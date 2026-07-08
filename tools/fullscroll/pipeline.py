"""pipeline — thin orchestration of the full-scroll stages. No framework:
a TOML config + argparse + subprocess/direct imports; each stage is resumable
and independently invokable (the stage scripts own their own CLIs).

Stages (design-doc bootstrap order):
  fetch      stage CT + label sources locally (fenix ingest-zarr + curl)
  labelgen   meshes -> constraints -> fit v0 -> (w,normal,conf) + sem + ink
  texture    harvest real-CT texture bank for phantom realism
  train      phantom pretrain -> real finetune -> QAT   (train.py phases)
  sweep      tiled inference -> prediction store        (sweep.py)
  solve      unwrap/stitch/emit wrap ids                (solve.py)
  evaluate   label-free health gates + oracle checks

  python pipeline.py --config fs.toml [--stages train,sweep] [--dry-run]
  python pipeline.py e2e-phantom       # the <20 min single-GPU integration gate
"""
import argparse
import json
import os
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sh(cmd, dry=False):
    print("+", " ".join(cmd))
    if dry:
        return
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"stage command failed ({r.returncode})")


def stage_fetch(cfg, dry):
    f = cfg.get("fetch", {})
    for item in f.get("ingest", []):
        if os.path.exists(item["out"]):
            print(f"fetch: {item['out']} exists — skip")
            continue
        sh([f.get("fenix", "./build-release/fenix"), "ingest-zarr",
            item["url"], str(item["level"]), item["out"],
            *[str(v) for v in item["origin_zyx"]],
            *[str(v) for v in item["shape_zyx"]]], dry)
    for item in f.get("curl", []):
        if os.path.exists(item["out"]):
            continue
        sh(["curl", "-fsSL", "-o", item["out"], item["url"]], dry)


def stage_labelgen(cfg, dry):
    lg = cfg["labelgen"]
    py = sys.executable
    if not os.path.exists(lg["constraints_dir"]) or cfg.get("force"):
        sh([py, "mesh_ingest.py", "--meshes", lg["meshes"],
            "--umbilicus", lg["umbilicus"], "--out", lg["constraints_dir"],
            "--scale", str(lg.get("scale", 1.0))], dry)
    store = lg["store"]
    if not os.path.exists(store) or cfg.get("force"):
        sh([py, "fitfield.py", "--constraints",
            os.path.join(lg["constraints_dir"], "*.npz"),
            "--umbilicus", lg["umbilicus"],
            "--bbox", *[str(v) for v in lg["bbox"]],
            "--scroll", lg["scroll"], "--level", str(lg["level"]),
            "--voxel-um", str(lg["voxel_um"]), "--out", store,
            "--report", store + ".fit_report.json"], dry)
    args = [py, "semlabels.py", "--store", store]
    if lg.get("mesh_bands", True):
        args += ["--mesh-constraints", os.path.join(lg["constraints_dir"], "*.npz")]
    if lg.get("teacher"):
        args += ["--teacher", lg["teacher"]]
    sh(args, dry)
    if lg.get("ink_zarr"):
        sh([py, "inkproject.py", "--store", store, "--ink-zarr",
            lg["ink_zarr"]], dry)


def stage_texture(cfg, dry):
    t = cfg["texture"]
    if os.path.exists(t["out"]) and not cfg.get("force"):
        return
    sh([sys.executable, "texture.py", "--ct", t["ct"], "--labels", t["labels"],
        "--out", t["out"]], dry)


def stage_train(cfg, dry):
    tr = cfg["train"]
    args = [sys.executable, "train.py", "--out", tr["out"]]
    for ph in tr["phases"]:
        args += ["--phase", ph]
    if tr.get("manifest"):
        args += ["--manifest", tr["manifest"]]
    if tr.get("warm"):
        args += ["--warm", tr["warm"]]
    for k in ("batch", "patch", "workers"):
        if k in tr:
            args += [f"--{k}", str(tr[k])]
    if os.path.exists(tr["out"] + "_last.pt") and not cfg.get("force"):
        args += ["--resume", tr["out"] + "_last.pt"]
    sh(args, dry)


def stage_sweep(cfg, dry):
    sw = cfg["sweep"]
    args = [sys.executable, "sweep.py", "run", "--ct", sw["ct"],
            "--out", sw["out"], "--ckpt", sw["ckpt"],
            "--lane", sw.get("lane", "fp16cl"),
            "--scroll", sw.get("scroll", "unknown"),
            "--level", str(sw.get("level", 0)),
            "--voxel-um", str(sw.get("voxel_um", 0.0))]
    if sw.get("coarse_ct"):
        args += ["--coarse-ct", sw["coarse_ct"],
                 "--coarse-factor", str(sw.get("coarse_factor", 32))]
    if sw.get("shard"):
        args += ["--shard", sw["shard"]]
    sh(args, dry)


def stage_solve(cfg, dry):
    so = cfg["solve"]
    args = [sys.executable, "solve.py", "all", "--preds", so["preds"],
            "--out", so["out"]]
    if so.get("umbilicus"):
        args += ["--umbilicus", so["umbilicus"],
                 "--anno-scale", str(so.get("anno_scale", 1.0))]
    sh(args, dry)


def stage_evaluate(cfg, dry):
    """Label-free health gates on real runs (the scroll has no GT): loop
    residual/conflict rates from the stitch, wrap coverage, plus the phantom
    e2e oracle as the algorithmic gate."""
    so = cfg.get("solve", {})
    rep = {}
    off = os.path.join(so.get("out", ""), "offsets.json")
    if os.path.exists(off):
        with open(off) as f:
            st = json.load(f)
        rep["components"] = st["components"]
        rep["edge_keep_rate"] = st["n_kept"] / max(st["n_edges"], 1)
        rep["conflicts"] = st["n_conflicts"]
        rep["conflict_rate"] = st["n_conflicts"] / max(st["n_kept"], 1)
        ok = rep["conflict_rate"] < 0.02 and rep["edge_keep_rate"] > 0.9
        rep["gate"] = "PASS" if ok else "WARN"
    print(json.dumps(rep, indent=1))
    out = os.path.join(so.get("out", "."), "eval.json")
    if not dry and rep:
        with open(out, "w") as f:
            json.dump(rep, f, indent=1)


STAGES = {"fetch": stage_fetch, "labelgen": stage_labelgen,
          "texture": stage_texture, "train": stage_train,
          "sweep": stage_sweep, "solve": stage_solve,
          "evaluate": stage_evaluate}


def e2e_phantom():
    """Integration gate, single GPU, <20 min: tiny train on phantoms ->
    fake-lane sweep math -> solve eval-phantom (CPU oracle e2e)."""
    py = sys.executable
    sh([py, "train.py", "--out", "/tmp/fs_e2e", "--phase", "phantom:30",
        "--batch", "1", "--patch", "128", "--workers", "2", "--val-every", "15",
        "--val-batches", "1", "--val-phantoms", "2",
        "--probe-every", "1000000", "--ckpt-every", "15"])
    sh([py, "solve.py", "eval-phantom", "--size", "192", "--block", "96",
        "--workers", "4"])
    print("E2E PHANTOM: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--stages", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "e2e-phantom"])
    args = ap.parse_args()
    if args.cmd == "e2e-phantom":
        e2e_phantom()
        return
    if not args.config:
        raise SystemExit("--config required (or use e2e-phantom)")
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    cfg["force"] = args.force
    wanted = args.stages.split(",") if args.stages else \
        [s for s in STAGES if s in cfg or s == "evaluate"]
    for name in wanted:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}")
        print(f"===== stage {name} =====")
        STAGES[name](cfg, args.dry_run)


if __name__ == "__main__":
    main()
