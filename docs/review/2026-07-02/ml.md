# Review: unit "ml" — src/ml/ (infer.hpp, inference.cpp, tiling.hpp, weights.hpp, augment.hpp, augment_cli.hpp)

Overall the module is in good shape for its age: the sliding-window tile math (`tile_starts`,
`gaussian1d`, the factorized `Wz·Wy·Wx` weight profile) is consistent between scatter and
normalization (edge guards match, the 0.1 Gaussian floor guarantees `w > 0` everywhere covered,
so there is no divide-by-zero at coverage boundaries); all 48 octahedral TTA inverse mappings were
traced and are correct (`flip⁻¹` then `permute⁻¹` with `q[pp[i]] = i`, flips applied in the
permuted frame both ways); the producer/consumer ring is race-free and deadlock-free on the happy
path (filled-count protocol, slot copied out via the fp16 cast before release). The problems
cluster around the NEW resumable checkpoint (a real accumulator-pollution bug on a bad file, and
a header that doesn't identify the model/input it belongs to — dangerous because checkpointing is
on by default), error handling (torch exceptions escape into a joinable-thread `std::terminate`,
CLI `stoi` crashes), the untrusted `.fxweights` parse, and a missing NaN guard on the fp16
forward.

## [high/correctness] `ckpt_load` pollutes the accumulator on a short/corrupt checkpoint, then reports "no resume"

**Verdict:** CONFIRMED — The finding is accurate. In src/ml/infer.hpp:157-162, the payload loop sets `ok = std::fread(...) == m;` and then unconditionally runs `for (j) acc[i+j] = buf[j]*sc;` — so on a short read the partial/stale buffer chunk is written, and every previously-read chunk was already stored into acc. The function then returns -1 (line 163/167), contradicting its own contract comment at line 139 ("leaves acc untouched"). The caller at infer.hpp:246-252 passes prob.data() (a freshly zeroed Volume, line 182) and only acts on `got > 0`; on got == -1 it proceeds with resume_from = 0 and Gaussian-accumulates the full run on top of the polluted baseline, and normalize() at 214-225 divides only by the fresh run's weight profile — the leftover checkpoint mass is never divided out, so the whole output volume is biased. Reachability: checkpointing is on by default (src/ml/inference.cpp:207 sets `opt.ckpt_path = outpath + ".ckpt"` unless `ckpt=off`), and ckpt_save (infer.hpp:132-134) does fflush + rename but no fsync/fdatasync — a power loss or crash after rename can leave a renamed file whose data blocks were never persisted, i.e. a valid-header short file; that is exactly the crash-then-resume scenario the checkpoint exists for. The header match at 145-149 checks dims/config but nothing validates payload length before writing. No guard elsewhere; not a documented stub. Severity high is appropriate: silent, plausible-looking wrong probabilities across the entire output with no log message (got=-1 is indistinguishable from "no checkpoint").

**Fix notes:** The proposed fix is correct but can be simplified. (1) Moving the copy loop inside `if (ok)` alone is insufficient — earlier successfully-read chunks are already in acc, so the zero-on-failure part is the load-bearing piece. Since the only caller passes a freshly zeroed buffer, `std::memset(acc, 0, nvox*sizeof(float))` (or a fill loop) before returning -1 exactly restores prior state. (2) The fstat size pre-check is fine as belt-and-suspenders but must account for the format: when h.acc_scale <= 0 ckpt_save writes header only (no payload), so expected size is sizeof(CkptHeader) when acc_scale<=0 and sizeof(CkptHeader)+2*nvox otherwise; the short-fread detection already catches truncation without it. (3) Consider also logging a warning on a header-matching but truncated/corrupt checkpoint so the restart isn't silent, and optionally fsync in ckpt_save before rename to reduce the truncation window (weigh against the 17 GB checkpoint size / best-effort design).

**Location:** src/ml/infer.hpp:157-164 (caller: infer.hpp:247-252)

**Evidence:**
```cpp
for (s64 i = 0; i < nvox && ok; ) {
    const s64 m = std::min<s64>(static_cast<s64>(buf.size()), nvox - i);
    ok = std::fread(buf.data(), sizeof(u16), static_cast<std::size_t>(m), f) == static_cast<std::size_t>(m);
    for (s64 j = 0; j < m; ++j) acc[static_cast<usize>(i + j)] = static_cast<float>(buf[static_cast<usize>(j)]) * sc;
    i += m;
}
if (ok) n = h.n_done;
```
The comment above the function claims it "leaves acc untouched" on a short file. It does not:
every chunk read *before* the failure is already written into `acc`, and the *failing* chunk is
also written (the `for j` copy runs unconditionally even when `ok` just became false, copying the
previous chunk's stale bytes). The caller then sees `got == -1`, keeps `resume_from = 0`, and
starts full inference **on top of the garbage baseline** instead of the zeroed `prob`:
```cpp
const long got = detail::ckpt_load(opt.ckpt_path, ckpt_hdr, prob.data(), d.count());
if (got > 0) { resume_from = got; ... }   // -1 → silently proceed with polluted prob
```

**Failure scenario:** a truncated `<out>.ckpt` (disk-full that lied to fflush, external
truncation, copy of a ckpt from another box) whose header matches → the whole prediction is
silently biased by a residual quantized accumulator added under every Gaussian window; the output
`.fxvol` looks plausible but is wrong everywhere the partial data landed. Checkpointing is ON by
default, so any such file next to the output triggers this without any user opt-in.

**Suggested fix:** on any mismatch/short-read after the header matched, `memset(acc, 0, nvox*4)`
(or defer writes: read the whole payload into a scratch buffer / validate the file size ==
`sizeof(CkptHeader) + 2*nvox` up front via fstat before writing a single value into `acc`). Also
move the `for j` copy inside `if (ok)`.

## [high/correctness] Checkpoint header does not identify the model, weights, input, or norm — default-on resume silently mixes runs

**Verdict:** CONFIRMED — The finding is accurate. CkptHeader (src/ml/infer.hpp:98-108) records only magic/dims/patch/overlap/tta/tile_shift/total_tiles, and ckpt_load (infer.hpp:145-149) matches on exactly those fields — nothing identifies the weights file, the input volume beyond its dimensions, or the task config (sigmoid/channel/norm, infer.hpp:36-38, which differ between predict-surface and predict-ink per inference.cpp:349/362, both of which funnel into the same run_predict). Checkpointing is default-ON at <out>.ckpt (inference.cpp:205-207) and only removed on success (infer.hpp:385/435), so a killed run always leaves a stale checkpoint. Re-running to the same output path with retrained weights, a different same-dims input, or the other predict stage (if grid params coincide) passes the header match at infer.hpp:246-247 and silently blends two different models' Gaussian-weighted accumulations. The comment "auto-resumes if the run is re-launched with the same args" (inference.cpp:205-206) documents the assumption but there is no guard and no warning — a documented assumption without enforcement does not make the reachable corruption intentional. Nothing elsewhere prevents it: the ensemble wrappers (scales/rots/noise/offsets) clear ckpt_path for inner members (infer.hpp:487/556/627) but the base single-config run checkpoints, which is exactly the vulnerable path.

**Fix notes:** The proposed fix is right in direction, with three corrections: (1) noise_member_seed does NOT need to be in the header — noise-TTA runs go through the ensemble wrapper which clears ckpt_path (infer.hpp:487/556/627), so a checkpointed run is always the clean base member; including it is harmless but dead. (2) For weights identity prefer a content hash (e.g. FNV-1a over the .fxweights bytes, or reuse the file's existing header/params hash if .fxweights carries one) over path+size+mtime — weights files get copied/re-downloaded and mtime churn would spuriously invalidate multi-hour checkpoints; hashing a few-hundred-MB weights file is negligible next to inference cost. (3) Do include sigmoid+channel+norm (they distinguish predict-surface from predict-ink) and an input identity (source path hash plus, for .fxvol, the archive's provenance/params hash rather than raw content — hashing a 2048³ volume is not free but is a one-time streaming pass if wanted). On mismatch, log a clear warning and start fresh (ignore/overwrite the checkpoint) rather than hard-fail — checkpointing is documented best-effort (infer.hpp:119) and a hard error would block legitimate re-runs with intentionally changed settings; the CkptHeader magic version byte ('2') should also be bumped so old-format checkpoints are rejected cleanly per the project's no-compat rule.

**Location:** src/ml/infer.hpp:98-108 (CkptHeader), src/ml/inference.cpp:207

**Evidence:**
```cpp
struct CkptHeader { char magic[8]; s64 dz, dy, dx; int patch; double overlap; int tta;
                    s64 tile_shift; long total_tiles; long n_done; double acc_scale; };
...
if (opt.ckpt_path.empty()) opt.ckpt_path = outpath + ".ckpt";   // resumable by DEFAULT
```
The match check is dims/patch/overlap/tta/tile_shift/total_tiles only. Nothing identifies the
input volume (beyond dims), the weights file, the task (`sigmoid`/`channel`/`norm`), or the noise
member seed.

**Failure scenario:** `fenix predict-surface scroll.fxvol w_v1.fxweights out.fxvol` is killed
mid-run; the user re-runs with retrained `w_v2.fxweights` (or runs `predict-ink` to the same
`out.fxvol`, or the same-named path with a *different* input volume of identical dims — trivially
common when re-encoding). The header matches, the run "RESUMES", and the final probability volume
is a Gaussian-blended average of **two different models' predictions** with no warning. Because
checkpointing is on by default, this needs no `ckpt=` flag to happen.

**Suggested fix:** add to the header a hash of (weights-file content hash or path+mtime+size,
input path+content id, `norm`, `sigmoid`, `channel`, `noise_apply`/`noise_member_seed`) and
reject on mismatch; log a warning when a ckpt is found but rejected.

## [high/resource-safety] Torch exceptions (CUDA OOM) escape the predict loop; in the batched path they unwind past a joinable `std::thread` → `std::terminate`

**Verdict:** CONFIRMED — Confirmed, not refuted. (1) Exceptions are live in this TU: CMakeLists.txt:89-92 only applies -fno-exceptions/-fno-rtti `if(NOT FENIX_ML)`, and the comment there explicitly says ML builds keep exceptions+RTTI for libtorch; infer.hpp is included solely by src/ml/inference.cpp:11 (the one torch TU), which only exists under FENIX_ML. So c10::Error from net->forward (infer.hpp:365, 410), .to(dev) (360, 398), or softmax can propagate. (2) A repo-wide grep for try/catch across src/ml/ and apps/ finds zero handlers — nothing in predict_surface_filled, inference.cpp, or the driver catches, and no std::set_terminate exists. (3) In the batched path the throwing torch calls (infer.hpp:357-368) sit between the construction of the joinable `std::thread producer` (infer.hpp:330) and `producer.join()` (infer.hpp:381); unwinding destroys a joinable std::thread → guaranteed std::terminate. In the serial path the uncaught exception also terminates (and would cross into -fno-exceptions frames only in mixed builds, but uncaught-exception terminate happens regardless). (4) The scenario is reachable: the module's own docs flag VRAM pressure (infer.hpp:35 "256^3 needs >8 GB VRAM"; src/ml/CLAUDE.md build notes say use patch=128 on 8 GB cards), and mid-run CUDA OOM/driver errors are ordinary operational events on shared GPUs. No project invariant blesses abort-on-torch-error; the module contract is "don't leak torch types past this API" and the project error contract is std::expected. One accuracy correction to the claim's impact: the checkpoint file is only removed on success (infer.hpp:385, 435), so an abort still leaves a resumable checkpoint — work since the last cadence save is lost, but the run is restartable; this softens impact slightly but does not refute the terminate.

**Fix notes:** The proposed fix is sound and compilable (infer.hpp is only compiled inside inference.cpp, where exceptions are on). Additions: (a) also wrap the PRODUCER body — prep() can throw std::bad_alloc via the filler's allocations; on producer-side error set the same stop flag, set prod_done, notify cv_filled, and have the consumer drain/join then return the error; (b) the consumer-side stop flag must be checked in the producer's cv_free.wait predicate ("filled < kSlots || stop") AND after wake (break on stop) so it can't re-block; (c) catch `const std::exception&` (c10::Error derives from it) plus a `catch (...)` fallback rather than naming c10::Error, keeping torch types out of the error path; (d) before returning the error in the batched path, call save_ckpt(resume_from + n_tiles) so already-scattered tiles are preserved without waiting for the cadence; (e) the same catch-translate wrapper is needed around the setup torch calls in run_predict/inference.cpp (weight load, net->to(dev), TorchScript load), not just the loop — .to(dev) at model setup is an equally likely OOM site; (f) serial path (B==1) needs the same wrapper even though there is no thread — uncaught still terminates.

**Location:** src/ml/infer.hpp:330-381 (producer thread + consumer forward), inference.cpp module-wide

**Evidence:**
```cpp
std::thread producer([&] { ... });
for (;;) {
    ...
    auto logits = net->forward(xin);   // torch throws c10::Error on CUDA OOM etc.
    ...
}
producer.join();
```
FENIX_ML builds enable exceptions for the libtorch ABI, and every torch op (`forward`, `.to(dev)`,
`softmax`) throws `c10::Error` on failure — CUDA OOM is explicitly plausible here (the batch=3
comment documents b4 as a "VRAM-pressure regression"; patch=256 needs >8 GB). There is no
try/catch anywhere in `predict_surface_filled` or `run_predict`. If `forward` throws mid-run in
the batched path, the exception unwinds past the local `producer` thread while it is joinable →
guaranteed `std::terminate`. Even in the serial path, the exception propagates up into driver
frames compiled `-fno-exceptions` → terminate. Either way the project's `std::expected` error
contract is violated: a recoverable GPU error becomes a process abort.

**Failure scenario:** long whole-volume run, another process grabs VRAM (or fragmentation after
thousands of tiles) → CUDA OOM on tile 2000/3375 → `terminate()` with no Error, no final
checkpoint flush (only the last cadence save survives), and a core dump instead of a message.

**Suggested fix:** wrap the torch-touching body of `predict_surface_filled` (and `run_predict`'s
torch calls) in `try { ... } catch (const std::exception& e) { stop the producer (set a stop flag
+ notify cv_free), join, return fenix::err(Errc::internal, e.what()); }`. The producer wait
predicate must also check the stop flag or it deadlocks on `cv_free` when the consumer exits
early with both slots filled.

## [medium/correctness] `.fxweights` parser: unbounded record walk + u64 overflow in the blob range check + shape unvalidated → OOB reads / SIGBUS on a corrupt file

**Verdict:** unverified (medium/low)

**Location:** src/ml/weights.hpp:70-81

**Evidence:**
```cpp
for (std::uint32_t i = 0; i < count; ++i) {
    std::uint16_t nlen = 0; rd(cur, nlen); cur += 2;                 // no cur+2 <= sz check
    std::string name(reinterpret_cast<const char*>(p + cur), nlen);  // no cur+nlen <= sz check
    ...
    for (std::uint8_t d = 0; d < ndim; ++d) { rd(cur, shape[d]); cur += 8; }
    if (data_off + nbytes > sz) return fail(...);                    // u64 wrap bypasses this
    torch::Tensor t = torch::from_blob(const_cast<std::uint8_t*>(p) + data_off, shape, opts).clone();
```
Three holes: (1) `cur` marches through record headers with zero bounds checks — a `count` larger
than the file supports, or a huge `nlen`, reads past the mmap end (SIGBUS on a file-backed map, or
silent garbage inside the last page). (2) `data_off + nbytes` is an unsigned add:
`data_off = UINT64_MAX-16, nbytes = 64` wraps small, passes the check, and `from_blob` reads from
`p + data_off` — wild pointer. (3) `shape` is never validated against `nbytes`
(`Π shape[d] · itemsize == nbytes`); `from_blob(...).clone()` reads `numel·itemsize` bytes, which
a corrupt shape can push far past the checked blob. Negative shape entries also make torch throw
(→ terminate per the previous finding).

**Failure scenario:** a truncated download of a multi-GB `.fxweights` (the realistic case: these
files travel between the RunPod box and dev machines) crashes with SIGBUS or silently loads
garbage weights that pass `load_into` name matching.

**Suggested fix:** check `cur + needed <= sz` before every read; validate
`data_off <= sz && nbytes <= sz - data_off` (subtraction form, no overflow); validate
`ndim <= 8`, each `shape[d] > 0`, and `Π shape · itemsize(dtype) == nbytes`; reject unknown dtype
codes instead of defaulting to kFloat32.

## [medium/correctness] No NaN/Inf guard on the fp16 forward output: one overflowed patch poisons its whole overlap neighborhood, and the f32→u8 cast of NaN is UB

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:365-371 (scatter of `surf`), src/ml/inference.cpp:328-332 (u8 quantize)

**Evidence:**
```cpp
auto logits = net->forward(xin);                              // fp16 forward, opt.half default true
auto pr = ... torch::softmax(logits, 1).index(...);           // NaN logits → NaN probs
...
p8(z, y, x) = static_cast<u8>(std::clamp(pvv(z, y, x), 0.0f, 1.0f) * 255.0f + 0.5f);
```
fp16 intermediate overflow (|x| > 65504 in a conv accumulation) is the classic nnU-Net half-
precision hazard: it yields inf logits, `softmax(inf - inf)` = NaN. A NaN in one tile's output is
scattered with Gaussian weights into every voxel of the P³ region, and NaN + anything = NaN, so
it also destroys the contributions of all *overlapping* tiles (at overlap 0.5 that is up to 8
neighbors). Nothing scrubs it: `normalize()` divides NaN by w, the u16 checkpoint quantize
(`acc[i] * inv + 0.5f` → cast to u16) is UB on NaN and bakes garbage into the resumable state,
and the final `std::clamp(NaN,...)` → `static_cast<u8>(NaN)` is UB. Note `std::isnan` is
unreliable under this project's `-ffast-math`, so the fix must be arithmetic
(e.g. `v == v` is also unreliable; use `torch::nan_to_num` on the GPU tensor, where libtorch is
not compiled fast-math).

**Failure scenario:** one hot patch (bright inclusion, dense mineral) overflows fp16 → a P³+halo
region of the output silently becomes 0 (or garbage) in the u8 artifact; downstream `predictions`
treats it as confident air.

**Suggested fix:** `pr = torch::nan_to_num(pr, /*nan=*/0.0, /*posinf=*/1.0, /*neginf=*/0.0)` right
after softmax/sigmoid (runs on GPU, torch is IEEE-safe), and/or detect nonfinite logits per batch
and retry that batch in fp32.

## [medium/bug] CLI parsing: `std::stoi` on positional slots crashes on keyword args; `ckpt_every=0` → integer division by zero

**Verdict:** unverified (medium/low)

**Location:** src/ml/inference.cpp:180-184, 203; src/ml/infer.hpp:377, 428

**Evidence:**
```cpp
if (args.size() >= 4) opt.patch = std::stoi(std::string(args[3]));   // "scales=0.8" → throws
...
if (a.size() > 11 && a.substr(0, 11) == "ckpt_every=") opt.ckpt_every = std::stol(...);  // 0 accepted
...
if (ckpt_on && (n_tiles % opt.ckpt_every < nb)) save_ckpt(done);     // % 0 → SIGFPE
```
The keyword args are documented as "scanned anywhere in args", but only slot 6 skips
`=`-containing tokens. `fenix predict-surface in.fxvol w.fxweights out.fxvol scales=0.8,1.0`
feeds `"scales=0.8,1.0"` to `std::stoi` → `std::invalid_argument` thrown → unwinds into
`-fno-exceptions` driver frames → terminate, instead of a usage error. Separately,
`ckpt_every=0` (or negative) is accepted and hits `n_tiles % opt.ckpt_every` → SIGFPE.

**Suggested fix:** skip any `=`-containing token in the positional slots 3-5 (same test as slot
6); replace all `std::stoi/stol/stod` here with `std::from_chars` returning
`Errc::invalid_argument`; clamp `ckpt_every = std::max(1L, ...)`.

## [medium/hygiene] `fenix augment` writes an f32-coded `.fxvol` — violates the u8-native / never-f32-on-disk rule

**Verdict:** unverified (medium/low)

**Location:** src/ml/augment_cli.hpp:44-47

**Evidence:**
```cpp
auto out = codec::VolumeArchive::create(outpath, s.image.dims(), codec::DctParams{});
if (auto r = out->write_volume(s.image.view()); !r) ...   // VolumeView<const f32> overload
```
`write_volume(VolumeView<const f32>)` sets the archive `src_dtype_` to f32
(src/codec/archive.hpp:270-272,290), so the augmented training crop is stored as an f32 volume —
4× the bytes of the u8 source and a direct violation of the project's "u8-native storage, f32
must never be written to disk / never promote u8 CT for storage" invariant. The input to
`augment` is u8-origin CT read via `read_volume(0)` (f32 decode); the whole augmentation chain is
designed around ~0..255 values, so the output should round-trip to u8. The predict path in
inference.cpp:324-336 does this correctly — augment_cli forgot.

**Failure scenario:** offline generation of augmented training crops produces f32 archives 4× the
budgeted size, and the student then trains on f32-decoded data whose quantization state differs
from the u8 `.fxvol` it will run on — the very mismatch the `compression` augmentation exists to
prevent.

**Suggested fix:** clamp to [0,255], round to `Volume<u8>`, and `write_volume<u8>` like
inference.cpp does.

## [medium/performance] Default-on checkpointing: serial O(voxels) max-scan + quantize + multi-GB write on the GPU consumer thread every 128 tiles

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:113-136 (ckpt_save), 377; src/ml/inference.cpp:207 (default on)

**Evidence:**
```cpp
float mx = 0.0f;
for (s64 i = 0; i < nvox; ++i) mx = std::max(mx, acc[...]);   // serial scan of the whole volume
...
for (s64 j = 0; j < m; ++j) buf[j] = static_cast<u16>(acc[i+j] * inv + 0.5f);  // serial quantize
ok = std::fwrite(...);                                        // blocking write, consumer thread
```
At 2048³ each save is a serial 8.6G-element max scan + quantize (seconds of single-core work) plus
a 17 GB blocking `fwrite` — all executed inline on the consumer thread between batches, so the GPU
sits idle for the duration. With the default `ckpt_every=128` and ~3375 tiles that is ~26 saves ≈
440 GB written per run and potentially tens of minutes of added wall time — for a feature the
user never asked for (`ckpt` defaults ON at `<out>.ckpt`). The pipelining win this file carefully
measured (prep overlap, fp16 D2H) is dwarfed by this.

**Suggested fix:** parallelize the max/quantize with `parallel_for`; do the write on a detached
IO worker double-buffered against the accumulator (it is safe to snapshot: only the consumer
mutates `prob`); scale `ckpt_every` with volume size (target e.g. ≤5% of projected wall time) or
make checkpointing opt-in for in-RAM-sized runs.

## [low/correctness] `predict_surface_rots` without angle 0 in the list leaves rotation-clipped corners as silent zeros

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:596-602

**Evidence:**
```cpp
const float w = wsum[...];
if (w > 1e-6f) av(z, y, x) /= w;   // else: av keeps its (near-zero) raw weighted sum
```
Every non-zero angle's validity mask is ~0 in the corners the rotation clips. If the user passes
`rots=15,30,45` (nothing forces including 0 — the code special-cases 0 but doesn't require it),
corner voxels have `wsum ≈ 0` and are emitted as probability ≈ 0 — fabricated "no surface" with
no warning, the exact absent-vs-failed conflation the project bans.

**Suggested fix:** if 0 is not in `opt.rots`, either inject it or, after fusing, fill
`wsum <= eps` voxels from a dedicated unrotated pass (or error out).

## [low/hygiene] `CkptHeader` is fwritten whole with two uninitialized 4-byte padding holes

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:98-108, 120

**Evidence:**
```cpp
struct CkptHeader { ...; int patch = 0; double overlap = 0; int tta = 0; s64 tile_shift = 0; ... };
...
bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
```
`int patch` → `double overlap` and `int tta` → `s64 tile_shift` each leave 4 padding bytes that
are never initialized (NSDMIs don't cover padding of a copied struct) and are written to disk.
Functionally harmless (the load compares fields, not bytes), but it writes uninitialized stack
memory to a file — MSan (a blocking CI gate) will flag the `fwrite`, and the ckpt bytes are
nondeterministic.

**Suggested fix:** reorder fields so the struct is padding-free (group the two `int`s together),
or `std::memset(&h, 0, sizeof h)`-then-assign before writing; add
`static_assert(std::has_unique_object_representations_v<CkptHeader>)`.
