# shakedown — training-pipeline validation runs

Box-side scripts exercising the training data plane + loop end-to-end on a GPU pod
(paths assume /root + /workspace/fenix; adjust per box). Run history + findings live in
the git log (search `Suite`/`Pass-A`/`shakedown`).

- `suite.sh` — the 6-phase functional suite: warm-cache speed, resume-from-checkpoint,
  feeder-kill/restart recovery, teacher/KD channel, patch=256, torchao QAT prepare.
- `passA.sh` + `drain.py` — feeder throughput sweep (threads × per-stage timing;
  drain.py is a max-rate ring consumer).
- `passB.sh` + `ring_dump.py` — determinism (same seed ⇒ identical draws), kill-9
  mid-fill cache crash-integrity, slot label-content sanity.
- `passC.sh` — multi-volume corpus + 1000-step soak (RSS + throughput drift).
- `roundD.sh` — trainer balance sweep (base×batch), multi-resolution canonical feed
  (um= resampling), validation-ring cadence.
- `roundF.sh` — QAT full cycle: prepare → train → convert → int8 forward.
- Round E (dress rehearsal: bulk-KD subset → KD train → student export →
  `fenix predict-surface` → eval firewall) lands once the student export path exists.

Bugs these runs have caught so far: CachedVolume lock-across-fetch serialization,
shell-stamp ~300× overdraw, ring O_TRUNC SIGBUS on live consumers, read-only cache
reopen, S3 fetch self-congestion, 256 MiB block-cache thrash, torchao API drift.
