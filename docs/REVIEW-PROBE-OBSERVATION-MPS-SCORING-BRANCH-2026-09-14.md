# Review: claude/probe-observation-mps-scoring

Date: 2026-09-14. Reviewed at branch head 31b39a4 (one commit over main
56c2667, fast-forwardable). Origin: the P7-B local qualification run on an
Apple Silicon Mac earlier the same day, in which every study-measurement
reading (6,912 of 6,912) came back `missing` while generation and the policy
runtime worked.

## 1. What the branch does

- **Transfer before cast.** `probe_observation.py` converted each observed
  activation with one combined `.to(device='cpu', dtype=torch.float64)`. On
  MPS that asks the source device for float64, which it lacks: for
  single-position selections it raised (5,588 readings, `Cannot convert a MPS
  Tensor to float64`), and for whole-prompt selections with storage offset zero
  and a page-aligned host buffer it returned undefined host memory without
  raising (1,324 readings, `The probe produced a non-finite score.`, with a
  silently wrong finite score also possible). The agent reproduced both
  behaviours on this machine. The new `to_cpu_float64` helper transfers to
  CPU first and casts there, which is bit-identical on CPU and CUDA.
- **Stage-named non-finite reasons.** A non-finite reading stays `missing`
  but its reason now says whether the observed activation, the
  standardization, or a layer's arithmetic produced it. ReLU is finite-in
  finite-out, so no check follows it.
- **Audit, tests, docs.** `audit-residual-runtime.py` models the rewrite
  exactly, keeps standardization and layer arithmetic pinned to the P3 landing,
  pins the helper's two-step body, and rejects the combined call as a third
  negative control. Four tests: the two transfer orders agree on CPU across
  bf16/fp16/fp32 and three selection shapes; on MPS (skipped when
  unavailable, executed here) a bf16 `(1, 40, 2560)` tensor with single,
  whole, and scattered selections equals `tensor.cpu().double()`, and the
  observer runs end to end on the toy decoder on MPS with scores equal to the
  reference arithmetic; the stage names are reached by a denormal scale and an
  overflowing layer weight. One paragraph in `docs/PROBE-STUDY-MEASUREMENTS.md`,
  a changelog entry, and the regenerated Python client identity.

## 2. What I checked

- Read the full diff (6 files, +131/−13). The arithmetic lines are unchanged;
  the only behavioural change on CPU/CUDA is that a non-finite result is now
  attributed to a stage. The P5 policy path (float32 on device,
  `.detach().cpu().tolist()` for evidence) was confirmed free of the pattern
  and is untouched.
- The mechanism against the live evidence: the run had `lastNonPadding`
  decode readings (single-position selections) for the raising case and
  prefill readings of whole prompts for the undefined-memory case; the counts
  5,588 and 1,324 match that split.
- Gates and suites on the branch tree: see §4.
- Live acceptance: the local P7-B leg rerun on MPS after landing (§5).

## 3. Findings

No landing fix. Notes:

- **N1 Silent wrong scores were possible before this fix** on MPS for
  whole-prompt selections (the undefined-memory path can return zeros). No
  MPS study evidence existed before today's run, and that run's readings are
  all `missing`, so no recorded score is affected.
- **N2 CUDA evidence unaffected**: CUDA supports float64, so the combined call
  produced correct values there; the P7-B CUDA run needs no rerun for this.

## 4. Suite results on the branch tree

Python (`Server/`, main venv, `HF_HUB_OFFLINE=1`): 6,616 passed, 9 skipped, 8 warnings
in 210 s (4 new tests; the MPS test executed rather than skipped). `check-generated.py
--audits` PASS with the residual-runtime audit's new transfer negative control,
`public_scan.py` clean, `git diff --check` clean, vocabulary grep clean.

Swift (Xcode beta, Metal toolchain 32023.920.1, serial, coverage mapping off):
TEST SUCCEEDED, 4954 passed, 5 skipped, 0 failed (identity gate satisfied by the
regenerated constant).

## 5. Landing and live acceptance

Fast-forward of main to 31b39a4, this review on top, app rebuilt (Python
client identity changed), engine pushed when the cluster session allows. Live
acceptance passed: the same frozen five-arm study rerun on the Mac's Python
engine (mps:0, bfloat16) after landing recorded 6,912 of 6,912 readings with
no missing reasons (before the fix: 0 of 6,912). On the fixed-strength arm the
post-action minus pre-action probe score averaged −0.4306 against the linear
prediction of −0.4308 (8 × wᵀ(v/scale)), median relative error 0.4% and
maximum 2.1% over 1,152 positions, consistent with bf16 rounding of the
addition; on the threshold arm the 511 positions with strength 0 show a delta
of exactly 0. The policy runtime's float32 device scores and the observer's
float64 CPU scores agree to 9e-7 at the same positions. Outputs are identical
to the pre-fix run, so the fix changed measurement only.
