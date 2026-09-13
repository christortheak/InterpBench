# J-lens dimension-batch benchmark: what ran, what we learned, what to change

- Date: 2026-09-12
- From: the maintainer's integration agent, on main `b1190f7`
- For: the refactor agents (§6 is the work list) and the study record
- Follows: `docs/JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md` §R1
  and the "Live acceptance" list in `docs/REVIEW-JLENS-SCALING-BRANCH-2026-09-11.md`

## 1. Summary

Six `jlens-fit-benchmark` jobs ran on the first site during 2026-09-12,
all through the managed workflow (interview → publish → package → stage →
plan → submit → export → fetch) with no hand edits and, for the second
round onward, with the per-request GPU type landed the same morning.

What we learned, in order of consequence:

1. **Dimension batch 4 is the ceiling on an 80 GB card** for the 27B
   hybrid-attention checkpoint in bfloat16, on both the A100 and the H100.
   Batch 8 and 16 exhaust device memory at the first backward pass, right
   after weight load. Batch 4 peaks at 79.3 GB of 80.
2. **Batching scales almost linearly to 4.** Two rows take 8,289 s at batch
   1 on the A100 and 2,104 s at batch 4; on the H100, 5,249 s and 1,334 s.
   That is 3.9× on both cards.
3. **The H100 is 1.6× faster than the A100 at every batch** on this
   workload, which runs the linear-attention layers on the Torch fallback.
4. **Batched and unbatched fits disagree by 5–7% relative Frobenius norm at
   early layers in bfloat16, and this is precision, not a defect.** The same
   comparison on a 4B model in float32 agrees to 1e-5; in bfloat16 on that
   same model the gap is 13–14%. The gap decays smoothly with depth in every
   run. Batch 2 and batch 4 differ from batch 1 by the identical amount at
   every layer, so the batched path agrees with itself.
5. **Consequence for the science:** any bfloat16 J-lens fit, including the
   published reference lenses fitted at batch 64, carries a percent-scale
   early-layer reproducibility floor under a change of kernel path. Early-
   layer readouts are to be interpreted through the stability protocol
   (agreement across independent fits), not read as exact.
6. **The benchmark tool needs three changes** (§6): a relative tolerance,
   pairwise case comparison with saved case matrices, and trimmed failure
   records.

## 2. What ran

All runs: two rows of the study's neutral maintenance-prose corpus (the
pilot corpus, first eight rows, 300+ characters), `maxSeqLen 128`,
`skipFirst 16`, kernel policy `current`, no compilation, tolerances
`rtol = atol = 1e-4`, one GPU, 80 GB host memory, 24 h walltime. Each
case is a fresh subprocess with its own compiler cache; the first case
(batch 1) is the comparison baseline.

| Round | Model, precision | Cases | GPU | Wall time | Request |
|---|---|---|---|---|---|
| 1 | 27B, bfloat16 | 1, 8, 16 | A100 | 2 h 22 min | `requests/jlens-bench-27b-dimbatch` |
| 1 | 27B, bfloat16 | 1, 8, 16 | H100 | 1 h 29 min | same, `--gpu-type H100` |
| 2 | 27B, bfloat16 | 1, 2, 4 | A100 | 4 h 35 min | `requests/jlens-bench-27b-dimbatch-small` |
| 2 | 27B, bfloat16 | 1, 2, 4 | H100 | 2 h 54 min | same, `--gpu-type H100` |
| 3 | 4B, float32 | 1, 2, 4 | H100 | 12 min | `requests/jlens-bench-4b-float32-dimbatch` |
| 3 | 4B, bfloat16 | 1, 2, 4 | H100 | 9 min | `requests/jlens-bench-4b-bfloat16-dimbatch` |

The 27B is the study's hybrid-attention 27B checkpoint (width 5120, 63
source layers, 48 linear-attention layers on the Torch fallback because the
fused kernels are not installed). The 4B is the dense-attention 4B
qualification model already in the site's offline cache (width 2560, 33
source layers); it was chosen because float32 weights fit an 80 GB card
with room for batch 4, so precision could be isolated from batching.

Each benchmark's two-row fitting request is published beside it
(`requests/jlens-fit-27b-bench2`, `requests/jlens-fit-4b-float32-bench2`,
`requests/jlens-fit-4b-bfloat16-bench2`). Every plan, submission, and plan
hash is recorded under the request directory (`<GPU>/remote-plan.json`,
`remote-submit.json`, `plan-hash`). Reports were exported and fetched with
custody receipts into the workspace as `runs/jlens-benchmark-<id>/
benchmark-report.json`.

## 3. Results

### 3.1 Round 1: batch 8 and 16 do not fit

| GPU | Batch 1, 2 rows | Rows/hour | Peak device | Batch 8 | Batch 16 |
|---|---|---|---|---|---|
| A100 80 GB | 7,971 s | 0.90 | 60.9 GB | out of memory | out of memory |
| H100 80 GB | 4,826 s | 1.49 | 61.0 GB | out of memory | out of memory |

Weights take 54.7 GB; one row's backward at batch 1 needs about 6 GB more.
Both larger batches failed at the first backward with under 10 MB free,
which is the early-refusal behaviour R1 asked for. The reference fits used
batch 64 on a 179 GB card with fused kernels and a dense-attention model;
our ceiling is lower for all three reasons.

### 3.2 Round 2: the ceiling is 4, and the speedup is real

| GPU | Batch | Time, 2 rows | Rows/hour | Peak device | Worst mean-lens relative error vs batch 1 |
|---|---|---|---|---|---|
| A100 | 1 | 8,289 s | 0.87 | 60.9 GB | baseline |
| A100 | 2 | 4,265 s | 1.69 | 67.1 GB | 5.45% |
| A100 | 4 | 2,104 s | 3.42 | 79.3 GB | 5.62% |
| H100 | 1 | 5,249 s | 1.37 | 61.0 GB | baseline |
| H100 | 2 | 2,651 s | 2.72 | 67.1 GB | 5.56% |
| H100 | 4 | 1,334 s | 5.40 | 79.3 GB | 5.56% |

Per-prompt matrices show the same picture with a worst case of 7.1%. In
every case the error is largest at layer 0 (5.3–5.6%) and falls
monotonically to 0.06% at layer 62. Batch 2 and batch 4 report identical
per-layer errors against the baseline to three significant figures on the
H100 (5.3e-2 at layer 0 … 6.2e-4 at layer 62, both cases).

### 3.3 Round 3: precision, not batching

The same benchmark on the 4B model, two rows, one H100:

| Precision | Batch | Time, 2 rows | Peak device | Worst relative error vs batch 1 (layer 0) | Error at the last layer |
|---|---|---|---|---|---|
| float32 | 1 | 189 s | 18.1 GB | baseline | |
| float32 | 2 | 140 s | 19.0 GB | 8.7e-6 | 1.4e-7 |
| float32 | 4 | 126 s | 20.8 GB | 1.3e-5 | 2.0e-7 |
| bfloat16 | 1 | 199 s | 9.3 GB | baseline | |
| bfloat16 | 2 | 102 s | 9.8 GB | 14% | 0.10% |
| bfloat16 | 4 | 53 s | 11.0 GB | 13% | 0.12% |

Same code, same rows, same card: four orders of magnitude between the two
precisions, and the same depth-decaying shape in both. In float32 the
"agrees: false" verdicts come only from the absolute tolerance of 1e-4,
which raw-scale Jacobian entries cannot meet (maximum absolute differences
of 1.6e-3 to 3.6e-3 against relative errors of 1e-5); see §6.1.

## 4. Interpretation

**Why batched and unbatched differ in bfloat16.** At batch 1 the
vector-Jacobian products are matrix-vector shaped; at batch 2 and above
they are matrix-matrix shaped, and the BLAS library selects different
kernels with different accumulation order. In bfloat16 (seven mantissa
bits) the two paths round differently at every layer, and the backward
pass carries that difference through every layer between source and
target. A source at layer 0 accumulates the most; a source at the last
layer accumulates almost none. That is exactly the observed shape, and it
is why batch 2 and batch 4, which share the matrix-matrix path, agree with
each other and differ from batch 1 by the same amount. In float32 the
per-layer rounding is about 2^-24 instead of 2^-8 and the compounded
difference is negligible.

**Why this is not a bug.** A defect in the batching logic (wrong row
alignment, wrong reduction, off-by-one in the dimension chunking) would
produce errors that do not vanish in float32 and that would not decay
smoothly with depth. Both signatures are absent.

**What it says about the instrument.** Early-layer J-lens Jacobians fitted
in bfloat16 are only reproducible to a few percent under an innocuous
change of kernel path. The published reference lenses were fitted in
bfloat16 at batch 64 and carry the same class of noise relative to any
other kernel path, including ours. This does not invalidate them; it says
the early-layer readouts have a noise floor that the qualification record
must state, and that the standing stability protocol (interpret only what
is stable across independent fits) is the right way to read them. Late
layers, which is where readout comparisons are most informative, are
stable to 0.1% or better.

**Cross-hardware.** The A100 and H100 agree with each other on every
aggregate (same peak memory, same 5.5% batched-versus-unbatched gap, same
3.9× speedup). Whether two fits on different GPU types agree with each
other at the matrix level has not been measured, because a benchmark
report holds agreement numbers, not matrices (§6.2). The checkpoint
identity permits merging across GPU types; the numerical question stays
open until two real fits are compared.

## 5. Decisions

- **Fit at dimension batch 4** on either 80 GB GPU type. Do not request 8.
- **Prefer the H100 when the queue offers it; accept the A100.** Single-GPU
  requests have started within minutes on both pools; four-GPU H100
  requests wait days, which is why a round of single-GPU shards is the
  right shape.
- **Budget:** at batch 4, a reference-scale 800-row fit is about 150
  H100-hours or 235 A100-hours; eight shards bring that to roughly a day
  of wall time.
- **Report the bfloat16 early-layer noise floor** in the qualification
  record of any lens fitted this way, and read early-layer results through
  the stability protocol.
- **Stopping rule** stays as designed (`max-layer-relative-frobenius-
  running-mean-change`, reference 0.002 / window 10 / min 100 prompts);
  note that the statistic's own floor in bfloat16 is now known to be far
  below the stopping threshold at late layers and comparable to it at
  early layers over a single kernel-path change, which argues for
  evaluating the rule on late layers or on the full-depth maximum with
  that context recorded.

## 6. Work for the agents

### 6.1 Relative agreement criterion (F)

`jlens-fit-benchmark` compares each case with `rtol` and `atol` applied
elementwise. Raw Jacobian entries for these models have magnitudes where
an absolute allowance of 1e-4 is below float32 noise, so the float32 4B
cases report `agrees: false` at a relative Frobenius error of 1e-5. Make
the headline verdict a relative Frobenius criterion per layer (default,
say, 1e-3 for float32 and a researcher-declared value for bfloat16), keep
the elementwise maximum as a reported statistic, and print the worst layer
and the trend with depth in the summary. The current numbers stay in the
report; only the verdict and its default change.

### 6.2 Pairwise comparison and retained case matrices (F)

Today each case is compared with the baseline only, and the per-case mean
matrices are deleted after comparison. Add an all-pairs comparison table
(case i vs case j, per layer) and an option to retain each case's mean
matrices in the run directory as ordinary safetensors, so "batch 2 and
batch 4 agree with each other" and "A100 vs H100" can be measured rather
than inferred. Retained matrices are 6.6 GB per case for the 27B; make
retention explicit in the request and in the cost review.

### 6.3 Trim failure records (N)

A failed case's `reason` is the child's entire stderr, including hundreds
of progress-bar redraws. Keep the last twenty lines plus the first line
that names the exception, and put the full stderr in a file beside the
report.

### 6.4 Guide text (N)

The J-lens guide's benchmark section should state the measured ceiling
(batch 4 on 80 GB for a 27B in bfloat16), the 3.9× speedup, the H100/A100
ratio, and the precision finding in §4, with the caveat that these are
measurements on this model, corpus, and runtime.

### 6.5 Stopping-rule evaluation scope (N, for discussion)

Given §5's last point, consider letting the stopping rule declare which
layers it is evaluated over (all, or a late-layer band), recorded in the
identity as it is today.

## 7. Evidence

Workspace runs (all with custody receipts):

| Run | Content |
|---|---|
| `runs/jlens-benchmark-27b67f87…` | 27B round 1, A100 |
| `runs/jlens-benchmark-f42436fa…` | 27B round 1, H100 |
| `runs/jlens-benchmark-5929ab75…` | 27B round 2, A100 |
| `runs/jlens-benchmark-e32e0c69…` | 27B round 2, H100 |
| `runs/jlens-benchmark-fab8769b…` | 4B float32, H100 |
| `runs/jlens-benchmark-676004be…` | 4B bfloat16, H100 |

Round 1 ran on the engine before the hardware-provenance landing; rounds 2
and 3 carry the runtime hardware block. Per-row telemetry (seconds, device
memory before and after, host RSS) is in every completed case's
`telemetry.rowMeasurementsFile`.
