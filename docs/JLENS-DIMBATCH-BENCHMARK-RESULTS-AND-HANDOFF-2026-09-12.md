# J-lens dimension-batch benchmark: what ran, what we learned, what to change

- Date: 2026-09-12
- From: the maintainer's integration agent, on main `b1190f7`
- For: the refactor agents (§6 is the work list) and the study record
- Follows: `docs/JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md` §R1
  and the "Live acceptance" list in `docs/REVIEW-JLENS-SCALING-BRANCH-2026-09-11.md`
- Revised the same day after the refactor agents' review: the
  interpretation in §1, §4, and §5 was too confident in the first version.
  This version states what was measured, marks inferences as inferences,
  and withdraws the stopping-rule suggestion. The tables are unchanged.

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
   early layers in bfloat16, and the disagreement is strongly
   precision-sensitive.** The same comparison on a 4B model in float32
   agrees to 1e-5; in bfloat16 on that same model the gap is 13–14%. The
   gap decays smoothly with depth in every run. This makes a
   precision-independent batching mistake very unlikely; it does not by
   itself exclude a dtype-specific implementation difference, and the
   kernels responsible have not been identified (§4). Batch 2 and batch 4
   differ from batch 1 by the identical amount at every layer, which is
   strong circumstantial evidence that they agree with each other, but a
   direct comparison has not been made (§6.2).
5. **What this means, and does not mean, for the science:** we have
   measured the sensitivity of bfloat16 fits to one change of execution
   configuration on two short rows. That is not repeated-run variability,
   not corpus-sampling uncertainty, and not error against a precise
   reference; it says nothing measured about the published reference
   lenses. It does say that a bfloat16 early-layer matrix should not be read
   as exact to better than a few percent without evidence, and that the
   question that matters, whether the difference changes readouts, is open
   and answerable with the assessment operation (§5).
6. **The benchmark tool needs three changes** (§6): a relative agreement
   criterion, pairwise case comparison with saved case matrices, and
   trimmed failure records.

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

**Why batched and unbatched fits may differ in bfloat16 (inference).** At
batch 1 the vector-Jacobian products are matrix-vector shaped; at batch 2
and above they are matrix-matrix shaped, and the BLAS library may select
kernels with a different accumulation order. PyTorch documents that
batched and individual computations can differ numerically for this
reason. In bfloat16 (seven mantissa bits) the two paths would round
differently at every layer, and the backward pass carries the difference
through every layer between source and target, so a source at layer 0
accumulates the most and a source at the last layer almost none. That is
the observed shape, and it is consistent with batch 2 and batch 4 (which
share the matrix-matrix path) differing from batch 1 by the same amount.
The kernels actually dispatched have not been identified, and this
explanation has not been tested beyond the float32 control.

**What the float32 control does and does not show.** It shows that the
batching arithmetic (row alignment, reduction, dimension chunking) is
correct at float32 precision to about 1e-5. It does not exclude a
bfloat16-specific difference in the model's own forward or backward
(autocast behaviour, a fused kernel taking a different path for batched
input); such a difference would look the same in these measurements. The
signatures of a logic error, errors that persist in float32 and that do
not decay with depth, are absent.

**What has been measured about the instrument.** Sensitivity of a
two-row bfloat16 fit to dimension batching: 5–7% relative Frobenius at
layer 0 falling to 0.06% at layer 62 on the 27B, 13–14% falling to 0.1% on
the 4B. Not measured: variability across repeated runs of one
configuration, sensitivity to corpus sampling, error against a
sufficiently precise reference, and the corresponding numbers for the
published reference lenses. Independent fits taken along the same
numerical path could agree with each other while sharing a bias, so
cross-fit agreement is not a complete check either. The right record is
"observed sensitivity to dimension batching" with the values and
conditions above, not a general reproducibility floor.

**Whether it matters.** A global matrix norm cannot say whether a 5%
difference in an early-layer Jacobian changes any token prediction,
readout ranking, or conclusion a researcher would draw. That is a
held-out readout question, and the landed assessment operation
(`jlens-fit-assess`) answers it directly: fit the same rows at batch 1 and
batch 4, register both lenses, and compare their readouts with each other
and with the final residual on held-out text (§5).

**Cross-hardware.** The A100 and H100 agree with each other on every
aggregate (same peak memory, same 5.5% batched-versus-unbatched gap, same
3.9× speedup). Whether two fits on different GPU types agree at the matrix
level has not been measured, because a benchmark report holds agreement
numbers, not matrices (§6.2). The checkpoint identity permits merging
across GPU types; the numerical question stays open until two real fits
are compared.

## 5. Decisions

- **Use dimension batch 4 for the next, longer pilot, with batch 2 as the
  lower-memory fallback.** Peak memory grew 61.0 → 67.1 → 79.3 GB across
  batches 1, 2, 4, about 6 GB per unit of batch, so batch 5 projects to
  roughly 85 GB and cannot fit an 80 GB card; batch 4 is the largest
  feasible setting, not merely the largest tested. It leaves under 1 GB of
  measured headroom over two rows, and behaviour over a hundred rows
  (allocator fragmentation, the longest rows at `maxSeqLen`) is untested.
  Do not request 8.
- **Prefer the H100 when the queue offers it; accept the A100.** Single-GPU
  requests have started within minutes on both pools; four-GPU H100
  requests wait days, which is why a round of single-GPU shards is the
  right shape.
- **Budget, as an extrapolation from two rows:** at batch 4, a
  reference-scale 800-row fit is about 150 H100-hours or 235 A100-hours;
  eight shards bring that to roughly a day of wall time. Treat these as
  planning figures until the longer pilot measures them.
- **Next experiment, before a funded round:** fit the pilot's eight rows at
  batch 4 (the eight-row batch-1 lens is already registered), register the
  result, and run `jlens-fit-assess` between the two lenses and against
  the final residual on held-out text. This is the readout-level answer to
  whether the batching difference matters, and it is the assessment
  operation's live acceptance. **Done; see §5a.** Record the batching
  sensitivity from §3 in the qualification record of any lens fitted at
  batch 4, with its conditions.
- **Stopping rule: unchanged.** The running-mean-change statistic measures
  how much one more row moves the estimate; the benchmark measured how a
  change of computation moves it. They are different quantities, and
  nothing here justifies changing which layers the rule evaluates. (The
  first version of this document suggested otherwise; that suggestion is
  withdrawn.)

## 5a. Readout-level result (added the same night)

The experiment named in §5 ran. The pilot's eight rows were fitted at
dimension batch 4 on an H100 (1 h 29 min, about 640 s per row; the
running-mean-change statistic tracked the batch-1 fit row for row:
0.478 / 0.363 / 0.306 / 0.257 against 0.473 / 0.370 / 0.311 / 0.254 on rows
4–7). The result was registered and assessed against the batch-1 eight-row
lens with `jlens-fit-assess` on sixteen held-out 700-character windows cut
from the study's own case text (a different source from the fitting
corpus), 64 positions per row after skipping 16, top-10 overlap:

| Comparison | JS divergence, min / median / max over 63 layers | Top-10 overlap, layer 0 → 62 |
|---|---|---|
| batch-1 lens vs batch-4 lens | 0.0000 / 0.0001 / 0.0032 | 0.89 → 1.00 (1.00 from layer 48) |
| batch-1 lens vs final residual | 0.141 / 0.662 / 0.689 | 0.00 → 0.60 |
| batch-4 lens vs final residual | 0.141 / 0.662 / 0.689 | 0.00 → 0.60 |

At readout level the two lenses are the same instrument on this text: the
largest between-lens divergence at any layer is 0.003, the two lenses'
agreement with the final residual is identical to three decimals at every
layer, and their top-10 token sets coincide at 89% of positions even at
layer 0, where the matrices differ by 5%. The 5% early-layer matrix gap
does not change what a researcher would read. This is one held-out set of
sixteen rows from one source, so it is evidence for this model, corpus,
and text population, not a general guarantee; but it is the measurement
§4 said was needed, and it favours fitting at batch 4.

Two side observations. The assessment took five minutes on an H100 for
16 rows and 126 lens-layer reads (671 MB of staged activations), which is
the live acceptance of the assessment-reuse landing. And both eight-row
lenses agree poorly with the final residual at every layer (JS above 0.6
until the last few layers), which is what an eight-row fit should look
like against a reference recipe that used about 800 rows; it says nothing
about batching and everything about the funded round's row budget.

Run: `runs/jlens-assessment-f29facc3…` in the workspace, with receipt.
Requests: `requests/jlens-fit-27b-8rows-dimbatch4`,
`requests/jlens-assess-27b-batch1-vs-batch4`; held-out corpus
`prompts/fitting/jlens-heldout-ladder-12week` (derivation README in
`prompts/fitting/sources/`).

## 6. Work for the agents

### 6.1 Relative agreement criterion (F)

`jlens-fit-benchmark` decides `agrees` with the combined elementwise
criterion `|a−b| ≤ atol + rtol·|b|`. At raw Jacobian scale that criterion
fails the float32 4B cases whose relative Frobenius error is 1e-5
(elementwise maxima of 1.6e-3 to 3.6e-3), so the verdict is not
attributable to the absolute term alone; the elementwise criterion itself
is the wrong headline for matrices of this scale. Make the headline verdict
a per-layer relative Frobenius criterion with an explicit zero-norm rule
(default, say, 1e-3 for float32 and a researcher-declared value for
bfloat16), keep the elementwise maxima and the current tolerances as
reported diagnostics, record the selected criterion and threshold in the
report, and print the worst layer and the trend with depth in the summary.

### 6.2 Pairwise comparison and retained case matrices (F, first)

Today each case is compared with the baseline only, and the per-case mean
matrices are deleted after comparison. Add an all-pairs comparison table
(case i vs case j, per layer) and an option to retain each case's mean
matrices in the run directory as ordinary safetensors, so "batch 2 and
batch 4 agree with each other" and "A100 vs H100" can be measured rather
than inferred. Retained matrices are 6.6 GB per case for the 27B; make
retention explicit in the request and in the cost review. This comes
first because §1 item 4 and §4 rest on it.

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
