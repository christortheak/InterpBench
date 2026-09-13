# J-lens numerical testing program: results and handoff

Date: 2026-09-13.
Status: program T0–T7 complete; this is the handoff to the coding, reviewing,
and running agents. The decision it supports is
`docs/JLENS-NUMERICS-PROGRAM-DECISION-2026-09-13.md`; the program it executes
is `docs/JLENS-NUMERICAL-REVIEW-AND-TEST-PROGRAM-2026-09-13.md`. Everything
below was run by the researcher's session on the study cluster and the Mac
between 08:46 and 13:20 local time; every number is read from a report file
listed in §9, and every interpretation is written beside its report in the
study workspace (`diagnostics/jlens-numerics-2026-09-13/INTERPRETATION-*.md`).
Paths below are workspace-relative unless stated.

Summary in one paragraph (see §10 for the 2026-09-13 corrections). The
estimator is exact in float64, the engine path reproduces the reference
kernel's batch-shape distances (direct matrix identity is J2 work), batch-1
fits are bitwise repeatable, and the 5–16% early-layer matrix gap between batch-1 and batched
fits is bfloat16 forward-pass kernel selection that appears on dense models
as much as on the hybrid. A 1000-prompt Gemma-3-4B reproduction of the
published recipe registers cleanly and is closer to the model's own final
distribution than the published 546-prompt lens at all 33 source layers on
two held-out corpora. On the same readout metric, at the five sampled layers, both Jacobian
lenses beat the plain logit lens at layer 8, tie it in JS at layer 32, and
are clearly worse at layers 17 and 25. The dense Gemma-3-27B fits at 26 rows per hour at dimension batch 4 in
64 GB on an H100. Work orders for the engine are in §6.

## 1. Fixed identities

| Item | Value |
|---|---|
| Reference kernel | `jlens` @ 581d398613e5602a5af361e1c34d3a92ea82ba8e (pinned extra); kernel sha256 5be8959d…, driver sha256 32eb0b6f… in every report |
| Engine | main 3b42c98 on the cluster and the Mac throughout |
| 4B model | google/gemma-3-4b-it @ 093f9f388b31de276ce2de164bdc2081324b9767 (hidden 2560; 33 source layers, target 33) |
| 27B hybrid | Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (hidden 5120; 63 source layers, target 63; linear-attention fast path unavailable, PyTorch fallback) |
| 27B dense | google/gemma-3-27b-it @ 005ad3404e59d6023443cb575daa05336842228a (61 source layers, target 61) |
| Published lenses | neuronpedia/jacobian-lens @ a4114d7752d11eb546e6cf372213d7e75526d3a1; the 4B lens is stored as float16 (33 layers, 546 prompts, dim_batch 128, B200, compile on) and is registered as `custom-lens-976f9c12fd854cbeb8ba29d5eb1e382f`; 27B lens 828 prompts, dim_batch 64 |
| Fitting corpus | `prompts/fitting/jlens-wikitext103-train-first1000` (first 1000 WikiText-103-raw-v1 train records with ≥600 stripped characters; dataset revision b08601e04326c79dfdd32d625aee71d232d685c3) |
| Held-out, general | `prompts/fitting/jlens-heldout-wikitext103-validation-first64` (64 records, same rule, validation split) |
| Held-out, application | `prompts/fitting/jlens-heldout-ladder-12week` (16 windows of 700 characters of the study case text) |
| Benchmark rows | `prompts/fitting/jlens-pilot-neutral-27b` (neutral rows, 115 and 119 tokens for the first two) |
| Token rule | maxSeqLen 128, skipFirst 16; assessments read 64 positions per row after the skip; bf16 readout through the output head unless stated |
| Runtime | eager attention, compile off, TF32 matmul off, float32 matmul precision "highest", deterministic algorithms off, kernel policy "current" |

Unknowns that stay unknown: the published script's record filter (only
max_chars 2000 is declared) and its exact stopping-window semantics (see §3
T5).

## 2. What ran

| Test | Where | Job(s) | Wall time | Output |
|---|---|---|---|---|
| T0 spec | Mac | — | — | `diagnostics/jlens-numerics-2026-09-13/SPEC.md` |
| T1 apparatus, T2 repeatability, T4 localization (one 128-token WikiText validation row, batch 1 vs 4) | H100, A100 (4B); H100 (27B hybrid) | hand sbatch, `jlens_numerics_diag.py` | minutes | `gemma4b-bf16/`, `gemma4b-bf16-a100/`, `qwen27b-bf16/` (report.json + summary.txt) |
| T1 logit-lens baselines on the assessment corpora | H100 | hand sbatch | seconds after load | `gemma4b-bf16-t1-wikitext64/`, `gemma4b-bf16-t1-ladder16/` |
| T3 derivative check | Mac CPU, float64 | — | minutes | `t3-hybrid-fd/report.json` |
| T4b kernel path + per-layer forward scan (neutral rows, batch 1 vs 4) | H100 and A100 (4B), H100 (27B hybrid scan) | hand sbatch, `jlens_numerics_t4b.py` | 8–14 min | `t4b-gemma4b/`, `t4b-gemma4b-a100/`, `t4b-qwen27b/` |
| 4B batch sweep 1/16/32/64 (managed benchmark) | H100 | ae6bfdc0 | ~5 min | `runs/jlens-benchmark-ae6bfdc0a20b4e09b8a38a8e58636996` |
| T5 reproduction fit (managed `jlens-fit`, dimBatch 64, 1000 rows, stopping {0.002, 10, 100}, checkpoint every 25) | H100 | fc1a8d162aaa | 3 h 05 min | `runs/jlens-fit-ccdaa9738cbe4fd0801302ccd69c566a`; lens `custom-lens-895a4f3a7dae43c3878ab081c703c365` |
| T5 assessments (managed `jlens-fit-assess`, candidate = ours, reference = published) | H100 | 245c594100da (WikiText 64), a2763491c013 (ladder 16) | 2 min, 1 min | `runs/jlens-assessment-2da5e363cd8a4f978be44134aac3a5bb`, `runs/jlens-assessment-badd5838ca434a7c9ba705e90b9c4bfc` |
| T5 direct matrix comparison | Mac | — | — | `diagnostics/jlens-numerics-2026-09-13/t5-matrix-comparison.json` |
| T6 dense 27B benchmark 1/2/4 (managed) | H100 | 7e9c8988c6dd | 45 min | `runs/jlens-benchmark-c9f1b6e4b7ef49298ae37b384021fbdd` |

All managed jobs went through interview/draft/publish → input-plan → package →
stage → plan → submit with `--gpu-type H100`, export/fetch, and
`verify-custody`; every receipt verified. The hand-submitted diagnostics ran
outside the engine on purpose, as an independent apparatus; their scripts are
copied beside their outputs.

## 3. Results by test

### T1 — the assessment's measuring apparatus is exact

- The wrapper's `unembed` of the final residual reproduces the model's native
  logits bit for bit at every assessed position (JS ≤ 1.3e-16, top-1
  identical), on both models and both cards.
- A lens compared with itself gives JS ≤ 1.1e-16 and overlap 1.0.
- Float32 readout vs the bf16 output head, same residuals: mean JS 7.0e-5,
  p95 2.1e-4, max 1.1e-3; top-1 agreement 0.98; top-10 0.99. On the hybrid:
  mean JS 6.6e-5, top-1 0.975.
- Logit-lens (identity transport) baseline, 4B, 64 WikiText validation rows,
  4096 positions, mean JS / top-10 overlap: L0 0.690 / 0.004; L8 0.684 /
  0.009; L17 0.597 / 0.239; L25 0.295 / 0.479; L32 0.070 / 0.791. Ladder
  windows (1024 positions): L17 0.576 / 0.300; L25 0.249 / 0.533; L32 0.053 /
  0.803. Hybrid 27B (16 WikiText rows): L0 0.648; L31 0.651; L48 0.620; L62
  0.138 (eight ladder windows, 16 components; not WikiText rows — §10).

### T2 — batch-1 execution is bitwise repeatable and order-invariant

Same retained graph twice, descending component order, seeded-shuffled order,
and a fresh forward: bitwise identical component rows at every tested layer
(4B: 0, 8, 17, 25, 32 on H100 and A100, 32 components; hybrid 27B: 0, 16,
31, 48, 62 on H100 only, 16 components, eight ladder windows), without
deterministic-algorithm flags. Component subsets are subsets (§10). The repeatability baseline is zero,
so every nonzero batch-shape difference below is a real execution difference.

### T3 — the estimator is mathematically correct

Float64, CPU, a tiny random hybrid (two linear-attention and two
full-attention blocks, 12 tokens, skipFirst 2). The reference
`jacobian_for_prompt` at batches 1, 2, 4 equals an independent
`torch.autograd.functional.jacobian` construction to relative Frobenius
5.3e-8–5.9e-8 at every layer; batches are bitwise identical; central finite
differences agree at their floor (≤ 3e-4 relative at step 1e-4, worse at
smaller steps as round-off dominates); transposed reference, wrong position
mask, and sum-instead-of-mean reduction are all detected.

### T4 and T4b — the batch-shape gap is bf16 forward-pass kernel selection

Per-layer relative Frobenius between single-row and replicated-batch
execution (forward residuals) and between batch 1 and batch 4 estimator rows
(backward):

| Model, card, row | Forward | Backward (Jacobian rows) |
|---|---|---|
| 4B, H100, WikiText 128 tokens | bitwise identical, every layer | bitwise identical |
| 4B, H100, neutral 115 / 119 tokens | 5.3e-5 at L0 → 6.4e-3 at L33 | L0 6.0e-2 / 1.6e-1; L32 1.5e-3 / 2.3e-3 |
| 4B, A100, WikiText 128 tokens | 2.1e-3 at L0 → 1.3e-2 at L33 | L0 1.1e-1; L17 2.4e-2; L32 3.0e-3 |
| 4B, A100, neutral rows | 1.7e-3 at L0 → 7.7e-3 at L33 | L0 8.3e-2 / 9.5e-2; L32 2.3e-3 / 1.8e-3 |
| 27B hybrid, H100, WikiText 128 tokens | 2.6e-5 at L0 → 1.2e-2 at L48, plateau | L0 5.7e-2; L31 2.2e-2; L62 8.2e-4 |
| 27B hybrid, H100, neutral row (scan) | 2.6e-5 at L0 (linear attention), 1.7e-3 by L2, 1.2e-2 from L32 | — |

Findings.

1. The difference begins in the forward pass at the first block (a
   sliding-attention block on the 4B, a linear-attention block on the 27B)
   and grows with depth; the Jacobian inherits it and amplifies it at shallow
   source layers because a layer-0 row's backward pass traverses every
   downstream block. Forward 1e-5 at layer 0 becomes 6–16% in layer-0
   Jacobian rows and 0.1–0.3% in the deepest rows.
2. It is not a hybrid-architecture effect: dense Gemma 4B shows it as
   strongly. It is not an estimator effect: T3 is exact in float64. The
   engine-wrapped estimator (model loaded through `jlens_fit_model.load`,
   telemetry wrapper, kernel observation) reproduced the bare reference's
   batch-4-vs-batch-1 distances at every layer for both rows on both cards;
   the direct engine-vs-bare matrix comparison at the same batch was not
   made by T4b and is the first J2 control (§10).
3. It depends on card and sequence shape, and the paths are discrete. On the
   H100 the 128-token row is bitwise identical across batch shapes while 115-
   and 119-token rows are not; on the A100 every row differs. In the
   benchmark reports, dense 4B batches 4, 16, 32 and 64 share one identical
   agreement record against batch 1 while batch 2 has another; dense 27B
   batches 2 and 4 share one; hybrid 27B batches 2 and 4 share one on the
   H100 but not on the A100. Identical records against batch 1 are strong
   evidence that the tensors coincide, not a direct comparison; the tensors
   were not retained, and J2 retains and compares them (§10). The pattern is
   consistent with GEMM algorithm selection by problem shape (M = batch ×
   tokens) with bf16 reduction-order rounding; the chosen algorithm was not
   captured (`kernelDispatch.executedModules` is empty; only an explicit
   disabled fast path is observable) and backward arithmetic may contribute,
   so the mechanism is an inference from the pattern.
4. Therefore batch-1, batched, and published fits are outcomes of the same
   bf16 rounding process under different shapes; none is the reference for
   the others. They are deterministic per configuration and not proven
   unbiased draws, so averaging across arbitrary configurations is not shown
   to cancel them (§10). The float32 4B
   benchmark (`runs/jlens-benchmark-fab8769bcdbf48469e7f523de6e5de01`,
   batches agree to 1e-5) is the batch-insensitive reference when one is
   needed.

### T5 — reproducing the published Gemma-3-4B lens

Fit. Same model, dataset, split, length limits, precision, and stopping rule
as the published recipe; dimension batch 64 (not 128), no compile, H100 (not
B200), stand-in prompt filter. 1000 of 1000 rows fitted, none skipped, 128
tokens and 111 valid positions every row, 10.9 s per row, 3 h 05 min. The
stopping rule never fired: the statistic (the reference kernel's exact
formula, max over source layers of ‖J_i − mean‖ / ((n+1)‖mean‖)) fell
roughly as 1/n — rows 100–200 mean 0.022; 500–600 mean 0.0062, min 0.0031;
900–1000 mean 0.0039, min 0.0018 — and dipped below 0.002 on one row. The
published fit reports final_mean_rel_change 0.00179 at 546 prompts. Either
the published script averaged the statistic over its window (ours requires
ten consecutive rows strictly below threshold) or its prompt set was more
homogeneous; the rule as we implement it would not have stopped this fit
within 1000 prompts.

Matrices. Against the published lens (float16 vs our float32, both promoted
to float64): relative Frobenius 0.269 (L0), 0.226 (L5), 0.150 (L10), 0.091
(L16), 0.057 (L24), 0.017 (L32); cosine 0.965 → 0.9999; norm ratio within 3%
at every layer; both lenses are 25× the identity at layer 0 and 0.96× at
layer 32. The difference is sampling variance over different prompt sets
(546 vs 1000) plus the bf16 batch-path effect of T4; the comparison cannot
separate them.

Readout, between the lenses (mean JS / top-10 overlap):

| source layer | WikiText 64 rows | ladder 16 windows |
|---|---|---|
| 0 | 0.100 / 0.82 | 0.087 / 0.84 |
| 4 | 0.283 / 0.64 | 0.314 / 0.63 |
| 5 | 0.186 / 0.75 | 0.201 / 0.74 |
| 8 | 0.085 / 0.83 | 0.094 / 0.82 |
| 16 | 0.034 / 0.89 | 0.032 / 0.90 |
| 24 | 0.006 / 0.96 | 0.005 / 0.96 |
| 32 | 0.0003 / 0.98 | 0.0004 / 0.98 |

Layers 4 and 5 are outliers: readout disagreement there is three times the
neighbours although the matrices differ no more there than at layers 0–3.
The readout at those layers is unusually sensitive to the lens (§7).

Readout, each lens against the model's final distribution: our lens is
closer than the published lens at every one of the 33 source layers on both
corpora, by 0.0002–0.0065 JS (WikiText L8 0.677 vs 0.682; L25 0.435 vs 0.440;
L32 0.070 vs 0.072; ladder L25 0.442 vs 0.448). Sixty-six cells, one sign.
The cause is not identified: prompt set, batch, compile, hardware, and
storage precision all differ between the two fits (§10).

Readout, against the logit lens (same positions, same bf16 readout):

| source layer | logit lens, WikiText 64 | our J-lens, WikiText 64 | logit lens, ladder | our J-lens, ladder |
|---|---|---|---|---|
| 0 | 0.690 / 0.004 | 0.691 / 0.006 | 0.685 / 0.007 | 0.693 / 0.003 |
| 8 | 0.684 / 0.009 | 0.677 / 0.028 | 0.682 / 0.010 | 0.677 / 0.025 |
| 17 | 0.597 / 0.239 | 0.645 / 0.066 | 0.576 / 0.300 | 0.653 / 0.048 |
| 25 | 0.295 / 0.479 | 0.435 / 0.299 | 0.249 / 0.533 | 0.442 / 0.286 |
| 32 | 0.070 / 0.791 | 0.070 / 0.766 | 0.053 / 0.803 | 0.061 / 0.761 |

Both Jacobian lenses (the published one tracks ours within 0.006 JS) sit at
the JS ceiling (ln 2 = 0.693) with the logit lens at layer 0, beat it at
layer 8 on both metrics on both corpora, are equal in JS but lower in
top-10 overlap at layer 32, and are clearly worse at layers 17 and 25. The
baseline was measured at these five layers only (§10). This is a
property of the method as read out by our assessment (J_l · h_l, then final
norm and unembedding, the reference kernel's convention), not of our fit. A
mean-Jacobian lens is not fitted to minimise readout divergence and its
intended use is transporting directions and features, so this is not a
verdict on the lens; it is a verdict on "agreement with the final
distribution" as a reason to prefer a J-lens over a logit lens on this model.

Not done: intermediate-budget assessment (546 rows, say). The fit retains
only its final checkpoint, so it needs either checkpoint retention (§6, item
5) or a deliberate 546-row fit.

### T6 — the dense 27B is affordable and shows the same pattern

| dimBatch | seconds (2 rows) | rows/hour | peak device GB | vs batch 1 |
|---|---|---|---|---|
| 1 | 771 | 9.3 | 57.4 | reference |
| 2 | 411 | 17.5 | 59.7 | L0 6.6e-2 → L60 8.9e-4 |
| 4 | 279 | 25.8 | 64.3 | identical record to batch 2 |

The hybrid 27B on the same card: 1.4 rows/hour at batch 1 (61 GB), 5.4 at
batch 4 (79.3 GB, the ceiling). The dense model is 6.8× faster at batch 1 and
4.8× at batch 4 with 15 GB to spare. The published 828-prompt recipe is about
32 GPU-hours on one H100 at batch 4, or 8 hours across four shards.

## 4. What we learned, in order of consequence

1. **The numerical path is correct as far as tested.** Estimator mathematics
   (T3, float64, tiny hybrid), apparatus (T1), engine-path distances (T4b),
   and repeatability (T2) all pass at their floors; the direct engine-vs-bare
   matrix identity is J2 work. The
   earlier handoffs' worry that the 27B numerical path might be wrong is
   closed.
2. **Batch shape is a rounding draw, not a defect.** Fit at the largest
   feasible dimension batch on whatever card the queue offers; merge shards
   across cards. Record card and batch (already in shard identity and
   `runtimeHardware`) so draws stay distinguishable. Stop describing batch 1
   as the reference.
3. **Judge lenses by readout, with the right baselines.** The readout
   apparatus is exact; the bf16 readout of the final residual contributes JS
   7e-5 and 2% top-1 flips, which is the scale of between-batch lens
   differences, so a float32 readout must be available (this figure is not a
   bound on transported intermediate residuals). The logit-lens baseline
   must sit beside every J-lens readout: on Gemma-3-4B the J-lens beats it
   at layer 8 and loses to it at layers 17 and 25.
4. **The managed reproduction succeeds and compares favourably with the
   published lens** by the readout metric at every layer under documented
   recipe differences (cause of the margin unidentified), which validates
   the whole managed chain
   (corpus preparation, fit, export, custody, registration, assessment) on a
   second model family and a real recipe.
5. **The stopping rule as implemented is stricter than the published one
   in practice**, and its statistic is noisy row to row (0.002–0.010 at
   n ≈ 950). Treat the row budget as the operative stop until §6 item 4 is
   settled.
6. **Early-layer readouts on Gemma-3-4B are lens-sensitive at layers 4–5**
   in a way the matrices do not predict. Investigate before any early-layer
   use (§7).

## 5. Corrections to earlier documents

These supersede the corresponding passages of
`docs/JLENS-DIMBATCH-BENCHMARK-RESULTS-AND-HANDOFF-2026-09-12.md` and
`docs/JLENS-BATCH-ASSESSMENT-HANDOFF-2026-09-13.md`.

- The "5–14% early-layer matrix gap" is a forward-pass bf16 rounding
  difference, discrete in batch shape, present on dense models; it is not a
  hybrid-kernel or batched-path defect and not evidence about convergence.
- The benchmark's `agrees` flag (rtol/atol 1e-4 against batch 1) is false
  for every bf16 batched case on every model tested and carries no
  information (§6 item 1).
- "Poor agreement with final predictions" at shallow layers is shared by the
  logit lens and by the published lens; it is a property of shallow residuals
  and of the readout, not of an under-fitted lens.
- The readout precision question is answered: JS 7e-5, top-1 0.98 (T1).

## 6. Work orders for the engine (priority order)

1. **Benchmark operation (`jlens-fit-benchmark`).** Retain every case's
   lens tensors in the run directory; add pairwise agreement between all
   batched cases (not only against batch 1); replace the `agrees` flag with
   a relative criterion: report each case's relative Frobenius against a
   float32 repeat of batch 1 when the request carries `dtype float32`, and
   otherwise label bf16 agreement rows "rounding draw, not a defect
   criterion". Keep `kernelDispatch` but state its limitation in the report
   header. Evidence: §3 T4/T6.
2. **Assessment operation (`jlens-fit-assess`).** Add `readoutDtype`
   (bfloat16 default, float32 option) and report both when float32 is
   chosen; add the identity-transport (logit-lens) baseline as a third
   readout column at every layer, computed from the same staged activations
   (it costs one unembed per layer and needs no lens); add a per-layer
   readout-sensitivity number (JS between the two lenses divided by their
   matrix relative Frobenius) so outliers like layers 4–5 are visible in the
   report. Evidence: §3 T1, T5.
3. **Assessment operation, second.** Accept a `baselineOnly` request (no
   candidate lens) so the logit-lens baseline can be measured through the
   managed chain instead of a hand script; the hand-run T1 numbers above
   should be reproducible by it.
4. **Fit operation (`jlens-fit`) stopping.** Keep the reference statistic
   and the consecutive-window semantics as the recorded default; add an
   optional `windowMean` rule variant (mean of the last `window` values below
   threshold) so a fit can be run with either semantics, and record both
   window statistics in `progress.jsonl` and `fit-report.json` whether or
   not either rule is active. Document in the guide that the published
   fit's window semantics are unknown and that the two variants differ by a
   factor of about 3 in effective strictness on this corpus. Evidence: §3 T5.
5. **Fit operation checkpoints.** Add `retainCheckpointsAt` (a list of row
   counts) so a fit keeps immutable snapshots of `sums` at preselected
   budgets alongside the rolling checkpoint; each retained snapshot must be
   importable as a lens with its row count in provenance. This is what the
   546-vs-1000 comparison needs.
6. **Fit operation records.** Expose `windowsPerRecord` in the corpus
   preparation and fit config (the agents' earlier item) and trim failure
   records to the fields a reader needs (status, index, id, reason, tokens).
7. **Guide text (`docs/` J-lens guides and the science guide entries).**
   Describe batch-1 and batched fits as draws of one rounding process; drop
   "reference" for batch 1; name the float32 benchmark as the
   batch-insensitive check; require the logit-lens baseline beside any J-lens
   readout claim; state the readout-precision numbers.
8. **Provenance surfacing.** `runtimeHardware` and shard identity already
   carry card and dimension batch; surface them in the merge report and the
   Mac lens inspector.

Each item should land with a test that reads a fixture report and asserts the
new columns, and the guide text should be checked by `check-generated.py
--audits` as usual. None of these changes the plan hash of an existing
request except item 4 when the variant is selected and item 5 when the list
is given.

## 7. Open scientific questions (not engine work)

- **Layers 4–5 readout sensitivity on Gemma-3-4B.** Between-lens JS peaks at
  0.28–0.31 there against 0.09–0.10 at layers 0–3 and 8, while matrix
  differences are flat across layers 0–5. Hypothesis: high-norm residual
  directions in Gemma's early blocks make the readout of J · h dominated by
  a few coordinates where the two lenses differ. Test: per-coordinate
  contribution to the readout logits at layers 3–6 for a handful of
  positions; compare the two lenses' columns for the top-norm coordinates.
- **J-lens versus logit lens at middle depth.** On this readout metric the
  J-lens is worse at layers 17–25 for both lenses. Whether the intended use
  (direction transport for feature attribution) is affected is a different
  question, answered by transporting known directions and checking effects,
  not by readout divergence. Do not fit a 27B lens for readout purposes on
  the strength of this evidence; fit it if direction transport is the use.
- **Stopping semantics of the published script.** Recoverable only by
  finding the script; until then, report both window statistics (§6 item 4).
- **Hybrid-27B early layers.** The hybrid's logit-lens baseline is flat at
  JS ≈ 0.65 from layer 0 to layer 48 and only reaches 0.14 at layer 62; the
  8-row lenses assessed earlier were no better. Whether a full-budget hybrid
  lens changes that is untested and, given the 4B result, should not be
  assumed.

## 8. Operating numbers and notes

| Model | Card | dimBatch | rows/hour | peak GB |
|---|---|---|---|---|
| Qwen3.8-27B hybrid | A100 | 1 / 2 / 4 | 0.9 / 1.7 / 3.4 | 61 / 67 / 79 |
| Qwen3.8-27B hybrid | H100 | 1 / 2 / 4 | 1.4 / 2.7 / 5.4 | 61 / 67 / 79 |
| Gemma-3-27B dense | H100 | 1 / 2 / 4 | 9.3 / 17.5 / 25.8 | 57 / 60 / 64 |
| Gemma-3-4B dense bf16 | H100 | 1 / 4 / 16 / 64 | 36 / 137 / 308 / 325 | 9 / 11 / 18 / 45 |
| Gemma-3-4B dense float32 | H100 | 1 / 4 | 38 / 57 | 18 / 21 |
| Gemma-3-4B, T5 fit | H100 | 64 | 330 | — |

- Per-request GPU type worked throughout (`--gpu-type H100` on plan and
  submit; `requestedGPUType` and `deviceName` in every record).
- Export of the 1.6 GB fit bundle took under two minutes; the two 1.2 GB
  assessment closures each staged, planned, and submitted in about four
  minutes; the 12 GB 27B closures of the previous days remain the slow case.
- The 4B model loads from cache in seconds, so its diagnostics are
  queue-bound, not compute-bound.
- The controller was started for the managed chain and stopped after the
  last fetch; hand diagnostics never needed it.

## 9. Evidence index

Workspace `diagnostics/jlens-numerics-2026-09-13/`: `SPEC.md`,
`INTERPRETATION-T1-T2.md`, `INTERPRETATION-T3.md`, `INTERPRETATION-T4.md`,
`INTERPRETATION-T5.md`, `INTERPRETATION-T6.md`, `t5-matrix-comparison.json`,
`gemma4b-bf16/`, `gemma4b-bf16-a100/`, `qwen27b-bf16/`,
`gemma4b-bf16-t1-wikitext64/`, `gemma4b-bf16-t1-ladder16/`, `t3-hybrid-fd/`,
`t4b-gemma4b/`, `t4b-gemma4b-a100/`, `t4b-qwen27b/`, the diagnostic scripts
and sbatch files, and the three T4b job logs.

Workspace `runs/`: `jlens-fit-ccdaa9738cbe4fd0801302ccd69c566a` (T5 fit),
`jlens-assessment-2da5e363cd8a4f978be44134aac3a5bb` (WikiText),
`jlens-assessment-badd5838ca434a7c9ba705e90b9c4bfc` (ladder),
`jlens-benchmark-c9f1b6e4b7ef49298ae37b384021fbdd` (T6),
`jlens-benchmark-ae6bfdc0a20b4e09b8a38a8e58636996` (4B sweep),
`jlens-benchmark-fab8769bcdbf48469e7f523de6e5de01` (4B float32),
`jlens-lenses/custom-lens-895a4f3a7dae43c3878ab081c703c365` (our lens),
`jlens-lenses/custom-lens-976f9c12fd854cbeb8ba29d5eb1e382f` (published).

Workspace `requests/`: `jlens-fit-gemma4b-wikitext-reproduction`,
`jlens-assess-t5-4b-repro-vs-published-wikitext`,
`jlens-assess-t5-4b-repro-vs-published-ladder`,
`jlens-fit-gemma27b-bfloat16-bench2`, `jlens-bench-gemma27b-bfloat16-dimbatch`,
`jlens-bench-4b-bfloat16-dimbatch-large`, each with `H100/remote-plan.json`,
`H100/remote-submit.json`, and `H100/plan-hash`.

## 10. Corrections of 2026-09-13 (J1 of the closure handoff)

Applied after the refactor agents' closure review (baseline 168aee8); the
same six corrections are recorded in §8 of the decision document. Reports
are unchanged; the text above is amended in place with pointers here.

| # | Original claim | Corrected statement | Evidence to come |
|---|---|---|---|
| 1 | Engine path reproduces the bare reference "to the bit" | T4b compared batch-4-vs-batch-1 distances separately in each path; equal distance summaries at every layer, no direct matrix comparison | J2: engine vs bare at the same batch, same tokens, layers, components; hashes over a documented serialization; mutation and permutation controls |
| 2 | Batches 2 and 4 (dense 27B), 4 through 64 (dense 4B) "coincide" | Identical agreement records against batch 1; tensors not retained, so an inference | J2: retained tensors and direct pairwise comparison, plus a repeat of one configuration |
| 3 | J-lens "does not beat the logit lens at any depth"; "at chance" at layers 0–8 | Better at layer 8 on both metrics and corpora; at the JS ceiling at layer 0; five sampled layers only | none needed; numbers already in §3 T5 |
| 4 | Batch shapes are "draws of the same rounding process" that merging averages out | Deterministic per configuration; mechanism inferred, kernel selection not captured; mixed-GPU scientific equivalence unqualified | J2 deferred item: matched small fits across proposed configurations vs a uniform configuration on a fixed corpus budget |
| 5 | "A 1000-prompt mean estimates the population mean Jacobian better" | Managed fit succeeds and compares favourably; cause of the margin unidentified (prompt set, batch, compile, hardware, storage precision differ) | retained-checkpoint or nested-budget fit (§6 item 5) |
| 6 | T1/T2 "both models, both cards", hybrid "16 WikiText rows" | Hybrid: eight ladder windows, H100 only, 16 components; dense 4B: 16 WikiText rows, both cards, 32 components; T3 certifies mathematics only; the readout figure is for the final residual | none |
