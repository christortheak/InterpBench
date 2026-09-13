# J-lens numerics: findings and decision (T0–T7), 2026-09-13

This closes the numerical testing program proposed in the agents' 2026-09-13
review of the J-lens fit studies. Every claim below is backed by a file in
the study workspace under `diagnostics/jlens-numerics-2026-09-13/` (the
per-test `INTERPRETATION-*.md` files and their `report.json`) or by a fetched
run under `runs/`. Nothing here rests on the earlier benchmark and assessment
handoffs' interpretations; where those were wrong, this document supersedes
them.

## 1. Decision

Fit J-lenses in bfloat16 at the largest dimension batch the card allows
(4 on an 80 GB card for the hybrid 27B; 64 or more for the dense 4B;
4 fits the dense 27B in 64 GB), on whichever GPU type the queue offers, and
merge shards across cards. Judge a fitted lens by readout agreement against
the model's own next-token distribution, with a float32 readout available,
and by the stopping statistic; never by bitwise or tolerance agreement with
another bfloat16 fit. Record card and dimension batch in shard identity and
provenance (already done) so different draws stay distinguishable.

## 2. What each test established

| Test | Question | Result | Evidence |
|---|---|---|---|
| T0 | What exactly do we test and what counts as passing? | Specification written before any run | `SPEC.md` |
| T1 | Does the assessment's "final prediction" equal the model's own logits, and how much does the bf16 readout cost? | Readout reproduces native logits bit for bit; bf16-vs-fp32 readout moves the distribution by mean JS 7e-5 and flips top-1 at ~2% of positions | `INTERPRETATION-T1-T2.md` |
| T2 | Is a batch-1 fit repeatable and order-invariant? | Bitwise identical across repeats, descending and shuffled component orders, and fresh forwards, on both models and both cards | same |
| T3 | Is the estimator mathematically correct? | In float64 on a tiny hybrid, the reference estimator equals an independent explicit Jacobian to 6e-8, batches 1/2/4 are bitwise identical, finite differences agree at their floor, and three deliberate mutations are detected | `INTERPRETATION-T3.md` |
| T4 | Where does the batch-shape difference arise? | In the bf16 forward pass itself, from the first block, on dense and hybrid models alike; card- and sequence-length-dependent; the engine path is bit-identical to the bare reference | `INTERPRETATION-T4.md` |
| T5 | Does our pipeline reproduce the published Gemma-3-4B lens's recipe and quality? | Yes at readout level: our 1000-prompt lens is closer to the model's final distribution than the published 546-prompt lens at every source layer on both held-out corpora; the two lenses differ 10–30% at shallow layers and converge with depth; neither beats the plain logit lens at any depth on this readout metric | `INTERPRETATION-T5.md` |
| T6 | Is a dense 27B affordable and does the same batch pattern appear? | 26 rows/hour at batch 4 on an H100 (hybrid: 1.4 at batch 1); batch 2 and 4 identical against batch 1; same depth profile | `INTERPRETATION-T6.md` |

## 3. The batch-shape difference, resolved

The earlier handoffs treated the 5–16% early-layer matrix difference between
batch-1 and batched fits as a property of the estimator or of the hybrid
architecture. The program shows it is neither.

- The estimator is exact (T3) and the engine's wrapping of it changes no bit
  (T4b).
- The bf16 forward pass returns different residuals when the same row is
  presented alone versus replicated, starting at layer 0 (1e-5 to 1e-3
  relative) and growing with depth to about 1e-2. Dense Gemma 4B shows it as
  strongly as the hybrid Qwen 27B and the dense Gemma 27B.
- Whether it appears depends on the card and the sequence length: on the
  H100 a 128-token row is bitwise identical across batch shapes and 115- and
  119-token rows are not; on the A100 every tested row differs. The paths
  are discrete: on the dense 27B batch 2 and batch 4 agree with each other
  and only batch 1 differs; on the dense 4B batches 4 through 64 agree with
  each other while batch 2 and batch 1 each differ.
  This is the signature of GEMM algorithm selection by problem shape: a
  single-row matmul and a batched matmul may run different kernels with
  different reduction orders, and bf16 keeps about three significant digits
  per partial sum. The diagnostics cannot observe the chosen algorithm, so
  this is an inference from the pattern rather than a captured dispatch.
- The Jacobian inherits the forward difference and amplifies it at shallow
  source layers (6–16% at layer 0, 0.1–0.3% at the deepest source layer)
  because a layer-0 row's backward pass traverses every downstream block.
- Consequently a batch-1 fit, a batch-4 fit, and the published fits (batch
  128 and 64 on a B200 with compile on) are three draws of the same bf16
  rounding process. None is the reference for the others. The float32 4B
  benchmark, where batch shapes agree to 1e-5, is the batch-insensitive
  reference when one is needed.

## 4. What this means for the readout-level assessment

T1 fixes the meaning of the assessment's numbers. "Final prediction" is
exactly the model's distribution. The bf16 readout contributes a mean JS of
7e-5 and a 2% top-1 flip rate on its own, which is the same order as the
batch-1-vs-batch-4 between-lens JS (median 1e-4, max 3.2e-3). Future
assessments should offer a float32 readout so lens differences are not
confounded with readout rounding. The logit-lens baseline (identity
transport) is recorded for both models by depth, so "better than no
transport" is now a measurable claim rather than an assumption.

## 5. T5 — reproduction of the published Gemma-3-4B lens

The fit: same model, dataset, split, length limits, precision and stopping
rule as the published recipe; dimension batch 64 instead of 128, no compile,
H100 instead of B200, and a stand-in prompt filter (first 1000 WikiText-103
train records with at least 600 characters) because the published script's
filter is not recoverable. 1000 rows in 3 h 05 min at 10.9 s per row. The
stopping rule never fired: the statistic (the reference kernel's exact
formula) decayed roughly as 1/n but stayed 2–5× above the published fit's
final value at equal prompt counts and dipped below 0.002 on one row only.
The published script either averaged the statistic over its window or fitted
a more homogeneous prompt set; the rule as implemented would not have stopped
this fit within 1000 prompts.

The lens: registered as `custom-lens-895a4f3a7dae43c3878ab081c703c365`.
Against the published matrices the relative Frobenius difference is 0.27 at
layer 0 falling to 0.017 at layer 32 (cosine 0.965 → 0.9999, norms within 3%);
both lenses are 25× the identity at layer 0, so they encode the same
strongly non-trivial transport and differ in the details, from sampling
variance over different prompt sets plus the bf16 batch-path effect.

Readout-level, on 64 held-out WikiText-103 validation records (4096
positions) and 16 ladder windows of the case text (1024 positions):

- Between the lenses: mean JS 0.10 at layer 0, peaking at 0.28–0.31 at layers
  4–5 (top-10 overlap 0.63), then falling to 0.0003 at layer 32 (overlap
  0.98). Same profile on both corpora. The layer 4–5 peak is a readout
  sensitivity, not a matrix anomaly, and deserves a look before any
  early-layer use.
- Against the model's own final distribution, our lens is closer than the
  published lens at all 33 source layers on both corpora, by 0.0002–0.0065
  JS. Sixty-six cells, one sign: a 1000-prompt mean estimates the population
  mean Jacobian better than a 546-prompt mean. The reproduction succeeds.
- Against the logit lens (identity transport, same positions and readout,
  measured on the same 64 WikiText rows and 16 ladder windows): both
  Jacobian lenses are at chance with it at layers 0–8, equal in JS but lower
  in top-10 overlap at layer 32, and clearly worse at layers 17 and 25
  (layer 25, WikiText: JS 0.44 and overlap 0.30 for the J-lens versus 0.30
  and 0.48 for the plain residual; ladder: 0.44 / 0.29 versus 0.25 / 0.53).
  This is a property of the method as read out (J_l · h_l, final norm,
  unembed), not of our fit. A mean-Jacobian lens is not fitted to minimise
  readout divergence and its intended use is transporting directions, but
  any study reading mid-depth residuals through a J-lens on this model must
  report the logit-lens baseline beside it.

## 6. Engineering follow-ups for the agents

1. Benchmark operation: retain the batched lenses, add pairwise comparisons
   (batch 2 vs 4, not only against batch 1), and replace the rtol/atol 1e-4
   "agrees" flag with a relative criterion against a float32 or repeated
   batch-1 reference; the current flag is false for every bf16 batched fit on
   every model tested and carries no information.
2. Assessment operation: offer a float32 readout option and report both.
3. Fit operation: expose `windowsPerRecord` and trim failure records as the
   agents proposed; the stopping statistic is already the reference kernel's
   exact formula (max over source layers of the relative running-mean shift),
   with consecutive-window semantics.
4. Guide text: describe batch-1 and batched fits as draws of the same
   rounding process, drop "reference" language for batch 1, and point to the
   float32 benchmark as the batch-insensitive check.
5. Provenance: shard identity and `runtimeHardware` already record card and
   dimension batch; keep them and surface them in the merge report.

## 7. Operating numbers for planning

| Model | Card | dimBatch | rows/hour | peak GB |
|---|---|---|---|---|
| Qwen3.8-27B hybrid | A100 | 1 / 2 / 4 | 0.9 / 1.7 / 3.4 | 61 / 67 / 79 |
| Qwen3.8-27B hybrid | H100 | 1 / 2 / 4 | 1.4 / 2.7 / 5.4 | 61 / 67 / 79 |
| Gemma-3-27B dense | H100 | 1 / 2 / 4 | 9.3 / 17.5 / 25.8 | 57 / 60 / 64 |
| Gemma-3-4B dense bf16 | H100 | 1 / 4 / 16 / 64 | 36 / 137 / 308 / 325 | 9 / 11 / 18 / 45 |
| Gemma-3-4B dense float32 | H100 | 1 / 4 | 38 / 57 | 18 / 21 |
| Gemma-3-4B dense, T5 fit | H100 | 64 | ~330 (10.9 s/row) | (see T5) |
