# J-lens numerical review and proposed testing program

Date: 2026-09-13
Status: draft for discussion with the coding, reviewing, and running agents.
Purpose: distinguish verified implementation, numerical sensitivity, and research usefulness before committing a large fitting budget.

This document reviews the dimension-batch benchmark and subsequent readout assessment, incorporates the researcher's questions, and proposes an ordered program of tests. It does not authorize cluster submissions, model downloads, deployments, or changes to existing study artifacts. No new full lens fit or component-order experiment has been run as part of this review.

## 1. Recommendation

The live work is valuable. It establishes that fitting, registration, managed assessment, GPU placement, and collection can work on the tested 27B configuration. Batch 4 is a promising execution setting. The evidence does not yet establish that the complete 27B numerical path is correct, that either eight-row lens has converged, or that increasing the budget to approximately 800 rows will resolve the poor agreement with final predictions.

Proceed with small correctness controls, then a full **Gemma 3 4B reference-reproduction experiment**. Add a small **Gemma 3 27B batching benchmark** to compare sizes within a model family. Keep an independent derivative check for the Qwen hybrid architecture. Use those results to decide the configuration and budget of the larger Qwen fitting round.

This is a research recommendation, not a proposal for new software refusals. Continue to let researchers explore while displaying the measured limitations. Correctness failures require investigation; missing qualification alone should remain an explicit status with helpful guidance.

## 2. Evidence reviewed

Primary repository records:

- [Dimension-batch benchmark and handoff](JLENS-DIMBATCH-BENCHMARK-RESULTS-AND-HANDOFF-2026-09-12.md).
- [Batch-1 versus batch-4 assessment and handoff](JLENS-BATCH-ASSESSMENT-HANDOFF-2026-09-13.md).
- [Earlier pilot and scaling handoff, especially §8.1](JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md).

The review examined the current assessment, fitting, reference-wrapper, and input-closure implementations. It also read the collected benchmark and assessment JSON reports in the research workspace. The numbers below were extracted from those reports, not independently recomputed on a GPU.

Five focused existing tests were rerun and passed: the independent analytic estimator fixture; fitting-mean/reference agreement; both small hybrid-model loader variants; and assessment equivalence with the previous owner. The hybrid-model cases are execution smoke tests, not independent derivative checks. Repository files were not changed during that review.

For the running agents, the relevant report IDs are:

| Report | Run ID |
|---|---|
| Gemma 3 4B, bfloat16, H100 | `jlens-benchmark-676004bea6e94e69a76f6c8eced2a8ae` |
| Gemma 3 4B, float32, H100 | `jlens-benchmark-fab8769bcdbf48469e7f523de6e5de01` |
| Qwen 27B, bfloat16, H100 | `jlens-benchmark-e32e0c695de8479fbc875fe869aa05da` |
| Qwen 27B, bfloat16, A100 | `jlens-benchmark-5929ab75f9b54c8c87ffc58e9a3b27bb` |
| Eight-row Qwen lens assessment | `jlens-assessment-f29facc3b590439aae8743dc5f298463` |

Keep the reports, corpus records, and their hashes in the research workspace. Future comparison reports should bind the full input identities, not just these human-readable run IDs.

## 3. What the experiments establish

### 3.1 Batching is a numerical execution change, not different fitting data

Dimension batch 1 runs one prompt copy and computes output components in successive backward passes through a retained forward computation. Dimension batch 4 runs four identical prompt copies; each copy supplies a different output-component cotangent in the same backward pass. Corresponding components should represent the same mathematical derivatives when examples do not interact across the batch dimension.

Changing the batch shape can change numerical execution in both the forward and backward computations. Floating-point operation order and kernel choice can then change the result. This is a documented possibility, not proof of the cause or acceptable magnitude of our discrepancy. See [PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations).

At dimension batch 1, changing only the order in which independent components are evaluated should not change their mathematical values. Rounding is not intrinsically random. With the same forward computation and deterministic execution, exact agreement is the appropriate expectation. Nondeterministic kernels or mutable state must be distinguished from batch-shape sensitivity.

### 3.2 The tested 4B model is Gemma, not Qwen

The benchmark used `google/gemma-3-4b-it` and `Qwen/Qwen3.8-27B`. The first has conventional attention; the second uses a hybrid architecture with linear-attention blocks. The 4B experiment therefore supplies a precision control within Gemma, not a direct numerical validation of Qwen's hybrid blocks.

For batch 4 versus batch 1 on an H100, the relative Frobenius difference of the **two-row mean lens** is:

| Relative depth | Gemma 3 4B, bfloat16 | Qwen 27B, bfloat16 |
|---|---:|---:|
| First source layer | 12.91% at layer 0 | 5.32% at layer 0 |
| Approximately one-quarter | 3.77% at layer 8 | 4.39% at layer 16 |
| Middle | 0.98% at layer 16 | 1.86% at layer 31 |
| Approximately three-quarters | 0.56% at layer 24 | 0.43% at layer 48 |
| Last source layer | 0.12% at layer 32 | 0.062% at layer 62 |

The corresponding Gemma float32 differences are much smaller: approximately 0.00131% at layer 0 and 0.000137% at layer 16. These are relative matrix differences, not token error rates or differences between predictions and ground truth.

Larger early-layer differences, decreasing with depth, and the float32 control are consistent with accumulated numerical effects along longer differentiation paths. They do not establish the source of the error. In particular, a successful small Gemma control does not exclude a separate Qwen-specific problem.

### 3.3 The readout assessment adds useful, narrower evidence

The two eight-row Qwen lenses agree closely in aggregate on sixteen passages from one held-out source. At layer 31:

- Between lenses: mean JS divergence approximately **0.000125**, mean top-10 overlap **98.1%**.
- Either lens versus the final residual's readout: mean JS divergence approximately **0.662**, mean top-10 overlap approximately **1.13%**.

These are separate findings. The small between-lens difference does not explain or validate the much larger lens-to-final difference. With the natural-log JS calculation used here, the upper bound is `ln(2)`, approximately 0.693; an early-layer mean of 0.689 is close to that bound.

The live assessment also establishes useful operational behavior: one forward capture per usable row, one pair of lens matrices resident at a time, scratch cleanup on the successful path, and successful managed execution on the reported GPU. It is not a complete failure/recovery or scientific qualification result.

## 4. Corrections to the handoffs' interpretation

Apply these corrections to both the assessment handoff and the benchmark handoff's added assessment summary.

1. **Replace “indistinguishable” and “the same instrument.”** The maximum reported JS value is the maximum of the per-layer means across assessed positions. It is not the maximum individual-position difference. Agreement to three decimal places is not an equivalence test. Use “closely agreeing aggregate readouts under the tested conditions.”
2. **Correct top-10 overlap semantics.** A mean overlap of 0.89 means an average of 8.9 shared tokens out of ten. It does not mean that complete top-10 sets coincide at 89% of positions. It also says nothing directly about their order or the identity of the top prediction. Values rounding to 1.00 should not be described as exact equality without the unrounded evidence.
3. **Do not infer convergence from agreement with final predictions.** The fitter estimates an average Jacobian; it does not optimize final-token prediction loss. Poor final-readout agreement could reflect sampling, corpus differences, approximation limitations, or an implementation/numerical issue. More rows need not eliminate it.
4. **Retain the earlier ruling on row budgets.** Approximately 800 rows was an observation from other reference fits, not a universal criterion. Similar running-mean-change trajectories indicate similar update behavior, not proof that either estimate has converged or is accurate. Decreasing updates alone are also insufficient: averaging mechanically reduces the influence of each additional row.
5. **Remove “bound on trust.”** Final-readout agreement is a diagnostic. No calibrated bound on interpretive reliability has been established. The code's `qualification: notPerformed` remains appropriate.
6. **Qualify the float32 claim.** The strong float32 result came from the separate 4B Gemma control. It was not a float32 fit of the 27B Qwen checkpoint, and the measured differences became small rather than literally vanishing.
7. **Describe the held-out sample accurately.** Sixteen non-overlapping windows from one source are useful study-specific coverage, not sixteen independent documents. Resetting context for each window and assessing only the first eligible positions limits transfer to full-context study use. Different file hashes alone do not establish independence.
8. **Record assessment precision separately from fitting precision.** The current assessment transports residuals through float32 matrices, then calls the reference wrapper's `unembed`, which casts to the model output head's dtype before final normalization and projection. In this run that was bfloat16. Test whether agreement persists with a float32 readout; do not assume rounding explains it or is harmless.

Suggested replacement summary:

> On sixteen passages from one held-out source, the eight-row batch-1 and batch-4 lenses showed close aggregate readout agreement: the largest per-layer mean JS divergence was 0.0032, and mean top-10 overlap ranged from approximately 0.89 to values rounding to 1.00. This supports batch 4 as a candidate execution setting for further fitting. It does not establish convergence, research adequacy, or equivalence at every position. The large disagreement with final predictions remains a separate diagnostic question.

## 5. Existing coverage and the important gaps

| Evidence | What it supports | What it does not establish |
|---|---|---|
| Analytic causal toy model | Orientation, source/future-target aggregation, component chunking, and prompt averaging against an independent calculation | Correct derivatives through the real hybrid architecture |
| Agreement with the pinned reference fitter | Our orchestration follows that implementation | Absence of a shared error or unsuitable model adapter |
| Small real hybrid-model fit | Actual loader and backward path execute with finite matrices of expected shape | Numerical correctness of those derivatives |
| Assessment AST/fixture equivalence | Refactor preserved the previous calculation | Correctness of the inherited measuring apparatus |
| Gemma float32/bfloat16 benchmark | Strong precision sensitivity within one model | Explanation of the exact Qwen kernel behavior |
| Qwen batch readout comparison | Aggregate stability on the selected text and readout precision | Accuracy, causal validity, or adequacy for a specific research conclusion |

The objective of the following program is to add independent controls where agreement with ourselves or the reference is insufficient.

## 6. Proposed testing program

### T0 — Freeze the comparison specification and repair reporting

Before new measurements, capture exact model and tokenizer revisions, reference package commit, local driver/kernel hashes, package versions, hardware, selected kernels, compilation, precision settings, token IDs, position masks, and corpus selection. Record precision separately for model execution, gradients, accumulation, saved matrices, transport, and vocabulary readout.

Correct the report interpretations above. Preserve original reports and add a new interpretation record rather than rewriting run evidence. Retain the current elementwise diagnostics even when adding a relative Frobenius headline. Define the denominator and a zero-norm rule explicitly; report absolute differences when a relative comparison is undefined. Separate “numerically close under this criterion” from “scientifically adequate.”

Deliverable: a reviewed experiment specification and precise metric definitions. Acceptance tolerances should be fixed before examining the new comparison results and justified by the calculation and its measured numerical floor, not widened afterward to obtain a pass.

### T1 — Validate the assessment's measuring apparatus

Start with small models locally, then selected real-checkpoint positions on the cluster.

- Compare a lens with itself: identical readout tensors must have zero distributional difference and full set overlap, subject to explicitly identified tie handling.
- Capture the final residual and apply the assessment's normalization and unembedding. Compare with the native Hugging Face model's logits at exactly the same input positions, before sampling processors. This checks that “final prediction” is actually the model output.
- Exercise identity transport at the target residual as an internal diagnostic. Do not manufacture an ordinary registered lens that violates the public source-layer contract.
- Compare the assessment readout with the production full-vocabulary and watchlist owners under matched precision. Check orientation, final-norm gain, epsilon, softcapping where applicable, token IDs, and layer numbering.
- Repeat readouts in float32 using the same captured activations and matrix bytes. This isolates readout rounding; it is not a float32 model-forward or fitting experiment.
- Compare against a direct-logit-lens baseline to show what adding the learned transport changes.

Deliverable: numerical control results with a repairable explanation for every mismatch. Investigate failed identity/native-logit controls before interpreting fitted-lens quality or spending on a full reproduction.

### T2 — Test repeatability and component-order invariance

Run first on the existing small fixtures and Gemma 3 4B, then on selected Qwen 27B components. A full matrix is unnecessary for the initial control. Use fixed component indices spread across the output width and early, middle, and late source layers.

Separate three experiments:

1. Repeated component derivatives in the same order on one retained forward computation.
2. Ascending, descending, and seeded-shuffled component orders on that same retained forward computation.
3. Repeated fresh forward computations with identical tokens and settings.

Keep `dimBatch=1`; retain the graph until all comparisons finish; collect returned gradients without parameter-gradient accumulation. Reassemble results by component identity, never execution order. Record bitwise equality, absolute difference, and relative difference for each component and layer.

With deterministic kernels, order should not matter. If ordinary execution is not repeatable, record the baseline variation, investigate dispatch/state, and test a supported deterministic configuration separately. Do not silently enable different kernels and call the result the original configuration. A component-order effect exceeding same-order repeatability warrants investigation before attributing differences to batch shape.

Deliverable: an explicit result for “same calculation, different component order.” Passing this control strengthens, but does not prove, the batch-shape explanation.

### T3 — Independently check hybrid derivatives

Extend the tiny real hybrid decoder test beyond shapes and finiteness. Use a small nonlinear hybrid configuration with actual linear-attention and full-attention blocks, fixed weights, and fixed inputs.

- Compute a small explicit Jacobian independently of the fitting helper and reduce it using the intended estimator.
- Check selected scalar derivatives with central finite differences across a range of perturbation sizes in float64 where supported, otherwise a justified higher-precision configuration.
- Match the actual mathematical objective: the reference estimator sums selected target positions and averages gradients over selected source positions. A same-token-only Jacobian is a different estimator.
- Check causal masks, source/target offsets, component ordering, batch independence, and the final partial dimension chunk.
- Compare batches 1, 2, and 4 against the independent result.
- Include deliberate orientation, mask, and reduction mutations so the controls demonstrably detect relevant errors.

Finite differences through quantized bfloat16 computations are not a high-precision ground truth; perturbations can be rounded away. Establish the derivative calculation in higher precision, then characterize the lower-precision execution separately. A forward-mode versus reverse-mode comparison can supplement this but should not replace an independent perturbation check.

Deliverable: numerical derivative validation for the hybrid implementation, with tolerances and unsupported operations explained. Tiny-model success still requires selected real-model checks; it does not certify every large-model kernel.

### T4 — Localize batching and precision sensitivity

On identical token IDs, compare single-copy and replicated-batch execution before computing a whole lens:

- Capture forward residuals at early, middle, and late layers and compare corresponding copies.
- Compare selected backward components for the same input, target mask, and source sites.
- Separate repeated-run variation, component-order effects, batch-shape effects, and dtype effects.
- Record actual executed kernels, not just configured function names or installed packages. Test a changed reduction/kernel setting as a separate, explicit case.
- Use high precision on selected blocks or a small homologous model where a full high-precision 27B forward is impractical. Do not describe partial upcasting as a full float32 reference fit.

Complete the existing benchmark work: retain case matrices when requested; add direct all-pairs comparison, including A100/H100 when available; add per-layer relative metrics with zero-norm handling; trim report failure summaries while preserving full logs. Equal distances to one baseline do not prove that two candidates agree with each other.

Deliverable: evidence about where the difference first appears and which execution changes affect it. A plausible rounding story becomes stronger only when these interventions predictably change the discrepancy.

### T5 — Produce a full Gemma 3 4B lens and compare with the published lens

This is the preferred first substantial fitting expenditure. It tests the complete fitting, registration, transport, and readout workflow against an external instrument on a tractable model.

**Reproduction preparation:** pin the published lens and its metadata; identify the original model/tokenizer revisions, dataset revision, row selection, text processing, tokenization, position mask, estimator, and stopping implementation. Match what is recoverable. Record unknowns rather than inventing exact reproducibility. A dataset name, maximum character count, and final prompt count are not enough to reconstruct the original selected bytes.

Use the published recipe as the starting point. If its batch size or hardware is unavailable, choose a measured local configuration and record that departure. Do not transfer Qwen's memory ceiling to Gemma. Prefer a matched bfloat16 reproduction first; a full float32 alternative is a separate question and should have a separate cost review.

Retain intermediate snapshots at preselected budgets and at the final stopping point. Keep sampling, operational stopping, and scientific adequacy distinct. A cap or stopping threshold ends computation; it does not confer qualification.

Assess our intermediate and final lenses, the published lens, and the identity/logit-lens baseline on the same captured activations for each evaluation set:

- A general-language held-out set, with documents separated from fitting sources where possible.
- A researcher-selected application set, including the context and rendering actually used in the study where feasible.
- An unchanged measurement set across checkpoints; reserve a separate final evaluation set if development choices are tuned on the first set.

Measure per-layer matrix differences, mean and tail JS divergence, top-1 disagreement, top-k set overlap and ranks, and target/control score differences. Retain per-document results and selected position-level examples. Compare signs, rankings, and conclusions used in the research, not only vocabulary-wide averages. Account for published-artifact storage precision separately from fitting precision.

Interpretation:

| Outcome | Next interpretation or action |
|---|---|
| Our lens closely agrees with the published lens and both pass apparatus controls | Stronger end-to-end evidence on Gemma; not numerical ground truth or Qwen qualification |
| Both lenses disagree similarly with native final predictions | Investigate the metric, text population, and approximation; the effect is not unique to our fit |
| Published lens is substantially better on the relevant comparisons | Investigate corpus selection, fitting budget, numerical settings, and implementation |
| Successive fits stabilize but differ from the published lens | More rows alone may not resolve recipe differences, numerical bias, or approximation limits |
| Vocabulary averages agree but target/control conclusions differ | The lens is not equivalent for that research use under the chosen criteria |

Deliverable: one immutable reproduction report stating what was matched, what differed, and what the comparison supports. Exact matrix equality is not expected across different corpora, execution configurations, or storage precision.

### T6 — Add a controlled Gemma 3 27B comparison

Use a small fitting benchmark, not a full reproduction initially. Match the Gemma 4B corpus and preparation, token settings, GPU, precision, and dimension batches as closely as feasible. Record tokenization differences if identical text does not yield identical token sequences. Compare corresponding fractions of depth and block types, not just identical layer numbers.

This provides a cleaner within-family size comparison than Gemma 4B versus Qwen 27B. It is not a pure parameter-count experiment: weights, depth, width, and architectural dimensions also change. It cannot certify Qwen's hybrid derivatives.

If a published Gemma 27B lens is available locally or its acquisition is approved, assessing it through the validated apparatus can be useful without fitting a new full lens. Do not assume that its fitting budget or achieved readout quality transfers to Qwen.

Deliverable: a measured answer to whether batching sensitivity grows or persists in a larger Gemma model, alongside a separate account of Qwen-specific behavior.

### T7 — Decide the larger Qwen fitting round from the evidence

Review T1–T6 before allocating the substantial round. The independent checks may identify a repair, support a particular kernel/precision setting, or leave a limited but usable instrument. Record the result plainly.

If proceeding, choose dimension batch and numerical settings using the measurements and memory headroom. Retain useful checkpoints and assess successive merged fits on fixed held-out inputs, including the actual research contrasts. Keep deterministic shard merging, exact provenance, continuation compatibility, queue monitoring, and verified collection in the managed workflow. Do not assume more data averages away a systematic numerical bias.

Deliverable: a justified budget and qualification statement scoped to the tested model revision, runtime, corpus, and research use. Useful exploratory results can remain available with limitations even when broader qualification is incomplete.

## 7. Published Gemma lenses: verified reference metadata

Neuronpedia hosts both lenses on Hugging Face. These are external comparison artifacts, not artifacts fitted by Hugging Face itself.

| Published lens | Model execution dtype during fitting | Dimension batch | Prompts fitted | GPU |
|---|---|---:|---:|---|
| `google/gemma-3-4b-it` | bfloat16 | 128 | 546 | NVIDIA B200 |
| `google/gemma-3-27b-it` | bfloat16 | 64 | 828 | NVIDIA B200 |

Both configurations record `Salesforce/wikitext`, `wikitext-103-raw-v1`, training split, maximum 2,000 characters per text, maximum sequence length 128, compilation enabled, a 1,000-prompt cap, and early stopping at delta 0.002 with a minimum of 100 prompts and window 10. Confirm the exact stopping statistic and preprocessing from the generating implementation before calling a new experiment a reproduction. Their declared bfloat16 fitting dtype does not imply that every intermediate or saved tensor is bfloat16.

Verified from the published configurations during this review:

- [Gemma 3 4B IT configuration](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/gemma-3-4b-it/jlens/Salesforce-wikitext/config.yaml).
- [Gemma 3 27B IT configuration](https://huggingface.co/neuronpedia/jacobian-lens/blob/main/gemma-3-27b-it/jlens/Salesforce-wikitext/config.yaml).

These URLs follow `main`. The execution agent must pin an immutable repository revision and content hashes before acquisition or comparison. Published use of bfloat16 establishes precedent, not proof that our runtime's observed numerical discrepancy is acceptable.

## 8. Tooling improvements that support the program

Keep changes in small scientific owners, with the app, CLI, API, and agent instructions sharing the same operations and report semantics.

- **Benchmark evidence:** requested matrix retention, direct pairwise and cross-hardware comparisons, explicit metric/tolerance provenance, and compact failure summaries with full logs retained.
- **Assessment evidence:** per-document reductions, position-level tails or bounded diagnostic traces, top-1 agreement, target/control comparisons, exact counts, and explicit readout precision. Preserve the aggregate legacy metrics with their correct meanings.
- **Corpus preparation:** an explicit `windowsPerRecord` option with deterministic, non-overlapping selection; source hashes, document identity, offsets, window lengths, selection algorithm, and seed in the receipt. Describe offset units. Keep source grouping through analysis and split by document before generating fitting/evaluation windows where independence matters. Warn when a short source cannot supply the requested number of windows rather than silently duplicating it.
- **Compute review:** show memory/disk estimates and requested matrix-retention cost. The 27B batch-4 measurements had little device-memory headroom; longer execution is not guaranteed by a two-row benchmark.
- **Transport:** preserve the landed lens closure change: ship the record, available receipt, and converted execution tensors; retain original source bytes in the authoritative workspace. Do not reintroduce duplicate multi-gigabyte payloads. Improve repeated hashing only through reviewed receipts bound to exact immutable bytes and their lifetime, never by trusting stale path names.

## 9. Operating and review instructions

This draft records proposed work; it does not claim that T1–T7 are implemented. Do not turn speculative diagnostics into mandatory product gates or claim scientific success merely because both test suites pass.

Keep study inputs and results in the research workspace. Runs and frozen artifacts remain immutable. Keep site names and study-case vocabulary out of repository changes and commit messages. Credentials remain in the approved credential store. Avoid rebuilding or changing the active app/client payload during a live command chain; deployment and controller restart must respect running jobs.

For implementation, branch from current main and include the landed lens-closure fix. The separate P1–P6 probes/interventions stack remains under independent review; this document does not authorize merging or modifying it. Read the complete diff, run both suites serially, run required generated-resource, public-hygiene, and scientific audits, and provide an AST audit for any claimed mechanical move. Use Xcode beta, the installed Metal toolchain, and scratch outside the repository for native checks. Independent reviewers report through the user before integration.

For each scientific test, report the question, exact inputs, configuration, comparison, observed result, uncertainty, and next decision. Distinguish implementation checks, numerical accuracy, stability, and research usefulness. The researcher should be able to understand what was learned without interpreting a wall of raw matrix diagnostics.
