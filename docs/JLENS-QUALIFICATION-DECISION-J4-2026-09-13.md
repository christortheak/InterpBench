# J4: scoped J-lens qualification decision

Date: 2026-09-13.
Baseline: main b821e3f (J1 corrections, J2 evidence, J3 landed at 9f99ae8; engine
deployed and running at 9f99ae81).
Recorded by: the researcher's running session. Independent review through the
user is the next step; this record does not substitute for it.
Companions: `docs/JLENS-CLOSURE-AND-P7-QUALIFICATION-HANDOFF-2026-09-13.md` (the
program this closes), `docs/JLENS-NUMERICS-PROGRAM-DECISION-2026-09-13.md` and
`docs/JLENS-NUMERICS-PROGRAM-RESULTS-AND-HANDOFF-2026-09-13.md` (with their J1
and J2 sections), `docs/REVIEW-JLENS-ASSESSMENT-READOUT-BRANCH-2026-09-13.md`
(J3). Detailed evidence lives in the study workspace under
`diagnostics/jlens-numerics-2026-09-13/` (`INTERPRETATION-*.md`, `j2-*/`, and
the T-series report directories) and `runs/`.

## 1. Decision

**Qualified, within the tested scope:** the managed J-lens fitting,
registration, and readout-assessment workflow on Python Compute with CUDA,
for the configurations in §3, as an instrument whose numerics are correct
(estimator), exactly reproducible within a configuration, and whose
cross-configuration differences are measured and bounded. Fits may be made at
the largest feasible dimension batch on whichever card the queue offers.
Lenses are judged by readout against the model's own final distribution with
the built-in logit-lens baseline and, when wanted, the paired float32
readout; never by bitwise or tolerance agreement with another bf16 fit.

**Not qualified:** scientific equivalence of lenses or shards fitted on
different GPU configurations (operationally allowed, recorded in provenance,
unverified as equivalent); any full-size hybrid 27B lens; usefulness of any
fitted lens for direction-transport research; readout usefulness at middle
depths, where the plain residual reads out better than either J-lens on
Gemma-3-4B.

**Gates for proceeding to P7:** J1 corrected (4a87703); J2 direct controls
accounted for (b821e3f); J3 landed, locally verified, and now verified live on
the deployed engine (§4.4). Nothing in the J-lens work blocks the
probes/interventions qualification, which is an independent workflow. P7 may
start its live acceptance.

## 2. What was tested (combinations)

| Estimator | Model | Precision | Dimension batch | Hardware | Readout | Tests |
|---|---|---|---|---|---|---|
| reference `jlens` @581d3986 (`jacobian_for_prompt`), bare path | tiny random hybrid (2 linear + 2 full attention) | float64 | 1, 2, 4 | Mac CPU | — | T3 |
| reference, bare and engine paths | Gemma-3-4B @093f9f38 | bf16 | 1, 2, 4, 16 | H100 | bf16 head, float32 head | T1, T2, T4, T4b, J2, T5 assessments, J3 live |
| reference, bare and engine paths | Gemma-3-4B | bf16 | 1, 4 | A100 | bf16 head | T1, T2, T4, T4b, J2 |
| managed benchmark | Gemma-3-4B | bf16 | 1, 2, 4, 16, 32, 64 | H100 | — | benchmarks |
| managed benchmark | Gemma-3-4B | float32 | 1, 2, 4 | H100 | — | float32 benchmark |
| reference, bare and engine paths | Qwen3.8-27B hybrid @1d4bf0f2 (PyTorch linear-attention fallback) | bf16 | 1, 4 | H100 | bf16 head | T1, T2, T4, T4b (scan), J2 (layers 0, 16, 31, 48, 62) |
| managed benchmark | Qwen3.8-27B hybrid | bf16 | 1, 2, 4 | A100, H100 | — | benchmarks |
| reference, bare path | Gemma-3-27B dense @005ad340 | bf16 | 1, 2, 4 | H100 | — | T6 (managed benchmark), J2 (layers 0, 15, 30, 45, 60) |
| managed fit | Gemma-3-4B | bf16 | 64 | H100 | — | T5 (1000 prompts) |
| managed assessment | Gemma-3-4B, two registered lenses | bf16 forward | — | H100 | bf16 head; paired float32 | T5 (WikiText 64, ladder 16), J3 live (WikiText 64, float32) |

Runtime everywhere: eager attention, compile off, TF32 matmul off, float32
matmul precision "highest", deterministic algorithms off, kernel policy
"current", no optional fast kernels. Not tested: compile on, TF32 on, B200,
MPS or any non-CUDA GPU path for fitting, 4B batches above 64, hybrid
batches other than 1 and 4 in direct comparison, any hybrid lens beyond eight
rows.

## 3. Supported configurations (explicit)

- Fitting: `jlens-fit` on CUDA, bf16, Gemma-3-4B at dimension batch up to
  64; Gemma-3-27B dense at 1–4 (64 GB at 4 on an 80 GB card); Qwen3.8-27B
  hybrid at 1–4 (79 GB at 4, the ceiling), on A100 or H100, with card and
  batch recorded in shard identity and `runtimeHardware`.
- Assessment: `jlens-fit-assess` on CUDA with the historical native readout,
  the built-in logit-lens baseline, matrix statistics, and the optional
  paired float32 readout (needs about 4 × vocabulary × hidden bytes of extra
  device memory; 2.7 GB on Gemma-3-4B).
- Registration and custody: fit export/fetch/verify/register through the
  managed chain, verified on 1.6 GB (4B) and 12 GB (hybrid 27B) closures.

## 4. Findings, by the five categories the closure handoff asks for

### 4.1 Derivative correctness controls

Established. In float64 on a tiny hybrid, the reference estimator equals an
independent explicit Jacobian to 6e-8 at every layer; batches 1, 2, 4 are
bitwise identical; finite differences agree at their floor; transposed
reference, wrong mask, and wrong reduction are detected (T3). The engine's
measuring apparatus reproduces the model's native logits bit for bit (T1).
On real models, a mutated coordinate and a permuted component order fail the
identity check (J2). Scope limit: T3 certifies the estimator's mathematics,
not every checkpoint, precision, or kernel; the real-model controls certify
the comparison apparatus.

### 4.2 Reproducibility within a configuration

Exact. Every repeat of a (model, card, batch, sequence) configuration is
bitwise identical: retained-graph repeats, component-order permutations, and
fresh forwards at batch 1 (T2, dense 4B on both cards and hybrid on H100);
whole-estimator repeats at batch 1 on dense 4B (both cards), hybrid 27B, and
dense 27B (J2). The engine path equals the bare reference bitwise at the
same batch on dense 4B (both cards, four batches, three rows, 33 layers) and
hybrid 27B (two batches, five layers) (J2).

### 4.3 Numerical agreement across configurations

Measured, deterministic, discrete, and bounded. Changing the dimension batch,
the card, or the sequence length can change the bf16 forward pass from the
first block (1e-5 to 1e-3 relative at layer 0, growing to about 1e-2 with
depth) and hence the Jacobian rows (5–18% at layer 0, 0.1–0.3% at the
deepest source layer) (T4, T4b). The outcomes are discrete: on dense 4B/H100
batches 4 and 16 coincide bitwise and batch 2 is distinct; for the
128-token row batches 1, 4, 16 coincide; on dense 27B batches 2 and 4
coincide; on the A100 every tested pair differs (J2). Kernel selection was
not captured; the mechanism is inferred. Configuration differences are not
random draws, and averaging across configurations is not shown to cancel
them. At readout level, the one measured pair of batch-1 and batch-4 8-row
hybrid lenses differed by at most JS 3.2e-3 with top-10 overlap ≥ 0.89. The
float32 4B benchmark shows batch shapes agreeing to 1e-5, the
batch-insensitive reference when one is needed.

Impact: fits are valid at any tested batch on either card; a fitted lens is a
configuration-specific bf16 outcome and should be compared with others by
readout, not bitwise. Mixed-configuration merging is operational, not
scientifically qualified (§5).

### 4.4 Managed execution and custody

Verified live. The managed chain (interview → publish → input-plan → package
→ stage → plan → submit with per-request GPU type → export → fetch →
verify-custody → register) ran end to end for the 1000-prompt 4B fit, five
assessments, and four benchmarks, every receipt verified. The J3 slice was
verified live on the deployed engine 9f99ae81 in
`runs/jlens-assessment-73fadf92d4b84c7285c20e5776318d1f`: the request review
carried the readout block; the built-in logit-lens baseline reproduced the
hand-run T1 baseline exactly at all five checked layers on the same 4096
positions; the historical groups matched the earlier assessment to the
printed digits; the matrix statistics matched the hand comparison; the extra
float32 head storage was 2,685,020,160 bytes as predicted; float32 readout
moved every lens number by at most 2e-4 JS (`INTERPRETATION-J3-LIVE.md`).
Not done: manual inspection of the new report in the app's Results view; the
ladder-corpus float32 run.

### 4.5 Usefulness for readout or direction-transport research

Readout: measured, mixed. Our 1000-prompt Gemma-3-4B lens is closer to the
model's final distribution than the published 546-prompt lens at all 33
layers on both held-out corpora (cause unidentified: prompt set, batch,
compile, hardware, and storage precision differ). Against the plain residual
(logit lens) at full depth on 64 WikiText rows, the J-lens is better at
layers 6–15 (by ≤ 0.015 JS, both near the ceiling), worse from layer 16 to
31 (by up to 0.14 JS at layer 25), and tied in JS at layer 32 with lower
top-10 overlap. The two lenses disagree most at layers 4–5 (JS 0.28–0.31,
overlap 0.63), and float32 readout does not change that, so it is a property
of the lenses and the residual, not the readout. Direction transport: not
tested; predicting final tokens is not evidence of accurate local
directional transport.

Impact: any study that reads mid-depth residuals through a J-lens on this
model must report the logit-lens baseline beside it (the assessment now does
so automatically). Fitting a full 27B lens for readout purposes is not
supported by this evidence; fitting it for direction transport is a separate,
untested research question.

## 5. Limitations and deferred questions, with stated impact

| Item | Status | Impact |
|---|---|---|
| Mixed-GPU scientific equivalence | Deferred (matched small fits across configurations vs a uniform configuration on a fixed corpus budget not run) | Merging across cards stays operational with provenance; do not claim equivalence in any study; does not block P7 |
| Full-lens hybrid 27B agreement | Bounded subset only (five layers, one row) | Hybrid fits beyond eight rows are unqualified; none is planned on this evidence |
| Kernel-selection mechanism | Inferred, dispatch not captured | Explains nothing more than the pattern; no engineering action depends on it |
| Published stopping-window semantics | Unknown; our rule (ten consecutive rows below 0.002) did not fire in 1000 prompts | Treat the row budget as the operative stop; record both window statistics when the fit operation gains that option |
| Layer 4–5 lens sensitivity on Gemma-3-4B | Open; confirmed in float32 | Do not use early-layer readouts on this model without a dedicated look at transported-vector norms and final-norm behaviour |
| Intermediate-budget comparison (546 vs 1000 rows) | Not possible from the rolling checkpoint | Needs retained checkpoints or a deliberate 546-row fit; a research nicety, not a gate |
| App Results view of the new report | Not inspected in this session | Running agents' P7-D surface pass covers it |
| Independent review of this record | Pending | Required before the support statement is published |

No unresolved estimator or shared-runtime discrepancy remains: the two J1
claims that lacked direct evidence (engine-vs-bare identity, batch-pair
identity) are now measured (J2).

## 6. Evidence index

Repo: the four companion documents above. Workspace
`diagnostics/jlens-numerics-2026-09-13/`: `SPEC.md`,
`INTERPRETATION-T1-T2.md`, `-T3.md`, `-T4.md`, `-T5.md`, `-T6.md`, `-J2.md`,
`-J3-LIVE.md`; report directories `gemma4b-bf16*/`, `qwen27b-bf16/`,
`t3-hybrid-fd/`, `t4b-*/`, `j2-*/` (with `content-hashes.json`; retained
matrices with their sha256 stay in the cluster diagnostics directory);
`t5-matrix-comparison.json`. Workspace `runs/`: fit
`jlens-fit-ccdaa9738cbe4fd0801302ccd69c566a`; assessments
`jlens-assessment-2da5e363…`, `…badd5838…`, `…73fadf92…`; benchmarks
`jlens-benchmark-c9f1b6e4…`, `…ae6bfdc0…`, `…fab8769b…`, `…e32e0c69…`,
`…5929ab75…`; lenses `custom-lens-895a4f3a…` (ours) and
`custom-lens-976f9c12…` (published).
