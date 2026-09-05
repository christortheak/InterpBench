# Production validation matrix (pre-declared, not yet executed)

Status: **DECLARED 2026-09-05, UNEXECUTED.** This document is the
pre-registration of a bounded validation matrix on the production models and
hardware this project actually uses. It fixes the configurations, the checks,
the comparison each check makes, and the tolerance it is held to, *before*
any outcome is inspected. It is not evidence. A row becomes evidence only when
its result artifact exists, is linked from the "Results" column, and its
tolerance was the one declared here. Rows whose runner does not yet exist say
so; they are open work, not skipped checks.

Why it exists: the tiny-model regression suites under `Server/tests/` prove
that each contract holds on CPU fixtures. They cannot show that the same
contracts hold across MLX, CUDA/bf16, 8-bit quantized 27B, chunked prefill at
production prompt lengths, or the real chat templates. The external review of
2026-09-05 (S-06) named that gap; the remaining-work handoff (REM-08) asks for
the matrix below.

## 1. Configuration slots

Each executed row pins every field in this table. Values are read from the
stamps the engines already write (`config.json` schema 4, the adapter sidecar,
the J-lens qualification record, `site qualify`'s report), never typed by hand.
The matrix is bounded to configurations that a study in this project has run
or is about to run; it is not a Cartesian product over models the project does
not use.

| Field | Where it is stamped | Notes |
|---|---|---|
| model id, revision | `config.json` (`modelID`, `revision`), adapter sidecar | a revision that is not a commit sha is a refusal on the freeze path already |
| tokenizer / chat-template identity | `prompts/models/<owner>--<repo>@<revision>.json` capability record | probed, not inferred |
| engine, package versions | `config.json` `appVersion`, `pythonEnvironment`; `site qualify` `dependencyLock` | |
| build commit | adapter sidecar `buildIdentity`; J-space report `engine.buildCommit` (this branch); the study run `config.json` carries `appVersion` only | needed by the impact ledger to classify by producing revision; a run without a commit stamp classifies as `unknown` |
| hardware | `site qualify` `cudaProbe`; Slurm job stamps | one GPU type per study's shards (standing rule) |
| dtype / quantization | `config.json` `dtype`; J-lens record `runtimeNumerics` | bf16 and 8-bit are different configurations, never one row |
| artifact and data hashes | manifest pins (`stimulusHash`, `neutralCorpusHash`, `devPromptsHash`, `batteryHash`, adapter `documents`) | |
| reading positions, layers, doses, seeds | manifest concepts / conditions; sweep grid; `manifest.seeds` | doses in residual-norm units under the stamped denominator rule |

Configurations expected to appear (fill from the workspace, do not assume):
the 4B instruction-tuned Gemma tier used for local and CPU-side checks; the
27B production tier on the cluster in bf16; the 8-bit 27B variant where it has
been used for J-lens work; and the Qwen3 models used as judge or as
verdict-rigidity subjects. A configuration that no study used gets no row.

## 2. Checks

Each row: what is compared, which existing runner produces it (or "no runner"),
the declared tolerance, and what a pass licenses. Tolerances marked *code* are
constants the engine already enforces; tolerances marked *declare* must be set
here, by the researcher, before the row runs.

| # | Check | Comparison | Runner today | Tolerance | A pass licenses |
|---|---|---|---|---|---|
| V1 | Build and environment identity | node reproduces the deployed commit and lock | `steerlab-server site qualify` (`buildIdentity`, `pythonEnvironment`, `dependencyLock`) | exact | every other row's provenance |
| V2 | Render and tokenization goldens | committed golden render/tokens vs the node | `site qualify` (`goldenRender`, `goldenTokens`) | exact | the prompt bytes a study sees are the pinned ones |
| V3 | Same-engine vector-parity arithmetic | synthetic fixtures through `vectors compare` | `site qualify` (`vectorParity`) | *code* (the harness's own) | the parity arithmetic, NOT cross-substrate agreement |
| V4 | Zero-intervention equivalence | steered pass at α = 0 vs unsteered pass, hidden states and logits, prefill and decode | tiny-model: `tests/test_injector_equivalence.py` (`test_post_hook_hidden_states_and_logits_match_the_legacy_path`, `test_generated_token_ids_match_the_legacy_path`); **no production runner** | *declare*: bit-identical in the runtime dtype is the tiny-model result; production must state whether bit-identity or a bound is claimed | the hook path itself changes nothing |
| V5 | Chunked vs single-pass prefill under steering | logits and continuation | tiny-model: `tests/test_chunked_prefill.py` (`test_chunked_prefill_matches_single_pass_logits_and_continuation`, rtol 1e-4 / atol 1e-5; `test_intermediate_chunks_are_bit_identical_and_prompt_end_steers`); **no production runner** | *declare* (the tiny-model bound is rtol 1e-4 / atol 1e-5; the external review measured ≤ 7.5e-8 on tiny Llama) | prompt-end injection is position-exact at production prompt lengths |
| V6 | Injection fires on every decode step | per-token firing count | tiny-model: `tests/test_injection_fires_per_token.py`; the live smoke test | exact | the classic prefill-only bug is absent on this runtime |
| V7 | Nominal vs realized dose | ‖(steered − baseline) at the injection layer‖ vs α·‖v‖, per dtype | `optvec jspace` realized-dose block (Python engine, this branch) | *declare* per dtype; report the measured floor beside it | low-dose readouts are read against a measured floor, not an assumed one |
| V8 | Ablation vs independent projection | `h − λPh` vs a QR-based projection at every position | tiny-model: `tests/test_ablator.py`; the review measured ≤ 1.8e-7; **no production runner** | *declare* | the removed subspace is the declared one on this runtime |
| V9 | Adapter activation and removal | outputs with adapter on vs off vs base model; foreign-substrate refusal | `tests/test_adapter_substrate.py`, `tests/test_lora_training_dtype.py`; **no production runner** for the on/off delta | *declare* | an adapter changes what it claims and only that |
| V10 | Training objective on the runtime | accumulated gradient equals the combined-batch gradient; partial groups unscaled | `tests/test_lora_train_objective.py` (`test_gradient_is_invariant_to_micro_batch_partitioning`, `test_accumulated_gradient_matches_a_hand_written_combined_batch`, `test_partial_final_micro_batch_and_group_are_not_scaled_down`); **no production runner** | *declare* (bf16 accumulation will not be bit-exact; a relative bound is the honest form) | the objective the sidecar stamps is the one optimized on this hardware |
| V11 | Completed-answer target render (Swift) | prompt render is a token prefix of the completed render; no generation suffix supervised | `FineTuneTokenizationTests.swift` on the committed template fixtures; **not run end-to-end on a trained adapter** | exact | the supervised span is the answer |
| V12 | J-lens reference agreement, both vocabulary paths | canonical readout vs the pinned reference on fixed residuals, watch-list and full-vocabulary | `steerlab-server jlens qualify <lens-id> <model-id>` (`referenceAgreement`) | *code*: `REFERENCE_TOLERANCE = 5e-2` max abs logit difference with the reference head in float32 | the lens's arithmetic on this runtime |
| V13 | J-lens stability, causal smoke, capability guard | committed prompts repeatable; a derived token direction shows a dose response; the battery holds at the intended range | `jlens qualify` (`stableReadouts`, `causalSmoke`, `capabilityGuard`) | *code*: `CAPABILITY_TOLERANCE = 0.15` | the lens is usable at that range; not that it explains behavior |
| V14 | Capability battery under steering | battery accuracy at each promoted cell vs baseline | `sweep` / `run` battery records | *code*: 0.15 battery tolerance, 0.45 distinct-2 floor (sweep selection rule) | a promoted cell is not confounded by degradation |
| V15 | Gradient check, trainable vector | finite-difference vs autograd through the sphere projection | `tests/test_optvec_gradient.py` (real tiny model); **no production runner** | *declare* | the optimized direction's gradient is the stated one |

## 3. Comparison discipline

- **Tiny-model rows are unit evidence.** A row whose only runner is a
  `Server/tests/` case stays labeled "tiny-model" in the results table. It is
  never relabeled as a large-model run.
- **Cross-engine identity is not the criterion.** MLX and CUDA activations do
  not byte-match, and a study's vectors are re-extracted per substrate
  (CONDUCTING-A-STUDY §6). Rows compare an engine against its own reference or
  against an independent computation on the same substrate.
- **Declare, then run.** Every *declare* tolerance is written into this file,
  with its justification, before the row's first execution. A tolerance
  changed after a failure is recorded as a change, with the failing result kept.
- **Skips are counted, not hidden.** A configuration that could not be run is
  a row with result "unavailable" and the reason, exactly as `site qualify`
  reports skips beside passes.

## 4. Results (to be filled)

| # | Configuration (slot values) | Command / job | Result artifact | Outcome | Notes |
|---|---|---|---|---|---|
| — | — | — | — | — | no row executed yet |

## 5. What is missing before this can run

- Production runners for V4, V5, V8, V9, V10 and V15: today these contracts are
  only exercised on tiny CPU fixtures. Each needs a verb or script that loads
  the production model on the node, performs the paired computation, and writes
  a stamped result under `diagnostics/` in the workspace (never under `runs/`).
- The bf16 floors for V7 on the 27B tier, measured at the doses the studies
  actually use; the J-space realized-dose block reports them once a run exists.
- Declared tolerances for every *declare* row, with a one-line justification
  each, entered here by the researcher before execution.

Until those exist, the correct statement in any report is: "production-scale
validation of these contracts is pre-declared in `docs/VALIDATION-MATRIX.md`
and has not been executed."
