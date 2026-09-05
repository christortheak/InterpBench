# Execution stages and draft/freeze policy handoff

Branch: `codex/maintainability-refactor`

Date: 2026-09-05

Starting commit for this slice: `6e416f4`.

Latest integrated main baseline: `5309b5f2efc8f70dfb7b1a444feb5312b27c7704`.

## Result and scope

Python execution is no longer implemented in `tasks.py`. The facade keeps the
existing entry points and import spellings, and binds the pipeline's stage/model
capabilities. Model acquisition, vector materialization, condition execution,
sweeps, panel runs and judging have independent owners. The facade shrank from
9,580 lines to 348; the benefit is the dependency direction and ownership, not
the line count alone.

Python and Swift save/draft admission, protocol edits, declaration rules and
freeze-evidence decisions now operate on supplied values. The stores gather
capability/evidence facts and retain persistence, pinned-input copying,
preregistration export and Git sequencing. This does not replace every artifact
authoring operation with a policy object or split the app's presentation state.

This slice preserves the integrated scientific baseline. It does not complete
the separate scientific handoff or J-lens recomputation, and it does not integrate
main's subsequent commits. The active checkout and running installation are not
modified by development in this worktree.

## Python execution ownership

Paths are under `Server/steerlab_server/experiment/`.

| Concern | Owner | Boundary |
|---|---|---|
| Model access | `model_resources.py` | Provider lock/context, resident dtype checks, revision pinning |
| Vectors | `vector_materialization.py` | Extracted/pinned bundles and provenance persistence |
| Layer resolution | `layer_resolution.py` | Existing layer declarations and resolution rules |
| Extraction | `extraction_workflow.py` | Model scope, extraction and diagnostics |
| Validation | `validation_workflow.py` | Probes, cosine comparisons and battery evidence |
| Conditions | `condition_execution.py` | Effective interventions, adapter scope and one-condition execution |
| Run | `run_workflow.py` | Source resolution, record/checkpoint orchestration and completion |
| Pre-model checks | `run_preflight.py` | Prompt/instrument/artifact/scenario and resume admission |
| Panel runs | `panel_workflow.py` | Scenario execution, transcript retention and interrupted-run reporting |
| Optional readouts | `run_readouts.py` | Verified variant identities and J-lens trace lifecycle |
| Sweep | `sweep_workflow.py` | Declared/legacy grids, shared model/judge scopes and checkpoint coordination |
| Sweep evidence | `sweep_evidence.py` | Retained generations, progress and deferred judgment completion |
| Sweep judging | `sweep_judging.py` | Judge admission, shared judge scope and packet emission |
| Scoring | `choice_scoring.py` | Choice/logprob scoring and explicit battery backend binding |
| Judge resources | `judge_resources.py` | Model identities, capacity, column release and local generation |
| Evaluation | `evaluation_workflow.py` | Paired judging and response coding |
| Evaluation evidence | `evaluation_evidence.py` | Judgment provenance, human validation and retry admission |
| Deferred evaluation | `deferred_evaluation.py` | Packet emission, fanout workers and completion |
| Rubrics | `rubric_inputs.py` | Pinned rubric resolution and refusals |
| Forward references | `forward_resolution.py` | Promoted-agent identity and concrete artifact resolution |
| Execution reporting | `execution_reporting.py` | Substrate/sampling stamps and configuration advisories |

These owners do not import `tasks.py`. Cross-owner function calls address their
owning module, so a dependency has one injection point. Existing model-provider,
cancellation, checkpoint, release and run-directory callback signatures remain.
Context-manager scopes and adapter cleanup were moved with their workflows.

The API's variant-battery route uses `choice_scoring`; the tokenizer preflight
route uses `task_inputs`. Existing pipeline/API tests still replace public stage
entry points at the facade. Tests replacing private helpers now patch their
canonical owner; their behavioral assertions have not been removed.

## Lifecycle ownership

| Python owner | Responsibility |
|---|---|
| `manifest_errors.py` | Shared `ExperimentStoreError` identity and structured refusal fields |
| `manifest_mutation_policy.py` | Save, draft upload/replacement, freeze-status admission and server-pin preservation |
| `draft_protocol_policy.py` | Protocol vocabulary, field validation, clearing and merged reasoning/system declarations |
| `manifest_declaration_policy.py` | Model-output applicability, OptVec exemption, judge identities, dtype/revision rules and judge validity |
| `freeze_policy.py` | Ordered gates over `FreezeEvidence`, force/refusal disposition, variant and battery evidence decisions |

`experiment_store` continues to expose old helper names. It reads the existing
manifest and capability/evidence records, delegates decisions, and performs the
existing atomic save or freeze transaction. `FreezeEvidence` contains values,
not callbacks into the store. The numerical and rendering rules continue to use
their existing domain implementations.

Swift paths are under `Sources/ExperimentKit/`:

| Owner | Responsibility |
|---|---|
| `ManifestMutationPolicy.swift` | Save admission plus draft-edit and freeze-status admission |
| `ManifestDraftEdits.swift` | Thirteen protocol/field editors, sampling validation and vocabulary over an in-memory manifest |
| `ManifestDeclarationPolicy.swift` | Manifest-only pipeline/judge/dtype/revision rules and applicability |
| `FreezePolicy.swift` | Gate outcomes, refusal identity, validation/variant/battery decisions over supplied facts |
| `FreezeGateContracts.swift` | Existing nested `ExperimentStore` gate contracts retained for source compatibility |

`ExperimentStore.updateDraft` still loads fresh, admits, mutates and saves.
Capability records are gathered by the store and supplied to the field editor.
Freeze's gate table gathers matching evidence on the requested run substrate and
supplies values to policy. The store retains the never-skippable verification,
input snapshotting, commit/stamp ordering and post-commit cleanliness check.
Existing Python/Swift lifecycle differences are preserved rather than silently
harmonized as part of a refactor.

## Verification

- All **136 Python stage definitions** match the previous parsed code after
  normalizing import qualification. Fourteen constants/aliases also moved with
  their owners. No numerical kernel was rewritten.
- Offline-import tests cover the lifecycle policy owners without Torch,
  Transformers, PEFT or `tasks`. Execution owners also import with `tasks`
  explicitly blocked.
- Six new Python value-policy tests cover clearing intent, frozen immutability,
  omitted versus cleared pins, refusal without draft mutation, panel scope,
  vacuous validation and force/refusal disposition.
- Five new Swift tests cover panel scope, vacuous validation, substrate-specific
  repairs, preservation of unexpected errors and draft clearing/status rules.
- Swift full suite: **277 SteeringKit + 4,355 ExperimentKit tests passed**
  (`TEST SUCCEEDED`), serially under Xcode beta and the supplied Metal toolchain.
- Python full suite: **5,612 passed, 9 skipped**, including loopback runner
  integration tests. The final focused stage/lifecycle rerun also passed all
  **147 tests** before the full run.

Swift validation uses the byte-checked source/fixture copy under
`/private/tmp/interpbench-refactor-validation` and external DerivedData. Follow
the toolchain invocation in `MAINTAINABILITY-REFACTOR.md`; do not share the active
checkout's scratch or install this branch into a running agent's environment.

## Existing defect found during the move audit

`complete_sweep_judgment` calls `_append_progress` when `select_cell` returns
`None`. That helper is local to `_sweep_with_spec`, so it is not in scope during
deferred completion. This was present at the starting commit and was also
observed in main during this review. It now appears in `sweep_evidence.py`.

The correction agent should cover a deferred judged sweep with no eligible or
improving cell. Completion should retain the no-selection recommendation and
write its normal verified completion artifacts instead of raising `NameError`.
Choose the appropriate completion-path writer; do not merely import a helper
whose closure belongs to a different sweep run. This defect is unchanged by
the move, and passing the full existing suite does not cover this branch.

## Integrating later main fixes

1. Merge the then-current main into this worktree before final review/merge.
2. Follow the ownership tables above when resolving edits to `tasks.py` or the
   stores. Apply fixes in their new implementation owner, and carry regression
   tests across with their original assertions.
3. Keep facade stage injection where API/CLI tests replace an entire operation;
   move private-helper injection to the helper's actual owner.
4. Recheck semantic conflicts even after a clean Git merge: gate ordering,
   force stamps, missing evidence, model release, adapter cleanup, checkpoints,
   artifact provenance and scientific aborts are the important contracts.
5. Run affected tests and both full suites, then hand the branch to the main
   coding agent for independent review and merge. J-lens recomputation remains
   owned by the existing science workstream.

The subsequent [Swift offline analysis slice](REFACTOR-SWIFT-ANALYSIS.md) is now
implemented, followed by the [study UI state/component slice](REFACTOR-STUDY-UI.md).
That handoff records remaining editor commands and feature views. Compatibility
bridges can be retired after their production callers have migrated.
