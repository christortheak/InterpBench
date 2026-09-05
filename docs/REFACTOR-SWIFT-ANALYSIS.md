# Swift offline analysis refactor handoff

Branch: `codex/maintainability-refactor`

Date: 2026-09-05

Starting commit: `35f686cdc22d640057077638512cd4eeb60c15a4`

Latest integrated main baseline: `5309b5f2efc8f70dfb7b1a444feb5312b27c7704`.

## Result

Swift `analyze` and `rescore-style` now separate evidence loading, calculation,
report serialization and filesystem publication. `ExperimentTasks` retains the
existing API and manifest-verification entry gate, then delegates to the offline
workflow. Calculations consume captured manifest, source provenance, generations,
pinned taxonomy, attention checks and declared-target values. They neither
select a workspace nor reopen source files.

`ExperimentTasks.swift` shrank from 9,640 to 8,464 lines. The new analysis owners
are 35–336 lines each; shared prompt parsing/loading are 222 and 70 lines. The
remaining task file still owns substantial model execution. This slice completes
the offline-analysis boundary, not every Swift workflow or app state refactor.

## Ownership

Paths are under `Sources/ExperimentKit/`.

| Owner | Responsibility |
|---|---|
| `StudyAnalysisWorkflow.swift` | Coordinate evidence, calculation, rendering and publication; emit returned diagnostics |
| `StudyAnalysisRepository.swift` | Select completed runs; enforce source epoch; load generations and pinned measurement inputs from supplied roots |
| `StudyAnalysisContracts.swift` | Captured input, calculation results, diagnostics and existing report wire contracts |
| `StudyAnalysisCalculator.swift` | Decode sampled/instrument records, apply declared exclusions, assemble choice and stratified results, rescore style |
| `StudyAnalysisStatistics.swift` | Paired sampled/ordinal effects, bootstrap/Wilcoxon wiring, multiple-comparison families, strata and style means |
| `StudyAnalysisRendering.swift` | Report JSON, CSV escaping/columns, optional artifact presence and human-readable summaries |
| `StudyAnalysisWriter.swift` | Allocate a fresh run directory, stamp manifest/config provenance and persist supplied report bytes |
| `StudyPromptRepository.swift` | Read prompt bytes from a supplied root and enforce existing pin/override rules |
| `StudyPromptParsing.swift` | Decode prompts, validate duplicate IDs/factors/transcripts and collect attention checks |

The existing `ExperimentTasks` helper names delegate to these owners, including
run-inline statistics and prompt parsing. There is one implementation of each
moved rule. Nested record/report type names remain source compatible through the
contract files; new owners reference those types but never call task execution
or `ExperimentPanel`. No model acquisition or MLX operation enters offline
calculation. This is a responsibility boundary within the existing package, not
a new separately compiled package without MLX dependencies.

The explicit-root overload of `ExperimentStore.loadPinnedReasoningStyle` uses the
same pin validation as its compatibility overload. `ExperimentStore.manifestHash`
remains the shared pure fingerprint function; there is no second hash algorithm.

## Admission and compatibility

- Public analyze/rescore calls still run `loadVerified` before the offline
  workflow. The repository/workflow are internal and expect an already verified
  manifest; they are not public replacements that bypass manifest admission.
- Source selection retains the completed-run filename pattern, descending name
  ordering and required generations/report/snapshot files.
- Epoch refusals, foreign-substrate refusals, tolerated measurement drift and
  explicit acceptance of unstamped runs retain the shared `RunEpoch` policy.
- Workspace and prompt roots are supplied explicitly. The compatibility boundary
  preserves the existing distinction: experiments/runs/taxonomies use
  `ExperimentStore.workspaceRoot`; prompt files use `VectorCatalog.projectRoot`.
  It does not silently change the existing override seams.
- The source provenance stamp is captured with evidence, before calculation.
  Calculation and rendering use those values even if source files later become
  unavailable. This does not introduce an atomic filesystem snapshot or a lock
  against external writers; source runs retain their existing immutability
  contract.
- Sample/ordinal pairing, deterministic bootstrap seeds, correction families,
  within-item diagnostic exclusions, first-record factor metadata, declared
  target authority and null-versus-absent endpoint handling are preserved.
- Choice-margin diagnostics continue to use the original collected readouts,
  before declared exclusions. Changing that scientific behavior belongs in a
  separately reviewed correction, not this move.
- Report schemas, optional artifact presence, provenance and offline config
  fields are preserved. Source runs are never written. Repeated publication
  allocates distinct directories.
- Reports are serialized before publication. Result diagnostics are buffered as
  values and emitted after successful publication; epoch warnings retain their
  admission-time emission. This changes diagnostic timing on a later write
  failure, not scientific results or normal successful-run wording.

The UI and CLI still call the compatibility facade. Separating panel selection,
background-job ownership and workspace admission from global UI state remains
part of the panel-state slice; this change does not claim to solve workspace
switch races before the facade has captured its inputs.

## Verification

A move audit compared 21 extracted function bodies with the starting version.
They match after whitespace normalization, without expression substitutions:
statistics/corrections/strata, prompt/schema parsing, endpoint decoding, baseline
warnings, CSV rendering and epoch/source readers.

Six new tests in `Tests/ExperimentKitTests/StudyAnalysisBoundaryTests.swift` cover:

1. Two explicit workspaces with the same experiment/run names produce their own
   results, and calculation/rendering succeed after a captured source is removed.
2. Publication preserves every source byte, allocates distinct outputs and stamps
   the existing offline configuration/provenance.
3. Prompt loading honors its explicit root, captures checks/targets and refuses
   subsequent hash drift.
4. Null endpoint exclusions remain distinct from absent endpoints, including the
   exclusion artifact and accepted-unstamped provenance.
5. Style rescoring uses captured taxonomy values after the taxonomy file is
   removed and retains diagnostic status, pin provenance and exact CSV output.
6. A mismatched epoch refuses even with the unverified flag and publishes nothing.

Full Swift validation passed: **277 SteeringKit + 4,361 ExperimentKit tests
(4,638 total)**, with `TEST SUCCEEDED`. The initial run caught a directory-URL
trailing-slash mismatch in one new assertion; comparing filesystem paths fixed
that test, and the complete suite passed on rerun. Final log:
`/private/tmp/interpbench-analysis-swift-verified.log`.

The existing full suite continues to exercise pooled/stratified and ordinal
statistics, choice-delta coverage, exclusion rules, frozen prompt overrides,
taxonomy scoring, manifest epochs, measurement drift and CLI/UI compatibility.

Validation uses a byte-checked source/fixture copy under
`/private/tmp/interpbench-refactor-validation` and external scratch. Every build
uses Xcode beta and `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1`; tests run
serially with `CLANG_COVERAGE_MAPPING=NO`. No Python implementation changed in
this slice; its preceding full-suite result remains recorded in the stage and
lifecycle handoff.

## Integration and remaining work

Further main fixes remain intentionally deferred. Apply incoming edits to the
owners above, retain their regression assertions, and inspect semantic conflicts
in pin checks, source admission, pairing/exclusions, correction families and
artifact presence even when Git reports a clean merge. Integrate then-current
main before the main coding agent reviews and merges this branch, and run the
affected checks and both full suites.

The subsequent [study UI slice](REFACTOR-STUDY-UI.md) separates draft fields,
job controllers, result state and run/result view components. Its handoff records
remaining editor commands, global workspace boundaries and other feature views.
The science-agent handoff and J-lens recomputation remain separate.
