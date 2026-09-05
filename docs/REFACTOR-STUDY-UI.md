# Study UI state and component refactor handoff

Branch: `codex/maintainability-refactor`

Date: 2026-09-05

State/controller slice starting commit: `22daaca`.

Authoring-component continuation starting commit: `d494826`.

Latest integrated main baseline: `b6949ffdef0e48bc32813247f231feead2e9ed44`.

The [updated integration record](MAIN-INTEGRATION.md) covers the subsequent
main merge; the slice-specific results below remain its historical validation.

## Result

The study UI now separates editable protocol state, result selection and reads,
local operation lifetime, remote job following, and submission options.
`ExperimentPanel` composes these owners and retains study-authoring commands,
route admission, freeze coordination and import transactions. Its existing
binding names forward to the owners, so callers outside the extracted views
continue to observe the same state.

`UnifiedStudyRunner` no longer accepts an `ExperimentPanel`. It receives an
immutable submission request, a remote-job controller, the connection, a local
run capability and a notice sink. Submission options, server capabilities and
the displayed substrate label are captured before network suspension. Editing
the remote controls during upload cannot change the executor, verb, resources,
resume policy or parallel-job count already being submitted.

`ExperimentsPanelView.swift` decreased from 4,501 to 2,720 lines in the first
UI slice, then to 1,300 lines in the authoring continuation. Run controls,
preparation controls, live progress, local results, recent jobs, notices and
review/import/rename windows have their own components. Live progress and result
selection read their owners directly. Draft and compute-option bindings in the
extracted UI address their field owners. The authoring sections now have focused
view components too. Study authoring still calls the coordinator's admitted
store operations.

`ExperimentPanel.swift` decreased from 6,295 to 5,214 lines; 551 lines of old
binding spellings are isolated in `StudyPanelBindings.swift`. That bridge is not
another copy of state: every getter/setter addresses the actual owner. The panel
is still substantial because authoring and lifecycle transactions remain there.

## State and controller ownership

Paths are under `Sources/ExperimentKit/`.

| Owner | Responsibility |
|---|---|
| `StudyDraftState.swift` | Editable protocol/condition/attachment fields, form errors, prompt document backing, judge-kind stashes, selection synchronization over supplied defaults |
| `StudyResultsState.swift` | Local run selection/detail through an explicit `StudyResultRepository`; remote listing, bounded detail fetches and result status |
| `StudyLocalJobController.swift` | Run/validate/extract/sweep/paired-judge lifetime, cancellation flags, completion paths and live generation/judgment state |
| `StudyRemoteJobController.swift` | Reconnect identity, bounded logs, recent jobs, server job identities and terminal-state following |
| `StudySubmissionOptions.swift` | Editable compute controls and their immutable `StudySubmissionRequest` snapshot |
| `UnifiedStudyRunner.swift` | Unified bundle preflight, packaging/upload/submission and handoff to the remote-job owner |
| `StudyJobPresentation.swift` | Explicit notice/status/refresh/result-selection/live-log capabilities, supplied by the composition boundary |
| `StudySubmissionPresentation.swift` | Existing bundle-status wording |
| `StudyPanelBindings.swift` | Source-compatible projections for remaining panel consumers |

The state/controller owners never import or retain `ExperimentPanel`. The panel
installs weak presentation callbacks once. Local job completion names its study
when requesting result selection; the coordinator only selects that result when
that study is still selected. A user changing the editor's selection does not
reset the executing controller's cancellation/progress state.

The draft's synchronization guard preserves unsaved text on ordinary refresh.
Selection changes and explicit forced synchronization apply the manifest again.
Model, judge, variant and scenario defaults are resolved at the coordinator, then
supplied as values. The guard runs before those library/workspace reads, so a
routine progress refresh does not introduce extra inventory scans.

Public `ExperimentPanel` binding spellings and nested form/job/report type names
remain available. Setters that commands need across files are internal rather
than file-private; this does not expose formerly read-only fields for mutation
by clients outside ExperimentKit. The unified runner's method signature changed;
the app's two production call sites use the new explicit capabilities.

## View ownership

Paths are under `Sources/SteerLabApp/`.

| Component | Responsibility |
|---|---|
| `StudyRunControlsView.swift` | Substrate override, unified submission/preflight presentation, forced retry and remote options |
| `StudyPreparationControlsView.swift` | Validation/extraction controls and server residency callout |
| `StudyLiveRunView.swift` | Live progress and stop controls over `StudyLocalJobController` |
| `StudyResultsView.swift` | Local results over `StudyResultsState`, with supplied refresh/judge controls |
| `StudyRecentJobsView.swift` | Recent jobs over the remote controller, with supplied resume/import/refresh actions |
| `StudyResultReviewWindow.swift` | Generation/judgment review sheet and payload |
| `StudyPromptImportSheet.swift` | Existing JSONL import preview and action |
| `StudyRenameWindow.swift` | Existing canonical-name/display-label editor |
| `NoticesViews.swift` | Notice bell and feed |
| `StudyControlCopy.swift` | Shared existing help/caption strings |
| `StudySetupSection.swift` | Question/purpose, baseline model and revision/precision, prompt rendering, reasoning/token settings, scenario/baseline selection, funnel phase and Save Study Setup |
| `StudySamplingControls.swift` | Samples/play-throughs, seed policy/list and existing stochastic-design advisories |
| `StudyConceptsSection.swift` | Pinned recipes, method/rendering/reading-position/reference/corpus attachment controls and detach action |
| `StudyInjectionConditionsSection.swift` | Native conditions, baseline display, sign/random controls and removal; composes the existing AddConditionEditor |
| `StudyArmsSection.swift` | Saved/forward-referenced agent arms, confirmation policy and non-blocking evidence notes over supplied evidence/library values |
| `StudySeatsSection.swift` | Seat pickers, casting status/advisories and save/permutation actions |
| `StudyTaskPromptsEditor.swift` | Prompt loading/editing/pinning, import entry points and instrument-field preservation warnings |
| `StudyEvaluationSection.swift` | Instrument activation, supplied judging controls, analysis declarations and evaluation save action |

The view hierarchy retains the existing control order, labels, help text,
confirmation dialogs, GPU warning gate and report contents. Run/preflight state
now belongs to its component; review-sheet presentation belongs to the results
component. No installed application was replaced or launched for this refactor.

## Authoring-component continuation

Eight focused SwiftUI components replace the protocol/condition/seat/prompt and
evaluation sections formerly implemented inside `ExperimentsPanelView`. The
largest new file is 323 lines. These are separate View types, not extensions
that retain the original view's entire state and service access.

The editors receive the selected manifest and the existing coordinator. Editable
bindings continue to target `StudyDraftState`; commands still use the admitted
`ExperimentPanel` operations. This slice does not move authoring transactions out
of that coordinator. Setup receives a substrate label value rather than the chat
service. The arms view receives robustness reports, the substrate and available
variants; it performs no directory scan or library refresh. The parent retains
the existing asynchronous evidence-loading task. Evaluation receives a judging
view builder, so only its host needs to supply the service-dependent judging UI.

Import-sheet visibility and text remain in the parent and reach the prompt
editor as bindings. The parent still presents the import sheet and resolves its
destination/action through the selected study. This preserves the existing
presentation lifetime when the prompt disclosure is collapsed. Template/rename,
GPU warning, freeze confirmation and result dialogs keep their existing owners.

No control order, label/help text, frozen-study enablement, seed-policy legacy
value, confirmation condition, pin command or save path intentionally changed.
The scientific decisions still belong to ExperimentKit and its store/policy
owners. In particular, this is UI organization, not a new implementation of
extraction, control generation, reasoning budgets or instrument scoring.

## Behavior preserved and focused transition corrections

- Local/server routing and existing gate checks remain before task execution.
  Store writes, frozen-study restrictions, prompt preservation and freeze/import
  transactions were not rewritten.
- Local task calls, cooperative cancellation, partial-artifact wording and
  completion/error paths are retained. Four local execution bodies match after
  normalizing only explicit refresh/result callbacks. Twelve other moved
  state/progress/cancellation/presentation bodies match after whitespace
  normalization alone.
- UserDefaults retains the existing reconnect-job key. Recent-job limits,
  bounded logs, terminal status vocabulary, checkpoint guidance and the
  post-stream polling fallback are unchanged.
- A stopped or superseded remote stream cannot append a late line or overwrite
  a newer stream's status. Cancellation is not surfaced as a connection error.
- An older remote detail request cannot replace the status of a newer detail
  request. Remote listing requests also reject superseded results, and leaving
  server results clears an older loading indicator.
- Local result selection is bound to an explicit workspace repository. Equal run
  IDs from two workspaces do not reuse the previous workspace's detail.
- UI observation follows the owner through the compatibility getters/setters;
  no duplicated stored field must be manually kept in sync.

These transition protections concern presentation and operation identity. No
extraction, injection, fine-tuning, J-space calculation, scientific endpoint or
artifact schema changed.

## Validation

Full serial Swift validation passed: **277 SteeringKit + 4,369 ExperimentKit
tests (4,646 total)**, with `TEST SUCCEEDED`. The first UI slice log is
`/private/tmp/interpbench-ui-swift-final.log`. The authoring continuation also
passed the full serial suite, including compilation of all eight components in
`SteerLabApp-product`; log: `/private/tmp/interpbench-ui-authoring-swift.log`.

A mechanical preservation audit accounted for all **51 existing helper function
bodies** and the two inline Study Setup/Evaluation sections. They match after
whitespace normalization and the four explicit composition substitutions:
substrate label input, supplied variant library, judging view builder and
sampling component. Twenty-six helpers moved to the new components. The rest
remain in the parent. No new tests were added for this structural move;
existing authoring, seat casting, prompt preservation, instrument, draft-state
and scientific regression tests remain in the full suite. Interactive visual QA
was not performed; the installed application was not launched or replaced.

Eight tests added in the state/controller slice in
`Tests/ExperimentKitTests/StudyUIOwnershipTests.swift` cover:

1. Unsaved editor values survive ordinary refresh; forced/changed selection and
   workspace-specific defaults synchronize explicitly.
2. Editor/result resets leave active generation progress and cancellation state
   with the job owner; Run cancellation does not cancel validation or sweep.
3. Observation propagates through old binding names and direct owner writes;
   condition-mode edits retain strength-reset and inline-error clearing behavior.
4. Submitted options retain their values after later edits, including nested
   resume-policy changes.
5. A delayed failed remote detail fetch cannot overwrite a newer successful one.
6. A stopped stream cannot append delayed lines or replace newer job status.
7. Leaving server results clears the old loading indicator.
8. Identical run IDs in two explicit workspaces load the correct local detail.

The asynchronous tests use controlled continuations/streams and supplied
fetchers; they do not submit jobs, load models or require a running server.
The full Swift build includes the extracted SwiftUI components. Existing tests
continue through the compatibility API, retaining authoring, routing, freeze,
remote-detail, cancellation, result and scientific assertions.

Validation follows the established byte-checked source/fixture copy under
`/private/tmp/interpbench-refactor-validation`, with Xcode beta, the supplied
Metal toolchain, external DerivedData, serial testing and
`CLANG_COVERAGE_MAPPING=NO`. The initial full run found a new assertion comparing
macOS `/var` and `/private/var` aliases; the final test verifies distinct file
contents through the selected result path instead of relying on URL spelling.
No Python files changed in this slice.

## Subsequent study/design management slice

The [management handoff](REFACTOR-STUDY-MANAGEMENT.md) records the next slice:
commands and inventory moved to `StudyManagementController` and
`StudyDesignLibrary`, and management dialogs/actions moved to their own views.
The current main view is 875 lines; the coordinator is 4,653 lines. The earlier
counts and tests above describe the authoring-editor slice.

## Remaining work and integration

This is the study UI slice, not a rewrite of every app panel. The following work
remains appropriate as separately reviewed changes:

1. Decompose remaining remote submission/pipeline coordination and presentation,
   followed by remaining protocol/condition/prompt authoring commands. The
   study/design inventory and management actions have moved in the subsequent
   slice linked above. [Freeze coordination](REFACTOR-FREEZE-COORDINATION.md)
   has also moved; the main view is now 699 lines and the panel 4,252 lines.
2. Extend explicit request/workspace binding through the legacy submission and
   store/task boundaries. Those compatibility paths still use existing global
   workspace lookup; this slice does not make all model operations safe against
   arbitrary workspace switches during execution.
3. Apply the same ownership approach to other large feature panels/views, then
   retire compatibility projections when their production callers migrate.

Main fixes through `b6949ff` are now integrated. Before the main coding
agent's review and merge, recheck main and integrate any newer commits, applying
edits to the owners above. Pay particular attention to new panel field declarations: add storage to
the proper owner and a compatibility projection when needed. Carry regression
assertions over with changes to routing, admission, callbacks, cancellation,
remote pin checks, resume policy and report parsing. A clean textual merge does
not establish that asynchronous behavior or a scientific fix survived a move.

The separate science handoff and J-lens recomputation remain with their existing
workstream. This branch has not been installed into the running agent's setup.
