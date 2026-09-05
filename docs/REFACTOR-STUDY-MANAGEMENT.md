# Study and design management refactor handoff

Branch: `codex/maintainability-refactor`

Starting commit: `afe5250`.

Integrated main baseline: `b6949ffdef0e48bc32813247f231feead2e9ed44` (0.9.5).

Date: 2026-09-05

## Result

Study creation, duplication, rename, trash, design creation/editing, metadata
updates, rename/delete and save-back now execute in `StudyManagementController`.
`StudyDesignLibrary` owns the design inventory and cached lineage. Neither owner
retains `ExperimentPanel` or `ChatService`. The controller can perform these
operations with a supplied draft and workspace model choices, without a panel.

`ExperimentPanel.swift` decreases from **5,214 to 4,653 lines**. Its original
management API remains available through 152 lines of compatibility projections
and delegates in `StudyManagementBindings.swift`; those introduce no duplicate
storage or transaction implementations. The largest new owner is 438 lines.

`ExperimentsPanelView.swift` decreases from **1,300 to 875 lines**. Study
selection/lifecycle controls and their dialogs live in `StudyManagementSection`;
lineage and the two design-save destinations live in `StudyDesignActionsView`.
The existing editor, results and run components remain in place.

## Owners and boundaries

| File | Responsibility |
|---|---|
| `Sources/ExperimentKit/StudyManagementController.swift` | Study inventory, selected study, display labels and rename invitation; create/duplicate/rename/trash and design write commands |
| `Sources/ExperimentKit/StudyDesignLibrary.swift` | Templates, selected/source design fields, instantiation invitation, new-study design choice, cached lineage, save-back availability and display summaries |
| `Sources/ExperimentKit/StudyManagementPresentation.swift` | Supplied workspace model choices and notice/selection/refresh callbacks |
| `Sources/ExperimentKit/StudyManagementBindings.swift` | Existing panel field/method spellings forwarded to their owners; host model choices resolved at the composition boundary |
| `Sources/SteerLabApp/StudyManagementSection.swift` | Study picker, creation/rename/duplicate/trash controls, advanced creation, rename/instantiation sheets and invitation consumption |
| `Sources/SteerLabApp/StudyDesignActionsView.swift` | Cached lineage/sibling display, save-back confirmation and save-as-new action |

Creation receives `StudyCreationContext` with a workspace default model and
inventory. It retains the existing carried-model rule: use the editable model
when available (or the inventory is unknown/empty), otherwise use the supplied
workspace default and explain the fallback. Legacy panel callers still do
nothing without a host/context. Draft fields remain in the shared
`StudyDraftState`, including the existing success-only clearing behavior.

The controller refreshes study inventory, labels and the design cache, validates
the selected name, then calls the coordinator's detail-refresh capability.
A changed selection invokes the coordinator's existing result/freeze/pipeline
reset and draft synchronization sequence. Local job progress/cancellation stays
with the local-job owner. Callbacks capture the coordinator weakly.

Design lineage still refreshes with inventory rather than reading files from a
view body. Selecting/removing a design still resolves the new-study choice and
clears missing library selections. Commands use the same `ExperimentStore` and
`StudyTemplateStore` operations, with their current global workspace resolution;
this slice does not introduce a new repository or cross-workspace transaction
model.

## Behavior preserved

- Placeholder creation invites rename only when the draft landed; advanced
  creation uses the same revision normalization and generation defaults.
- Canonical rename occurs before the optional display-label write. Store
  admission remains authoritative for frozen-study restrictions. Labels remain
  hash-exempt, and duplication produces an editable draft.
- Trash keeps the existing recoverable destination and refusal messaging.
  The UI still counts stamped runs at click time for its confirmation.
- Design deduplication, divergence messages, metadata-only description updates,
  edit-draft creation and in-place save-back preserve their prior sequencing.
  Save-back retains the original study's lifecycle and lineage stamps.
- The management section is always present, including with no selected study.
  It owns rename/instantiation presentation and consumes cross-tab invitations
  on change and appearance. Its appearance handler performs the existing
  refresh-before-consume sequence.
- The design-action component owns its save-back confirmation. Existing labels,
  help text and destructive-action confirmations are retained.

A mechanical audit compared all **19 moved command/library bodies**, all
**11 moved UI helper bodies**, and the composed selection/refresh sequence with
`afe5250`. They match after normalizing only explicit owner/input names and
bindings. No scientific calculation, store admission rule, artifact schema,
freeze transaction or server submission behavior was rewritten.

## Validation

Full serial Swift validation passed: **277 SteeringKit + 4,377 ExperimentKit
= 4,654 tests**, with `TEST SUCCEEDED`. The build includes the SwiftUI app
target and both new management views.

Final log: `/private/tmp/interpbench-study-management-swift-final.log`.

The initial full run exposed a new test-fixture assumption: deduplication applies
to an unchanged study with design lineage, not repeated export of an unlinked
source. The test now creates an unchanged edit instance through the real command
before checking deduplication. Production behavior was not changed to satisfy
that assertion.

Five new tests in `Tests/ExperimentKitTests/StudyManagementOwnershipTests.swift`
exercise the actual boundaries:

1. A standalone controller creates a study from supplied workspace choices,
   retains a valid carried model, and invites rename after successful creation.
2. Frozen rename/trash refuse, labels remain editable, and duplication yields
   a draft that can be moved to trash without deleting the original.
3. Design mint/dedup, rename, metadata update, edit/save-back and removal work
   without a panel and update the owned inventory.
4. A selection made through the owner propagates through the panel's observable
   compatibility property, synchronizes the editor/results, preserves unsaved
   text on ordinary refresh and leaves an active local job's progress intact.
5. Presentation callbacks do not keep the panel alive when its owner is retained.

Existing design round-trip, rename, lifecycle, store, scientific and UI-owner
regressions remain unchanged. Validation uses Xcode beta, the supplied Metal
toolchain, serial testing and the byte-checked external source/fixture copy and
DerivedData under `/private/tmp`. No Python source changed, so the full Python
result from the main integration remains applicable. No installed application
was replaced or launched; interactive visual QA remains outstanding.

## Remaining work and integration

The subsequent [freeze coordination slice](REFACTOR-FREEZE-COORDINATION.md)
is implemented. Next is remote submission/pipeline coordination and presentation. Protocol/condition/prompt authoring transactions and confirmation
study construction still reside in the coordinator; the existing template
instantiation table still composes with the panel. Other feature panels and
remaining global workspace boundaries remain separate work.

Before the other coding agents review and merge this branch into main, check
main again and integrate any newer commits. Apply changes to the owning commands
and library fields above, retaining compatibility delegates only while callers
need them. Carry regression assertions across when resolving moved-code
conflicts. The integration record documents the current main baseline.
