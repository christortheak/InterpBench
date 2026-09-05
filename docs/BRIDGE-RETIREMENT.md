# Swift bridge retirement: required before 1.0

The next substantive Swift refactor is the panel-authoring migration. Its exit
criterion is removal of all four transitional bridge files, with callers using
focused owners or intentional panel commands. No compatibility bridge may ship
in 1.0 or in the codebase linked from the launch blog.

## Baseline and callers

| File under `Sources/ExperimentKit/` | Members scanned | Caller files |
|---|---:|---:|
| `StudyFreezeBindings.swift` | 11 | 3 |
| `StudyManagementBindings.swift` | 32 | 26 |
| `StudyPanelBindings.swift` | 109 | 41 |
| `StudyRemoteCoordinationBindings.swift` | 8 | 3 |

The bridge files total 915 lines. The panel bindings contain 109 members; the
other counts include the public/internal command and context members needed by
the scan. The complete per-member, per-file caller inventory with occurrence
counts is [swift-bridge-callers.json](../scripts/ci/swift-bridge-callers.json).
The lists below identify source and test consumers for each bridge; consult the
JSON for the exact members each uses.

### StudyFreezeBindings.swift

- `Sources/ExperimentKit/WebServer.swift`
- `Sources/SteerLabApp/ExperimentsPanelView.swift`
- `Sources/SteerLabApp/StudyIssuesSection.swift`

### StudyManagementBindings.swift

- `Sources/ExperimentKit/ExperimentPanel+Factorial.swift`
- `Sources/ExperimentKit/ExperimentPanel.swift`
- `Sources/ExperimentKit/WebServer.swift`
- `Sources/SteerLabApp/ExperimentsPanelView.swift`
- `Sources/SteerLabApp/HomeDashboardView.swift`
- `Sources/SteerLabApp/ModelVariantsPanelView.swift`
- `Sources/SteerLabApp/OptimizationRunsView.swift`
- `Sources/SteerLabApp/StudyManagementSection.swift`
- `Sources/SteerLabApp/StudyPreparationControlsView.swift`
- `Sources/SteerLabApp/StudyRenameWindow.swift`
- `Sources/SteerLabApp/StudyTypeOverviewColumn.swift`
- `Sources/SteerLabApp/TemplateInstantiationSheet.swift`
- `Sources/SteerLabApp/TemplatesPanelView.swift`
- `Tests/ExperimentKitTests/ConfirmationDraftShortcutTests.swift`
- `Tests/ExperimentKitTests/EvaluationDeclarationTests.swift`
- `Tests/ExperimentKitTests/ExtractionDeclarationPickerTests.swift`
- `Tests/ExperimentKitTests/HeadlessProtocolSaveTests.swift`
- `Tests/ExperimentKitTests/InstrumentActivationTests.swift`
- `Tests/ExperimentKitTests/JudgeKindFieldOwnershipTests.swift`
- `Tests/ExperimentKitTests/OptimizationLifecycleTests.swift`
- `Tests/ExperimentKitTests/SeatCastingTests.swift`
- `Tests/ExperimentKitTests/StudyDesignRoundTripTests.swift`
- `Tests/ExperimentKitTests/StudyDesignTests.swift`
- `Tests/ExperimentKitTests/StudyManagementOwnershipTests.swift`
- `Tests/ExperimentKitTests/StudyTemplateBatchTests.swift`
- `Tests/ExperimentKitTests/SweepJudgeResolutionTests.swift`

### StudyPanelBindings.swift

- `Sources/ExperimentKit/ChatService.swift`
- `Sources/ExperimentKit/ExperimentPanel+Factorial.swift`
- `Sources/ExperimentKit/ExperimentPanel.swift`
- `Sources/ExperimentKit/WebServer.swift`
- `Sources/SteerLabApp/ActivityFeedColumn.swift`
- `Sources/SteerLabApp/AddConditionEditor.swift`
- `Sources/SteerLabApp/DataReadinessSection.swift`
- `Sources/SteerLabApp/DiscriminantControlsSection.swift`
- `Sources/SteerLabApp/EvaluationClarityViews.swift`
- `Sources/SteerLabApp/ExperimentsPanelView.swift`
- `Sources/SteerLabApp/HomeDashboardView.swift`
- `Sources/SteerLabApp/JudgingSectionView.swift`
- `Sources/SteerLabApp/ModelVariantsPanelView.swift`
- `Sources/SteerLabApp/OptimizationRunsView.swift`
- `Sources/SteerLabApp/RunResultsViews.swift`
- `Sources/SteerLabApp/SectionContainers.swift`
- `Sources/SteerLabApp/StudyArmsSection.swift`
- `Sources/SteerLabApp/StudyConceptsSection.swift`
- `Sources/SteerLabApp/StudyEvaluationSection.swift`
- `Sources/SteerLabApp/StudyManagementSection.swift`
- `Sources/SteerLabApp/StudyPreparationControlsView.swift`
- `Sources/SteerLabApp/StudyRunControlsView.swift`
- `Sources/SteerLabApp/StudySamplingControls.swift`
- `Sources/SteerLabApp/StudySetupSection.swift`
- `Sources/SteerLabApp/StudyTaskPromptsEditor.swift`
- `Sources/SteerLabApp/TemplatesPanelView.swift`
- `Sources/SteerLabApp/ValidateStudyButtonRow.swift`
- `Tests/ExperimentKitTests/ConfirmationDraftShortcutTests.swift`
- `Tests/ExperimentKitTests/EvaluationDeclarationTests.swift`
- `Tests/ExperimentKitTests/ExtractionDeclarationPickerTests.swift`
- `Tests/ExperimentKitTests/HeadlessProtocolSaveTests.swift`
- `Tests/ExperimentKitTests/JudgeKindFieldOwnershipTests.swift`
- `Tests/ExperimentKitTests/JudgingCustodyAdvisoryTests.swift`
- `Tests/ExperimentKitTests/OptimizationLifecycleTests.swift`
- `Tests/ExperimentKitTests/RemoteSubmissionLabelTests.swift`
- `Tests/ExperimentKitTests/RobustnessJudgeSurfaceTests.swift`
- `Tests/ExperimentKitTests/SeatCastingTests.swift`
- `Tests/ExperimentKitTests/StudyDesignRoundTripTests.swift`
- `Tests/ExperimentKitTests/StudyManagementOwnershipTests.swift`
- `Tests/ExperimentKitTests/StudyUIOwnershipTests.swift`
- `Tests/ExperimentKitTests/WorkbenchViewerHygieneTests.swift`

### StudyRemoteCoordinationBindings.swift

- `Sources/ExperimentKit/ExperimentPanel.swift`
- `Sources/SteerLabApp/ExperimentsPanelView.swift`
- `Tests/ExperimentKitTests/StudyRemoteCoordinationTests.swift`

## Migration order

1. Move protocol, prompt-file and condition-authoring transactions from
   `ExperimentPanel` into focused authoring owners. Give each operation an
   explicit workspace/request and preserve draft-only writes, pin invalidation,
   arm compatibility, refusal text and refresh behavior.
2. Migrate draft field consumers to `StudyDraftState`; migrate local/remote job,
   result and submission consumers directly to their existing owners. Update
   SwiftUI bindings, web endpoints, CLI-related consumers and tests together.
3. Migrate study/design consumers to `StudyManagementController` and
   `StudyDesignLibrary`. Keep the panel as composition/navigation glue, not a
   second source of study or template state.
4. Move freeze and remote request construction into intentional coordinators or
   command interfaces. Preserve the captured context, cancellation and stale
   response guards; do not delete the bridges by merely copying their bodies
   back into `ExperimentPanel` or renaming the files.
5. Remove the four bridge files and confirm no compatibility forwarding remains
   elsewhere. Keep legitimate public operations; remove temporary access paths.

## Enforcement and exit criteria

```sh
python scripts/ci/check-swift-bridge-retirement.py
python scripts/ci/check-swift-bridge-retirement.py --release
```

The normal CI check permits decreasing use but refuses new bridge members,
new caller/member occurrences, and additional `*Bindings.swift` extensions of
`ExperimentPanel`. The 1.0 check refuses any remaining listed bridge. CI selects
that strict check automatically when the package version or release tag reaches
1.x or later. Explicitly run `--release` before publishing the launch article,
regardless of version.

This is a syntactic ratchet, not Swift compiler reference resolution. It scans
known panel receivers, direct `.experiments` accesses and unqualified uses in
panel extensions. Aliases passed through arbitrary functions or renamed files
need manual review. The checked-in inventory includes all currently detected
callers; do not regenerate it to approve new debt. Reductions should be recorded
as migrations land so retired dependencies cannot be reintroduced up to an old
allowance. Inventory increases require explicit review of the migration design.

Completion requires:

- No listed bridge files and no replacement compatibility forwarding layer.
- Callers migrated to owners or deliberate command APIs; one mutable state owner
  for each concern.
- Preserved scientific write/identity gates and async context behavior, with
  focused regressions for the migrated authoring transactions.
- Full serial Swift suite and app build using Xcode beta and the Metal toolchain.
- Interactive authoring, freeze, run, cancellation and results smoke checks.
- Current main integrated and the main coding agents' review completed.
