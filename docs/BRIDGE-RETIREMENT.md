# Swift bridge retirement: required before 1.0

The next substantive Swift refactor is the panel-authoring migration. Its exit
criterion is removal of all four transitional bridge files, with callers using
focused owners or intentional panel commands. No compatibility bridge may ship
in 1.0 or in the codebase linked from the launch blog.

## Current implementation status

The workflow implementation has retired `StudyPanelBindings.swift`: all 109
forwarding properties are gone. The app, local HTTP handlers and tests now read
and mutate `StudyDraftState`, local/remote job controllers, result state and
submission state directly. SwiftUI edits bind to the relevant observable owner.
The other three bridges and the substantive panel-authoring migration remain;
the 1.0 retirement gate is not yet satisfied.

The historical inventory below remains the ratchet baseline. Its old property
bridge file/count describes what was removed, not an existing compatibility API.
No baseline allowance was expanded.

The property-access migration has a reproducible parsed syntax-tree audit against
`a9545df`. `scripts/ci/audit-panel-owner-access.swift` derives all 109 accessor
mappings from that baseline and requires the source-file census to differ only
by deletion of the bridge. It compares complete syntax trees after normalizing
those owner accesses, equivalent `@Bindable` aliases, and the explicit initializer
for one formerly shorthand optional binding. Trivia is excluded; statement,
literal, argument, control-flow and declaration structure otherwise remain.
This is a syntax audit, not compiler-resolved type proof; the compiler and both
suites remain required. The audit passed for 45 changed source/test files.

To reproduce from this checkout (all scratch stays outside the workspace):

```sh
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
export TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1
audit_scratch=$(mktemp -d /private/tmp/panel-owner-audit.XXXXXX)
mkdir "$audit_scratch/before" "$audit_scratch/after"
git archive a9545df | tar -x -C "$audit_scratch/before"
git archive 22024f7 | tar -x -C "$audit_scratch/after"
audit_host="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/host"
xcrun swiftc -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
  -target arm64-apple-macosx15.0 -I "$audit_host" -L "$audit_host" \
  -Xlinker -rpath -Xlinker "$audit_host" \
  scripts/ci/audit-panel-owner-access.swift -o "$audit_scratch/audit"
"$audit_scratch/audit" "$audit_scratch/after" "$audit_scratch/before"
```

This checkpoint audit intentionally flags subsequent semantic work; rerun it on
the property-retirement commit for its original proof, rather than weakening its
comparison to accommodate later features.

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
