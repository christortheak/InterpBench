import Foundation
import Testing

@testable import ExperimentKit

/// Write-funnel normalization of sweep-spec instrument paths
/// (`SweepSpecForm.workspaceRelativeNormalized`, field gap 2026-08-07): the
/// composer's free-text instrument fields accept a hand-picked ABSOLUTE
/// path, and one inside the workspace must serialize workspace-relative —
/// the same rule agents get at `ModelVariantStore.save`
/// (`ArtifactIdentity.workspaceRelative`), because an absolute Mac path in
/// a manifest resolves on this machine and dies on the cluster. Paths
/// outside the workspace pass through untouched: already non-portable, and
/// rewriting them would only hide it.
struct SweepSpecNormalizationTests {

    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "sweep-norm-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    @Test func absolutePathsInsideTheWorkspaceNormalizeInEveryField() throws {
        try withWorkspace { workspace in
            var spec = ExperimentManifest.SweepSpec(
                devPromptsFile: workspace
                    .appending(path: "prompts/dev/d.jsonl").path,
                batteryFile: workspace
                    .appending(path: "prompts/batteries/b.jsonl").path)
            spec.selection = .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: workspace
                        .appending(path: "prompts/choices/c.jsonl").path))
            let normalized = SweepSpecForm.workspaceRelativeNormalized(spec)
            #expect(normalized.devPromptsFile == "prompts/dev/d.jsonl")
            #expect(normalized.batteryFile == "prompts/batteries/b.jsonl")
            #expect(
                normalized.selection?.objective?.choicePromptsFile
                    == "prompts/choices/c.jsonl")
        }
    }

    @Test func perConceptMapValuesNormalizeIndividually() throws {
        try withWorkspace { workspace in
            var spec = ExperimentManifest.SweepSpec()
            spec.selection = .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFiles: [
                        "fear": workspace
                            .appending(path: "prompts/choices/fear.jsonl").path,
                        "courage": "prompts/choices/courage.jsonl",
                    ]))
            let normalized = SweepSpecForm.workspaceRelativeNormalized(spec)
            #expect(
                normalized.selection?.objective?.choicePromptsFiles == [
                    "fear": "prompts/choices/fear.jsonl",
                    "courage": "prompts/choices/courage.jsonl",
                ])
        }
    }

    /// Outside the workspace there is nothing to relativize against; a
    /// relative path is already the written shape. Both pass through, and
    /// nothing else in the spec (grid, constraints, controls) is touched.
    @Test func foreignAndRelativePathsPassThroughUntouched() throws {
        try withWorkspace { _ in
            var spec = ExperimentManifest.SweepSpec(
                devPromptsFile: "prompts/dev/dev-prompts.jsonl",
                batteryFile: "/scratch/other-ws/prompts/batteries/b.jsonl")
            spec.selection = .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: "/scratch/other-ws/choices.jsonl"),
                constraints: .init(
                    capabilityTolerance: 0.2, coherenceFloor: 0.5),
                controls: .init(matchedNormRandomMargin: 0.05))
            let normalized = SweepSpecForm.workspaceRelativeNormalized(spec)
            #expect(normalized == spec)
        }
    }

    /// A spec with no selection block (the historical default criterion)
    /// still gets its dev/battery paths normalized, nothing else invented.
    @Test func aSelectionlessSpecNormalizesJustTheInstrumentPaths() throws {
        try withWorkspace { workspace in
            let spec = ExperimentManifest.SweepSpec(
                devPromptsFile: workspace
                    .appending(path: "prompts/dev/d.jsonl").path,
                batteryFile: "prompts/batteries/basic.jsonl")
            let normalized = SweepSpecForm.workspaceRelativeNormalized(spec)
            #expect(normalized.devPromptsFile == "prompts/dev/d.jsonl")
            #expect(normalized.batteryFile == "prompts/batteries/basic.jsonl")
            #expect(normalized.selection == nil)
        }
    }
}

/// Store-backed write-shape test — extends the serialized
/// `ExperimentStoreTests` suite because it uses the process-global
/// workspace override (same pattern as the `setSweepSpec` round-trip
/// tests in `OptimizationLifecycleTests`).
extension ExperimentStoreTests {

    private func withNormalizationWorkspace<T>(
        _ body: (URL) throws -> T
    ) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "sweep-norm-funnel-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    /// The funnel both declare flows share (`ExperimentPanel.setSweepSpec`)
    /// applies the rule: a declare carrying an absolute in-workspace
    /// choice-prompts path STORES the workspace-relative form — and the
    /// save-time instrument validation (the engine's own loader) runs
    /// against that normalized shape.
    @Test @MainActor func setSweepSpecStoresWorkspaceRelativeInstrumentPaths() throws {
        try withNormalizationWorkspace { workspace in
            let choicesDirectory = workspace
                .appending(path: "prompts/choices")
            try FileManager.default.createDirectory(
                at: choicesDirectory, withIntermediateDirectories: true)
            let choicesURL = choicesDirectory.appending(path: "c.jsonl")
            try #"{"id": "a", "prompt": "P", "options": ["yes", "no"]}"#
                .write(to: choicesURL, atomically: true, encoding: .utf8)
            _ = try ExperimentStore.create(
                name: "norm-funnel", description: "", modelID: "test/model")
            let panel = ExperimentPanel()

            var spec = ExperimentManifest.SweepSpec()
            spec.selection = .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: choicesURL.path))
            #expect(panel.setSweepSpec(spec, for: "norm-funnel"))
            let loaded = try ExperimentStore.load(name: "norm-funnel")
            #expect(
                loaded.sweep?.selection?.objective?.choicePromptsFile
                    == "prompts/choices/c.jsonl")
        }
    }
}
