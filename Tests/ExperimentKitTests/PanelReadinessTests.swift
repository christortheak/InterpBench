import Foundation
import Testing

@testable import ExperimentKit

/// The reported cluster failure: a panel could be authored and selected, yet
/// Data & Prompts insisted no scenario was pinned — and "Create from template"
/// produced the file while the blocker stayed put.
@Suite("PanelReadinessTests")
struct PanelReadinessTests {

    private func workspace() throws -> URL {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-readiness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appending(components: "prompts", "panels"),
            withIntermediateDirectories: true)
        return root
    }

    private func manifest() -> ExperimentManifest {
        var m = ExperimentManifest(
            name: "panel-study", description: "", modelID: "test/model")
        m.studyKind = .multiAgent
        return m
    }

    @Test("an unpinned panel says the pin is written on SAVE")
    func unpinnedMessageNamesTheRealAction() throws {
        let root = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let rows = StudyDataReadiness.requirements(for: manifest(), workspaceRoot: root)
        let scenario = try #require(rows.first { $0.kind == .multiAgentScenario })

        #expect(scenario.status == .missing)
        // The old text said "author agents and turns", which is exactly what
        // the researcher had already done — selection simply does not pin.
        #expect(scenario.detail.contains("does NOT pin"))
        #expect(scenario.detail.contains("SAVE"))
    }

    @Test("scaffolding a panel pins it, so the blocker actually clears")
    func scaffoldingPinsThePanel() throws {
        let root = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        var m = manifest()

        let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
        let requirement = try #require(rows.first { $0.kind == .multiAgentScenario })

        // Stand in for the seed the app ships.
        let seed = root.appending(path: DataTemplates.scenario.seedRelativePath)
        try FileManager.default.createDirectory(
            at: seed.deletingLastPathComponent(), withIntermediateDirectories: true)
        let panel = MultiAgentScenario(
            name: "template", baseModelID: "m",
            agents: [.init(id: "a", name: "A", baseModelID: "m")],
            turns: [.init(id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go")])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(panel).write(to: seed)

        let created = try StudyDataReadiness.scaffold(requirement: requirement, in: root)
        #expect(FileManager.default.fileExists(atPath: created.path))

        // Before pinning the blocker is unchanged — the file alone is not enough.
        let stillMissing = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            .first { $0.kind == .multiAgentScenario }
        #expect(stillMissing?.status == .missing)

        #expect(
            try StudyDataReadiness.pinScaffolded(
                requirement: requirement, createdPath: requirement.path,
                into: &m, workspaceRoot: root))

        let after = try #require(
            StudyDataReadiness.requirements(for: m, workspaceRoot: root)
                .first { $0.kind == .multiAgentScenario })
        #expect(after.status == .present)
        #expect(m.multiAgentScenarioHash != nil)
    }

    @Test("a pinned file missing under a server target says where to look")
    func serverWorkspaceMissingFileIsExplained() throws {
        let root = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        var m = manifest()
        m.multiAgentScenarioPath = "prompts/panels/absent.json"

        let local = try #require(
            StudyDataReadiness.requirements(for: m, workspaceRoot: root)
                .first { $0.kind == .multiAgentScenario })
        let server = try #require(
            StudyDataReadiness.requirements(
                for: m, workspaceRoot: root, workspaceIsServer: true
            ).first { $0.kind == .multiAgentScenario })

        #expect(local.detail.contains("restore it"))
        // Under a server target the file may legitimately live on the server;
        // telling the researcher to re-author it would be wrong.
        #expect(server.detail.contains("exists on the server"))
    }
}

@Suite("PanelUnsavedWorkTests")
struct PanelUnsavedWorkTests {

    @MainActor
    @Test("New Scenario refuses to discard unsaved work without confirmation")
    func newScenarioRefusesWhenDirty() {
        let panel = MultiAgentPanel()
        panel.sharedMaterials = "a case record I just typed in"

        #expect(panel.hasUnsavedChanges)
        // The reported loss: this used to wipe the editor silently.
        #expect(panel.newScenario(discardingChanges: false) == false)
        #expect(panel.sharedMaterials == "a case record I just typed in")

        // Confirmed discard is the only way through.
        #expect(panel.newScenario(discardingChanges: true))
        #expect(panel.sharedMaterials.isEmpty)
    }
}
