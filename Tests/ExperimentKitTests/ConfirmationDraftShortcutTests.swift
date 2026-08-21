import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The confirm shortcut (Agents → Optimizations → "Open Studies — Confirm
/// agent") must CREATE a new confirmation draft — a confirmation is a new
/// preregistered study — and leave the screen study untouched, never
/// flipping its phase in place (P1 fix 2026-07-19). Declared as an
/// extension of the serialized `ExperimentStoreTests` suite because it uses
/// the process-global workspace override (the same seam as the composer
/// tests: the panel reads experiments AND the model-variant library, so the
/// whole workspace root moves, not just the experiments root).
extension ExperimentStoreTests {

    private func makeConfirmShortcutWorkspace() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "confirm-shortcut-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        return temp
    }

    private func cleanupConfirmShortcutWorkspace(_ root: URL) {
        WorkspaceRoot.programmaticOverride = nil
        ExperimentRootOverrideLock.release()
        try? FileManager.default.removeItem(at: root)
    }

    /// Plants a confirmable agent artifact (matching base model, single
    /// injection, no adapters) in the workspace's model-variant library,
    /// optionally with a promotion birth certificate naming its source
    /// experiment.
    private func plantConfirmableAgent(
        named name: String, promotedFrom experiment: String?
    ) throws {
        let directory = ModelVariantStore.directory.appending(component: name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        var artifact = ModelVariantArtifact(
            name: name, baseModelID: "test/model",
            injections: [
                .init(concept: "ghost", vectorArtifactID: "v", layer: 4, alpha: 2)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
        if let experiment {
            artifact.promotion = .init(
                experiment: experiment, experimentHash: "00",
                promotedAt: "2026-07-19T00:00:00Z", promotedBy: "criterion",
                sweepRun: "runs/x",
                winningCell: .init(layer: 4, alpha: 2),
                substrate: "swift-mlx", appVersion: "test")
        }
        try JSONEncoder().encode(artifact).write(
            to: directory.appending(component: "model-variant.json"))
    }

    @Test @MainActor func confirmShortcutCreatesANewDraftAndLeavesTheScreenStudyAlone()
        throws
    {
        let root = try makeConfirmShortcutWorkspace()
        defer { cleanupConfirmShortcutWorkspace(root) }

        // The screen study carries BOTH its scientific pins and a full
        // load of EXECUTION state — the draft must inherit only the
        // former (P1 second pass 2026-07-19: inherit-by-allowlist).
        var screen = try ExperimentStore.create(
            name: "screen-study", description: "screen", modelID: "test/model")
        screen.phase = "screen"
        screen.taskPromptsFile = "prompts/tasks/screen.jsonl"
        screen.taskPromptsHash = "screen-pool-hash"
        screen.modelRevision = "abc123"
        screen.concepts = [
            .init(
                name: "ghost", stimulusSetHash: "00",
                options: ExtractionOptions(method: .meanDifference))
        ]
        screen.neutralCorpusHash = "neutral-hash"
        screen.markersHash = "markers-hash"
        screen.judges = [.init(name: "opus", kind: "claude", model: "claude-x")]
        screen.judgeRubricFile = "prompts/rubrics/r.md"
        screen.judgeRubricHash = "rubric-hash"
        screen.capabilityBatteryFile = "prompts/batteries/b.jsonl"
        screen.capabilityBatteryHash = "battery-hash"
        screen.outcomeInstruments = ["answerTokenLogprob"]
        screen.caseFamily = "sentencing"
        screen.humanBaseline = .init(path: "prompts/baselines/h.csv", hash: "hb")
        screen.temperature = 0.7
        screen.samplesPerItem = 5
        screen.seedPolicy = "derivedSHA256"
        screen.systemPrompt = "You are a judge."
        // Execution state that must NOT travel:
        screen.sweep = .init(devPromptsFile: "prompts/dev/dev.jsonl")
        screen.pipeline = .object(["stages": .array([])])
        screen.conditions = [
            .init(name: "baseline", slots: []),
            .init(
                name: "ghost-recommended",
                slots: [.init(concept: "ghost", layer: 4, alpha: 2)],
                selection: .init(
                    sweepRun: "runs/x",
                    criterion: .init(objective: .init(metric: "markerDensity")),
                    devPromptsHash: "00",
                    winningCell: .init(layer: 4, alpha: 2), metrics: [:])),
        ]
        screen.variantConditions = [
            .init(
                name: "arm", artifactPath: "p", artifactHash: "h",
                artifact: .init(
                    name: "arm", baseModelID: "test/model",
                    promptMode: "chatAssistant", qwenThinkingEnabled: false,
                    temperature: 0, systemPrompt: ""))
        ]
        screen.promotionRule = .init(
            fdrThreshold: 0.05, doseMonotone: true, exceedsRandomFloor: true,
            capabilityGate: "battery within 0.05")
        screen.perturbationPolicy = .init(
            sourceAgent: .init(
                name: "old-agent", artifactPath: "p", artifactHash: "h",
                promoted: true),
            concept: "ghost", cell: .init(layer: 4, alpha: 2),
            alphaDeltas: [0.2], includeMatchedNormControl: true,
            declaredAt: "2026-07-19T00:00:00Z")
        screen.readerRefs = [.init(path: "prompts/readers/g.json", hash: "rh", concept: "ghost")]
        screen.acknowledgeUnequalOptionLengths = true
        try ExperimentStore.save(screen)
        let screenBefore = try ExperimentStore.load(name: "screen-study")

        // Two promoted agents in the library: one from ANOTHER study that
        // sorts first alphabetically, and the screen study's own. The
        // shortcut must preselect the screen study's, not merely the first.
        try plantConfirmableAgent(named: "aaa-agent", promotedFrom: "other-study")
        try plantConfirmableAgent(named: "screen-agent", promotedFrom: "screen-study")

        let panel = ExperimentPanel()
        panel.notices = PanelNotices(
            fileURL: root.appending(component: "notices.jsonl"))
        panel.refresh()
        panel.createConfirmationDraft(from: "screen-study")

        // The new draft: phase confirm, declared concept study, the screen
        // pool pinned as the held-out reference, its own prompts CLEARED
        // (confirm needs held-out items; readiness will demand them).
        let draft = try ExperimentStore.load(name: "screen-study-confirm")
        #expect(draft.status == .draft)
        #expect(draft.phase == "confirm")
        #expect(draft.studyType == "conceptStudy")
        #expect(draft.studyKind == .modelOutput)
        #expect(draft.screenTaskPromptsHash == "screen-pool-hash")
        #expect(draft.taskPromptsFile == nil)
        #expect(draft.taskPromptsHash == nil)

        // INHERITED: the scientific pins — model identity, concepts,
        // measurement declarations, generation settings.
        #expect(draft.modelID == "test/model")
        #expect(draft.modelRevision == "abc123")
        #expect(draft.concepts.map(\.name) == ["ghost"])
        #expect(draft.neutralCorpusHash == "neutral-hash")
        #expect(draft.markersHash == "markers-hash")
        #expect(draft.judges?.map(\.name) == ["opus"])
        #expect(draft.judgeRubricFile == "prompts/rubrics/r.md")
        #expect(draft.judgeRubricHash == "rubric-hash")
        #expect(draft.capabilityBatteryFile == "prompts/batteries/b.jsonl")
        #expect(draft.capabilityBatteryHash == "battery-hash")
        #expect(draft.outcomeInstruments == ["answerTokenLogprob"])
        #expect(draft.caseFamily == "sentencing")
        #expect(draft.humanBaseline?.path == "prompts/baselines/h.csv")
        #expect(draft.temperature == 0.7)
        #expect(draft.samplesPerItem == 5)
        #expect(draft.seedPolicy == "derivedSHA256")
        #expect(draft.systemPrompt == "You are a judge.")
        #expect(draft.seeds == screenBefore.seeds)

        // NOT inherited: the screen's EXECUTION state (inherit-by-allowlist
        // — a confirmation preregisters its own).
        #expect(draft.sweep == nil)
        #expect(draft.pipeline == nil)
        #expect(draft.conditions.isEmpty)  // incl. the stamped recommendation
        #expect(draft.variantConditions.isEmpty)
        #expect(draft.promotionRule == nil)
        #expect(draft.perturbationPolicy == nil)
        #expect(draft.readerRefs == nil)
        #expect(draft.acknowledgeUnequalOptionLengths == nil)
        #expect(draft.frozenAt == nil)
        #expect(draft.freezeHash == nil)
        #expect(draft.gitCommit == nil)

        // The screen study is untouched — same manifest, same phase.
        let screenAfter = try ExperimentStore.load(name: "screen-study")
        #expect(screenAfter == screenBefore)
        #expect(screenAfter.phase == "screen")

        // The panel lands on the new draft with the SCREEN study's promoted
        // agent preselected in the confirm controls.
        #expect(panel.selectedName == "screen-study-confirm")
        let preselected = panel.confirmableAgents.first {
            $0.id == panel.confirmAgentID
        }
        #expect(preselected?.artifact.name == "screen-agent")

        // Running the shortcut again suffixes — never overwrites the first
        // confirmation draft.
        panel.createConfirmationDraft(from: "screen-study")
        #expect(panel.selectedName == "screen-study-confirm-2")
        #expect((try? ExperimentStore.load(name: "screen-study-confirm-2")) != nil)
        // And the first draft still exists, untouched.
        #expect((try? ExperimentStore.load(name: "screen-study-confirm")) != nil)
    }

    @Test @MainActor func confirmShortcutRefusesAMissingScreenStudy() throws {
        let root = try makeConfirmShortcutWorkspace()
        defer { cleanupConfirmShortcutWorkspace(root) }

        let panel = ExperimentPanel()
        panel.notices = PanelNotices(
            fileURL: root.appending(component: "notices.jsonl"))
        panel.refresh()
        panel.createConfirmationDraft(from: "no-such-study")
        // Nothing created, nothing selected.
        #expect((try? ExperimentStore.load(name: "no-such-study-confirm")) == nil)
        #expect(panel.selectedName == nil)
        #expect(panel.status?.contains("no-such-study") == true)
    }
}
