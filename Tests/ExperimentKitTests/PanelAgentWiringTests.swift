import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Panels and agents, wired together (FR6 integration: public issues #15,
/// #16, and #17), on the Mac engine:
///
/// * a panel seat cast with an agent thinks at the AGENT's level, and the
///   turn records the level the template renders;
/// * one budget predicate: an off study may declare `reasoningMaxTokens`
///   when an agent arm thinks OR a cast panel seat thinks;
/// * the agents a panel's seats are cast with meet the agent-file rules
///   (less the answer-token and revision rules) at `panel check`, verify,
///   submission, and run start; a seat whose agent file has a problem
///   refuses when it plays.
///
/// Twin of `Server/tests/test_panel_agent_wiring.py`.
@Suite(.serialized) struct PanelAgentWiringTests {
    static let qwen = "test/qwen-panel"
    static let plain = "org/tiny"
    static let revision = String(repeating: "a", count: 40)
    static let agentPath = "runs/model-variants/seat-agent/model-variant.json"

    private func withWorkspace<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .resolvingSymlinksInPath()
            .appending(component: "fr6-wiring-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    /// Writes the seat's agent file, by hand (so it may break a rule).
    private func writeAgent(_ root: URL, _ fields: [String: Any], model: String = qwen) throws {
        var document: [String: Any] = ["name": "seat-agent", "baseModelID": model]
        for (key, value) in fields { document[key] = value }
        let url = root.appending(path: Self.agentPath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: document).write(to: url)
    }

    private func scenario(model: String = qwen, cast: Bool = true) -> MultiAgentScenario {
        MultiAgentScenario(
            name: "p", baseModelID: model,
            agents: [
                .init(id: "a", name: "Ava", baseModelID: model,
                      variantArtifactPath: cast ? Self.agentPath : nil),
                .init(id: "b", name: "Ben", baseModelID: model),
            ],
            turns: [
                .init(id: "t1", title: "Ava speaks", speakerAgentID: "a",
                      promptTemplate: "Speak.", outputLabel: "ava1"),
                .init(id: "t2", title: "Ben replies", speakerAgentID: "b",
                      promptTemplate: "Ava said {{outputs.ava1}}", outputLabel: "ben1"),
            ])
    }

    /// A panel study whose seat Ava is cast with the agent file in `root`.
    private func panelStudy(
        _ root: URL, model: String = qwen, status: ExperimentManifest.Status = .draft,
        cast: Bool = true
    ) throws -> ExperimentManifest {
        let url = root.appending(path: "prompts/panels/p.json")
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(scenario(model: model, cast: cast)).write(to: url)
        var manifest = ExperimentManifest(name: "study", description: "", modelID: model)
        manifest.studyKind = .multiAgent
        manifest.multiAgentScenarioPath = "prompts/panels/p.json"
        manifest.multiAgentSemanticScenarioPath = "prompts/panels/semantic/p.json"
        manifest.modelRevision = Self.revision
        manifest.status = status
        return manifest
    }

    // MARK: - 1. The seat's level

    @Test func aSeatAgentThinksAtItsOwnLevelAndRecordsWhatRendered() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"])
            let seat = scenario().agents[0]
            let alone = try MultiAgentRunner.seatReasoning(
                for: seat, studyEffort: .off, studyBudget: nil)
            #expect(alone.effort == .medium && alone.budget == nil)
            let budgeted = try MultiAgentRunner.turnReasoning(
                for: seat, modelID: Self.qwen, revision: Self.revision,
                studyEffort: .off, studyBudget: 300)
            #expect(budgeted.effort == .medium && budgeted.budget == 300)
            #expect(budgeted.recorded == .medium)
            // On a template with no thinking switch the level renders as off,
            // and the turn records that, not the agent's declaration.
            let unrendered = try MultiAgentRunner.turnReasoning(
                for: seat, modelID: Self.plain, revision: Self.revision,
                studyEffort: .off, studyBudget: 300)
            #expect(unrendered.effort == .medium && unrendered.recorded == .off)
            // The bare seat follows the study, and gets no budget when off.
            let bare = try MultiAgentRunner.turnReasoning(
                for: scenario().agents[1], modelID: Self.qwen, revision: Self.revision,
                studyEffort: .off, studyBudget: 300)
            #expect(bare.effort == .off && bare.budget == nil && bare.recorded == .off)
        }
    }

    @Test func aSeatAgentFileWithAProblemRefusesWhenTheSeatPlays() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium", "qwenThinkingEnabled": false])
            let seat = scenario().agents[0]
            for attempt in [
                { _ = try MultiAgentRunner.seatReasoning(for: seat, studyEffort: .off, studyBudget: nil) },
                { _ = try MultiAgentRunner.runtimeSettings(for: seat) },
            ] as [() throws -> Void] {
                do {
                    try attempt()
                    Issue.record("expected a refusal")
                } catch let error as ExperimentError {
                    #expect(error.reason.hasPrefix("seat 'Ava': This agent file says"))
                    #expect(error.lifecycleRefusal?.gate == .studyDeclaration)
                }
            }
        }
    }

    // MARK: - 2. One budget predicate

    @Test func anOffPanelStudyMayDeclareABudgetOnlyWhenACastSeatThinks() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"])
            var manifest = try panelStudy(root)
            manifest.reasoningEffort = "off"
            #expect(ModelVariantArtifact.budgetPermittedWithoutEffort(manifest))
            try ManifestDraftEdits.setSamplingProtocol(
                reasoningMaxTokens: 128, experimentName: "study", manifest: &manifest)
            #expect(manifest.reasoningMaxTokens == 128)
            #expect(manifest.reasoningProtocolViolations.isEmpty)

            try writeAgent(root, ["reasoningEffort": "off"])
            #expect(!ModelVariantArtifact.budgetPermittedWithoutEffort(manifest))
            #expect(manifest.reasoningProtocolViolations
                == [ReasoningEffort.budgetWithoutEffortReason])
            var fresh = try panelStudy(root)
            fresh.reasoningEffort = "off"
            #expect(throws: ExperimentError.self) {
                try ManifestDraftEdits.setSamplingProtocol(
                    reasoningMaxTokens: 128, experimentName: "study", manifest: &fresh)
            }
        }
    }

    // MARK: - 3. The agent rules for the agents seats are cast with

    @Test func aDraftPanelStudyNamesEachSeatAgentRule() async throws {
        try await withWorkspace { root in
            for (fields, model, phrase) in [
                (["reasoningEffort": "medium"] as [String: Any], Self.plain,
                 "has no thinking switch"),
                (["reasoningEffort": "medium", "qwenThinkingEnabled": false], Self.qwen,
                 "one says the agent thinks"),
                (["adapters": [["adapterDirectory": "a"], ["adapterDirectory": "b"]]],
                 Self.qwen, "lists 2 adapters"),
            ] {
                try writeAgent(root, fields, model: model)
                let manifest = try panelStudy(root, model: model)
                let problems = ModelVariantArtifact.manifestArmProblems(
                    manifest, workspaceRoot: root)
                #expect(problems.contains { $0.hasPrefix("seat 'Ava': ") && $0.contains(phrase) },
                        "\(phrase): \(problems)")
            }
        }
    }

    @Test func aSeatAgentIsNotHeldToTheRevisionOrAnswerTokenRules() async throws {
        try await withWorkspace { root in
            try writeAgent(root, [
                "baseRevision": String(repeating: "b", count: 40), "qwenThinkingEnabled": true,
            ])
            var manifest = try panelStudy(root)
            manifest.outcomeInstruments = ["answerTokenLogprob"]
            #expect(ModelVariantArtifact.manifestArmFindings(manifest, workspaceRoot: root).isEmpty)
        }
    }

    @Test func aFrozenPanelStudyIsToldAtVerifyAndRefusedAtSubmissionAndRunStart() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            let manifest = try panelStudy(root, model: Self.plain, status: .frozen)
            let note = try #require(ModelVariantArtifact.frozenArmProblemsAdvisory(
                manifest, workspaceRoot: root))
            #expect(note.hasPrefix("this study will be refused when its run is submitted or "
                + "started, because seat 'Ava': "))
            #expect(note.contains("steerlab-cli experiment duplicate study study-v2"))
            #expect(note.contains("steerlab-cli panel compile prompts/panels/semantic/p.json "
                + "--experiment study-v2 --seat <seat>=<agent-path>"))
            // Submission (every client site asks this) and run start
            // (`ExperimentTasks.runImpl` asks `armRefusal`).
            let refusal = try #require(SubstrateRouting.agentArmSubmissionRefusal(
                manifest: manifest, verb: "run", workspaceRoot: root))
            #expect(refusal.reason.contains("has no thinking switch"))
            #expect(ModelVariantArtifact.armRefusal(manifest, workspaceRoot: root) != nil)
            #expect(SubstrateRouting.agentArmSubmissionRefusal(
                manifest: manifest, verb: "analyze", workspaceRoot: root) == nil)

            try writeAgent(root, ["reasoningEffort": "off"], model: Self.plain)
            #expect(ModelVariantArtifact.armRefusal(manifest, workspaceRoot: root) == nil)
        }
    }

    @Test func panelCheckRefusesASeatWhoseAgentBreaksARule() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            let url = root.appending(path: "proposed.json")
            try JSONEncoder().encode(scenario(model: Self.plain)).write(to: url)
            let refused = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "panel", ["check", url.path, "--json"])
            #expect(refused.envelope.exitCode == 65)
            #expect(refused.envelope.error?.reason.hasPrefix("seat 'Ava': ") == true)
            #expect(refused.envelope.error?.repairAction == ModelVariantArtifact.seatAgentRepair)

            try writeAgent(root, ["reasoningEffort": "off"], model: Self.plain)
            let passed = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "panel", ["check", url.path, "--json"])
            #expect(passed.envelope.exitCode == 0, "\(passed.envelope)")
        }
    }
}
