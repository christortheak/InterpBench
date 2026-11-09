import CryptoKit
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
            // On a template with no thinking switch no level reaches one. No
            // probe said the seat cannot think, so it is played as thinking
            // (split, and budgeted), and a turn played as thinking never
            // records off: it records `on`, not the agent's declaration.
            let unrendered = try MultiAgentRunner.turnReasoning(
                for: seat, modelID: Self.plain, revision: Self.revision,
                studyEffort: .off, studyBudget: 300)
            #expect(unrendered.effort == .medium && unrendered.budget == 300)
            #expect(unrendered.recorded == .on)
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
                + "--experiment study-v2 --seat a=<agent-path>"))
            #expect(note.contains("(seat ID a)"))
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

    @Test func aFileThatIsNotThePinnedPanelIsNeverJudged() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            var manifest = try panelStudy(root, model: Self.plain, status: .frozen)
            let bytes = try Data(contentsOf: root.appending(path: "prompts/panels/p.json"))
            manifest.multiAgentScenarioHash = SHA256.hash(data: bytes)
                .map { String(format: "%02x", $0) }.joined()
            #expect(!ModelVariantArtifact.manifestArmFindings(manifest, workspaceRoot: root).isEmpty)
            manifest.multiAgentScenarioHash = String(repeating: "0", count: 64)
            #expect(ModelVariantArtifact.manifestArmFindings(manifest, workspaceRoot: root).isEmpty)
            #expect(ModelVariantArtifact.armRefusal(manifest, workspaceRoot: root) == nil)
        }
    }

    // MARK: - Round 2

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A probed record saying the template has no thinking switch, as a
    /// run's `ensure` leaves one.
    private func writeNoSwitchProbe(_ root: URL, model: String, revision: String) throws {
        var record = ModelCapabilities.heuristic(modelID: model, revision: revision)
        record.source = .probe
        record.detected.thinkingSwitch = .unsupported
        try ModelCapabilitiesStore.write(record, root: root)
    }

    @Test func aFrozenPanelStudyKeepsVerifyingAfterItsOwnRunProbesARecord() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "on"])
            var frozen = try panelStudy(root, status: .frozen)
            frozen.reasoningEffort = "off"
            frozen.reasoningMaxTokens = 256
            var draft = frozen
            draft.status = .draft
            #expect(frozen.reasoningProtocolViolations.isEmpty)
            #expect(draft.reasoningProtocolViolations.isEmpty)
            try writeNoSwitchProbe(root, model: Self.qwen, revision: Self.revision)
            // Frozen: read by the rules it froze under, never the new probe.
            #expect(frozen.reasoningProtocolViolations.isEmpty)
            // A draft still reads it.
            #expect(draft.reasoningProtocolViolations == [ReasoningEffort.budgetWithoutEffortReason])
        }
    }

    @Test(arguments: [nil, ""] as [String?])
    func theSeatRepairForAHandBoundPanelPinsACopyInTheStudyEditor(_ semantic: String?) async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            var manifest = try panelStudy(root, model: Self.plain)
            manifest.multiAgentSemanticScenarioPath = semantic
            let findings = ModelVariantArtifact.manifestArmFindings(manifest, workspaceRoot: root)
            #expect(findings.map(\.arm) == ["a"])
            let repair = ModelVariantArtifact.armProblemsRepair(
                manifest, findings: findings, frozen: false)
            #expect(repair.contains("(seat ID a)"))
            #expect(!repair.contains("panel compile"))
            #expect(repair.contains("save a copy of the hand-bound panel prompts/panels/p.json "
                + "in which each of those seats' variantArtifactPath and variantArtifactHash "
                + "name its new agent, and pin the copy in the Mac app's study editor (Setup)"))
            // Carry it out as the study editor does: a corrected agent, a copy
            // of the panel naming it in seat a, and a setup save selecting it.
            let fixedPath = "runs/model-variants/fixed/model-variant.json"
            let fixedURL = root.appending(path: fixedPath)
            try FileManager.default.createDirectory(
                at: fixedURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let fixed = try JSONSerialization.data(withJSONObject: [
                "name": "fixed", "baseModelID": Self.plain, "reasoningEffort": "off",
            ])
            try fixed.write(to: fixedURL)
            var copy = scenario(model: Self.plain)
            copy.agents[0].variantArtifactPath = fixedPath
            copy.agents[0].variantArtifactHash = sha(fixed)
            try JSONEncoder().encode(copy).write(to: root.appending(path: "prompts/panels/p-fixed.json"))
            var study = manifest
            study.status = .draft
            // A hand-bound panel as the study editor pins one (no semantic
            // path at all); the repair above reads an empty one the same way.
            study.multiAgentSemanticScenarioPath = nil
            try ExperimentStore.save(study, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            var fields = StudyProtocolFields(manifest: reviewed.manifest)
            fields.studyKind = .multiAgent
            let result = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields,
                scenario: StudyProtocolScenario(path: "prompts/panels/p-fixed.json", workspaceRoot: root))
            guard case .saved(let saved, _, _) = result else {
                Issue.record("expected a save"); return
            }
            #expect(saved.manifest.multiAgentScenarioPath == "prompts/panels/p-fixed.json")
            #expect(ModelVariantArtifact.manifestArmFindings(saved.manifest, workspaceRoot: root).isEmpty)
        }
    }

    @Test func aSeatAgentFileThatIsNotTheOneItsSeatPinsIsNeverJudged() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            var pinned = scenario(model: Self.plain)
            pinned.agents[0].variantArtifactHash = sha(
                try Data(contentsOf: root.appending(path: Self.agentPath)))
            #expect(!ModelVariantArtifact.scenarioSeatFindings(pinned, workspaceRoot: root).isEmpty)
            // Another file at the same path: not the agent the seat pins.
            try writeAgent(root, ["reasoningEffort": "high"], model: Self.plain)
            #expect(ModelVariantArtifact.scenarioSeatFindings(pinned, workspaceRoot: root).isEmpty)
        }
    }

    /// A draft panel study with a semantic panel of one seat, and the study
    /// editor's fields for it: effort off, the thinking agent cast in the seat.
    private func editorDraft(
        _ root: URL
    ) throws -> (DraftAuthoringSnapshot, StudyProtocolFields, StudyProtocolScenario) {
        try writeAgent(root, ["reasoningEffort": "on"])
        let semantic = MultiAgentScenario(
            name: "dialogue", baseModelID: "",
            agents: [.init(id: "speaker", name: "Speaker", baseModelID: "", systemPrompt: "Discuss.")],
            turns: [.init(id: "turn", title: "Response", speakerAgentID: "speaker",
                          promptTemplate: "Respond.", outputLabel: "response")])
        let semanticURL = root.appending(path: "prompts/panels/dialogue.json")
        try FileManager.default.createDirectory(
            at: semanticURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(semantic).write(to: semanticURL)
        let selection = try StudyProtocolScenario(
            path: "prompts/panels/dialogue.json", workspaceRoot: root)
        var created = ExperimentManifest(name: "study", description: "", modelID: Self.qwen)
        created.modelRevision = Self.revision
        try ExperimentStore.save(created, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
        var fields = StudyProtocolFields(manifest: reviewed.manifest)
        fields.studyKind = .multiAgent
        fields.reasoningEffort = "off"
        fields.seatCastingEdits = ["speaker": try SeatCasting.occupant(artifactPath: Self.agentPath)]
        return (reviewed, fields, selection)
    }

    /// One save that casts a thinking agent and declares its budget, with the
    /// study's effort off, is judged by the casting it saves: accepted.
    @Test func castingAThinkingAgentAndItsBudgetInOneSaveIsAccepted() async throws {
        try await withWorkspace { root in
            var (reviewed, fields, selection) = try editorDraft(root)
            fields.reasoningMaxTokens = 256
            guard case .saved(let cast, _, _) = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields, scenario: selection)
            else { Issue.record("expected a save"); return }
            #expect(cast.manifest.reasoningMaxTokens == 256)
        }
    }

    /// A save that recasts the only thinking seat to baseline while keeping
    /// the budget is refused, so no draft is saved that freeze would refuse.
    @Test func recastingTheOnlyThinkingSeatToBaselineRefusesTheBudget() async throws {
        try await withWorkspace { root in
            var (reviewed, fields, selection) = try editorDraft(root)
            // Cast first, then declare the budget, so the draft pins a
            // thinking seat whatever the rule reads.
            guard case .saved(let cast, _, _) = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields, scenario: selection)
            else { Issue.record("expected a save"); return }
            fields.reasoningMaxTokens = 256
            guard case .saved(let budgeted, _, _) = try StudyProtocolAuthoring.save(
                reviewed: cast, fields: fields, scenario: selection)
            else { Issue.record("expected a save"); return }
            #expect(budgeted.manifest.reasoningMaxTokens == 256)
            fields.seatCastingEdits = ["speaker": .baseline]
            do {
                _ = try StudyProtocolAuthoring.save(
                    reviewed: budgeted, fields: fields, scenario: selection)
                Issue.record("expected the budget to be refused")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(ReasoningEffort.budgetWithoutEffortReason))
            }
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data
                == budgeted.file.data)
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
