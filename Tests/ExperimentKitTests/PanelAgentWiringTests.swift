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

    /// The make-again lead every seat repair opens with on this command line.
    static let macMake = "make a new agent for each seat whose agent breaks a rule (seat ID a): "
        + "steerlab-cli agent create <new-name> --model <model> --revision <sha> … "
        + "(or the app's agent editor), then "
    static let macSeats = "in the Mac app's study editor (Seats), put each new agent in its "
        + "seat and save, which compiles the panel again"

    /// A corrected agent for seat a, in the library: its path and occupant.
    private func fixedAgent(_ root: URL) throws -> (path: String, data: Data) {
        let fixedPath = "runs/model-variants/fixed/model-variant.json"
        let fixedURL = root.appending(path: fixedPath)
        try FileManager.default.createDirectory(
            at: fixedURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fixed = try JSONSerialization.data(withJSONObject: [
            "name": "fixed", "baseModelID": Self.plain, "reasoningEffort": "off",
        ])
        try fixed.write(to: fixedURL)
        return (fixedPath, fixed)
    }

    /// A saved draft panel study whose seat a's agent breaks a rule, with
    /// the given semantic path and pin, exactly as the editor would read it.
    private func savedSeatDraft(
        _ root: URL, semantic: String?, pin: String
    ) throws -> DraftAuthoringSnapshot {
        try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
        var manifest = try panelStudy(root, model: Self.plain)
        if pin != "prompts/panels/p.json" {
            let url = root.appending(path: pin)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(contentsOf: root.appending(path: "prompts/panels/p.json")).write(to: url)
            manifest.multiAgentScenarioPath = pin
        }
        manifest.multiAgentSemanticScenarioPath = semantic
        try ExperimentStore.save(manifest, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        return try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
    }

    @Test func aHandBoundPanelIsRepairedByPinningACopyInSetup() async throws {
        try await withWorkspace { root in
            let reviewed = try savedSeatDraft(root, semantic: nil, pin: "prompts/panels/p.json")
            let findings = ModelVariantArtifact.manifestArmFindings(
                reviewed.manifest, workspaceRoot: root)
            #expect(findings.map(\.arm) == ["a"])
            #expect(ModelVariantArtifact.armProblemsRepair(
                reviewed.manifest, findings: findings, frozen: false)
                == Self.macMake + "save a copy of the panel prompts/panels/p.json in which "
                + "each of those seats' variantArtifactPath and variantArtifactHash name its "
                + "new agent, and pin the copy: in the Mac app's study editor (Setup), which "
                + "pins a hand-bound panel as it stands")
            // Carry it out as the study editor does: a copy of the panel
            // naming the corrected agent in seat a, selected in Setup.
            let fixed = try fixedAgent(root)
            var copy = scenario(model: Self.plain)
            copy.agents[0].variantArtifactPath = fixed.path
            copy.agents[0].variantArtifactHash = sha(fixed.data)
            try JSONEncoder().encode(copy).write(to: root.appending(path: "prompts/panels/p-fixed.json"))
            var fields = StudyProtocolFields(manifest: reviewed.manifest)
            fields.studyKind = .multiAgent
            guard case .saved(let saved, _, _) = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields,
                scenario: StudyProtocolScenario(path: "prompts/panels/p-fixed.json", workspaceRoot: root))
            else { Issue.record("expected a save"); return }
            #expect(saved.manifest.multiAgentScenarioPath == "prompts/panels/p-fixed.json")
            #expect(ModelVariantArtifact.manifestArmFindings(saved.manifest, workspaceRoot: root).isEmpty)
        }
    }

    /// An empty semantic path, or a pin in prompts/panels/compiled/: the
    /// editor reads a casting it recompiles (and ignores a new selection),
    /// so the repair is its Seats section, and doing that clears it.
    @Test(arguments: [
        ("", "prompts/panels/p.json"), (nil, "prompts/panels/compiled/p.json"),
    ] as [(String?, String)])
    func aCastingTheEditorRecompilesIsRepairedInItsSeats(
        _ semantic: String?, _ pin: String
    ) async throws {
        try await withWorkspace { root in
            let reviewed = try savedSeatDraft(root, semantic: semantic, pin: pin)
            let findings = ModelVariantArtifact.manifestArmFindings(
                reviewed.manifest, workspaceRoot: root)
            #expect(findings.map(\.arm) == ["a"])
            #expect(ModelVariantArtifact.armProblemsRepair(
                reviewed.manifest, findings: findings, frozen: false)
                == Self.macMake + Self.macSeats)
            let fixed = try fixedAgent(root)
            var fields = StudyProtocolFields(manifest: reviewed.manifest)
            fields.studyKind = .multiAgent
            fields.seatCastingEdits = ["a": try SeatCasting.occupant(artifactPath: fixed.path)]
            guard case .saved(let saved, let compiled, _) = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: fields)
            else { Issue.record("expected a save"); return }
            #expect(compiled)
            #expect(ModelVariantArtifact.manifestArmFindings(saved.manifest, workspaceRoot: root).isEmpty)
        }
    }

    @Test func printedSeatRepairsQuoteTheirValues() async throws {
        try await withWorkspace { root in
            var manifest = try panelStudy(root, model: Self.plain)
            manifest.multiAgentSemanticScenarioPath = "prompts/panels/my panel.json"
            // The same strings test_panel_agent_wiring.py asserts for the server.
            #expect(ModelVariantArtifact.seatRepair(
                manifest, seatIDs: ["judge one", "it's"], target: "study")
                == "make a new agent for each seat whose agent breaks a rule (seat IDs "
                + "judge one, it's): steerlab-cli agent create <new-name> --model <model> "
                + "--revision <sha> … (or the app's agent editor), then compile the panel "
                + "again, casting every seat, with each new agent in place of the one its "
                + "seat holds: steerlab-cli panel compile 'prompts/panels/my panel.json' "
                + "--experiment study --seat 'judge one'=<agent-path> --seat "
                + "'it'\"'\"'s'=<agent-path> --seat <each other seat>=<its agent-path>")
            #expect(ModelVariantArtifact.seatRepair(manifest, seatIDs: ["a=b"], target: "study")
                == "make a new agent for each seat whose agent breaks a rule (seat ID "
                + "a=b): steerlab-cli agent create <new-name> --model <model> --revision "
                + "<sha> … (or the app's agent editor), then compile the panel again, "
                + "casting every seat, with each new agent in place of the one its seat "
                + "holds: steerlab-cli panel compile 'prompts/panels/my panel.json' "
                + "--experiment study --casting <file> --file-sha256 <digest> "
                + "--manifest-sha256 <digest> (a casting file names each seat by its ID; "
                + "--seat cannot name one that contains '='), or in the Mac app's study "
                + "editor (Seats), put each new agent in its seat and save, which compiles "
                + "the panel again")
        }
    }

    @Test func aSeatAgentEditedInPlaceIsJudgedAsItIsOnDisk() async throws {
        try await withWorkspace { root in
            // The run loads the file as it is now (drift is a warning there),
            // so the seat checks judge that file, whatever hash the seat pins.
            try writeAgent(root, ["reasoningEffort": "off"], model: Self.plain)
            var pinned = scenario(model: Self.plain)
            pinned.agents[0].variantArtifactHash = sha(
                try Data(contentsOf: root.appending(path: Self.agentPath)))
            #expect(ModelVariantArtifact.scenarioSeatFindings(pinned, workspaceRoot: root).isEmpty)
            try writeAgent(root, ["reasoningEffort": "medium"], model: Self.plain)
            #expect(ModelVariantArtifact.scenarioSeatFindings(pinned, workspaceRoot: root)
                .map(\.arm) == ["a"])
        }
    }

    @Test func aFrozenStudyKeepsItsBudgetAfterASeatAgentIsEdited() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "on"])
            var pinned = scenario()
            pinned.agents[0].variantArtifactHash = sha(
                try Data(contentsOf: root.appending(path: Self.agentPath)))
            var frozen = try panelStudy(root, status: .frozen)
            try JSONEncoder().encode(pinned).write(to: root.appending(path: "prompts/panels/p.json"))
            frozen.reasoningEffort = "off"
            frozen.reasoningMaxTokens = 256
            var draft = frozen
            draft.status = .draft
            #expect(frozen.reasoningProtocolViolations.isEmpty)
            // The agent editor's Save Changes, in place, to off.
            try writeAgent(root, ["reasoningEffort": "off"])
            #expect(frozen.reasoningProtocolViolations.isEmpty)
            // A draft reads the file as it is now: nothing thinks any more.
            #expect(draft.reasoningProtocolViolations == [ReasoningEffort.budgetWithoutEffortReason])
        }
    }

    /// A casting that cannot compile is refused with the compile's own
    /// reason, not as a budget problem.
    @Test func aCastingThatCannotCompileRefusesWithItsOwnReason() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "on"])
            let broken = MultiAgentScenario(
                name: "dialogue", baseModelID: "",
                agents: [.init(id: "speaker", name: "Speaker", baseModelID: "", systemPrompt: "Discuss.")],
                turns: [
                    .init(id: "turn", title: "One", speakerAgentID: "speaker",
                          promptTemplate: "Respond.", outputLabel: "one"),
                    .init(id: "turn", title: "Two", speakerAgentID: "speaker",
                          promptTemplate: "Again.", outputLabel: "two"),
                ])
            let url = root.appending(path: "prompts/panels/dialogue.json")
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(broken).write(to: url)
            var created = ExperimentManifest(name: "study", description: "", modelID: Self.qwen)
            created.modelRevision = Self.revision
            try ExperimentStore.save(created, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            var fields = StudyProtocolFields(manifest: reviewed.manifest)
            fields.studyKind = .multiAgent
            fields.reasoningEffort = "off"
            fields.reasoningMaxTokens = 256
            fields.seatCastingEdits = ["speaker": try SeatCasting.occupant(artifactPath: Self.agentPath)]
            do {
                _ = try StudyProtocolAuthoring.save(
                    reviewed: reviewed, fields: fields,
                    scenario: StudyProtocolScenario(path: "prompts/panels/dialogue.json", workspaceRoot: root))
                Issue.record("expected the casting to be refused")
            } catch let error as ExperimentError {
                #expect(!error.reason.contains(ReasoningEffort.budgetWithoutEffortReason))
                #expect(error.lifecycleRefusal?.gate == .missingPrerequisite, "\(error.reason)")
            }
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

    // MARK: - Round 4

    /// Frozen: seat a has no pin and its agent file is gone; seat b's pinned
    /// agent was saved in place to off. Each seat is read on its own, and an
    /// unreadable file counts as thinking, so the budget stays allowed.
    @Test func anUnreadableUnpinnedSeatNeverHidesTheSeatsAfterIt() async throws {
        try await withWorkspace { root in
            var paths: [String: URL] = [:]
            for seat in ["a", "b"] {
                let url = root.appending(path: "runs/model-variants/\(seat)/model-variant.json")
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try JSONSerialization.data(withJSONObject: [
                    "name": seat, "baseModelID": Self.qwen, "reasoningEffort": "on",
                ]).write(to: url)
                paths[seat] = url
            }
            let pinB = sha(try Data(contentsOf: try #require(paths["b"])))
            let scenario = MultiAgentScenario(
                name: "p", baseModelID: Self.qwen,
                agents: [
                    .init(id: "a", name: "Ava", baseModelID: Self.qwen,
                          variantArtifactPath: "runs/model-variants/a/model-variant.json"),
                    .init(id: "b", name: "Ben", baseModelID: Self.qwen,
                          variantArtifactPath: "runs/model-variants/b/model-variant.json",
                          variantArtifactHash: pinB),
                ],
                turns: [
                    .init(id: "t1", title: "x", speakerAgentID: "a", promptTemplate: "Speak.",
                          outputLabel: "o1"),
                    .init(id: "t2", title: "y", speakerAgentID: "b", promptTemplate: "Speak.",
                          outputLabel: "o2"),
                ])
            try FileManager.default.removeItem(at: try #require(paths["a"]))
            try JSONSerialization.data(withJSONObject: [
                "name": "b", "baseModelID": Self.qwen, "reasoningEffort": "off",
            ]).write(to: try #require(paths["b"]))
            #expect(MultiAgentRunner.seatThinks(
                in: scenario, studyRevision: Self.revision, frozen: true))
        }
    }

    /// The other side of the frozen rule: every seat's agent is readable, is
    /// the one it pins, and is off, so nothing thinks and the budget beside
    /// an off effort is refused. A seat with an empty path is a bare seat.
    @Test func aFrozenPanelWhoseReadableSeatsAreAllOffRefusesItsBudget() async throws {
        try await withWorkspace { root in
            try writeAgent(root, ["reasoningEffort": "off"])
            var pinned = scenario()
            pinned.agents[0].variantArtifactHash = sha(
                try Data(contentsOf: root.appending(path: Self.agentPath)))
            pinned.agents[1].variantArtifactPath = ""
            #expect(!MultiAgentRunner.seatThinks(
                in: pinned, studyRevision: Self.revision, frozen: true))
            var frozen = try panelStudy(root, status: .frozen)
            try JSONEncoder().encode(pinned).write(to: root.appending(path: "prompts/panels/p.json"))
            frozen.reasoningEffort = "off"
            frozen.reasoningMaxTokens = 256
            #expect(frozen.reasoningProtocolViolations == [ReasoningEffort.budgetWithoutEffortReason])
        }
    }

    /// The repair reads the editor's rule against the workspace it is given,
    /// not whatever the process points at.
    @Test func theSeatRepairReadsTheGivenWorkspace() async throws {
        try await withWorkspace { _ in
            let other = FileManager.default.temporaryDirectory
                .resolvingSymlinksInPath()
                .appending(component: "fr6-other-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: other) }
            var manifest = ExperimentManifest(name: "study", description: "", modelID: Self.plain)
            manifest.studyKind = .multiAgent
            manifest.multiAgentScenarioPath = other.appending(
                path: "prompts/panels/compiled/p.json").path
            let findings = [ModelVariantArtifact.ArmFinding(arm: "a", sentence: "x", fix: .seat)]
            #expect(ModelVariantArtifact.armProblemsRepair(
                manifest, findings: findings, frozen: false, workspaceRoot: other)
                == Self.macMake + Self.macSeats)
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
