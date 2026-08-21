import Foundation
import Testing

@testable import ExperimentKit

/// The AUTHORING half of panel turn contracts: the semantic-panel machinery,
/// the protocol-template library and the editor model must carry a contract
/// turn and a seat's `role` through without touching them.
///
/// Why this is worth a suite of its own: every path here is a COPY of a panel
/// — strip the bindings, cast the seats, mint a protocol, instantiate it — and
/// a copy that silently drops a key produces a file that still decodes, still
/// validates and still runs, rendering different prose than the one the
/// researcher authored. There is no error to notice. The byte-level assertions
/// below (encode the contract sub-object, compare) are deliberately stricter
/// than `==` on the struct: they fail if a field is preserved in memory but
/// re-encoded differently.
@Suite(.serialized) struct PanelContractAuthoringTests {

    // MARK: - Fixtures

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "panel-contract-authoring-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    /// A semantic panel whose second turn is a CONTRACT turn, and whose first
    /// seat declares a role. Three seats, so the own-voice block and the roll
    /// call both render.
    private func contractPanel() -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: "contract-panel",
                description: "A panel with one contract turn.",
                baseModelID: "",
                sharedMaterials: "MATERIAL TEXT",
                agents: [
                    .init(
                        id: "ava", name: "Ava", baseModelID: "",
                        systemPrompt: "You are Ava.", role: "a reviewer"),
                    .init(id: "ben", name: "Ben", baseModelID: "", systemPrompt: "You are Ben."),
                    .init(id: "cal", name: "Cal", baseModelID: "", systemPrompt: "You are Cal."),
                ],
                turns: [
                    .init(
                        id: "t1a", title: "First draft — Ava", speakerAgentID: "ava",
                        promptTemplate: "Draft it.", outputLabel: "t1_ava"),
                    .init(
                        id: "t1b", title: "First draft — Ben", speakerAgentID: "ben",
                        promptTemplate: "Draft it.", outputLabel: "t1_ben"),
                    .init(
                        id: "t2", title: "Response — Ava", speakerAgentID: "ava",
                        promptTemplate: "", outputLabel: "t2_ava",
                        includeScenarioMaterials: true, includeSpeakerContext: false,
                        maxTokens: 400,
                        contract: TurnContract(
                            stage: "The group has exchanged first drafts.",
                            task: "Write a response memo to your colleagues.",
                            format: "Verdict: yes OR no",
                            inputs: ["t1_ava", "t1_ben"],
                            ownVoice: true,
                            materialsTitle: "THE RECORD ON APPEAL")),
                ]))
    }

    /// The contract sub-objects of a scenario, as canonical JSON. Stricter
    /// than struct equality: a field kept in memory but re-encoded differently
    /// fails here.
    private func contractBytes(_ scenario: MultiAgentScenario) throws -> [String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try scenario.turns.compactMap { turn in
            try turn.contract.map { String(decoding: try encoder.encode($0), as: UTF8.self) }
        }
    }

    private func render(
        _ scenario: MultiAgentScenario, turnID: String
    ) throws -> String {
        let turn = try #require(scenario.turns.first { $0.id == turnID })
        let speaker = try #require(scenario.agents.first { $0.id == turn.speakerAgentID })
        return MultiAgentRunner.renderPrompt(
            scenario: scenario, turn: turn, speakerName: speaker.name,
            speakerContext: "",
            outputsByLabel: ["t1_ava": "AVA DRAFT.", "t1_ben": "BEN DRAFT."])
    }

    // MARK: - The semantic form

    @Test func theSemanticFormKeepsEveryContractAndEveryRole() throws {
        // The bound file a legacy panel or a compiled casting looks like.
        var bound = contractPanel()
        bound.baseModelID = "test/model"
        bound.temperature = 0.7
        bound.maxTokens = 999
        for index in bound.agents.indices {
            bound.agents[index].baseModelID = "test/model"
            bound.agents[index].variantArtifactPath = "runs/model-variants/x.json"
            bound.agents[index].variantArtifactHash = "deadbeef"
        }

        let semantic = PanelComposition.semanticForm(bound)

        // Bindings gone…
        #expect(semantic.baseModelID.isEmpty)
        #expect(semantic.agents.allSatisfy { $0.baseModelID.isEmpty })
        #expect(semantic.agents.allSatisfy { $0.variantArtifactPath == nil })
        // …and the environment untouched, contracts included.
        #expect(semantic.turns == bound.turns)
        #expect(try contractBytes(semantic) == (try contractBytes(bound)))
        #expect(semantic.agents.map(\.role) == bound.agents.map(\.role))
        #expect(semantic.agents.first?.role == "a reviewer")
        // The file still declares schema 2: stripping bindings does not
        // un-declare the contract turn.
        #expect(semantic.requiredSchemaVersion == 2)
    }

    @Test func aContractPanelIsNotMistakenForABoundOne() throws {
        // `carriesBindings` is defined as "differs from its own semantic
        // form". If `semanticForm` ever dropped `role` or `contract`, every
        // contract panel would open READ-ONLY behind the migration banner —
        // the loudest possible symptom, and the reason it is asserted here.
        #expect(!PanelAuthoring.carriesBindings(contractPanel()))
        #expect(PanelAuthoring.mode(for: contractPanel()) == .semantic)
    }

    @Test func aSaveNormalisesWithoutTouchingContractsOrRoles() throws {
        let panel = contractPanel()
        let saved = PanelAuthoring.scenarioForSave(
            name: panel.name, description: panel.description,
            sharedMaterials: panel.sharedMaterials, agents: panel.agents, turns: panel.turns)
        #expect(saved.turns == panel.turns)
        #expect(try contractBytes(saved) == (try contractBytes(panel)))
        #expect(saved.agents.map(\.role) == panel.agents.map(\.role))
    }

    // MARK: - Compile

    @Test func compilingACastingCarriesTheContractsThrough() throws {
        let semantic = contractPanel()
        let seats = PanelComposition.seatIDs(semantic)
        let assignment = SeatAssignment(
            seatIDs: seats,
            ordered: [
                .agent(name: "sympathy", artifactPath: "runs/model-variants/s.json", artifactHash: "abc"),
                .baseline,
                .baseline,
            ])
        let bound = try PanelComposition.compile(
            semantic: semantic, assignment: assignment, modelID: "test/model",
            temperature: 0, maxTokens: 512)

        #expect(bound.turns == semantic.turns)
        #expect(try contractBytes(bound) == (try contractBytes(semantic)))
        #expect(bound.agents.map(\.role) == semantic.agents.map(\.role))
        #expect(bound.agents[0].variantArtifactPath == "runs/model-variants/s.json")
    }

    @Test func compilingStampsTheVersionTheTurnScriptRequires() throws {
        // `compileAndPin` writes its file with a bare encoder rather than
        // through the store, so a stale in-memory version would reach a PINNED
        // file. A compiled panel that says schema 1 while carrying contract
        // turns is a file that lies about its own contents to whichever
        // decoder reads it next.
        var stale = contractPanel()
        stale.schemaVersion = 1
        let seats = PanelComposition.seatIDs(stale)
        let bound = try PanelComposition.compile(
            semantic: stale,
            assignment: SeatAssignment(
                seatIDs: seats, ordered: Array(repeating: .baseline, count: seats.count)),
            modelID: "test/model", temperature: 0, maxTokens: 512)
        #expect(bound.schemaVersion == 2)
    }

    @Test func theRenderedPromptIsIdenticalBeforeAndAfterCompiling() throws {
        // The property that actually matters: casting changes who speaks, not
        // what the prompt says. Both sides go through the runner's real
        // renderer, so a lossy compile shows up as different BYTES here rather
        // than as a subtly different transcript in three weeks.
        let semantic = contractPanel()
        let seats = PanelComposition.seatIDs(semantic)
        let bound = try PanelComposition.compile(
            semantic: semantic,
            assignment: SeatAssignment(
                seatIDs: seats, ordered: Array(repeating: .baseline, count: seats.count)),
            modelID: "test/model", temperature: 0, maxTokens: 512)

        let boundPrompt = try render(bound, turnID: "t2")
        // The semantic panel cannot be VALIDATED (no model, on purpose) but it
        // renders, and the renderer is pure over the same fields.
        #expect(try render(semantic, turnID: "t2") == boundPrompt)
        // Sanity: it really is the contract renderer, with the role in it.
        #expect(boundPrompt.hasPrefix("You are Ava, a reviewer."))
        #expect(boundPrompt.contains("===== THE RECORD ON APPEAL ====="))
        #expect(boundPrompt.contains("===== YOUR OWN EARLIER OUTPUT — First draft — Ava ====="))
        #expect(boundPrompt.contains("or reply on behalf of Ben or Cal."))
        #expect(MultiAgentRunner.promptRenderer(for: bound.turns[2]) == "contract-v1")
    }

    @Test func everyCastingOfACompositionSweepKeepsTheSameContracts() throws {
        let semantic = contractPanel()
        let seats = PanelComposition.seatIDs(semantic)
        let sweep = PanelComposition.compositionSweep(
            seatIDs: seats,
            agent: .agent(
                name: "sympathy", artifactPath: "runs/model-variants/s.json",
                artifactHash: "abc"))
        // all-baseline + one per seat + all-treated
        #expect(sweep.count == seats.count + 2)

        let expected = try contractBytes(semantic)
        let expectedPrompt = try render(semantic, turnID: "t2")
        for assignment in sweep {
            let bound = try PanelComposition.compile(
                semantic: semantic, assignment: assignment, modelID: "test/model",
                temperature: 0, maxTokens: 512)
            try MultiAgentRunner.validate(bound)
            #expect(try contractBytes(bound) == expected)
            #expect(try render(bound, turnID: "t2") == expectedPrompt)
        }
    }

    // MARK: - The panel file

    @Test func aContractPanelSurvivesTheRoundTripThroughDisk() throws {
        try withTempWorkspace { _ in
            let record = try MultiAgentScenarioStore.save(contractPanel())
            let data = try Data(contentsOf: record.url)
            let reloaded = try JSONDecoder().decode(MultiAgentScenario.self, from: data)

            #expect(reloaded.turns == contractPanel().turns)
            #expect(try contractBytes(reloaded) == (try contractBytes(contractPanel())))
            #expect(reloaded.agents.map(\.role) == ["a reviewer", nil, nil])
            #expect(reloaded.schemaVersion == 2)
            // Re-saving the reload is a byte-for-byte no-op: a pinned panel
            // must not drift because it was opened.
            let again = try MultiAgentScenarioStore.update(reloaded, at: record.url)
            #expect(try Data(contentsOf: again.url) == data)
        }
    }

    // MARK: - Protocol templates

    @Test func aProtocolTemplateCarriesContractsAndRolesBothWays() throws {
        try withTempWorkspace { _ in
            let source = contractPanel()
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                source, named: "three-seat-contract-protocol",
                materialsChecklist: ["the record"])
            // Minting drops the case, not the script.
            #expect(template.protocolBody.sharedMaterials.isEmpty)
            #expect(template.protocolBody.turns == source.turns)
            #expect(try contractBytes(template.protocolBody) == (try contractBytes(source)))

            // Through its own file, and back.
            let saved = try ScenarioProtocolTemplateStore.save(template)
            let reloaded = try JSONDecoder().decode(
                ScenarioProtocolTemplate.self, from: try Data(contentsOf: saved.url))
            #expect(reloaded.protocolBody.turns == source.turns)
            #expect(reloaded.seats.map(\.role) == ["a reviewer", nil, nil])

            // Instantiating mints an ordinary panel around a new case.
            let minted = try ScenarioProtocolTemplateStore.instantiate(
                reloaded, name: "case-17", sharedMaterials: "NEW RECORD")
            let scenario = minted.record.scenario
            #expect(scenario.name == "case-17")
            #expect(scenario.sharedMaterials == "NEW RECORD")
            #expect(scenario.turns == source.turns)
            #expect(try contractBytes(scenario) == (try contractBytes(source)))
            #expect(scenario.agents.map(\.role) == ["a reviewer", nil, nil])
            #expect(scenario.requiredSchemaVersion == 2)
            // And the minted panel renders the contract, with the new record
            // inside the fence.
            #expect(try render(scenario, turnID: "t2").contains(
                "===== THE RECORD ON APPEAL =====\nNEW RECORD"))
        }
    }

    // MARK: - Editor helpers

    @Test func theInputsPickerOffersEarlierTurnsOnly() throws {
        let panel = contractPanel()
        let available = PanelAuthoring.availableInputs(in: panel, before: "t2")
        #expect(available.map(\.label) == ["t1_ava", "t1_ben"])
        #expect(available.map(\.speakerName) == ["Ava", "Ben"])
        #expect(available[0].summary == "t1_ava — First draft — Ava (Ava)")
        // Nothing is earlier than the first turn, and a turn that is not in
        // the script at all sees the whole script.
        #expect(PanelAuthoring.availableInputs(in: panel, before: "t1a").isEmpty)
        #expect(PanelAuthoring.availableInputs(in: panel, before: "nope").count == 3)
    }

    @Test func theLayoutSummaryDescribesWhatTheRendererWillDo() throws {
        let panel = contractPanel()
        let turn = try #require(panel.turns.first { $0.id == "t2" })
        let summary = PanelAuthoring.contractLayoutSummary(scenario: panel, turn: turn)

        #expect(summary.count == 8)
        #expect(summary[0].contains("You are Ava, a reviewer."))
        #expect(summary[1].contains("THE RECORD ON APPEAL"))
        #expect(summary[2].contains("t1_ava (your own earlier output)"))
        #expect(summary[2].contains("t1_ben (Ben)"))
        #expect(summary[3].contains("OFF for this turn"))
        #expect(summary[5].contains("Ben or Cal"))
        // A template turn has no layout to describe — the engine adds nothing.
        let template = try #require(panel.turns.first { $0.id == "t1a" })
        #expect(PanelAuthoring.contractLayoutSummary(scenario: panel, turn: template).isEmpty)
    }

    @Test func thePreviewGoesThroughTheRealRenderer() throws {
        let panel = contractPanel()
        let turn = try #require(panel.turns.first { $0.id == "t2" })
        let preview = PanelAuthoring.previewPrompt(scenario: panel, turn: turn)

        #expect(preview.hasPrefix("You are Ava, a reviewer."))
        // Runtime-only text is stood in for, visibly: an input block that
        // rendered empty would hide the very arrangement being previewed.
        #expect(preview.contains("===== YOUR OWN EARLIER OUTPUT — First draft — Ava ====="))
        #expect(preview.contains("«t1_ben:"))
        #expect(preview.hasSuffix("Reminder: you are Ava. Respond as Ava and as no one else."))
    }

    // MARK: - Conversion between the two renderers

    @Test func convertingATemplateTurnKeepsTheProseAndDropsTheLayoutPlaceholders() throws {
        let turn = MultiAgentScenario.Turn(
            id: "t", title: "Memo", speakerAgentID: "ava",
            promptTemplate: """
                You are {{agent.name}}.

                Shared materials:
                {{scenario.materials}}

                Prior context:
                {{agent.context}}

                Earlier draft:
                {{outputs.t1_ava}}

                Write a memo to the panel.
                """,
            outputLabel: "memo")
        let conversion = PanelAuthoring.asContractTurn(turn)

        #expect(conversion.turn.promptTemplate.isEmpty)
        let contract = try #require(conversion.turn.contract)
        // The author's own sentences survive; the slots the renderer owns do
        // not, because `validate` refuses them inside contract text.
        #expect(contract.task.contains("Write a memo to the panel."))
        #expect(contract.task.contains("You are {{agent.name}}."))
        #expect(!contract.task.contains("{{scenario.materials}}"))
        #expect(!contract.task.contains("{{agent.context}}"))
        #expect(!contract.task.contains("{{outputs."))
        // And it says so, rather than quietly editing the researcher's text.
        #expect(conversion.notes.count == 1)
        #expect(conversion.notes[0].contains("{{scenario.materials}}"))
        // The result is contract-legal.
        #expect(MultiAgentRunner.forbiddenContractPlaceholder(in: contract.task) == nil)
        // Everything else about the turn is untouched.
        #expect(conversion.turn.title == turn.title)
        #expect(conversion.turn.outputLabel == turn.outputLabel)
        // Converting an already-contract turn is a no-op.
        #expect(PanelAuthoring.asContractTurn(conversion.turn).turn == conversion.turn)
    }

    @Test func convertingBackReassemblesTheContractAndNamesWhatCannotSurvive() throws {
        let panel = contractPanel()
        let turn = try #require(panel.turns.first { $0.id == "t2" })
        let conversion = PanelAuthoring.asTemplateTurn(turn)

        #expect(conversion.turn.contract == nil)
        // Stage, task and format, in render order — nothing authored is lost.
        #expect(conversion.turn.promptTemplate == """
            The group has exchanged first drafts.

            Write a response memo to your colleagues.

            Verdict: yes OR no
            """)
        #expect(conversion.notes.contains { $0.contains("canonical layout is gone") })
        #expect(conversion.notes.contains { $0.contains("t1_ava, t1_ben") })
        // Round-tripping the turn back to a contract is legal again.
        #expect(PanelAuthoring.asContractTurn(conversion.turn).turn.contract != nil)
    }

    @Test func aTaskDraftCollapsesTheHolesTheSubstitutionMade() throws {
        let (task, dropped) = PanelAuthoring.contractTaskDraft(
            from: "Materials:\n{{scenario.materials}}\n\n{{agent.context}}\n\nDecide.")
        #expect(task == "Materials:\n\nDecide.")
        #expect(dropped == ["{{scenario.materials}}", "{{agent.context}}"])
    }

    // MARK: - The editor model

    @MainActor
    @Test func theEditorSwitchesATurnBetweenRenderersAndSavesTheResult() throws {
        try withTempWorkspace { _ in
            let panel = MultiAgentPanel()
            panel.newScenario()
            panel.name = "switching-panel"
            panel.sharedMaterials = "MATERIAL TEXT"
            let turnID = try #require(panel.turns.first).id

            panel.setTurnUsesContract(id: turnID, true)
            #expect(panel.turns[0].contract != nil)
            #expect(panel.turns[0].promptTemplate.isEmpty)
            panel.turns[0].contract?.task = "Write your private notes."

            // The inputs picker, the layout summary and the preview all read
            // the live editor state.
            #expect(panel.availableInputs(forTurn: turnID).isEmpty)
            #expect(panel.contractLayoutSummary(forTurn: turnID).count == 8)
            let preview = try #require(panel.previewPrompt(forTurn: turnID))
            #expect(preview.contains("Write your private notes."))

            panel.saveScenario()
            let saved = try #require(panel.selectedScenario).scenario
            #expect(saved.turns[0].contract?.task == "Write your private notes.")
            #expect(saved.requiredSchemaVersion == 2)
            #expect(!panel.hasUnsavedChanges)

            // …and back, without losing the sentence.
            panel.setTurnUsesContract(id: turnID, false)
            #expect(panel.turns[0].contract == nil)
            #expect(panel.turns[0].promptTemplate.contains("Write your private notes."))
            #expect(panel.turnNotices[turnID]?.isEmpty == false)
        }
    }

    @MainActor
    @Test func contractInputsAreStoredInScriptOrderHoweverTheyAreTicked() throws {
        try withTempWorkspace { _ in
            let panel = MultiAgentPanel()
            panel.newScenario()
            panel.turns = contractPanel().turns
            panel.agents = contractPanel().agents
            panel.setContractInput(turnID: "t2", label: "t1_ava", included: false)
            #expect(panel.turns[2].contract?.inputs == ["t1_ben"])
            // Ticked back in second, but stored first: the list is the order
            // the reader sees the documents in, not the order of the clicks.
            panel.setContractInput(turnID: "t2", label: "t1_ava", included: true)
            #expect(panel.turns[2].contract?.inputs == ["t1_ava", "t1_ben"])
        }
    }

    @MainActor
    @Test func theBuiltInContractPanelValidatesAndRendersContracts() throws {
        try withTempWorkspace { _ in
            let panel = MultiAgentPanel()
            #expect(panel.newContractScenario(discardingChanges: true))
            let semantic = PanelAuthoring.scenarioForSave(
                name: panel.name, description: panel.scenarioDescription,
                sharedMaterials: "THE RECORD OF THIS APPEAL.",
                agents: panel.agents, turns: panel.turns)

            // Every turn is a contract turn, and the panel is runnable once cast.
            #expect(semantic.turns.allSatisfy { $0.contract != nil })
            #expect(semantic.agents.allSatisfy { $0.role != nil })
            let bound = try PanelAuthoring.rehearsalScenario(
                semantic, modelID: "test/model", temperature: 0, maxTokens: 2048)
            try MultiAgentRunner.validate(bound)
            // Clean: no duplicate labels, no unroutable input, no strict-format
            // budget problem.
            #expect(MultiAgentRunner.advisories(bound).isEmpty)
            #expect(bound.requiredSchemaVersion == 2)

            // The built-in TEMPLATE panel is unchanged — a built-in button
            // that quietly started emitting a different prompt shape would
            // re-render every panel built out of habit.
            panel.newScenario()
            #expect(panel.turns.allSatisfy { $0.contract == nil })
            #expect(panel.turns.allSatisfy { !$0.promptTemplate.isEmpty })
            #expect(panel.agents.allSatisfy { $0.role == nil })
        }
    }
}
