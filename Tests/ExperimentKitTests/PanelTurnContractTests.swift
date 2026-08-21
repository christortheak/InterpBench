import Foundation
import Testing

@testable import ExperimentKit

/// Panel turn contracts (`docs/PANEL-TURN-CONTRACTS-SPEC.md`, 2026-08-17).
///
/// The load-bearing test is `workedExampleRendersByteForByte`: the spec's §2
/// example is the cross-engine contract, so it is asserted here against a
/// string copied out of the spec rather than against anything this file
/// computes. If the Python twin renders something else, one of the two suites
/// fails and the disagreement is visible before a panel runs on it.
@Suite("PanelTurnContractTests")
struct PanelTurnContractTests {

    // MARK: - Fixtures

    private func agent(_ id: String, _ name: String, role: String? = nil)
        -> MultiAgentScenario.Agent
    {
        MultiAgentScenario.Agent(
            id: id, name: name, baseModelID: "test/model", role: role)
    }

    /// The §2 scenario: Ava (a reviewer), Ben, Cal; two first drafts already
    /// produced; a contract turn spoken by Ava.
    private func workedExampleScenario() -> MultiAgentScenario {
        MultiAgentScenario(
            name: "panel",
            baseModelID: "test/model",
            sharedMaterials: "MATERIAL TEXT",
            agents: [agent("ava", "Ava", role: "a reviewer"), agent("ben", "Ben"),
                     agent("cal", "Cal")],
            turns: [
                .init(
                    id: "t1a", title: "First draft — Ava", speakerAgentID: "ava",
                    promptTemplate: "Draft.", outputLabel: "t1_ava"),
                .init(
                    id: "t1b", title: "First draft — Ben", speakerAgentID: "ben",
                    promptTemplate: "Draft.", outputLabel: "t1_ben"),
                .init(
                    id: "t2", title: "Response — Ava", speakerAgentID: "ava",
                    promptTemplate: "", outputLabel: "t2_ava",
                    includeScenarioMaterials: true, includeSpeakerContext: false,
                    contract: TurnContract(
                        stage: "The group has exchanged first drafts and you are now "
                            + "writing your response.",
                        task: "Write a response memo to your colleagues.",
                        format: "Use exactly this format and nothing else:\nVerdict: yes OR no",
                        inputs: ["t1_ava", "t1_ben"])),
            ])
    }

    private let workedExampleOutputs = ["t1_ava": "AVA DRAFT.", "t1_ben": "BEN DRAFT."]

    private func render(
        _ scenario: MultiAgentScenario,
        turnID: String,
        speakerName: String,
        context: String = "",
        outputs: [String: String] = [:]
    ) throws -> String {
        let turn = try #require(scenario.turns.first { $0.id == turnID })
        return MultiAgentRunner.renderPrompt(
            scenario: scenario, turn: turn, speakerName: speakerName,
            speakerContext: context, outputsByLabel: outputs)
    }

    private func validationFailure(_ scenario: MultiAgentScenario) -> String? {
        do {
            try MultiAgentRunner.validate(scenario)
            return nil
        } catch let error as ExperimentError {
            return error.reason
        } catch {
            return String(describing: error)
        }
    }

    // MARK: - The normative worked example (spec §2)

    @Test("the spec's worked example renders byte for byte")
    func workedExampleRendersByteForByte() throws {
        // Copied verbatim out of docs/PANEL-TURN-CONTRACTS-SPEC.md §2. Do not
        // regenerate this from the implementation: it is the cross-engine
        // contract, and a test that reproduces whatever the code does would
        // certify a divergence instead of catching it.
        let expected = """
            You are Ava, a reviewer. The other participants are Ben and Cal. The group has exchanged first drafts and you are now writing your response.

            ===== SHARED MATERIALS =====
            MATERIAL TEXT
            ===== END OF SHARED MATERIALS =====

            That was the shared material. Every participant has read it.

            ===== YOUR OWN EARLIER OUTPUT — First draft — Ava =====
            AVA DRAFT.
            ===== END OF YOUR OWN EARLIER OUTPUT =====

            That was your own earlier output, written by you, Ava.

            ===== OUTPUT OF Ben — First draft — Ben =====
            BEN DRAFT.
            ===== END OF OUTPUT OF Ben =====

            Those were the contributions of the other participants. You have now read them.

            ===== YOUR TASK =====
            You are Ava. Write a response memo to your colleagues.

            Write only your own response, in your own voice, as Ava and no one else. Do not write, draft, continue, quote at length, or reply on behalf of Ben or Cal. Their contributions above are finished documents; you are adding one document of your own.

            Use exactly this format and nothing else:
            Verdict: yes OR no

            Reminder: you are Ava. Respond as Ava and as no one else.
            """

        let rendered = try render(
            workedExampleScenario(), turnID: "t2", speakerName: "Ava",
            outputs: workedExampleOutputs)

        #expect(rendered == expected)
        // No trailing newline: the sandwich ends on the reminder.
        #expect(!rendered.hasSuffix("\n"))
    }

    @Test("a contract turn renders through the contract renderer, never the template one")
    func contractTurnsSkipTheTemplatePath() throws {
        let rendered = try render(
            workedExampleScenario(), turnID: "t2", speakerName: "Ava",
            context: "SOME-CONTEXT", outputs: workedExampleOutputs)

        // includeSpeakerContext is false, so the transcript block is absent —
        // and the template fallback's wording never appears on this path.
        #expect(!rendered.contains("SOME-CONTEXT"))
        #expect(!rendered.contains("Shared scenario materials:"))
        #expect(!rendered.contains("Visible prior context:"))
    }

    // MARK: - Block-by-block behaviour (spec §2)

    @Test("a single-agent scenario drops the roll call and the own-voice block")
    func singleAgentOpenerAndVoice() throws {
        let scenario = MultiAgentScenario(
            name: "solo", baseModelID: "test/model",
            agents: [agent("a", "Ava")],
            turns: [
                .init(
                    id: "t", title: "Note", speakerAgentID: "a", promptTemplate: "",
                    contract: TurnContract(task: "Write a note."))
            ])

        let rendered = try render(scenario, turnID: "t", speakerName: "Ava")

        #expect(
            rendered == """
                You are Ava.

                ===== YOUR TASK =====
                You are Ava. Write a note.

                Reminder: you are Ava. Respond as Ava and as no one else.
                """)
    }

    @Test("three colleagues take the Oxford comma, in both conjunctions")
    func colleagueListsUseTheOxfordComma() throws {
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava"), agent("b", "Ben"), agent("c", "Cal"),
                     agent("d", "Dee")],
            turns: [
                .init(
                    id: "t", title: "Memo", speakerAgentID: "a", promptTemplate: "",
                    contract: TurnContract(task: "Write."))
            ])

        let rendered = try render(scenario, turnID: "t", speakerName: "Ava")

        #expect(rendered.contains("The other participants are Ben, Cal, and Dee."))
        #expect(rendered.contains("on behalf of Ben, Cal, or Dee."))
    }

    @Test("an empty stage, format, or materials block leaves no glue line behind")
    func emptyBlocksAreOmittedEntirely() throws {
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            sharedMaterials: "",
            agents: [agent("a", "Ava"), agent("b", "Ben")],
            turns: [
                .init(
                    id: "t", title: "Memo", speakerAgentID: "a", promptTemplate: "",
                    contract: TurnContract(task: "Write."))
            ])

        let rendered = try render(scenario, turnID: "t", speakerName: "Ava")

        #expect(rendered.hasPrefix("You are Ava. The other participants are Ben.\n\n"))
        #expect(!rendered.contains("===== SHARED MATERIALS ====="))
        #expect(!rendered.contains("That was the shared material."))
        // The task fence is the only one left standing.
        #expect(rendered.components(separatedBy: "=====").count - 1 == 2)
        // No blank-line runs anywhere: blocks join with exactly one.
        #expect(!rendered.contains("\n\n\n"))
    }

    @Test("ownVoice false drops the constraint block and nothing else")
    func ownVoiceCanBeDeclaredOff() throws {
        var scenario = workedExampleScenario()
        scenario.turns[2].contract?.ownVoice = false

        let rendered = try render(
            scenario, turnID: "t2", speakerName: "Ava", outputs: workedExampleOutputs)

        #expect(!rendered.contains("Write only your own response"))
        #expect(rendered.contains("===== YOUR TASK ====="))
        #expect(rendered.contains("Reminder: you are Ava."))
    }

    @Test("the transcript block alone earns the finished-documents sentence")
    func transcriptAloneTriggersTheFinishedDocumentsSentence() throws {
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava"), agent("b", "Ben")],
            turns: [
                .init(
                    id: "t", title: "Memo", speakerAgentID: "a", promptTemplate: "",
                    includeScenarioMaterials: false,
                    contract: TurnContract(task: "Write."))
            ])

        let withTranscript = try render(
            scenario, turnID: "t", speakerName: "Ava", context: "[x] T — Ben\nBEN.")
        let without = try render(scenario, turnID: "t", speakerName: "Ava")

        #expect(withTranscript.contains("===== TRANSCRIPT SO FAR ====="))
        #expect(withTranscript.contains("Their contributions above are finished documents"))
        #expect(!without.contains("Their contributions above are finished documents"))
    }

    @Test("an own-only input draws no other-participants glue line")
    func ownOnlyInputsDrawNoOtherGlue() throws {
        var scenario = workedExampleScenario()
        scenario.turns[2].contract?.inputs = ["t1_ava"]

        let rendered = try render(
            scenario, turnID: "t2", speakerName: "Ava", outputs: workedExampleOutputs)

        #expect(rendered.contains("That was your own earlier output, written by you, Ava."))
        #expect(!rendered.contains("Those were the contributions of the other participants."))
        #expect(!rendered.contains("BEN DRAFT."))
        // No other-authored input and no transcript: the own-voice block stops
        // at the prohibition.
        #expect(!rendered.contains("Their contributions above are finished documents"))
    }

    @Test("the materials fence title is the contract's, verbatim")
    func materialsTitleIsDeclaredData() throws {
        var scenario = workedExampleScenario()
        scenario.turns[2].contract?.materialsTitle = "THE RECORD ON APPEAL"

        let rendered = try render(
            scenario, turnID: "t2", speakerName: "Ava", outputs: workedExampleOutputs)

        #expect(rendered.contains("===== THE RECORD ON APPEAL ====="))
        #expect(rendered.contains("===== END OF THE RECORD ON APPEAL ====="))
    }

    @Test("contract text takes the four scenario substitutions and no others")
    func contractTextTakesTheDeclaredSubstitutions() throws {
        var scenario = workedExampleScenario()
        scenario.description = "A scripted panel."
        scenario.turns[2].contract = TurnContract(
            stage: "{{scenario.name}}: {{scenario.description}}",
            task: "You, {{agent.name}}, are writing {{turn.title}}.")

        let rendered = try render(scenario, turnID: "t2", speakerName: "Ava")

        #expect(rendered.contains("panel: A scripted panel."))
        #expect(rendered.contains("You are Ava. You, Ava, are writing Response — Ava."))
    }

    // MARK: - Schema (spec §1)

    @Test("a contract turn round-trips through JSON")
    func contractRoundTrips() throws {
        let scenario = workedExampleScenario()
        let data = try JSONEncoder().encode(scenario)
        let decoded = try JSONDecoder().decode(MultiAgentScenario.self, from: data)

        #expect(decoded == scenario)
        #expect(decoded.turns[2].contract?.inputs == ["t1_ava", "t1_ben"])
        #expect(decoded.turns[2].contract?.ownVoice == true)
        #expect(decoded.turns[2].contract?.materialsTitle == "SHARED MATERIALS")
        #expect(decoded.agents[0].role == "a reviewer")
        #expect(decoded.agents[1].role == nil)
    }

    @Test("a contract turn decodes with promptTemplate absent as well as empty")
    func promptTemplateIsOptionalOnContractTurns() throws {
        // Writers emit `"promptTemplate": ""` for compatibility with older
        // builds; a file that simply omits it is the same turn and must not be
        // rejected as malformed.
        let json = """
            {"schemaVersion":2,"name":"p","baseModelID":"m","description":"",
             "sharedMaterials":"","temperature":0.0,"maxTokens":512,
             "agents":[{"id":"a","name":"A","baseModelID":"m","systemPrompt":"",
                        "role":"a reviewer"}],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","outputLabel":"",
                       "routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "contract":{"task":"Write."}}]}
            """
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(json.utf8))

        #expect(scenario.turns[0].promptTemplate.isEmpty)
        #expect(scenario.turns[0].contract?.task == "Write.")
        // Defaults fire on absence.
        #expect(scenario.turns[0].contract?.ownVoice == true)
        #expect(scenario.turns[0].contract?.stage == "")
        #expect(scenario.turns[0].contract?.materialsTitle == "SHARED MATERIALS")
        try MultiAgentRunner.validate(scenario)
    }

    @Test("a blank or padded materialsTitle normalizes at decode, like the Python twin")
    func materialsTitleNormalizesAtDecode() throws {
        // Python's `from_dict` stores the STRIPPED title and repairs
        // blank-after-strip to the default; a fence label of nothing would
        // render `=====  =====`. Decode is the shared surface, so Swift must
        // make the identical repair or the engines render different bytes
        // from the same file.
        func decoded(_ title: String) throws -> TurnContract? {
            let json = """
                {"task":"Write.","materialsTitle":\(title)}
                """
            return try JSONDecoder().decode(TurnContract.self, from: Data(json.utf8))
        }
        #expect(try decoded("\"\"")?.materialsTitle == "SHARED MATERIALS")
        #expect(try decoded("\"  \\n \"")?.materialsTitle == "SHARED MATERIALS")
        #expect(try decoded("\"  THE RECORD  \"")?.materialsTitle == "THE RECORD")
    }

    @Test("a half-written contract decodes and refuses at validate, not at decode")
    func decodeIsLenientAndValidateIsStrict() throws {
        // Deliberately unlike `TurnEndpoint`, which refuses on decode: an
        // authoring surface has to be able to HOLD a contract mid-thought.
        // What must never happen is running one, which is validate's job.
        let json = """
            {"name":"p","baseModelID":"m",
             "agents":[{"id":"a","name":"A","baseModelID":"m","systemPrompt":""}],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "contract":{"task":"","inputs":["nope"]}}]}
            """
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(json.utf8))

        #expect(scenario.turns[0].contract?.inputs == ["nope"])
        #expect(validationFailure(scenario)?.contains("with no task") == true)
    }

    @Test("a structurally wrong contract refuses on decode")
    func structurallyWrongContractsRefuse() {
        func panel(_ contract: String) -> String {
            """
            {"name":"p","baseModelID":"m","agents":[],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "contract":\(contract)}]}
            """
        }
        // Leniency is about MISSING and EMPTY, never about a value of the
        // wrong shape: those are not half-written contracts, they are files
        // that do not mean what they say.
        for broken in ["\"Write.\"", "{\"task\":\"Write.\",\"inputs\":\"t1\"}"] {
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(
                    MultiAgentScenario.self, from: Data(panel(broken).utf8))
            }
        }
    }

    @Test("a blank role is the same as no role, on disk and in memory")
    func blankRoleNormalizesToAbsent() throws {
        let json = """
            {"id":"a","name":"A","baseModelID":"m","systemPrompt":"","role":"   "}
            """
        let decoded = try JSONDecoder().decode(
            MultiAgentScenario.Agent.self, from: Data(json.utf8))
        #expect(decoded.role == nil)

        let text = try #require(
            String(data: try JSONEncoder().encode(decoded), encoding: .utf8))
        #expect(!text.contains("role"))
    }

    @Test("a contract-free panel gains no keys and keeps its bytes")
    func contractFreePanelsAreByteStable() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: "panel.json")
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "A")],
            turns: [.init(id: "t", title: "T", speakerAgentID: "a", promptTemplate: "go")])

        _ = try MultiAgentScenarioStore.update(scenario, at: url)
        let first = try MultiAgentScenarioStore.hash(url)
        let text = try String(contentsOf: url, encoding: .utf8)

        // The two new keys are absent, not null and not empty — a key that
        // appears out of nowhere on a no-op re-save is freeze drift on
        // somebody's pinned panel.
        #expect(!text.contains("contract"))
        #expect(!text.contains("role"))
        #expect(text.contains("\"schemaVersion\" : 1"))

        // Re-save from the DECODED scenario, which is the path a study edit
        // actually takes.
        let reloaded = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(contentsOf: url))
        _ = try MultiAgentScenarioStore.update(reloaded, at: url)
        #expect(try MultiAgentScenarioStore.hash(url) == first)
    }

    @Test("schemaVersion is stamped from the turn script, both ways")
    func schemaVersionFollowsTheTurnScript() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-contract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: "panel.json")

        var scenario = workedExampleScenario()
        #expect(scenario.requiredSchemaVersion == 2)
        let record = try MultiAgentScenarioStore.update(scenario, at: url)
        #expect(record.scenario.schemaVersion == 2)
        #expect(try String(contentsOf: url, encoding: .utf8).contains("\"schemaVersion\" : 2"))

        // Removing the last contract puts the file back to 1: the version
        // describes the contents, it is not a latch on the build.
        scenario.turns.removeLast()
        let downgraded = try MultiAgentScenarioStore.update(scenario, at: url)
        #expect(downgraded.scenario.schemaVersion == 1)
        #expect(try String(contentsOf: url, encoding: .utf8).contains("\"schemaVersion\" : 1"))
    }

    @Test("a schema-2 file decodes, and so does one with no schemaVersion at all")
    func bothSchemaVersionsDecode() throws {
        let contractFree = """
            {"name":"p","baseModelID":"m","agents":[],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"go",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true}]}
            """
        // Absent means "read it off the turns", not "assume whatever this
        // build writes today".
        #expect(
            try JSONDecoder().decode(
                MultiAgentScenario.self, from: Data(contractFree.utf8)
            ).schemaVersion == 1)

        let withContract = """
            {"name":"p","baseModelID":"m","agents":[],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "contract":{"task":"Write."}}]}
            """
        #expect(
            try JSONDecoder().decode(
                MultiAgentScenario.self, from: Data(withContract.utf8)
            ).schemaVersion == 2)
    }

    // MARK: - Validation (spec §4)

    private func contractScenario(_ contract: TurnContract, template: String = "")
        -> MultiAgentScenario
    {
        MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava"), agent("b", "Ben")],
            turns: [
                .init(
                    id: "t1", title: "Draft", speakerAgentID: "b",
                    promptTemplate: "go", outputLabel: "draft"),
                .init(
                    id: "t2", title: "Memo", speakerAgentID: "a",
                    promptTemplate: template, outputLabel: "memo",
                    contract: contract),
            ])
    }

    @Test("a contract with no task refuses")
    func emptyTaskRefuses() {
        let reason = validationFailure(contractScenario(TurnContract(task: "   ")))
        #expect(reason?.contains("declares a contract with no task") == true)
    }

    @Test("a contract plus a prompt template refuses")
    func contractPlusTemplateRefuses() {
        let reason = validationFailure(
            contractScenario(TurnContract(task: "Write."), template: "You are …"))
        #expect(reason?.contains("declares both a contract and a prompt template") == true)
    }

    @Test("a contract input naming an unknown or forward label refuses")
    func unresolvableInputRefuses() {
        let unknown = validationFailure(
            contractScenario(TurnContract(task: "Write.", inputs: ["nope"])))
        #expect(unknown?.contains("contract input 'nope' is not produced") == true)

        // Forward reference: the label exists, but on a LATER turn.
        var forward = contractScenario(TurnContract(task: "Write.", inputs: ["memo"]))
        forward.turns[1].contract?.inputs = ["later"]
        forward.turns.append(
            .init(id: "t3", title: "Late", speakerAgentID: "b",
                  promptTemplate: "go", outputLabel: "later"))
        #expect(
            validationFailure(forward)?.contains("contract input 'later' is not produced")
                == true)

        // The backward one is fine.
        #expect(
            validationFailure(contractScenario(TurnContract(task: "Write.", inputs: ["draft"])))
                == nil)
    }

    @Test("a layout placeholder inside contract text refuses, in every field")
    func forbiddenPlaceholdersRefuse() {
        let cases: [(TurnContract, String)] = [
            (TurnContract(stage: "See {{scenario.materials}}", task: "Write."),
             "{{scenario.materials}}"),
            (TurnContract(task: "Read {{agent.context}} and write."), "{{agent.context}}"),
            (TurnContract(task: "Write.", format: "Copy {{outputs.draft}}"),
             "{{outputs.draft}}"),
            (TurnContract(task: "Write.", materialsTitle: "{{agent.context}}"),
             "{{agent.context}}"),
        ]
        for (contract, placeholder) in cases {
            let reason = validationFailure(contractScenario(contract))
            #expect(reason?.contains(placeholder) == true, "for \(placeholder)")
            #expect(reason?.contains("canonical slot") == true, "for \(placeholder)")
        }
    }

    @Test("a template turn still needs its template")
    func templateTurnsStillNeedATemplate() {
        var scenario = contractScenario(TurnContract(task: "Write."))
        scenario.turns[1].contract = nil
        #expect(validationFailure(scenario)?.contains("needs a prompt template") == true)
    }

    // MARK: - Advisories (spec §4)

    @Test("reading an output this speaker was never routed is advised")
    func privateLabelReadIsAdvised() {
        // Ben's turn is private to Ben; Ava's contract reads it anyway.
        var scenario = contractScenario(TurnContract(task: "Write.", inputs: ["draft"]))
        scenario.turns[0].routing = .speakerOnly

        let notes = MultiAgentRunner.advisories(scenario)
        #expect(notes.contains { $0.contains("check whether a private turn is leaking") })

        // Same rule for a template's {{outputs.X}}.
        var template = scenario
        template.turns[1].contract = nil
        template.turns[1].promptTemplate = "Revise {{outputs.draft}}"
        #expect(
            MultiAgentRunner.advisories(template)
                .contains { $0.contains("check whether a private turn is leaking") })

        // Routed to everyone: no advisory.
        var routed = scenario
        routed.turns[0].routing = .all
        #expect(MultiAgentRunner.advisories(routed).isEmpty)
    }

    @Test("a fence marker in the shared materials is advised")
    func fenceCollisionIsAdvised() {
        var scenario = contractScenario(TurnContract(task: "Write."))
        scenario.sharedMaterials = "PART ONE\n===== EXHIBIT A =====\nText."

        #expect(
            MultiAgentRunner.advisories(scenario)
                .contains { $0.contains("fence marker") })
    }

    @Test("a strict-format turn with room to write an opinion is advised")
    func strictFormatBudgetIsAdvised() {
        let endpoint = TurnEndpoint(
            name: "vote", kind: .choice, marker: "Vote:", vocabulary: ["affirm", "reverse"])
        var scenario = contractScenario(TurnContract(task: "Write."))
        scenario.maxTokens = 2048
        scenario.turns[1].endpoint = endpoint

        // Falls back to the scenario default, which is far over budget.
        #expect(
            MultiAgentRunner.advisories(scenario)
                .contains { $0.contains("invites format contamination") })

        // A per-turn cap at the budget is quiet.
        scenario.turns[1].maxTokens = 512
        #expect(MultiAgentRunner.advisories(scenario).isEmpty)
    }

    @Test("a clean contract panel draws no advisories")
    func cleanContractPanelIsQuiet() {
        #expect(
            MultiAgentRunner.advisories(
                contractScenario(TurnContract(task: "Write.", inputs: ["draft"]))
            ).isEmpty)
    }

    // MARK: - Acknowledged cross-routing reads (spec §4.1)

    /// The real prec-delib shape in miniature: a private round 1, then three
    /// rounds that read all three round-1 memos by name. 3 seats × 3 later
    /// rounds × 2 COLLEAGUE memos = 18 private reads BY DESIGN (a seat's own
    /// memo was routed to it, so it draws nothing) — the number that made the
    /// advisory unreadable and the reason the declaration exists.
    private func blindRoundScenario(acknowledged: Bool) -> MultiAgentScenario {
        let seats = ["judge-1", "judge-2", "judge-3"]
        var turns: [MultiAgentScenario.Turn] = seats.map { seat in
            .init(
                id: "r1-\(seat)", title: "Round 1 — \(seat)", speakerAgentID: seat,
                promptTemplate: "", outputLabel: "r1_\(seat)", routing: .speakerOnly,
                contract: TurnContract(task: "Write your memo."))
        }
        for round in 2 ... 4 {
            for seat in seats {
                turns.append(
                    .init(
                        id: "r\(round)-\(seat)", title: "Round \(round) — \(seat)",
                        speakerAgentID: seat, promptTemplate: "",
                        outputLabel: "r\(round)_\(seat)", routing: .all,
                        contract: TurnContract(
                            task: "Respond.", inputs: seats.map { "r1_\($0)" }),
                        acknowledgedInputs: acknowledged
                            ? seats.filter { $0 != seat }.map { "r1_\($0)" } : nil))
            }
        }
        return MultiAgentScenario(
            name: "panel", baseModelID: "test/model", sharedMaterials: "M",
            agents: seats.map { agent($0, $0.capitalized) }, turns: turns)
    }

    @Test("a blind-round design is 18 advisories until it declares its reads")
    func acknowledgedReadsSilenceTheAdvisory() throws {
        let undeclared = MultiAgentRunner.advisories(blindRoundScenario(acknowledged: false))
        #expect(undeclared.count == 18)
        #expect(undeclared.allSatisfy { $0.contains("private turn is leaking") })

        let declared = blindRoundScenario(acknowledged: true)
        #expect(MultiAgentRunner.advisories(declared).isEmpty)
        try MultiAgentRunner.validate(declared)
    }

    @Test("an undeclared private read still fires beside the acknowledged ones")
    func suppressionIsPerLabelNotPerTurn() {
        // Declaring two reads must not buy silence for a third.
        var scenario = blindRoundScenario(acknowledged: true)
        scenario.turns.insert(
            .init(
                id: "r1-extra", title: "Round 1 — extra", speakerAgentID: "judge-2",
                promptTemplate: "go", outputLabel: "r1_extra", routing: .speakerOnly),
            at: 3)
        scenario.turns[4].contract?.inputs.append("r1_extra")

        let notes = MultiAgentRunner.advisories(scenario)

        #expect(notes.count == 1)
        #expect(notes.first?.contains("reads the output 'r1_extra'") == true)
        #expect(notes.first?.hasSuffix("check whether a private turn is leaking") == true)
    }

    /// One private producer, one reader that acknowledges `labels`.
    private func acknowledgedScenario(
        labels: [String],
        producerRouting: MultiAgentScenario.Turn.Routing = .speakerOnly,
        inputs: [String] = ["secret"]
    ) -> MultiAgentScenario {
        MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava"), agent("b", "Ben")],
            turns: [
                .init(
                    id: "t1", title: "Private note", speakerAgentID: "a",
                    promptTemplate: "go", outputLabel: "secret", routing: producerRouting),
                .init(
                    id: "t2", title: "Reader", speakerAgentID: "b", promptTemplate: "",
                    outputLabel: "two",
                    contract: TurnContract(task: "T", inputs: inputs),
                    acknowledgedInputs: labels),
            ])
    }

    @Test("an acknowledgment of a read that is not there is advised as stale")
    func staleAcknowledgmentIsAdvised() {
        // A silencer left behind after the read it silenced was edited away.
        #expect(
            MultiAgentRunner.advisories(acknowledgedScenario(labels: ["secret", "gone"]))
                == [
                    "turn 'Reader' acknowledges the read of 'gone' but does not read it "
                        + "— remove the stale acknowledgment"
                ])

        // Read, but produced by nothing at all: no advisory could ever have
        // fired for it, so the acknowledgment is the same kind of leftover.
        #expect(
            MultiAgentRunner.advisories(
                acknowledgedScenario(
                    labels: ["secret", "nowhere"], inputs: ["secret", "nowhere"]))
                == [
                    "turn 'Reader' acknowledges the read of 'nowhere' but does not read "
                        + "it — remove the stale acknowledgment"
                ])
    }

    @Test("an acknowledgment of a routed read is advised as doing nothing")
    func noOpAcknowledgmentIsAdvised() {
        #expect(
            MultiAgentRunner.advisories(
                acknowledgedScenario(labels: ["secret"], producerRouting: .all))
                == [
                    "turn 'Reader' acknowledges the read of 'secret', but its speaker "
                        + "was routed that output anyway — the acknowledgment is doing "
                        + "nothing"
                ])
    }

    @Test("acknowledgment hygiene sits between the private reads and the budget")
    func acknowledgmentHygieneKeepsItsPlaceInTheOrder() {
        // The cross-engine emission order, asserted as a list: unknown
        // reference, unacknowledged private read, acknowledgment hygiene,
        // strict-format budget, duplicate label.
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava"), agent("b", "Ben")],
            turns: [
                .init(
                    id: "t1", title: "One", speakerAgentID: "a", promptTemplate: "go",
                    outputLabel: "secret", routing: .speakerOnly),
                .init(
                    id: "t2", title: "Two", speakerAgentID: "b",
                    promptTemplate: "{{outputs.nope}} {{outputs.secret}}",
                    outputLabel: "secret",
                    endpoint: TurnEndpoint(
                        name: "v", kind: .choice, marker: "V:", vocabulary: ["y"]),
                    acknowledgedInputs: ["stale"]),
            ],
            maxTokens: 2048)

        let notes = MultiAgentRunner.advisories(scenario)

        #expect(notes.count == 5)
        #expect(notes[0].contains("interpolates {{outputs.nope}}"))
        #expect(notes[1].contains("private turn is leaking"))
        #expect(notes[2].contains("remove the stale acknowledgment"))
        #expect(notes[3].contains("invites format contamination"))
        #expect(notes[4].contains("reuses the output label 'secret'"))
    }

    @Test("acknowledgments are reported in declaration order")
    func acknowledgmentsKeepTheirDeclaredOrder() {
        let notes = MultiAgentRunner.advisories(
            acknowledgedScenario(labels: ["secret", "zulu", "alpha"]))

        #expect(notes.count == 2)
        #expect(notes[0].contains("'zulu'"))
        #expect(notes[1].contains("'alpha'"))
    }

    @Test("acknowledgments round-trip with their order intact")
    func acknowledgmentsRoundTrip() throws {
        var scenario = workedExampleScenario()
        scenario.turns[2].acknowledgedInputs = ["t1_ben", "t1_ava"]

        let data = try JSONEncoder().encode(scenario)
        let decoded = try JSONDecoder().decode(MultiAgentScenario.self, from: data)

        #expect(decoded == scenario)
        #expect(decoded.turns[2].acknowledgedInputs == ["t1_ben", "t1_ava"])
        #expect(
            String(data: data, encoding: .utf8)?.contains("acknowledgedInputs") == true)
    }

    @Test("an acknowledgment-free panel gains no key and keeps its bytes")
    func acknowledgmentFreePanelsAreByteStable() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-ack-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: "panel.json")
        var scenario = workedExampleScenario()

        _ = try MultiAgentScenarioStore.update(scenario, at: url)
        let first = try MultiAgentScenarioStore.hash(url)
        #expect(!(try String(contentsOf: url, encoding: .utf8)).contains("acknowledgedInputs"))

        // Empty is the same as absent, all the way to the bytes: a key that
        // appears out of nowhere on a no-op re-save is freeze drift.
        scenario.turns[2].acknowledgedInputs = []
        #expect(scenario.turns[2].acknowledgedInputs == nil)
        _ = try MultiAgentScenarioStore.update(scenario, at: url)
        #expect(try MultiAgentScenarioStore.hash(url) == first)

        let reloaded = try JSONDecoder().decode(
            MultiAgentScenario.self, from: Data(contentsOf: url))
        _ = try MultiAgentScenarioStore.update(reloaded, at: url)
        #expect(try MultiAgentScenarioStore.hash(url) == first)
    }

    @Test("acknowledgedInputs does not move the derived schemaVersion")
    func acknowledgmentsAreAdvisoryOnlyForTheSchema() throws {
        // Advisory-only semantics, spelled as a version claim: an older build
        // that ignores the key loses nothing measurable, so the key must not
        // declare a new schema on its own.
        var scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [agent("a", "Ava")],
            turns: [
                .init(id: "t1", title: "One", speakerAgentID: "a", promptTemplate: "go",
                      outputLabel: "one"),
                .init(id: "t2", title: "Two", speakerAgentID: "a",
                      promptTemplate: "Revise {{outputs.one}}", outputLabel: "two",
                      acknowledgedInputs: ["one"]),
            ])
        #expect(scenario.requiredSchemaVersion == 1)

        // The contract is what carries the version, and it still does with an
        // acknowledgment sitting next to it.
        scenario.turns[1].promptTemplate = ""
        scenario.turns[1].contract = TurnContract(task: "Revise.", inputs: ["one"])
        #expect(scenario.requiredSchemaVersion == 2)
    }

    @Test("a non-array acknowledgment refuses on decode")
    func nonArrayAcknowledgmentsRefuse() {
        func panel(_ value: String) -> String {
            """
            {"name":"p","baseModelID":"m","agents":[],
             "turns":[{"id":"t","title":"T","speakerAgentID":"a","promptTemplate":"go",
                       "outputLabel":"","routing":"all","routedAgentIDs":[],
                       "includeScenarioMaterials":true,"includeSpeakerContext":true,
                       "acknowledgedInputs":\(value)}]}
            """
        }
        // Leniency is about MISSING and EMPTY, never the wrong shape — the
        // same floor the contract's `inputs` has.
        for broken in ["\"one\"", "{\"labels\":[\"one\"]}", "3"] {
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(
                    MultiAgentScenario.self, from: Data(panel(broken).utf8))
            }
        }
        // Absent and null are both "none declared".
        for quiet in ["null", "[]"] {
            let scenario = try? JSONDecoder().decode(
                MultiAgentScenario.self, from: Data(panel(quiet).utf8))
            #expect(scenario?.turns[0].acknowledgedInputs == nil)
        }
    }

    // MARK: - Reader-aware transcript entries (spec §3.1)

    @Test("an entry names the reader's own output as their own")
    func contextEntriesAreReaderAware() {
        let speaker = agent("a", "Judge A")
        let turn = MultiAgentScenario.Turn(
            id: "t", title: "Memo", speakerAgentID: "a", promptTemplate: "go",
            outputLabel: "memo_a")

        let own = MultiAgentRunner.contextEntry(
            turn: turn, outputLabel: "memo_a", speaker: speaker, output: "TEXT",
            reader: "a")
        let other = MultiAgentRunner.contextEntry(
            turn: turn, outputLabel: "memo_a", speaker: speaker, output: "TEXT",
            reader: "b")

        #expect(own == "[memo_a] Memo — your own earlier output (Judge A)\nTEXT")
        #expect(other == "[memo_a] Memo — Judge A\nTEXT")
    }

    // MARK: - Template fallback placement (spec §3.2)

    @Test("the template fallback prepends: record, transcript, instruction")
    func fallbackIsPrependedInOrder() throws {
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model", sharedMaterials: "CASE-RECORD",
            agents: [agent("a", "Judge A")],
            turns: [
                .init(id: "t", title: "Notes", speakerAgentID: "a",
                      promptTemplate: "You are {{agent.name}}. Write notes.")
            ])

        let rendered = try render(
            scenario, turnID: "t", speakerName: "Judge A", context: "PRIOR-OUTPUT")

        #expect(
            rendered == """
                Shared scenario materials:
                CASE-RECORD

                Visible prior context:
                PRIOR-OUTPUT

                You are Judge A. Write notes.
                """)
    }

    @Test("a template that interpolates one placeholder still prepends only the other")
    func fallbackPrependsOnlyWhatTheTemplateOmitted() throws {
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model", sharedMaterials: "CASE-RECORD",
            agents: [agent("a", "Judge A")],
            turns: [
                .init(id: "t", title: "Notes", speakerAgentID: "a",
                      promptTemplate: "Record:\n{{scenario.materials}}\n\nWrite notes.")
            ])

        let rendered = try render(
            scenario, turnID: "t", speakerName: "Judge A", context: "PRIOR-OUTPUT")

        // The shipped parity bug this guards: the fallback tests the ORIGINAL
        // template, not the substituted string, so an interpolated placeholder
        // is never also prepended.
        #expect(
            rendered == """
                Visible prior context:
                PRIOR-OUTPUT

                Record:
                CASE-RECORD

                Write notes.
                """)
    }

    // MARK: - The committed cross-engine fixture

    @Test("the panel-render fixture replays byte for byte")
    func crossEngineFixtureReplays() throws {
        // Same directory the Python suite reads
        // (`Server/tests/test_panel_contracts.py`). scenario.json is INPUT —
        // it is written in the Python engine's dialect, so this replays it and
        // compares the rendered prompts, and never asserts that this encoder
        // reproduces those bytes.
        let root = VectorCatalog.bundledSeedRoot
            .appending(components: "prompts", "fixtures", "panel-render")
        let scenario = try JSONDecoder().decode(
            MultiAgentScenario.self,
            from: try Data(contentsOf: root.appending(component: "scenario.json")))

        // The fixture is deliberately clean, so either of these firing is a
        // regression rather than a property of the panel.
        //
        // The seats are cast with an empty `baseModelID` — the fixture is a
        // renderer input and never loads a model — and Swift's `validate` has
        // a rule the Python twin does not: a seat needs a base model before it
        // can RUN. That rule predates contracts and is not the spec's to
        // relax, so the copy validated here fills it in; nothing else about
        // the panel is touched, and the rendered bytes below come from the
        // fixture exactly as committed.
        var runnable = scenario
        for index in runnable.agents.indices { runnable.agents[index].baseModelID = "test/model" }
        try MultiAgentRunner.validate(runnable)
        #expect(MultiAgentRunner.advisories(scenario).isEmpty)

        // Stub outputs from the fixture README, keyed by output label.
        let stubs = [
            "t1_ava": "AVA DRAFT.", "t1_ben": "BEN DRAFT.", "t2_ava": "AVA RESPONSE.",
            "t3_ava": "AVA RECAP.", "t4_ben": "Verdict: yes", "t5_cal": "CAL NOTE.",
        ]

        // The replay is the run loop's own routing and labelling rules, not a
        // copy of them: same helpers, same order.
        var context: [String: [String]] = [:]
        var outputsByLabel: [String: String] = [:]
        for (index, turn) in scenario.turns.enumerated() {
            let label = MultiAgentRunner.normalizedOutputLabel(turn: turn, index: index)
            let speaker = try #require(scenario.agents.first { $0.id == turn.speakerAgentID })
            let rendered = MultiAgentRunner.renderPrompt(
                scenario: scenario, turn: turn, speakerName: speaker.name,
                speakerContext: (context[speaker.id] ?? []).joined(separator: "\n\n"),
                outputsByLabel: outputsByLabel)

            // Raw bytes: the expected files carry no trailing newline, and a
            // line-oriented read would quietly forgive one.
            let expected = try Data(
                contentsOf: root.appending(components: "expected", "\(label).txt"))
            #expect(Data(rendered.utf8) == expected, "rendered prompt for \(label)")

            let output = try #require(stubs[label])
            outputsByLabel[label] = output
            for reader in MultiAgentRunner.routedAgentIDs(for: turn, agents: scenario.agents) {
                context[reader, default: []].append(
                    MultiAgentRunner.contextEntry(
                        turn: turn, outputLabel: label, speaker: speaker,
                        output: output, reader: reader))
            }
        }
    }

    // MARK: - Turn-record provenance (spec §3.3)

    @Test("the record stamps which renderer produced its prompt")
    func turnRecordsStampTheRenderer() throws {
        let scenario = workedExampleScenario()
        #expect(MultiAgentRunner.promptRenderer(for: scenario.turns[0]) == "template-v1")
        #expect(MultiAgentRunner.promptRenderer(for: scenario.turns[2]) == "contract-v1")

        let record = MultiAgentTurnResult(
            turnID: "t", turnIndex: 1, title: "T", speakerAgentID: "a",
            speakerName: "Ava", modelRevision: nil, prompt: "p", output: "o",
            outputLabel: "l", routedAgentIDs: ["a"],
            promptRenderer: "contract-v1")
        let text = try #require(
            String(data: try JSONEncoder().encode(record), encoding: .utf8))
        #expect(text.contains("\"promptRenderer\":\"contract-v1\""))
    }

    @Test("a turn record written before the two renderers still decodes")
    func preContractTurnRecordDecodes() throws {
        let json = """
            {"turnID":"t1","turnIndex":1,"title":"Memo","speakerAgentID":"a",
             "speakerName":"Judge A","prompt":"p","output":"o",
             "outputLabel":"memo_a","routedAgentIDs":["a"]}
            """
        let turn = try JSONDecoder().decode(
            MultiAgentTurnResult.self, from: Data(json.utf8))

        // Absent, not defaulted: a run from before the split is a run whose
        // renderer nobody stamped.
        #expect(turn.promptRenderer == nil)
    }
}
