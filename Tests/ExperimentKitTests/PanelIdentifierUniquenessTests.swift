import Foundation
import Testing

@testable import ExperimentKit

/// Two seats, or two turns, of a panel may not share an ID.
///
/// A seat's ID (an entry in the panel's `agents` list) is what turns, routing,
/// and records name it by. A turn's ID keys turn-level resume and the per-turn
/// seed. A repeat fails SILENTLY — a resumed transcript replays the first
/// turn's recorded output in place of the second and never generates it — so
/// it is refused, with the ID named and a repair, wherever a panel is authored,
/// checked, cast, or about to run.
///
/// It is never refused on DECODE, so a run directory that already exists still
/// reads.
///
/// The two refusal strings are a cross-engine contract. Python twin:
/// `Server/tests/test_panel_identifier_uniqueness.py` asserts the same
/// literals.
///
/// Serialized: the CLI tests move the process-global workspace root, under
/// the shared `ExperimentRootOverrideLock`.
@Suite(.serialized) struct PanelIdentifierUniquenessTests {

    // The two refusals, spelled out in full. Identical on the Python engine.
    static let duplicateSeat =
        "seats 1 and 3 ('Alice' and 'Carol') share the ID 'a' — each seat needs "
        + "its own ID, because turns, routing, and records refer to a seat by its "
        + "ID; give one of them a different ID, and update any turn that should "
        + "name it"
    static let duplicateTurn =
        "turns 1 and 3 ('Alice opens' and 'Alice closes') share the ID 't1' — "
        + "each turn needs its own ID, because a run uses the turn ID to pick up "
        + "where it stopped and to set each turn's random seed; give one of them a "
        + "different ID"

    // MARK: - Fixtures

    private func scenario(
        agentIDs: [String] = ["a", "b", "c"],
        turnIDs: [String] = ["t1", "t2", "t3"]
    ) -> MultiAgentScenario {
        let names = ["Alice", "Bob", "Carol"]
        let titles = ["Alice opens", "Bob replies", "Alice closes"]
        let speakers = [agentIDs[0], agentIDs[1], agentIDs[0]]
        return MultiAgentScenario(
            name: "panel",
            baseModelID: "test/model",
            agents: zip(agentIDs, names).map {
                .init(id: $0, name: $1, baseModelID: "test/model")
            },
            turns: zip(turnIDs.indices, turnIDs).map { index, id in
                .init(
                    id: id, title: titles[index], speakerAgentID: speakers[index],
                    promptTemplate: "Speak.", outputLabel: "o\(index + 1)")
            })
    }

    private func refusal(_ body: () throws -> Void) -> ExperimentError? {
        do {
            try body()
            return nil
        } catch {
            return error as? ExperimentError
        }
    }

    /// A temporary workspace for the store AND for everything the CLI
    /// resolves (the panel library scan), so nothing here reads a real one.
    private func withTempWorkspace<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .resolvingSymlinksInPath()
            .appending(component: "panel-identity-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        let previousWorkspace = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = previousWorkspace
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    // MARK: - The validator

    @Test("a panel whose IDs are all distinct validates")
    func aCleanPanelValidates() throws {
        #expect(MultiAgentRunner.duplicateIdentifierRefusal(scenario()) == nil)
        try MultiAgentRunner.validate(scenario())
    }

    @Test("a repeated turn ID is refused by name, with a repair")
    func duplicateTurnIDsAreRefused() throws {
        let duplicated = scenario(turnIDs: ["t1", "t2", "t1"])

        let error = try #require(refusal { try MultiAgentRunner.validate(duplicated) })

        #expect(error.reason == Self.duplicateTurn)
        // Typed: a rule declined, with a repair — not an operational failure.
        let typed = try #require(error.lifecycleRefusal)
        #expect(typed.gate == .missingPrerequisite)
        #expect(typed.repairAction.contains("different ID"))
        #expect(typed.repairAction.contains("steerlab-cli panel check "))
    }

    @Test("a repeated seat ID is refused by name, with a repair")
    func duplicateSeatIDsAreRefused() throws {
        let duplicated = scenario(agentIDs: ["a", "b", "a"])

        let error = try #require(refusal { try MultiAgentRunner.validate(duplicated) })

        #expect(error.reason == Self.duplicateSeat)
        let typed = try #require(error.lifecycleRefusal)
        #expect(typed.gate == .missingPrerequisite)
        #expect(typed.repairAction.contains("steerlab-cli panel check "))
    }

    @Test("seats are named before turns, and the first collision wins")
    func refusalOrderIsFixed() throws {
        // Refusal ORDER is part of the cross-engine contract: a panel that
        // trips two rules must name the same one on both engines.
        let both = scenario(agentIDs: ["a", "b", "a"], turnIDs: ["t1", "t1", "t1"])
        #expect(
            refusal { try MultiAgentRunner.validate(both) }?.reason
                == Self.duplicateSeat)

        let turnsOnly = scenario(turnIDs: ["t1", "t1", "t1"])
        let reason = try #require(
            MultiAgentRunner.duplicateIdentifierRefusal(turnsOnly)?.reason)
        #expect(
            reason.hasPrefix(
                "turns 1 and 2 ('Alice opens' and 'Bob replies') share the ID 't1'"))
    }

    // MARK: - …but never when an existing run is read

    @Test("a panel with repeated IDs still decodes, and its readers still work")
    func decodeIsNotValidation() throws {
        // A run directory's `scenario.json` snapshot of a panel that ran
        // before this rule existed must keep opening.
        let duplicated = scenario(
            agentIDs: ["a", "b", "a"], turnIDs: ["t1", "t2", "t1"])
        let data = try JSONEncoder().encode(duplicated)

        let decoded = try JSONDecoder().decode(MultiAgentScenario.self, from: data)

        #expect(decoded.agents.map(\.id) == ["a", "b", "a"])
        #expect(decoded.turns.map(\.id) == ["t1", "t2", "t1"])
        // The authoring advisories and the panel-effects exposure map read it.
        _ = MultiAgentRunner.advisories(decoded)
        let treated = PanelEffects.treatedAgentIDs(in: decoded)
        _ = PanelEffects.exposureByTurn(scenario: decoded, treated: treated)
    }

    @Test("old turn records under a repeated turn ID still read")
    func recordsUnderARepeatedTurnIDStillRead() throws {
        let lines = [("first", 0), ("second", 1)].map { output, index in
            """
            {"condition":"configured","promptID":"t1","promptIndex":\(index),\
            "replicateIndex":0,"output":"\(output)","speakerName":"Alice",\
            "turnTitle":"Turn \(index + 1)"}
            """
        }.joined(separator: "\n")

        let parsed = RunResults.records(fromJSONL: lines)
        let transcript = try #require(
            PanelTranscript.transcripts(from: parsed.records).first)

        #expect(parsed.skippedLines == 0)
        #expect(transcript.turns.map(\.output) == ["first", "second"])
        #expect(transcript.turns.map(\.promptID) == ["t1", "t1"])
    }

    // MARK: - Authoring

    @Test("a casting over a repeated seat ID is refused, not trapped")
    func castingOverARepeatedSeatRefuses() throws {
        // The seats section builds its assignment from the seats it shows; a
        // repeated ID used to trap there, before `compile` could refuse it.
        let duplicated = PanelComposition.semanticForm(
            scenario(agentIDs: ["a", "b", "a"]))
        let state = SeatCasting.State(
            form: .uncast,
            seats: duplicated.agents.map { .init(id: $0.id, name: $0.name) },
            occupants: [:], semantic: duplicated, semanticPath: nil)

        let assignment = state.assignment

        #expect(assignment.seatIDs == ["a", "b", "a"])
        let error = try #require(
            refusal {
                _ = try PanelComposition.compile(
                    semantic: duplicated, assignment: assignment,
                    modelID: "test/model", temperature: 0, maxTokens: 64)
            })
        #expect(error.reason == Self.duplicateSeat)
    }

    @Test("the semantic-panel authoring check names the repeated ID")
    func semanticAuthoringNamesTheRepeat() throws {
        let repeatedTurn = PanelComposition.semanticForm(
            scenario(turnIDs: ["t1", "t2", "t1"]))
        let refusedTurn = try #require(
            refusal { try StudyPanelAuthoring.validate(repeatedTurn) })
        #expect(refusedTurn.reason == Self.duplicateTurn)
        #expect(refusedTurn.lifecycleRefusal?.gate == .missingPrerequisite)

        // This used to answer "Use a semantic panel with unique seats…" —
        // true, and no help in finding the pair.
        let repeatedSeat = PanelComposition.semanticForm(
            scenario(agentIDs: ["a", "b", "a"]))
        let refusedSeat = try #require(
            refusal { try StudyPanelAuthoring.validate(repeatedSeat) })
        #expect(refusedSeat.reason == Self.duplicateSeat)

        // A clean semantic panel still passes the same check.
        try StudyPanelAuthoring.validate(PanelComposition.semanticForm(scenario()))
    }

    // MARK: - The CLI

    @Test("panel check answers a refusal with a runnable repair")
    func panelCheckRefusesWithARepair() async throws {
        try await withTempWorkspace { root in
            let url = root.appending(path: "proposed.json")
            try JSONEncoder().encode(scenario(turnIDs: ["t1", "t2", "t1"]))
                .write(to: url)

            let outcome = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "panel", ["check", url.path])

            // The way an agent reads it: refused / 65, the gate as the code,
            // the ID in the reason, and a repair that names a command.
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.code == "missingPrerequisite")
            #expect(outcome.envelope.error?.gate == "missingPrerequisite")
            #expect(outcome.envelope.error?.reason == Self.duplicateTurn)
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("steerlab-cli panel check "))
            // Human mode keeps exit 1 and the same prose.
            #expect(outcome.exitCode == 1)
            #expect(outcome.failure?.reason == Self.duplicateTurn)
        }
    }

    @Test("panel check still passes a clean panel")
    func panelCheckPassesACleanPanel() async throws {
        try await withTempWorkspace { root in
            let url = root.appending(path: "proposed.json")
            try JSONEncoder().encode(scenario()).write(to: url)

            let outcome = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "panel", ["check", url.path])

            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.error == nil)
        }
    }

    @Test("the registry inventories the refusal site")
    func theRegistryNamesTheSite() throws {
        let site = try #require(RefusalSiteRegistry.site(for: .missingPrerequisite))
        #expect(site.verbs.contains("panel check"))
        #expect(site.origin.contains("MultiAgentRunner.duplicateIdentifierRefusal"))
    }
}
