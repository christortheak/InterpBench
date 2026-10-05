import Foundation
import Testing

@testable import ExperimentKit

/// The flattened panel record names its seat by ID.
///
/// `generations.jsonl` used to carry `speakerName` only. A name is a label:
/// two seats may share one, and a rename changes it, so a turn could be
/// attributed to its seat only by joining `turns.jsonl`.
///
/// `speakerAgentID` is additive. A record written before it existed has no
/// such key, and every decoder here keeps reading it: the lenient reader the
/// Results browser uses, the transcript built from it, and the engine's own
/// record type.
///
/// Python twin: `Server/tests/test_panel_record_seat.py`.
@Suite("PanelRecordSeatTests")
struct PanelRecordSeatTests {

    private let turnWithSeat = """
        {"condition":"configured","promptID":"t1","promptIndex":0,"sampleIndex":0,\
        "replicateIndex":0,"prompt":"p","output":"first words","wordCount":2,\
        "speakerAgentID":"seat-a","speakerName":"Reviewer","turnTitle":"First",\
        "routedAgentIDs":["seat-a","seat-b"]}
        """

    /// The same kind of record as a run written before the key existed has it.
    private let turnWithoutSeat = """
        {"condition":"configured","promptID":"t2","promptIndex":1,"sampleIndex":0,\
        "replicateIndex":0,"prompt":"p","output":"second words","wordCount":2,\
        "speakerName":"Reviewer","turnTitle":"Second","routedAgentIDs":["seat-b"]}
        """

    @Test("a flattened record decodes with and without the seat ID")
    func flattenedRecordDecodesWithAndWithoutTheSeat() throws {
        let parsed = RunResults.records(
            fromJSONL: turnWithSeat + "\n" + turnWithoutSeat)

        #expect(parsed.skippedLines == 0)
        #expect(parsed.records.count == 2)
        #expect(parsed.records.map(\.speakerAgentID) == ["seat-a", nil])
        // Nothing else about the old record changed: it is still a turn, with
        // its name, title and routing.
        #expect(parsed.records.allSatisfy { $0.isTurn })
        #expect(parsed.records.map(\.speakerName) == ["Reviewer", "Reviewer"])
        #expect(parsed.records[1].turnTitle == "Second")
        #expect(parsed.records[1].routedAgentIDs == ["seat-b"])
    }

    @Test("a seat ID that is not a string is absent, never a guess")
    func aNonStringSeatIsAbsent() throws {
        let line = """
            {"condition":"configured","promptID":"t1","speakerAgentID":null,\
            "speakerName":"Reviewer","turnTitle":"First"}
            """
        let record = try #require(RunResults.records(fromJSONL: line).records.first)
        #expect(record.speakerAgentID == nil)
        #expect(record.isTurn)
    }

    @Test("the transcript carries the seat, and reads old runs without it")
    func transcriptCarriesTheSeat() throws {
        let records = RunResults.records(
            fromJSONL: turnWithSeat + "\n" + turnWithoutSeat).records

        let transcript = try #require(
            PanelTranscript.transcripts(from: records).first)

        // Two turns by seats that share a display name: only the ID tells
        // them apart, and only where the run recorded it.
        #expect(transcript.turns.map(\.speaker) == ["Reviewer", "Reviewer"])
        #expect(transcript.turns.map(\.speakerAgentID) == ["seat-a", nil])
        #expect(transcript.turns.map(\.title) == ["First", "Second"])
    }

    // MARK: - The engine's own writer

    private func flattened() -> ExperimentTasks.GenerationRecord {
        let turn = MultiAgentTurnResult(
            turnID: "t1", turnIndex: 1, title: "First", speakerAgentID: "seat-a",
            speakerName: "Reviewer", modelRevision: "rev", prompt: "p",
            output: "first words", outputLabel: "one",
            routedAgentIDs: ["seat-a", "seat-b"], replicateIndex: 0,
            temperature: 0)
        var manifest = ExperimentManifest(
            name: "panel-study", description: "", modelID: "test/model")
        manifest.modelRevision = "rev"
        let row = ExperimentTasks.MetricRow(
            condition: "configured", seed: 7, promptIndex: 0, promptID: turn.turnID,
            wordCount: 2, distinct2: 1, markerDensity: [:], replicate: 0)
        let scenario = MultiAgentScenario(
            name: "panel", baseModelID: "test/model",
            agents: [
                .init(id: "seat-a", name: "Reviewer", baseModelID: "test/model"),
                .init(id: "seat-b", name: "Reviewer", baseModelID: "test/model"),
            ],
            turns: [
                .init(
                    id: "t1", title: "First", speakerAgentID: "seat-a",
                    promptTemplate: "Open.", outputLabel: "one")
            ])
        return ExperimentTasks.panelGenerationRecord(
            turn: turn, row: row, manifest: manifest, experimentHash: "h",
            scenario: scenario, scenarioPath: "prompts/panels/panel.json",
            scenarioHash: "sh", replicate: 0)
    }

    @Test("the Mac engine's flattened record names its seat by ID")
    func theEngineWritesTheSeat() throws {
        let json = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(flattened()))
                as? [String: Any])

        #expect(json["speakerAgentID"] as? String == "seat-a")
        // The neighbours it sits beside are what they always were.
        #expect(json["speakerName"] as? String == "Reviewer")
        #expect(json["turnTitle"] as? String == "First")
        #expect(json["routedAgentIDs"] as? [String] == ["seat-a", "seat-b"])
        #expect(json["promptID"] as? String == "t1")
        #expect(json["condition"] as? String == "configured")
        #expect(json["sampleIndex"] as? Int == 0)
        // …and the lenient reader takes it back.
        let read = try #require(RunResults.record(from: json))
        #expect(read.speakerAgentID == "seat-a")
    }

    @Test("the engine's own record type decodes with and without the seat ID")
    func theRecordTypeRoundTripsBothShapes() throws {
        let record = flattened()
        let withSeat = try JSONEncoder().encode(record)
        #expect(
            try JSONDecoder().decode(
                ExperimentTasks.GenerationRecord.self, from: withSeat
            ).speakerAgentID == "seat-a")

        // The old shape: the same record with the key simply not there.
        var old = try #require(
            try JSONSerialization.jsonObject(with: withSeat) as? [String: Any])
        old.removeValue(forKey: "speakerAgentID")
        let decoded = try JSONDecoder().decode(
            ExperimentTasks.GenerationRecord.self,
            from: JSONSerialization.data(withJSONObject: old))
        #expect(decoded.speakerAgentID == nil)
        #expect(decoded.speakerName == "Reviewer")

        // And a record that is not a panel turn gains no key at all.
        var plain = record
        plain.speakerAgentID = nil
        let plainJSON = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(plain))
                as? [String: Any])
        #expect(plainJSON["speakerAgentID"] == nil, "nil seat must be omitted")
    }
}
