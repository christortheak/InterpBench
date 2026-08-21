import Foundation
import Testing

@testable import ExperimentKit

@Suite("PanelTranscriptTests")
struct PanelTranscriptTests {

    private func turn(
        condition: String, replicate: Int?, index: Int, id: String,
        speaker: String, title: String, output: String,
        routed: [String]? = nil
    ) -> RunResults.Record {
        var record = RunResults.Record(condition: condition, promptID: id)
        record.promptIndex = index
        record.replicateIndex = replicate
        record.speakerName = speaker
        record.turnTitle = title
        record.output = output
        record.prompt = "p"
        record.wordCount = output.split(separator: " ").count
        record.routedAgentIDs = routed
        return record
    }

    @Test("transcripts group by condition and play-through, in script order")
    func groupsByConditionAndReplicate() {
        // Deliberately shuffled input: grouping must not depend on file order.
        let records = [
            turn(condition: "configured", replicate: 1, index: 1, id: "t2",
                 speaker: "B", title: "Second", output: "c1 second"),
            turn(condition: "baseline", replicate: 0, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "b0 first"),
            turn(condition: "configured", replicate: 0, index: 1, id: "t2",
                 speaker: "B", title: "Second", output: "c0 second"),
            turn(condition: "configured", replicate: 1, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "c1 first"),
            turn(condition: "configured", replicate: 0, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "c0 first"),
            turn(condition: "baseline", replicate: 0, index: 1, id: "t2",
                 speaker: "B", title: "Second", output: "b0 second"),
        ]

        let transcripts = PanelTranscript.transcripts(from: records)

        #expect(transcripts.count == 3)
        // Baseline reads first so the arms line up left-to-right.
        #expect(transcripts.first?.condition == "baseline")
        for transcript in transcripts {
            #expect(transcript.turns.map(\.title) == ["First", "Second"])
        }
        #expect(PanelTranscript.replicates(in: transcripts) == [0, 1])
    }

    @Test("arms of one play-through pair up for side-by-side reading")
    func armsPairByReplicate() {
        let records = [
            turn(condition: "baseline", replicate: 0, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "b"),
            turn(condition: "configured", replicate: 0, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "c"),
            turn(condition: "configured", replicate: 1, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "c2"),
        ]
        let transcripts = PanelTranscript.transcripts(from: records)

        let first = PanelTranscript.arms(in: transcripts, replicate: 0)
        #expect(first.map(\.condition) == ["baseline", "configured"])
        // Replicate 1 has no baseline arm; it is not invented.
        #expect(PanelTranscript.arms(in: transcripts, replicate: 1).map(\.condition)
            == ["configured"])
    }

    @Test("non-panel records yield no transcripts")
    func ordinaryStudyRecordsAreIgnored() {
        var ordinary = RunResults.Record(condition: "baseline", promptID: "case-1")
        ordinary.output = "an ordinary study generation"
        // No speaker/title: not a turn.
        #expect(PanelTranscript.transcripts(from: [ordinary]).isEmpty)
    }

    @Test("absent routing is distinguished from routing to nobody")
    func routingAbsenceIsNotEmptiness() {
        // A run predating routedAgentIDs must not be rendered as "private".
        let legacy = turn(
            condition: "configured", replicate: 0, index: 0, id: "t1",
            speaker: "A", title: "First", output: "x", routed: nil)
        let priv = turn(
            condition: "configured", replicate: 0, index: 1, id: "t2",
            speaker: "A", title: "Second", output: "y", routed: [])

        let turns = PanelTranscript.transcripts(from: [legacy, priv])[0].turns
        #expect(turns[0].routedTo == nil)
        #expect(turns[1].routedTo == [])
    }

    @Test("records with no replicate index fall into play-through 0")
    func legacyRecordsDefaultToFirstReplicate() {
        let records = [
            turn(condition: "configured", replicate: nil, index: 0, id: "t1",
                 speaker: "A", title: "First", output: "x")
        ]
        let transcripts = PanelTranscript.transcripts(from: records)
        #expect(transcripts.count == 1)
        #expect(transcripts[0].replicate == 0)
    }
}

@Suite("PanelEffectsTests")
struct PanelEffectsTests {

    private func turn(
        id: String, speaker: String, output: String
    ) -> MultiAgentTurnResult {
        MultiAgentTurnResult(
            turnID: id, turnIndex: 1, title: id, speakerAgentID: speaker,
            speakerName: speaker, modelRevision: nil, prompt: "p", output: output,
            outputLabel: id, routedAgentIDs: [])
    }

    @Test("direct and spillover split by whether the seat was treated")
    func splitsDirectFromSpillover() {
        // Seat "a" carries the variant; "b" only ever receives routed context.
        let configured = [
            turn(id: "t1", speaker: "a", output: "10"),
            turn(id: "t2", speaker: "b", output: "4"),
        ]
        let baseline = [
            turn(id: "t1", speaker: "a", output: "0"),
            turn(id: "t2", speaker: "b", output: "0"),
        ]

        let row = PanelEffects.compute(
            configured: configured, baseline: baseline, treated: ["a"],
            endpoint: "value", parse: { Double($0) }, groupTurnID: "t2")

        #expect(row.direct == 10)
        #expect(row.directN == 1)
        #expect(row.spillover == 4)
        #expect(row.spilloverN == 1)
        // group = the designated panel-outcome turn.
        #expect(row.group == 4)
        // How much of the injected stance survived deliberation.
        #expect(abs(row.transmissionRatio - 0.4) < 1e-9)
        #expect(abs(row.amplification - 0.4) < 1e-9)
    }

    @Test("unparseable turns are dropped and counted, never coerced")
    func unparseableTurnsAreCounted() {
        // Coercing a failed parse to zero would drag every mean toward
        // no-effect while looking like data.
        let configured = [
            turn(id: "t1", speaker: "a", output: "10"),
            turn(id: "t2", speaker: "a", output: "not a number"),
        ]
        let baseline = [
            turn(id: "t1", speaker: "a", output: "0"),
            turn(id: "t2", speaker: "a", output: "0"),
        ]

        let row = PanelEffects.compute(
            configured: configured, baseline: baseline, treated: ["a"],
            endpoint: "value", parse: { Double($0) })

        #expect(row.directN == 1)
        #expect(row.direct == 10)
        #expect(row.droppedTurns == 1)
    }

    @Test("ratios are undefined rather than infinite when there is no direct effect")
    func ratiosGuardAgainstZeroDirect() {
        let configured = [turn(id: "t1", speaker: "b", output: "5")]
        let baseline = [turn(id: "t1", speaker: "b", output: "0")]

        let row = PanelEffects.compute(
            configured: configured, baseline: baseline, treated: ["a"],
            endpoint: "value", parse: { Double($0) })

        #expect(row.directN == 0)
        #expect(row.direct.isNaN)
        #expect(row.transmissionRatio.isNaN)
    }

    @Test("CSV header matches the server's contract")
    func csvHeaderIsTheCrossEngineContract() {
        let text = PanelEffects.csv([
            PanelEffects.compute(
                configured: [turn(id: "t1", speaker: "a", output: "2")],
                baseline: [turn(id: "t1", speaker: "a", output: "1")],
                treated: ["a"], endpoint: "wordCount", parse: { Double($0) })
        ])
        let lines = text.split(separator: "\n")

        // Byte-identical to the server's `write_panel_effects` fieldnames.
        // `unexposed`/`unexposedN` joined both engines together when spillover
        // learned to test exposure — a header change is exactly what this
        // assertion exists to catch.
        #expect(
            lines[0]
                == "endpoint,direct,directN,spillover,spilloverN,group,groupN,"
                    + "transmissionRatio,amplification,unexposed,unexposedN,"
                    + "droppedTurns")
        // Reads back through the shared parser either engine's file uses.
        let parsed = RunResults.panelEffects(fromCSV: text)
        #expect(parsed?.count == 1)
        #expect(parsed?.first?.endpoint == "wordCount")
    }

    @Test("NaN is written as an empty field, matching the server")
    func nanFormatsAsEmpty() {
        #expect(PanelEffects.format(.nan) == "")
        #expect(PanelEffects.format(0.4) == "0.4")
    }
}


@Suite("PanelExposureTests")
struct PanelExposureTests {

    /// A treated seat (a) that speaks privately first, then to b only; b then
    /// speaks to c. So b is exposed second-hand and c third-hand.
    private func scenario() -> MultiAgentScenario {
        MultiAgentScenario(
            name: "panel", baseModelID: "m",
            agents: [
                .init(id: "a", name: "A", baseModelID: "m",
                      variantArtifactPath: "v.json"),
                .init(id: "b", name: "B", baseModelID: "m"),
                .init(id: "c", name: "C", baseModelID: "m"),
            ],
            turns: [
                .init(id: "p_a", title: "notes A", speakerAgentID: "a",
                      promptTemplate: "x", routing: .speakerOnly),
                .init(id: "p_b", title: "notes B", speakerAgentID: "b",
                      promptTemplate: "x", routing: .speakerOnly),
                .init(id: "memo_a", title: "memo A", speakerAgentID: "a",
                      promptTemplate: "x", routing: .selected, routedAgentIDs: ["b"]),
                .init(id: "memo_b", title: "memo B", speakerAgentID: "b",
                      promptTemplate: "x", routing: .selected, routedAgentIDs: ["c"]),
                .init(id: "memo_c", title: "memo C", speakerAgentID: "c",
                      promptTemplate: "x", routing: .all),
            ])
    }

    @Test("exposure follows routing, not seat identity")
    func exposureFollowsRouting() {
        let exposed = PanelEffects.exposureByTurn(
            scenario: scenario(), treated: ["a"])

        // B speaks before hearing anything from A: cannot carry the intervention.
        #expect(exposed["p_b"] == false)
        // After A's memo reaches B, B is exposed.
        #expect(exposed["memo_b"] == true)
        // C never heard A directly, only through B — second-order propagation.
        #expect(exposed["memo_c"] == true)
    }

    @Test("exposure follows what the prompt actually reads")
    func exposureFollowsThePrompt() {
        // Routing records who COULD hear; the prompt decides who DOES.
        let scenario = MultiAgentScenario(
            name: "p", baseModelID: "m",
            agents: [
                .init(id: "a", name: "A", baseModelID: "m",
                      variantArtifactPath: "v.json"),
                .init(id: "b", name: "B", baseModelID: "m"),
            ],
            turns: [
                .init(id: "t_a", title: "A speaks", speakerAgentID: "a",
                      promptTemplate: "x", outputLabel: "amemo", routing: .all),
                // Routed to, but reads no context: NOT exposed.
                .init(id: "blind", title: "B ignores", speakerAgentID: "b",
                      promptTemplate: "x", routing: .all,
                      includeSpeakerContext: false),
                // Reads the routed context: exposed.
                .init(id: "reads", title: "B reads", speakerAgentID: "b",
                      promptTemplate: "x", routing: .all),
                // No routing at all, but interpolates a treated output.
                .init(id: "interp", title: "B interpolates", speakerAgentID: "b",
                      promptTemplate: "see {{outputs.amemo}}", routing: .none,
                      includeSpeakerContext: false),
            ])

        let exposed = PanelEffects.exposureByTurn(scenario: scenario, treated: ["a"])
        #expect(exposed["blind"] == false)
        #expect(exposed["reads"] == true)
        #expect(exposed["interp"] == true)
    }

    @Test("pre-exposure turns are a placebo channel, not spillover")
    func unexposedTurnsAreNotSpillover() throws {
        let scenario = self.scenario()
        let exposed = PanelEffects.exposureByTurn(scenario: scenario, treated: ["a"])
        func turns(_ lift: String) -> [MultiAgentTurnResult] {
            scenario.turns.map { turn in
                MultiAgentTurnResult(
                    turnID: turn.id, turnIndex: 1, title: turn.title,
                    speakerAgentID: turn.speakerAgentID, speakerName: turn.speakerAgentID,
                    modelRevision: nil, prompt: "p",
                    output: turn.id == "memo_b" ? lift : "0",
                    outputLabel: turn.id, routedAgentIDs: [])
            }
        }

        let row = PanelEffects.compute(
            configured: turns("10"), baseline: turns("0"), treated: ["a"],
            endpoint: "v", parse: { Double($0) }, exposedTurns: exposed)

        // Only the two exposed untreated turns feed spillover; B's
        // pre-exposure private note goes to the placebo channel. The old rule
        // averaged that structural zero in and reported 3.33 instead of 5.
        #expect(row.spilloverN == 2)
        #expect(row.unexposedN == 1)
        #expect(abs(row.spillover - 5.0) < 1e-9)
    }
}

@Suite("PanelResumeTests")
struct PanelResumeTests {

    @Test("a torn final line is truncated, not left to poison the file")
    func tornTailIsTruncated() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-resume-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(component: "turns.jsonl")
        try (#"{"turnID":"t1","turnIndex":1,"title":"T","speakerAgentID":"a","#
            + #""speakerName":"A","prompt":"p","output":"kept","outputLabel":"l","#
            + #""routedAgentIDs":[]}"# + "\n" + #"{"turnID":"t2","outp"#)
            .write(to: url, atomically: true, encoding: .utf8)

        // Skipping the fragment is not enough — it has to leave the file.
        #expect(MultiAgentRunner.completedTurns(at: url).keys.sorted() == ["t1"])
        MultiAgentRunner.truncateTornTail(at: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        #expect(text.hasSuffix("\n"))
        #expect(!text.contains("outp\""))
        // Appending now produces a file whose every line parses.
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            #expect(
                (try? JSONDecoder().decode(
                    MultiAgentTurnResult.self, from: Data(line.utf8))) != nil)
        }
    }

    @Test("a complete record missing only its newline survives")
    func completeRecordWithoutNewlineSurvives() throws {
        // Loading before truncating LOST data, and Swift is the likelier
        // victim: the record and its newline are two separate writes. A crash
        // between them leaves complete JSON that parses — so it entered
        // `completed` (never regenerated) and was then truncated away.
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-resume-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(component: "turns.jsonl")
        try (#"{"turnID":"t1","turnIndex":1,"title":"T","speakerAgentID":"a","#
            + #""speakerName":"A","prompt":"p","output":"PRECIOUS","outputLabel":"l","#
            + #""routedAgentIDs":[]}"#)  // deliberately no trailing newline
            .write(to: url, atomically: true, encoding: .utf8)

        MultiAgentRunner.truncateTornTail(at: url)

        let kept = MultiAgentRunner.completedTurns(at: url)
        #expect(kept["t1"]?.output == "PRECIOUS")
        #expect(try String(contentsOf: url, encoding: .utf8).hasSuffix("\n"))
    }

    @Test("an intact file is left alone")
    func intactFileIsUntouched() throws {
        let root = URL(filePath: NSTemporaryDirectory())
            .appending(component: "steerlab-resume-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(component: "turns.jsonl")
        let original = #"{"turnID":"t1","turnIndex":1,"title":"T","speakerAgentID":"a","#
            + #""speakerName":"A","prompt":"p","output":"o","outputLabel":"l","#
            + #""routedAgentIDs":[]}"# + "\n"
        try original.write(to: url, atomically: true, encoding: .utf8)

        MultiAgentRunner.truncateTornTail(at: url)
        #expect(try String(contentsOf: url, encoding: .utf8) == original)
    }
}
