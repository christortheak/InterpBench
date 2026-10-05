import Foundation
import Testing

@testable import ExperimentKit

/// What one paired difference of a stored effect row is, as the Results views
/// settle it (`RunResults.resolveUnits`).
///
/// Before SteerLab 0.9.7 the Mac engine paired every response of a
/// multi-sample run with the baseline response to the same item and seed,
/// counted responses in `n`, and stamped no unit, so five seeds of one prompt
/// read as "5 paired items" with an interval. The rule is the Python
/// reader's (`results_export.resolve_units`), and the shared fixture
/// `Tests/Fixtures/cross-engine/effect-units.json` holds that reader's
/// answers. Server twin: `Server/tests/test_results_export.py`; explorer
/// twin: `results-explorer/test/effectUnits.test.ts`.
///
/// Every test binds an explicit temporary workspace.
struct EffectUnitTests {

    // MARK: - The shared fixture

    private struct Fixture: Decodable {
        let minimumPairs: Int
        let wording: Wording
        let cases: [Case]
    }

    private struct Wording: Decodable {
        let responseCaveat: String
        let unknownNote: String
        let responseUnitExplanation: String
        let unitLines: [String: String]
    }

    private struct Case: Decodable {
        let label: String
        let generations: [String]
        let stampedUnit: String?
        let effectRows: [Row]
        let expected: Expected
    }

    /// One stored row: `n` is the cell's text, "" when the file gave none.
    private struct Row: Decodable {
        let condition: String
        let endpoint: String
        let n: String
        let unit: String
        let stratifyBy: String
        let stratum: String
    }

    private struct Expected: Decodable {
        let pairedItems: [String: Int]
        let rows: [ExpectedRow]
    }

    private struct ExpectedRow: Decodable {
        let condition: String
        let endpoint: String
        let stratifyBy: String
        let stratum: String
        let unit: String
        let unitSource: String
        let pairedItems: Int?
        let count: String?
        let tooFew: Bool
    }

    private var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func fixture() throws -> Fixture {
        try JSONDecoder().decode(
            Fixture.self,
            from: Data(
                contentsOf: repository.appending(
                    path: "Tests/Fixtures/cross-engine/effect-units.json")))
    }

    private func fixtureCase(_ label: String) throws -> Case {
        try #require(try fixture().cases.first { $0.label == label })
    }

    private func row(
        _ stored: Row, meanDiff: Double = 1, ciLower: Double = 0.5, ciUpper: Double = 1.5
    ) -> RunResults.EffectSizeRow {
        RunResults.EffectSizeRow(
            condition: stored.condition, metric: stored.endpoint,
            n: Double(stored.n).map { Int($0) } ?? 0, meanDiff: meanDiff,
            ciLower: ciLower, ciUpper: ciUpper, wilcoxonW: nil, wilcoxonP: nil,
            adjustedP: nil, correction: nil, modality: nil,
            recordedUnit: stored.unit.isEmpty ? nil : stored.unit)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "effect-units-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func theFixtureCarriesTheCasesTheRuleIsDefinedBy() throws {
        let labels = try fixture().cases.map(\.label)
        for label in [
            "one prompt, five seeds", "four items, two seeds",
            "a current item-level analysis of four items, two seeds",
            "a condition the records cannot place", "records a reader leaves out",
            "a unit the analysis stamped",
        ] {
            #expect(labels.contains(label), "\(label)")
        }
    }

    /// Every case, row by row: the paired items counted from the raw lines
    /// (in memory and streamed from a file), and each row's unit, its
    /// source, its paired items, the count the sentence states, and the
    /// minimum-pairs rule — all as the Python reader answers them.
    @Test func everyCaseSettlesAsThePythonReaderSettlesIt() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for value in try fixture().cases {
            let text = value.generations.joined(separator: "\n") + "\n"
            let counted = RunResults.pairedItems(fromJSONL: text)
            #expect(counted == value.expected.pairedItems, "\(value.label)")
            let file = root.appending(component: "generations.jsonl")
            try Data(text.utf8).write(to: file)
            #expect(
                RunResults.pairedItems(generationsAt: file) == value.expected.pairedItems,
                "\(value.label)")

            #expect(value.effectRows.count == value.expected.rows.count, "\(value.label)")
            for (stored, expected) in zip(value.effectRows, value.expected.rows) {
                var settled = row(stored)
                settled.unit = RunResults.effectUnit(
                    condition: settled.condition, n: settled.n,
                    recordedUnit: settled.recordedUnit, stampedUnit: value.stampedUnit,
                    pairedItems: RunResults.PairedItems(counts: counted, complete: true))
                let where_ = "\(value.label): \(stored.stratifyBy) \(stored.stratum) \(stored.endpoint)"
                #expect(settled.unit.unit == expected.unit, "\(where_)")
                #expect(settled.unit.source.rawValue == expected.unitSource, "\(where_)")
                #expect(settled.unit.pairedItems == expected.pairedItems, "\(where_)")
                #expect(EffectNarrative.countPhrase(settled) == expected.count, "\(where_)")
                #expect(EffectNarrative.hasTooFewPairs(settled) == expected.tooFew, "\(where_)")
            }
        }
    }

    /// The pooled rows read from an effect-sizes.csv take the same units: the
    /// table reader keeps the `unit` column and the model settles from it.
    @Test func pooledRowsReadFromTheTableSettleTheSameWay() throws {
        for value in try fixture().cases {
            var lines = ["condition,metric,n,meanDiff,ciLower,ciUpper,stratifyBy,stratum,unit"]
            for stored in value.effectRows {
                lines.append(
                    [stored.condition, stored.endpoint, stored.n, "1.0", "0.5", "1.5",
                     stored.stratifyBy, stored.stratum, stored.unit].joined(separator: ","))
            }
            let rows = try #require(RunResults.effectSizes(fromCSV: lines.joined(separator: "\n")))
            let counted = RunResults.pairedItems(
                fromJSONL: value.generations.joined(separator: "\n"))
            let settled = RunResults.resolveUnits(rows, stampedUnit: value.stampedUnit) {
                RunResults.PairedItems(counts: counted, complete: true)
            }.rows
            let pooled = value.expected.rows.filter { $0.stratifyBy == "pooled" }
            #expect(settled.map(\.unit.unit) == pooled.map(\.unit), "\(value.label)")
            #expect(
                settled.map(\.unit.source.rawValue) == pooled.map(\.unitSource), "\(value.label)")
        }
    }

    @Test func theWordsAreThePythonReaders() throws {
        let value = try fixture()
        #expect(EffectNarrative.minimumPairsForInterval == value.minimumPairs)
        #expect(EffectNarrative.responseCaveat == value.wording.responseCaveat)
        #expect(EffectNarrative.unknownUnitNote == value.wording.unknownNote)
        #expect(EffectNarrative.responseUnitExplanation == value.wording.responseUnitExplanation)
        let records = RunResults.PairedItems(counts: ["formal": 4], complete: true)
        for (source, unit) in [
            (RunResults.EffectUnit.Source.engineDefault, "item"),
            (.inferredFromRecords, "response"), (.notEstablished, "unknown"),
        ] {
            var settled = row(
                Row(condition: "formal", endpoint: "wordCount", n: "4", unit: "",
                    stratifyBy: "pooled", stratum: ""))
            settled.unit = RunResults.EffectUnit(unit: unit, source: source)
            let line = try #require(value.wording.unitLines[source.rawValue])
            #expect(
                EffectNarrative.unitLines(rows: [settled], records: records)
                    == ["Unit of analysis: " + line])
        }
    }

    // MARK: - The sentence

    private func settled(_ label: String, row index: Int = 0) throws -> RunResults.EffectSizeRow {
        let value = try fixtureCase(label)
        var settled = row(value.effectRows[index])
        settled.unit = RunResults.effectUnit(
            condition: settled.condition, n: settled.n, recordedUnit: settled.recordedUnit,
            stampedUnit: value.stampedUnit,
            pairedItems: RunResults.PairedItems(
                counts: RunResults.pairedItems(
                    fromJSONL: value.generations.joined(separator: "\n")),
                complete: true))
        return settled
    }

    /// Five seeds of one prompt are five responses from ONE item: not five
    /// items, and too few items for an interval.
    @Test func onePromptFiveSeedsDoesNotReadAsItems() throws {
        let row = try settled("one prompt, five seeds")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains(
            "across 5 paired responses from 1 item (too few items for a confidence "
                + "interval; at least 3 are needed) — with so few items this "
                + "describes these items only, and is not a test."))
        #expect(sentence.hasSuffix(" " + EffectNarrative.responseCaveat + "."))
        #expect(!sentence.contains("paired item ") && !sentence.contains("paired items"))
        #expect(!sentence.contains("95% CI"))
        #expect(EffectNarrative.hasTooFewPairs(row))
        #expect(!EffectNarrative.hasReportableInterval(row))
        #expect(EffectNarrative.unitLabel(row) == "response (from the records)")
    }

    /// Four items with two seeds each, n = 8: eight paired responses from
    /// four items, with the stored interval and what it is not.
    @Test func fourItemsTwoSeedsReadsAsResponsesFromItems() throws {
        let row = try settled("four items, two seeds")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 8 paired responses from 4 items (95% CI 0.5 to 1.5)"))
        #expect(sentence.hasSuffix(" " + EffectNarrative.responseCaveat + "."))
        #expect(!EffectNarrative.hasTooFewPairs(row))
        #expect(EffectNarrative.hasReportableInterval(row))
    }

    /// Many responses from two items are still two items: too few.
    @Test func manyResponsesFromTwoItemsAreTooFew() throws {
        let row = try settled("two items, four seeds")
        #expect(row.n == 8)
        #expect(EffectNarrative.hasTooFewPairs(row))
        #expect(EffectNarrative.tooFewCaption([row])?.contains("counts its items") == true)
    }

    /// The same records analyzed the current way read exactly as before.
    @Test func aCurrentItemLevelRowIsUnchanged() throws {
        let row = try settled("a current item-level analysis of four items, two seeds")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("by +1 across 4 paired items (95% CI 0.5 to 1.5) — "))
        #expect(!sentence.contains("These are responses") && !sentence.contains("not established"))
        #expect(EffectNarrative.unitLabel(row) == "item (default)")
    }

    @Test func aRowNothingPlacesSaysSo() throws {
        let row = try settled("a condition the records cannot place", row: 1)
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 3 pairs (95% CI"))
        #expect(sentence.hasSuffix(" " + EffectNarrative.unknownUnitNote + "."))
        #expect(!sentence.contains("item"))
    }

    // MARK: - The Results model, end to end

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    /// The Mac engine's own effect table, in its dialect.
    private func macTable(_ rows: [(condition: String, n: Int, low: Double, high: Double)]) -> String {
        (["condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction"]
            + rows.map { "\($0.condition),wordCount,\($0.n),2.0,\($0.low),\($0.high),0,0.0625,," })
            .joined(separator: "\n") + "\n"
    }

    /// A run made before 0.9.7, with its effect table beside its records:
    /// the app reads the row as five responses from one item.
    @Test func anOldRunsOwnTableReadsAsResponses() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/20261001T000000000-exp-tone-run")
        let value = try fixtureCase("one prompt, five seeds")
        try write(value.generations.joined(separator: "\n") + "\n",
                  to: run.appending(component: "generations.jsonl"))
        try write(macTable([("formal", 5, 2, 2)]), to: run.appending(component: "effect-sizes.csv"))

        let model = RunResults.load(runDirectory: run)
        let row = try #require(model.effectSizes?.first)
        #expect(row.unit == RunResults.EffectUnit(
            unit: "response", source: .inferredFromRecords, pairedItems: 1))
        #expect(model.effectUnitRecords == RunResults.PairedItems(
            counts: ["formal": 1], complete: true))
        #expect(EffectNarrative.sentence(for: row, in: model.effectSizes ?? [])
            .contains("across 5 paired responses from 1 item (too few items"))
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: the response, not the item. "
                + EffectNarrative.responseUnitExplanation])
    }

    /// An analysis directory holds no records: its rows are settled against
    /// the run it analyzed, named by analysis.json or source-run.txt.
    @Test func anAnalysisIsSettledAgainstTheRunItAnalyzed() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let runs = root.appending(component: "runs")
        let source = "20261001T000000000-exp-tone-run"
        try write(
            try fixtureCase("four items, two seeds").generations.joined(separator: "\n") + "\n",
            to: runs.appending(components: source, "generations.jsonl"))

        // The current engine: averaged within items, n = 4. Unchanged.
        let current = runs.appending(component: "20261002T000000000-exp-tone-analyze")
        try write(macTable([("formal", 4, 1.5, 2.5)]), to: current.appending(component: "effect-sizes.csv"))
        try write(#"{"sourceRun": "\#(source)"}"#, to: current.appending(component: "analysis.json"))
        let item = try #require(RunResults.load(runDirectory: current).effectSizes?.first)
        #expect(item.unit == RunResults.EffectUnit(unit: "item", source: .engineDefault, pairedItems: 4))
        #expect(EffectNarrative.sentence(for: item, familySize: 1)
            .contains("by +2 across 4 paired items (95% CI 1.5 to 2.5) — "))

        // The engine before 0.9.7: every response paired, n = 8.
        let old = runs.appending(component: "20261003T000000000-exp-tone-analyze")
        try write(macTable([("formal", 8, 1.5, 2.5)]), to: old.appending(component: "effect-sizes.csv"))
        try write("runs/\(source)/\n", to: old.appending(component: "source-run.txt"))
        let responses = try #require(RunResults.load(runDirectory: old).effectSizes?.first)
        #expect(responses.unit == RunResults.EffectUnit(
            unit: "response", source: .inferredFromRecords, pairedItems: 4))
        #expect(EffectNarrative.sentence(for: responses, familySize: 1)
            .contains("across 8 paired responses from 4 items (95% CI 1.5 to 2.5)"))
    }

    /// A stamped unit settles every row, and the run's records are not read.
    @Test func aStampedUnitIsUsedWithoutReadingRecords() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let analysis = root.appending(path: "runs/20261002T000000000-exp-panel-analyze")
        try write(macTable([("formal", 4, 1, 3)]), to: analysis.appending(component: "effect-sizes.csv"))
        try write(#"{"unitOfAnalysis": "transcript"}"#,
                  to: analysis.appending(component: "unit-of-analysis.json"))
        try write(#"{"sourceRun": "20261001T000000000-exp-panel-run"}"#,
                  to: analysis.appending(component: "analysis.json"))
        let model = RunResults.load(runDirectory: analysis)
        let row = try #require(model.effectSizes?.first)
        #expect(row.unit == RunResults.EffectUnit(unit: "transcript", source: .recorded))
        #expect(model.stampedUnit == "transcript" && model.effectUnitRecords == nil)
        #expect(EffectNarrative.sentence(for: row, familySize: 1).contains("across 4 paired transcripts"))
    }

    /// An analysis whose source run is not on this machine cannot be
    /// settled, and says why.
    @Test func anAnalysisWithoutItsRunIsNotEstablished() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let analysis = root.appending(path: "runs/20261002T000000000-exp-tone-analyze")
        try write(macTable([("formal", 4, 1, 3)]), to: analysis.appending(component: "effect-sizes.csv"))
        try write(#"{"sourceRun": "20261001T000000000-exp-tone-run"}"#,
                  to: analysis.appending(component: "analysis.json"))
        let model = RunResults.load(runDirectory: analysis)
        let row = try #require(model.effectSizes?.first)
        #expect(row.unit == .unresolved)
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: not established. The analysis did not stamp it, and the "
                + "run's records are not available here."])
    }

    /// A local run larger than the app's preview read is counted whole.
    @Test func aTruncatedLocalReadIsCountedWhole() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/20261001T000000000-exp-tone-run")
        let lines = try fixtureCase("four items, two seeds").generations
        try write(lines.joined(separator: "\n") + "\n", to: run.appending(component: "generations.jsonl"))
        var artifacts = RunResults.ArtifactBytes()
        // The preview held only the baseline's records.
        artifacts.generationsText = lines.prefix(8).joined(separator: "\n")
        artifacts.generationsTruncated = true
        #expect(RunResults.localPairedItems(runDirectory: run, artifacts: artifacts)
            == RunResults.PairedItems(counts: ["formal": 4], complete: true))
    }

    /// A remote preview holds only the head of the records. A row within the
    /// items the head pairs is item-level whatever the rest holds; a row
    /// beyond them cannot be settled from a head.
    @Test func aRemoteHeadSettlesOnlyWhatItCan() throws {
        let lines = try fixtureCase("four items, two seeds").generations
        // Baseline (8 lines) and the first two items under the condition.
        let head = (lines.prefix(8) + lines[8..<12]).joined(separator: "\n")
        var artifacts = RunResults.ArtifactBytes()
        artifacts.generationsText = head
        artifacts.generationsTruncated = true
        artifacts.effectSizesText = macTable([("formal", 2, 1, 3)])
        let within = try #require(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first)
        #expect(within.unit == RunResults.EffectUnit(unit: "item", source: .engineDefault, pairedItems: 2))

        artifacts.effectSizesText = macTable([("formal", 4, 1, 3)])
        let model = RunResults.remoteModel(runID: "r", artifacts: artifacts)
        let beyond = try #require(model.effectSizes?.first)
        #expect(beyond.unit == RunResults.EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: 2))
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: not established. The analysis did not stamp it, and only "
                + "the first part of the run's records was read here, which does not settle it."])

        // The whole file, read: the same row is item-level.
        artifacts.generationsText = lines.joined(separator: "\n")
        artifacts.generationsTruncated = false
        #expect(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first?.unit
            == RunResults.EffectUnit(unit: "item", source: .engineDefault, pairedItems: 4))
    }

    /// The streamed count reads lines that cross the reader's chunks.
    @Test func aLargeFileIsCountedAcrossChunks() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let padding = String(repeating: "é", count: 3_000)
        var lines: [String] = []
        for condition in ["baseline", "formal"] {
            for item in 0..<1_500 where condition == "baseline" || item % 3 != 0 {
                lines.append(#"{"condition": "\#(condition)", "promptID": "item-\#(item)", "output": "\#(padding)"}"#)
            }
        }
        let text = lines.joined(separator: "\n") + "\n"
        #expect(text.utf8.count > 12 << 20)
        let file = root.appending(component: "generations.jsonl")
        try Data(text.utf8).write(to: file)
        #expect(RunResults.pairedItems(generationsAt: file) == ["formal": 1_000])
        #expect(RunResults.pairedItems(fromJSONL: text) == ["formal": 1_000])
    }
}
