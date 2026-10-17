import Foundation
import Testing

@testable import ExperimentKit

/// What one paired difference of a stored effect row is, as the Results views
/// settle it (`RunResults.resolveUnits`).
///
/// Before SteerLab 0.9.7 the Mac engine paired every response of a
/// multi-sample run with the baseline response to the same item and seed,
/// counted responses in `n`, and stamped no unit, so five seeds of one prompt
/// read as "5 paired items" with an interval; and fewer pairs than items
/// (after an exclusion, missing values, or in a subgroup) does not show a row
/// counts items either. The rule is the Python reader's
/// (`results_export.resolve_units`), and the shared fixture
/// `Tests/Fixtures/cross-engine/effect-units.json` holds that reader's
/// answers; the hand-written answers that prove the rule right are in
/// `Server/tests/test_effect_unit_interpretation.py`. Server twin:
/// `Server/tests/test_results_export.py`; explorer twin:
/// `results-explorer/test/effectUnits.test.ts`.
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
        let unitRecordsUnavailable: String
        let unitRecordsPartial: String
        let responseUnitExplanation: String
        let unitLines: [String: String]
    }

    private struct Case: Decodable {
        let label: String
        let generations: [String]
        /// The analysis directory's config.json, as far as the rule reads it.
        let config: Config?
        let stampedUnit: String?
        let effectRows: [Row]
        let expected: Expected

        var pairing: RunResults.EnginePairing? {
            RunResults.enginePairing(substrate: config?.substrate, appVersion: config?.appVersion)
        }
    }

    private struct Config: Decodable {
        let substrate: String?
        let appVersion: String?
    }

    private struct Samples: Decodable, Equatable {
        let responses: Int
        let readouts: Int
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
        let enginePairing: String?
        let pairedItems: [String: Int]
        let samplesPerItem: [String: Samples]
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
            n: Double(stored.n).map { Int($0) } ?? 0, storedN: Double(stored.n), meanDiff: meanDiff,
            ciLower: ciLower, ciUpper: ciUpper, wilcoxonW: nil, wilcoxonP: nil,
            adjustedP: nil, correction: nil, modality: nil,
            recordedUnit: stored.unit.isEmpty ? nil : stored.unit,
            stratifyBy: stored.stratifyBy)
    }

    /// `samplesPerItem` in the fixture's shape.
    private func samples(_ records: RunResults.PairedItems) -> [String: Samples] {
        records.samplesPerItem.mapValues { Samples(responses: $0.responses, readouts: $0.readouts) }
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
            "the same item-level rows with no engine record",
            "the reviewer's exclusion case, made by the Mac engine before 0.9.7",
            "the reviewer's exclusion case, with no engine version recorded",
            "a subgroup row within the run's items",
            "an ordinal readout row made by the Mac engine before 0.9.7",
            "a condition the records cannot place", "records a reader leaves out",
            "a unit the analysis stamped",
        ] {
            #expect(labels.contains(label), "\(label)")
        }
    }

    /// Every case, row by row: the paired items and samples per item counted
    /// from the raw lines (in memory and streamed from a file), the engine's
    /// pairing read from the case's config.json, and each row's unit, its
    /// source, its paired items, the count the sentence states, and the
    /// minimum-pairs rule — all as the Python reader answers them.
    @Test func everyCaseSettlesAsThePythonReaderSettlesIt() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for value in try fixture().cases {
            let text = value.generations.joined(separator: "\n") + "\n"
            let records = RunResults.pairedRecords(fromJSONL: text, complete: true)
            #expect(records.counts == value.expected.pairedItems, "\(value.label)")
            #expect(samples(records) == value.expected.samplesPerItem, "\(value.label)")
            #expect(RunResults.pairedItems(fromJSONL: text) == value.expected.pairedItems, "\(value.label)")
            let file = root.appending(component: "generations.jsonl")
            try Data(text.utf8).write(to: file)
            #expect(RunResults.pairedRecords(generationsAt: file) == records, "\(value.label)")
            #expect(value.pairing?.rawValue == value.expected.enginePairing, "\(value.label)")

            #expect(value.effectRows.count == value.expected.rows.count, "\(value.label)")
            for (stored, expected) in zip(value.effectRows, value.expected.rows) {
                var settled = row(stored)
                settled.unit = RunResults.effectUnit(
                    condition: settled.condition, storedN: settled.storedN ?? 0,
                    recordedUnit: settled.recordedUnit, stampedUnit: value.stampedUnit,
                    stratifyBy: stored.stratifyBy, metric: stored.endpoint, pairing: value.pairing,
                    pairedItems: records)
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
            let records = RunResults.pairedRecords(
                fromJSONL: value.generations.joined(separator: "\n"), complete: true)
            let settled = RunResults.resolveUnits(
                rows, stampedUnit: value.stampedUnit, pairing: value.pairing
            ) { records }.rows
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
        for (source, unit, key) in [
            (RunResults.EffectUnit.Source.oneSamplePerItem, "item", "one_sample_per_item"),
            (.analysisEngine, "item", "analysis_engine_item"),
            (.analysisEngine, "response", "analysis_engine_response"),
            (.inferredFromRecords, "response", "inferred_from_records"),
            (.notEstablished, "unknown", "not_established"),
        ] {
            var settled = row(
                Row(condition: "formal", endpoint: "wordCount", n: "4", unit: "",
                    stratifyBy: "pooled", stratum: ""))
            settled.unit = RunResults.EffectUnit(unit: unit, source: source)
            let line = try #require(value.wording.unitLines[key])
            #expect(EffectNarrative.unstampedUnitLines[key] == line)
            #expect(
                EffectNarrative.unitLines(rows: [settled], records: records)
                    == ["Unit of analysis: " + line])
            // What this reading saw of the records is added to the
            // not-established line only.
            let partial = RunResults.PairedItems(counts: ["formal": 4], complete: false)
            for (seen, clause) in [(nil, value.wording.unitRecordsUnavailable), (partial, value.wording.unitRecordsPartial)] {
                #expect(
                    EffectNarrative.unitLines(rows: [settled], records: seen)
                        == ["Unit of analysis: " + line + (key == "not_established" ? " " + clause : "")])
            }
        }
        #expect(EffectNarrative.unitRecordsUnavailable == value.wording.unitRecordsUnavailable)
        #expect(EffectNarrative.unitRecordsPartial == value.wording.unitRecordsPartial)
        // One Mac analysis from before 0.9.7 holds item rows (its ordinal
        // readouts) and response rows: one line for each.
        var ordinal = row(
            Row(condition: "formal", endpoint: "ordinalPosition", n: "4", unit: "",
                stratifyBy: "pooled", stratum: ""))
        ordinal.unit = RunResults.EffectUnit(unit: "item", source: .analysisEngine)
        var words = ordinal
        words.unit = RunResults.EffectUnit(unit: "response", source: .analysisEngine)
        #expect(EffectNarrative.unitLines(rows: [ordinal, words], records: records) == [
            "Unit of analysis for the rows marked item: "
                + (value.wording.unitLines["analysis_engine_item"] ?? ""),
            "Unit of analysis for the rows marked response: "
                + (value.wording.unitLines["analysis_engine_response"] ?? ""),
        ])
        // A recorded unit is defined, not only named.
        for unit in ["item", "transcript"] {
            var settled = row(
                Row(condition: "formal", endpoint: "wordCount", n: "4", unit: unit,
                    stratifyBy: "pooled", stratum: ""))
            settled.unit = RunResults.EffectUnit(unit: unit, source: .recorded)
            let line = try #require(value.wording.unitLines["recorded_" + unit])
            #expect(
                EffectNarrative.unitLines(rows: [settled], records: records)
                    == ["Unit of analysis: " + line])
        }
    }

    // MARK: - The sentence

    private func settled(_ label: String, row index: Int = 0) throws -> RunResults.EffectSizeRow {
        let value = try fixtureCase(label)
        let stored = value.effectRows[index]
        var settled = row(stored)
        settled.unit = RunResults.effectUnit(
            condition: settled.condition, storedN: settled.storedN ?? 0, recordedUnit: settled.recordedUnit,
            stampedUnit: value.stampedUnit, stratifyBy: stored.stratifyBy, metric: stored.endpoint,
            pairing: value.pairing,
            pairedItems: RunResults.pairedRecords(
                fromJSONL: value.generations.joined(separator: "\n"), complete: true))
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
    /// at most four items, with the stored interval and what it is not.
    @Test func fourItemsTwoSeedsReadsAsResponsesFromItems() throws {
        let row = try settled("four items, two seeds")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 8 paired responses from at most 4 items (95% CI 0.5 to 1.5)"))
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

    /// The same records analyzed by the Mac engine 0.9.7, whose config.json
    /// says so: items.
    @Test func aCurrentItemLevelRowIsItems() throws {
        let row = try settled("a current item-level analysis of four items, two seeds")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("by +1 across 4 paired items (95% CI 0.5 to 1.5) — "))
        #expect(!sentence.contains("These are responses") && !sentence.contains("did not record"))
        #expect(EffectNarrative.unitLabel(row) == "item (from the engine record)")
    }

    /// The same four pairs with nothing to say which engine made them could
    /// be four items, or four responses left by exclusions or missing values:
    /// the sentence says the analysis did not record which.
    @Test func fewerPairsThanItemsIsNotCalledItems() throws {
        let row = try settled("the same item-level rows with no engine record")
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 4 pairs (95% CI 0.5 to 1.5)"))
        #expect(sentence.hasSuffix(
            " The analysis that made this row did not record whether its pairs count items or responses."))
        #expect(!sentence.contains("paired item"))
        #expect(EffectNarrative.unitLabel(row) == "unknown")
    }

    /// The external reviewer's case: ten items, five samples, one item kept
    /// by an exclusion; the Mac engine before 0.9.7 stored five responses.
    @Test func theReviewersExclusionCaseIsResponses() throws {
        let row = try settled("the reviewer's exclusion case, made by the Mac engine before 0.9.7")
        #expect(row.unit == RunResults.EffectUnit(unit: "response", source: .analysisEngine, pairedItems: 10))
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 5 paired responses from at most 10 items"))
        #expect(sentence.hasSuffix(" " + EffectNarrative.responseCaveat + "."))
        #expect(EffectNarrative.unitLabel(row) == "response (from the engine record)")
        let unrecorded = try settled("the reviewer's exclusion case, with no engine version recorded")
        #expect(unrecorded.unit.unit == "unknown")
        #expect(EffectNarrative.sentence(for: unrecorded, familySize: 1).contains("across 5 pairs"))
    }

    @Test func aRowNothingPlacesSaysSo() throws {
        let row = try settled("a condition the records cannot place", row: 1)
        let sentence = EffectNarrative.sentence(for: row, familySize: 1)
        #expect(sentence.contains("across 3 pairs (95% CI"))
        #expect(sentence.hasSuffix(" " + EffectNarrative.unknownUnitNote + "."))
        #expect(!sentence.contains("paired item"))
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
            counts: ["formal": 1], complete: true,
            samplesPerItem: ["baseline": .init(responses: 5, readouts: 0),
                             "formal": .init(responses: 5, readouts: 0)]))
        #expect(EffectNarrative.sentence(for: row, in: model.effectSizes ?? [])
            .contains("across 5 paired responses from 1 item (too few items"))
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: the response, not the item. "
                + EffectNarrative.responseUnitExplanation])
    }

    /// The directory's config.json, as an engine writes it.
    private func config(_ appVersion: String, substrate: String = "swift-mlx") -> String {
        #"{"schemaVersion": 4, "runType": "analyze", "substrate": "\#(substrate)", "appVersion": "\#(appVersion)"}"#
    }

    /// External review F1, end to end: ten items with five samples each, a
    /// declared exclusion that kept one item, and an analysis by the Mac
    /// engine before 0.9.7 that stored n = 5. The app reads the analysis's
    /// config.json and says five responses, not five items; with no version
    /// there, it says the analysis did not record which.
    @Test func theReviewersExclusionCaseEndToEnd() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let runs = root.appending(component: "runs")
        let source = "20261001T000000000-exp-tone-run"
        try write(
            try fixtureCase("the reviewer's exclusion case, made by the Mac engine before 0.9.7")
                .generations.joined(separator: "\n") + "\n",
            to: runs.appending(components: source, "generations.jsonl"))
        let analysis = runs.appending(component: "20261002T000000000-exp-tone-analyze")
        try write(macTable([("formal", 5, 1.5, 2.5)]), to: analysis.appending(component: "effect-sizes.csv"))
        try write(#"{"sourceRun": "\#(source)"}"#, to: analysis.appending(component: "analysis.json"))
        try write(config("swift-app 0.9.6+0a1b2c3d"), to: analysis.appending(component: "config.json"))
        let model = RunResults.load(runDirectory: analysis)
        let row = try #require(model.effectSizes?.first)
        #expect(row.unit == RunResults.EffectUnit(unit: "response", source: .analysisEngine, pairedItems: 10))
        #expect(EffectNarrative.sentence(for: row, familySize: 1)
            .contains("across 5 paired responses from at most 10 items (95% CI 1.5 to 2.5)"))
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: " + (EffectNarrative.unstampedUnitLines["analysis_engine_response"] ?? "")])

        // The same analysis with no version in its config.json.
        try FileManager.default.removeItem(at: analysis.appending(component: "config.json"))
        try write(#"{"substrate": "swift-mlx"}"#, to: analysis.appending(component: "config.json"))
        let unrecorded = try #require(RunResults.load(runDirectory: analysis).effectSizes?.first)
        #expect(unrecorded.unit == RunResults.EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: 10))
        #expect(EffectNarrative.sentence(for: unrecorded, familySize: 1).hasSuffix(
            " " + EffectNarrative.unknownUnitNote + "."))
    }

    /// Which side of 0.9.7 a Mac analysis is, read from its appVersion.
    @Test func theEnginesPairingIsReadFromConfigJSON() {
        let cases: [(String?, String?, RunResults.EnginePairing?)] = [
            ("python-hf-transformers", nil, .items),
            ("python-hf-transformers", "steerlab-server 0.1.0", .items),
            ("swift-mlx", "swift-app 0.9.7", .items),
            ("swift-mlx", "swift-app 0.9.8+0a1b2c3d", .items),
            ("swift-mlx", "swift-app 1.0.0-beta+0a1b2c3d", .items),
            ("swift-mlx", "swift-app 0.10.0", .items),
            ("swift-mlx", "swift-app 0.9.6", .responses),
            ("swift-mlx", "swift-app 0.9.0-dev+abc12345", .responses),
            ("swift-mlx", "swift-app 0.9.7-dev+abc12345", nil),
            ("swift-mlx", "swift-app (fixture)", nil),
            ("swift-mlx", nil, nil),
            ("something-else", "swift-app 0.9.8", nil),
            (nil, nil, nil),
        ]
        for (substrate, appVersion, expected) in cases {
            #expect(
                RunResults.enginePairing(substrate: substrate, appVersion: appVersion) == expected,
                "\(substrate ?? "nil") \(appVersion ?? "nil")")
        }
        #expect(RunResults.enginePairing(configData: Data(#"{"substrate": "swift-mlx", "appVersion": 97}"#.utf8)) == nil)
        #expect(RunResults.enginePairing(configData: Data("[1]".utf8)) == nil)
        #expect(RunResults.enginePairing(configData: nil) == nil)
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

        // The engine from 0.9.7: averaged within items, n = 4, and its
        // config.json says which engine made it.
        let current = runs.appending(component: "20261002T000000000-exp-tone-analyze")
        try write(macTable([("formal", 4, 1.5, 2.5)]), to: current.appending(component: "effect-sizes.csv"))
        try write(#"{"sourceRun": "\#(source)"}"#, to: current.appending(component: "analysis.json"))
        try write(config("swift-app 0.9.7+0a1b2c3d"), to: current.appending(component: "config.json"))
        let item = try #require(RunResults.load(runDirectory: current).effectSizes?.first)
        #expect(item.unit == RunResults.EffectUnit(unit: "item", source: .analysisEngine, pairedItems: 4))
        #expect(EffectNarrative.sentence(for: item, familySize: 1)
            .contains("by +2 across 4 paired items (95% CI 1.5 to 2.5) — "))

        // The engine before 0.9.7, with no config.json: every response
        // paired, n = 8, more pairs than the records' items.
        let old = runs.appending(component: "20261003T000000000-exp-tone-analyze")
        try write(macTable([("formal", 8, 1.5, 2.5)]), to: old.appending(component: "effect-sizes.csv"))
        try write("runs/\(source)/\n", to: old.appending(component: "source-run.txt"))
        let responses = try #require(RunResults.load(runDirectory: old).effectSizes?.first)
        #expect(responses.unit == RunResults.EffectUnit(
            unit: "response", source: .inferredFromRecords, pairedItems: 4))
        #expect(EffectNarrative.sentence(for: responses, familySize: 1)
            .contains("across 8 paired responses from at most 4 items (95% CI 1.5 to 2.5)"))
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
            == ["Unit of analysis: not established. The analysis that made these rows did not record "
                + "whether their pairs count items or responses, and neither its config.json nor the run's "
                + "records settle it: fewer pairs than items can mean items, or responses from fewer items. "
                + "The numbers are shown as stored; analyzing the run again with this version records the unit. "
                + "The run's records are not available here, so they could not be checked."])

        // Its config.json names the Python engine, which always paired items:
        // that settles it without the records.
        try write(config("steerlab-server 0.9.6+0a1b2c3d", substrate: "python-hf-transformers"),
                  to: analysis.appending(component: "config.json"))
        #expect(RunResults.load(runDirectory: analysis).effectSizes?.first?.unit
            == RunResults.EffectUnit(unit: "item", source: .analysisEngine))
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
        let records = RunResults.localPairedItems(runDirectory: run, artifacts: artifacts)
        #expect(records?.counts == ["formal": 4] && records?.complete == true)
        #expect(records?.samplesPerItem == [
            "baseline": RunResults.SamplesPerItem(responses: 2, readouts: 0),
            "formal": RunResults.SamplesPerItem(responses: 2, readouts: 0),
        ])
    }

    /// A remote preview holds only the head of the records: its counts are
    /// lower bounds. Expected values written by hand:
    ///
    /// | config.json        | row        | unit, source                  |
    /// |--------------------|------------|-------------------------------|
    /// | none               | n = 2 or 4 | unknown, not established      |
    /// | Python engine      | n = 2      | item, analysis engine         |
    /// | Mac engine 0.9.6   | n = 4      | response, analysis engine, and |
    /// |                    |            | no item count: the head's two |
    /// |                    |            | items are not the most there  |
    ///
    /// The whole file, read, settles the row from the records.
    @Test func aRemoteHeadSettlesOnlyWhatItCan() throws {
        let lines = try fixtureCase("four items, two seeds").generations
        // Baseline (8 lines) and the first two items under the condition.
        let head = (lines.prefix(8) + lines[8..<12]).joined(separator: "\n")
        var artifacts = RunResults.ArtifactBytes()
        artifacts.generationsText = head
        artifacts.generationsTruncated = true
        artifacts.effectSizesText = macTable([("formal", 2, 1, 3)])
        let within = try #require(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first)
        #expect(within.unit == RunResults.EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: 2))

        artifacts.effectSizesText = macTable([("formal", 4, 1, 3)])
        let model = RunResults.remoteModel(runID: "r", artifacts: artifacts)
        let beyond = try #require(model.effectSizes?.first)
        #expect(beyond.unit == RunResults.EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: 2))
        #expect(EffectNarrative.unitLines(rows: model.effectSizes ?? [], records: model.effectUnitRecords)
            == ["Unit of analysis: " + (EffectNarrative.unstampedUnitLines["not_established"] ?? "")
                + " Only the first part of the run's records was read here; reading the whole file may settle it."])

        // A Python-engine analysis, as a cluster run is: items, whatever the
        // head holds.
        artifacts.configData = Data(config("steerlab-server 0.9.6", substrate: "python-hf-transformers").utf8)
        artifacts.effectSizesText = macTable([("formal", 2, 1, 3)])
        #expect(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first?.unit
            == RunResults.EffectUnit(unit: "item", source: .analysisEngine, pairedItems: 2))

        // The Mac engine before 0.9.7: the head already shows two responses
        // per item under both arms, so the row paired responses; how many
        // items they come from the head cannot say.
        artifacts.configData = Data(config("swift-app 0.9.6").utf8)
        artifacts.effectSizesText = macTable([("formal", 4, 1, 3)])
        let old = try #require(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first)
        #expect(old.unit == RunResults.EffectUnit(unit: "response", source: .analysisEngine, pairedItems: nil))
        #expect(EffectNarrative.countPhrase(old) == "4 paired responses")
        #expect(!EffectNarrative.hasTooFewPairs(old))

        // The whole file, read, with no config.json: four pairs from four
        // items sampled twice could be items or responses, and eight are
        // more pairs than items.
        artifacts.configData = nil
        artifacts.generationsText = lines.joined(separator: "\n")
        artifacts.generationsTruncated = false
        #expect(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first?.unit
            == RunResults.EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: 4))
        artifacts.effectSizesText = macTable([("formal", 8, 1, 3)])
        #expect(RunResults.remoteModel(runID: "r", artifacts: artifacts).effectSizes?.first?.unit
            == RunResults.EffectUnit(unit: "response", source: .inferredFromRecords, pairedItems: 4))
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

    /// The unit rule compares the stored count, as the Python reader does
    /// (8.5 pairs of 8 items are paired responses); the minimum-pairs rule
    /// reads its whole number toward zero, as the Python page reads the stored
    /// value (2.5 is fewer than three, 0.5 is fewer than one).
    @Test func aFractionalCountIsComparedAsStoredAndCountedTowardZero() {
        let records = RunResults.PairedItems(counts: ["formal": 8], complete: true)
        let unit = RunResults.effectUnit(condition: "formal", storedN: 8.5, recordedUnit: nil, stampedUnit: nil, pairedItems: records)
        #expect(unit.unit == "response")
        #expect(pairCount(2.5) == 2 && pairCount(0.5) == 0 && pairCount(3) == 3)
        #expect(pairCount(.nan) == 0 && pairCount(-4) == 0 && pairCount(1e300) == Int.max)
    }

    /// A string that opens with a combining mark beside a NaN elsewhere on the
    /// line: Python reads the line, so the count does too.
    @Test func aCombiningMarkAtAStringsStartDoesNotHideTheLine() {
        let lines = [
            #"{"condition": "baseline", "promptID": "i", "output": "a"}"#,
            "{\"condition\": \"formal\", \"promptID\": \"i\", \"output\": \"\u{0301}x\", \"v\": NaN}",
        ].joined(separator: "\n")
        #expect(RunResults.pairedItems(fromJSONL: lines) == ["formal": 1])
        #expect(RunResults.PairedItemCounter.withoutNonFinite("{\"a\": \"\u{0301}NaN\", \"b\": NaN}")
            == "{\"a\": \"\u{0301}NaN\", \"b\": null}")
    }
}
