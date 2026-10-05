import Foundation
import Testing

@testable import ExperimentKit

/// Paired effects when a study samples several responses per item.
///
/// The unit of analysis is the ITEM: each (condition, item) cell is averaged
/// over its samples, and the treatment mean pairs to the same item's
/// baseline mean — by promptID, never by seed. A derived seed includes the
/// condition name (`StudySampling.deriveSeed`), so the two records of a pair
/// carry different seeds by design.
///
/// The measurements come from the shared fixture
/// `Tests/Fixtures/cross-engine/sampled-effect-pairing.json`, which holds
/// one set of measurements under both seed policies together with the rows
/// the Python engine's `analyze` writes for them. Server twin:
/// `Server/tests/test_sampled_effect_pairing.py`.
///
/// Every test binds an explicit temporary workspace; none touches
/// `rootOverride`.
struct SampledEffectPairingTests {

    // MARK: - The shared fixture

    private struct Fixture: Decodable {
        let experimentHash: String
        let cases: [Case]
    }

    private struct Case: Decodable {
        let label: String
        let seedPolicy: String
        let samplesPerItem: Int
        let manifestSeeds: [UInt64]
        let records: [Record]
        let effectRows: [PythonRow]
    }

    /// One sampled record, in the keys both engines write.
    private struct Record: Codable {
        let condition: String
        let seed: UInt64
        let seedPolicy: String
        let sampleIndex: Int
        let promptIndex: Int
        let promptID: String
        let arm: String
        let wordCount: Int
        let distinct2: Double
    }

    /// One effect-sizes.csv row as the Python engine writes it (its column
    /// names; an empty cell is "" for text and null for a number).
    private struct PythonRow: Decodable {
        let condition: String
        let endpoint: String
        let n: Int
        let deltaMean: Double
        let wilcoxonW: Double?
        let wilcoxonP: Double?
        let adjustedP: Double?
        let correction: String
        let stratifyBy: String
        let stratum: String
        let unit: String
        let estimand: String
        let inference: String
    }

    private func fixtureCase(_ label: String) throws -> (hash: String, value: Case) {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "sampled-effect-pairing.json")
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: try Data(contentsOf: url))
        let value = try #require(fixture.cases.first { $0.label == label })
        return (fixture.experimentHash, value)
    }

    /// The manifest whose sampling policy produces the case's seeds.
    private func manifest(for value: Case) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "pairing", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.temperature = 0.7
        if value.seedPolicy == "derivedSHA256" {
            manifest.samplesPerItem = value.samplesPerItem
            manifest.seedPolicy = "derivedSHA256"
        } else {
            manifest.seeds = value.manifestSeeds
        }
        return manifest
    }

    // MARK: - The real analysis entry point, in a temporary workspace

    private struct Analysis {
        let entries: [ExperimentTasks.EffectSizeEntry]
        let csv: String
    }

    /// Writes the records as a completed run in a fresh temporary workspace,
    /// runs `experiment analyze`'s workflow over it, and reads back what it
    /// published.
    private func analyze(_ manifest: ExperimentManifest, records: [Record]) throws -> Analysis {
        let encoder = JSONEncoder()
        return try analyze(
            manifest,
            lines: try records.map { String(decoding: try encoder.encode($0), as: UTF8.self) })
    }

    private func analyze(_ manifest: ExperimentManifest, lines: [String]) throws -> Analysis {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "sampled-pairing-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/20261004T000000000Z-exp-pairing-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))

        let out = try StudyAnalysisWorkflow.analyze(
            manifest: manifest,
            repository: StudyAnalysisRepository(workspaceRoot: root, promptRoot: root),
            allowUnverifiedEpoch: false)
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: out.appending(component: "analysis.json")))
        let csv = try String(
            contentsOf: out.appending(component: "effect-sizes.csv"), encoding: .utf8)
        return Analysis(entries: report.effectSizes, csv: csv)
    }

    private func entry(
        _ entries: [ExperimentTasks.EffectSizeEntry], metric: String,
        stratifyBy: String? = nil, stratum: String? = nil
    ) throws -> ExperimentTasks.EffectSizeEntry {
        let matches = entries.filter {
            $0.metric == metric && $0.stratifyBy == stratifyBy && $0.stratum == stratum
        }
        #expect(
            matches.count == 1,
            "expected one \(stratifyBy ?? "pooled") \(stratum ?? "") \(metric) row, found \(matches.count)")
        return try #require(matches.first)
    }

    /// The normal-approximation p of a signed-rank W, written out from the
    /// textbook formula rather than through the engine: mean n(n+1)/4,
    /// variance n(n+1)(2n+1)/24 less the tie correction, continuity-corrected.
    private func twoSidedP(w: Double, n: Double, tieCorrection: Double = 0) -> Double {
        let mean = n * (n + 1) / 4
        let variance = n * (n + 1) * (2 * n + 1) / 24 - tieCorrection / 48
        let z = (w - mean + 0.5) / variance.squareRoot()
        return min(1, 1 + erf(z / 2.0.squareRoot()))
    }

    // MARK: - The fixture's seeds are the engine's own

    @Test(arguments: ["derivedSHA256", "manifestSeeds"])
    func fixtureSeedsAreTheOnesTheRunWouldDraw(label: String) throws {
        let (hash, value) = try fixtureCase(label)
        let manifest = manifest(for: value)
        #expect(StudySampling.policy(manifest) == value.seedPolicy)
        #expect(StudySampling.count(manifest) == 3)
        #expect(value.records.count == 24)
        for record in value.records {
            #expect(
                StudySampling.seed(
                    manifest, experimentHash: hash, condition: record.condition,
                    promptID: record.promptID, sampleIndex: record.sampleIndex)
                    == record.seed)
        }
        // What makes the seed unusable as a pairing key under derived seeds:
        // no treatment record shares a (seed, item) key with any baseline
        // record. Under manifest seeds every one does.
        let baselineKeys = Set(
            value.records.filter { $0.condition == "baseline" }
                .map { "\($0.seed)::\($0.promptID)" })
        let treatmentShares = value.records.filter { $0.condition == "steered" }
            .map { baselineKeys.contains("\($0.seed)::\($0.promptID)") }
        #expect(treatmentShares.count == 12)
        #expect(treatmentShares.allSatisfy { $0 == (label == "manifestSeeds") })
    }

    // MARK: - Hand-computed paired differences

    /// wordCount, three samples per cell (baseline | steered):
    ///
    ///     item-1  10 12 14 → 12  |  15 18 21 → 18   difference  +6
    ///     item-2  20 20 23 → 21  |  22 25 25 → 24   difference  +3
    ///     item-3  30 33 36 → 33  |  31 32 33 → 32   difference  −1
    ///     item-4  40 41 45 → 42  |  50 52 54 → 52   difference +10
    ///
    /// Four paired items, mean (6 + 3 − 1 + 10) / 4 = 4.5. Signed ranks of
    /// |−1| < |3| < |6| < |10| are 1, 2, 3, 4, so W− = 1, W+ = 9, W = 1.
    @Test(arguments: ["derivedSHA256", "manifestSeeds"])
    func analyzeAveragesOverSamplesAndPairsByItem(label: String) throws {
        let (_, value) = try fixtureCase(label)
        let analysis = try analyze(manifest(for: value), records: value.records)
        let entries = analysis.entries

        // Pooled: one difference per ITEM — n = 4, not 12 sample pairs and
        // not an absent row.
        let pooled = try entry(entries, metric: "wordCount")
        #expect(pooled.condition == "steered")
        #expect(pooled.n == 4)
        #expect(pooled.meanDiff == 4.5)
        #expect(pooled.wilcoxonW == 1)
        let pooledP = try #require(pooled.wilcoxonP)
        #expect(abs(pooledP - twoSidedP(w: 1, n: 4)) < 1e-12)  // 0.201243
        // A family of one: the adjusted p is the raw p.
        #expect(pooled.adjustedP == pooled.wilcoxonP)
        #expect(pooled.correction == "bh")
        // Pooled rows leave unit, estimand, and inference unset: their unit
        // is the run's default, the item.
        #expect(pooled.unit == nil && pooled.estimand == nil && pooled.inference == nil)
        // The interval is a bootstrap of those four differences.
        #expect(-1 <= pooled.ciLower && pooled.ciLower <= 4.5)
        #expect(4.5 <= pooled.ciUpper && pooled.ciUpper <= 10)

        // distinct2, per-item differences +0.25, 0, +0.25, −0.5: mean 0; the
        // zero drops, the two 0.25s tie at rank 1.5, so W+ = W− = 3 and the
        // p clamps to 1.
        let distinct = try entry(entries, metric: "distinct2")
        #expect(distinct.n == 4)
        #expect(distinct.meanDiff == 0)
        #expect(distinct.wilcoxonW == 3)
        #expect(distinct.wilcoxonP == 1)

        // Strata with two items stay item-level: arm x = {item-1, item-3} →
        // (6 − 1) / 2; arm y = {item-2, item-4} → (3 + 10) / 2.
        let armX = try entry(entries, metric: "wordCount", stratifyBy: "arm", stratum: "x")
        #expect(armX.n == 2 && armX.meanDiff == 2.5)
        #expect(armX.wilcoxonW == 1)  // ranks −1 → 1, +6 → 2
        #expect(armX.wilcoxonP == 1)  // z = 0
        let armY = try entry(entries, metric: "wordCount", stratifyBy: "arm", stratum: "y")
        #expect(armY.n == 2 && armY.meanDiff == 6.5)
        #expect(armY.wilcoxonW == 0)
        let armYP = try #require(armY.wilcoxonP)
        #expect(abs(armYP - twoSidedP(w: 0, n: 2)) < 1e-12)  // 0.371093
        // Benjamini–Hochberg over the two arm rows: the smaller p doubles.
        let armYAdjusted = try #require(armY.adjustedP)
        #expect(abs(armYAdjusted - 2 * twoSidedP(w: 0, n: 2)) < 1e-12)  // 0.742187
        #expect(armX.adjustedP == 1)
        for row in [armX, armY] {
            #expect(row.unit == "item")
            #expect(row.estimand == "itemLevel")
            #expect(row.inference == "corrected")
            #expect(row.correction == "bh")
        }

        // A one-item stratum drops to the sample axis, pairing sample k to
        // baseline sample k (by sampleIndex, not by seed), and is a
        // diagnostic held out of the correction.
        let bySample: [(item: String, diffs: [Double], mean: Double, w: Double, ties: Double)] = [
            ("item-1", [5, 6, 7], 6, 0, 0),  // 15−10, 18−12, 21−14
            ("item-2", [2, 5, 2], 3, 0, 6),  // the two 2s tie
            ("item-3", [1, -1, -3], -1, 1.5, 6),  // |1| and |−1| tie
            ("item-4", [10, 11, 9], 10, 0, 0),
        ]
        for expected in bySample {
            #expect(expected.diffs.reduce(0, +) / 3 == expected.mean)
            let row = try entry(
                entries, metric: "wordCount", stratifyBy: "promptID", stratum: expected.item)
            #expect(row.n == 3, "\(expected.item)")
            #expect(row.meanDiff == expected.mean, "\(expected.item)")
            #expect(row.wilcoxonW == expected.w, "\(expected.item)")
            let p = try #require(row.wilcoxonP)
            #expect(
                abs(p - twoSidedP(w: expected.w, n: 3, tieCorrection: expected.ties)) < 1e-12,
                "\(expected.item)")
            #expect(row.unit == "sample")
            #expect(row.estimand == "withinItemSamples")
            #expect(row.inference == "diagnostic")
            #expect(row.adjustedP == nil && row.correction == nil)
        }

        // The plain-language sentence counts what the row counts: items.
        let rows = try #require(RunResults.effectSizes(fromCSV: analysis.csv))
        let sentenceRow = try #require(rows.first { $0.metric == "wordCount" })
        #expect(sentenceRow.n == 4)
        #expect(
            EffectNarrative.sentence(for: sentenceRow, in: rows)
                .contains("by +4.5 across 4 paired items"))
    }

    // MARK: - Agreement with the Python engine on the same records

    @Test(arguments: ["derivedSHA256", "manifestSeeds"])
    func analyzeAgreesWithThePythonEngineOnTheSameRecords(label: String) throws {
        let (_, value) = try fixtureCase(label)
        let entries = try analyze(manifest(for: value), records: value.records).entries

        func key(_ stratifyBy: String, _ stratum: String, _ condition: String, _ metric: String)
            -> String
        { [stratifyBy, stratum, condition, metric].joined(separator: "|") }
        // Same rows on both engines: nothing missing, nothing extra.
        #expect(
            Set(entries.map { key($0.stratifyBy ?? "pooled", $0.stratum ?? "", $0.condition, $0.metric) })
                == Set(value.effectRows.map { key($0.stratifyBy, $0.stratum, $0.condition, $0.endpoint) }))
        #expect(entries.count == value.effectRows.count)
        #expect(value.effectRows.count == 14)

        // The fixture's numbers carry the CSV's six significant digits.
        func close(_ mine: Double?, _ theirs: Double?) -> Bool {
            switch (mine, theirs) {
            case (nil, nil): return true
            case (let mine?, let theirs?): return abs(mine - theirs) <= 2e-6 * max(1, abs(theirs))
            default: return false
            }
        }
        for expected in value.effectRows {
            let name = key(
                expected.stratifyBy, expected.stratum, expected.condition, expected.endpoint)
            let matches = entries.filter {
                key($0.stratifyBy ?? "pooled", $0.stratum ?? "", $0.condition, $0.metric) == name
            }
            guard let mine = matches.first, matches.count == 1 else { continue }
            #expect(mine.n == expected.n, "n drift on \(name)")
            #expect(close(mine.meanDiff, expected.deltaMean), "mean drift on \(name)")
            #expect(close(mine.wilcoxonW, expected.wilcoxonW), "W drift on \(name)")
            #expect(close(mine.wilcoxonP, expected.wilcoxonP), "p drift on \(name)")
            #expect(close(mine.adjustedP, expected.adjustedP), "adjusted p drift on \(name)")
            // An empty CSV cell on the Python side is an unset field here.
            #expect((mine.correction ?? "") == expected.correction, "correction drift on \(name)")
            #expect((mine.unit ?? "") == expected.unit, "unit drift on \(name)")
            #expect((mine.estimand ?? "") == expected.estimand, "estimand drift on \(name)")
            #expect((mine.inference ?? "") == expected.inference, "inference drift on \(name)")
        }
    }

    // MARK: - The run's own report (report.json) pairs the same way

    @Test(arguments: ["derivedSHA256", "manifestSeeds"])
    func runReportPairsByItem(label: String) throws {
        let (hash, value) = try fixtureCase(label)
        // The rows a run keeps in memory while it generates.
        let rows = value.records.map {
            ExperimentTasks.MetricRow(
                condition: $0.condition, seed: $0.seed, promptIndex: $0.promptIndex,
                promptID: $0.promptID, wordCount: $0.wordCount,
                distinct2: Float($0.distinct2), markerDensity: [:])
        }
        let report = ExperimentTasks.report(
            experiment: manifest(for: value), experimentHash: hash,
            taskPrompts: ("items.jsonl", "ph", []), rows: rows, conditionCount: 2,
            concepts: [])
        let effects = try #require(report.effectSizes)
        #expect(effects.count == 2)
        let wordCount = try #require(effects.first { $0.metric == "wordCount" })
        #expect(wordCount.n == 4)
        #expect(wordCount.meanDiff == 4.5)
        #expect(wordCount.wilcoxonW == 1)
        let p = try #require(wordCount.wilcoxonP)
        #expect(abs(p - twoSidedP(w: 1, n: 4)) < 1e-12)
        // The per-condition block still counts every generation.
        #expect(report.conditions["steered"]?.generations == 12)
    }

    // MARK: - What must NOT change

    /// A study with one response per item gets the numbers it always got —
    /// every column, bootstrap interval included. The expected text is the
    /// output of the engine BEFORE sampled cells were averaged, captured on
    /// these exact rows (five items, shuffled prompt indices, and seeds that
    /// differ between items, so the row order feeding the bootstrap is
    /// pinned too).
    @Test func oneResponsePerItemKeepsEveryNumber() {
        var rows: [ExperimentTasks.MetricRow] = []
        let cells: [(String, UInt64, Int, Int, Int, Float, Float)] = [
            ("q-a", 7, 3, 100, 103, 0.5, 0.75), ("q-b", 3, 1, 80, 79, 0.25, 0.5),
            ("q-c", 9, 5, 60, 64, 0.5, 0.25), ("q-d", 3, 2, 120, 121, 0.75, 0.75),
            ("q-e", 7, 4, 90, 95, 0.5, 0.625),
        ]
        for (promptID, seed, index, base, steered, baseDistinct, steeredDistinct) in cells {
            rows.append(
                ExperimentTasks.MetricRow(
                    condition: "baseline", seed: seed, promptIndex: index, promptID: promptID,
                    wordCount: base, distinct2: baseDistinct, markerDensity: ["warm": 0.125]))
            rows.append(
                ExperimentTasks.MetricRow(
                    condition: "steered", seed: seed, promptIndex: index, promptID: promptID,
                    wordCount: steered, distinct2: steeredDistinct,
                    markerDensity: ["warm": 0.125 * Float(index)]))
        }
        let factors: [String: [String: String]] = [
            "q-a": ["arm": "x"], "q-b": ["arm": "y"], "q-c": ["arm": "x"],
            "q-d": ["arm": "y"], "q-e": ["arm": "x"],
        ]
        let before = """
            condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,stratifyBy,stratum,unit,estimand,inference
            steered,wordCount,5,2.4,0.4,4.2,1.5,0.1362168698445676,0.1362168698445676,bh,pooled,,,,
            steered,distinct2,5,0.075,-0.1,0.225,3.0,0.570750388058174,0.570750388058174,bh,pooled,,,,
            steered,warmMarkerDensity,5,0.25,0.1,0.4,0.0,0.10034824646229079,0.10034824646229079,bh,pooled,,,,
            steered,wordCount,1,3.0,3.0,3.0,0.0,1.0,1.0,bh,promptID,q-a,item,itemLevel,corrected
            steered,distinct2,1,0.25,0.25,0.25,0.0,1.0,1.0,bh,promptID,q-a,item,itemLevel,corrected
            steered,warmMarkerDensity,1,0.25,0.25,0.25,0.0,1.0,1.0,bh,promptID,q-a,item,itemLevel,corrected
            steered,wordCount,1,-1.0,-1.0,-1.0,0.0,1.0,1.0,bh,promptID,q-b,item,itemLevel,corrected
            steered,distinct2,1,0.25,0.25,0.25,0.0,1.0,1.0,bh,promptID,q-b,item,itemLevel,corrected
            steered,warmMarkerDensity,1,0.0,0.0,0.0,,,,bh,promptID,q-b,item,itemLevel,corrected
            steered,wordCount,1,4.0,4.0,4.0,0.0,1.0,1.0,bh,promptID,q-c,item,itemLevel,corrected
            steered,distinct2,1,-0.25,-0.25,-0.25,0.0,1.0,1.0,bh,promptID,q-c,item,itemLevel,corrected
            steered,warmMarkerDensity,1,0.5,0.5,0.5,0.0,1.0,1.0,bh,promptID,q-c,item,itemLevel,corrected
            steered,wordCount,1,1.0,1.0,1.0,0.0,1.0,1.0,bh,promptID,q-d,item,itemLevel,corrected
            steered,distinct2,1,0.0,0.0,0.0,,,,bh,promptID,q-d,item,itemLevel,corrected
            steered,warmMarkerDensity,1,0.125,0.125,0.125,0.0,1.0,1.0,bh,promptID,q-d,item,itemLevel,corrected
            steered,wordCount,1,5.0,5.0,5.0,0.0,1.0,1.0,bh,promptID,q-e,item,itemLevel,corrected
            steered,distinct2,1,0.125,0.125,0.125,0.0,1.0,1.0,bh,promptID,q-e,item,itemLevel,corrected
            steered,warmMarkerDensity,1,0.375,0.375,0.375,0.0,1.0,1.0,bh,promptID,q-e,item,itemLevel,corrected
            steered,wordCount,3,4.0,3.0,5.0,0.0,0.18144920772142048,0.36289841544284096,bh,arm,x,item,itemLevel,corrected
            steered,distinct2,3,0.041666666666666664,-0.25,0.25,2.5,1.0,1.0,bh,arm,x,item,itemLevel,corrected
            steered,warmMarkerDensity,3,0.375,0.25,0.5,0.0,0.18144920772142048,0.36289841544284096,bh,arm,x,item,itemLevel,corrected
            steered,wordCount,2,0.0,-1.0,1.0,1.5,1.0,1.0,bh,arm,y,item,itemLevel,corrected
            steered,distinct2,2,0.125,0.0,0.25,0.0,1.0,1.0,bh,arm,y,item,itemLevel,corrected
            steered,warmMarkerDensity,2,0.0625,0.0,0.125,0.0,1.0,1.0,bh,arm,y,item,itemLevel,corrected

            """
        let entries =
            ExperimentTasks.effectSizes(rows: rows, concepts: ["warm"])
            + ExperimentTasks.stratifiedEffectSizes(
                rows: rows, concepts: ["warm"], factorsByItem: factors)
        #expect(ExperimentTasks.effectSizesCSV(entries) == before)
    }

    /// Under shared manifest seeds the within-item diagnostic rows were
    /// already right, and they keep every number (captured from the engine
    /// before the change). Records written before `sampleIndex` was stamped
    /// pair within an item by seed, and analyze to the same rows.
    @Test func sharedSeedWithinItemRowsKeepEveryNumber() throws {
        let (_, value) = try fixtureCase("manifestSeeds")
        let before = [
            "steered,wordCount,3,6.0,5.0,7.0,0.0,0.18144920772142048,,,promptID,item-1,sample,withinItemSamples,diagnostic",
            "steered,distinct2,3,0.25,0.25,0.25,0.0,0.1489146731787656,,,promptID,item-1,sample,withinItemSamples,diagnostic",
            "steered,wordCount,3,3.0,2.0,5.0,0.0,0.17356816655592167,,,promptID,item-2,sample,withinItemSamples,diagnostic",
            "steered,distinct2,3,0.0,-0.25,0.25,1.5,1.0,,,promptID,item-2,sample,withinItemSamples,diagnostic",
            "steered,wordCount,3,-1.0,-3.0,1.0,1.5,0.5862136810731401,,,promptID,item-3,sample,withinItemSamples,diagnostic",
            "steered,distinct2,3,0.25,0.25,0.25,0.0,0.1489146731787656,,,promptID,item-3,sample,withinItemSamples,diagnostic",
            "steered,wordCount,3,10.0,9.0,11.0,0.0,0.18144920772142048,,,promptID,item-4,sample,withinItemSamples,diagnostic",
            "steered,distinct2,3,-0.5,-0.5,-0.5,0.0,0.1489146731787656,,,promptID,item-4,sample,withinItemSamples,diagnostic",
        ]
        func withinItemRows(_ csv: String) -> [String] {
            csv.split(separator: "\n").map(String.init).filter { $0.contains(",promptID,") }
        }
        let stamped = try analyze(manifest(for: value), records: value.records)
        #expect(withinItemRows(stamped.csv) == before)

        struct UnstampedRecord: Encodable {
            let condition: String
            let seed: UInt64
            let promptIndex: Int
            let promptID: String
            let arm: String
            let wordCount: Int
            let distinct2: Double
        }
        let encoder = JSONEncoder()
        let unstamped = try analyze(
            manifest(for: value),
            lines: try value.records.map {
                String(
                    decoding: try encoder.encode(
                        UnstampedRecord(
                            condition: $0.condition, seed: $0.seed,
                            promptIndex: $0.promptIndex, promptID: $0.promptID,
                            arm: $0.arm, wordCount: $0.wordCount, distinct2: $0.distinct2)),
                    as: UTF8.self)
            })
        #expect(unstamped.csv == stamped.csv)
    }

    // MARK: - One item, many samples

    /// A study of ONE item sampled three times has one paired item, not
    /// three: baseline 10 12 14 → 12, steered 15 18 21 → 18, difference +6.
    /// With a single difference there is nothing to resample (the interval
    /// collapses onto 6) and the signed-rank test has z = 0, p = 1. What
    /// the three samples do say is kept, labelled for what it is, in the
    /// item's own stratum: +5, +6, +7 as a within-item diagnostic.
    @Test func oneItemSampledSeveralTimesIsOnePairedItem() throws {
        let (_, value) = try fixtureCase("derivedSHA256")
        let records = value.records.filter { $0.promptID == "item-1" }
        #expect(records.count == 6)
        let analysis = try analyze(manifest(for: value), records: records)

        let pooled = try entry(analysis.entries, metric: "wordCount")
        #expect(pooled.n == 1)
        #expect(pooled.meanDiff == 6)
        #expect(pooled.ciLower == 6 && pooled.ciUpper == 6)
        #expect(pooled.wilcoxonW == 0)
        #expect(pooled.wilcoxonP == 1)

        let within = try entry(
            analysis.entries, metric: "wordCount", stratifyBy: "promptID", stratum: "item-1")
        #expect(within.n == 3 && within.meanDiff == 6)
        #expect(within.unit == "sample" && within.inference == "diagnostic")
        #expect(within.adjustedP == nil)

        let table = try #require(RunResults.effectSizes(fromCSV: analysis.csv))
        let row = try #require(table.first { $0.metric == "wordCount" })
        #expect(
            EffectNarrative.sentence(for: row, in: table)
                .contains("by +6 across 1 paired item ("))
    }

    // MARK: - Cells of unequal size

    /// When the two arms kept different numbers of samples (an interrupted
    /// run, or declared exclusions), each cell is averaged over what it has
    /// and the item still contributes ONE difference. Hand-computed:
    ///
    ///     item-a  baseline 10 20 → 15      steered 30 → 30          +15
    ///     item-b  baseline 10 → 10         steered 13 15 17 → 15     +5
    ///     item-c  baseline 5 6 7 → 6       steered 8 · 12 → 10       +4
    ///
    /// Three paired items, mean (15 + 5 + 4) / 3 = 8; all positive, so
    /// W = 0. (Pairing sample with sample instead would give 20, 3, 3, and
    /// 5 — four differences with mean 7.75.)
    @Test func unevenCellsContributeOneDifferencePerItem() throws {
        var seed: UInt64 = 1000
        func row(_ condition: String, _ item: String, _ index: Int, _ sample: Int, _ words: Int)
            -> ExperimentTasks.MetricRow
        {
            // Derived seeds: every (condition, item, sample) draws its own.
            seed += 17
            return ExperimentTasks.MetricRow(
                condition: condition, seed: seed, promptIndex: index, promptID: item,
                wordCount: words, distinct2: 0.5, markerDensity: [:], sampleIndex: sample)
        }
        let rows = [
            row("baseline", "item-a", 1, 0, 10), row("baseline", "item-a", 1, 1, 20),
            row("steered", "item-a", 1, 0, 30),
            row("baseline", "item-b", 2, 0, 10),
            row("steered", "item-b", 2, 0, 13), row("steered", "item-b", 2, 1, 15),
            row("steered", "item-b", 2, 2, 17),
            row("baseline", "item-c", 3, 0, 5), row("baseline", "item-c", 3, 1, 6),
            row("baseline", "item-c", 3, 2, 7),
            row("steered", "item-c", 3, 0, 8), row("steered", "item-c", 3, 2, 12),
        ]
        let pooled = try #require(
            ExperimentTasks.effectSizes(rows: rows, concepts: [])
                .first { $0.metric == "wordCount" })
        #expect(pooled.n == 3)
        #expect(pooled.meanDiff == 8)
        #expect(pooled.wilcoxonW == 0)
        let p = try #require(pooled.wilcoxonP)
        #expect(abs(p - twoSidedP(w: 0, n: 3)) < 1e-12)  // 0.181449

        let stratified = ExperimentTasks.stratifiedEffectSizes(
            rows: rows, concepts: [], factorsByItem: [:])
        // item-a and item-b each have ONE sample index present in both arms,
        // which is not enough for a within-item read: they stay item-level.
        let itemA = try entry(
            stratified, metric: "wordCount", stratifyBy: "promptID", stratum: "item-a")
        #expect(itemA.n == 1 && itemA.meanDiff == 15 && itemA.unit == "item")
        let itemB = try entry(
            stratified, metric: "wordCount", stratifyBy: "promptID", stratum: "item-b")
        #expect(itemB.n == 1 && itemB.meanDiff == 5 && itemB.unit == "item")
        // item-c pairs samples 0 and 2 by index (8 − 5 = 3 and 12 − 7 = 5);
        // baseline sample 1 has no partner and is left out of this row.
        let itemC = try entry(
            stratified, metric: "wordCount", stratifyBy: "promptID", stratum: "item-c")
        #expect(itemC.n == 2 && itemC.meanDiff == 4)
        #expect(itemC.unit == "sample" && itemC.inference == "diagnostic")
    }

    /// A declared exclusion drops a RECORD, and its cell keeps the samples
    /// that survive — what the exclusion stamp's own note says. Excluding
    /// item-1's third steered sample (21 words) leaves 15 and 18 → 16.5, so
    /// item-1's difference is 16.5 − 12 = 4.5 and the pooled mean is
    /// (4.5 + 3 − 1 + 10) / 4 = 4.125 over the same four items.
    @Test func anExcludedSampleLeavesItsCellTheSurvivingSamples() throws {
        let (_, value) = try fixtureCase("derivedSHA256")
        var manifest = manifest(for: value)
        manifest.exclusionRules = [ExclusionRule(rule: ExclusionEngine.ruleUnparseableEndpoint)]
        let encoder = JSONEncoder()
        let lines = try value.records.map { record -> String in
            let line = String(decoding: try encoder.encode(record), as: UTF8.self)
            guard record.condition == "steered", record.promptID == "item-1",
                record.sampleIndex == 2
            else { return line }
            // A failed numeric parse: the endpoint key is present and null.
            return String(line.dropLast()) + #","parsedMonths":null}"#
        }
        let analysis = try analyze(manifest, lines: lines)

        let pooled = try entry(analysis.entries, metric: "wordCount")
        #expect(pooled.n == 4)
        #expect(pooled.meanDiff == 4.125)
        // The item's own stratum pairs the two samples both arms still have:
        // 15 − 10 = 5 and 18 − 12 = 6.
        let within = try entry(
            analysis.entries, metric: "wordCount", stratifyBy: "promptID", stratum: "item-1")
        #expect(within.n == 2 && within.meanDiff == 5.5)
        #expect(within.unit == "sample")
    }
}
