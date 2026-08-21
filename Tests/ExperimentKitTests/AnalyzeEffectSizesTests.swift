import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// StudyStatistics wiring: paired effect sizes in the report path, the
/// effect-sizes.csv contract, the battery score/report layer, and the
/// headless `experiment analyze` verb with its epoch guard. All engine-pure
/// (fixture rows, no model). Declared as an extension of the serialized
/// `ExperimentStoreTests` suite because the analyze tests share its
/// `rootOverride` test seam (a process-global).
extension ExperimentStoreTests {

    private func row(
        condition: String, promptID: String, wordCount: Int,
        distinct2: Float = 0.9, markers: [String: Float] = [:],
        seed: UInt64 = 1, promptIndex: Int = 0
    ) -> ExperimentTasks.MetricRow {
        ExperimentTasks.MetricRow(
            condition: condition, seed: seed, promptIndex: promptIndex,
            promptID: promptID, wordCount: wordCount, distinct2: distinct2,
            markerDensity: markers)
    }

    @Test func pairedEffectSizesAgainstSameItemBaseline() {
        // Condition adds exactly +10 words on every paired item.
        let rows = [
            row(condition: "baseline", promptID: "p1", wordCount: 100),
            row(condition: "baseline", promptID: "p2", wordCount: 120),
            row(condition: "baseline", promptID: "p3", wordCount: 90),
            row(condition: "steered", promptID: "p1", wordCount: 110,
                markers: ["fear": 0.2]),
            row(condition: "steered", promptID: "p2", wordCount: 130,
                markers: ["fear": 0.4]),
            row(condition: "steered", promptID: "p3", wordCount: 100,
                markers: ["fear": 0.3]),
        ]
        let entries = ExperimentTasks.effectSizes(
            rows: rows, concepts: ["fear"], replicates: 500)
        let wordCount = entries.first {
            $0.condition == "steered" && $0.metric == "wordCount"
        }
        #expect(wordCount?.n == 3)
        #expect(wordCount?.meanDiff == 10)
        // Constant differences: the bootstrap CI collapses onto the mean and
        // Wilcoxon is defined (all positive).
        #expect(wordCount?.ciLower == 10)
        #expect(wordCount?.ciUpper == 10)
        #expect(wordCount?.wilcoxonW != nil)

        // distinct2 identical everywhere → all-zero diffs → Wilcoxon
        // undefined → nils, mean 0.
        let distinct = entries.first {
            $0.condition == "steered" && $0.metric == "distinct2"
        }
        #expect(distinct?.meanDiff == 0)
        #expect(distinct?.wilcoxonW == nil)
        #expect(distinct?.wilcoxonP == nil)

        // Marker density diffs vs a zero baseline are positive.
        let fear = entries.first {
            $0.condition == "steered" && $0.metric == "fearMarkerDensity"
        }
        #expect(fear.map { $0.meanDiff > 0.29 && $0.meanDiff < 0.31 } == true)

        // Metrics per condition: wordCount, distinct2, fearMarkerDensity.
        #expect(entries.count == 3)
    }

    @Test func effectSizesRequireBaselineAndPairing() {
        // No baseline rows → no entries.
        #expect(
            ExperimentTasks.effectSizes(
                rows: [row(condition: "steered", promptID: "p1", wordCount: 10)],
                concepts: []
            ).isEmpty)
        // Unpaired items (different promptIDs) → no entries either.
        #expect(
            ExperimentTasks.effectSizes(
                rows: [
                    row(condition: "baseline", promptID: "p1", wordCount: 10),
                    row(condition: "steered", promptID: "p2", wordCount: 20),
                ],
                concepts: []
            ).isEmpty)
    }

    @Test func effectSizesCSVHasContractColumns() {
        let entries = [
            ExperimentTasks.EffectSizeEntry(
                condition: "steered", metric: "wordCount", n: 3, meanDiff: 10,
                ciLower: 8, ciUpper: 12, wilcoxonW: 0, wilcoxonP: 0.25,
                adjustedP: 0.5, correction: "bh"),
            ExperimentTasks.EffectSizeEntry(
                condition: "steered", metric: "distinct2", n: 3, meanDiff: 0,
                ciLower: 0, ciUpper: 0, wilcoxonW: nil, wilcoxonP: nil),
        ]
        let csv = ExperimentTasks.effectSizesCSV(entries)
        let lines = csv.split(separator: "\n").map(String.init)
        // The correction and stratification columns carry the server's exact
        // contract names (adjustedP, correction, stratifyBy, stratum, unit)
        // so one reader parses both dialects.
        #expect(
            lines[0] == "condition,metric,n,meanDiff,ciLower,ciUpper,"
                + "wilcoxonW,wilcoxonP,adjustedP,correction,"
                + "stratifyBy,stratum,unit,estimand,inference")
        // Entries without stratification provenance are the pooled rows —
        // and pooled rows leave estimand/inference EMPTY (the two columns
        // describe a stratified row's claim, and a pooled row's estimand was
        // never in question).
        #expect(
            lines[1] == "steered,wordCount,3,10.0,8.0,12.0,0.0,0.25,0.5,bh,"
                + "pooled,,,,")
        // Undefined Wilcoxon/adjusted cells are empty, never NaN.
        #expect(lines[2] == "steered,distinct2,3,0.0,0.0,0.0,,,,,pooled,,,,")

        // A stratified entry writes its family, cell label, unit, and what
        // it estimates / may be claimed from.
        let stratified = ExperimentTasks.EffectSizeEntry(
            condition: "steered", metric: "wordCount", n: 1, meanDiff: 36,
            ciLower: 36, ciUpper: 36, wilcoxonW: nil, wilcoxonP: nil,
            stratifyBy: "arm×caseID", stratum: "notLegal×loan", unit: "item",
            estimand: "itemLevel", inference: "corrected")
        let stratifiedLines = ExperimentTasks.effectSizesCSV([stratified])
            .split(separator: "\n").map(String.init)
        #expect(
            stratifiedLines[1] == "steered,wordCount,1,36.0,36.0,36.0,,,,,"
                + "arm×caseID,notLegal×loan,item,itemLevel,corrected")
    }

    /// Within-item sample rows are a DIAGNOSTIC, not a test (2026-08-06,
    /// cross-engine with the server's appended `estimand`/`inference`
    /// columns).
    ///
    /// A single-item stratum with several draws pairs sample against sample
    /// WITHIN one item: it reads within-item variability, not an item-level
    /// effect, so it is not an independent test of the pre-registered
    /// hypothesis. It used to join the familywise correction anyway —
    /// inflating the family (shrinking every real row's adjustedP) and
    /// stamping an `adjustedP` that read as citable. It is now held out:
    /// raw Wilcoxon and CI kept, no adjustedP, no correction stamp.
    @Test func withinItemSampleRowsAreHeldOutOfTheCorrectionFamily() throws {
        var rows: [ExperimentTasks.MetricRow] = []
        for seed in 1 ... 6 {
            rows.append(
                row(condition: "baseline", promptID: "live", wordCount: 100,
                    seed: UInt64(seed)))
            rows.append(
                row(condition: "steered", promptID: "live",
                    wordCount: 110 + seed % 2, seed: UInt64(seed)))
        }
        rows.append(row(condition: "baseline", promptID: "flat", wordCount: 50))
        rows.append(row(condition: "steered", promptID: "flat", wordCount: 60))

        let stratified = ExperimentTasks.stratifiedEffectSizes(
            rows: rows, concepts: [], factorsByItem: [:], replicates: 200)

        let within = try #require(
            stratified.first { $0.stratum == "live" && $0.metric == "wordCount" })
        #expect(within.unit == "sample")
        #expect(within.estimand == "withinItemSamples")
        #expect(within.inference == "diagnostic")
        // Held OUT of the family: no adjusted p, no correction stamp…
        #expect(within.adjustedP == nil)
        #expect(within.correction == nil)
        // …but still readable — the raw test and the CI survive.
        #expect((within.wilcoxonP ?? 1) < 0.05)
        #expect(within.ciLower < within.ciUpper || within.n == 1)

        // The item-level stratum in the SAME family is corrected as before.
        let itemLevel = try #require(
            stratified.first { $0.stratum == "flat" && $0.metric == "wordCount" })
        #expect(itemLevel.estimand == "itemLevel")
        #expect(itemLevel.inference == "corrected")
        #expect(itemLevel.correction == "bh")

        // Every stratified row is classified; pooled rows are not.
        #expect(stratified.allSatisfy { $0.estimand != nil && $0.inference != nil })
        #expect(
            ExperimentTasks.effectSizes(rows: rows, concepts: [], replicates: 200)
                .allSatisfy { $0.estimand == nil && $0.inference == nil })
    }

    // MARK: - Stratified effect sizes (server twin)

    @Test func stratifiedEffectSizesSurfaceACellMaskedByPooling() {
        // The K&Z failure mode, engine-pure: one live cell (+36 words) among
        // three saturated cells. The pooled row dilutes the effect to +9 —
        // its semantics are untouched — while the stratified rows carry the
        // cell at full strength under every family that contains it.
        var rows: [ExperimentTasks.MetricRow] = []
        for (promptID, shift) in [
            ("loan-notLegal", 36), ("loan-legal", 0),
            ("lease-notLegal", 0), ("lease-legal", 0),
        ] {
            rows.append(row(condition: "baseline", promptID: promptID, wordCount: 100))
            rows.append(
                row(condition: "steered", promptID: promptID, wordCount: 100 + shift))
        }
        let factors: [String: [String: String]] = [
            "loan-notLegal": ["arm": "notLegal", "caseID": "loan"],
            "loan-legal": ["arm": "legal", "caseID": "loan"],
            "lease-notLegal": ["arm": "notLegal", "caseID": "lease"],
            "lease-legal": ["arm": "legal", "caseID": "lease"],
        ]
        let pooled = ExperimentTasks.effectSizes(
            rows: rows, concepts: [], replicates: 200)
        let pooledWordCount = pooled.first { $0.metric == "wordCount" }
        #expect(pooledWordCount?.meanDiff == 9)
        #expect(pooledWordCount?.n == 4)
        // Pooled entries carry no stratification keys (report bytes stable).
        #expect(pooledWordCount?.stratifyBy == nil)
        #expect(pooledWordCount?.unit == nil)

        let stratified = ExperimentTasks.stratifiedEffectSizes(
            rows: rows, concepts: [], factorsByItem: factors, replicates: 200)
        // promptID family: the live item's own row names the mover.
        let live = stratified.first {
            $0.stratifyBy == "promptID" && $0.stratum == "loan-notLegal"
                && $0.metric == "wordCount"
        }
        #expect(live?.n == 1)
        #expect(live?.meanDiff == 36)
        #expect(live?.unit == "item")
        // Marginal factor family: notLegal pools its two items at item unit.
        let armRow = stratified.first {
            $0.stratifyBy == "arm" && $0.stratum == "notLegal"
                && $0.metric == "wordCount"
        }
        #expect(armRow?.n == 2)
        #expect(armRow?.meanDiff == 18)
        #expect(armRow?.unit == "item")
        // The full cross names the cell the way the researcher reads it
        // (keys sorted alphabetically, levels in key order).
        let cell = stratified.first {
            $0.stratifyBy == "arm×caseID" && $0.stratum == "notLegal×loan"
                && $0.metric == "wordCount"
        }
        #expect(cell?.n == 1)
        #expect(cell?.meanDiff == 36)
        // Family inventory: promptID always, both factor marginals, and the
        // cross — each corrected independently (stamp present on all rows).
        #expect(
            Set(stratified.compactMap(\.stratifyBy))
                == ["promptID", "arm", "caseID", "arm×caseID"])
        #expect(stratified.allSatisfy { $0.correction == "bh" })
    }

    @Test func stratifiedSingleItemStrataDropToTheSampleAxis() {
        // A single-item stratum with several paired replicates resolves the
        // pairs WITHIN the item — the resolution pooling averages away. Unit
        // says so ("sample"), and the sign test fires from one item's data.
        var rows: [ExperimentTasks.MetricRow] = []
        for seed in 1 ... 6 {
            rows.append(
                row(condition: "baseline", promptID: "live", wordCount: 100,
                    seed: UInt64(seed)))
            rows.append(
                row(condition: "steered", promptID: "live",
                    wordCount: 110 + seed % 2, seed: UInt64(seed)))
        }
        rows.append(row(condition: "baseline", promptID: "flat", wordCount: 50))
        rows.append(row(condition: "steered", promptID: "flat", wordCount: 50))

        let stratified = ExperimentTasks.stratifiedEffectSizes(
            rows: rows, concepts: [], factorsByItem: [:], replicates: 200)
        let live = stratified.first {
            $0.stratum == "live" && $0.metric == "wordCount"
        }
        #expect(live?.unit == "sample")
        #expect(live?.n == 6)
        #expect(live.map { $0.meanDiff > 10 && $0.meanDiff < 11 } == true)
        #expect(live.map { ($0.wilcoxonP ?? 1) < 0.05 } == true)
        // No factor metadata → the promptID family is the only one.
        #expect(Set(stratified.compactMap(\.stratifyBy)) == ["promptID"])
    }

    // MARK: - Multiple-comparison correction (server-analyze twin)

    @Test func correctionMethodFollowsTheServersPhaseRule() {
        // tasks.py analyze: "holm" if manifest.phase == "confirm" else "bh".
        #expect(ExperimentTasks.correctionMethod(phase: "confirm") == "holm")
        #expect(ExperimentTasks.correctionMethod(phase: "screen") == "bh")
        #expect(ExperimentTasks.correctionMethod(phase: "shakedown") == "bh")
        #expect(ExperimentTasks.correctionMethod(phase: nil) == "bh")
    }

    private func entry(
        _ condition: String, _ metric: String, p: Double?
    ) -> ExperimentTasks.EffectSizeEntry {
        ExperimentTasks.EffectSizeEntry(
            condition: condition, metric: metric, n: 3, meanDiff: 1,
            ciLower: 0.5, ciUpper: 1.5, wilcoxonW: 0, wilcoxonP: p)
    }

    @Test func applyCorrectionGroupsPerMetricAndFollowsPhase() {
        // Two metric families across two conditions + one undefined p. The
        // family boundary is the METRIC (the server groups per endpoint):
        // wordCount's correction must never see distinct2's p-values.
        let entries = [
            entry("c1", "wordCount", p: 0.02),
            entry("c2", "wordCount", p: 0.03),
            entry("c1", "distinct2", p: 0.01),
            entry("c2", "distinct2", p: nil),
        ]

        // Absent/screen phase → BH per family. bh([0.02, 0.03]) == [0.03,
        // 0.03] — the identical dense values are pinned server-side in
        // test_correction_parity.py's apply_correction fixture.
        for phase in [nil, "screen"] as [String?] {
            let bh = ExperimentTasks.applyCorrection(entries, phase: phase)
            #expect(bh[0].adjustedP == 0.03 && bh[0].correction == "bh")
            #expect(bh[1].adjustedP == 0.03 && bh[1].correction == "bh")
            // distinct2 family has ONE usable p → adjusted == raw (m = 1);
            // were the family global (m = 3), 0.01 would adjust to 0.03.
            #expect(bh[2].adjustedP == 0.01 && bh[2].correction == "bh")
            // Undefined p: skipped by the adjustment, still stamped with
            // the method (server apply_correction semantics).
            #expect(bh[3].adjustedP == nil && bh[3].correction == "bh")
        }

        // Confirm phase → Holm. holm([0.02, 0.03]) == [0.04, 0.04].
        let holm = ExperimentTasks.applyCorrection(entries, phase: "confirm")
        #expect(holm[0].adjustedP == 0.04 && holm[0].correction == "holm")
        #expect(holm[1].adjustedP == 0.04 && holm[1].correction == "holm")
        #expect(holm[2].adjustedP == 0.01 && holm[2].correction == "holm")
        #expect(holm[3].adjustedP == nil && holm[3].correction == "holm")

        // Entry order is preserved (the CSV/report row order is the
        // condition-appearance order, not the correction's sort).
        #expect(holm.map(\.metric) == entries.map(\.metric))
        #expect(holm.map(\.condition) == entries.map(\.condition))
    }

    @Test func effectSizesStampCorrectionAndRoundTripThroughRunResults() throws {
        // The run-inline path: effectSizes(...) itself applies the phase's
        // correction, so report.json and effect-sizes.csv agree.
        let rows = [
            row(condition: "baseline", promptID: "p1", wordCount: 100),
            row(condition: "baseline", promptID: "p2", wordCount: 90),
            row(condition: "steered", promptID: "p1", wordCount: 111),
            row(condition: "steered", promptID: "p2", wordCount: 99),
        ]
        let entries = ExperimentTasks.effectSizes(
            rows: rows, concepts: [], replicates: 200, phase: "confirm")
        let wordCount = try #require(
            entries.first { $0.metric == "wordCount" })
        #expect(wordCount.correction == "holm")
        #expect(wordCount.adjustedP != nil)
        // distinct2 diffs are all zero → Wilcoxon undefined → adjusted
        // absent but the family stamp present.
        let distinct = try #require(
            entries.first { $0.metric == "distinct2" })
        #expect(distinct.adjustedP == nil && distinct.correction == "holm")

        // The Swift CSV dialect round-trips through the ONE cross-engine
        // reader the effect views consume.
        let parsed = try #require(
            RunResults.effectSizes(fromCSV: ExperimentTasks.effectSizesCSV(entries)))
        let parsedWordCount = try #require(
            parsed.first { $0.metric == "wordCount" })
        #expect(parsedWordCount.adjustedP == wordCount.adjustedP)
        #expect(parsedWordCount.correction == "holm")
        let parsedDistinct = try #require(
            parsed.first { $0.metric == "distinct2" })
        #expect(parsedDistinct.adjustedP == nil)
        #expect(parsedDistinct.correction == "holm")
    }

    @Test func batteryScoreLayerIsPureAndStampsContractFields() throws {
        let batteryJSONL = """
            {"prompt": "2+2?", "answer": "4"}
            {"prompt": "Is fire hot?", "answer": "yes"}
            {"prompt": "Capital of France?", "answer": "Paris"}

            """
        let url = FileManager.default.temporaryDirectory
            .appending(component: "battery-\(UUID().uuidString).jsonl")
        try batteryJSONL.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        let battery = try CapabilityBattery(url: url)

        // The pure seam moved with the format-2 twin (2026-08-19): readings
        // are taken by the arming-aware scorer, and `scoreBatteryReadings`
        // turns already-taken readings into records + summary. Same intent —
        // the score/report layer is exercised with NO model — and this now
        // runs the real legacy grader on the way in.
        let arming = battery.resolveArming(promptMode: .chatAssistant)
        let scored = ExperimentTasks.scoreBatteryReadings(
            condition: "steered",
            battery: battery,
            arming: arming,
            scores: VariantRobustness.batteryScores(
                battery, ["The answer is 4", "No.", "Paris, of course"]),
            batteryHash: "feedface")
        #expect(scored.summary.itemCount == 3)
        #expect(scored.summary.batteryHash == "feedface")
        #expect(abs(scored.summary.accuracy - 2.0 / 3.0) < 1e-9)
        #expect(scored.records.count == 3)
        #expect(scored.records.map(\.correct) == [true, false, true])
        #expect(scored.records.allSatisfy { $0.condition == "steered" })

        // The report block carries the cross-engine keys.
        let encoded = String(
            decoding: try JSONEncoder().encode(scored.summary), as: UTF8.self)
        #expect(encoded.contains("\"accuracy\""))
        #expect(encoded.contains("\"itemCount\""))
        #expect(encoded.contains("\"batteryHash\""))
    }

    @Test func reportStampsBatteryPerConditionAndEffectSizes() throws {
        // Report assembly (engine-pure): every condition with a battery
        // summary carries the "capabilityBattery" block — including
        // choice-only conditions that produced no sampled rows.
        let manifest = ExperimentManifest(
            name: "report-fixture", description: "", modelID: "test/model")
        let rows = [
            row(condition: "baseline", promptID: "p1", wordCount: 100),
            row(condition: "steered", promptID: "p1", wordCount: 110),
        ]
        let summaries = [
            "baseline": ExperimentTasks.CapabilityBatterySummary(
                accuracy: 1.0, itemCount: 4, batteryHash: "aa"),
            "steered": ExperimentTasks.CapabilityBatterySummary(
                accuracy: 0.75, itemCount: 4, batteryHash: "aa"),
            "choice-only": ExperimentTasks.CapabilityBatterySummary(
                accuracy: 0.5, itemCount: 4, batteryHash: "aa"),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: rows, conditionCount: 3,
            concepts: [], batterySummaries: summaries)
        #expect(report.conditions["baseline"]?.capabilityBattery?.accuracy == 1.0)
        #expect(report.conditions["steered"]?.capabilityBattery?.accuracy == 0.75)
        #expect(report.conditions["choice-only"]?.capabilityBattery?.itemCount == 4)
        #expect(report.conditions["choice-only"]?.generations == 0)
        #expect(report.effectSizes?.isEmpty == false)
        // The run-inline effect rows carry the phase's correction — an
        // unphased manifest gets the screen default (BH), exactly like the
        // server's analyze.
        let wordCount = try #require(
            report.effectSizes?.first { $0.metric == "wordCount" })
        #expect(wordCount.correction == "bh")
        #expect(wordCount.adjustedP != nil)

        // Encoded report.json carries the contract keys under conditions.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(report), as: UTF8.self)
        #expect(json.contains("\"capabilityBattery\""))
        #expect(json.contains("\"effectSizes\""))
        #expect(json.contains("\"adjustedP\""))
        #expect(json.contains("\"correction\""))

        // A confirm-phase manifest flips the SAME report path to Holm.
        var confirmManifest = manifest
        confirmManifest.phase = "confirm"
        let confirmReport = ExperimentTasks.report(
            experiment: confirmManifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: rows, conditionCount: 3,
            concepts: [], batterySummaries: summaries)
        #expect(
            confirmReport.effectSizes?.first { $0.metric == "wordCount" }?
                .correction == "holm")
    }

    @Test func reportStampsCategoricalReadoutsLikeTheServer() throws {
        let manifest = ExperimentManifest(
            name: "choice-report", description: "", modelID: "test/model")
        let readouts = [
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "A", target: "A"),
            ExperimentTasks.ReportChoiceReadout(
                condition: "steered", promptID: "p1", sampleIndex: nil,
                source: "instrument", selected: "B", target: "A"),
            ExperimentTasks.ReportChoiceReadout(
                condition: "baseline", promptID: "p1", sampleIndex: 7,
                source: "parsed", selected: "A", target: "A"),
            ExperimentTasks.ReportChoiceReadout(
                condition: "steered", promptID: "p1", sampleIndex: 7,
                source: "parsed", selected: "A", target: "A"),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hh",
            taskPrompts: ("f.jsonl", "ph", []), rows: [], conditionCount: 2,
            concepts: [], choiceReadouts: readouts)
        let baseline = try #require(report.conditions["baseline"])
        let steered = try #require(report.conditions["steered"])
        #expect(baseline.generations == 0)
        #expect(baseline.choiceReadouts == 1)
        #expect(baseline.choiceRate == 1.0)
        #expect(steered.choiceReadouts == 1)
        #expect(steered.choiceRate == 1.0)
        #expect(steered.agreementWithBaseline?.n == 2)
        #expect(steered.agreementWithBaseline?.agreement == 0.5)

        let json = String(decoding: try JSONEncoder().encode(report), as: UTF8.self)
        #expect(json.contains("\"choiceReadouts\":1"))
        #expect(json.contains("\"choiceRate\":1"))
        #expect(json.contains("\"agreementWithBaseline\""))
    }

    // MARK: - analyze verb + epoch guard

    private func withAnalyzeTempRoot<T>(_ body: () throws -> T) rethrows -> T {
        // Shared cross-suite lock for the process-global rootOverride.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "analyze") { _ in
            try body()
        }
    }

    /// Fabricates a completed study run for the manifest: generations.jsonl
    /// with paired baseline/steered records, report.json, manifest snapshot,
    /// and (optionally) the experiment-hash stamp.
    private func fabricateRun(
        for manifest: ExperimentManifest, stamped: Bool = true,
        directoryName: String = "20260712T000000000Z-exp-NAME-run"
    ) throws -> URL {
        let dir = ExperimentStore.runsDirectory.appending(
            component: directoryName.replacingOccurrences(
                of: "NAME", with: manifest.name))
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        if stamped {
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
        }
        var lines = ""
        for (promptID, base, steered) in [("p1", 100, 111), ("p2", 90, 99)] {
            lines += """
                {"condition": "baseline", "seed": 1, "promptID": "\(promptID)", "promptIndex": 1, "wordCount": \(base), "distinct2": 0.9, "markerDensity": {"french": 0.0}}
                {"condition": "steered", "seed": 1, "promptID": "\(promptID)", "promptIndex": 1, "wordCount": \(steered), "distinct2": 0.8, "markerDensity": {"french": 0.5}}

                """
        }
        // A choice-instrument record must be skipped by analysis.
        lines += """
            {"condition": "steered", "promptID": "p1", "instrument": "answerTokenLogprob", "selected": "affirm", "margin": 0.5}

            """
        try lines.write(
            to: dir.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: dir.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        return dir
    }

    private func makeVerifiedManifest(name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        let french = try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
        manifest.concepts.append(
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: french.hash, options: .init()))
        try ExperimentStore.save(manifest)
        return manifest
    }

    @Test func analyzeComputesEffectSizesFromNewestRun() throws {
        try withAnalyzeTempRoot {
            let manifest = try makeVerifiedManifest(name: "analyze-me")
            _ = try fabricateRun(for: manifest)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-me")
            let data = try Data(contentsOf: out.appending(component: "analysis.json"))
            let object = try #require(
                try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["experiment"] as? String == "analyze-me")
            #expect(object["epochUnverified"] == nil)
            #expect(
                object["sourceRunExperimentHash"] as? String
                    == ExperimentStore.manifestHash(manifest))
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let wordCount = try #require(
                entries.first {
                    $0["metric"] as? String == "wordCount"
                        && $0["condition"] as? String == "steered"
                })
            #expect(wordCount["n"] as? Int == 2)
            #expect(wordCount["meanDiff"] as? Double == 10)
            // An unphased manifest gets the screen default (BH) — one row
            // per family here, so adjusted == raw (m = 1).
            #expect(wordCount["correction"] as? String == "bh")
            #expect(
                wordCount["adjustedP"] as? Double
                    == wordCount["wilcoxonP"] as? Double)
            // The CSV rides along with the contract header (correction
            // columns carry the server's exact names).
            let csv = try String(
                contentsOf: out.appending(component: "effect-sizes.csv"),
                encoding: .utf8)
            #expect(
                csv.hasPrefix(
                    "condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,"
                        + "wilcoxonP,adjustedP,correction"))
        }
    }

    /// End-to-end for the stratified companion rows: a run whose records
    /// carry arm/caseID metadata analyzes into per-stratum rows BESIDE the
    /// pooled rows — same CSV, extra rows, pooled semantics untouched — and
    /// the semantic RunResults reader keeps returning pooled rows only.
    @Test func analyzeEmitsStratifiedRowsBesidePooledOnes() throws {
        try withAnalyzeTempRoot {
            let manifest = try makeVerifiedManifest(name: "analyze-strata")
            let dir = ExperimentStore.runsDirectory.appending(
                component: "20260806T000000000Z-exp-analyze-strata-run")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: dir.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            var lines = ""
            // One live cell (+36 words) among three saturated cells.
            for (promptID, arm, caseID, shift) in [
                ("loan-notLegal", "notLegal", "loan", 36),
                ("loan-legal", "legal", "loan", 0),
                ("lease-notLegal", "notLegal", "lease", 0),
                ("lease-legal", "legal", "lease", 0),
            ] {
                for (condition, wordCount) in [
                    ("baseline", 100), ("steered", 100 + shift),
                ] {
                    lines += """
                        {"condition": "\(condition)", "seed": 1, "promptID": "\(promptID)", "promptIndex": 1, "wordCount": \(wordCount), "distinct2": 0.9, "arm": "\(arm)", "caseID": "\(caseID)"}

                        """
                }
            }
            try lines.write(
                to: dir.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(
                to: dir.appending(component: "report.json"),
                atomically: true, encoding: .utf8)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-strata")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            // Pooled rows first (no stratification keys — legacy shape).
            let pooled = try #require(
                entries.first {
                    $0["metric"] as? String == "wordCount"
                        && $0["stratifyBy"] == nil
                })
            #expect(pooled["n"] as? Int == 4)
            #expect(pooled["meanDiff"] as? Double == 9)
            // The cross family names the live cell with its own statistics.
            let cell = try #require(
                entries.first {
                    $0["stratifyBy"] as? String == "arm×caseID"
                        && $0["stratum"] as? String == "notLegal×loan"
                        && $0["metric"] as? String == "wordCount"
                })
            #expect(cell["n"] as? Int == 1)
            #expect(cell["meanDiff"] as? Double == 36)
            #expect(cell["unit"] as? String == "item")

            let csv = try String(
                contentsOf: out.appending(component: "effect-sizes.csv"),
                encoding: .utf8)
            #expect(csv.contains("arm×caseID,notLegal×loan,item"))
            // The semantic reader (effect table/narrative/forest plot input)
            // filters the stratified rows: one row per (condition, metric).
            let parsed = try #require(RunResults.effectSizes(fromCSV: csv))
            #expect(parsed.count == 2)
            #expect(Set(parsed.map(\.metric)) == ["wordCount", "distinct2"])
        }
    }

    /// End-to-end for the confirm-phase correction: a confirm-phase
    /// manifest's analyze output stamps Holm-adjusted p-values in
    /// analysis.json AND effect-sizes.csv, and the CSV rows decode through
    /// the RunResults reader with the corrected fields present — the
    /// narrative/table's "survives correction" path now fires from
    /// Swift-local data.
    @Test func analyzeAppliesHolmForConfirmPhase() throws {
        try withAnalyzeTempRoot {
            var manifest = try makeVerifiedManifest(name: "analyze-confirm")
            manifest.phase = "confirm"
            // Funnel gate: a confirm-phase study pins the screen pool it is
            // held out from (any distinct hash satisfies verify here).
            manifest.screenTaskPromptsHash = "feedfacedeadbeef"
            try ExperimentStore.save(manifest)
            _ = try fabricateRun(for: manifest)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-confirm")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let wordCount = try #require(
                entries.first { $0["metric"] as? String == "wordCount" })
            #expect(wordCount["correction"] as? String == "holm")
            #expect(wordCount["adjustedP"] is Double)

            let csv = try String(
                contentsOf: out.appending(component: "effect-sizes.csv"),
                encoding: .utf8)
            let rows = try #require(RunResults.effectSizes(fromCSV: csv))
            let parsed = try #require(rows.first { $0.metric == "wordCount" })
            #expect(parsed.correction == "holm")
            #expect(parsed.adjustedP != nil)
            #expect(parsed.significantAfterCorrection != nil)
        }
    }

    /// End-to-end for the ordinalScale instrument's statistics layer: a
    /// fixture run whose generations.jsonl holds ONLY instrument records
    /// with ordinalPosition flows through analyze into an "ordinalPosition"
    /// effect row (paired to the same-item baseline readout, bootstrap CI +
    /// Wilcoxon), and the CSV round-trips through the RunResults decoder the
    /// effect views render — no model, no GPU.
    @Test func analyzeComputesOrdinalEffectsFromInstrumentRecords() throws {
        try withAnalyzeTempRoot {
            let manifest = try makeVerifiedManifest(name: "analyze-ordinal")
            let dir = ExperimentStore.runsDirectory.appending(
                component: "20260719T000000000Z-exp-analyze-ordinal-run")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: dir.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            var lines = ""
            for (promptID, base, steered) in [
                ("p1", 2.0, 2.3), ("p2", 1.0, 1.7),
            ] {
                lines += """
                    {"condition": "baseline", "promptID": "\(promptID)", "instrument": "answerTokenLogprob", "selected": "2", "ordinalPosition": \(base), "ordinalDistribution": [0.2, 0.3, 0.5]}
                    {"condition": "steered", "promptID": "\(promptID)", "instrument": "answerTokenLogprob", "selected": "3", "ordinalPosition": \(steered), "ordinalDistribution": [0.1, 0.2, 0.7]}

                    """
            }
            try lines.write(
                to: dir.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(
                to: dir.appending(component: "report.json"),
                atomically: true, encoding: .utf8)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-ordinal")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let ordinal = try #require(
                entries.first {
                    $0["metric"] as? String == "ordinalPosition"
                        && $0["condition"] as? String == "steered"
                })
            #expect(ordinal["n"] as? Int == 2)
            // Diffs +0.3/+0.7 → paired mean +0.5, CI within the diff range.
            let meanDiff = try #require(ordinal["meanDiff"] as? Double)
            #expect(abs(meanDiff - 0.5) < 1e-9)
            let ciLower = try #require(ordinal["ciLower"] as? Double)
            let ciUpper = try #require(ordinal["ciUpper"] as? Double)
            #expect(ciLower >= 0.3 - 1e-9 && ciUpper <= 0.7 + 1e-9)

            // effect-sizes.csv decodes through the SAME RunResults parser the
            // effect views (table, narrative, forest plot) consume.
            let csv = try String(
                contentsOf: out.appending(component: "effect-sizes.csv"),
                encoding: .utf8)
            let rows = try #require(RunResults.effectSizes(fromCSV: csv))
            let row = try #require(rows.first { $0.metric == "ordinalPosition" })
            #expect(row.condition == "steered")
            #expect(row.n == 2)
            #expect(abs(row.meanDiff - 0.5) < 1e-9)
        }
    }

    /// End-to-end twin of the server's `test_choice_deltas.py`: a run whose
    /// generations.jsonl holds ONLY answer-token instrument records flows
    /// through analyze into `choice-deltas.csv` + `choice-deltas.json` — the
    /// citable per-item Δ the results explorer otherwise derives itself — and
    /// a second analyze of the same run reproduces both byte for byte.
    @Test func analyzeWritesPerItemChoiceDeltas() throws {
        try withAnalyzeTempRoot {
            let manifest = try makeVerifiedManifest(name: "analyze-choice")
            let dir = ExperimentStore.runsDirectory.appending(
                component: "20260805T000000000Z-exp-analyze-choice-run")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: dir.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            // p1: target A up 3 nats, no flip. p2: up 5 nats AND flips B → A.
            // p3: measured under steered only — skipped AND counted.
            let lines = """
                {"condition": "baseline", "promptID": "p2", "instrument": "answerTokenLogprob", "target": "A", "options": ["A", "B"], "logOdds": {"A": -2.0, "B": 2.0}, "choiceProbability": {"A": 0.25, "B": 0.75}, "selected": "B"}
                {"condition": "steered", "promptID": "p2", "instrument": "answerTokenLogprob", "target": "A", "options": ["A", "B"], "logOdds": {"A": 3.0, "B": -3.0}, "choiceProbability": {"A": 0.9, "B": 0.1}, "selected": "A"}
                {"condition": "baseline", "promptID": "p1", "instrument": "answerTokenLogprob", "target": "A", "options": ["A", "B"], "logOdds": {"A": 1.0, "B": -1.0}, "choiceProbability": {"A": 0.8, "B": 0.2}, "selected": "A"}
                {"condition": "steered", "promptID": "p1", "instrument": "answerTokenLogprob", "target": "A", "options": ["A", "B"], "logOdds": {"A": 4.0, "B": -4.0}, "choiceProbability": {"A": 0.95, "B": 0.05}, "selected": "A"}
                {"condition": "steered", "promptID": "p3", "instrument": "answerTokenLogprob", "target": "A", "options": ["A", "B"], "logOdds": {"A": 7.0, "B": -7.0}, "choiceProbability": {"A": 0.99, "B": 0.01}, "selected": "A"}

                """
            try lines.write(
                to: dir.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(
                to: dir.appending(component: "report.json"),
                atomically: true, encoding: .utf8)

            let out = try ExperimentTasks.analyze(experimentName: "analyze-choice")
            let csv = try String(
                contentsOf: out.appending(component: "choice-deltas.csv"),
                encoding: .utf8)
            let rows = csv.split(separator: "\n").map(String.init)
            #expect(rows[0] == ChoiceDeltas.header.joined(separator: ","))
            // Sorted by (condition, promptID) — never by run order.
            #expect(rows.count == 3)
            #expect(rows[1] == "steered,p1,A,1,4,3,A,A,0,0.8,0.95,0.15")
            #expect(rows[2] == "steered,p2,A,-2,3,5,B,A,1,0.25,0.9,0.65")

            let summary = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(component: "choice-deltas.json")))
                    as? [String: Any])
            #expect(summary["records"] as? Int == 2)
            // The unpaired p3 readout is skipped AND counted: silent
            // truncation would read as coverage.
            #expect(summary["skippedNoBaseline"] as? Int == 1)
            let conditions = try #require(summary["conditions"] as? [String: Any])
            let steered = try #require(conditions["steered"] as? [String: Any])
            #expect(steered["n"] as? Int == 2)
            #expect(steered["flipped"] as? Int == 1)
            #expect(steered["skippedNoBaseline"] as? Int == 1)
            #expect(abs(try #require(steered["deltaTargetLogOddsMean"] as? Double) - 4) < 1e-9)
            #expect(steered["replicates"] as? Int == 10_000)

            // Determinism: same run in → byte-identical artifacts out.
            let again = try ExperimentTasks.analyze(experimentName: "analyze-choice")
            #expect(again != out)
            for artifact in ["choice-deltas.csv", "choice-deltas.json"] {
                #expect(
                    try Data(contentsOf: out.appending(component: artifact))
                        == (try Data(contentsOf: again.appending(component: artifact))),
                    "\(artifact) is not reproducible")
            }
        }
    }

    /// Absence over empty artifacts: a run with no choice readouts must not
    /// grow a table implying it had some.
    @Test func analyzeWritesNoChoiceDeltasWithoutChoiceReadouts() throws {
        try withAnalyzeTempRoot {
            let manifest = try makeVerifiedManifest(name: "analyze-no-choice")
            let dir = ExperimentStore.runsDirectory.appending(
                component: "20260805T000000000Z-exp-analyze-no-choice-run")
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(
                to: dir.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            let lines = """
                {"condition": "baseline", "seed": 1, "promptID": "p1", "promptIndex": 0, "wordCount": 100, "distinct2": 0.9}
                {"condition": "steered", "seed": 1, "promptID": "p1", "promptIndex": 0, "wordCount": 110, "distinct2": 0.8}

                """
            try lines.write(
                to: dir.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(
                to: dir.appending(component: "report.json"),
                atomically: true, encoding: .utf8)

            let out = try ExperimentTasks.analyze(
                experimentName: "analyze-no-choice")
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "choice-deltas.csv").path))
            #expect(
                !FileManager.default.fileExists(
                    atPath: out.appending(component: "choice-deltas.json").path))
        }
    }

    @Test func analyzeRefusesEpochMismatchAndFlagsUnstampedRuns() throws {
        try withAnalyzeTempRoot {
            var manifest = try makeVerifiedManifest(name: "epoch")
            _ = try fabricateRun(for: manifest)

            // Mutate the manifest AFTER the run: the epoch guard refuses.
            manifest.maxTokens = 512
            try ExperimentStore.save(manifest)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.analyze(experimentName: "epoch")
            }

            // An UNSTAMPED newer run refuses without the flag…
            let unstamped = try fabricateRun(
                for: manifest, stamped: false,
                directoryName: "20260713T000000000Z-exp-NAME-run")
            #expect(
                ExperimentTasks.runExperimentHashStamp(at: unstamped) == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.analyze(experimentName: "epoch")
            }
            // …and is accepted WITH it, stamping epochUnverified: true.
            let out = try ExperimentTasks.analyze(
                experimentName: "epoch", allowUnverifiedEpoch: true)
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            #expect(object["epochUnverified"] as? Bool == true)

            // The flag NEVER bypasses a stamped mismatch: with only the
            // stale-stamped run present, --allow-unverified-epoch still
            // refuses.
            try FileManager.default.removeItem(at: unstamped)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.analyze(
                    experimentName: "epoch", allowUnverifiedEpoch: true)
            }
        }
    }
}
