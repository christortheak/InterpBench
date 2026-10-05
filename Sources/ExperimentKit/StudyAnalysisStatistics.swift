import Foundation
import SteeringKit

/// Value-only rules; no workspace discovery or artifact writes.
enum StudyAnalysisStatistics {
    typealias EffectSizeEstimand = ExperimentTasks.EffectSizeEstimand
    typealias EffectSizeInference = ExperimentTasks.EffectSizeInference
    typealias MetricRow = ExperimentTasks.MetricRow
    typealias ReportChoiceReadout = ExperimentTasks.ReportChoiceReadout
    typealias EffectSizeEntry = ExperimentTasks.EffectSizeEntry
    typealias ReasoningStyleFeatureStat = ExperimentTasks.ReasoningStyleFeatureStat
    typealias ReasoningStyleConditionReport = ExperimentTasks.ReasoningStyleConditionReport

    /// One deterministic answer-token value for a (condition, item): the
    /// input of an outcome that has no sample axis. `analyze` builds these
    /// for `choiceLogOdds`, from the readouts whose target the item
    /// declared.
    struct ItemReadout: Equatable {
        let condition: String
        let promptID: String
        let value: Double
    }

    /// One paired outcome built from the sampled responses: how a response
    /// is read, and how a cell's readings become the item's value.
    /// `StudyAnalysisOutcomes` holds the same rules in words.
    struct SampledOutcome {
        enum Reduction {
            /// The mean of the cell's readings.
            case mean
            /// The mean of the cell's parsed numbers, summed in ascending
            /// order (the Python engine's `judicial.summarize`).
            case parsedMean
            /// The sample standard deviation (n − 1) of the cell's parsed
            /// numbers; no value with fewer than two.
            case parsedSpread
        }

        let name: String
        /// One response's reading; nil when the response carries none.
        let reading: (MetricRow) -> Double?
        var reduction: Reduction = .mean
        /// Whether a stratum in which one item pairs may drop to that
        /// item's sample axis. False for a spread (it is a property of the
        /// whole cell) and for the neutral parsed-value names, which the
        /// Python engine has never read sample by sample.
        var pairsWithinItem = true

        /// The item's value in one condition; nil when the cell has no
        /// reading to form one from.
        func itemValue(_ cell: [MetricRow]) -> Double? {
            let readings = cell.compactMap(reading)
            switch reduction {
            case .mean:
                guard !readings.isEmpty else { return nil }
                return readings.reduce(0, +) / Double(readings.count)
            case .parsedMean, .parsedSpread:
                let parsed = readings.filter { !$0.isNaN }.sorted()
                guard !parsed.isEmpty else { return nil }
                let mean = parsed.reduce(0, +) / Double(parsed.count)
                if reduction == .parsedMean { return mean }
                guard parsed.count > 1 else { return nil }
                let squares = parsed.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
                return (squares / Double(parsed.count - 1)).squareRoot()
            }
        }
    }

    /// The outcomes the sampled responses can support, in emission order:
    /// the surface measures, each concept's marker density, the
    /// reasoning-style features (declared taxonomy order), then the outcomes
    /// a record carries only when the study measured them — reader scores,
    /// the parsed number and its spread, and the target-choice rate. An
    /// outcome no response carries a reading for produces no row.
    ///
    /// `numericParserKind` is the declared numeric parser's kind, nil when
    /// the study declares none: any kind but `durationMonths` reports the
    /// parsed number under the neutral `parsedValue…` names as well (Python
    /// twin: `analysis_endpoints.endpoint_values`).
    static func sampledOutcomes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String],
        numericParserKind: String?
    ) -> [SampledOutcome] {
        var outcomes: [SampledOutcome] = [
            SampledOutcome(name: "wordCount", reading: { Double($0.wordCount) }),
            SampledOutcome(name: "distinct2", reading: { Double($0.distinct2) }),
        ]
        for concept in concepts.sorted() {
            outcomes.append(
                SampledOutcome(
                    name: "\(concept)\(StudyAnalysisOutcomes.markerDensitySuffix)",
                    reading: { Double($0.markerDensity[concept] ?? 0) }))
        }
        // Reasoning-style features join the same paired machinery, one
        // numeric metric per feature (declared taxonomy order).
        for id in styleFeatureIDs {
            outcomes.append(
                SampledOutcome(
                    name: "\(StudyAnalysisOutcomes.reasoningStylePrefix)\(id)",
                    reading: { $0.reasoningStyle[id] ?? 0 }))
        }
        for concept in Set(rows.flatMap { $0.readerScores.keys }).sorted() {
            outcomes.append(
                SampledOutcome(
                    name: "\(StudyAnalysisOutcomes.readerScorePrefix)\(concept)",
                    reading: { $0.readerScores[concept] }))
        }
        outcomes.append(
            SampledOutcome(
                name: "meanMonths", reading: \.parsedValue, reduction: .parsedMean))
        outcomes.append(
            SampledOutcome(
                name: "monthsSpread", reading: \.parsedValue,
                reduction: .parsedSpread, pairsWithinItem: false))
        if let numericParserKind, numericParserKind != "durationMonths" {
            outcomes.append(
                SampledOutcome(
                    name: "parsedValueMean", reading: \.parsedValue,
                    reduction: .parsedMean, pairsWithinItem: false))
            outcomes.append(
                SampledOutcome(
                    name: "parsedValueSpread", reading: \.parsedValue,
                    reduction: .parsedSpread, pairsWithinItem: false))
        }
        outcomes.append(
            SampledOutcome(
                name: "choiceRate", reading: { $0.choseTarget.map { $0 ? 1 : 0 } }))
        return outcomes
    }

    static func effectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        targetLogOdds: [ItemReadout] = [], numericParserKind: String? = nil,
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        applyCorrection(
            sampledEffectSizes(
                rows: rows,
                outcomes: sampledOutcomes(
                    rows: rows, concepts: concepts,
                    styleFeatureIDs: styleFeatureIDs,
                    numericParserKind: numericParserKind),
                replicates: replicates)
                + ordinalEffectSizes(choiceReadouts, replicates: replicates)
                + readoutEffectSizes(
                    metric: "choiceLogOdds", readouts: targetLogOdds,
                    replicates: replicates),
            phase: phase)
    }

    /// What a multi-agent analysis says about its unit, written beside its
    /// effect rows as `unit-of-analysis.json`. Python twin: the same file
    /// from `analysis_workflow.analyze`, same keys and words.
    struct TranscriptUnit: Codable, Equatable {
        static let reasonText =
            "turns within a transcript are dependent; each transcript is "
            + "reduced to its mean paired difference before testing"
        /// The log line for a skipped outcome (Python twin, same words).
        static let skippedNote =
            "effect sizes: some endpoints skipped — 1 transcript per "
            + "condition supports a point estimate but no interval. "
            + "Re-run with samplesPerItem > 1."

        var unitOfAnalysis = "transcript"
        var reason = reasonText
        /// True when an outcome a condition measured paired fewer than two
        /// transcripts, so it has no row.
        var skippedForSingleTranscript: Bool
    }

    /// A multi-agent study's paired effects, per CONVERSATION (plan D1).
    ///
    /// Turns are not independent observations: turn k is conditioned on
    /// turns 1..k-1, and after the first turn the two arms diverge, so what
    /// pairs across conditions is a script POSITION within one play-through.
    /// A turn's value pairs with the baseline's value at the same turn of the
    /// same replicate; the turn differences are averaged within each
    /// transcript; and the paired bootstrap and Wilcoxon run over one value
    /// per transcript, so `n` counts transcripts, never turns. An outcome a
    /// condition measured that pairs fewer than two transcripts has no row —
    /// one transcript supports a point estimate but no interval — and sets
    /// `skippedForSingleTranscript`. Every row is a member of its outcome's
    /// correction family, and nothing is stratified: the per-turn strata are
    /// the dependent observations this estimator exists to aggregate.
    ///
    /// Python twin: `analysis_endpoints.key_records_by_transcript` and
    /// `transcript_level_diffs` in `analysis_workflow.analyze`; both engines
    /// are pinned to `Tests/Fixtures/cross-engine/panel-transcript-analysis.json`.
    /// Answer-token readouts carry no replicate, so they read as one
    /// transcript, exactly as on the Python engine. Row order follows this
    /// engine's pooled order: condition, then outcome.
    static func transcriptEffectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        targetLogOdds: [ItemReadout] = [], numericParserKind: String? = nil,
        replicates: Int = 10_000, phase: String? = nil
    ) -> (entries: [EffectSizeEntry], unit: TranscriptUnit) {
        struct Cell: Hashable {
            let condition: String
            let promptID: String
            let transcript: Int
        }
        var skipped = false
        var entries: [EffectSizeEntry] = []
        /// One outcome for one condition: its per-cell values beside the
        /// baseline's, reduced to one mean paired difference per transcript.
        func entry(
            condition: String, metric: String, values: [(Cell, Double)],
            baseline: [Cell: Double]
        ) {
            guard !values.isEmpty else { return }
            var byTranscript: [Int: [Double]] = [:]
            for (cell, value) in values {
                let partner = Cell(
                    condition: "baseline", promptID: cell.promptID,
                    transcript: cell.transcript)
                guard let base = baseline[partner] else { continue }
                byTranscript[cell.transcript, default: []].append(value - base)
            }
            let diffs = byTranscript.keys.sorted().map { key in
                let turns = byTranscript[key] ?? []
                return turns.reduce(0, +) / Double(turns.count)
            }
            guard diffs.count >= 2 else {
                skipped = true
                return
            }
            let ci = StudyStatistics.pairedBootstrapCI(
                diffs, replicates: replicates, seed: 0)
            let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
            entries.append(
                EffectSizeEntry(
                    condition: condition, metric: metric, n: ci.n,
                    meanDiff: ci.mean, ciLower: ci.ciLower, ciUpper: ci.ciUpper,
                    wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                    wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p))
        }

        var conditionOrder: [String] = []
        var seen = Set<String>()
        for condition in rows.map(\.condition) + choiceReadouts.map(\.condition)
            + targetLogOdds.map(\.condition) where condition != "baseline"
        {
            if seen.insert(condition).inserted { conditionOrder.append(condition) }
        }
        // Sampled outcomes: each (condition, turn, transcript) cell's value,
        // in the order the cells first appear in run order.
        var cellOrder: [Cell] = []
        var rowsByCell: [Cell: [MetricRow]] = [:]
        for row in inRunOrder(rows) {
            let cell = Cell(
                condition: row.condition, promptID: row.promptID,
                transcript: row.replicate ?? 0)
            if rowsByCell[cell] == nil { cellOrder.append(cell) }
            rowsByCell[cell, default: []].append(row)
        }
        let outcomes = sampledOutcomes(
            rows: rows, concepts: concepts, styleFeatureIDs: styleFeatureIDs,
            numericParserKind: numericParserKind)
        // Answer-token readouts: one value per (condition, item), last
        // readout wins, as in the pooled analysis.
        func readoutValues(_ pairs: [(condition: String, promptID: String, value: Double)])
            -> [Cell: Double]
        {
            var values: [Cell: Double] = [:]
            for pair in pairs {
                values[Cell(condition: pair.condition, promptID: pair.promptID, transcript: 0)] =
                    pair.value
            }
            return values
        }
        let ordinal = readoutValues(
            choiceReadouts.compactMap { readout in
                guard readout.source == "instrument", let position = readout.ordinalPosition
                else { return nil }
                return (readout.condition, readout.promptID, position)
            })
        let logOdds = readoutValues(
            targetLogOdds.map { ($0.condition, $0.promptID, $0.value) })
        for condition in conditionOrder {
            for outcome in outcomes {
                var baseline: [Cell: Double] = [:]
                var values: [(Cell, Double)] = []
                for cell in cellOrder
                where cell.condition == condition || cell.condition == "baseline" {
                    guard let value = outcome.itemValue(rowsByCell[cell] ?? []) else { continue }
                    if cell.condition == "baseline" {
                        baseline[cell] = value
                    } else {
                        values.append((cell, value))
                    }
                }
                entry(
                    condition: condition, metric: outcome.name, values: values,
                    baseline: baseline)
            }
        }
        // The readout outcomes follow the sampled ones, as in the pooled
        // analysis (`effectSizes`).
        for (metric, readings) in [("ordinalPosition", ordinal), ("choiceLogOdds", logOdds)] {
            let baseline = readings.filter { $0.key.condition == "baseline" }
            for condition in conditionOrder {
                let values = readings.filter { $0.key.condition == condition }
                    .sorted { $0.key.promptID < $1.key.promptID }
                    .map { ($0.key, $0.value) }
                entry(condition: condition, metric: metric, values: values, baseline: baseline)
            }
        }
        return (
            applyCorrection(entries, phase: phase),
            TranscriptUnit(skippedForSingleTranscript: skipped)
        )
    }

    static func correctionMethod(phase: String?) -> String {
        phase == "confirm" ? "holm" : "bh"
    }

    static func applyCorrection(
        _ entries: [EffectSizeEntry], phase: String?
    ) -> [EffectSizeEntry] {
        let method = correctionMethod(phase: phase)
        var result = entries
        var families: [String: [Int]] = [:]
        for (index, entry) in entries.enumerated() {
            families[entry.metric, default: []].append(index)
        }
        for indices in families.values {
            let usable = indices.filter { result[$0].wilcoxonP != nil }
            let dense = usable.compactMap { result[$0].wilcoxonP }
            let adjusted =
                method == "holm"
                ? StudyStatistics.holm(dense) : StudyStatistics.bhFDR(dense)
            for (offset, index) in usable.enumerated() {
                result[index].adjustedP = adjusted[offset]
            }
            for index in indices {
                result[index].correction = method
            }
        }
        return result
    }

    /// The run's canonical row order, (seed, promptIndex, promptID): it fixes
    /// the order the paired differences enter the bootstrap, so a given run
    /// always reproduces the same interval.
    private static func inRunOrder(_ rows: [MetricRow]) -> [MetricRow] {
        rows.sorted {
            ($0.seed, $0.promptIndex, $0.promptID)
                < ($1.seed, $1.promptIndex, $1.promptID)
        }
    }

    /// What identifies one sample WITHIN an item, for pairing sample k of a
    /// condition with sample k of the baseline: the record's `sampleIndex`.
    /// Never the seed where an index exists — a derived seed includes the
    /// condition name, so the two sides of a pair carry different seeds by
    /// design (`StudySampling.deriveSeed`). Rows from records that predate
    /// the stamp fall back to the seed, which every such run shared across
    /// conditions.
    private static func sampleKey(_ row: MetricRow) -> String {
        row.sampleIndex.map { "index:\($0)" } ?? "seed:\(row.seed)"
    }

    /// The paired unit is the ITEM. Each (condition, item) cell is averaged
    /// over its samples, and the condition's mean is paired with the same
    /// item's baseline mean by promptID — so `n` counts items, however many
    /// responses were sampled per item, and a study with one response per
    /// item gets exactly the per-item differences it always did. Server
    /// twin: `analysis_endpoints.endpoint_values` and the pooled loop in
    /// `analysis_workflow.analyze`; both engines are pinned to the same
    /// records by `Tests/Fixtures/cross-engine/sampled-effect-pairing.json`.
    ///
    /// When `stratum` is set the rows have already been restricted to one
    /// stratum's items; every produced entry carries the stratification
    /// provenance, and `unit` says what one paired difference is. A stratum
    /// in which several items pair is item-level, like the pooled row. A
    /// stratum in which exactly ONE item pairs drops to that item's sample
    /// axis when at least two of its samples pair (sample k with baseline
    /// sample k): `unit` is then "sample", and the row is a within-item
    /// diagnostic rather than an item-level effect. Server twin:
    /// `analysis_endpoints.stratified_effect_rows`.
    ///
    /// Pairing is per OUTCOME: an item pairs for an outcome when both arms
    /// have a value for it. Every response has a word count, so the surface
    /// measures pair every item both arms measured; a response the numeric
    /// parser could not read has no parsed number, so an item whose baseline
    /// cell holds no readable response is left out of `meanMonths` alone.
    private static func sampledEffectSizes(
        rows: [MetricRow], outcomes: [SampledOutcome],
        replicates: Int, stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        let baselineByItem = Dictionary(
            grouping: inRunOrder(rows.filter { $0.condition == "baseline" }),
            by: \.promptID)
        guard !baselineByItem.isEmpty else { return [] }

        // Conditions in first-appearance order; items in the order they
        // first appear among the condition's rows in run order, so the
        // bootstrap draws are reproducible for a given run (and, with one
        // response per item, are the draws this analysis has always made).
        var conditionOrder: [String] = []
        var seen = Set<String>()
        for row in rows where row.condition != "baseline" {
            if seen.insert(row.condition).inserted { conditionOrder.append(row.condition) }
        }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let conditionRows = inRunOrder(rows.filter { $0.condition == condition })
            var itemOrder: [String] = []
            var rowsByItem: [String: [MetricRow]] = [:]
            for row in conditionRows {
                if rowsByItem[row.promptID] == nil { itemOrder.append(row.promptID) }
                rowsByItem[row.promptID, default: []].append(row)
            }
            for outcome in outcomes {
                // An item pairs when both arms have a value for this
                // outcome — for the surface measures, whenever both arms
                // measured the item at least once.
                var pairedItems: [String] = []
                var diffs: [Double] = []
                for item in itemOrder {
                    guard let baselineCell = baselineByItem[item],
                        let value = outcome.itemValue(rowsByItem[item] ?? []),
                        let baseline = outcome.itemValue(baselineCell)
                    else { continue }
                    pairedItems.append(item)
                    diffs.append(value - baseline)
                }
                guard !diffs.isEmpty else { continue }
                var withinItem = false
                if stratum != nil, outcome.pairsWithinItem, pairedItems.count == 1,
                    let item = pairedItems.first
                {
                    var baselineBySample: [String: Double] = [:]
                    for row in baselineByItem[item] ?? [] {
                        if let reading = outcome.reading(row) {
                            baselineBySample[sampleKey(row)] = reading
                        }
                    }
                    let sampleDiffs = (rowsByItem[item] ?? []).compactMap { row -> Double? in
                        guard let reading = outcome.reading(row),
                            let baseline = baselineBySample[sampleKey(row)]
                        else { return nil }
                        return reading - baseline
                    }
                    if sampleDiffs.count >= 2 {
                        diffs = sampleDiffs
                        withinItem = true
                    }
                }
                let ci = StudyStatistics.pairedBootstrapCI(
                    diffs, replicates: replicates, seed: 0)
                let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
                entries.append(
                    EffectSizeEntry(
                        condition: condition,
                        metric: outcome.name,
                        n: ci.n,
                        meanDiff: ci.mean,
                        ciLower: ci.ciLower,
                        ciUpper: ci.ciUpper,
                        wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                        wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                        stratifyBy: stratum?.family,
                        stratum: stratum?.label,
                        unit: stratum.map { _ in withinItem ? "sample" : "item" },
                        // The unit IS the estimand: one difference per item
                        // is the pooled estimand restricted to this stratum
                        // and belongs in the correction family; several
                        // draws of one item are a within-item variability
                        // read and are reported as a diagnostic instead.
                        estimand: stratum.map { _ in
                            withinItem
                                ? EffectSizeEstimand.withinItemSamples
                                : EffectSizeEstimand.itemLevel
                        },
                        inference: stratum.map { _ in
                            withinItem
                                ? EffectSizeInference.diagnostic
                                : EffectSizeInference.corrected
                        }))
            }
        }
        return entries
    }

    /// The ordinalScale instrument's paired effects: per-item ladder-position
    /// differences against the SAME-item baseline instrument readout (one
    /// deterministic readout per condition × prompt, so pairing is by
    /// promptID), through the same bootstrap CI + Wilcoxon as every other
    /// metric — no new statistics. The metric name "ordinalPosition" is the
    /// pinned cross-engine contract (server `_endpoint_values` twin).
    private static func ordinalEffectSizes(
        _ readouts: [ReportChoiceReadout], replicates: Int,
        stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        let ordinal = readouts.filter {
            $0.source == "instrument" && $0.ordinalPosition != nil
        }
        // Defensive last-wins on a duplicated promptID, matching `report`.
        var baselineByItem: [String: Double] = [:]
        for readout in ordinal where readout.condition == "baseline" {
            baselineByItem[readout.promptID] = readout.ordinalPosition
        }
        guard !baselineByItem.isEmpty else { return [] }
        var conditionOrder: [String] = []
        var seen = Set<String>()
        for readout in ordinal where readout.condition != "baseline" {
            if seen.insert(readout.condition).inserted {
                conditionOrder.append(readout.condition)
            }
        }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let diffs: [Double] =
                ordinal
                .filter { $0.condition == condition }
                .sorted { $0.promptID < $1.promptID }
                .compactMap { readout in
                    guard let base = baselineByItem[readout.promptID],
                        let position = readout.ordinalPosition
                    else { return nil }
                    return position - base
                }
            guard !diffs.isEmpty else { continue }
            let ci = StudyStatistics.pairedBootstrapCI(
                diffs, replicates: replicates, seed: 0)
            let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
            entries.append(
                EffectSizeEntry(
                    condition: condition,
                    metric: "ordinalPosition",
                    n: ci.n,
                    meanDiff: ci.mean,
                    ciLower: ci.ciLower,
                    ciUpper: ci.ciUpper,
                    wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                    wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                    stratifyBy: stratum?.family,
                    stratum: stratum?.label,
                    // One deterministic readout per (condition, prompt):
                    // the instrument has no sample axis, so a stratified
                    // ordinal pair is always per-item — item-level, and
                    // therefore always a member of the correction family.
                    unit: stratum.map { _ in "item" },
                    estimand: stratum.map { _ in EffectSizeEstimand.itemLevel },
                    inference: stratum.map { _ in EffectSizeInference.corrected }))
        }
        return entries
    }

    /// Paired effects for an outcome with ONE deterministic value per
    /// (condition, item) — `choiceLogOdds`, the target option's log-odds
    /// from the answer-token readout. The item's value in a condition pairs
    /// with the same item's baseline value by promptID, through the same
    /// bootstrap CI + Wilcoxon as every other outcome. Python twin: the
    /// `choiceLogOdds` endpoint of `analysis_endpoints.endpoint_values`.
    ///
    /// A (condition, item) read twice keeps its last readout, as on the
    /// Python engine. There is no sample axis, so a stratified row is
    /// always item-level and always a member of its correction family.
    private static func readoutEffectSizes(
        metric: String, readouts: [ItemReadout], replicates: Int,
        stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        var conditionOrder: [String] = []
        var valuesByCondition: [String: [String: Double]] = [:]
        for readout in readouts {
            if valuesByCondition[readout.condition] == nil, readout.condition != "baseline" {
                conditionOrder.append(readout.condition)
            }
            valuesByCondition[readout.condition, default: [:]][readout.promptID] = readout.value
        }
        guard let baseline = valuesByCondition["baseline"], !baseline.isEmpty else { return [] }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let values = valuesByCondition[condition] ?? [:]
            let diffs: [Double] = values.keys.sorted().compactMap { item in
                guard let value = values[item], let base = baseline[item] else { return nil }
                return value - base
            }
            guard !diffs.isEmpty else { continue }
            let ci = StudyStatistics.pairedBootstrapCI(
                diffs, replicates: replicates, seed: 0)
            let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
            entries.append(
                EffectSizeEntry(
                    condition: condition,
                    metric: metric,
                    n: ci.n,
                    meanDiff: ci.mean,
                    ciLower: ci.ciLower,
                    ciUpper: ci.ciUpper,
                    wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                    wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                    stratifyBy: stratum?.family,
                    stratum: stratum?.label,
                    unit: stratum.map { _ in "item" },
                    estimand: stratum.map { _ in EffectSizeEstimand.itemLevel },
                    inference: stratum.map { _ in EffectSizeInference.corrected }))
        }
        return entries
    }

    static func stratificationFamilies(
        factorsByItem: [String: [String: String]], items: Set<String>
    ) -> [(name: String, strata: [(label: String, items: Set<String>)])] {
        var families: [(name: String, strata: [(label: String, items: Set<String>)])] = [
            (
                name: "promptID",
                strata: items.sorted().map { (label: $0, items: Set([$0])) }
            )
        ]
        let keys = Set(factorsByItem.values.flatMap(\.keys)).sorted()
        for key in keys {
            var strata: [String: Set<String>] = [:]
            for item in items {
                if let level = factorsByItem[item]?[key] {
                    strata[level, default: []].insert(item)
                }
            }
            if strata.count >= 2 {
                families.append(
                    (
                        name: key,
                        strata: strata.sorted { $0.key < $1.key }
                            .map { (label: $0.key, items: $0.value) }
                    ))
            }
        }
        if keys.count >= 2 {
            var cells: [String: Set<String>] = [:]
            for item in items {
                let levels = factorsByItem[item] ?? [:]
                let values = keys.compactMap { levels[$0] }
                guard values.count == keys.count else { continue }
                cells[values.joined(separator: "×"), default: []].insert(item)
            }
            if cells.count >= 2 {
                families.append(
                    (
                        name: keys.joined(separator: "×"),
                        strata: cells.sorted { $0.key < $1.key }
                            .map { (label: $0.key, items: $0.value) }
                    ))
            }
        }
        return families
    }

    static func stratifiedEffectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        targetLogOdds: [ItemReadout] = [], numericParserKind: String? = nil,
        factorsByItem: [String: [String: String]],
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        var items = Set(rows.map(\.promptID))
        for readout in choiceReadouts
        where readout.source == "instrument" && readout.ordinalPosition != nil {
            items.insert(readout.promptID)
        }
        items.formUnion(targetLogOdds.map(\.promptID))
        guard !items.isEmpty else { return [] }
        // One outcome list for every stratum, built from ALL the rows: the
        // same outcomes the pooled rows report.
        let outcomes = sampledOutcomes(
            rows: rows, concepts: concepts, styleFeatureIDs: styleFeatureIDs,
            numericParserKind: numericParserKind)
        var entries: [EffectSizeEntry] = []
        for family in stratificationFamilies(
            factorsByItem: factorsByItem, items: items)
        {
            var familyEntries: [EffectSizeEntry] = []
            for (label, members) in family.strata {
                let stratum = (family: family.name, label: label)
                familyEntries += sampledEffectSizes(
                    rows: rows.filter { members.contains($0.promptID) },
                    outcomes: outcomes,
                    replicates: replicates, stratum: stratum)
                familyEntries += ordinalEffectSizes(
                    choiceReadouts.filter { members.contains($0.promptID) },
                    replicates: replicates, stratum: stratum)
                familyEntries += readoutEffectSizes(
                    metric: "choiceLogOdds",
                    readouts: targetLogOdds.filter { members.contains($0.promptID) },
                    replicates: replicates, stratum: stratum)
            }
            // The phase's correction, per metric WITHIN this family —
            // applyCorrection groups by metric over exactly the entries it
            // is handed — and ONLY over the item-level rows.
            //
            // A `withinItemSamples` row pairs several draws of the SAME item
            // against that item's baseline draws: it measures within-item
            // variability, not an item-level effect, so it is not an
            // independent test of the pre-registered hypothesis. Correcting
            // across those rows inflated the family (shrinking every real
            // row's adjustedP) AND stamped an `adjustedP` that read as a
            // citable test. They are emitted as `diagnostic` instead — raw
            // Wilcoxon and bootstrap CI kept, no adjustedP, no correction
            // stamp. Order is preserved so the CSV row order is unchanged.
            let corrected = applyCorrection(
                familyEntries.filter { !$0.isWithinItemSamples }, phase: phase)
            var correctedRows = corrected.makeIterator()
            entries += familyEntries.map { entry in
                entry.isWithinItemSamples
                    ? entry : (correctedRows.next() ?? entry)
            }
        }
        return entries
    }

    static func reasoningStyleReport(
        rows: [MetricRow], style: PinnedReasoningStyle?
    ) -> ReasoningStyleConditionReport? {
        guard let style, !rows.isEmpty else { return nil }
        let features = Dictionary(
            uniqueKeysWithValues: style.taxonomy.featureIDs.map { id in
                (
                    id,
                    ReasoningStyleFeatureStat(
                        mean: rows.map { $0.reasoningStyle[id] ?? 0 }.reduce(0, +)
                            / Double(rows.count),
                        n: rows.count)
                )
            })
        return ReasoningStyleConditionReport(
            taxonomy: style.taxonomy.name, taxonomyHash: style.hash,
            taxonomyFile: style.path, diagnosticOnly: true,
            features: features)
    }
}
