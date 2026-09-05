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

    static func effectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        applyCorrection(
            sampledEffectSizes(
                rows: rows, concepts: concepts,
                styleFeatureIDs: styleFeatureIDs, replicates: replicates)
                + ordinalEffectSizes(choiceReadouts, replicates: replicates),
            phase: phase)
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

    /// When `stratum` is set the rows have already been restricted to one
    /// stratum's items; every produced entry carries the stratification
    /// provenance, and `unit` says what one paired difference is: "item"
    /// when each joined item contributes exactly one pair, "sample" when
    /// the pairs resolve within items (multiple seeds of the same item).
    private static func sampledEffectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String],
        replicates: Int, stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        var baselineByKey: [String: MetricRow] = [:]
        for row in rows where row.condition == "baseline" {
            baselineByKey["\(row.seed)::\(row.promptID)"] = row
        }
        guard !baselineByKey.isEmpty else { return [] }

        var metrics: [(name: String, value: (MetricRow) -> Double)] = [
            ("wordCount", { Double($0.wordCount) }),
            ("distinct2", { Double($0.distinct2) }),
        ]
        for concept in concepts.sorted() {
            metrics.append(
                ("\(concept)MarkerDensity", { Double($0.markerDensity[concept] ?? 0) }))
        }
        // Reasoning-style features join the same paired machinery, one
        // numeric metric per feature (declared taxonomy order).
        for id in styleFeatureIDs {
            metrics.append(("rs_\(id)", { $0.reasoningStyle[id] ?? 0 }))
        }

        // Conditions in first-appearance order; items in a deterministic
        // (seed, promptIndex, promptID) order so the bootstrap draws are
        // reproducible for a given run.
        var conditionOrder: [String] = []
        var seen = Set<String>()
        for row in rows where row.condition != "baseline" {
            if seen.insert(row.condition).inserted { conditionOrder.append(row.condition) }
        }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let conditionRows =
                rows
                .filter { $0.condition == condition }
                .sorted {
                    ($0.seed, $0.promptIndex, $0.promptID)
                        < ($1.seed, $1.promptIndex, $1.promptID)
                }
            for metric in metrics {
                var diffs: [Double] = []
                var pairedItems = Set<String>()
                for row in conditionRows {
                    guard let base = baselineByKey["\(row.seed)::\(row.promptID)"] else {
                        continue
                    }
                    diffs.append(metric.value(row) - metric.value(base))
                    pairedItems.insert(row.promptID)
                }
                guard !diffs.isEmpty else { continue }
                let ci = StudyStatistics.pairedBootstrapCI(
                    diffs, replicates: replicates, seed: 0)
                let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
                entries.append(
                    EffectSizeEntry(
                        condition: condition,
                        metric: metric.name,
                        n: ci.n,
                        meanDiff: ci.mean,
                        ciLower: ci.ciLower,
                        ciUpper: ci.ciUpper,
                        wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                        wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                        stratifyBy: stratum?.family,
                        stratum: stratum?.label,
                        unit: stratum.map {
                            _ in
                            diffs.count == pairedItems.count
                                ? "item" : "sample"
                        },
                        // The unit IS the estimand: one pair per item is the
                        // pooled estimand restricted to this stratum and
                        // belongs in the correction family; several draws of
                        // the same item are a within-item variability read
                        // and are reported as a diagnostic instead.
                        estimand: stratum.map {
                            _ in
                            diffs.count == pairedItems.count
                                ? EffectSizeEstimand.itemLevel
                                : EffectSizeEstimand.withinItemSamples
                        },
                        inference: stratum.map {
                            _ in
                            diffs.count == pairedItems.count
                                ? EffectSizeInference.corrected
                                : EffectSizeInference.diagnostic
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
        factorsByItem: [String: [String: String]],
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        var items = Set(rows.map(\.promptID))
        for readout in choiceReadouts
        where readout.source == "instrument" && readout.ordinalPosition != nil {
            items.insert(readout.promptID)
        }
        guard !items.isEmpty else { return [] }
        var entries: [EffectSizeEntry] = []
        for family in stratificationFamilies(
            factorsByItem: factorsByItem, items: items)
        {
            var familyEntries: [EffectSizeEntry] = []
            for (label, members) in family.strata {
                let stratum = (family: family.name, label: label)
                familyEntries += sampledEffectSizes(
                    rows: rows.filter { members.contains($0.promptID) },
                    concepts: concepts, styleFeatureIDs: styleFeatureIDs,
                    replicates: replicates, stratum: stratum)
                familyEntries += ordinalEffectSizes(
                    choiceReadouts.filter { members.contains($0.promptID) },
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
