import Foundation
import SteeringKit

/// Value-only rules; no workspace discovery or artifact writes.
enum StudyAnalysisCalculator {
    typealias MetricRow = ExperimentTasks.MetricRow
    typealias ReportChoiceReadout = ExperimentTasks.ReportChoiceReadout

    static func analyze(_ input: StudyAnalysisInput) throws -> StudyAnalysisResult {
        let manifest = input.manifest
        let style = input.style
        let text = input.generations
        let exclusionRules = manifest.exclusionRules ?? []
        let exclusionChecks = input.exclusionChecks
        let declaredTargets = input.declaredTargets
        var diagnostics: [StudyAnalysisDiagnostic] = []
        func log(_ text: String) { diagnostics.append(StudyAnalysisDiagnostic(text: text)) }
        let exclusionEndpoints = Set(
            exclusionRules
                .filter { $0.rule != ExclusionEngine.ruleFailedAttentionCheck }
                .map(ExclusionEngine.resolvedEndpoint))
        var exclusionViews: [ExclusionEngine.RecordView] = []
        var instrumentExclusionViews: [ExclusionEngine.InstrumentRecordView] = []

        var rows: [MetricRow] = []
        var ordinalReadouts: [ReportChoiceReadout] = []
        var conceptSet = Set<String>()
        // D3: per-condition option logprobs, for the distance-from-boundary
        // diagnostics written alongside the effect sizes.
        var optionLogprobsByCondition: [String: [[String: Double]]] = [:]
        // Phase 3: per-item choice deltas, paired to the same item's baseline
        // readout. Collected in run order; the pairing and the sort happen in
        // ChoiceDeltas (server twin: choice_deltas.rows).
        var choiceReadouts: [ChoiceDeltas.Readout] = []
        // Declared-target map from the PINNED task file (open-issues #6, the
        // exact authority — server twin: `tasks.analyze`'s `declared_targets`).
        // It keeps a mixed instrument's legitimate endpoint (an item with both
        // a declared A/B target and an ordinal readout on one record) while
        // dropping the ordinalScale items whose "target" was synthesized. An
        // unloadable prompts file falls back to the per-record ladder inside
        // `ChoiceDeltas.targetIsDeclared`.
        // Item → declared factor levels (arm/caseID + the factorial
        // `factors` object), for the stratified effect rows. Records carry
        // the item metadata verbatim, so no rejoin of the task-prompts file;
        // first record per item wins (items stamp identically).
        var factorsByItem: [String: [String: String]] = [:]
        // Every condition name that actually produced a record. A run with
        // no NON-baseline condition has nothing to pair against — analyze
        // still writes its (empty) artifacts and exits 0, so the fact has to
        // be said out loud on stderr (WP0 dry run #0, P0-2).
        var conditionsSeen = Set<String>()
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            guard
                let record = try? decoder.decode(
                    AnalysisGeneration.self, from: Data(line.utf8))
            else { continue }
            conditionsSeen.insert(record.condition)
            if factorsByItem[record.promptID] == nil {
                var levels: [String: String] = [:]
                if let arm = record.arm, !arm.isEmpty { levels["arm"] = arm }
                if let caseID = record.caseID, !caseID.isEmpty {
                    levels["caseID"] = caseID
                }
                for (key, value) in record.factors ?? [:] where !value.isEmpty {
                    levels[key] = value
                }
                factorsByItem[record.promptID] = levels
            }
            if let logprobs = record.optionLogprobs, !logprobs.isEmpty {
                optionLogprobsByCondition[record.condition, default: []]
                    .append(logprobs)
            }
            if record.instrument != nil {
                // Instrument records carry no sampled metrics, but an
                // ordinalScale run's ladder positions are per-item numeric
                // data for the SAME paired effect-size machinery — and
                // under scope allRecordTypes the declared rules consider
                // the readout itself (its own endpoints; its cell's
                // attention evidence).
                if !exclusionRules.isEmpty {
                    instrumentExclusionViews.append(
                        ExclusionEngine.InstrumentRecordView(
                            condition: record.condition,
                            promptID: record.promptID,
                            endpoints: analysisEndpoints(
                                jsonLine: Data(line.utf8),
                                names: exclusionEndpoints)))
                }
                // Every answer-token readout with a DECLARED target is
                // collected, including one whose logOdds is missing or has no
                // entry for that target: that is an unreadable measurement,
                // counted as such downstream, never quietly absent from the
                // coverage numbers. A readout whose target was never declared
                // (open-issues #6) is not an unreadable choice — it is not a
                // choice measurement at all, so it stays out of the table
                // rather than inflating its skip counts.
                if record.instrument == ChoiceDeltas.instrument,
                    ChoiceDeltas.targetIsDeclared(
                        promptID: record.promptID,
                        targetSource: record.targetSource,
                        ordinalPosition: record.ordinalPosition,
                        declaredTargets: declaredTargets)
                {
                    choiceReadouts.append(
                        ChoiceDeltas.Readout(
                            condition: record.condition,
                            promptID: record.promptID,
                            sampleIndex: record.sampleIndex.map { String($0) } ?? "",
                            target: record.target ?? "",
                            logOdds: record.logOdds ?? [:],
                            choiceProbability: record.choiceProbability ?? [:],
                            selected: record.selected ?? ""))
                }
                if let position = record.ordinalPosition {
                    ordinalReadouts.append(
                        ReportChoiceReadout(
                            condition: record.condition,
                            promptID: record.promptID,
                            sampleIndex: nil,
                            source: "instrument",
                            selected: "",
                            target: nil,
                            ordinalPosition: position))
                }
                continue
            }
            guard let wordCount = record.wordCount else { continue }
            if !exclusionRules.isEmpty {
                exclusionViews.append(
                    ExclusionEngine.RecordView(
                        condition: record.condition,
                        seed: record.seed ?? 0,
                        promptID: record.promptID,
                        output: record.output ?? "",
                        endpoints: analysisEndpoints(
                            jsonLine: Data(line.utf8),
                            names: exclusionEndpoints)))
            }
            let markerDensity = record.markerDensity ?? [:]
            conceptSet.formUnion(markerDensity.keys)
            let reasoningStyle: [String: Double] =
                if let style, let output = record.output {
                    style.taxonomy.score(output)
                } else {
                    [:]
                }
            rows.append(
                MetricRow(
                    condition: record.condition,
                    seed: record.seed ?? 0,
                    promptIndex: record.promptIndex ?? 0,
                    promptID: record.promptID,
                    wordCount: wordCount,
                    distinct2: record.distinct2 ?? 0,
                    markerDensity: markerDensity,
                    reasoningStyle: reasoningStyle))
        }
        // Choice readouts count as analyzable material alongside sampled
        // generations and ordinal readouts: a study whose whole instrument is
        // the answer-token logprob (no prose arm at all) has per-item deltas
        // to report, and refusing it here would make choice-deltas.csv
        // unreachable on this engine. The server has never had this guard.
        guard !rows.isEmpty || !ordinalReadouts.isEmpty || !choiceReadouts.isEmpty
        else {
            throw ExperimentError(
                reason: "run '\(input.sourceRunName)' has no sampled "
                    + "generations or instrument readouts to analyze")
        }
        // Records exist, but every one of them is the baseline: the paired
        // statistics have no contrast to compute, so this analysis is
        // structurally empty however many generations it read. A warning,
        // not a refusal — the run's own artifacts are still legitimate
        // material — but never silent (WP0 dry run #0, P0-2). Server twin:
        // the same line in `tasks.analyze`.
        if let warning = baselineOnlyAnalysisWarning(
            runName: input.sourceRunName, conditions: conditionsSeen)
        {
            diagnostics.append(StudyAnalysisDiagnostic(text: warning, standardError: true))
        }
        var exclusionStamp: ExclusionStamp?
        if !exclusionRules.isEmpty {
            let outcome = ExclusionEngine.evaluate(
                rules: exclusionRules, checks: exclusionChecks,
                views: exclusionViews,
                instrumentViews: instrumentExclusionViews)
            exclusionStamp = outcome.stamp
            rows = rows.filter {
                !outcome.excludedKeys.contains(
                    ExclusionEngine.rowKey(
                        condition: $0.condition, seed: $0.seed,
                        promptID: $0.promptID))
            }
            ordinalReadouts = ordinalReadouts.filter {
                !outcome.excludedInstrumentKeys.contains(
                    ExclusionEngine.instrumentKey(
                        condition: $0.condition, promptID: $0.promptID))
            }
            // Same drop for the choice-delta table: an excluded readout must
            // not reappear as a citable per-item delta.
            choiceReadouts = choiceReadouts.filter {
                !outcome.excludedInstrumentKeys.contains(
                    ExclusionEngine.instrumentKey(
                        condition: $0.condition, promptID: $0.promptID))
            }
            log(
                "exclusions: \(outcome.stamp.excludedRecords) record(s) "
                    + "excluded by \(exclusionRules.count) declared rule(s); "
                    + "surviving N per condition: "
                    + outcome.stamp.survivingN.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }
                    .joined(separator: ", "))
        }
        let pooledEntries = StudyAnalysisStatistics.effectSizes(
            rows: rows, concepts: conceptSet.sorted(),
            styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
            choiceReadouts: ordinalReadouts,
            phase: manifest.phase)
        // Per-cell strata beside the pooled rows (same file, extra rows):
        // pooling across items has both hidden a real single-cell effect
        // behind saturated cells and manufactured pooled effects from one
        // cell's parse garbage. Pooled entries keep their exact semantics
        // and correction family; each stratified family is corrected
        // independently. Server twin: tasks.analyze.
        let entries =
            pooledEntries
            + StudyAnalysisStatistics.stratifiedEffectSizes(
                rows: rows, concepts: conceptSet.sorted(),
                styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
                choiceReadouts: ordinalReadouts,
                factorsByItem: factorsByItem,
                phase: manifest.phase)

        var marginReports: [String: ChoiceMarginDiagnostics.Report] = [:]
        for (condition, logprobs) in optionLogprobsByCondition {
            let block = ChoiceMarginDiagnostics.report(
                optionLogprobsPerItem: logprobs)
            if block.scoredItems > 0 { marginReports[condition] = block }
        }
        return StudyAnalysisResult(
            entries: entries, pooledCount: pooledEntries.count,
            sampledCount: rows.count, ordinalCount: ordinalReadouts.count,
            exclusions: exclusionStamp, choiceDeltas: ChoiceDeltas.table(choiceReadouts),
            margins: marginReports, diagnostics: diagnostics)
    }

    static func rescoreStyle(_ input: StudyAnalysisInput) throws -> StudyStyleResult {
        guard let style = input.style else {
            throw ExperimentError(reason: "rescore-style requires a pinned taxonomy")
        }
        let text = input.generations
        var rows: [MetricRow] = []
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            guard
                let record = try? decoder.decode(
                    AnalysisGeneration.self, from: Data(line.utf8)),
                record.instrument == nil,
                let output = record.output
            else { continue }
            rows.append(
                MetricRow(
                    condition: record.condition,
                    seed: record.seed ?? 0,
                    promptIndex: record.promptIndex ?? 0,
                    promptID: record.promptID,
                    wordCount: record.wordCount ?? 0,
                    distinct2: record.distinct2 ?? 0,
                    markerDensity: [:],
                    reasoningStyle: style.taxonomy.score(output)))
        }
        guard !rows.isEmpty else {
            throw ExperimentError(
                reason: "run '\(input.sourceRunName)' has no sampled "
                    + "generations to rescore")
        }

        let grouped = Dictionary(grouping: rows, by: \.condition)
        return StudyStyleResult(
            rows: rows,
            conditions: grouped.compactMapValues { conditionRows in
                StudyAnalysisStatistics.reasoningStyleReport(rows: conditionRows, style: style)
                    .map {
                        ExperimentTasks.RescoreStyleReport.ConditionBlock(features: $0.features)
                    }
            })
    }

    static func analysisEndpoints(
        jsonLine: Data, names: Set<String>
    ) -> [String: Double?] {
        guard !names.isEmpty,
            let object = try? JSONSerialization.jsonObject(with: jsonLine),
            let record = object as? [String: Any]
        else { return [:] }
        var endpoints: [String: Double?] = [:]
        for name in names {
            guard let value = record[name] else { continue }
            if value is NSNull {
                // `updateValue`, never subscript-assign: the key must appear
                // with a nil VALUE (a parse failure); the subscript would
                // remove it instead.
                endpoints.updateValue(nil, forKey: name)
            } else if let number = value as? NSNumber,
                CFGetTypeID(number) != CFBooleanGetTypeID()
            {
                endpoints.updateValue(number.doubleValue, forKey: name)
            }
        }
        return endpoints
    }

    static func baselineOnlyAnalysisWarning(
        runName: String, conditions: Set<String>
    ) -> String? {
        guard !conditions.isEmpty,
            !conditions.contains(where: { $0 != "baseline" })
        else { return nil }
        return "WARNING: run '\(runName)' contains only BASELINE records — "
            + "there is no non-baseline condition to pair against, so this "
            + "analysis will produce no effect sizes. Check the study's "
            + "conditions before citing it."
    }
}
