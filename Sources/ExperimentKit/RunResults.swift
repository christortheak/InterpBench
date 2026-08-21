import Foundation

/// Semantic decoding of a STUDY RUN directory for the Results browser —
/// the layer between `RunBrowser`'s generic bounded previews and the
/// categorical/statistics views (app gaps A3/A6/A13; team findings P3/P4).
///
/// Everything here is read-only and cross-engine: records are decoded
/// LENIENTLY from the shared camelCase contract keys, tolerating each
/// engine's extras (`seedInert` on Swift, `interventionState`/`sampleIndex`
/// on the server) and both report.json dialects. Absence is shown, never
/// invented — every derived quantity is optional when its inputs are.
///
/// Pure parsers + derivations live as static functions (unit-tested on
/// fixture strings); `Model.load` is the thin disk wrapper.
public enum RunResults {

    // MARK: - Lenient generation/choice record (A6)

    /// Tri-state for the outcome-endpoint parse keys, whose JSON contract
    /// distinguishes key-absent (endpoint does not apply) from `null`
    /// (a parse FAILURE — a first-class coherence endpoint).
    public enum ParseField<Value: Sendable & Equatable>: Sendable, Equatable {
        case absent
        case failure
        case value(Value)

        public var isFailure: Bool { self == .failure }
        public var value: Value? {
            if case .value(let value) = self { return value }
            return nil
        }
    }

    /// One generations.jsonl record, reduced to the analysis surface. Covers
    /// BOTH shapes both engines write: sampled generation records and
    /// answer-token-logprob choice records (`instrument:
    /// "answerTokenLogprob"`). Unknown keys are ignored; missing keys decode
    /// as nil — never a decode error for engine-specific extras.
    public struct Record: Sendable, Equatable {
        public var condition: String
        public var promptID: String
        public var promptIndex: Int?
        public var sampleIndex: Int?
        public var prompt: String?
        public var output: String?
        public var wordCount: Int?
        public var distinct2: Double?
        /// "answerTokenLogprob" on choice records; nil on sampled records.
        public var instrument: String?
        public var options: [String]?
        public var target: String?
        /// Choice-instrument readouts.
        public var selected: String?
        public var margin: Double?
        public var choiceProbability: [String: Double]?
        public var logOdds: [String: Double]?
        /// ordinalScale-instrument readouts (cross-engine keys
        /// "ordinalPosition"/"ordinalDistribution"), present only on
        /// instrument records of a run that declared ordinalScale.
        public var ordinalPosition: Double?
        public var ordinalDistribution: [Double]?
        /// Built-in outcome-endpoint parses (absent / null=failure / value).
        public var parsedChoice: ParseField<String> = .absent
        public var parsedMonths: ParseField<Double> = .absent
        /// Science-layer item metadata, carried verbatim.
        public var arm: String?
        public var caseID: String?
        /// Run-time resolved model revision (P4 revision chip input).
        public var modelRevision: String?
        /// Variant-condition provenance, when the record came from one.
        public var variantArtifactPath: String?
        public var variantArtifactHash: String?
        /// Server error records (a failed variant condition emits one).
        public var error: String?
        /// Multi-agent turn identity (nil on every other study kind). These
        /// are what let the transcript be rebuilt from generations.jsonl,
        /// so `transcript.md` never has to be a measurement input.
        public var speakerName: String?
        public var turnTitle: String?
        public var routedAgentIDs: [String]?
        public var replicateIndex: Int?

        /// Does this record come from a multi-agent turn?
        public var isTurn: Bool { speakerName != nil || turnTitle != nil }

        public var isChoiceRecord: Bool { instrument != nil }
        /// Option-bearing: this record's item carries a categorical endpoint.
        public var isCategorical: Bool {
            instrument != nil || options != nil || parsedChoice != .absent
        }

        public init(condition: String, promptID: String) {
            self.condition = condition
            self.promptID = promptID
        }
    }

    /// Decode one JSONL line's parsed object. Returns nil when the object
    /// carries no condition (not a study record — judgments, custom shapes).
    static func record(from dictionary: [String: Any]) -> Record? {
        guard let condition = dictionary["condition"] as? String else { return nil }
        // promptID is stringly on both engines, but tolerate numeric ids.
        let promptID: String
        if let id = dictionary["promptID"] as? String {
            promptID = id
        } else if let id = dictionary["promptID"] as? NSNumber {
            promptID = "\(id)"
        } else {
            return nil
        }
        var record = Record(condition: condition, promptID: promptID)
        record.promptIndex = intValue(dictionary["promptIndex"])
        record.sampleIndex = intValue(dictionary["sampleIndex"])
        record.prompt = dictionary["prompt"] as? String
        record.output = dictionary["output"] as? String
        record.wordCount = intValue(dictionary["wordCount"])
        record.distinct2 = doubleValue(dictionary["distinct2"])
        record.instrument = dictionary["instrument"] as? String
        record.speakerName = dictionary["speakerName"] as? String
        record.turnTitle = dictionary["turnTitle"] as? String
        record.routedAgentIDs = (dictionary["routedAgentIDs"] as? [Any])?
            .compactMap { $0 as? String }
        record.replicateIndex = intValue(dictionary["replicateIndex"])
        record.options = (dictionary["options"] as? [Any])?
            .compactMap { $0 as? String }
        record.target = dictionary["target"] as? String
        record.selected = dictionary["selected"] as? String
        record.margin = doubleValue(dictionary["margin"])
        record.choiceProbability = doubleDictionary(dictionary["choiceProbability"])
        record.logOdds = doubleDictionary(dictionary["logOdds"])
        record.ordinalPosition = doubleValue(dictionary["ordinalPosition"])
        record.ordinalDistribution = (dictionary["ordinalDistribution"] as? [Any])?
            .compactMap { doubleValue($0) }
        record.parsedChoice = parseField(dictionary, key: "parsedChoice") {
            $0 as? String
        }
        record.parsedMonths = parseField(dictionary, key: "parsedMonths") {
            doubleValue($0)
        }
        record.arm = dictionary["arm"] as? String
        record.caseID = dictionary["caseID"] as? String
        record.modelRevision = dictionary["modelRevision"] as? String
        record.variantArtifactPath = dictionary["variantArtifactPath"] as? String
        record.variantArtifactHash = dictionary["variantArtifactHash"] as? String
        record.error = dictionary["error"] as? String
        return record
    }

    private static func parseField<Value>(
        _ dictionary: [String: Any], key: String, cast: (Any) -> Value?
    ) -> ParseField<Value> {
        guard let raw = dictionary[key] else { return .absent }
        if raw is NSNull { return .failure }
        guard let value = cast(raw) else { return .failure }
        return .value(value)
    }

    private static func intValue(_ raw: Any?) -> Int? {
        (raw as? NSNumber)?.intValue
    }

    private static func doubleValue(_ raw: Any?) -> Double? {
        (raw as? NSNumber)?.doubleValue
    }

    private static func doubleDictionary(_ raw: Any?) -> [String: Double]? {
        guard let dictionary = raw as? [String: Any] else { return nil }
        return dictionary.compactMapValues { doubleValue($0) }
    }

    /// All records of a generations.jsonl text (whole lines only — pair with
    /// a bounded read). Undecodable lines are counted, never dropped
    /// silently.
    public static func records(
        fromJSONL text: String
    ) -> (records: [Record], skippedLines: Int) {
        var records: [Record] = []
        var skipped = 0
        for line in text.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard
                let object = try? JSONSerialization.jsonObject(
                    with: Data(trimmed.utf8)),
                let dictionary = object as? [String: Any],
                let record = record(from: dictionary)
            else {
                skipped += 1
                continue
            }
            records.append(record)
        }
        return (records, skipped)
    }

    // MARK: - report.json (lenient, cross-engine)

    /// One condition's block of report.json. Field presence follows the
    /// engines: Swift stamps meanWordCount/meanDistinct2/effect sizes;
    /// the server adds choiceReadouts, error strings for failed variant
    /// conditions, and (2026-07) `agreementWithBaseline` on choice-bearing
    /// runs — read when present, tolerated when absent.
    public struct ConditionSummary: Sendable, Equatable {
        public var generations: Int
        public var meanWordCount: Double?
        public var meanDistinct2: Double?
        /// Per-concept mean marker density (Swift engine stamps it; the
        /// server omits it). The manipulation-check DIAGNOSTIC per
        /// CLAUDE.md — it must survive decoding so the app can show it (F2).
        public var meanMarkerDensity: [String: Double]?
        public var choiceReadouts: Int?
        public var choiceRate: Double?
        /// ordinalScale-instrument summary both engines stamp on report.json
        /// (contract keys "ordinalMean"/"ordinalSD"; SD is the population
        /// standard deviation) — absent on non-ordinal runs.
        public var ordinalMean: Double?
        public var ordinalSD: Double?
        public var error: String?
        public var capabilityBatteryAccuracy: Double?
        public var capabilityBatteryItemCount: Int?
        /// Server-stamped exact agreement vs baseline (fraction in 0…1).
        /// Accepts a bare number, {matches,total}, or the canonical
        /// cross-engine {n,agreement} object.
        public var agreementWithBaseline: Double?
        public var agreementMatches: Int?
        public var agreementTotal: Int?

        public init(generations: Int) { self.generations = generations }
    }

    public struct Report: Sendable, Equatable {
        public var experiment: String?
        public var experimentHash: String?
        public var promptCount: Int?
        public var conditionCount: Int?
        public var seedCount: Int?
        public var conditions: [String: ConditionSummary] = [:]
        public var effectSizes: [EffectSizeRow] = []

        public init() {}
    }

    public static func report(fromJSON data: Data) -> Report? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return nil }
        var report = Report()
        report.experiment = dictionary["experiment"] as? String
        report.experimentHash = dictionary["experimentHash"] as? String
        report.promptCount = intValue(dictionary["promptCount"])
        report.conditionCount = intValue(dictionary["conditionCount"])
        report.seedCount = intValue(dictionary["seedCount"])
        if let conditions = dictionary["conditions"] as? [String: Any] {
            for (name, raw) in conditions {
                guard let block = raw as? [String: Any] else { continue }
                var summary = ConditionSummary(
                    generations: intValue(block["generations"]) ?? 0)
                summary.meanWordCount = doubleValue(block["meanWordCount"])
                summary.meanDistinct2 = doubleValue(block["meanDistinct2"])
                summary.meanMarkerDensity = doubleDictionary(block["meanMarkerDensity"])
                summary.choiceReadouts = intValue(block["choiceReadouts"])
                summary.choiceRate = doubleValue(block["choiceRate"])
                summary.ordinalMean = doubleValue(block["ordinalMean"])
                summary.ordinalSD = doubleValue(block["ordinalSD"])
                summary.error = block["error"] as? String
                if let battery = block["capabilityBattery"] as? [String: Any] {
                    summary.capabilityBatteryAccuracy = doubleValue(battery["accuracy"])
                    summary.capabilityBatteryItemCount = intValue(battery["itemCount"])
                }
                if let agreement = block["agreementWithBaseline"] {
                    if let fraction = doubleValue(agreement) {
                        summary.agreementWithBaseline = fraction
                    } else if let object = agreement as? [String: Any] {
                        let total = intValue(object["total"])
                            ?? intValue(object["n"])
                        let explicitFraction = doubleValue(object["agreement"])
                        // Only EXPLICIT match counts are stored — the field
                        // holds engine-stamped truth, never a decode-time
                        // reconstruction (the one consumer that needs a count
                        // derives it from the fraction itself).
                        let matches = intValue(object["matches"])
                            ?? intValue(object["agreed"])
                        summary.agreementMatches = matches
                        summary.agreementTotal = total
                        if let explicitFraction {
                            summary.agreementWithBaseline = explicitFraction
                        } else if let matches, let total, total > 0 {
                            summary.agreementWithBaseline =
                                Double(matches) / Double(total)
                        }
                    }
                }
                report.conditions[name] = summary
            }
        }
        if let entries = dictionary["effectSizes"] as? [[String: Any]] {
            report.effectSizes = entries.compactMap(effectSizeRow(fromJSON:))
        }
        return report
    }

    // MARK: - Choice matrix + derivations (P3)

    /// The canonical baseline condition name on both engines.
    public static let baselineConditionName = "baseline"

    /// One (item × condition) cell of the comparison table.
    public struct ChoiceCell: Sendable, Equatable {
        /// The chosen option: instrument `selected` when instrumented, else
        /// the (majority) sampled `parsedChoice`. nil with `parseFailures>0`
        /// means every sample failed to parse.
        public var choice: String?
        /// True when the chosen option equals the item's target; nil when
        /// the item declares no target or nothing was chosen.
        public var matchesTarget: Bool?
        /// Instrumented readouts (choice probability / log odds OF THE
        /// TARGET option), when an answerTokenLogprob record exists.
        public var targetProbability: Double?
        public var targetLogOdds: Double?
        public var margin: Double?
        public var instrumented: Bool = false
        /// Sampled-record tallies (0 for instrument-only cells).
        public var sampleCount: Int = 0
        public var parseFailures: Int = 0

        public init() {}
    }

    public struct ItemRow: Identifiable, Sendable, Equatable {
        public var promptID: String
        public var promptIndex: Int
        public var arm: String?
        public var target: String?
        public var options: [String]?
        public var cells: [String: ChoiceCell]
        public var id: String { promptID }
    }

    public struct ChoiceMatrix: Sendable, Equatable {
        /// Conditions in presentation order: baseline first, rest sorted.
        public var conditions: [String]
        public var items: [ItemRow]

        public var isEmpty: Bool { items.isEmpty }
    }

    /// Presentation order for condition columns: baseline first, the rest
    /// in sorted order (stable across engines and reloads).
    public static func orderedConditions(_ names: some Sequence<String>) -> [String] {
        let unique = Set(names)
        var ordered: [String] = []
        if unique.contains(baselineConditionName) {
            ordered.append(baselineConditionName)
        }
        ordered.append(
            contentsOf: unique.subtracting([baselineConditionName]).sorted())
        return ordered
    }

    /// Item × condition comparison table over the run's OPTION-BEARING
    /// items. Instrument records win the cell; sampled records contribute a
    /// majority choice + parse-failure tally.
    public static func choiceMatrix(records: [Record]) -> ChoiceMatrix {
        let categorical = records.filter { $0.isCategorical && $0.error == nil }
        let conditions = orderedConditions(categorical.map(\.condition))
        var byItem: [String: [Record]] = [:]
        for record in categorical {
            byItem[record.promptID, default: []].append(record)
        }
        let items = byItem.map { promptID, itemRecords in
            var row = ItemRow(
                promptID: promptID,
                promptIndex: itemRecords.compactMap(\.promptIndex).min() ?? 0,
                arm: itemRecords.compactMap(\.arm).first,
                target: itemRecords.compactMap(\.target).first,
                options: itemRecords.compactMap(\.options).first,
                cells: [:])
            for condition in conditions {
                let cellRecords = itemRecords.filter { $0.condition == condition }
                guard !cellRecords.isEmpty else { continue }
                row.cells[condition] = choiceCell(
                    records: cellRecords, target: row.target)
            }
            return row
        }
        .sorted { ($0.promptIndex, $0.promptID) < ($1.promptIndex, $1.promptID) }
        return ChoiceMatrix(conditions: conditions, items: items)
    }

    static func choiceCell(records: [Record], target: String?) -> ChoiceCell {
        var cell = ChoiceCell()
        let sampled = records.filter { !$0.isChoiceRecord }
        cell.sampleCount = sampled.count
        cell.parseFailures = sampled.count(where: { $0.parsedChoice.isFailure })
        if let instrument = records.first(where: { $0.isChoiceRecord }) {
            cell.instrumented = true
            cell.choice = instrument.selected
            cell.margin = instrument.margin
            let target = target ?? instrument.target
            if let target {
                cell.targetProbability = instrument.choiceProbability?[target]
                cell.targetLogOdds = instrument.logOdds?[target]
            }
        } else {
            // Majority vote over sampled parses (ties break alphabetically —
            // deterministic, and a tie is visible via the sample counts).
            let choices = sampled.compactMap(\.parsedChoice.value)
            let tally = Dictionary(grouping: choices, by: { $0 })
                .mapValues(\.count)
            cell.choice = tally.max {
                ($0.value, $1.key) < ($1.value, $0.key)
            }?.key
        }
        if let target, let choice = cell.choice {
            cell.matchesTarget = choice == target
        }
        return cell
    }

    // MARK: Arm aggregation

    public struct ArmSummary: Identifiable, Sendable, Equatable {
        public var arm: String
        public var condition: String
        public var items: Int
        /// Fraction of the arm's items whose chosen option matched target.
        public var targetMatchRate: Double?
        public var meanTargetProbability: Double?
        public var meanTargetLogOdds: Double?
        public var id: String { "\(arm)\u{1F}\(condition)" }
    }

    /// Aggregation by the experimental arm metadata (P3), one row per
    /// (arm, condition). Empty when no item carries an `arm`.
    public static func armAggregation(matrix: ChoiceMatrix) -> [ArmSummary] {
        let armed = matrix.items.filter { $0.arm != nil }
        guard !armed.isEmpty else { return [] }
        var summaries: [ArmSummary] = []
        let byArm = Dictionary(grouping: armed, by: { $0.arm ?? "" })
        for (arm, items) in byArm.sorted(by: { $0.key < $1.key }) {
            for condition in matrix.conditions {
                let cells = items.compactMap { $0.cells[condition] }
                guard !cells.isEmpty else { continue }
                let matches = cells.compactMap(\.matchesTarget)
                let probabilities = cells.compactMap(\.targetProbability)
                let logOdds = cells.compactMap(\.targetLogOdds)
                summaries.append(
                    ArmSummary(
                        arm: arm, condition: condition, items: cells.count,
                        targetMatchRate: matches.isEmpty
                            ? nil
                            : Double(matches.count(where: { $0 }))
                                / Double(matches.count),
                        meanTargetProbability: probabilities.isEmpty
                            ? nil
                            : probabilities.reduce(0, +) / Double(probabilities.count),
                        meanTargetLogOdds: logOdds.isEmpty
                            ? nil
                            : logOdds.reduce(0, +) / Double(logOdds.count)))
            }
        }
        return summaries
    }

    // MARK: Parse failures

    public struct ParseFailure: Identifiable, Sendable, Equatable {
        public var condition: String
        public var promptID: String
        public var endpoint: String
        public var outputExcerpt: String
        public var id: String { "\(condition)\u{1F}\(promptID)\u{1F}\(endpoint)" }
    }

    /// Every record whose endpoint parse key is JSON `null` — the parse
    /// FAILURES (first-class coherence endpoint, P3), with an output excerpt
    /// so the failure is inspectable without opening the raw JSONL.
    public static func parseFailures(
        records: [Record], excerptLength: Int = 160
    ) -> [ParseFailure] {
        var failures: [ParseFailure] = []
        for record in records {
            for (field, endpoint) in [
                (record.parsedChoice.isFailure, "choice"),
                (record.parsedMonths.isFailure, "months"),
            ] where field {
                let output = (record.output ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                failures.append(
                    ParseFailure(
                        condition: record.condition,
                        promptID: record.promptID,
                        endpoint: endpoint,
                        outputExcerpt: output.count > excerptLength
                            ? String(output.prefix(excerptLength)) + "…"
                            : output))
            }
        }
        return failures.sorted {
            ($0.condition, $0.promptID, $0.endpoint)
                < ($1.condition, $1.promptID, $1.endpoint)
        }
    }

    // MARK: Cross-condition exact agreement

    public struct Agreement: Identifiable, Sendable, Equatable {
        public var condition: String
        public var matches: Int
        public var comparable: Int
        /// True when the value came from the server's stamped
        /// `agreementWithBaseline` key rather than a local recount.
        public var fromServerStamp: Bool
        public var id: String { condition }

        public var fraction: Double? {
            comparable > 0 ? Double(matches) / Double(comparable) : nil
        }
    }

    /// Exact choice agreement of every non-baseline condition with the
    /// baseline, per item (acceptance test 5's UI half: "no-intervention
    /// variant matches baseline 12/12"). Computed locally from the matrix;
    /// the server's stamped `conditions[].agreementWithBaseline` is
    /// preferred when present (tolerating its absence).
    public static func baselineAgreement(
        matrix: ChoiceMatrix, report: Report? = nil
    ) -> [Agreement] {
        guard matrix.conditions.contains(baselineConditionName) else { return [] }
        var agreements: [Agreement] = []
        for condition in matrix.conditions
        where condition != baselineConditionName {
            var matches = 0
            var comparable = 0
            for item in matrix.items {
                guard
                    let base = item.cells[baselineConditionName]?.choice,
                    let choice = item.cells[condition]?.choice
                else { continue }
                comparable += 1
                if base == choice { matches += 1 }
            }
            if let stamped = report?.conditions[condition],
                let fraction = stamped.agreementWithBaseline
            {
                let total = stamped.agreementTotal ?? comparable
                let agreed =
                    stamped.agreementMatches
                    ?? Int((fraction * Double(total)).rounded())
                agreements.append(
                    Agreement(
                        condition: condition, matches: agreed,
                        comparable: total, fromServerStamp: true))
            } else if comparable > 0 {
                agreements.append(
                    Agreement(
                        condition: condition, matches: matches,
                        comparable: comparable, fromServerStamp: false))
            }
        }
        return agreements
    }

    // MARK: Completion status

    public struct ConditionCompletion: Identifiable, Sendable, Equatable {
        public var condition: String
        public var completed: Int
        public var planned: Int
        public var error: String?
        public var id: String { condition }
    }

    public struct Completion: Sendable, Equatable {
        public var completed: Int
        /// Derivable planned total: the union of observed generation slots
        /// applied uniformly across conditions (plus report error
        /// conditions). nil when nothing was decodable.
        public var planned: Int?
        public var perCondition: [ConditionCompletion]

        public var summaryLine: String {
            guard let planned else { return "\(completed) measurement records recorded" }
            return "\(completed) of \(planned) measurement records completed"
        }
    }

    /// Completion where derivable: the study matrix is uniform by
    /// construction (every condition visits every item slot), so the union
    /// of observed (kind, promptID, sampleIndex) slots across conditions is
    /// the per-condition plan. Report `error` conditions count as 0 of plan.
    public static func completion(
        records: [Record], report: Report? = nil
    ) -> Completion {
        struct Slot: Hashable {
            let kind: Bool  // isChoiceRecord
            let promptID: String
            let sampleIndex: Int
        }
        var slots: Set<Slot> = []
        var byCondition: [String: Set<Slot>] = [:]
        var errors: [String: String] = [:]
        for record in records {
            if let error = record.error {
                errors[record.condition] = error
                continue
            }
            let slot = Slot(
                kind: record.isChoiceRecord,
                promptID: record.promptID,
                sampleIndex: record.sampleIndex ?? 0)
            slots.insert(slot)
            byCondition[record.condition, default: []].insert(slot)
        }
        for (condition, summary) in report?.conditions ?? [:] {
            if let error = summary.error { errors[condition] = error }
        }
        let planned = slots.count
        let conditions = orderedConditions(
            Set(byCondition.keys).union(errors.keys))
        let perCondition = conditions.map { condition in
            ConditionCompletion(
                condition: condition,
                completed: byCondition[condition]?.count ?? 0,
                planned: planned,
                error: errors[condition])
        }
        let completed = perCondition.map(\.completed).reduce(0, +)
        return Completion(
            completed: completed,
            planned: planned > 0 ? planned * perCondition.count : nil,
            perCondition: perCondition)
    }

    // MARK: Categorical metric display (acceptance 7)

    /// Conditions whose records carry ANY option-bearing item: text-diversity
    /// metrics are not meaningful there (wordCount=1 expected, distinct2=0
    /// mathematically inevitable for one-token answers).
    public static func categoricalConditions(records: [Record]) -> Set<String> {
        var result: Set<String> = []
        for record in records where record.isCategorical {
            result.insert(record.condition)
        }
        return result
    }

    /// Categorical evidence from BOTH layers: option-bearing records, plus
    /// report conditions stamped with choice metrics (choiceReadouts /
    /// choiceRate / agreementWithBaseline). A run whose generations.jsonl is
    /// missing or truncated away can still prove it was categorical through
    /// its report — the categorical presentation (columns + text-metric N/A
    /// rule) must not hide stamped choice metrics behind zero decoded
    /// records.
    public static func categoricalConditions(
        records: [Record], report: Report?
    ) -> Set<String> {
        var result = categoricalConditions(records: records)
        for (name, summary) in report?.conditions ?? [:]
        where summary.choiceReadouts != nil || summary.choiceRate != nil
            || summary.agreementWithBaseline != nil
            || summary.agreementTotal != nil
        {
            result.insert(name)
        }
        return result
    }

    public static let categoricalMetricPlaceholder = "N/A (categorical output)"

    /// The single display rule for wordCount/distinct2 wherever they render:
    /// categorical conditions show the N/A placeholder, non-categorical show
    /// the formatted value, absent values show an em dash.
    public static func textMetricDisplay(
        _ value: Double?, categorical: Bool, fractionDigits: Int = 2
    ) -> String {
        if categorical { return categoricalMetricPlaceholder }
        guard let value else { return "—" }
        return String(format: "%.\(fractionDigits)f", value)
    }
}
