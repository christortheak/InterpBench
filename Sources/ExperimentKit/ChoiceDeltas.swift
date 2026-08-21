import Foundation

/// Per-item paired deltas for the answer-token choice instrument — the
/// CITABLE, engine-computed version of the per-item Δ the results explorer
/// otherwise derives client-side. A number a paper quotes should come out of
/// the engine, under the epoch guard, not out of a viewer's arithmetic.
///
/// `analyze` writes `choice-deltas.csv`: one row per NON-baseline instrument
/// record, joined to the same item's baseline record.
///
///     condition, promptID, target,
///     baselineTargetLogOdds, conditionTargetLogOdds, deltaTargetLogOdds,
///     baselineSelected, conditionSelected, flipped,
///     baselineTargetProbability, conditionTargetProbability,
///     deltaTargetProbability
///
/// Server twin: `steerlab_server/experiment/choice_deltas.py`. The pinned
/// decisions, implemented identically on both engines:
///
/// - **The join key is `promptID` alone.** The answer-token instrument emits
///   exactly ONE deterministic readout per (condition, prompt) — it never
///   samples — so instrument records carry no `sampleIndex` to join on.
///   Should one ever appear it joins as part of the key, rather than
///   silently collapsing replicates.
/// - **A non-baseline record with no baseline partner is skipped AND
///   COUNTED** (`skippedNoBaseline`). Silent truncation reads as coverage: a
///   table of 8 rows where 12 items were measured must say so.
/// - **The row's quantity is the TARGET option's log-odds**, the same
///   endpoint the paired effect sizes use. A record whose `logOdds` has no
///   entry for its own target is unreadable, not zero: skipped and counted
///   (`skippedNoTargetValue`).
/// - **Probabilities are a convenience column**, not the row's reason to
///   exist: when `choiceProbability` is missing those three cells are empty
///   and the log-odds row still stands.
/// - **`flipped` needs both sides' `selected`.** With either missing the cell
///   is 0 and both selected columns are empty — a flip is a claim about two
///   observed choices, never an inference from one.
/// - **Rows sort by (condition, promptID)**, stably over run order, so the
///   same run in produces a byte-identical file out.
///
/// The per-condition summary reuses `StudyStatistics.pairedBootstrapCI` at
/// its default conventions (10 000 replicates, seed 0) — the same machinery
/// `effect-sizes.csv` runs on. No new statistics live here. (As with every
/// bootstrap in this project, the two engines' RNGs differ, so replicate
/// draws are not bit-identical across substrates; the estimator is.)
public enum ChoiceDeltas {

    public static let header = [
        "condition", "promptID", "target",
        "baselineTargetLogOdds", "conditionTargetLogOdds", "deltaTargetLogOdds",
        "baselineSelected", "conditionSelected", "flipped",
        "baselineTargetProbability", "conditionTargetProbability",
        "deltaTargetProbability",
    ]

    /// The instrument id whose records this reads. `choiceProbability` and
    /// `ordinalScale` ride the SAME record, so one id covers the family.
    public static let instrument = "answerTokenLogprob"

    /// Whether a record's `target` was DECLARED by its task item, and so
    /// whether this table may treat the target's log-odds as an endpoint
    /// (open-issues #6). Server twin: `choice_deltas._declared`.
    ///
    /// The run loop used to default an absent target to `options[0]`, which
    /// for an ordinalScale item is the rating ladder's minimum — so a likert
    /// run's `choice-deltas.csv` tabulated the paired shift in log-odds of
    /// answering "1" under a `target` column, a quantity nobody declared and
    /// which entangles pole movement with distribution sharpening.
    ///
    /// The authority ladder, in order:
    ///
    /// 1. **The pinned task file's per-item map**, when the caller could load
    ///    it. Exact, and the only rung that keeps a MIXED instrument honest —
    ///    an item carrying both a declared A/B target and an ordinal readout
    ///    on one record is a legitimate choice item.
    /// 2. **The record's own `targetSource` stamp** (records written after the
    ///    fix): `"declared"` and nothing else counts.
    /// 3. **The historical backstop**: an unstamped record whose only "target"
    ///    rides an `ordinalPosition` readout is the observed failure class.
    ///    Every legitimate choice study declares targets in its item file, so
    ///    an unstamped record with no ordinal readout keeps its endpoint.
    public static func targetIsDeclared(
        promptID: String, targetSource: String?, ordinalPosition: Double?,
        declaredTargets: [String: Bool]?
    ) -> Bool {
        if let declaredTargets, let declared = declaredTargets[promptID] {
            return declared
        }
        if let targetSource { return targetSource == "declared" }
        return ordinalPosition == nil
    }

    public static let baselineCondition = "baseline"

    /// One instrument readout, reduced to what the delta table needs.
    public struct Readout: Sendable, Equatable {
        public let condition: String
        public let promptID: String
        /// "" when the record carries no `sampleIndex` (today's instrument
        /// path never writes one). Absent normalizes to "" rather than 0, so
        /// a future sampled instrument cannot pair a sample-0 record with an
        /// unsampled one.
        public let sampleIndex: String
        public let target: String
        public let logOdds: [String: Double]
        public let choiceProbability: [String: Double]
        public let selected: String

        public init(
            condition: String, promptID: String, sampleIndex: String = "",
            target: String, logOdds: [String: Double],
            choiceProbability: [String: Double] = [:], selected: String = ""
        ) {
            self.condition = condition
            self.promptID = promptID
            self.sampleIndex = sampleIndex
            self.target = target
            self.logOdds = logOdds
            self.choiceProbability = choiceProbability
            self.selected = selected
        }
    }

    /// Per-condition rollup. Optional fields are omitted from JSON when the
    /// condition paired nothing — an absent mean is not a zero mean.
    public struct ConditionSummary: Codable, Sendable, Equatable {
        public let n: Int
        public let flipped: Int
        public let skippedNoBaseline: Int
        public let skippedNoTargetValue: Int
        public let deltaTargetLogOddsMean: Double?
        public let ciLower: Double?
        public let ciUpper: Double?
        public let replicates: Int?
        public let seed: UInt64?
    }

    /// The `choice-deltas.json` payload (cross-engine key contract).
    public struct Summary: Codable, Sendable, Equatable {
        public let conditions: [String: ConditionSummary]
        public let records: Int
        public let skippedNoBaseline: Int
        public let skippedNoTargetValue: Int
    }

    /// `(rows, summary)` over a run's instrument readouts. An empty
    /// `summary.conditions` means the run had nothing for this artifact to be
    /// about — the caller writes nothing rather than an empty table.
    public static func table(_ readouts: [Readout]) -> (rows: [[String]], summary: Summary) {
        var baselines: [String: Readout] = [:]
        for readout in readouts where readout.condition == baselineCondition {
            let key = itemKey(readout)
            if baselines[key] == nil { baselines[key] = readout }
        }

        var rows: [[String]] = []
        var diffs: [String: [Double]] = [:]
        var flips: [String: Int] = [:]
        var skippedNoBaseline: [String: Int] = [:]
        var skippedNoTarget: [String: Int] = [:]
        var conditions = Set<String>()
        for readout in readouts where readout.condition != baselineCondition {
            conditions.insert(readout.condition)
            guard let base = baselines[itemKey(readout)] else {
                skippedNoBaseline[readout.condition, default: 0] += 1
                continue
            }
            guard
                let conditionOdds = finite(readout.logOdds[readout.target]),
                let baselineOdds = finite(base.logOdds[readout.target])
            else {
                skippedNoTarget[readout.condition, default: 0] += 1
                continue
            }
            let delta = conditionOdds - baselineOdds
            diffs[readout.condition, default: []].append(delta)

            let conditionP = finite(readout.choiceProbability[readout.target])
            let baselineP = finite(base.choiceProbability[readout.target])
            let deltaP: Double? =
                if let conditionP, let baselineP { conditionP - baselineP } else { nil }

            let bothSelected = !base.selected.isEmpty && !readout.selected.isEmpty
            let flipped = bothSelected && base.selected != readout.selected
            if flipped { flips[readout.condition, default: 0] += 1 }

            rows.append([
                readout.condition,
                readout.promptID,
                readout.target,
                format(baselineOdds),
                format(conditionOdds),
                format(delta),
                bothSelected ? base.selected : "",
                bothSelected ? readout.selected : "",
                flipped ? "1" : "0",
                format(baselineP),
                format(conditionP),
                format(deltaP),
            ])
        }

        // Sorted so the same run in gives a byte-identical file out. The sort
        // is made stable by carrying the original index — Swift's `sort` is
        // not guaranteed stable, and two rows sharing a key must not swap
        // between runs.
        rows = rows.enumerated()
            .sorted {
                ($0.element[0], $0.element[1], $0.offset)
                    < ($1.element[0], $1.element[1], $1.offset)
            }
            .map(\.element)

        var summaries: [String: ConditionSummary] = [:]
        for condition in conditions {
            let values = diffs[condition] ?? []
            let ci = values.isEmpty ? nil : StudyStatistics.pairedBootstrapCI(values)
            summaries[condition] = ConditionSummary(
                n: values.count,
                flipped: flips[condition] ?? 0,
                skippedNoBaseline: skippedNoBaseline[condition] ?? 0,
                skippedNoTargetValue: skippedNoTarget[condition] ?? 0,
                deltaTargetLogOddsMean: ci?.mean,
                ciLower: ci?.ciLower,
                ciUpper: ci?.ciUpper,
                replicates: ci?.replicates,
                seed: ci?.seed)
        }
        let summary = Summary(
            conditions: summaries,
            records: rows.count,
            skippedNoBaseline: skippedNoBaseline.values.reduce(0, +),
            skippedNoTargetValue: skippedNoTarget.values.reduce(0, +))
        return (rows, summary)
    }

    /// The CSV text for a set of rows (header always present).
    public static func csv(_ rows: [[String]]) -> String {
        var lines = [header.map(escape).joined(separator: ",")]
        for row in rows {
            lines.append(row.map(escape).joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - Internals

    private static func itemKey(_ readout: Readout) -> String {
        "\(readout.promptID)\u{1F}\(readout.sampleIndex)"
    }

    private static func finite(_ value: Double?) -> Double? {
        guard let value, !value.isNaN else { return nil }
        return value
    }

    /// 6 significant digits, empty for absent/NaN — matching what
    /// `effect-sizes.csv` already publishes on the server. One dialect for
    /// engine-computed numbers.
    private static func format(_ value: Double?) -> String {
        guard let value, !value.isNaN else { return "" }
        return String(format: "%.6g", value)
    }

    private static func escape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
