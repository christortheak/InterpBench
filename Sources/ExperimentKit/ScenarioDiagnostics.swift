import CryptoKit
import Foundation

/// Per-scenario validation diagnostics (D1).
///
/// `scenarioAccuracy` computes every projection and the class means that
/// define the midpoint, then throws all of it away and returns one number.
/// When nine virtues all score near chance, that number cannot distinguish the
/// two explanations a researcher actually needs to tell apart:
///
/// - the direction does not read the concept at all, or
/// - it ranks scenarios correctly but the MIDPOINT sits in the wrong place.
///
/// The second is a threshold problem, not a vector problem, and it is
/// invisible to accuracy. So this keeps the working: per-row projections and
/// margins, the class means and threshold that produced them, and a
/// threshold-free ranking statistic (`auc`) that separates the two cases.
///
/// The 27B validates of 2026-08-01 turned the hypothetical into an incident:
/// every `designatedReference` concept thresholded its held-out scenarios at
/// the midpoint of two STORY-corpus projections, and the scenario
/// distribution sat entirely on one side of it (`fair`: tp=20 fp=20 tn=0
/// fn=0, accuracy 0.50, AUC 0.855). Two additions carry that lesson:
/// `heldOutCalibration` (the same confusion arithmetic at the held-out items'
/// OWN class-mean midpoint — one disclosed scalar spent on the labels, and
/// deliberately NOT sign-oriented, so an inverted direction reads below 0.5
/// exactly as it does in AUC) and `oneSidedPredictions` (the transfer
/// threshold put every item on one side — `accuracy` is measuring the
/// threshold, not the vector). `accuracy` keeps its historical meaning: does
/// the EXTRACTION-derived threshold transfer to held-out text.
///
/// **AUC and the calibration are DIAGNOSTICS.** No freeze gate reads them,
/// and none should start to without an explicit policy decision — they are
/// reported so a near-chance accuracy can be interpreted, not so they can be
/// optimised against.
///
/// Statistics here are deliberately DESCRIPTIVE (D2): confusion matrix,
/// balanced accuracy, class counts, Wilson intervals. No p-values. The
/// binomial null assumes independence, and these scenario sets are generated
/// per concept by one agent over shared topics with deliberately matched
/// pairs — matched pairs and topic clustering both induce correlation, so a
/// binomial p-value would understate variance and read optimistically. The
/// inferential design is a decision to be declared, not a default to ship.
///
/// Cross-engine twin: `Server/steerlab_server/experiment/scenario_diagnostics.py`.
public enum ScenarioDiagnostics {

    /// Identity of a scenario ROW, independent of its position in the file —
    /// so a diagnostic record still names the right item after a re-order.
    public static func rowHash(text: String, label: Bool) -> String {
        // Canonical row, not text alone. Two scenarios with identical text
        // and opposite `expresses` labels are different rows; hashing the
        // text alone would give them one identity and make a diagnostic
        // record ambiguous about which one it describes.
        let canonical = #"{"expresses":\#(label),"text":"# + quoted(text) + "}"
        return SHA256.hash(data: Data(canonical.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// Python's `json.dumps` encoding, reproduced exactly.
    ///
    /// The previous version escaped only backslash, quote and newline, while
    /// `json.dumps` defaults to `ensure_ascii=True` and escapes EVERY
    /// non-ASCII character plus every control character. So `café` hashed as
    /// `caf\u00e9` on the server and `café` on the Mac, and a tab or a smart
    /// quote diverged the same way — invisible in an ASCII-only fixture, and
    /// near-certain in LLM-generated prose.
    ///
    /// Rules, matching CPython's `json.encoder`: `\"` `\\` `\b` `\f` `\n`
    /// `\r` `\t` as short escapes; every other character below 0x20 and
    /// every character above 0x7E as `\uXXXX`, with astral characters as
    /// surrogate pairs (which UTF-16 code units give directly).
    static func quoted(_ value: String) -> String {
        var out = "\""
        for unit in value.utf16 {
            switch unit {
            case 0x22: out += "\\\""
            case 0x5C: out += "\\\\"
            case 0x08: out += "\\b"
            case 0x0C: out += "\\f"
            case 0x0A: out += "\\n"
            case 0x0D: out += "\\r"
            case 0x09: out += "\\t"
            case 0x20 ... 0x7E:
                out.append(Character(UnicodeScalar(unit)!))
            default:
                out += String(format: "\\u%04x", unit)
            }
        }
        return out + "\""
    }

    /// NAIVE item-level Wilson score interval (95% by default).
    ///
    /// "Naive" is not modesty — it is the assumption. Wilson assumes
    /// independent Bernoulli trials, and these scenario sets are generated
    /// per concept by one agent over shared topics with deliberately matched
    /// pairs. Matched pairs and topic clustering both induce correlation, so
    /// the true interval is WIDER than this one. Calling the output
    /// "descriptive" does not repair that; an interval is an inferential
    /// object whatever it is labelled, so the label states the assumption
    /// instead.
    ///
    /// Kept rather than dropped because small-N imprecision is real and worth
    /// seeing: 20/20 and 200/200 are different evidence. A cluster-aware
    /// bootstrap would be the honest replacement, and it needs the blocking
    /// structure that is still undeclared (D2).
    ///
    /// Wilson rather than normal-approximation because these sets are small
    /// (40 items) and often near 0 or 1, where the normal interval runs past
    /// the ends of the scale and reports impossible bounds.
    public static func wilsonInterval(
        successes: Int, total: Int, z: Double = 1.959963984540054
    ) -> (low: Double, high: Double)? {
        guard total > 0 else { return nil }
        let p = Double(successes) / Double(total)
        let n = Double(total)
        let denominator = 1 + z * z / n
        let centre = (p + z * z / (2 * n)) / denominator
        let spread = (z * (p * (1 - p) / n + z * z / (4 * n * n)).squareRoot())
            / denominator
        return (max(0, centre - spread), min(1, centre + spread))
    }

    /// Threshold-free separation: the probability that a randomly chosen
    /// positive scenario projects above a randomly chosen negative one.
    ///
    /// Mann–Whitney U over midranks, so TIES CONTRIBUTE 0.5 — an all-ties
    /// direction scores exactly 0.5 rather than 0 or 1 depending on
    /// comparison order. Nil when either class is empty: with one class there
    /// is nothing to separate, and any number would be an artefact.
    public static func auc(projections: [Double], labels: [Bool]) -> Double? {
        var positives: [Double] = []
        var negatives: [Double] = []
        for (projection, label) in zip(projections, labels) {
            if label { positives.append(projection) } else { negatives.append(projection) }
        }
        guard !positives.isEmpty, !negatives.isEmpty else { return nil }
        var wins = 0.0
        for p in positives {
            for n in negatives {
                if p > n {
                    wins += 1
                } else if p == n {
                    wins += 0.5
                }
            }
        }
        return wins / Double(positives.count * negatives.count)
    }

    /// Confusion arithmetic at the held-out items' OWN class-mean midpoint.
    ///
    /// The transfer threshold comes from the extraction classes, and when
    /// those are a different text family (designatedReference's story corpora
    /// vs scenario validation items) the whole held-out distribution can land
    /// on one side of it. This measures separation at the boundary the
    /// held-out classes themselves define.
    public struct Calibration: Codable, Sendable, Equatable {
        public var threshold: Double
        public var classMeans: [String: Double]
        public var accuracy: Double
        public var balancedAccuracy: Double
        public var confusion: [String: Int]
    }

    /// Nil when either class is empty — with one class there is no boundary
    /// to define. Plain `reduce(+)/count` means to match the Python twin's
    /// `sum/len`; the committed fixture pins the bytes.
    public static func heldOutCalibration(
        projections: [Double], labels: [Bool]
    ) -> Calibration? {
        var positives: [Double] = []
        var negatives: [Double] = []
        for (projection, label) in zip(projections, labels) {
            if label { positives.append(projection) } else { negatives.append(projection) }
        }
        guard !positives.isEmpty, !negatives.isEmpty else { return nil }
        let meanPositive = positives.reduce(0, +) / Double(positives.count)
        let meanNegative = negatives.reduce(0, +) / Double(negatives.count)
        let threshold = (meanPositive + meanNegative) / 2
        let tp = positives.filter { $0 > threshold }.count
        let fn = positives.count - tp
        let fp = negatives.filter { $0 > threshold }.count
        let tn = negatives.count - fp
        let sensitivity = Double(tp) / Double(positives.count)
        let specificity = Double(tn) / Double(negatives.count)
        return Calibration(
            threshold: threshold,
            classMeans: ["positive": meanPositive, "negative": meanNegative],
            accuracy: Double(tp + tn) / Double(projections.count),
            balancedAccuracy: (sensitivity + specificity) / 2,
            confusion: ["tp": tp, "fp": fp, "tn": tn, "fn": fn])
    }

    public struct Row: Codable, Sendable, Equatable {
        public var index: Int
        public var id: String
        public var rowHash: String
        public var label: Bool
        public var projection: Double
        public var predicted: Bool
        /// Signed distance from the decision boundary. Near zero means the
        /// item barely decided either way — the rows to read first when an
        /// accuracy is disappointing.
        public var margin: Double
        public var correct: Bool
    }

    public struct Report: Codable, Sendable, Equatable {
        public var layer: Int
        public var threshold: Double
        public var classMeans: [String: Double]
        public var directionNorm: Double
        public var scenarioCount: Int
        public var classCounts: [String: Int]
        public var accuracy: Double?
        /// NAIVE item-level interval — assumes independent items, which
        /// these sets violate. See `wilsonInterval`.
        public var naiveItemLevelInterval95: [Double]?
        public var balancedAccuracy: Double?
        public var sensitivity: Double?
        public var specificity: Double?
        public var confusion: [String: Int]
        /// DIAGNOSTIC ONLY — see the type's documentation.
        public var auc: Double?
        /// The transfer threshold put every item on one side: `accuracy` is
        /// measuring the threshold, not the vector. Optional so reports
        /// written before 2026-08-01 still decode.
        public var oneSidedPredictions: Bool?
        /// DIAGNOSTIC ONLY — separation at the held-out classes' own
        /// midpoint. Optional for the same pre-2026-08-01 decode reason.
        public var heldOutCalibration: Calibration?
        public var rows: [Row]
    }

    /// The rich record: everything the accuracy number was computed FROM.
    ///
    /// The caller supplies projections because how activations are obtained
    /// differs between the paired and grand-mean paths, while the arithmetic
    /// here does not.
    public struct UnequalInputs: Error, CustomStringConvertible {
        public let detail: String
        public var description: String { detail }
    }

    public static func report(
        scenarioIDs: [String?],
        scenarioTexts: [String],
        projections: [Double],
        labels: [Bool],
        threshold: Double,
        classMeans: [String: Double],
        layer: Int,
        directionNorm: Double
    ) throws -> Report {
        // Refuse rather than pad. This used to invent `false` for a missing
        // label and "" for a missing text, while the Python twin truncated
        // via `zip` — so the same malformed input produced a different
        // (and in Swift's case, fabricated) answer on each engine.
        let counts = [
            "scenarioIDs": scenarioIDs.count, "scenarioTexts": scenarioTexts.count,
            "projections": projections.count, "labels": labels.count,
        ]
        if Set(counts.values).count > 1 {
            throw UnequalInputs(
                detail: "scenario diagnostics received unequal inputs "
                    + counts.sorted { $0.key < $1.key }
                        .map { "\($0.key) \($0.value)" }
                        .joined(separator: ", ")
                    + " — a padded or truncated row set would silently change "
                    + "which items were scored")
        }
        var rows: [Row] = []
        var correct = 0
        var tp = 0, fp = 0, tn = 0, fn = 0
        for index in projections.indices {
            let projection = projections[index]
            let label = labels[index]
            let predicted = projection > threshold
            let isCorrect = predicted == label
            if isCorrect { correct += 1 }
            switch (label, predicted) {
            case (true, true): tp += 1
            case (true, false): fn += 1
            case (false, true): fp += 1
            case (false, false): tn += 1
            }
            let text = scenarioTexts[index]
            rows.append(
                Row(
                    // Position AND identity: a line number alone stops
                    // meaning anything the moment the file is re-ordered.
                    index: index,
                    id: scenarioIDs[index] ?? "scenario-\(index + 1)",
                    // The canonical ROW, not the text alone: two scenarios
                    // with identical text and opposite labels are different
                    // rows and must not share an identity.
                    rowHash: rowHash(text: text, label: label),
                    label: label,
                    projection: projection,
                    predicted: predicted,
                    margin: projection - threshold,
                    correct: isCorrect))
        }

        let total = rows.count
        let positives = tp + fn
        let negatives = tn + fp
        let sensitivity = positives > 0 ? Double(tp) / Double(positives) : nil
        let specificity = negatives > 0 ? Double(tn) / Double(negatives) : nil
        let balanced: Double? =
            if let sensitivity, let specificity { (sensitivity + specificity) / 2 }
            else { nil }
        let interval = wilsonInterval(successes: correct, total: total)

        return Report(
            layer: layer,
            threshold: threshold,
            classMeans: classMeans,
            directionNorm: directionNorm,
            scenarioCount: total,
            classCounts: ["positive": positives, "negative": negatives],
            accuracy: total > 0 ? Double(correct) / Double(total) : nil,
            naiveItemLevelInterval95: interval.map { [$0.low, $0.high] },
            balancedAccuracy: balanced,
            sensitivity: sensitivity,
            specificity: specificity,
            confusion: ["tp": tp, "fp": fp, "tn": tn, "fn": fn],
            auc: auc(projections: projections, labels: labels),
            oneSidedPredictions: total > 0 && (tp + fp == total || tn + fn == total),
            heldOutCalibration: heldOutCalibration(
                projections: projections, labels: labels),
            rows: rows)
    }
}
