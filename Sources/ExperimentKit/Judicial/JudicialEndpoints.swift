import Foundation

/// Built-in outcome-endpoint parsers + derived endpoints (Swift twin of the
/// server's `steerlab_server/experiment/judicial.py` — the two share fixture
/// values, so changes here must land in both). Everything is pure
/// text/arithmetic — no model, no GPU.
///
/// These are the engine's two SHIPPED parsers, both concept-agnostic text
/// readers over whatever prose a study's items elicit:
///
/// - **categorical choice**: the option a response settled on, read from a
///   JSON object with a declared key, an exact whole-response match, or the
///   earliest word-boundary mention. Derived endpoints over it are outcome
///   rates and between-item contrasts.
/// - **duration in months**: a numeric quantity stated in years and/or
///   months; endpoints are the mean shift, spread, anchor slope, and
///   proportionality; the parse-failure rate is itself a coherence signal.
///
/// Neither parser knows any domain. A study picks one by declaring a parser in
/// its manifest (see `ParserRegistry` for the declared-grammar path, which
/// generalizes both). The case families of the judicial-decision study that
/// motivated them are one worked example: a choice-of-law answer and a
/// rule-adherence choice use the categorical parser, and a sentence length in
/// months uses the duration parser.
public enum Judicial {

    // MARK: - Categorical option parsing

    /// Extracts a categorical choice from sampled prose.
    ///
    /// Tries, in order: a JSON object carrying `jsonKey` (the usual
    /// answer-in-JSON schema), an exact normalized match of the whole
    /// response, then the option whose earliest word-boundary mention appears
    /// first in the text. Returns the matched option verbatim, or nil (a parse
    /// failure — count it).
    public static func parseChoice(
        _ text: String, options: [String], jsonKey: String = "answer"
    ) -> String? {
        guard !text.isEmpty, !options.isEmpty else { return nil }
        let normalized = Dictionary(
            options.map { (normalize($0), $0) },
            uniquingKeysWith: { _, later in later })

        for payload in jsonObjects(in: text) {
            if let value = payload[jsonKey] as? String,
                let match = normalized[normalize(value)]
            {
                return match
            }
        }

        if let match = normalized[normalize(text)] { return match }

        let haystack = normalizeSpaces(text)
        let haystackRange = NSRange(haystack.startIndex..., in: haystack)
        var earliest: (location: Int, option: String)?
        for option in options {
            let pattern =
                "\\b" + NSRegularExpression.escapedPattern(for: normalize(option)) + "\\b"
            guard
                let regex = try? NSRegularExpression(
                    pattern: pattern, options: [.caseInsensitive]),
                let hit = regex.firstMatch(in: haystack, range: haystackRange)
            else { continue }
            if earliest == nil || hit.range.location < (earliest?.location ?? .max) {
                earliest = (hit.range.location, option)
            }
        }
        return earliest?.option
    }

    private static func normalize(_ text: String) -> String {
        var result = normalizeSpaces(text).trimmingCharacters(in: .whitespaces)
        while result.hasSuffix(".") { result.removeLast() }
        while result.hasPrefix(".") { result.removeFirst() }
        return result.lowercased()
    }

    private static func normalizeSpaces(_ text: String) -> String {
        text.replacing(/\s+/, with: " ")
    }

    /// JSON objects embedded in the text (fenced or bare), outermost first —
    /// a model told to answer in JSON often wraps it in prose or a fence.
    private static func jsonObjects(in text: String) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        let fence = /```(?:json)?\s*(\{.*?\})\s*```/.dotMatchesNewlines()
        for match in text.matches(of: fence) {
            if let payload = decodeObject(String(match.1)) {
                objects.append(payload)
            }
        }
        var searchIndex = text.startIndex
        while let start = text[searchIndex...].firstIndex(of: "{") {
            if let end = balancedObjectEnd(in: text, from: start),
                let payload = decodeObject(String(text[start ... end]))
            {
                objects.append(payload)
            }
            searchIndex = text.index(after: start)
        }
        return objects
    }

    /// Index of the `}` closing the object opened at `start`, tracking JSON
    /// string/escape state so braces inside strings do not count (approximates
    /// Python's `JSONDecoder.raw_decode`, which tolerates trailing prose).
    private static func balancedObjectEnd(
        in text: String, from start: String.Index
    ) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let char = text[index]
            if inString {
                if escaped {
                    escaped = false
                } else if char == "\\" {
                    escaped = true
                } else if char == "\"" {
                    inString = false
                }
            } else {
                switch char {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 { return index }
                default: break
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func decodeObject(_ candidate: String) -> [String: Any]? {
        guard
            let object = try? JSONSerialization.jsonObject(with: Data(candidate.utf8))
        else { return nil }
        return object as? [String: Any]
    }

    // MARK: - Duration-in-months parsing

    /// Number words for the spelled-out formal register — e.g. a
    /// judicial-decision study's "I sentence the defendant, A, to **ten years
    /// and six months'** imprisonment" (an anchoring-arm run, 2026-08-10:
    /// 58/1320 records unparsed, and dose-dependently, because steering pushes
    /// the formal register — so a digit-only grammar loses records
    /// NON-randomly).
    /// One through twelve, accepted wherever a digit number is in the
    /// COMPOUND and SINGLE terms only — ranges stay digit-only. Server twin:
    /// `judicial.NUMBER_WORDS`; the registry's durationMonths grammar shares
    /// this vocabulary (`ParserRegistry.numberWordFragment`).
    static let numberWords: [String: Double] = [
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12,
    ]

    /// Extracts a duration in months from prose (e.g. a sentence length in a
    /// judicial-decision study).
    ///
    /// Handles "18 months", "2 years", "1.5 years", "18 to 24 months"
    /// (midpoint), compound durations ("8 years 3 months" → 99 — a compound
    /// must never drop a term), the spelled-out formal register ("ten years
    /// and six months" → 126 — number words one through twelve), German units
    /// with identical semantics ("2 Jahre 6 Monate" → 30, "18 Monate" → 18),
    /// and comma decimals. Years normalize ×12. Returns nil on failure —
    /// parse-failure *rate* is a first-class coherence endpoint, so failures
    /// must be counted, never silently dropped or coerced to 0.
    ///
    /// Cross-engine contract: must agree fixture-for-fixture with the
    /// server's `parse_months` (`Server/steerlab_server/experiment/`).
    public static func parseMonths(_ text: String) -> Double? {
        guard !text.isEmpty else { return nil }
        let range =
            /(\d+(?:[.,]\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:[.,]\d+)?)\s*(years?|yrs?|months?|mos?|jahre?n?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: range) {
            guard let low = toDouble(match.1), let high = toDouble(match.2) else {
                return nil
            }
            let months = (low + high) / 2.0
            return isYears(match.3) ? months * 12.0 : months
        }
        // Compound "X years Y months" (optionally joined by ","/"and"/"und"):
        // both terms count — matching only the first silently corrupts the
        // duration DV (96 where the model said 99). Number terms are
        // digits-or-word; the \b pair keeps "brighten years" from reading as
        // "ten years".
        let compound =
            /(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(?:years?|yrs?|jahre?n?)\b,?\s*(?:(?:and|und)\s+)?(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(?:months?|mos?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: compound) {
            guard let years = toDouble(match.1), let months = toDouble(match.2) else {
                return nil
            }
            return years * 12.0 + months
        }
        let single =
            /(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(years?|yrs?|months?|mos?|jahre?n?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: single) {
            guard let value = toDouble(match.1) else { return nil }
            return isYears(match.2) ? value * 12.0 : value
        }
        return nil
    }

    private static func toDouble(_ token: Substring) -> Double? {
        if let word = numberWords[token.lowercased()] { return word }
        return Double(token.replacingOccurrences(of: ",", with: "."))
    }

    private static func isYears(_ unit: Substring) -> Bool {
        let lowered = unit.lowercased()
        return lowered.hasPrefix("year") || lowered.hasPrefix("yr")
            || lowered.hasPrefix("jahr")
    }

    /// Fraction of items whose parse came back nil.
    public static func parseFailureRate(_ parsed: [Double?]) -> Double {
        guard !parsed.isEmpty else { return 0 }
        return Double(parsed.count(where: { $0 == nil })) / Double(parsed.count)
    }

    // MARK: - Derived endpoints

    /// P(parsed choice == target) over successfully parsed items.
    public static func outcomeRate(_ choices: [String?], target: String) -> Double {
        let hits = choices.compactMap { $0 }
        guard !hits.isEmpty else { return .nan }
        return Double(hits.count(where: { $0 == target })) / Double(hits.count)
    }

    /// How much larger the concept's outcome-rate shift is on the
    /// high-discretion arm than on the low-discretion one.
    ///
    /// A generic two-arm contrast of two already-computed deltas. In the
    /// judicial-decision study it compares a doctrine framed as a standard
    /// (high discretion) against the same doctrine framed as a rule (low
    /// discretion).
    public static func sympathyGap(
        standardArmDelta: Double, ruleArmDelta: Double
    ) -> Double {
        standardArmDelta - ruleArmDelta
    }

    /// Per-item difference-in-differences:
    /// (treated − baseline | standard) − (treated − baseline | rule).
    /// Feed the result to `StudyStatistics`.
    public static func ruleVsStandardInteraction(
        ruleBaseline: [Double], ruleTreated: [Double],
        standardBaseline: [Double], standardTreated: [Double]
    ) -> [Double] {
        let n = min(
            min(ruleBaseline.count, ruleTreated.count),
            min(standardBaseline.count, standardTreated.count))
        return (0 ..< n).map { i in
            (standardTreated[i] - standardBaseline[i])
                - (ruleTreated[i] - ruleBaseline[i])
        }
    }

    /// OLS slope of the numeric outcome on the numeric value presented to the
    /// model in the item — the anchoring endpoint (Englich & Mussweiler's
    /// design; in the judicial-decision study, sentence months on anchor
    /// months). NaN when degenerate.
    public static func anchorSlope(anchors: [Double?], sentences: [Double?]) -> Double {
        let pairs = pairedValues(anchors, sentences)
        guard pairs.count >= 2 else { return .nan }
        let meanX = pairs.map(\.0).reduce(0, +) / Double(pairs.count)
        let meanY = pairs.map(\.1).reduce(0, +) / Double(pairs.count)
        let varX = pairs.map { ($0.0 - meanX) * ($0.0 - meanX) }.reduce(0, +)
        guard varX != 0 else { return .nan }
        let cov = pairs.map { ($0.0 - meanX) * ($0.1 - meanY) }.reduce(0, +)
        return cov / varX
    }

    /// Pearson correlation of the numeric outcome with each item's declared
    /// severity — does the model still scale its response to the item under
    /// the intervention? (In the judicial-decision study: sentence length
    /// against offense severity.)
    public static func proportionality(
        severities: [Double?], sentences: [Double?]
    ) -> Double {
        let pairs = pairedValues(severities, sentences)
        guard pairs.count >= 2 else { return .nan }
        let meanX = pairs.map(\.0).reduce(0, +) / Double(pairs.count)
        let meanY = pairs.map(\.1).reduce(0, +) / Double(pairs.count)
        let varX = pairs.map { ($0.0 - meanX) * ($0.0 - meanX) }.reduce(0, +)
        let varY = pairs.map { ($0.1 - meanY) * ($0.1 - meanY) }.reduce(0, +)
        guard varX != 0, varY != 0 else { return .nan }
        let cov = pairs.map { ($0.0 - meanX) * ($0.1 - meanY) }.reduce(0, +)
        return cov / (varX * varY).squareRoot()
    }

    private static func pairedValues(
        _ xs: [Double?], _ ys: [Double?]
    ) -> [(Double, Double)] {
        zip(xs, ys).compactMap { x, y in
            guard let x, let y else { return nil }
            return (x, y)
        }
    }

    /// Distributional readout over sampled numeric outcomes for one item —
    /// the mean AND spread endpoints, computed per (condition, prompt).
    public struct DistributionSummary: Codable, Sendable, Equatable {
        public let count: Int
        public let mean: Double
        /// Sample standard deviation (n − 1 denominator).
        public let stdev: Double
        public let min: Double
        public let q25: Double
        public let median: Double
        public let q75: Double
        public let max: Double

        public init(
            count: Int, mean: Double, stdev: Double, min: Double,
            q25: Double, median: Double, q75: Double, max: Double
        ) {
            self.count = count
            self.mean = mean
            self.stdev = stdev
            self.min = min
            self.q25 = q25
            self.median = median
            self.q75 = q75
            self.max = max
        }
    }

    /// Summary statistics over the successfully parsed samples of one item.
    public static func summarize(_ values: [Double?]) -> DistributionSummary? {
        let clean = values.compactMap { value -> Double? in
            guard let value, !value.isNaN else { return nil }
            return value
        }.sorted()
        guard let first = clean.first, let last = clean.last else { return nil }
        let n = clean.count
        let mean = clean.reduce(0, +) / Double(n)
        let stdev =
            n > 1
            ? (clean.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n - 1))
                .squareRoot()
            : 0.0
        return DistributionSummary(
            count: n, mean: mean, stdev: stdev, min: first,
            q25: quantile(clean, 0.25), median: quantile(clean, 0.5),
            q75: quantile(clean, 0.75), max: last)
    }

    /// Linear-interpolation quantile over a pre-sorted, non-empty list.
    private static func quantile(_ ordered: [Double], _ q: Double) -> Double {
        guard ordered.count > 1 else { return ordered[0] }
        let position = q * Double(ordered.count - 1)
        let low = Int(position.rounded(.down))
        let high = Swift.min(low + 1, ordered.count - 1)
        let weight = position - Double(low)
        return ordered[low] * (1 - weight) + ordered[high] * weight
    }
}
