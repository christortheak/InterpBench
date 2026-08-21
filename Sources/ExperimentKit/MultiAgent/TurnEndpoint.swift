import Foundation

/// A quantity a panel turn is supposed to produce (Wave-2 contract,
/// 2026-08-05), declared as DATA on the scenario:
///
/// ```json
/// "endpoint": {"name": "vote", "kind": "choice", "marker": "Vote:",
///              "vocabulary": ["affirm", "reverse", "vacate", "remand"]}
/// "endpoint": {"name": "months", "kind": "number", "marker": "Sentence:",
///              "min": 0, "max": 600}
/// ```
///
/// The runner parses each turn's generated text at write time and stamps the
/// parse onto the turn record, so a panel's votes are stored data rather than
/// something a viewer mines out of free text later.
///
/// Server twin: `Server/steerlab_server/experiment/turn_endpoint.py`. The
/// committed golden fixture `prompts/fixtures/panel-endpoints/` is exercised
/// by BOTH engines' tests, so a divergence fails on both sides.
public struct TurnEndpoint: Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable, CaseIterable {
        case choice
        case number
    }

    public var name: String
    public var kind: Kind
    public var marker: String
    /// Choice only. Non-empty, with non-empty members (enforced on decode).
    public var vocabulary: [String]?
    /// Number only, inclusive.
    public var min: Double?
    public var max: Double?

    public init(
        name: String, kind: Kind, marker: String,
        vocabulary: [String]? = nil, min: Double? = nil, max: Double? = nil
    ) {
        self.name = name
        self.kind = kind
        self.marker = marker
        self.vocabulary = vocabulary
        self.min = min
        self.max = max
    }

    /// Validating decode. A malformed declaration is a LOUD refusal, not a
    /// tolerated unknown: the scenario is pinned, reviewed data, and a
    /// declaration that silently parsed nothing — a typo'd kind, an empty
    /// marker, a choice with no vocabulary — would be indistinguishable from
    /// a panel that never answered the question.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let name = (try container.decodeIfPresent(String.self, forKey: .name) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "endpoint needs a name")
        }
        let rawKind = (try container.decodeIfPresent(String.self, forKey: .kind) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let kind = Kind(rawValue: rawKind) else {
            throw ExperimentError(
                reason: "endpoint '\(name)' has unknown kind '\(rawKind)' — "
                    + "expected one of \(Kind.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        let marker = try container.decodeIfPresent(String.self, forKey: .marker) ?? ""
        guard !marker.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ExperimentError(
                reason: "endpoint '\(name)' needs a non-empty marker")
        }
        let vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary)
        let minimum = try container.decodeIfPresent(Double.self, forKey: .min)
        let maximum = try container.decodeIfPresent(Double.self, forKey: .max)
        switch kind {
        case .choice:
            guard let vocabulary, !vocabulary.isEmpty else {
                throw ExperimentError(
                    reason: "choice endpoint '\(name)' needs a non-empty vocabulary")
            }
            guard
                vocabulary.allSatisfy({
                    !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                })
            else {
                throw ExperimentError(
                    reason: "choice endpoint '\(name)' has an empty vocabulary member")
            }
            guard minimum == nil, maximum == nil else {
                throw ExperimentError(
                    reason: "choice endpoint '\(name)' declares min/max, which "
                        + "only a number endpoint reads")
            }
        case .number:
            guard vocabulary == nil else {
                throw ExperimentError(
                    reason: "number endpoint '\(name)' declares a vocabulary, "
                        + "which only a choice endpoint reads")
            }
            if let minimum, let maximum, minimum > maximum {
                throw ExperimentError(
                    reason: "number endpoint '\(name)' has min > max")
            }
        }
        self.name = name
        self.kind = kind
        self.marker = marker
        self.vocabulary = vocabulary
        self.min = minimum
        self.max = maximum
    }
}

/// A parsed endpoint value: a vocabulary member, or a number.
public enum TurnEndpointValue: Codable, Sendable, Equatable {
    case choice(String)
    case number(Double)

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .choice(text)
        } else {
            self = .number(try container.decode(Double.self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .choice(let text): try container.encode(text)
        case .number(let value): try container.encode(value)
        }
    }
}

/// What the runner writes onto a turn record: a value, or an explicit
/// unparsed marker. Never a guess.
///
/// `value` is encoded as JSON `null` when unparsed rather than omitted — the
/// key's presence is what says "this turn declared an endpoint and it could
/// not be read", which is a first-class finding (a seat that stopped
/// answering the question), not an absence.
public struct TurnEndpointStamp: Codable, Sendable, Equatable {
    public let name: String
    public let value: TurnEndpointValue?
    public let unparsed: Bool?

    public init(name: String, value: TurnEndpointValue?) {
        self.name = name
        self.value = value
        self.unparsed = value == nil ? true : nil
    }

    private enum CodingKeys: String, CodingKey {
        case name, value, unparsed
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.value = try container.decodeIfPresent(
            TurnEndpointValue.self, forKey: .value)
        self.unparsed = try container.decodeIfPresent(Bool.self, forKey: .unparsed)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        if let value {
            try container.encode(value, forKey: .value)
        } else {
            try container.encodeNil(forKey: .value)
            try container.encode(true, forKey: .unparsed)
        }
    }
}

/// The parser. One pure function, no regex anywhere — parsing is a literal,
/// case-insensitive scan: find the marker, then read the following 80
/// characters. Two engines byte-agree on literal scans trivially; regex
/// dialects diverge (character classes, boundary semantics, greediness), and
/// this has a Python twin that must agree character for character.
///
/// Pinned edge decisions (the twin implements each one identically):
///
/// * **Case-insensitivity is ASCII folding** (A–Z ↔ a–z); every other
///   character compares exactly. Unicode lowercasing is not length-preserving
///   (`İ` lowercases to two scalars), which would desynchronise the folded
///   index from the original text — and the two engines' Unicode tables need
///   not agree on the same release.
/// * **Positions are Unicode scalars**, matching Python's code-point string
///   indexing.
/// * **The window is the 80 scalars following the marker**, and a match must
///   fit ENTIRELY inside it. A value that starts inside the window and runs
///   past it is unparsed, never a truncated number.
/// * **Whole-word means not adjacent to a letter** (Unicode general category
///   L\*), checked against the FULL text so a word clipped by the window edge
///   is correctly seen as clipped. This is what keeps `affirm` out of
///   `reaffirmed`.
/// * **Choice takes the earliest POSITION**, not the declaration order. Ties
///   (only reachable via a duplicated vocabulary) break on longest, then
///   declaration order.
/// * The stamped choice value is the DECLARED vocabulary member, not the
///   matched substring, so counts do not fragment across casings.
/// * **The marker's FIRST occurrence is the only one scanned.** If its window
///   holds no value the turn is unparsed even when a later marker would have
///   parsed — "the first thing it said" is the endpoint, not the most
///   convenient thing it said.
/// * Number: optional sign, digits, optional `.`-fraction; the FIRST such
///   number in the window, refused when outside a declared `min`/`max`
///   (inclusive). An out-of-range first number is unparsed — the scan does
///   not go looking for a second, more agreeable number.
public enum TurnEndpointParser {
    /// Scalars scanned after the marker.
    public static let window = 80

    public static func parse(_ endpoint: TurnEndpoint, in text: String) -> TurnEndpointValue? {
        let scalars = Array(text.unicodeScalars)
        guard let start = markerEnd(scalars, marker: endpoint.marker) else { return nil }
        switch endpoint.kind {
        case .choice:
            return parseChoice(endpoint, scalars, from: start).map(TurnEndpointValue.choice)
        case .number:
            return parseNumber(endpoint, scalars, from: start).map(TurnEndpointValue.number)
        }
    }

    /// The turn-record stamp. The record keeps the full text either way —
    /// this never replaces the evidence, only indexes it.
    public static func stamp(_ endpoint: TurnEndpoint, in text: String) -> TurnEndpointStamp {
        TurnEndpointStamp(name: endpoint.name, value: parse(endpoint, in: text))
    }

    // MARK: literal scanning

    // `fold` / `isLetter` are module-internal rather than private on purpose:
    // `VoiceLint` scans the same turn text with the same conventions, and two
    // copies of "ASCII fold" in one engine is exactly how a twin drifts.
    static func fold(_ scalar: Unicode.Scalar) -> UInt32 {
        // ASCII case fold, length-preserving by construction, so an index
        // into the folded view is an index into the original.
        (scalar.value >= 65 && scalar.value <= 90) ? scalar.value + 32 : scalar.value
    }

    private static func folded(_ text: String) -> [UInt32] {
        text.unicodeScalars.map(fold)
    }

    private static func folded(_ scalars: [Unicode.Scalar]) -> [UInt32] {
        scalars.map(fold)
    }

    static func isLetter(_ scalar: Unicode.Scalar) -> Bool {
        // Unicode general category L* — the twin's
        // `unicodedata.category(ch).startswith("L")`, spelled out rather than
        // delegated to `isLetter`, whose Alphabetic property is a different
        // (wider) set.
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
            .modifierLetter, .otherLetter:
            true
        default:
            false
        }
    }

    private static func isDigit(_ scalar: Unicode.Scalar) -> Bool {
        // ASCII digits only: a Unicode decimal would parse here and then
        // diverge from the Python twin, or fail conversion outright.
        scalar.value >= 48 && scalar.value <= 57
    }

    /// Index just past the FIRST case-insensitive occurrence of `marker`.
    private static func markerEnd(_ scalars: [Unicode.Scalar], marker: String) -> Int? {
        guard let index = find(folded(scalars), folded(marker), from: 0) else { return nil }
        return index + marker.unicodeScalars.count
    }

    /// Naive literal search — deliberately, so both engines agree on it.
    private static func find(_ haystack: [UInt32], _ needle: [UInt32], from: Int) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var start = Swift.max(0, from)
        while start + needle.count <= haystack.count {
            var offset = 0
            while offset < needle.count, haystack[start + offset] == needle[offset] {
                offset += 1
            }
            if offset == needle.count { return start }
            start += 1
        }
        return nil
    }

    /// Whole-word: neither neighbour in the FULL text is a letter.
    private static func boundaryOK(_ scalars: [Unicode.Scalar], _ start: Int, _ end: Int) -> Bool {
        if start > 0, isLetter(scalars[start - 1]) { return false }
        if end < scalars.count, isLetter(scalars[end]) { return false }
        return true
    }

    private static func parseChoice(
        _ endpoint: TurnEndpoint, _ scalars: [Unicode.Scalar], from start: Int
    ) -> String? {
        let windowEnd = Swift.min(scalars.count, start + window)
        let haystack = folded(scalars)
        // (position, -length, declaration order) — earliest position wins;
        // ties break on longest, then declaration order.
        var best: (position: Int, length: Int, order: Int, member: String)?
        for (order, member) in (endpoint.vocabulary ?? []).enumerated() {
            let needle = folded(member)
            var index = find(haystack, needle, from: start)
            while let found = index, found + needle.count <= windowEnd {
                if boundaryOK(scalars, found, found + needle.count) {
                    let candidate = (found, needle.count, order, member)
                    if best == nil
                        || candidate.0 < best!.position
                        || (candidate.0 == best!.position
                            && (candidate.1 > best!.length
                                || (candidate.1 == best!.length && candidate.2 < best!.order)))
                    {
                        best = candidate
                    }
                    break
                }
                index = find(haystack, needle, from: found + 1)
            }
        }
        return best?.member
    }

    private static func parseNumber(
        _ endpoint: TurnEndpoint, _ scalars: [Unicode.Scalar], from start: Int
    ) -> Double? {
        let windowEnd = Swift.min(scalars.count, start + window)
        var index = start
        while index < windowEnd {
            var cursor = index
            if scalars[cursor] == "+" || scalars[cursor] == "-" { cursor += 1 }
            guard cursor < windowEnd, isDigit(scalars[cursor]) else {
                index += 1
                continue
            }
            while cursor < windowEnd, isDigit(scalars[cursor]) { cursor += 1 }
            if cursor + 1 < windowEnd, scalars[cursor] == ".", isDigit(scalars[cursor + 1]) {
                cursor += 1
                while cursor < windowEnd, isDigit(scalars[cursor]) { cursor += 1 }
            }
            // Clipped by the window edge: the digits continue past it, so
            // what we can see is a PREFIX of the number, not the number.
            // Reading it would invent a value ("6" out of "600") — the worst
            // possible failure for a numeric endpoint.
            if cursor == windowEnd, cursor < scalars.count {
                let tail = scalars[cursor]
                if isDigit(tail)
                    || (tail == "." && cursor + 1 < scalars.count && isDigit(scalars[cursor + 1]))
                {
                    return nil
                }
            }
            guard
                let value = Double(String(String.UnicodeScalarView(scalars[index ..< cursor])))
            else { return nil }
            if let minimum = endpoint.min, value < minimum { return nil }
            if let maximum = endpoint.max, value > maximum { return nil }
            return value
        }
        return nil
    }
}
