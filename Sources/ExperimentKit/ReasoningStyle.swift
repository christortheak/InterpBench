import CryptoKit
import Foundation

/// Reasoning-style measurement as pinned DATA (Python twin:
/// `Server/steerlab_server/experiment/reasoning_style.py`).
///
/// A taxonomy is a versioned JSON file of surface features that measure how a
/// generated response ARGUES (hedging, certainty, enumeration, …) — distinct
/// from the paired judge's holistic preference and from concept marker
/// density. Everything here is concept- and domain-agnostic: features arrive
/// as data (`prompts/taxonomies/<name>.json`), pinned by hash into the
/// manifest (`reasoningStyleTaxonomyPath` + `reasoningStyleTaxonomyHash`),
/// and scored deterministically from the generated text — no model access,
/// so results are recomputable post-hoc (`experiment rescore-style`).
///
/// Cross-engine scoring contract (fixture-tested to 1e-9 on both engines,
/// `Fixtures/reasoning-style/reasoning-style-parity.json`):
/// - generated text AND every pattern are NFC-normalized before any matching
///   (`precomposedStringWithCanonicalMapping` / Python
///   `unicodedata.normalize("NFC", …)`) — decomposed and precomposed accents
///   score identically;
/// - words for matching = maximal runs of Unicode scalars whose general
///   category is a letter (Lu/Ll/Lt/Lm/Lo) or a decimal digit (Nd), taken
///   from the NFC text after mapping EACH scalar through its unconditional
///   full lowercase mapping (per-scalar, never `String.lowercased()` —
///   CPython's `str.lower` applies the Final_Sigma context rule and ICU's
///   whole-string lowercasing does not, so neither whole-string API is the
///   rule; scalars, never grapheme clusters);
/// - `wordList` patterns match as whole-word (contiguous multi-word) sequences;
/// - `regex` patterns are case-insensitive non-overlapping leftmost match
///   counts, restricted to a PARSED portable grammar (`PortableRegexParser`
///   — anything outside it is a load error). Pinned semantics: `.` matches
///   any character except `\n` (INCLUDING `\r` — hence
///   `.useUnixLineSeparators`, without which ICU also excludes `\r`, NEL,
///   LS, PS); `^`/`$` anchor to the whole text (`$` also matches just
///   before one trailing `\n` on both engines); empty-match advancement
///   (e.g. `(?:ab)*`) is fixture-pinned;
/// - `perSentence` divides by sentence count (a `.`/`!`/`?` followed by
///   whitespace or end of text; minimum 1);
/// - `per1kWords` scales by 1000 / whitespace-token count (minimum 1);
/// - `rawCount` is the count itself.
public struct ReasoningStyleTaxonomy: Sendable, Equatable {
    public enum FeatureKind: String, Codable, Sendable {
        case wordList
        case regex
    }

    public enum Normalize: String, Codable, Sendable {
        case perSentence
        case per1kWords
        case rawCount
    }

    public struct Feature: Sendable, Equatable {
        public let id: String
        public let title: String
        public let kind: FeatureKind
        public let patterns: [String]
        public let normalize: Normalize

        public init(
            id: String, title: String, kind: FeatureKind,
            patterns: [String], normalize: Normalize
        ) {
            self.id = id
            self.title = title
            self.kind = kind
            self.patterns = patterns
            self.normalize = normalize
        }
    }

    public let name: String
    public let features: [Feature]

    /// Feature ids in declared taxonomy order — the deterministic column /
    /// endpoint order (`rs_<id>`) on both engines.
    public var featureIDs: [String] { features.map(\.id) }

    // MARK: - Load + validation

    private struct File: Decodable {
        struct RawFeature: Decodable {
            var id: String?
            var title: String?
            var kind: String?
            var patterns: [String]?
            var normalize: String?
        }
        var schemaVersion: Int?
        var name: String?
        var features: [RawFeature]?
    }

    static let featureIDAllowed = CharacterSet(
        charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")

    /// Loads and VALIDATES a taxonomy: unknown schema versions, unknown
    /// kinds/normalize modes, empty/duplicate/unsafe feature ids, empty
    /// pattern lists, word-list patterns with no matchable words, and regex
    /// patterns that fail to compile (or use non-portable lookbehind) are all
    /// load errors — a taxonomy that cannot score identically on both
    /// engines must never be pinned.
    public static func load(data: Data) throws -> ReasoningStyleTaxonomy {
        let file: File
        do {
            file = try JSONDecoder().decode(File.self, from: data)
        } catch {
            throw ExperimentError(reason: "taxonomy is not valid JSON: \(error)")
        }
        guard file.schemaVersion == 1 else {
            throw ExperimentError(
                reason: "taxonomy schemaVersion must be 1 "
                    + "(got \(file.schemaVersion.map(String.init) ?? "none"))")
        }
        let name = (file.name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "taxonomy needs a non-empty \"name\"")
        }
        guard let rawFeatures = file.features, !rawFeatures.isEmpty else {
            throw ExperimentError(reason: "taxonomy needs a non-empty \"features\" list")
        }
        var seen = Set<String>()
        var features: [Feature] = []
        for (index, raw) in rawFeatures.enumerated() {
            let id = (raw.id ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !id.isEmpty else {
                throw ExperimentError(reason: "feature #\(index + 1) has no id")
            }
            guard id.unicodeScalars.allSatisfy(featureIDAllowed.contains) else {
                throw ExperimentError(
                    reason: "feature id '\(id)' must use only [A-Za-z0-9_.-] "
                        + "(it becomes the rs_<id> metric column)")
            }
            guard seen.insert(id).inserted else {
                throw ExperimentError(reason: "duplicate feature id '\(id)'")
            }
            guard let kindRaw = raw.kind, let kind = FeatureKind(rawValue: kindRaw) else {
                throw ExperimentError(
                    reason: "feature '\(id)': unknown kind "
                        + "'\(raw.kind ?? "none")' (expected wordList|regex)")
            }
            guard let normalizeRaw = raw.normalize,
                let normalize = Normalize(rawValue: normalizeRaw)
            else {
                throw ExperimentError(
                    reason: "feature '\(id)': unknown normalize "
                        + "'\(raw.normalize ?? "none")' "
                        + "(expected perSentence|per1kWords|rawCount)")
            }
            // Patterns are NFC-normalized at load — matching happens in NFC
            // space on both engines (generated text is normalized in score()).
            let patterns = (raw.patterns ?? []).filter {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }.map(\.precomposedStringWithCanonicalMapping)
            guard !patterns.isEmpty else {
                throw ExperimentError(
                    reason: "feature '\(id)' needs a non-empty \"patterns\" list")
            }
            switch kind {
            case .wordList:
                for pattern in patterns where matchTokens(pattern).isEmpty {
                    throw ExperimentError(
                        reason: "feature '\(id)': word-list pattern "
                            + "'\(pattern)' contains no matchable words")
                }
            case .regex:
                for pattern in patterns {
                    // Portability guard: the pattern must PARSE inside the
                    // restricted cross-engine grammar — "compiles here" is
                    // not a portability proof (ICU and Python `re` each
                    // accept constructs the other rejects or reads
                    // differently).
                    if let violation = PortableRegexParser.violation(in: pattern) {
                        throw ExperimentError(
                            reason: "feature '\(id)': regex '\(pattern)': "
                                + "\(violation) — not in the portable "
                                + "cross-engine regex subset (see "
                                + "prompts/templates/reasoning-style/README.md)")
                    }
                    do {
                        _ = try NSRegularExpression(
                            pattern: pattern, options: Self.regexOptions)
                    } catch {  // unreachable post-validation
                        throw ExperimentError(
                            reason: "feature '\(id)': regex '\(pattern)' does not "
                                + "compile: \(error.localizedDescription)")
                    }
                }
            }
            features.append(
                Feature(
                    id: id, title: raw.title ?? id, kind: kind,
                    patterns: patterns, normalize: normalize))
        }
        return ReasoningStyleTaxonomy(name: name, features: features)
    }

    public static func load(url: URL) throws -> ReasoningStyleTaxonomy {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ExperimentError(reason: "cannot read taxonomy at \(url.path)")
        }
        return try load(data: data)
    }

    // MARK: - Scoring (the cross-engine math)

    /// The pinned regex option set: case-insensitive, and `\n` as the ONLY
    /// line separator so `.` matches everything except `\n` — Python `re`'s
    /// default (without `.useUnixLineSeparators`, ICU's `.` also excludes
    /// `\r`, NEL, LS and PS, and counts diverge on `\r`-bearing text).
    static let regexOptions: NSRegularExpression.Options = [
        .caseInsensitive, .useUnixLineSeparators,
    ]

    /// Token scalars are letters (Lu/Ll/Lt/Lm/Lo) and DECIMAL digits (Nd) —
    /// a general-category rule, not `Character.isLetter`/`isNumber`
    /// (`isNumber` accepts Roman numerals, Python's `isdigit` accepts
    /// superscript ²; the category rule is identical on both engines).
    static func isTokenScalar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
            .modifierLetter, .otherLetter, .decimalNumber:
            return true
        default:
            return false
        }
    }

    /// The WORD tokenizer for `wordList` matching (identical rule in
    /// `reasoning_style.match_tokens`): NFC-normalize, then map EACH scalar
    /// through its unconditional full lowercase mapping (per-scalar
    /// `lowercaseMapping` — NOT `String.lowercased()`, whose Final_Sigma
    /// context rule CPython applies and ICU's whole-string API does not…
    /// in opposite directions), then take maximal runs of
    /// letter/decimal-digit scalars. Iterates Unicode scalars (== Python
    /// code points), never grapheme clusters — grapheme iteration is why
    /// decomposed accents used to tokenize differently across engines.
    static func matchTokens(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for scalar in text.precomposedStringWithCanonicalMapping.unicodeScalars {
            for mapped in scalar.properties.lowercaseMapping.unicodeScalars {
                if isTokenScalar(mapped) {
                    current.unicodeScalars.append(mapped)
                } else if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
            }
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    /// Sentence count: `.`/`!`/`?` followed by whitespace or end of text;
    /// minimum 1 (so empty text scores 0, never divides by zero).
    static func sentenceCount(_ text: String) -> Int {
        let characters = Array(text)
        var count = 0
        for (index, character) in characters.enumerated()
        where character == "." || character == "!" || character == "?" {
            if index == characters.count - 1 || characters[index + 1].isWhitespace {
                count += 1
            }
        }
        return max(count, 1)
    }

    /// Whitespace-separated token count, minimum 1 — the `per1kWords`
    /// denominator (deliberately the same "words" as `wordCount` metrics).
    static func whitespaceWordCount(_ text: String) -> Int {
        max(text.split(whereSeparator: \.isWhitespace).count, 1)
    }

    /// Raw match count of one feature in `text` (before normalization).
    static func matchCount(feature: Feature, in text: String) -> Int {
        switch feature.kind {
        case .wordList:
            let tokens = matchTokens(text)
            var total = 0
            for pattern in feature.patterns {
                let patternTokens = matchTokens(pattern)
                guard !patternTokens.isEmpty, patternTokens.count <= tokens.count
                else { continue }
                for start in 0 ... (tokens.count - patternTokens.count)
                where Array(tokens[start ..< start + patternTokens.count]) == patternTokens {
                    total += 1
                }
            }
            return total
        case .regex:
            let range = NSRange(text.startIndex ..< text.endIndex, in: text)
            var total = 0
            for pattern in feature.patterns {
                guard
                    let regex = try? NSRegularExpression(
                        pattern: pattern, options: Self.regexOptions)
                else { continue }  // unreachable post-validation
                total += regex.numberOfMatches(in: text, range: range)
            }
            return total
        }
    }

    /// Per-generation feature values, keyed by feature id. Deterministic,
    /// pure CPU; empty text scores 0 on every feature. The text is
    /// NFC-normalized up front (patterns were normalized at load), so
    /// decomposed and precomposed input scores identically on both engines.
    public func score(_ rawText: String) -> [String: Double] {
        let text = rawText.precomposedStringWithCanonicalMapping
        // Denominators computed once per text, not per feature.
        lazy var sentences = Double(Self.sentenceCount(text))
        lazy var words = Double(Self.whitespaceWordCount(text))
        var values: [String: Double] = [:]
        for feature in features {
            let count = Double(Self.matchCount(feature: feature, in: text))
            switch feature.normalize {
            case .perSentence: values[feature.id] = count / sentences
            case .per1kWords: values[feature.id] = count * 1000.0 / words
            case .rawCount: values[feature.id] = count
            }
        }
        return values
    }
}

// MARK: - Portable regex grammar (the cross-engine contract)

/// Recursive-descent validator for the PORTABLE cross-engine regex grammar.
///
/// "Compiles on both engines" is NOT a portability proof: ICU accepts `\R`
/// and `(?<name>…)` which Python rejects, Python accepts `(?P<name>…)` and a
/// literal bare `{` which ICU rejects, and class syntax silently DIVERGES
/// (ICU treats `[[a]]` as a nested set and `[a&&b]` as an intersection;
/// Python treats both as literals). So this parser accepts a restricted
/// grammar and rejects everything outside it by construct name + scalar
/// position. Identical implementation in Python
/// `reasoning_style._PortableRegexParser` — keep the two in lockstep; the
/// shared acceptance/rejection vectors live in
/// `Fixtures/reasoning-style/portable-regex-vectors.json`.
///
/// Accepted grammar:
///
///     pattern     ::= alternation                      (then end of pattern)
///     alternation ::= sequence ("|" sequence)*         (empty branches legal)
///     sequence    ::= term*
///     term        ::= atom quantifier?
///     atom        ::= literal | "." | "^" | "$" | escape | class
///                   | "(?:" alternation ")"            (non-capturing only)
///     literal     ::= any scalar EXCEPT  \ ^ $ . | ? * + ( ) [ { }
///     escape      ::= "\" ( d D w W s S               class escapes
///                         | b                          word boundary (not in
///                                                      classes, unquantifiable)
///                         | n t r                      control literals
///                         | ASCII punctuation )        escaped literal
///     class       ::= "[" "^"? member+ "]"
///     member      ::= class-literal | escape-in-class | range
///     range       ::= single-char "-" single-char      (left <= right; "-" is
///                                                      literal at start, before
///                                                      "]", or escaped)
///     quantifier  ::= ("*" | "+" | "?" | "{m}" | "{m,}" | "{m,n}") "?"?
///                                                      (m <= n <= 9999; lazy
///                                                      "?" ok, possessive "+"
///                                                      rejected)
///
/// Rejected by parse (not by substring): capturing `(`, named groups (both
/// dialects), backreferences/octal, all four lookarounds, inline flags,
/// atomic/conditional/comment groups, `\p{…}`, `\x`/`\u`/`\U`/`\N`, `\R` and
/// every other letter escape, nested `[` / `&&` / `\b` inside classes, class
/// escapes in ranges, reversed ranges, and bare `{` `}` that are not
/// well-formed quantifiers.
struct PortableRegexParser {
    private struct Violation: Error {
        let construct: String
        let position: Int
        var message: String { "\(construct) at position \(position)" }
    }

    private static let classSetEscapes = Set("dDwWsS".unicodeScalars)
    private static let controlEscapes: [Unicode.Scalar: Unicode.Scalar] = [
        "n": "\n", "t": "\t", "r": "\r",
    ]
    private static let asciiDigits = Set("0123456789".unicodeScalars)
    private static let asciiPunctuation = Set(
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".unicodeScalars)
    private static let maxGroupDepth = 64
    private static let maxQuantifierBound = 9999

    private let scalars: [Unicode.Scalar]
    private var index = 0
    private var depth = 0

    private init(_ pattern: String) {
        scalars = Array(pattern.unicodeScalars)
    }

    /// nil when `pattern` is inside the portable grammar; otherwise
    /// "<construct> at position <N>" (0-based Unicode scalar offsets ==
    /// Python string indices).
    static func violation(in pattern: String) -> String? {
        var parser = PortableRegexParser(pattern)
        do {
            try parser.parsePattern()
            return nil
        } catch let violation as Violation {
            return violation.message
        } catch {
            return "unexpected regex validation failure"  // unreachable
        }
    }

    private func peek(_ offset: Int = 0) -> Unicode.Scalar? {
        let j = index + offset
        return j < scalars.count ? scalars[j] : nil
    }

    private func reject(_ construct: String, at position: Int) throws -> Never {
        throw Violation(construct: construct, position: position)
    }

    private mutating func parsePattern() throws {
        try parseAlternation()
        if index < scalars.count {  // only a stray ')' can stop the parse
            try reject("unmatched ')'", at: index)
        }
    }

    private mutating func parseAlternation() throws {
        try parseSequence()
        while peek() == "|" {
            index += 1
            try parseSequence()
        }
    }

    private mutating func parseSequence() throws {
        while let c = peek(), c != "|", c != ")" {
            try parseTerm()
        }
    }

    private mutating func parseTerm() throws {
        let quantifiable = try parseAtom()
        try parseOptionalQuantifier(quantifiable: quantifiable)
    }

    /// Consumes one atom; returns true when a quantifier may follow it.
    private mutating func parseAtom() throws -> Bool {
        let position = index
        guard let next = peek() else {
            try reject("unexpected end of pattern", at: position)  // unreachable
        }
        switch next {
        case "(":
            try parseGroup()
            return true
        case "[":
            try parseClass()
            return true
        case "*", "+", "?":
            try reject("quantifier with nothing to repeat", at: position)
        case "{":
            if scanBraceQuantifier() != nil {
                try reject("quantifier with nothing to repeat", at: position)
            }
            try reject("literal '{' must be escaped as '\\{'", at: position)
        case "}":
            try reject("literal '}' must be escaped as '\\}'", at: position)
        case "^", "$":
            index += 1
            return false
        case "\\":
            if case .boundary = try parseEscape(inClass: false) { return false }
            return true
        default:
            index += 1  // every other scalar is a literal (incl. ']' and '.')
            return true
        }
    }

    private mutating func parseOptionalQuantifier(quantifiable: Bool) throws {
        let start = index
        switch peek() {
        case "*"?, "+"?, "?"?:
            index += 1
        case "{"?:
            guard let (end, low, high) = scanBraceQuantifier() else {
                try reject("literal '{' must be escaped as '\\{'", at: start)
            }
            if low > Self.maxQuantifierBound || (high ?? 0) > Self.maxQuantifierBound {
                try reject("quantifier bound too large", at: start)
            }
            if let high, low > high {
                try reject("reversed '{m,n}' quantifier", at: start)
            }
            index = end
        default:
            return
        }
        if !quantifiable {
            try reject("quantifier on an anchor", at: start)
        }
        if peek() == "?" { index += 1 }  // lazy
        if peek() == "+" {
            try reject("possessive quantifier", at: index)
        }
    }

    /// Probes (without consuming) a `{m}`/`{m,}`/`{m,n}` at `index`. Returns
    /// (index after `}`, m, n or nil for open-ended) or nil when it is not
    /// one — the two engines DISAGREE about a bare `{` (Python: literal,
    /// ICU: error), so a `{` must be a well-formed quantifier or an escaped
    /// literal.
    private func scanBraceQuantifier() -> (end: Int, low: Int, high: Int?)? {
        var j = index + 1
        var lowDigits = ""
        while j < scalars.count, Self.asciiDigits.contains(scalars[j]) {
            lowDigits.unicodeScalars.append(scalars[j])
            j += 1
        }
        guard !lowDigits.isEmpty, let low = Int(lowDigits) else { return nil }
        if j < scalars.count, scalars[j] == "}" {
            return (j + 1, low, low)
        }
        if j < scalars.count, scalars[j] == "," {
            j += 1
            var highDigits = ""
            while j < scalars.count, Self.asciiDigits.contains(scalars[j]) {
                highDigits.unicodeScalars.append(scalars[j])
                j += 1
            }
            if j < scalars.count, scalars[j] == "}" {
                return (j + 1, low, highDigits.isEmpty ? nil : Int(highDigits))
            }
        }
        return nil
    }

    private mutating func parseGroup() throws {
        let start = index
        index += 1  // '('
        guard peek() == "?" else {
            try reject("capturing group '('", at: start)
        }
        index += 1
        switch peek() {
        case ":"?:
            index += 1
        case "P"?:
            try reject("named group '(?P'", at: start)
        case "<"?:
            if peek(1) == "=" { try reject("lookbehind '(?<='", at: start) }
            if peek(1) == "!" { try reject("lookbehind '(?<!'", at: start) }
            try reject("named group '(?<'", at: start)
        case "="?:
            try reject("lookahead '(?='", at: start)
        case "!"?:
            try reject("lookahead '(?!'", at: start)
        case ">"?:
            try reject("atomic group '(?>'", at: start)
        case "("?:
            try reject("conditional group '(?('", at: start)
        case "#"?:
            try reject("comment group '(?#'", at: start)
        default:
            try reject("inline flags '(?'", at: start)
        }
        depth += 1
        if depth > Self.maxGroupDepth {
            try reject("group nesting too deep", at: start)
        }
        try parseAlternation()
        guard peek() == ")" else {
            try reject("unterminated group", at: start)
        }
        index += 1
        depth -= 1
    }

    private enum EscapeKind {
        case set  // \d \D \w \W \s \S
        case boundary  // \b
        case literal(Unicode.Scalar)
    }

    /// Consumes a `\`-escape (cursor on the backslash).
    private mutating func parseEscape(inClass: Bool) throws -> EscapeKind {
        let start = index
        index += 1  // '\'
        guard let c = peek() else {
            try reject("trailing backslash", at: start)
        }
        if Self.classSetEscapes.contains(c) {
            index += 1
            return .set
        }
        if c == "b" {
            if inClass {
                try reject("'\\b' inside a character class", at: start)
            }
            index += 1
            return .boundary
        }
        if let control = Self.controlEscapes[c] {
            index += 1
            return .literal(control)
        }
        if ("1" ... "9").contains(c) {
            try reject("backreference escape '\\\(c)'", at: start)
        }
        if c == "0" {
            try reject("octal escape '\\0'", at: start)
        }
        if c == "p" || c == "P" {
            try reject("unicode property escape '\\\(c)'", at: start)
        }
        if c == "x" || c == "u" || c == "U" || c == "N" {
            try reject("hex/unicode escape '\\\(c)'", at: start)
        }
        if Self.asciiPunctuation.contains(c) {
            index += 1
            return .literal(c)
        }
        try reject("unsupported escape '\\\(c)'", at: start)
    }

    private enum ClassMember {
        case single(Unicode.Scalar)
        case set
        case range
    }

    private mutating func parseClass() throws {
        let start = index
        index += 1  // '['
        if peek() == "^" { index += 1 }
        if peek() == "]" {
            try reject("empty character class", at: start)
        }
        var last: ClassMember?
        while true {
            guard let c = peek() else {
                try reject("unterminated character class", at: start)
            }
            if c == "]" {
                index += 1
                return
            }
            if c == "[" {
                try reject("nested '[' in a character class", at: index)
            }
            if c == "&", peek(1) == "&" {
                try reject("set operation '&&' in a character class", at: index)
            }
            if c == "-", let previous = last, peek(1) != "]" {
                let dash = index
                switch previous {
                case .set:
                    try reject("class escape in a character range", at: dash)
                case .range:
                    try reject("ambiguous '-' in a character class", at: dash)
                case .single(let low):
                    index += 1
                    let high = try parseRangeEndpoint(dash: dash, classStart: start)
                    if low.value > high.value {
                        try reject("reversed character range", at: dash)
                    }
                    last = .range
                    continue
                }
            }
            if c == "\\" {
                switch try parseEscape(inClass: true) {
                case .set: last = .set
                case .literal(let scalar): last = .single(scalar)
                case .boundary: last = nil  // unreachable: rejected above
                }
                continue
            }
            index += 1
            last = .single(c)
        }
    }

    private mutating func parseRangeEndpoint(
        dash: Int, classStart: Int
    ) throws -> Unicode.Scalar {
        guard let c = peek() else {
            try reject("unterminated character class", at: classStart)
        }
        if c == "[" {
            try reject("nested '[' in a character class", at: index)
        }
        if c == "\\" {
            switch try parseEscape(inClass: true) {
            case .set, .boundary:
                try reject("class escape in a character range", at: dash)
            case .literal(let scalar):
                return scalar
            }
        }
        index += 1
        return c
    }
}

/// A hash-verified, ready-to-score pinned taxonomy.
public struct PinnedReasoningStyle: Sendable {
    public let taxonomy: ReasoningStyleTaxonomy
    public let path: String
    public let hash: String
}

extension ExperimentStore {
    /// THE workspace-relative home of reasoning-style taxonomies.
    public static func taxonomiesRelativeDirectory() -> String { "prompts/taxonomies" }

    /// Loads the manifest's pinned reasoning-style taxonomy, HASH-CHECKED
    /// against the pinned `reasoningStyleTaxonomyHash` (drift throws — the
    /// scoring path must never read a file verify() would reject). Returns
    /// nil when the manifest pins no taxonomy (absent = no reasoning-style
    /// scoring, no violation).
    public static func loadPinnedReasoningStyle(
        _ manifest: ExperimentManifest
    ) throws -> PinnedReasoningStyle? {
        guard let path = manifest.reasoningStyleTaxonomyPath else {
            guard manifest.reasoningStyleTaxonomyHash == nil else {
                throw ExperimentError(
                    reason: "reasoningStyleTaxonomyHash pinned without a path — "
                        + "an unresolvable pin certifies nothing")
            }
            return nil
        }
        guard let pinned = manifest.reasoningStyleTaxonomyHash else {
            throw ExperimentError(
                reason: "reasoning-style taxonomy pin is incomplete — "
                    + "reasoningStyleTaxonomyPath and reasoningStyleTaxonomyHash "
                    + "must both be set (re-pin with "
                    + "'experiment set-style-taxonomy')")
        }
        let url = resolveProjectPath(path)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(
                reason: "pinned reasoning-style taxonomy missing (\(path))")
        }
        let live = sha256Hex(data)
        guard live == pinned else {
            throw ExperimentError(
                reason: "reasoning-style taxonomy changed since pinning "
                    + "(have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)")
        }
        return PinnedReasoningStyle(
            taxonomy: try ReasoningStyleTaxonomy.load(data: data),
            path: path, hash: pinned)
    }

    /// The one true pin gesture (`experiment set-style-taxonomy`): validates
    /// the taxonomy LOADS (both-engine loadability is the contract), stamps
    /// `reasoningStyleTaxonomyPath` + `reasoningStyleTaxonomyHash` (SHA-256
    /// of the file bytes), and saves. Draft-only, like every manifest edit —
    /// `save` refuses frozen manifests.
    @discardableResult
    public static func pinReasoningStyleTaxonomy(
        experimentName: String, path: String
    ) throws -> ExperimentManifest {
        var manifest = try load(name: experimentName)
        let url = resolveProjectPath(path)
        guard let data = try? Data(contentsOf: url) else {
            // The third pin verb with the same hole (2026-08-18): the PROSE
            // already named the convention directory, and the envelope still
            // said `failed`/70/`verbFailed` with "read the reason" — an
            // operational failure, which tells an agent to retry rather than
            // to author the file. Prose byte-preserved; the gate and the
            // runnable repair are new.
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "no taxonomy file at \(path) — author one under "
                    + "\(taxonomiesRelativeDirectory())/ (see "
                    + "prompts/templates/reasoning-style/)",
                repair: "author \(path) under \(taxonomiesRelativeDirectory())/ "
                    + "(copy one from prompts/templates/reasoning-style/), then "
                    + "steerlab-cli experiment set-style-taxonomy "
                    + "\(experimentName) \(path)")
        }
        _ = try ReasoningStyleTaxonomy.load(data: data)  // validation gate
        manifest.reasoningStyleTaxonomyPath = path
        manifest.reasoningStyleTaxonomyHash = sha256Hex(data)
        try save(manifest)
        return manifest
    }
}
