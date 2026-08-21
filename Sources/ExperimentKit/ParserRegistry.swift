import CryptoKit
import Foundation

/// Workspace-declared numeric answer parsers (USABILITY-PLAN Phase-4 item
/// 18): numeric-outcome grammars become workspace DATA, not app code. A study
/// names a parser from `prompts/parsers/parser-registry.json` (manifest key
/// `numericParser`) and the run parses its numeric outcome with that
/// declared grammar instead of the built-in duration parser.
///
/// Two kinds exist:
/// - `durationMonths`: a unit table as data (token → months multiplier),
///   joiner words as data, compound support ("8 years 3 months" = 99, both
///   terms count), ranges under a declared policy. The SHIPPED default
///   entry ("sentencing-months", named for the judicial-decision study it was
///   written for) reproduces `Judicial.parseMonths` exactly — fixture-locked
///   on both engines.
/// - `number`: plain numeric extraction with declared percent and range
///   policies.
///
/// Cross-engine contract (server twin:
/// `Server/steerlab_server/experiment/parser_registry.py`): the registry
/// file path, the manifest keys (`numericParser` + `parserRegistryHash`),
/// the spec vocabulary (kinds, range/percent policies), and the report.json
/// `numericParser` provenance block are identical. When no parser is named,
/// behavior is exactly the historical one (`caseFamily == "sentencing"` →
/// built-in `parseMonths`); no new file is required and legacy studies gain
/// no violations. That implicit selection is DEPRECATED as of 2026-08-18:
/// it still works, and every site where it fires now says so
/// (`ExperimentManifest.implicitCaseFamilyAdvisory`, CLI advisory code
/// `deprecatedImplicitSelection`). Declaring a parser is the mechanism.
public enum ParserRegistry {

    /// The one workspace location the registry lives at (seeded into new
    /// workspaces from the repo template; both engines resolve this path).
    public static let registryFile = "prompts/parsers/parser-registry.json"

    public static let knownKinds = ["durationMonths", "number"]
    public static let rangePolicies = ["mean", "refuse", "first"]
    public static let percentPolicies = ["accept", "refuse", "fraction"]

    /// One parser declaration as it appears in the registry JSON. All
    /// fields optional at decode; `validate` reports what is missing in
    /// plain words.
    public struct Spec: Decodable, Sendable {
        public var kind: String?
        public var description: String?
        public var units: [String: Double]?
        public var joiners: [String]?
        public var range: String?
        public var percent: String?
        public var decimalComma: Bool?
    }

    private struct Registry: Decodable {
        var schemaVersion: Int?
        var parsers: [String: Spec]?
    }

    /// The report.json `numericParser` block (cross-engine contract keys:
    /// name, kind, registryFile, registryHash).
    public struct NumericParserProvenance: Codable, Sendable, Equatable {
        public let name: String
        public let kind: String
        public let registryFile: String
        public let registryHash: String
    }

    /// A resolved, ready-to-run declared parser plus the provenance the run
    /// stamps into report.json.
    public struct ResolvedNumericParser: Sendable {
        public let name: String
        public let kind: String
        public let registryHash: String
        let parseFunction: @Sendable (String) -> Double?

        public func parse(_ text: String) -> Double? { parseFunction(text) }

        public var provenance: NumericParserProvenance {
            NumericParserProvenance(
                name: name, kind: kind,
                registryFile: ParserRegistry.registryFile,
                registryHash: registryHash)
        }
    }

    /// The registry's workspace location (test-seam aware, like every other
    /// pinned input).
    static var registryURL: URL {
        ExperimentStore.resolveProjectPath(registryFile)
    }

    /// SHA-256 over the registry file's raw bytes, or nil when absent —
    /// the value freeze pins as `parserRegistryHash`.
    public static func liveHash(at url: URL? = nil) -> String? {
        guard let data = try? Data(contentsOf: url ?? registryURL) else {
            return nil
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The parsed registry's parser table, or a plain-language refusal.
    static func loadSpecs(at url: URL? = nil) throws -> [String: Spec] {
        let location = url ?? registryURL
        guard let data = try? Data(contentsOf: location) else {
            throw ExperimentError(
                reason: "no parser registry exists at \(registryFile) — new "
                    + "workspaces are seeded with a template; create the file "
                    + "(or remove the study's numericParser)")
        }
        let registry: Registry
        do {
            registry = try JSONDecoder().decode(Registry.self, from: data)
        } catch {
            throw ExperimentError(
                reason: "the parser registry \(registryFile) is not readable "
                    + "as JSON (\(error.localizedDescription)) — it must be an "
                    + "object like {\"schemaVersion\": 1, \"parsers\": "
                    + "{\"<name>\": {…}}}")
        }
        guard let parsers = registry.parsers else {
            throw ExperimentError(
                reason: "the parser registry \(registryFile) must be an object "
                    + "with a \"parsers\" object mapping parser names to their "
                    + "declarations — start from the shipped template")
        }
        guard registry.schemaVersion == 1 else {
            throw ExperimentError(
                reason: "the parser registry \(registryFile) declares "
                    + "schemaVersion \(registry.schemaVersion.map(String.init) ?? "nothing") "
                    + "— this engine reads version 1")
        }
        return parsers
    }

    /// The validated spec for one named parser.
    static func spec(named name: String, at url: URL? = nil) throws -> Spec {
        let parsers = try loadSpecs(at: url)
        guard let spec = parsers[name] else {
            let known = parsers.keys.sorted().joined(separator: ", ")
            throw ExperimentError(
                reason: "the registry defines no parser named '\(name)' — "
                    + "defined: \(known.isEmpty ? "none" : known)")
        }
        try validate(spec, name: name)
        return spec
    }

    /// Shape-check one parser declaration (the pin-time schema-validation
    /// rule: say what is wrong at the moment of declaration, not at the
    /// failing run).
    static func validate(_ spec: Spec, name: String) throws {
        guard let kind = spec.kind, knownKinds.contains(kind) else {
            throw ExperimentError(
                reason: "parser '\(name)' declares kind "
                    + "'\(spec.kind ?? "nothing")' — known kinds: "
                    + knownKinds.joined(separator: ", "))
        }
        if let range = spec.range, !rangePolicies.contains(range) {
            throw ExperimentError(
                reason: "parser '\(name)' declares range '\(range)' — declare "
                    + "one of " + rangePolicies.joined(separator: ", ")
                    + " (what to do with '5-7': the midpoint, a parse "
                    + "failure, or the first value)")
        }
        if kind == "durationMonths" {
            guard let units = spec.units, !units.isEmpty else {
                throw ExperimentError(
                    reason: "parser '\(name)' is a durationMonths parser but "
                        + "declares no \"units\" table — declare unit tokens "
                        + "mapping to a months multiplier, e.g. "
                        + "{\"years\": 12, \"months\": 1}")
            }
            for (token, multiplier) in units {
                if token.trimmingCharacters(in: .whitespaces).isEmpty {
                    throw ExperimentError(
                        reason: "parser '\(name)': every units key must be a "
                            + "non-empty unit token (a word like 'years')")
                }
                if !(multiplier > 0) {
                    throw ExperimentError(
                        reason: "parser '\(name)': unit '\(token)' has "
                            + "multiplier \(multiplier) — each unit maps to a "
                            + "positive number of months (years: 12, months: 1)")
                }
            }
            if let joiners = spec.joiners,
                joiners.contains(where: {
                    $0.trimmingCharacters(in: .whitespaces).isEmpty
                })
            {
                throw ExperimentError(
                    reason: "parser '\(name)': \"joiners\" must be a list of "
                        + "words that may join compound terms, e.g. "
                        + "[\"and\", \"und\"]")
            }
        } else if let percent = spec.percent, !percentPolicies.contains(percent) {
            throw ExperimentError(
                reason: "parser '\(name)' declares percent '\(percent)' — "
                    + "declare one of " + percentPolicies.joined(separator: ", ")
                    + " (what to do with '42%': read 42, a parse failure, or "
                    + "read 0.42)")
        }
    }

    /// Resolve a manifest's `numericParser` to a runnable parser, or nil
    /// when the study names none (the legacy path). Refuses — with the file
    /// and remedy named — on a missing registry, an undefined or malformed
    /// parser, or drift from the pinned registry hash.
    public static func resolveNumericParser(
        _ manifest: ExperimentManifest
    ) throws -> ResolvedNumericParser? {
        let name = manifest.numericParser?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !name.isEmpty else { return nil }
        let spec = try spec(named: name)
        guard let hash = liveHash() else {
            throw ExperimentError(
                reason: "no parser registry exists at \(registryFile)")
        }
        if let pinned = manifest.parserRegistryHash, pinned != hash {
            throw ExperimentError(
                reason: "parser registry '\(registryFile)' drifted from the "
                    + "pinned hash (have \(hash.prefix(12))…, pinned "
                    + "\(pinned.prefix(12))…)")
        }
        return ResolvedNumericParser(
            name: name, kind: spec.kind ?? "", registryHash: hash,
            parseFunction: try buildParser(name: name, spec: spec))
    }

    /// Verify-surface checks for the declared parser + its registry pin
    /// (server `parser_pin_violations` twin). A study that names no parser
    /// and pins no hash gets an EMPTY list — legacy manifests gain no
    /// violations.
    static func pinViolations(_ manifest: ExperimentManifest) -> [String] {
        let name = manifest.numericParser?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !name.isEmpty else {
            if manifest.parserRegistryHash != nil {
                return [
                    "parserRegistryHash is pinned but no numericParser is "
                        + "declared — an unused pin certifies nothing; declare "
                        + "the parser or remove the pin"
                ]
            }
            return []
        }
        let live = liveHash()
        if let pinned = manifest.parserRegistryHash {
            guard let live else {
                return ["pinned parser registry missing at \(registryFile)"]
            }
            if live != pinned {
                return [
                    "parser registry changed since pinning (have "
                        + "\(live.prefix(12))…, pinned \(pinned.prefix(12))…)"
                ]
            }
        }
        do {
            _ = try spec(named: name)
        } catch {
            let reason = (error as? ExperimentError)?.reason ?? "\(error)"
            return ["numericParser '\(name)': \(reason)"]
        }
        return []
    }

    // MARK: - Parser construction

    /// A `text → value-or-nil` closure (nil = parse failure, counted —
    /// never coerced to 0 or a partial number).
    static func buildParser(
        name: String, spec: Spec
    ) throws -> @Sendable (String) -> Double? {
        try validate(spec, name: name)
        if spec.kind == "durationMonths" {
            return try buildDurationParser(name: name, spec: spec)
        }
        return buildNumberParser(spec: spec)
    }

    private static func numberPattern(decimalComma: Bool) -> String {
        decimalComma ? #"(\d+(?:[.,]\d+)?)"# : #"(\d+(?:\.\d+)?)"#
    }

    /// The spelled-out formal register's number-word alternation (one through
    /// twelve — `Judicial.numberWords`), inside the digits-or-word capturing
    /// group of `numberOrWordPattern`.
    private static let numberWordFragment =
        #"\b(?:"#
        + Judicial.numberWords.keys.sorted().joined(separator: "|")
        + #")\b"#

    /// Digits-or-number-word, one capturing group — formal registers spell
    /// their numbers out ("ten years and six months' imprisonment"), and a
    /// digit-only grammar would drop those records non-randomly.
    /// durationMonths compound/single terms only; ranges and the plain
    /// `number` kind stay digit-only (server twin
    /// `_number_or_word_pattern`).
    private static func numberOrWordPattern(decimalComma: Bool) -> String {
        let digits = decimalComma ? #"\d+(?:[.,]\d+)?"# : #"\d+(?:\.\d+)?"#
        return "(" + digits + "|" + numberWordFragment + ")"
    }

    private static func toDouble(_ token: String, decimalComma: Bool) -> Double? {
        if let word = Judicial.numberWords[token.lowercased()] { return word }
        return Double(decimalComma ? token.replacingOccurrences(of: ",", with: ".") : token)
    }

    /// One capturing alternation over unit tokens, longest first (so
    /// 'monaten' wins over 'mo' regardless of declaration order).
    private static func alternation(_ tokens: [String]) -> String {
        let ordered = tokens.sorted {
            $0.count != $1.count ? $0.count > $1.count : $0 < $1
        }
        return "("
            + ordered.map(NSRegularExpression.escapedPattern(for:))
                .joined(separator: "|") + ")"
    }

    /// Case-insensitive, simple-word-boundary compile — the same options the
    /// built-in `Judicial.parseMonths` regexes use.
    private static func compile(_ pattern: String) -> Regex<AnyRegexOutput>? {
        guard let regex = try? Regex(pattern) else { return nil }
        return regex.ignoresCase().wordBoundaryKind(.simple)
    }

    private static func group(
        _ match: Regex<AnyRegexOutput>.Match, _ index: Int
    ) -> String? {
        guard index < match.output.count,
            let substring = match.output[index].substring
        else { return nil }
        return String(substring)
    }

    /// The declared-unit-grammar generalization of `Judicial.parseMonths`:
    /// compound (larger unit then smaller unit, optional comma/joiner), then
    /// range, then single — the shipped default table reproduces the
    /// built-in duration behavior fixture-for-fixture on both engines.
    private static func buildDurationParser(
        name: String, spec: Spec
    ) throws -> @Sendable (String) -> Double? {
        let units = Dictionary(
            (spec.units ?? [:]).map { ($0.key.lowercased(), $0.value) },
            uniquingKeysWith: { first, _ in first })
        guard let minimum = units.values.min() else {
            throw ExperimentError(
                reason: "parser '\(name)' has an empty units table")
        }
        let minors = units.filter { $0.value == minimum }.map(\.key)
        let majors = units.filter { $0.value != minimum }.map(\.key)
        let joiners = (spec.joiners ?? []).map { $0.lowercased() }
        let rangePolicy = spec.range ?? "mean"
        let decimalComma = spec.decimalComma ?? true

        let num = numberPattern(decimalComma: decimalComma)
        let numOrWord = numberOrWordPattern(decimalComma: decimalComma)
        let anyUnits = alternation(Array(units.keys))
        let joinerPart =
            joiners.isEmpty
            ? ""
            : #"(?:(?:"#
                + joiners.map(NSRegularExpression.escapedPattern(for:))
                    .joined(separator: "|")
                + #")\b\s*)?"#
        let compoundPattern: String? =
            (majors.isEmpty || minors.isEmpty)
            ? nil
            : numOrWord + #"\s*"# + alternation(majors) + #"\b\s*,?\s*"#
                + joinerPart
                + numOrWord + #"\s*"# + alternation(minors) + #"\b"#
        let rangePattern =
            num + #"\s*(?:to|-|–|—)\s*"# + num + #"\s*"# + anyUnits + #"\b"#
        let singlePattern = numOrWord + #"\s*"# + anyUnits + #"\b"#
        // Surface an unbuildable grammar at resolve time, in plain words —
        // never as a nil parse at run time.
        for pattern in [compoundPattern, rangePattern, singlePattern] {
            if let pattern, compile(pattern) == nil {
                throw ExperimentError(
                    reason: "parser '\(name)': could not build a matcher from "
                        + "its unit table — check the unit tokens and joiners "
                        + "for unusual characters")
            }
        }

        // Pattern STRINGS (Sendable) are captured; each parse re-compiles —
        // compilation was proven above, and study-scale record counts make
        // the cost negligible.
        return { text in
            guard !text.isEmpty else { return nil }
            if let compoundPattern, let compound = compile(compoundPattern),
                let match = text.firstMatch(of: compound),
                let first = group(match, 1)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let majorUnit = group(match, 2),
                let second = group(match, 3)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let minorUnit = group(match, 4),
                let majorMultiplier = units[majorUnit.lowercased()],
                let minorMultiplier = units[minorUnit.lowercased()]
            {
                return first * majorMultiplier + second * minorMultiplier
            }
            if let ranged = compile(rangePattern),
                let match = text.firstMatch(of: ranged),
                let low = group(match, 1)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let high = group(match, 2)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let unit = group(match, 3),
                let multiplier = units[unit.lowercased()]
            {
                if rangePolicy == "refuse" { return nil }
                let value = rangePolicy == "first" ? low : (low + high) / 2.0
                return value * multiplier
            }
            if let single = compile(singlePattern),
                let match = text.firstMatch(of: single),
                let value = group(match, 1)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let unit = group(match, 2),
                let multiplier = units[unit.lowercased()]
            {
                return value * multiplier
            }
            return nil
        }
    }

    /// Plain numeric extraction: the FIRST number in the text, with declared
    /// range and percent policies (identical logic on both engines —
    /// fixture-locked).
    private static func buildNumberParser(
        spec: Spec
    ) -> @Sendable (String) -> Double? {
        let rangePolicy = spec.range ?? "refuse"
        let percentPolicy = spec.percent ?? "accept"
        let decimalComma = spec.decimalComma ?? true
        let num = numberPattern(decimalComma: decimalComma)
        let rangePattern = num + #"\s*(?:to|-|–|—)\s*"# + num
        let singlePattern = num

        @Sendable func applyPercent(
            _ value: Double, text: String, end: String.Index
        ) -> Double? {
            let tail = text[end...].drop(while: \.isWhitespace)
            if tail.first == "%" {
                if percentPolicy == "refuse" { return nil }
                if percentPolicy == "fraction" { return value / 100.0 }
            }
            return value
        }

        return { text in
            guard !text.isEmpty,
                let single = compile(singlePattern),
                let first = text.firstMatch(of: single),
                let firstValue = group(first, 1)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) })
            else { return nil }
            if let ranged = compile(rangePattern),
                let span = text.firstMatch(of: ranged),
                span.range.lowerBound <= first.range.lowerBound,
                let low = group(span, 1)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) }),
                let high = group(span, 2)
                    .flatMap({ toDouble($0, decimalComma: decimalComma) })
            {
                if rangePolicy == "refuse" { return nil }
                let value = rangePolicy == "first" ? low : (low + high) / 2.0
                return applyPercent(value, text: text, end: span.range.upperBound)
            }
            return applyPercent(
                firstValue, text: text, end: first.range.upperBound)
        }
    }
}
