import Foundation

/// Per-response coding instrument (2026-08-04) — cross-engine twin of the
/// server's `response_coding.py`.
///
/// The paired judge answers "which response is preferred?"; this instrument
/// answers "what does each response contain?" — the K&Z §S9 shape: for each
/// individual response, a blinded coder records declared, typed fields
/// (booleans, integers, categories). No comparison, no winner. A rubric
/// file opts in with a strict frontmatter block:
///
///     ---
///     mode: perResponseCoding
///     field: citesTheGivenRule boolean
///     field: severity integer optional
///     field: stance enum(ruleBound|discretionary|mixed)
///     ---
///     <markdown rubric body the coder reads>
///
/// A rubric with no frontmatter is a paired rubric, byte-for-byte today's
/// behavior. The schema rides inside the rubric file, so the existing
/// `judgeRubricFile` + `judgeRubricHash` pin covers it. The prompt wrapper
/// is BYTE-IDENTICAL to the server's `response_coding.build_prompt`, pinned
/// by the committed goldens in `prompts/fixtures/coding-judge/` — change it
/// only deliberately, on both engines at once, regenerating the goldens.
public enum ResponseCoding {

    /// Cross-engine refusal (server twin: `NO_CODEABLE_MESSAGE`).
    public static let noCodeableMessage =
        "per-response coding found no codeable records: every record is an "
        + "instrument readout or an error record — nothing carries sampled "
        + "text to code. Refusing to write an empty coding report."

    public struct Field: Sendable, Equatable, Codable {
        public let name: String
        /// boolean | integer | number | string | enum
        public let type: String
        public let optional: Bool
        /// Enum vocabulary; empty for every other type.
        public let values: [String]

        /// The field's line in the coding prompt — part of the byte-pinned
        /// wrapper contract.
        var describe: String {
            let base =
                type == "enum" ? "one of: " + values.joined(separator: " | ") : type
            if optional {
                return "- \(name) (\(base); optional — null allowed)"
            }
            return "- \(name) (\(base))"
        }

        // The report's field entry omits `values` for non-enums (server
        // twin spreads it in only when non-empty).
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            try container.encode(type, forKey: .type)
            try container.encode(optional, forKey: .optional)
            if !values.isEmpty {
                try container.encode(values, forKey: .values)
            }
        }
    }

    public struct Schema: Sendable, Equatable {
        public let fields: [Field]
        /// The rubric body BELOW the frontmatter — what the coder reads.
        public let body: String
    }

    /// Field names are `[A-Za-z][A-Za-z0-9_]*` (computed per call: `Regex`
    /// is not `Sendable`, and a shared static would trip strict
    /// concurrency).
    private static var namePattern: Regex<Substring> {
        /^[A-Za-z][A-Za-z0-9_]*$/
    }

    /// The coding declaration of a rubric, or nil for a paired rubric.
    /// Malformed declarations THROW — a typo'd field line silently skipped
    /// would drop a coding field from every judgment. Grammar identical to
    /// the server's `parse_rubric`.
    public static func parseRubric(_ text: String) throws -> Schema? {
        let lines = text.components(separatedBy: "\n")
        guard let first = lines.first,
            first.trimmingCharacters(in: .whitespaces) == "---"
        else { return nil }
        guard
            let end = (1..<lines.count).first(where: {
                lines[$0].trimmingCharacters(in: .whitespaces) == "---"
            })
        else {
            throw ExperimentError(
                reason: "rubric frontmatter never closes: the opening '---' "
                    + "has no matching '---' line")
        }
        var mode: String?
        var fields: [Field] = []
        for raw in lines[1..<end] {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("mode:") {
                guard mode == nil else {
                    throw ExperimentError(
                        reason: "rubric frontmatter declares 'mode:' twice")
                }
                mode = String(line.dropFirst("mode:".count))
                    .trimmingCharacters(in: .whitespaces)
                continue
            }
            if line.hasPrefix("field:") {
                fields.append(
                    try parseFieldLine(
                        String(line.dropFirst("field:".count))
                            .trimmingCharacters(in: .whitespaces)))
                continue
            }
            throw ExperimentError(
                reason: "unrecognized rubric frontmatter line: '\(line)' — "
                    + "only 'mode:' and 'field:' lines are allowed")
        }
        guard let mode else {
            throw ExperimentError(
                reason: "rubric frontmatter has no 'mode:' line — a "
                    + "frontmatter block must declare its mode "
                    + "(mode: perResponseCoding)")
        }
        guard mode == "perResponseCoding" else {
            throw ExperimentError(
                reason: "unknown rubric mode '\(mode)' — this engine knows "
                    + "'perResponseCoding' (a rubric with no frontmatter is "
                    + "a paired rubric)")
        }
        guard !fields.isEmpty else {
            throw ExperimentError(
                reason: "a perResponseCoding rubric must declare at least "
                    + "one 'field:' line")
        }
        var seen = Set<String>()
        for field in fields {
            guard seen.insert(field.name).inserted else {
                throw ExperimentError(
                    reason: "rubric declares field '\(field.name)' twice")
            }
        }
        let body = lines[(end + 1)...].joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Schema(fields: fields, body: body)
    }

    private static func parseFieldLine(_ spec: String) throws -> Field {
        let tokens = spec.split(separator: " ").map(String.init)
        guard tokens.count >= 2 else {
            throw ExperimentError(
                reason: "malformed field declaration '\(spec)' — expected "
                    + "'field: <name> <type> [optional]'")
        }
        let name = tokens[0]
        guard name.wholeMatch(of: namePattern) != nil else {
            throw ExperimentError(
                reason: "invalid field name '\(name)' — names are "
                    + "[A-Za-z][A-Za-z0-9_]*")
        }
        let typeToken = tokens[1]
        var optional = false
        if tokens.count == 3, tokens[2] == "optional" {
            optional = true
        } else if tokens.count > 2 {
            throw ExperimentError(
                reason: "malformed field declaration '\(spec)' — the only "
                    + "modifier after the type is 'optional'")
        }
        if typeToken.hasPrefix("enum("), typeToken.hasSuffix(")") {
            let rawValues = typeToken.dropFirst("enum(".count).dropLast()
            let values = rawValues.split(
                separator: "|", omittingEmptySubsequences: false
            ).map { $0.trimmingCharacters(in: .whitespaces) }
            guard !values.isEmpty, !values.contains(where: \.isEmpty),
                !rawValues.isEmpty
            else {
                throw ExperimentError(
                    reason: "malformed enum declaration '\(typeToken)' — "
                        + "expected enum(value|value|…) with non-empty values")
            }
            return Field(
                name: name, type: "enum", optional: optional, values: values)
        }
        guard ["boolean", "integer", "number", "string"].contains(typeToken)
        else {
            throw ExperimentError(
                reason: "unknown field type '\(typeToken)' — types are "
                    + "boolean, integer, number, string, or enum(a|b|…)")
        }
        return Field(
            name: name, type: typeToken, optional: optional, values: [])
    }

    /// Refuse paired-only machinery a coding rubric (server twin:
    /// `refuse_if_coding` — keep the wording identical). A coding rubric
    /// records per-response codes and no preference, so a preference-shaped
    /// consumer running it would force the judge to improvise a winner.
    public static func refuseIfCoding(
        _ rubricText: String, context: String, rubricFile: String? = nil
    ) throws {
        guard try parseRubric(rubricText) != nil else { return }
        let label =
            rubricFile.map { "rubric '\($0)'" } ?? "the pinned rubric"
        throw ExperimentError(
            reason: "\(label) declares perResponseCoding — a coding rubric "
                + "records per-response codes and no preference, so it "
                + "cannot drive \(context). Pin a paired-preference rubric "
                + "instead.")
    }

    /// The canonical coding-prompt contract — BYTE-IDENTICAL to the
    /// server's `response_coding.build_prompt`, pinned by the goldens in
    /// `prompts/fixtures/coding-judge/`.
    public static func buildPrompt(
        schema: Schema, response: String, taskPrompt: String?
    ) -> String {
        let trimmedBody = schema.body.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let body = trimmedBody.isEmpty
            ? "Code the declared fields exactly as named." : trimmedBody
        let fieldLines = schema.fields.map(\.describe).joined(separator: "\n")
        let task = (taskPrompt ?? "").trimmingCharacters(
            in: .whitespacesAndNewlines)
        let taskBlock = task.isEmpty
            ? ""
            : "=== Task prompt (the response answered this) ===\n" + task
                + "\n\n"
        return "You are a blinded coder annotating ONE model response to a "
            + "task prompt. Do not infer or guess which experimental "
            + "condition produced the response; code only what the response "
            + "text contains.\n\n"
            + "Coding rubric:\n" + body + "\n\n"
            + "Code these fields:\n" + fieldLines + "\n\n"
            + taskBlock
            + "=== Response ===\n"
            + response.trimmingCharacters(in: .whitespacesAndNewlines)
            + "\n\n"
            + "Return JSON only: {\"codes\": {\"<field>\": <value>, ...}, "
            + "\"brief_reason\": \"at most two sentences\"}. Every listed "
            + "field must appear in \"codes\" with a value of its declared "
            + "type. Use null only for a field marked optional, and only "
            + "when the response text cannot settle it. Never code "
            + "information the response does not contain, and keep "
            + "\"brief_reason\" to at most two sentences — never an essay."
    }

    /// One validated coding verdict.
    public struct Verdict: Sendable, Equatable {
        /// The DECLARED fields only — the measurement. Everything the
        /// aggregates, the agreement statistics, and the analysis read comes
        /// from here, and every key in it is named by the pinned rubric.
        public let codes: [String: JSONValue]
        /// Keys the coder invented that the rubric never declared, kept
        /// verbatim and kept OUT of `codes` (cross-engine key:
        /// `undeclaredCodes`).
        ///
        /// Two rules meet here. Dropping them would be dishonest — a coder
        /// that keeps volunteering a field is telling you something about
        /// the rubric, and silently deleting evidence of it is exactly what
        /// this instrument exists not to do. Leaving them beside the
        /// declared codes was the hazard the review found: a reader (or a
        /// future aggregator) sees one flat `codes` object and cannot tell
        /// the pinned measurement from a judge's improvisation. So they are
        /// persisted, in their own block, marked as not evidence.
        public let undeclaredCodes: [String: JSONValue]
        public let briefReason: String
        /// OpenRouter coders only: the VERIFIED serving provider.
        public var provider: String?
    }

    /// An unparseable/invalid coding response, carrying the raw text (the
    /// diagnostic record the run-status invalid log persists).
    public struct InvalidCodesError: Error, CustomStringConvertible {
        public let reason: String
        public let raw: String
        public var description: String { reason }
    }

    /// Extract and validate one coding verdict against the schema; throws
    /// `InvalidCodesError` on any failure — the retry-once-then-refuse
    /// closure handles it (server twin: `parse_codes`).
    public static func parseCodes(
        _ text: String, schema: Schema
    ) throws -> Verdict {
        struct Raw: Decodable {
            let codes: [String: JSONValue]?
            let briefReason: String?
            enum CodingKeys: String, CodingKey {
                case codes
                case briefReason = "brief_reason"
            }
        }
        guard
            let json = PairedJudgeVerdictParser.firstBalancedJSONObject(
                in: text),
            let raw = try? JSONDecoder().decode(Raw.self, from: Data(json.utf8))
        else {
            throw InvalidCodesError(
                reason: "coding response had no parseable JSON object",
                raw: text)
        }
        guard let codes = raw.codes else {
            throw InvalidCodesError(
                reason: "invalid codes: coding response has no \"codes\" "
                    + "object",
                raw: text)
        }
        let problems = validate(codes: codes, schema: schema)
        guard problems.isEmpty else {
            throw InvalidCodesError(
                reason: "invalid codes: " + problems.joined(separator: "; "),
                raw: text)
        }
        // The split is the contract (server twin: `parse_codes`): `codes`
        // holds exactly the declared fields, and anything else the coder
        // volunteered moves to `undeclaredCodes` — never dropped, never
        // mixed in with the measurement.
        let declared = Set(schema.fields.map(\.name))
        return Verdict(
            codes: codes.filter { declared.contains($0.key) },
            undeclaredCodes: codes.filter { !declared.contains($0.key) },
            briefReason: raw.briefReason ?? "")
    }

    /// Every problem with a parsed codes object (empty = valid). Declared
    /// fields are checked for presence and type; extra keys the judge added
    /// are neither a problem nor a measurement — `parseCodes` moves them to
    /// the verdict's `undeclaredCodes` block.
    static func validate(
        codes: [String: JSONValue], schema: Schema
    ) -> [String] {
        var problems: [String] = []
        for field in schema.fields {
            guard let value = codes[field.name] else {
                problems.append("missing field '\(field.name)'")
                continue
            }
            if let problem = typeProblem(field, value) {
                problems.append(problem)
            }
        }
        return problems
    }

    private static func typeProblem(
        _ field: Field, _ value: JSONValue
    ) -> String? {
        if case .null = value {
            if field.optional { return nil }
            return "required field '\(field.name)' is null"
        }
        switch field.type {
        case "boolean":
            if case .bool = value { return nil }
            return "field '\(field.name)' must be a boolean, got "
                + value.displayString
        case "integer":
            // JSON does not distinguish 3 from 3.0 — an integral number
            // passes on both engines; a boolean never does.
            if case .number(let number) = value,
                number == number.rounded()
            {
                return nil
            }
            return "field '\(field.name)' must be an integer, got "
                + value.displayString
        case "number":
            if case .number = value { return nil }
            return "field '\(field.name)' must be a number, got "
                + value.displayString
        case "string":
            if case .string = value { return nil }
            return "field '\(field.name)' must be a string, got "
                + value.displayString
        case "enum":
            if case .string(let string) = value,
                field.values.contains(string)
            {
                return nil
            }
            return "field '\(field.name)' must be one of "
                + "\(field.values.joined(separator: "|")), got "
                + value.displayString
        default:
            return "field '\(field.name)' has unknown declared type "
                + "'\(field.type)'"
        }
    }

    /// Call the coder and require codes that satisfy the declared schema —
    /// the exact closure rule of `judgmentWithValidWinner` / the server's
    /// `valid_codes`: retry once, then refuse the whole phase naming the
    /// judge, the item, and what came back; never record invented data.
    /// `onInvalid` records every invalid attempt, raw text included.
    ///
    /// The refusal is `JudgeNoncompliantError` (Christian, 2026-08-09): the
    /// coder ANSWERED, so the coding loop records the record as a
    /// `codes: null` row and continues, while a transport failure from
    /// `obtain` propagates unchanged and still fails the session. Server
    /// twin: `response_coding.valid_codes` raising
    /// `paired_judge.JudgeNoncompliant`.
    static func codesWithValidSchema(
        judgeName: String,
        item: String,
        schema: Schema,
        onInvalid: (@Sendable ([String: String]) async -> Void)? = nil,
        _ obtain: () async throws -> (text: String, provider: String?)
    ) async throws -> Verdict {
        var failures: [String] = []
        for attempt in 1...2 {
            let (text, provider) = try await obtain()
            do {
                var verdict = try parseCodes(text, schema: schema)
                verdict.provider = provider
                return verdict
            } catch let error as InvalidCodesError {
                failures.append(error.reason)
                await onInvalid?([
                    "attempt": String(attempt),
                    "error": error.reason,
                    "rawResponse": error.raw,
                    "judge": judgeName,
                    "item": item,
                ])
            }
        }
        throw JudgeNoncompliantError(
            reason: "judge '\(judgeName)' returned invalid codes twice for "
                + "\(item): " + failures.joined(separator: "; then ")
                + " — refusing to record invented data; nothing was "
                + "recorded for this phase — fix the judge or rubric, then "
                + "re-run it")
    }

    /// Whitespace-run word count — computed by the ENGINE, deterministically,
    /// for every coded record (server twin: `word_count`). A rubric asking
    /// the judge to estimate a count the engine knows exactly was one of the
    /// seeded-rubric defects this instrument replaces.
    public static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    /// The agreement label for a coded value (cross-engine): booleans code
    /// as "true"/"false", null as "null", strings as themselves, numbers by
    /// their display form.
    static func categoricalLabel(_ value: JSONValue?) -> String {
        switch value {
        case .none, .some(.null): return "null"
        case .some(.bool(let bool)): return bool ? "true" : "false"
        case .some(.string(let string)): return string
        case .some(let other): return other.displayString
        }
    }
}

/// One coding row's live-progress preview (the coding instrument's sibling
/// of `StudyJudgePreview`).
public struct StudyCodingPreview: Identifiable, Codable, Sendable, Equatable {
    public var id: String { "\(judge)-\(condition)-\(promptID)-\(sampleIndex)" }
    public let judge: String
    public let condition: String
    public let sampleIndex: UInt64
    public let promptID: String
    /// Engine-computed (never judge-estimated) whitespace-run word count.
    public let wordCount: Int
    public let codes: [String: JSONValue]
    public let briefReason: String
}
