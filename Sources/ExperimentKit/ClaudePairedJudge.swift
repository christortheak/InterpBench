import Foundation

public struct PairedJudgeError: Error, CustomStringConvertible {
    public let reason: String
    public var description: String { reason }
}

/// The judge ANSWERED, twice, and neither answer parsed to a verdict (or to
/// a valid code set).
///
/// Distinct from transport failure ON PURPOSE (Christian, 2026-08-09): a
/// judge that answers garbage for one item is a per-item, classifiable
/// outcome the evaluate loops RECORD as a row and continue past; a judge
/// whose CALL fails (HTTP error, network, credential) still fails the whole
/// session, because the resume machinery exists to complete exactly those
/// without re-paying finished judgments — and because recording a
/// rate-limited call as "noncompliance" would misclassify a healthy judge.
/// Callers must therefore catch THIS type, never a generic error.
///
/// Server twin: `paired_judge.JudgeNoncompliant`.
public struct JudgeNoncompliantError: Error, CustomStringConvertible {
    public let reason: String
    public var description: String { reason }

    public init(reason: String) { self.reason = reason }
}

/// How much judge noncompliance a run tolerates before it is systemic
/// failure rather than flakiness.
///
/// Christian, 2026-08-09: a few invalid verdicts become recorded rows and
/// the run completes; a judge failing more than this fraction of its column
/// fails the evaluation, because a report built mostly on holes is not a
/// result. Shared by the paired-judge evaluate loop and the per-response
/// coding loop. Server twin: `paired_judge.NONCOMPLIANCE_CAP` — keep the
/// two values identical.
public enum JudgeNoncompliance {
    public static let cap = 0.25

    /// The cap as the refusal copy states it ("25%"), so both engines'
    /// messages read the same.
    static var capPercentText: String {
        "\(Int((cap * 100).rounded()))%"
    }

    /// The reason string a recorded row keeps, bounded the way the server
    /// bounds it (`str(exc)[:2000]`) — a raw judge dump is diagnostic, not
    /// an artifact budget.
    static func recordedReason(_ error: some Error) -> String {
        String("\(error)".prefix(2000))
    }
}

public struct PairedJudgeResponse: Codable, Sendable, Equatable {
    public let aScores: [String: Int]?
    public let bScores: [String: Int]?
    public let structuredFields: [String: JSONValue]?
    public let winner: String
    public let confidence: Double
    public let briefReason: String
    /// Truncation-salvage marker (2026-07-22 incident, cross-engine key
    /// `reasoningTruncated` with the server's `paired_judge`): true when the
    /// judge's JSON never closed — the winner was legibly complete but the
    /// reasoning commentary outran the generation cap, so `briefReason` is
    /// only the salvageable prefix (possibly empty). Absent (nil) on a
    /// fully-parsed verdict.
    public let reasoningTruncated: Bool?
    /// OpenRouter judges only: the canonical slug of the provider that
    /// actually SERVED this verdict, read off the response and verified
    /// against the pin before the verdict was accepted (2026-07-24).
    ///
    /// Set by the transport, not the parser, which is why it is a `var` —
    /// the parser sees only the model's text. Nil for every other judge
    /// kind, which have no provider dimension. Cross-engine twin: the
    /// server's verdict carries `provider` in the same position.
    public var provider: String?
    /// What this judgment cost the judge, when the transport could read it
    /// off the response (OpenRouter today). Set by the TRANSPORT, not the
    /// parser — which is why it is a `var` — exactly like `provider`.
    /// Cross-engine key `usage` with the server's verdict dict. Recorded
    /// provenance only: no code path reads it to gate anything.
    public var usage: PairedJudgeUsage?
    /// How the CALL finished, as the provider reported it (`stop`,
    /// `length`, a provider's own spelling…). Stamped on every judgment
    /// record, not only failures (2026-08-06): a clean verdict whose
    /// finishReason is "length" is a salvage candidate a report reader must
    /// be able to spot without re-running the evaluation. Set by the
    /// transport, like `provider`; nil for judges whose transport reports
    /// none. Cross-engine key `finishReason` with the server's verdict.
    public var finishReason: String?
    /// Whether the ENDPOINT accepted the JSON output constraint:
    /// "json_object" when the request carried `response_format`,
    /// "unsupported" when the endpoint rejected it and the call fell back
    /// unconstrained. A verdict that still arrives as prose under
    /// "json_object" is a different diagnosis from one the endpoint never
    /// promised to constrain. Cross-engine key `responseFormat`.
    public var responseFormat: String?

    enum CodingKeys: String, CodingKey {
        case aScores = "a_scores"
        case bScores = "b_scores"
        case structuredFields = "structured_fields"
        case winner
        case confidence
        case briefReason = "brief_reason"
        case reasoningTruncated
        case provider
        case usage
        case finishReason
        case responseFormat
    }

    public init(
        aScores: [String: Int]? = nil,
        bScores: [String: Int]? = nil,
        structuredFields: [String: JSONValue]? = nil,
        winner: String,
        confidence: Double,
        briefReason: String,
        reasoningTruncated: Bool? = nil,
        provider: String? = nil,
        usage: PairedJudgeUsage? = nil,
        finishReason: String? = nil,
        responseFormat: String? = nil
    ) {
        self.aScores = aScores
        self.bScores = bScores
        self.structuredFields = structuredFields
        self.winner = winner
        self.confidence = confidence
        self.briefReason = briefReason
        self.reasoningTruncated = reasoningTruncated
        self.provider = provider
        self.usage = usage
        self.finishReason = finishReason
        self.responseFormat = responseFormat
    }

    /// Decoded leniently for `provider`: verdict payloads written before
    /// 2026-07-24 carry no such key, and a synthesized decoder would reject
    /// them outright rather than reading them as "no provider recorded".
    /// Same for `usage` (2026-08-06).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        aScores = try c.decodeIfPresent([String: Int].self, forKey: .aScores)
        bScores = try c.decodeIfPresent([String: Int].self, forKey: .bScores)
        structuredFields = try c.decodeIfPresent(
            [String: JSONValue].self, forKey: .structuredFields)
        winner = try c.decode(String.self, forKey: .winner)
        confidence = try c.decode(Double.self, forKey: .confidence)
        briefReason = try c.decode(String.self, forKey: .briefReason)
        reasoningTruncated = try c.decodeIfPresent(
            Bool.self, forKey: .reasoningTruncated)
        provider = try c.decodeIfPresent(String.self, forKey: .provider)
        usage = try c.decodeIfPresent(PairedJudgeUsage.self, forKey: .usage)
        finishReason = try c.decodeIfPresent(String.self, forKey: .finishReason)
        responseFormat = try c.decodeIfPresent(
            String.self, forKey: .responseFormat)
    }
}

/// The ONE generation cap for judge verdicts, on every judging path of this
/// engine (Claude, OpenRouter, and the local-model judge).
///
/// The cap exists to bound RUNAWAY generation — a judge stuck in a loop must
/// not bill or block forever. It must never RATION reasoning: a judge that
/// thinks harder than we expected is doing its job well, and a run must
/// never stop because a judge spent tokens thinking (researcher directive
/// 2026-08-06).
///
/// 1024 was the 2026-07-22 value (a winner + confidence + two-sentence
/// reason needs no more, and the server's local judge at 512 had truncated a
/// legible verdict on the cluster). Reasoning-first models broke that
/// arithmetic twice, because hidden reasoning is billed against the SAME cap
/// as the visible answer:
/// - 2026-08-05: `deepseek/deepseek-v4-flash-0731` returned HTTP 200 with
///   EMPTY content ("carried no content") — the whole 1024 went to
///   reasoning;
/// - 2026-08-06: `google/gemini-3.6-flash` @ google-ai-studio returned
///   ~170-200-character mid-sentence prose fragments with no JSON, and the
///   run refused after its retry (workspace run
///   20260805T232636162-exp-replication-1-evaluate, raw fragments in
///   judge-failures.jsonl).
/// Both judges were doing nothing wrong; our budget truncated good analysis.
/// 8192 accommodates them, and the OpenRouter transport escalates further
/// (`OpenRouterPairedJudge.escalationLimit`) when a provider still reports a
/// length cut.
///
/// Server twin: `paired_judge.JUDGE_MAX_TOKENS`
/// (Server/steerlab_server/experiment/paired_judge.py) — keep the two
/// values identical.
public enum PairedJudgeBudget {
    public static let maxTokens = 8192
}

/// What one judgment cost the judge, as the provider reported it.
///
/// REPORTED, NEVER GATED (researcher directive 2026-08-06): nothing in
/// either engine reads these numbers to refuse, throttle, or shrink a
/// judgment — they exist so the researcher can SEE that a judge is spending
/// more than expected on hidden reasoning. Cross-engine JSON keys with the
/// server's verdict `usage` block and judge-report token sums.
public struct PairedJudgeUsage: Codable, Sendable, Equatable {
    /// Total completion tokens billed for the response. Providers differ on
    /// whether this already includes `reasoningTokens`, so the two are
    /// recorded as reported, never differenced.
    public var completionTokens: Int?
    /// Tokens the model spent on hidden reasoning, when the provider
    /// reports them (`completion_tokens_details.reasoning_tokens`, or a
    /// top-level `reasoning_tokens`).
    public var reasoningTokens: Int?

    public init(completionTokens: Int? = nil, reasoningTokens: Int? = nil) {
        self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens
    }

    /// Whether the provider reported anything at all worth recording.
    public var isEmpty: Bool { completionTokens == nil && reasoningTokens == nil }
}

public enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var displayString: String {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return String(format: "%.4g", value)
        case .bool(let value):
            return value ? "true" : "false"
        case .object, .array:
            let data = (try? JSONEncoder().encode(self)) ?? Data()
            return String(decoding: data, as: UTF8.self)
        case .null:
            return "null"
        }
    }
}

public enum ClaudePairedJudge {
    public static let defaultModel = "claude-opus-4-8"
    private static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!

    private struct APIResponse: Decodable {
        struct ContentBlock: Decodable {
            let type: String
            let text: String?
        }
        let content: [ContentBlock]
    }

    private struct APIError: Decodable {
        struct Detail: Decodable {
            let type: String
            let message: String
        }
        let error: Detail
    }

    private static var responseSchema: [String: Any] { [
        "type": "object",
        "properties": [
            "a_scores": [
                "type": "object",
                "additionalProperties": ["type": "integer"],
            ],
            "b_scores": [
                "type": "object",
                "additionalProperties": ["type": "integer"],
            ],
            "structured_fields": [
                "type": "object",
                "additionalProperties": true,
            ],
            "winner": ["type": "string", "enum": ["A", "B", "tie"]],
            "confidence": ["type": "number", "minimum": 0, "maximum": 1],
            "brief_reason": ["type": "string"],
        ],
        "required": ["winner", "confidence", "brief_reason"],
        "additionalProperties": false,
    ] }

    /// The request body, extracted so tests can assert the unified judge
    /// generation cap (`PairedJudgeBudget.maxTokens`) is what this client
    /// actually sends.
    static func requestBody(
        model: String,
        rubric: String,
        structuredPrompt: String?,
        prompt: String,
        responseA: String,
        responseB: String
    ) -> [String: Any] {
        [
            "model": apiModelName(model),
            "max_tokens": PairedJudgeBudget.maxTokens,
            "output_config": ["format": ["type": "json_schema", "schema": responseSchema]],
            "messages": [
                [
                    "role": "user",
                    "content": PairedJudgePrompt.build(
                        rubric: rubric, structuredPrompt: structuredPrompt,
                        prompt: prompt,
                        responseA: responseA, responseB: responseB),
                ]
            ],
        ]
    }

    public static func judge(
        model: String,
        rubric: String,
        structuredPrompt: String?,
        prompt: String,
        responseA: String,
        responseB: String
    ) async throws -> PairedJudgeResponse {
        let body = requestBody(
            model: model, rubric: rubric, structuredPrompt: structuredPrompt,
            prompt: prompt, responseA: responseA, responseB: responseB)
        let text = try await completeText(body: body)
        // The shared verdict parser: a structured-output response is normally
        // complete JSON, but a max-tokens truncation can still cut it mid-
        // reasoning — the salvage rule accepts a legibly-complete winner
        // (2026-07-22 incident closure).
        return try PairedJudgeVerdictParser.parse(text)
    }

    /// One raw Claude completion under the unified judge cap — the
    /// transport shared by the paired judge and the per-response coding
    /// instrument (2026-08-04; server twin: `paired_judge.claude_complete`).
    /// `responseSchema` constrains the output when the caller has one.
    static func complete(
        model: String,
        prompt: String,
        responseSchema: [String: Any]? = nil
    ) async throws -> String {
        var body: [String: Any] = [
            "model": apiModelName(model),
            "max_tokens": PairedJudgeBudget.maxTokens,
            "messages": [["role": "user", "content": prompt]],
        ]
        if let responseSchema {
            body["output_config"] = [
                "format": ["type": "json_schema", "schema": responseSchema]
            ]
        }
        return try await completeText(body: body)
    }

    private static func completeText(body: [String: Any]) async throws -> String {
        guard let apiKey = ClaudeStimulusGenerator.apiKey else {
            throw PairedJudgeError(
                reason: "no API key — set ANTHROPIC_API_KEY or save a key in "
                    + "the Compute section (stored in the macOS Keychain)")
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 300

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw PairedJudgeError(reason: "no HTTP response")
        }
        guard http.statusCode == 200 else {
            if let apiError = try? JSONDecoder().decode(APIError.self, from: data) {
                throw PairedJudgeError(
                    reason: "\(apiError.error.type): \(apiError.error.message)")
            }
            throw PairedJudgeError(reason: "HTTP \(http.statusCode)")
        }
        let decoded = try JSONDecoder().decode(APIResponse.self, from: data)
        guard let text = decoded.content.first(where: { $0.type == "text" })?.text else {
            throw PairedJudgeError(reason: "response did not contain a JSON text block")
        }
        return text
    }

    /// The structured-output schema for a per-response coding call
    /// (2026-08-04): a `codes` object plus the brief reason. The engine's
    /// own `ResponseCoding.validate` enforces the declared field types —
    /// the API schema only guarantees the envelope shape.
    static var codingResponseSchema: [String: Any] { [
        "type": "object",
        "properties": [
            "codes": ["type": "object", "additionalProperties": true],
            "brief_reason": ["type": "string"],
        ],
        "required": ["codes", "brief_reason"],
        "additionalProperties": false,
    ] }

    public static func isClaudeModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.isEmpty
            || normalized.hasPrefix("claude-")
            || normalized.hasPrefix("anthropic:")
    }

    private static func apiModelName(_ model: String) -> String {
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("anthropic:") {
            return String(trimmed.dropFirst("anthropic:".count))
        }
        return trimmed.isEmpty ? defaultModel : trimmed
    }
}

/// The canonical judge-prompt wrapper (wrapper unification 2026-07-22):
/// BYTE-IDENTICAL to the server's `paired_judge.build_prompt`
/// (Server/steerlab_server/experiment/paired_judge.py) — both are pinned by
/// the committed goldens in `prompts/fixtures/paired-judge/` and each
/// engine's golden test fails loudly on drift. It explicitly demands a
/// BRIEF reason: the 2026-07-22 incident judge stated a valid winner and
/// then wrote an essay-length reasoning that outran the generation cap, so
/// the JSON never closed. Change the wrapper only deliberately, on both
/// engines at once, regenerating the goldens.
enum PairedJudgePrompt {
    static func build(
        rubric: String,
        structuredPrompt: String? = nil,
        prompt: String,
        responseA: String,
        responseB: String
    ) -> String {
        let trimmedRubric = rubric.trimmingCharacters(in: .whitespacesAndNewlines)
        let rubricText = trimmedRubric.isEmpty
            ? "Judge which response is higher quality and more appropriate."
            : trimmedRubric
        let structured = structuredPrompt?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let structuredBlock = structured.isEmpty
            ? ""
            : "Structured comparison fields:\n" + structured + "\n\n"
                + "Fill \"structured_fields\" with the requested conclusions. Use stable "
                + "machine-readable values: booleans for yes/no findings, numbers for "
                + "scales or deltas, and short strings for categories. If a field "
                + "cannot be determined, use null rather than guessing. For "
                + "directional per-response scores, name fields with A/B prefixes, "
                + "such as \"a_severity\" and \"b_severity\".\n\n"
        let task = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let taskBlock = task.isEmpty
            ? ""
            : "=== Task prompt (both responses answered this) ===\n" + task + "\n\n"
        return "You are a blinded evaluator comparing two model responses, A and "
            + "B, to the same task prompt. Do not infer which response is a "
            + "baseline or which was manipulated.\n\n"
            + "Evaluation rubric:\n" + rubricText + "\n\n"
            + structuredBlock + taskBlock
            + "=== Response A ===\n"
            + responseA.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            + "=== Response B ===\n"
            + responseB.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n"
            + "Ignore style, length, and eloquence unless the rubric explicitly "
            + "makes them relevant. Prefer \"tie\" when the evidence is genuinely "
            + "indistinguishable.\n\n"
            + "Return JSON only, stating the verdict fields first: "
            + "{\"winner\": \"A\"|\"B\"|\"tie\", \"confidence\": 0..1, \"brief_reason\": "
            + "\"at most two sentences\", \"a_scores\": {dim: 1-7}, \"b_scores\": "
            + "{dim: 1-7}}. Keep \"brief_reason\" to at most two sentences — never "
            + "an essay. If the rubric names scalar dimensions, use them for the "
            + "1-7 scores; use null if a dimension does not apply. If structured "
            + "comparison fields were requested, also include \"structured_fields\"."
    }
}

/// The ONE verdict parser for every Swift judging path (Claude, OpenRouter,
/// local model) — the cross-engine twin of the server's
/// `paired_judge.parse_response`, salvage rule included (2026-07-22
/// incident closure).
enum PairedJudgeVerdictParser {

    /// Lenient verdict shape for balanced-JSON responses that don't satisfy
    /// the strict `PairedJudgeResponse` contract (a judge model without
    /// structured output may answer "reasoning" instead of "brief_reason",
    /// or omit confidence).
    private struct LenientVerdict: Decodable {
        let winner: String
        let confidence: Double?
        let briefReason: String?
        let reasoning: String?
        let aScores: [String: Int]?
        let bScores: [String: Int]?

        enum CodingKeys: String, CodingKey {
            case winner, confidence, reasoning
            case briefReason = "brief_reason"
            case aScores = "a_scores"
            case bScores = "b_scores"
        }
    }

    /// Extract the judge's verdict from a (possibly fenced) response.
    ///
    /// Truncation salvage (2026-07-22 incident, identical rule to the
    /// server): when the braces never balance — the judge stated its
    /// verdict fields and then a too-long reasoning hit the generation cap,
    /// so the object never closed — a COMPLETE `"winner"` field whose value
    /// is exactly A/B/tie is still a legible, explicitly-written verdict,
    /// not invented data. Such a verdict is accepted with
    /// `reasoningTruncated: true` (plus the complete `confidence` number
    /// when present and whatever reasoning prefix was written). If the
    /// winner field itself is absent, incomplete, or out-of-vocabulary, the
    /// refusal path is unchanged.
    ///
    /// A leading `<think>` block is stripped first (see
    /// `strippingLeadingThinkBlock`) — some providers inline the model's
    /// reasoning in the visible content.
    static func parse(_ raw: String) throws -> PairedJudgeResponse {
        let text = strippingLeadingThinkBlock(raw)
        guard text.contains("{") else {
            throw PairedJudgeError(reason: "judge response had no JSON object")
        }
        if let json = firstBalancedJSONObject(in: text) {
            let data = Data(json.utf8)
            if let full = try? JSONDecoder().decode(
                PairedJudgeResponse.self, from: data)
            {
                return full
            }
            let lenient: LenientVerdict
            do {
                lenient = try JSONDecoder().decode(LenientVerdict.self, from: data)
            } catch {
                throw PairedJudgeError(
                    reason: "judge verdict JSON did not decode: \(error)")
            }
            return PairedJudgeResponse(
                aScores: lenient.aScores,
                bScores: lenient.bScores,
                structuredFields: nil,
                winner: lenient.winner,
                confidence: lenient.confidence ?? 0,
                briefReason: lenient.briefReason ?? lenient.reasoning ?? "")
        }
        // Salvage only within the CANDIDATE object (first "{" onward) —
        // prose before the JSON must not contribute verdict fields.
        if let start = text.firstIndex(of: "{"),
            let salvaged = salvageTruncatedVerdict(in: String(text[start...]))
        {
            return salvaged
        }
        throw PairedJudgeError(reason: "unbalanced JSON in judge response")
    }

    /// Drop a leading `<think>…</think>` reasoning preamble.
    ///
    /// Some providers INLINE the model's reasoning in the visible content
    /// instead of delivering it out of band (2026-08-06 round). That
    /// preamble can contain braces, so it has to come off before JSON
    /// extraction or the scan walks a reasoning aside looking for a verdict.
    ///
    /// An UNCLOSED leading `<think>` (the response was cut before the
    /// closing tag, or the provider simply omits it) is treated the same
    /// way: everything after the opening tag is the candidate text. Only a
    /// LEADING block is stripped — a `<think>` after the verdict is content.
    /// Server twin: `paired_judge.strip_leading_think_block`.
    static func strippingLeadingThinkBlock(_ text: String) -> String {
        let leading = text.drop(while: { $0.isWhitespace })
        guard leading.lowercased().hasPrefix("<think"),
            let tagEnd = leading.firstIndex(of: ">")
        else { return text }
        let rest = leading[leading.index(after: tagEnd)...]
        if let close = rest.range(of: "</think>", options: [.caseInsensitive]) {
            return String(rest[close.upperBound...])
        }
        return String(rest)
    }

    /// The first `{…}` with balanced braces — brace-depth scan, ignoring
    /// braces inside JSON strings.
    static func firstBalancedJSONObject(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let character = text[index]
            if escaped {
                escaped = false
            } else if inString {
                if character == "\\" { escaped = true }
                if character == "\"" { inString = false }
            } else {
                switch character {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        return String(text[start...index])
                    }
                default: break
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    /// The truncated-verdict salvage rule (identical on both engines): a
    /// complete A/B/tie winner is accepted; the verdict is stamped
    /// `reasoningTruncated: true` (cross-engine key); confidence rides
    /// along only when its number is complete (digits followed by a
    /// delimiter — a number cut mid-digits is dropped, decoding as 0 here);
    /// the reasoning text is whatever complete prefix was written (possibly
    /// empty; the pre-unification "reasoning" spelling salvages too).
    /// Returns nil when there is nothing legible to salvage — including
    /// when MORE than one complete winner field exists (finding 5,
    /// 2026-07-23, identical rule to the server): conflicting or
    /// duplicated winners in a truncated response have no single legible
    /// verdict, and picking the first would parse arbitrary data.
    static func salvageTruncatedVerdict(in text: String) -> PairedJudgeResponse? {
        let winnerMatches = text.matches(of: /"winner"\s*:\s*"(A|B|tie)"/)
        guard winnerMatches.count == 1, let winnerMatch = winnerMatches.first
        else { return nil }
        var confidence = 0.0
        if let confidenceMatch = text.firstMatch(
            of: /"confidence"\s*:\s*(\d+(?:\.\d+)?)(?=[\s,}])/),
            let value = Double(confidenceMatch.output.1)
        {
            confidence = value
        }
        var reason = ""
        if let reasonMatch = text.firstMatch(
            of: /"(?:brief_reason|reasoning)"\s*:\s*"((?:[^"\\]|\\.)*)/)
        {
            reason = unescapeStringPrefix(String(reasonMatch.output.1))
        }
        return PairedJudgeResponse(
            winner: String(winnerMatch.output.1),
            confidence: confidence,
            briefReason: reason,
            reasoningTruncated: true)
    }

    /// Minimal JSON-string unescape for a possibly-truncated prefix.
    /// Identical logic to the server's `_unescape_string_prefix`: known
    /// escapes map, unknown escapes keep the escaped character, a trailing
    /// lone backslash (truncated mid-escape) is dropped.
    static func unescapeStringPrefix(_ raw: String) -> String {
        var out = ""
        var iterator = raw.makeIterator()
        while let character = iterator.next() {
            guard character == "\\" else {
                out.append(character)
                continue
            }
            guard let next = iterator.next() else { break }
            switch next {
            case "n": out.append("\n")
            case "t": out.append("\t")
            case "r": out.append("\r")
            default: out.append(next)
            }
        }
        return out
    }
}
