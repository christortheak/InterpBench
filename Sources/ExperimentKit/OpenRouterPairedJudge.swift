import Foundation

/// OpenRouter-backed paired judge (2026-07-19), sibling to
/// `ClaudePairedJudge` for the Mac's DEFERRED judging path. OpenRouter is a
/// router: the same model slug can be served by different backends running
/// different quantizations with different outputs — so every call pins its
/// provider (`allow_fallbacks: false`) and refuses a response served
/// off-pin. The prompt is `PairedJudgePrompt.build` — the same information
/// set every Mac judging path sends.
public enum OpenRouterPairedJudge {

    static let endpoint = URL(
        string: "https://openrouter.ai/api/v1/chat/completions")!

    /// How many times ONE call may DOUBLE its generation cap when the
    /// provider reports a length cut: 8192 → 16384 → 32768. Escalation is
    /// per-call and stateless — no learned cap is persisted, because the
    /// next judgment may be a short one and a remembered cap would be an
    /// unaudited change to the measurement. Server twin:
    /// `paired_judge.JUDGE_MAX_ESCALATIONS`.
    public static let escalationLimit = 2

    /// The HTTP seam, so the escalation loop is testable without a network.
    public typealias Transport =
        @Sendable (URLRequest) async throws -> (Data, URLResponse)

    public static let liveTransport: Transport = {
        try await URLSession.shared.data(for: $0)
    }

    /// Test seam for the credential lookup (same convention as
    /// `ExperimentTasks.judgeOverrideForTesting`): the escalation loop is
    /// exercised through `transport`, and a test runner must never be sent
    /// to the Keychain — nor may it depend on a process-wide
    /// `OPENROUTER_API_KEY`.
    nonisolated(unsafe) static var keyOverrideForTesting: String?

    /// One completion's result: the text plus what the transport learned
    /// about how the response ENDED — provenance and diagnosis material,
    /// never a gate.
    struct Completion {
        let text: String
        /// The VERIFIED canonical serving provider.
        let provider: String
        var usage: PairedJudgeUsage?
        /// The provider's finish reason for the LAST attempt.
        var finishReason: String?
        /// Whether that finish reason says the cap cut the visible answer.
        var truncated: Bool
        /// The cap actually sent on the last attempt (after escalation).
        var capReached: Int
        /// Whether the endpoint accepted the JSON output constraint:
        /// `constrainedFormat` when the accepted request carried
        /// `response_format`, `unsupportedFormat` after the fallback.
        var responseFormat: String
    }

    /// `responseFormat` provenance values (cross-engine strings with the
    /// server's `usage_out["responseFormat"]`).
    static let constrainedFormat = "json_object"
    static let unsupportedFormat = "unsupported"

    /// HTTP statuses that mean "this endpoint will not take
    /// `response_format`" under a no-fallback provider pin: 400 invalid
    /// parameter, 404 no matching endpoint, 422 unprocessable. Auth (401),
    /// billing (402/403), rate (429) and server (5xx) errors are NOT about
    /// the constraint — dropping it there would just re-send a doomed
    /// request (and double-bill a rate-limited one). Server twin: the same
    /// triple in `paired_judge.openrouter_complete`.
    static let constraintRejectionStatuses: Set<Int> = [400, 404, 422]

    /// finish_reason spellings that mean "the generation cap cut this off".
    /// Providers disagree (OpenAI-style `length`, Google `MAX_TOKENS`,
    /// others `max_tokens`/`model_length`), and OpenRouter surfaces its own
    /// normalization in `finish_reason` with the provider's raw value in
    /// `native_finish_reason` — both are checked, because either one alone
    /// has been the only signal in practice. Server twin:
    /// `paired_judge.is_length_finish`.
    static let lengthFinishReasons: Set<String> = [
        "length", "max_tokens", "maxtokens", "max_output_tokens",
        "model_length", "token_limit", "max_completion_tokens",
    ]

    static func isLengthFinish(_ reason: String?) -> Bool {
        guard let reason else { return false }
        let normalized = reason
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        return lengthFinishReasons.contains(normalized)
    }

    private struct APIResponse: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable { let content: String? }
            let message: Message?
            let finishReason: String?
            let nativeFinishReason: String?

            enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
                case nativeFinishReason = "native_finish_reason"
            }

            /// Whether EITHER finish-reason field says the cap truncated it.
            var wasTruncated: Bool {
                OpenRouterPairedJudge.isLengthFinish(finishReason)
                    || OpenRouterPairedJudge.isLengthFinish(nativeFinishReason)
            }
        }

        /// Read defensively: every field is optional, and providers put
        /// reasoning tokens in either `completion_tokens_details` or a
        /// top-level `reasoning_tokens`.
        struct Usage: Decodable {
            struct Details: Decodable {
                let reasoningTokens: Int?
                enum CodingKeys: String, CodingKey {
                    case reasoningTokens = "reasoning_tokens"
                }
            }
            let completionTokens: Int?
            let reasoningTokens: Int?
            let completionTokensDetails: Details?

            enum CodingKeys: String, CodingKey {
                case completionTokens = "completion_tokens"
                case reasoningTokens = "reasoning_tokens"
                case completionTokensDetails = "completion_tokens_details"
            }

            var judgeUsage: PairedJudgeUsage {
                PairedJudgeUsage(
                    completionTokens: completionTokens,
                    reasoningTokens: completionTokensDetails?.reasoningTokens
                        ?? reasoningTokens)
            }
        }

        let choices: [Choice]?
        let provider: String?
        let usage: Usage?
    }

    /// The request body, extracted so tests can assert the unified judge
    /// generation cap (`PairedJudgeBudget.maxTokens`) — and the escalated
    /// caps — are what this client actually sends.
    static func requestBody(
        model: String,
        provider: String,
        rubric: String,
        structuredPrompt: String?,
        prompt: String,
        responseA: String,
        responseB: String,
        maxTokens: Int = PairedJudgeBudget.maxTokens,
        constrained: Bool = true
    ) -> [String: Any] {
        requestBody(
            model: model, provider: provider,
            prompt: PairedJudgePrompt.build(
                rubric: rubric, structuredPrompt: structuredPrompt,
                prompt: prompt,
                responseA: responseA, responseB: responseB),
            maxTokens: maxTokens, constrained: constrained)
    }

    /// The raw-prompt request body — the one place the wire shape is built.
    static func requestBody(
        model: String,
        provider: String,
        prompt: String,
        maxTokens: Int = PairedJudgeBudget.maxTokens,
        constrained: Bool = true
    ) -> [String: Any] {
        let providerSlug = OpenRouterProviderIdentity.canonical(provider)
        var body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            // Exclude only skips DELIVERING the reasoning text; it does NOT
            // limit thinking. There are deliberately no `effort`/`max_tokens`
            // reasoning knobs here: we welcome smarter models and do not
            // ration how hard a judge thinks (researcher directive
            // 2026-08-06). Not delivering the reasoning keeps the visible
            // answer — the JSON verdict — from sharing the response body
            // with an essay we never read.
            "reasoning": ["exclude": true],
            "provider": [
                "order": [providerSlug], "allow_fallbacks": false,
            ],
            "messages": [["role": "user", "content": prompt]],
        ]
        // Ask the ENDPOINT to constrain output to a JSON object (2026-08-06,
        // an evaluate post-mortem): every caller of this transport
        // demands JSON ("Return JSON only" is in both prompt contracts), and
        // 22 of 36 judgments on run 20260805T232636162 arrived as partial
        // JSON rescued by regex salvage. `json_object` is the portable
        // spelling; `json_schema` is deliberately NOT sent — the verdict's
        // shape is rubric-dependent (structured_fields is an open object the
        // wrapper leaves to the judge), and pinning a schema here would be a
        // second, drifting copy of the prompt contract. The caller falls back
        // unconstrained when the pinned endpoint rejects the parameter; the
        // parser and truncation salvage behind it are unchanged, so an
        // unconstrained judge is exactly as safe as before.
        if constrained {
            body["response_format"] = ["type": constrainedFormat]
        }
        return body
    }

    public static func judge(
        model: String,
        provider: String,
        rubric: String,
        structuredPrompt: String?,
        prompt: String,
        responseA: String,
        responseB: String,
        transport: Transport = liveTransport
    ) async throws -> PairedJudgeResponse {
        let completion = try await complete(
            model: model, provider: provider,
            prompt: PairedJudgePrompt.build(
                rubric: rubric, structuredPrompt: structuredPrompt,
                prompt: prompt, responseA: responseA, responseB: responseB),
            transport: transport)
        // The shared verdict parser (cross-engine twin of the server's
        // `paired_judge.parse_response`): balanced-JSON extraction with
        // lenient field mapping, plus the 2026-07-22 truncation salvage —
        // a legibly-complete winner whose reasoning outran the cap is
        // accepted with `reasoningTruncated`.
        var verdict: PairedJudgeResponse
        do {
            verdict = try PairedJudgeVerdictParser.parse(completion.text)
        } catch let error as PairedJudgeError {
            throw truncationDiagnosis(error, model: model, completion: completion)
        }
        // Carry the VERIFIED serving provider out with the verdict
        // (2026-07-24): a recorded string is provenance only if it is the
        // verified one, never the manifest's raw pin.
        verdict.provider = completion.provider
        // Token provenance (2026-08-06), stamped by the transport exactly
        // like the provider. Nothing reads it to gate.
        if let usage = completion.usage, !usage.isEmpty {
            verdict.usage = usage
        }
        // How the call FINISHED, and under which output contract, on EVERY
        // judgment record — not only failures (2026-08-06, server twin
        // `openrouter_judge_pair`). A clean verdict with finishReason
        // "length" is a salvage candidate a report reader must be able to
        // spot without re-running; a prose answer under "json_object" is a
        // different diagnosis from one served by an endpoint that never
        // promised to constrain. Recorded provenance — nothing gates on it.
        verdict.finishReason = completion.finishReason
        verdict.responseFormat = completion.responseFormat
        return verdict
    }

    /// Re-word an unparseable-response refusal when the evidence says the
    /// judge was TRUNCATED rather than incoherent (2026-08-06).
    ///
    /// "judge response had no JSON object" sent the researcher looking for a
    /// broken rubric when the real mechanism was a reasoning-first model
    /// spending the whole cap on hidden reasoning. Still a
    /// `PairedJudgeError`, so `judgmentWithValidWinner`'s
    /// retry-once-then-refuse semantics and the raw-response failure log are
    /// untouched — only the wording changes. Server twin:
    /// `paired_judge.truncation_diagnosis`.
    static func truncationDiagnosis(
        _ error: PairedJudgeError,
        model: String,
        completion: Completion
    ) -> PairedJudgeError {
        let spentReasoning = (completion.usage?.reasoningTokens ?? 0) > 0
        guard completion.truncated || spentReasoning else { return error }
        return PairedJudgeError(
            reason: "judge '\(model)' returned no parseable verdict: the "
                + "judge's visible answer was truncated after "
                + "\(completion.capReached) tokens despite escalation "
                + "(reasoning models spend the cap on hidden reasoning) — "
                + "the raw text is retained in judge-failures.jsonl")
    }

    /// One raw provider-pinned OpenRouter completion — the transport shared
    /// by the paired judge and the per-response coding instrument
    /// (2026-08-04; server twin: `paired_judge.openrouter_complete`).
    ///
    /// Truncation escalation (2026-08-06): a response whose finish_reason
    /// says "length" is retried with the cap DOUBLED, up to
    /// `escalationLimit` times. A reasoning-first model spends the cap on
    /// hidden reasoning, so the visible answer arrives truncated or empty
    /// through no fault of the judge; the fix is to give it room, never to
    /// refuse it for its token appetite.
    static func complete(
        model: String,
        provider: String,
        prompt: String,
        transport: Transport = liveTransport
    ) async throws -> Completion {
        guard let apiKey = keyOverrideForTesting
            ?? JudgeKeyStore.resolveKey(kind: "openrouter")
        else {
            throw PairedJudgeError(
                reason: "no OpenRouter key — save one in the Compute "
                    + "section (External judge key, stored in the macOS "
                    + "Keychain) or set OPENROUTER_API_KEY")
        }
        let trimmedProvider = provider.trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard !trimmedProvider.isEmpty else {
            throw PairedJudgeError(
                reason: "openrouter judge for '\(model)' has no pinned "
                    + "provider — refusing an unpinned judgment")
        }

        var cap = PairedJudgeBudget.maxTokens
        var escalations = 0
        var decoded: APIResponse
        var choice: APIResponse.Choice?
        var usage: PairedJudgeUsage?
        var truncated = false
        // Two independent retry axes that compose: the JSON constraint may
        // be dropped ONCE (single-shot — a second client error is a real
        // failure, not another axis), and the cap may be doubled up to
        // `escalationLimit` times. Server twin: `openrouter_complete`.
        var constrained = true
        while true {
            let body = requestBody(
                model: model, provider: trimmedProvider, prompt: prompt,
                maxTokens: cap, constrained: constrained)

            var request = URLRequest(url: endpoint)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            request.timeoutInterval = 300

            let (data, response) = try await transport(request)
            guard let http = response as? HTTPURLResponse else {
                throw PairedJudgeError(reason: "no HTTP response")
            }
            guard http.statusCode == 200 else {
                // Graceful fallback, ONE axis at a time: a client error
                // against the CONSTRAINED request is how OpenRouter surfaces
                // an endpoint that does not support `response_format` under a
                // no-fallback provider pin. The retry keeps the same cap —
                // the constraint was the problem, not the budget.
                if constrained,
                    constraintRejectionStatuses.contains(http.statusCode)
                {
                    constrained = false
                    print(
                        "judge '\(model)' @ \(trimmedProvider) rejected "
                            + "response_format \(constrainedFormat) (HTTP "
                            + "\(http.statusCode)) — retrying unconstrained; "
                            + "the JSON parser and truncation salvage still "
                            + "apply")
                    continue
                }
                let detail = String(decoding: data.prefix(300), as: UTF8.self)
                throw PairedJudgeError(
                    reason: "OpenRouter HTTP \(http.statusCode): \(detail)")
            }
            decoded = try JSONDecoder().decode(APIResponse.self, from: data)
            choice = decoded.choices?.first
            usage = decoded.usage?.judgeUsage
            if let reasoning = usage?.reasoningTokens {
                // Transparency, not a gate: the researcher sees what the
                // judge spent. Providers differ on whether completion tokens
                // already include the reasoning tokens, so both numbers are
                // printed as reported rather than differenced.
                print(
                    "judge '\(model)' used \(reasoning) reasoning + "
                        + "\(usage?.completionTokens ?? 0) answer tokens")
            }
            truncated = choice?.wasTruncated ?? false
            guard truncated, escalations < escalationLimit else { break }
            escalations += 1
            let previous = cap
            cap *= 2
            print(
                "judge '\(model)' hit its \(previous)-token cap "
                    + "(finish_reason "
                    + "\(choice?.finishReason ?? choice?.nativeFinishReason ?? "?")) "
                    + "— retrying with \(cap); a judge is never refused for "
                    + "spending tokens on reasoning")
        }

        // The response must NAME its serving provider (engineer review
        // 2026-07-18, provider-evidence pass): a missing field would
        // otherwise record the REQUESTED provider as if it were verified —
        // provenance is refused rather than believed, never assumed.
        guard let servedBy = decoded.provider, !servedBy.isEmpty else {
            throw PairedJudgeError(
                reason: "OpenRouter response for '\(model)' named no "
                    + "serving provider — cannot verify the provider pin; "
                    + "refusing an unattributed judgment")
        }
        // allow_fallbacks=false should make a mismatch impossible; verify
        // anyway — a judgment served off-pin is not the declared judge.
        if !OpenRouterProviderIdentity.equivalent(servedBy, trimmedProvider) {
            throw PairedJudgeError(
                reason: "OpenRouter served the judgment via '\(servedBy)' "
                    + "but the pinned provider is '\(trimmedProvider)' — "
                    + "refusing an off-pin judgment")
        }
        let content = choice?.message?.content ?? ""
        // Empty content WITH a length finish is the 2026-08-05 incident
        // (deepseek-v4-flash: HTTP 200, no content, the whole cap eaten by
        // hidden reasoning). Escalation has already been exhausted by the
        // time we get here, so the empty text goes on to the normal parse
        // path, which names the mechanism — refusing here as "carried no
        // content" would hide it. Empty content WITHOUT a length finish
        // keeps the old refusal: that is a genuinely empty answer, not a
        // truncated one.
        if content.isEmpty && !truncated {
            throw PairedJudgeError(
                reason: "OpenRouter response for '\(model)' carried no content")
        }
        // The CANONICAL serving provider rides out with the text: the
        // server's inline path stamps `canonical(served_by)`, and a
        // recorded string is provenance only if it is the verified one.
        return Completion(
            text: content,
            provider: OpenRouterProviderIdentity.canonical(servedBy),
            usage: (usage?.isEmpty ?? true) ? nil : usage,
            finishReason: choice?.finishReason ?? choice?.nativeFinishReason,
            truncated: truncated,
            capReached: cap,
            responseFormat: constrained ? constrainedFormat : unsupportedFormat)
    }
}
