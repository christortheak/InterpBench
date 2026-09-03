import Foundation
import Testing

@testable import ExperimentKit

/// The 2026-08-06 reasoning-accommodation round: welcome smarter judges.
///
/// Two incidents, one mechanism. A reasoning-first model bills its HIDDEN
/// reasoning against the same generation cap as its visible answer, so our
/// 1024-token judge cap truncated good analysis:
///
/// - 2026-08-05: `deepseek/deepseek-v4-flash-0731` returned HTTP 200 with
///   EMPTY content — the transport refused with "carried no content";
/// - 2026-08-06: `google/gemini-3.6-flash` @ google-ai-studio returned
///   ~170-200-character mid-sentence prose fragments with no JSON, and the
///   run refused after its one retry (workspace run
///   20260805T232636162-exp-replication-1-evaluate).
///
/// Neither judge was doing anything wrong. The contract under test (server
/// twin: `Server/tests/test_judge_reasoning_accommodation.py`):
///
/// - the shared cap is 8192, cross-engine;
/// - OpenRouter requests carry `reasoning: {"exclude": true}` — skip
///   DELIVERING the reasoning text, never limit the thinking (deliberately
///   no effort/max_tokens reasoning knobs);
/// - a length finish_reason ESCALATES the cap (8192 → 16384 → 32768), per
///   call and stateless, including when the content came back empty;
/// - empty content WITHOUT a length finish keeps the old refusal;
/// - token usage is stamped on the verdict and summed per judge into the
///   report — and NOTHING reads it to gate;
/// - a leading `<think>` block (closed or unclosed) is stripped before JSON
///   extraction;
/// - when the refusal does fire after escalation, the message names the
///   mechanism instead of "no JSON object".
///
/// The invented-data firewall is untouched: a judge that HAD its budget and
/// still returned garbage still gets retried once and then refuses.
@Suite(.serialized) struct PairedJudgeReasoningAccommodationTests {

    // MARK: - Scripted transport

    /// Replays canned OpenRouter responses and records what was sent, so the
    /// escalation loop is exercised without a network.
    private actor ScriptedOpenRouter {
        private var remaining: [String]
        private(set) var requestBodies: [Data] = []

        init(_ responses: [String]) { remaining = responses }

        func handle(_ request: URLRequest) throws -> (Data, URLResponse) {
            requestBodies.append(request.httpBody ?? Data())
            guard !remaining.isEmpty else {
                throw PairedJudgeError(
                    reason: "scripted transport exhausted — the client made "
                        + "more calls than the test scripted")
            }
            let response = HTTPURLResponse(
                url: OpenRouterPairedJudge.endpoint, statusCode: 200,
                httpVersion: nil, headerFields: nil)!
            return (Data(remaining.removeFirst().utf8), response)
        }

    }

    /// Request bodies as dictionaries. Decoded OUTSIDE the actor because
    /// `[String: Any]` is not Sendable — `Data` crosses the boundary.
    private static func decoded(_ bodies: [Data]) -> [[String: Any]] {
        bodies.compactMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        }
    }

    private static func transport(
        _ script: ScriptedOpenRouter
    ) -> OpenRouterPairedJudge.Transport {
        { request in try await script.handle(request) }
    }

    /// One OpenRouter response body.
    private static func payload(
        content: String,
        finishReason: String? = nil,
        nativeFinishReason: String? = nil,
        provider: String = "Google AI Studio",
        usage: String? = nil
    ) -> String {
        var choice = "{\"message\": {\"content\": \(quoted(content))}"
        if let finishReason {
            choice += ", \"finish_reason\": \(quoted(finishReason))"
        }
        if let nativeFinishReason {
            choice += ", \"native_finish_reason\": \(quoted(nativeFinishReason))"
        }
        choice += "}"
        var body = "{\"provider\": \(quoted(provider)), \"choices\": [\(choice)]"
        if let usage { body += ", \"usage\": \(usage)" }
        return body + "}"
    }

    private static func quoted(_ text: String) -> String {
        String(
            decoding: try! JSONSerialization.data(
                withJSONObject: [text], options: [.fragmentsAllowed]),
            as: UTF8.self)
        .dropFirst().dropLast().description
    }

    private static let verdictJSON =
        "{\"winner\": \"A\", \"confidence\": 0.8, \"brief_reason\": \"A is tighter.\"}"

    /// Runs `body` with the credential seam armed, always clearing it.
    private func withKey<T>(_ body: () async throws -> T) async rethrows -> T {
        OpenRouterPairedJudge.keyOverrideForTesting = "sk-or-test"
        defer { OpenRouterPairedJudge.keyOverrideForTesting = nil }
        return try await body()
    }

    private func judge(
        model: String = "google/gemini-3.6-flash",
        provider: String = "Google AI Studio",
        script: ScriptedOpenRouter
    ) async throws -> PairedJudgeResponse {
        try await withKey {
            try await OpenRouterPairedJudge.judge(
                model: model, provider: provider, rubric: "r",
                structuredPrompt: nil, prompt: "p", responseA: "a",
                responseB: "b", transport: Self.transport(script))
        }
    }

    // MARK: - The cap and the reasoning knob

    @Test func theCapAccommodatesReasoningModels() {
        #expect(PairedJudgeBudget.maxTokens == 8192)
        #expect(OpenRouterPairedJudge.escalationLimit == 2)
    }

    @Test func requestBodyExcludesReasoningDeliveryWithoutLimitingIt() {
        let body = OpenRouterPairedJudge.requestBody(
            model: "google/gemini-3.6-flash", provider: "Google AI Studio",
            rubric: "r", structuredPrompt: nil, prompt: "p",
            responseA: "a", responseB: "b")
        let reasoning = body["reasoning"] as? [String: Any]
        #expect(reasoning?["exclude"] as? Bool == true)
        // No effort/token knobs: we do not ration how hard a judge thinks.
        #expect(reasoning?.count == 1)
        #expect(body["max_tokens"] as? Int == PairedJudgeBudget.maxTokens)
    }

    @Test func theRawPromptBodyCarriesTheSameStance() {
        // The per-response coding instrument shares this transport, so it
        // inherits the accommodation for free.
        let body = OpenRouterPairedJudge.requestBody(
            model: "google/gemini-3.6-flash", provider: "Google AI Studio",
            prompt: "code this")
        #expect((body["reasoning"] as? [String: Any])?["exclude"] as? Bool == true)
        #expect(body["max_tokens"] as? Int == PairedJudgeBudget.maxTokens)
    }

    // MARK: - Truncation escalation

    @Test func lengthFinishEscalatesTheCapAndThenParses() async throws {
        let script = ScriptedOpenRouter([
            Self.payload(content: "{\"winner\": \"A\", \"confi",
                         finishReason: "length"),
            Self.payload(content: Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.winner == "A")
        let bodies = Self.decoded(await script.requestBodies)
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384])
        // Everything else about the escalated request is IDENTICAL.
        #expect(
            (bodies[0]["messages"] as? NSArray)
                == (bodies[1]["messages"] as? NSArray))
        #expect(
            (bodies[1]["reasoning"] as? [String: Any])?["exclude"] as? Bool == true)
    }

    @Test func aNativeFinishReasonAloneEscalates() async throws {
        // OpenRouter normalizes into finish_reason, but a provider's own
        // spelling (Google: MAX_TOKENS) has been the only signal present.
        let script = ScriptedOpenRouter([
            Self.payload(content: "Response A is stronger because it",
                         finishReason: "stop",
                         nativeFinishReason: "MAX_TOKENS"),
            Self.payload(content: Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.winner == "A")
        let bodies = Self.decoded(await script.requestBodies)
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384])
    }

    @Test func emptyContentWithALengthFinishEscalates() async throws {
        // Incident 1 regression (2026-08-05, deepseek-v4-flash): HTTP 200,
        // empty content, the whole cap spent on hidden reasoning. That must
        // buy the judge more room, never a "carried no content" refusal.
        let script = ScriptedOpenRouter([
            Self.payload(content: "", finishReason: "length",
                         provider: "Anthropic"),
            Self.payload(content: Self.verdictJSON, provider: "Anthropic"),
        ])

        let verdict = try await judge(
            model: "deepseek/deepseek-v4-flash-0731", provider: "Anthropic",
            script: script)

        #expect(verdict.winner == "A")
        let bodies = Self.decoded(await script.requestBodies)
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384])
    }

    @Test func emptyContentWithoutALengthFinishStillRefuses() async throws {
        // A genuinely empty answer is not a truncated one — the old refusal
        // stands, unchanged, and buys no escalation.
        let script = ScriptedOpenRouter([
            Self.payload(content: "", finishReason: "stop")
        ])

        await #expect(throws: PairedJudgeError.self) {
            try await judge(script: script)
        }
        #expect(await script.requestBodies.count == 1)
    }

    @Test func escalationStopsAfterTwoDoublings() async throws {
        let truncated = Self.payload(
            content: "Response A begins to make the stronger",
            finishReason: "length")
        let script = ScriptedOpenRouter([truncated, truncated, truncated])

        await #expect(throws: PairedJudgeError.self) {
            try await judge(script: script)
        }
        let bodies = Self.decoded(await script.requestBodies)
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384, 32768])
    }

    @Test func escalationIsStatelessAcrossCalls() async throws {
        // No learned cap: the NEXT judgment starts at the shared cap again.
        let script = ScriptedOpenRouter([
            Self.payload(content: "", finishReason: "length"),
            Self.payload(content: Self.verdictJSON),
            Self.payload(content: Self.verdictJSON),
        ])

        _ = try await judge(script: script)
        _ = try await judge(script: script)

        let bodies = Self.decoded(await script.requestBodies)
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384, 8192])
    }

    @Test func theExhaustedEscalationDiagnosisNamesTheMechanism() async throws {
        // The 2026-08-06 shape: a mid-sentence prose fragment, no JSON at
        // all. "judge response had no JSON object" sent the researcher after
        // the rubric; the message must name truncation instead.
        let fragment = Self.payload(
            content: "Response A adopts the more formalist posture, tracking "
                + "the statute's",
            finishReason: "length")
        let script = ScriptedOpenRouter([fragment, fragment, fragment])

        var reason = ""
        do {
            _ = try await judge(script: script)
            Issue.record("expected a refusal")
        } catch let error as PairedJudgeError {
            reason = error.reason
        }
        #expect(reason.contains("truncated after 32768 tokens despite escalation"))
        #expect(reason.contains("hidden reasoning"))
        #expect(reason.contains("judge-failures.jsonl"))
    }

    @Test func aBudgetedJudgeThatReturnsGarbageStillRefusesPlainly() async throws {
        // The firewall is intact: no length finish, no reasoning tokens —
        // the judge had its budget and answered incoherently. Plain refusal
        // wording, no truncation excuse.
        let script = ScriptedOpenRouter([
            Self.payload(content: "I refuse to pick a winner.",
                         finishReason: "stop")
        ])

        var reason = ""
        do {
            _ = try await judge(script: script)
            Issue.record("expected a refusal")
        } catch let error as PairedJudgeError {
            reason = error.reason
        }
        #expect(reason == "judge response had no JSON object")
    }

    @Test(arguments: [
        "length", "max_tokens", "MAX_TOKENS", "max-tokens", "model_length",
    ])
    func lengthFinishVocabulary(reason: String) {
        #expect(OpenRouterPairedJudge.isLengthFinish(reason))
    }

    @Test(arguments: ["stop", "end_turn", "content_filter", "", "tool_calls"])
    func nonLengthFinishReasonsDoNotEscalate(reason: String) {
        #expect(!OpenRouterPairedJudge.isLengthFinish(reason))
        #expect(!OpenRouterPairedJudge.isLengthFinish(nil))
    }

    // MARK: - Usage transparency (report, never gate)

    @Test func usageIsStampedOnTheVerdict() async throws {
        let script = ScriptedOpenRouter([
            Self.payload(
                content: Self.verdictJSON,
                usage: "{\"completion_tokens\": 900, "
                    + "\"completion_tokens_details\": {\"reasoning_tokens\": 812}}")
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.usage?.completionTokens == 900)
        #expect(verdict.usage?.reasoningTokens == 812)
    }

    @Test func aTopLevelReasoningTokensFieldIsReadDefensively() async throws {
        let script = ScriptedOpenRouter([
            Self.payload(
                content: Self.verdictJSON,
                usage: "{\"completion_tokens\": 12, \"reasoning_tokens\": 5}")
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.usage == PairedJudgeUsage(
            completionTokens: 12, reasoningTokens: 5))
    }

    @Test func aResponseWithoutUsageStampsNothing() async throws {
        let script = ScriptedOpenRouter([
            Self.payload(content: Self.verdictJSON)
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.usage == nil)
    }

    @Test func usageSumsPerJudgeForTheReport() {
        // Cross-engine twin of the server's per-judge report block sums
        // (`paired_judge.sum_judgment_usage`), which this engine keys by
        // panel name because its `judges` are bare strings.
        let records = [
            Self.record(judge: "j1", usage: .init(completionTokens: 100,
                                                  reasoningTokens: 90)),
            Self.record(judge: "j1", usage: .init(completionTokens: 120,
                                                  reasoningTokens: 100)),
            Self.record(judge: "j2", usage: .init(completionTokens: 40,
                                                  reasoningTokens: nil)),
            Self.record(judge: "local", usage: nil),
        ]

        let totals = ExperimentTasks.judgeUsageTotals(records: records)

        #expect(totals?["j1"] == PairedJudgeUsage(completionTokens: 220,
                                                 reasoningTokens: 190))
        #expect(totals?["j2"] == PairedJudgeUsage(completionTokens: 40,
                                                 reasoningTokens: nil))
        // A judge whose transport reports nothing contributes no entry.
        #expect(totals?["local"] == nil)
    }

    @Test func aRunWithNoReportedUsageOmitsTheKeyEntirely() {
        #expect(
            ExperimentTasks.judgeUsageTotals(
                records: [Self.record(judge: "j1", usage: nil)]) == nil)
    }

    private static func record(
        judge: String, usage: PairedJudgeUsage?
    ) -> ExperimentTasks.PairedJudgeRecord {
        ExperimentTasks.PairedJudgeRecord(
            experiment: "exp", experimentHash: "h", sourceRunDirectory: "/run",
            judgeName: judge, judgeKind: "openrouter",
            judgeModel: "google/gemini-3.6-flash",
            judgePrompt: "r", judgeRubricFile: nil, judgeRubricHash: nil,
            structuredPrompt: nil, condition: "fear", sampleIndex: 0,
            baselineSeed: 1, variantSeed: 2, promptID: "p0", prompt: "q",
            baselineWas: "A", conditionWas: "B",
            judgment: PairedJudgeResponse(
                winner: "A", confidence: 0.8, briefReason: "r", usage: usage),
            conditionResult: "baseline")
    }

    // MARK: - Parser robustness: inlined reasoning

    @Test func aClosedThinkBlockIsStrippedBeforeJSONExtraction() throws {
        let verdict = try PairedJudgeVerdictParser.parse(
            "<think>Let me weigh {rigor} against clarity. A cites the "
                + "statute; B does not. So {\"winner\": \"B\"} would be "
                + "wrong.</think>\n{\"winner\": \"A\", \"confidence\": 0.9, "
                + "\"brief_reason\": \"A cites it.\"}")
        #expect(verdict.winner == "A")
        #expect(verdict.confidence == 0.9)
        #expect(verdict.reasoningTruncated == nil)
    }

    @Test func anUnclosedThinkBlockTreatsTheRestAsCandidateText() throws {
        let verdict = try PairedJudgeVerdictParser.parse(
            "<think>Weighing the two responses…\n{\"winner\": \"B\", "
                + "\"confidence\": 0.6, \"brief_reason\": \"B is tighter.\"}")
        #expect(verdict.winner == "B")
        #expect(verdict.briefReason == "B is tighter.")
    }

    @Test func thinkStrippingIsLeadingOnlyAndWhitespaceTolerant() {
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "  \n<think>hidden</think>rest") == "rest")
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "<THINK id=\"1\">hidden</THINK>rest") == "rest")
        // A think tag AFTER the verdict is content, not a preamble.
        let tail = "{\"winner\": \"A\"} <think>afterthought</think>"
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(tail) == tail)
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock("no tags here")
                == "no tags here")
    }

    @Test func anOrphanClosingThinkTagTerminatesTheReasoningPreamble() throws {
        // Qwen3 with enable_thinking: the chat template emits the OPENING
        // `<think>` as the last token of the PROMPT, and generation decodes
        // with skip_prompt, so the text begins INSIDE the reasoning block
        // and the first `</think>` is the only boundary it carries
        // (2026-09-03).
        let thinking =
            "Let me compare. A cites the statute; B does not. A draft verdict\n"
            + "might be {\"winner\": \"B\"} but that is wrong.\n</think>\n\n"
        let verdictJSON =
            "{\"winner\": \"A\", \"confidence\": 0.9, "
            + "\"brief_reason\": \"A cites it.\"}"
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                thinking + verdictJSON) == "\n\n" + verdictJSON)
        let verdict = try PairedJudgeVerdictParser.parse(thinking + verdictJSON)
        #expect(verdict.winner == "A")
        #expect(verdict.confidence == 0.9)
        #expect(verdict.reasoningTruncated == nil)
        // Case-insensitive, like the opening-tag path.
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "hidden</THINK>rest") == "rest")
    }

    @Test func allFourThinkBoundaryShapes() {
        // The four shapes the stripper has to tell apart, side by side.
        let body = "{\"winner\": \"A\"}"
        // 1. Opening tag present (closed): the whole block comes off.
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "<think>hidden</think>" + body) == body)
        // 2. Orphan closing tag only (Qwen3 thinking mode): terminator.
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "hidden</think>" + body) == body)
        // 3. No tags at all: untouched.
        #expect(PairedJudgeVerdictParser.strippingLeadingThinkBlock(body) == body)
        // 4. Unclosed/truncated reasoning. With an opening tag the rest is
        //    the candidate; WITHOUT one (thinking mode cut before
        //    `</think>`) the text carries no tag and is indistinguishable
        //    from plain prose, so it is returned unchanged.
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(
                "<think>still deliberating " + body)
                == "still deliberating " + body)
        let truncated = "still deliberating about {A} and {B"
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(truncated)
                == truncated)
    }

    @Test func anOrphanClosingTagAfterAThinkAsideIsNotATerminator() throws {
        // The orphan rule needs the closing tag to precede EVERY `<think>`:
        // a `<think>…</think>` aside after the verdict is content, and its
        // closing tag must not drag the verdict away with it.
        let tail = "{\"winner\": \"A\"} <think>afterthought</think> more"
        #expect(PairedJudgeVerdictParser.strippingLeadingThinkBlock(tail) == tail)
        #expect(try PairedJudgeVerdictParser.parse(tail).winner == "A")
        // Likewise a leading block followed by an aside: only the leading
        // block comes off, the aside stays.
        let both = "<think>hidden</think>{\"winner\": \"B\"} <think>aside</think>"
        #expect(
            PairedJudgeVerdictParser.strippingLeadingThinkBlock(both)
                == "{\"winner\": \"B\"} <think>aside</think>")
    }

    @Test func aThinkingModeResponseThatIsAllReasoningStillRefuses() {
        // Orphan-closing-tag shape with nothing after the boundary: no
        // verdict to salvage, so the refusal path is unchanged.
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(
                "I compared {A} and {B} at length.</think>\n")
        }
    }

    @Test func aThinkOnlyResponseStillRefuses() {
        // Nothing but reasoning: no verdict to salvage, so the refusal path
        // is unchanged.
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(
                "<think>I am still deliberating about {A} and {B}")
        }
    }
}
