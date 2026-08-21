import Foundation
import Testing

@testable import ExperimentKit

/// Constrain the judge's output format at the ENDPOINT, and make every
/// judgment record carry its own diagnosis (2026-08-06 evaluate
/// post-mortem). Server twin:
/// `Server/tests/test_judge_structured_output_and_provenance.py`.
///
/// The server hardened; this engine did not, so the same judge behaved
/// differently per substrate. Contract under test here:
///
/// - OpenRouter requests carry `response_format: {"type": "json_object"}`
///   (`json_schema` is deliberately NOT sent — the verdict's shape is
///   rubric-dependent and a schema here would be a drifting second copy of
///   the prompt contract);
/// - an endpoint that rejects the constraint (client error under the
///   no-fallback provider pin) gets ONE unconstrained retry, at the SAME
///   cap; auth/billing/rate/server errors never trigger the fallback, and
///   the fallback is single-shot;
/// - the constraint survives cap escalation (the two retry axes compose);
/// - every verdict stamps `finishReason` and `responseFormat` (plus `usage`
///   when the response reported it) — recorded provenance, gating nothing.
@Suite(.serialized) struct OpenRouterJudgeConstraintTests {

    /// Replays canned `(status, body)` responses and records what was sent.
    private actor ScriptedOpenRouter {
        private var remaining: [(Int, String)]
        private(set) var requestBodies: [Data] = []

        init(_ responses: [(Int, String)]) { remaining = responses }

        func handle(_ request: URLRequest) throws -> (Data, URLResponse) {
            requestBodies.append(request.httpBody ?? Data())
            guard !remaining.isEmpty else {
                throw PairedJudgeError(
                    reason: "scripted transport exhausted — the client made "
                        + "more calls than the test scripted")
            }
            let (status, body) = remaining.removeFirst()
            let response = HTTPURLResponse(
                url: OpenRouterPairedJudge.endpoint, statusCode: status,
                httpVersion: nil, headerFields: nil)!
            return (Data(body.utf8), response)
        }
    }

    /// `[String: Any]` is not Sendable — the `Data` crosses the boundary.
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

    private static let verdictJSON =
        #"{"winner": "A", "confidence": 0.8, "brief_reason": "A is tighter."}"#

    private static func ok(
        _ content: String, finishReason: String? = nil, usage: String? = nil
    ) -> (Int, String) {
        var choice = "{\"message\": {\"content\": \(quoted(content))}"
        if let finishReason {
            choice += ", \"finish_reason\": \(quoted(finishReason))"
        }
        choice += "}"
        var body = "{\"provider\": \"Anthropic\", \"choices\": [\(choice)]"
        if let usage { body += ", \"usage\": \(usage)" }
        return (200, body + "}")
    }

    private static func error(_ status: Int) -> (Int, String) {
        (status, #"{"error": {"message": "no endpoints found"}}"#)
    }

    private static func quoted(_ text: String) -> String {
        String(
            decoding: try! JSONSerialization.data(
                withJSONObject: [text], options: [.fragmentsAllowed]),
            as: UTF8.self)
        .dropFirst().dropLast().description
    }

    private func withKey<T>(_ body: () async throws -> T) async rethrows -> T {
        OpenRouterPairedJudge.keyOverrideForTesting = "sk-or-test"
        defer { OpenRouterPairedJudge.keyOverrideForTesting = nil }
        return try await body()
    }

    private func judge(
        script: ScriptedOpenRouter
    ) async throws -> PairedJudgeResponse {
        try await withKey {
            try await OpenRouterPairedJudge.judge(
                model: "anthropic/claude-opus-4.8", provider: "Anthropic",
                rubric: "rubric", structuredPrompt: nil, prompt: "p",
                responseA: "a", responseB: "b",
                transport: Self.transport(script))
        }
    }

    private static func responseFormat(_ body: [String: Any]) -> String? {
        (body["response_format"] as? [String: Any])?["type"] as? String
    }

    // MARK: - The endpoint-side JSON constraint

    @Test func theRequestConstrainsOutputToAJSONObject() async throws {
        let script = ScriptedOpenRouter([Self.ok(Self.verdictJSON)])

        let verdict = try await judge(script: script)

        let bodies = Self.decoded(await script.requestBodies)
        #expect(Self.responseFormat(bodies[0]) == "json_object")
        // json_schema is deliberately absent: one prompt contract, not two.
        #expect(
            (bodies[0]["response_format"] as? [String: Any])?.count == 1)
        #expect(verdict.winner == "A")
        #expect(verdict.responseFormat == "json_object")
    }

    @Test func theCodingTransportSharesTheJSONConstraint() {
        // The per-response coding instrument's prompt also demands "Return
        // JSON only" — the shared raw transport constrains both callers.
        let body = OpenRouterPairedJudge.requestBody(
            model: "anthropic/claude-opus-4.8", provider: "Anthropic",
            prompt: "code this")
        #expect(Self.responseFormat(body) == "json_object")
    }

    @Test func anEndpointThatRejectsTheConstraintFallsBackUnconstrained()
        async throws
    {
        let script = ScriptedOpenRouter([
            Self.error(404), Self.ok(Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        let bodies = Self.decoded(await script.requestBodies)
        #expect(Self.responseFormat(bodies[0]) == "json_object")
        #expect(bodies[1]["response_format"] == nil)
        // The fallback burns no cap escalation — the constraint was the
        // problem, not the budget — and the verdict SAYS the constraint was
        // not in force for the text that came back.
        #expect(
            bodies.map { $0["max_tokens"] as? Int }
                == [PairedJudgeBudget.maxTokens, PairedJudgeBudget.maxTokens])
        #expect(verdict.winner == "A")
        #expect(verdict.responseFormat == "unsupported")
    }

    @Test(arguments: [400, 404, 422])
    func everyConstraintRejectionStatusFallsBack(status: Int) async throws {
        let script = ScriptedOpenRouter([
            Self.error(status), Self.ok(Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.responseFormat == "unsupported")
        #expect(await script.requestBodies.count == 2)
    }

    @Test func theFallbackIsSingleShot() async throws {
        // A second client error after dropping the constraint is a real
        // failure, not another axis to retry.
        let script = ScriptedOpenRouter([Self.error(400), Self.error(400)])

        await #expect(throws: PairedJudgeError.self) {
            try await judge(script: script)
        }
        #expect(await script.requestBodies.count == 2)
    }

    @Test(arguments: [401, 402, 429, 500])
    func authBillingAndRateErrorsNeverTriggerTheFallback(
        status: Int
    ) async throws {
        // These are not about the constraint — dropping it would just
        // re-send a doomed request (and double-bill a rate-limited one).
        let script = ScriptedOpenRouter([Self.error(status)])

        await #expect(throws: PairedJudgeError.self) {
            try await judge(script: script)
        }
        #expect(await script.requestBodies.count == 1)
    }

    @Test func theConstraintSurvivesCapEscalation() async throws {
        // The two retry axes compose: a length cut escalates the cap and the
        // escalated request still asks for a JSON object.
        let script = ScriptedOpenRouter([
            Self.ok(#"{"winner": "A", "confi"#, finishReason: "length"),
            Self.ok(Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        let bodies = Self.decoded(await script.requestBodies)
        #expect(verdict.winner == "A")
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 16384])
        #expect(bodies.allSatisfy { Self.responseFormat($0) == "json_object" })
    }

    @Test func theConstraintStaysDroppedAcrossEscalation() async throws {
        // Dropped once, dropped for the rest of the CALL: an escalation
        // after the fallback must not re-send the parameter the endpoint
        // just rejected.
        let script = ScriptedOpenRouter([
            Self.error(422),
            Self.ok("Response A begins to", finishReason: "length"),
            Self.ok(Self.verdictJSON),
        ])

        let verdict = try await judge(script: script)

        let bodies = Self.decoded(await script.requestBodies)
        #expect(verdict.responseFormat == "unsupported")
        #expect(bodies.map { $0["max_tokens"] as? Int } == [8192, 8192, 16384])
        #expect(bodies.dropFirst().allSatisfy { $0["response_format"] == nil })
    }

    // MARK: - Transport provenance on every judgment record

    @Test func finishReasonAndUsageAreStampedOnEveryVerdict() async throws {
        let script = ScriptedOpenRouter([
            Self.ok(
                Self.verdictJSON, finishReason: "stop",
                usage: #"{"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 812}}"#)
        ])

        let verdict = try await judge(script: script)

        #expect(verdict.finishReason == "stop")
        #expect(verdict.responseFormat == "json_object")
        #expect(
            verdict.usage
                == PairedJudgeUsage(completionTokens: 900, reasoningTokens: 812))
    }

    @Test func aSalvagedVerdictStampsItsLengthFinish() async throws {
        // Escalation exhausts, salvage rescues the winner: the verdict must
        // say BOTH that its reasoning was truncated and how the call ended,
        // so a salvage is never indistinguishable from a clean verdict.
        let salvageable = #"{"winner": "A", "confidence": 0.9, "brief_reason": "A cites"#
        let script = ScriptedOpenRouter(
            Array(repeating: Self.ok(salvageable, finishReason: "length"), count: 3))

        let verdict = try await judge(script: script)

        #expect(verdict.winner == "A")
        #expect(verdict.reasoningTruncated == true)
        #expect(verdict.finishReason == "length")
    }

    /// The stamps must SURVIVE the round trip: judgments.jsonl is where a
    /// report reader looks, and cross-engine key parity is the point.
    @Test func theStampsRoundTripUnderTheServersKeys() throws {
        let verdict = PairedJudgeResponse(
            winner: "A", confidence: 0.8, briefReason: "r",
            usage: .init(completionTokens: 10, reasoningTokens: 4),
            finishReason: "length", responseFormat: "json_object")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(verdict)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])

        #expect(object["finishReason"] as? String == "length")
        #expect(object["responseFormat"] as? String == "json_object")
        #expect(
            try JSONDecoder().decode(PairedJudgeResponse.self, from: data)
                == verdict)

        // A legacy verdict carrying neither key still decodes (the keys are
        // additive provenance, never a decoding requirement).
        let legacy = Data(
            #"{"winner": "A", "confidence": 0.5, "brief_reason": "r"}"#.utf8)
        let decoded = try JSONDecoder().decode(
            PairedJudgeResponse.self, from: legacy)
        #expect(decoded.finishReason == nil)
        #expect(decoded.responseFormat == nil)
    }
}
