import Foundation
import Testing

@testable import ExperimentKit

/// The 2026-07-22 judging incident round: truncated-verdict salvage, the
/// unified judge generation cap, and the golden-pinned prompt wrapper.
///
/// A cluster paired-judge evaluate failed after 15 minutes because a local
/// judge stated a valid winner and confidence FIRST, then wrote an
/// essay-length reasoning that outran its token cap — the JSON never
/// closed, balanced-brace parsing refused, and a legible verdict died.
/// Contract under test (server twin:
/// `Server/tests/test_judge_wrapper_and_salvage.py`):
///
/// - salvage: a truncated response with a COMPLETE `"winner"` (exactly
///   A/B/tie) is accepted with `reasoningTruncated: true` — no retry
///   burned; truncation BEFORE the winner completes (or an
///   out-of-vocabulary winner) keeps the refusal path;
/// - cap: `PairedJudgeBudget.maxTokens` (8192 since the 2026-08-06
///   reasoning-accommodation round — see
///   `PairedJudgeReasoningAccommodationTests`; twin of the server's
///   `paired_judge.JUDGE_MAX_TOKENS`) is what every judge client on this
///   engine actually passes — Claude, OpenRouter, and the local judge;
/// - wrapper: `PairedJudgePrompt.build` is byte-identical to the committed
///   goldens in `prompts/fixtures/paired-judge/` (the Python builder is
///   pinned to the same bytes).
@Suite struct PairedJudgeWrapperAndSalvageTests {

    /// The exact incident shape (cluster evaluate, 2026-07-22): fenced
    /// JSON, verdict fields complete, reasoning cut mid-word by the cap.
    static let incidentText = "```json\n{\"winner\": \"A\", \"confidence\": 0.95, "
        + "\"reasoning\": \"Response A exhibits a more distinctly Californian "
        + "sensibility in its treatment of the evo"

    // MARK: - Truncated-verdict salvage

    @Test func salvagesTheExactIncidentShape() throws {
        let verdict = try PairedJudgeVerdictParser.parse(Self.incidentText)
        #expect(verdict.winner == "A")
        #expect(verdict.confidence == 0.95)
        #expect(verdict.reasoningTruncated == true)
        #expect(verdict.briefReason.hasPrefix("Response A exhibits"))
        #expect(verdict.briefReason.hasSuffix("the evo"))
    }

    @Test func salvagedVerdictConsumesNoRetry() async throws {
        // The winner was explicitly written — the valid-winner wrapper must
        // accept it on the FIRST call, never burning the single retry on a
        // legible verdict.
        let calls = Counter()
        let verdict = try await ExperimentTasks.judgmentWithValidWinner(
            judgeName: "google/gemma-3-4b-it", item: "pair fear/p0"
        ) {
            await calls.increment()
            return try PairedJudgeVerdictParser.parse(Self.incidentText)
        }
        #expect(verdict.winner == "A")
        #expect(verdict.reasoningTruncated == true)
        #expect(await calls.value == 1)
    }

    @Test func completeFencedJSONStillParsesWithoutTruncationMark() throws {
        let verdict = try PairedJudgeVerdictParser.parse(
            "```json\n{\"winner\": \"B\", \"confidence\": 0.7, "
                + "\"brief_reason\": \"B is tighter.\"}\n```")
        #expect(verdict.winner == "B")
        #expect(verdict.confidence == 0.7)
        #expect(verdict.briefReason == "B is tighter.")
        #expect(verdict.reasoningTruncated == nil)
    }

    @Test(arguments: [
        "```json\n{\"winner\": \"A",
        "```json\n{\"win",
        "```json\n{\"winner\": \"",
        "{\"confidence\": 0.9, \"reasoning\": \"A seems better",
    ])
    func truncationBeforeTheWinnerCompletesStillRefuses(text: String) {
        // No closing quote after the winner value — nothing legible to
        // salvage; the refusal path (identical wording to the server's
        // parse_response) stays engaged.
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(text)
        }
    }

    @Test(arguments: [
        "{\"winner\": \"C\", \"confidence\": 0.9, \"reasoning\": \"cut",
        "{\"winner\": \"AB\", \"confidence\": 0.9, \"reasoning\": \"cut",
        "{\"winner\": \"A and B are equal\", \"reasoning\": \"cut",
        "{\"winner\": \"both\", \"reasoning\": \"cut",
    ])
    func salvageNeverAcceptsAnOutOfVocabularyWinner(text: String) {
        // A complete winner field whose value is outside A/B/tie is
        // invented data, truncated or not — refuse.
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(text)
        }
    }

    @Test func salvageDropsAnIncompleteConfidenceNumber() throws {
        // The text ends mid-number: "0.9" could have been 0.95 — a number
        // without a trailing delimiter is not complete and is dropped
        // (decoding as 0 on this engine's non-optional field).
        let verdict = try PairedJudgeVerdictParser.parse(
            "{\"winner\": \"tie\", \"confidence\": 0.9")
        #expect(verdict.winner == "tie")
        #expect(verdict.confidence == 0)
        #expect(verdict.reasoningTruncated == true)
        #expect(verdict.briefReason.isEmpty)
    }

    @Test func salvageUnescapesTheReasoningPrefix() throws {
        let verdict = try PairedJudgeVerdictParser.parse(
            "{\"winner\": \"B\", \"confidence\": 0.6, "
                + "\"brief_reason\": \"Cites \\\"Rule 4\\\" precisely.\\nAlso ")
        #expect(verdict.briefReason == "Cites \"Rule 4\" precisely.\nAlso ")
        #expect(verdict.reasoningTruncated == true)
    }

    @Test func responseWithNoJSONAtAllNamesThat() {
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse("I cannot decide, sorry.")
        }
    }

    // MARK: - Finding 5 tightening (2026-07-23)

    @Test func validJSONWithBracesInsideStringsParsesExactly() throws {
        // The brace scan is string-aware: a verdict whose reason mentions
        // braces is VALID JSON and parses completely — never mistaken for
        // truncation (the server's old counter made exactly that mistake).
        let verdict = try PairedJudgeVerdictParser.parse(
            "{\"winner\": \"B\", \"confidence\": 0.8, "
                + "\"brief_reason\": \"B closes every {brace} it opens; A "
                + "writes } without opening.\"}")
        #expect(verdict.winner == "B")
        #expect(verdict.confidence == 0.8)
        #expect(verdict.briefReason.contains("{brace}"))
        #expect(verdict.reasoningTruncated == nil)
    }

    @Test func duplicateWinnerFieldsInTruncationRefuse() {
        // Two complete winner fields in a truncated response — conflicting
        // OR agreeing — have no single legible verdict: salvage refuses
        // into the retry-then-refuse path instead of parsing whichever came
        // first (identical rule to the server, 2026-07-23).
        let conflicting = "{\"winner\": \"A\", \"notes\": {\"winner\": \"B\", "
            + "\"confidence\": 0.9, \"brief_reason\": \"an essay that never"
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(conflicting)
        }
        let agreeing = "{\"winner\": \"A\", \"restated\": {\"winner\": \"A\", "
            + "\"brief_reason\": \"an essay that never"
        #expect(throws: PairedJudgeError.self) {
            try PairedJudgeVerdictParser.parse(agreeing)
        }
    }

    @Test func salvageIgnoresWinnerShapedProseBeforeTheJSON() throws {
        // Salvage is scoped to the CANDIDATE object (first "{" onward):
        // verdict-shaped prose before the JSON contributes nothing.
        let text = "I might have said \"winner\": \"B\" earlier, but: "
            + "{\"winner\": \"A\", \"brief_reason\": \"cut off mid"
        let verdict = try PairedJudgeVerdictParser.parse(text)
        #expect(verdict.winner == "A")
        #expect(verdict.reasoningTruncated == true)
    }

    // MARK: - The unified generation cap

    @Test func theUnifiedCapIs8192AndNamed() {
        // Twin constant: server paired_judge.JUDGE_MAX_TOKENS — keep
        // identical. Raised from 1024 on 2026-08-06: the cap bounds runaway
        // generation, it must never ration a reasoning model's thinking
        // (two incidents — see PairedJudgeReasoningAccommodationTests).
        #expect(PairedJudgeBudget.maxTokens == 8192)
    }

    @Test func claudeJudgeRequestBodyCarriesTheUnifiedCap() {
        let body = ClaudePairedJudge.requestBody(
            model: "claude-opus-4-8", rubric: "r", structuredPrompt: nil,
            prompt: "p", responseA: "a", responseB: "b")
        #expect(body["max_tokens"] as? Int == PairedJudgeBudget.maxTokens)
    }

    @Test func openRouterJudgeRequestBodyCarriesTheUnifiedCap() {
        let body = OpenRouterPairedJudge.requestBody(
            model: "google/gemma-3-27b-it", provider: "DeepInfra",
            rubric: "r", structuredPrompt: nil, prompt: "p",
            responseA: "a", responseB: "b")
        #expect(body["max_tokens"] as? Int == PairedJudgeBudget.maxTokens)
    }

    @Test func openRouterProviderDisplayNameAndSlugShareOneIdentity() {
        #expect(
            OpenRouterProviderIdentity.equivalent(
                "Google AI Studio", "google-ai-studio"))
        #expect(
            !OpenRouterProviderIdentity.equivalent(
                "Google AI Studio", "google-vertex"))

        let body = OpenRouterPairedJudge.requestBody(
            model: "google/gemini-3.6-flash",
            provider: "Google AI Studio", rubric: "r",
            structuredPrompt: nil, prompt: "p", responseA: "a",
            responseB: "b")
        let routing = body["provider"] as? [String: Any]
        #expect(routing?["order"] as? [String] == ["google-ai-studio"])
    }

    @Test func localJudgeGeneratesWithTheUnifiedCap() {
        // LocalPairedJudge passes `generationMaxTokens` to
        // ExperimentTasks.generate — the incident class was exactly a
        // smaller ad-hoc local-judge cap.
        #expect(LocalPairedJudge.generationMaxTokens == PairedJudgeBudget.maxTokens)
    }

    // MARK: - The golden-pinned wrapper

    /// Repo root derived from this file's compile-time path (same anchor
    /// as `GoldenRenderFixtureTests`).
    private static var goldenDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "paired-judge")
    }

    private func golden(_ name: String) throws -> String {
        let url = Self.goldenDirectory.appending(component: name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            Issue.record(
                Comment(
                    rawValue: "missing committed golden \(url.path) — the "
                        + "paired-judge wrapper is byte-pinned cross-engine; "
                        + "restore the fixture, never delete it"))
            throw PairedJudgeError(reason: "missing golden \(name)")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    @Test func wrapperMatchesTheFullGolden() throws {
        // Inputs identical to the Python twin test — the two engines'
        // builders are pinned to the SAME committed bytes.
        let built = PairedJudgePrompt.build(
            rubric: "Which response answers the task prompt with the sounder "
                + "legal reasoning? Dimensions: rigor, clarity.",
            structuredPrompt: "a_severity and b_severity: sentence severity "
                + "on a 1-9 ordinal scale.",
            prompt: "Draft the holding for Case 12, stating the disposition "
                + "in one paragraph.",
            responseA: "Response A body.\nSecond line of A.",
            responseB: "Response B body.")
        let expected = try golden("wrapper-full.golden.txt")
        #expect(
            built == expected,
            """
            paired-judge wrapper drifted from its committed golden — the \
            wrapper is byte-pinned cross-engine (the server's \
            paired_judge.build_prompt reads the same fixture); change it \
            only deliberately on BOTH engines and regenerate \
            prompts/fixtures/paired-judge/.
            golden: \(expected.debugDescription)
            built:  \(built.debugDescription)
            """)
    }

    @Test func wrapperMatchesTheMinimalGolden() throws {
        let built = PairedJudgePrompt.build(
            rubric: "", structuredPrompt: nil, prompt: "",
            responseA: "A text.", responseB: "B text.")
        let expected = try golden("wrapper-minimal.golden.txt")
        #expect(
            built == expected,
            Comment(
                rawValue: "paired-judge wrapper (minimal shape) drifted from "
                    + "its committed golden — see the wrapper-full assertion "
                    + "for the contract"))
    }

    @Test func wrapperDemandsABriefReason() {
        // The incident's root behavior: judges writing essays. The unified
        // wrapper must explicitly bound the reason and put the verdict
        // fields first.
        let built = PairedJudgePrompt.build(
            rubric: "r", prompt: "p", responseA: "a", responseB: "b")
        #expect(built.contains("at most two sentences"))
        #expect(built.contains("stating the verdict fields first"))
    }
}

/// Tiny async-safe call counter for the no-retry assertion.
private actor Counter {
    private(set) var value = 0
    func increment() { value += 1 }
}
