import Foundation
import MLXLMCommon

enum LocalPairedJudge {
    /// The unified judge generation cap (`PairedJudgeBudget.maxTokens`,
    /// server twin `paired_judge.JUDGE_MAX_TOKENS`) — exposed as its own
    /// name so a test can assert this is what local judge generation
    /// actually passes. The 2026-07-22 incident cap (the server's old 512)
    /// truncated a legible verdict mid-reasoning; the old 768 here had the
    /// same failure mode.
    static let generationMaxTokens = PairedJudgeBudget.maxTokens

    static func judge(
        container: ModelContainer,
        modelID: String,
        rubric: String,
        structuredPrompt: String?,
        prompt: String,
        responseA: String,
        responseB: String
    ) async throws -> PairedJudgeResponse {
        let response = try await ExperimentTasks.generate(
            container,
            prompt: PairedJudgePrompt.build(
                rubric: rubric, structuredPrompt: structuredPrompt, prompt: prompt,
                responseA: responseA, responseB: responseB),
            modelID: modelID,
            maxTokens: generationMaxTokens,
            temperature: 0)
        // The shared verdict parser (cross-engine twin of the server's
        // `paired_judge.parse_response`): balanced-JSON extraction plus the
        // 2026-07-22 truncation salvage — a legibly-complete winner whose
        // reasoning outran the cap is accepted with `reasoningTruncated`.
        return try PairedJudgeVerdictParser.parse(response)
    }
}
