import Foundation
import Testing

@testable import ExperimentKit

/// C1 (local half) — a LABELLED estimate, never a claim.
///
/// Token counts belong to a specific tokenizer at a specific model revision.
/// The Mac does not have a cluster-only model's tokenizer and never will, so
/// local readiness can only estimate. Presenting a character-derived number as
/// if it were a token count would be worse than presenting nothing: a study
/// that "passed readiness" and then died on the cluster teaches the researcher
/// to distrust readiness generally.
struct TokenBudgetHeuristicTests {

    @Test func theEstimateLeansHigh() {
        // A false "might not fit" costs one exact check; a false "fits" costs
        // a cluster run. So the divisor is below the ~4 chars/token that
        // English prose actually runs at.
        #expect(TokenBudgetHeuristic.charactersPerToken < 4.0)
        // 350 characters ÷ 3.5 = 100.
        #expect(TokenBudgetHeuristic.estimatedTokens(String(repeating: "x", count: 350)) == 100)
        // Partial tokens round UP.
        #expect(TokenBudgetHeuristic.estimatedTokens("xxxx") == 2)
    }

    @Test func theBudgetSubtractsGenerationAndReserve() throws {
        let estimate = try #require(
            TokenBudgetHeuristic.estimate(
                texts: [String(repeating: "x", count: 350)],
                contextWindow: 1000, maxTokens: 200))
        #expect(estimate.promptBudget == 1000 - 200 - 16)
        #expect(estimate.longestEstimatedTokens == 100)
        #expect(estimate.itemsOverEstimate == 0)
    }

    @Test func itCountsTheItemsWorthCheckingExactly() throws {
        let short = String(repeating: "x", count: 35)     // ≈10 tokens
        let long = String(repeating: "x", count: 3500)    // ≈1000 tokens
        let estimate = try #require(
            TokenBudgetHeuristic.estimate(
                texts: [short, long, long], contextWindow: 500, maxTokens: 100))
        #expect(estimate.itemCount == 3)
        #expect(estimate.itemsOverEstimate == 2)
    }

    @Test func theDetailAlwaysSaysItIsAnEstimateAndNamesTheExactCheck() throws {
        let estimate = try #require(
            TokenBudgetHeuristic.estimate(
                texts: [String(repeating: "x", count: 3500)],
                contextWindow: 500, maxTokens: 100))
        #expect(estimate.detail.contains("ESTIMATED"))
        #expect(estimate.detail.contains("token preflight"))
        #expect(estimate.detail.contains("pinned revision's tokenizer"))
    }

    @Test func anUnknownContextWindowSaysNothingAtAll() {
        // Every number would be invented. Silence is the honest output.
        #expect(
            TokenBudgetHeuristic.estimate(
                texts: ["hello"], contextWindow: nil, maxTokens: 100) == nil)
        #expect(
            TokenBudgetHeuristic.estimate(
                texts: [], contextWindow: 1000, maxTokens: 100) == nil)
    }

    @Test func aGenerationBudgetLargerThanTheWindowFlagsEverything() throws {
        let estimate = try #require(
            TokenBudgetHeuristic.estimate(
                texts: ["a", "b"], contextWindow: 100, maxTokens: 200))
        #expect(estimate.promptBudget == 0)
        #expect(estimate.itemsOverEstimate == 2)
    }

    @Test @MainActor func theCatalogSuppliesWindowsForKnownModelsOnly() {
        #expect(
            TokenBudgetHeuristic.declaredContextWindow(
                modelID: "mlx-community/gemma-3-27b-it-8bit") == 131_072)
        // A cluster-only model this Mac has never seen — exactly why the
        // exact check lives on the server.
        #expect(
            TokenBudgetHeuristic.declaredContextWindow(
                modelID: "some-org/unseen-model") == nil)
    }

    @Test func theReserveMatchesTheEnginesRuntimeReserve() {
        // Drift here would make the estimate disagree with the check it is
        // estimating (generate.CONTEXT_BUDGET_RESERVE / contextBudgetReserve).
        #expect(TokenBudgetHeuristic.contextBudgetReserve == 16)
    }
}
