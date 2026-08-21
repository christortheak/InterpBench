import Foundation

/// A LABELLED estimate of prompt length for local readiness (C1).
///
/// Token counts belong to a specific tokenizer at a specific model revision.
/// The Mac cannot settle whether a prompt fits a server-only model — it does
/// not have that revision's tokenizer, and for a cluster study it never will.
/// So local readiness gets an estimate that says, in as many words, that it
/// is an estimate, and points at the exact check
/// (`POST /api/experiment/{name}/token-preflight`) that can settle it.
///
/// The alternative — presenting a character-derived number as if it were a
/// token count — is worse than presenting nothing, because a study that
/// "passed readiness" and then died on the cluster teaches the researcher to
/// distrust readiness generally.
public enum TokenBudgetHeuristic {

    /// Characters per token. English prose through a modern BPE tokenizer
    /// runs about 4; this uses a deliberately LOW divisor so the estimate
    /// leans high. A false "might not fit" costs one exact check; a false
    /// "fits" costs a cluster run.
    public static let charactersPerToken = 3.5

    /// Matches `ExperimentTasks.contextBudgetReserve` and the server's
    /// `CONTEXT_BUDGET_RESERVE`.
    public static let contextBudgetReserve = 16

    public static func estimatedTokens(_ text: String) -> Int {
        Int((Double(text.count) / charactersPerToken).rounded(.up))
    }

    /// The readiness verdict over a set of prompts, or nil when no context
    /// window is known for the model — in which case saying nothing is
    /// correct, since every number would be invented.
    public struct Estimate: Sendable, Equatable {
        public var itemCount: Int
        public var longestEstimatedTokens: Int
        public var promptBudget: Int
        /// Items whose ESTIMATE exceeds the budget. Not a count of failures —
        /// a count of items worth checking exactly.
        public var itemsOverEstimate: Int

        public var detail: String {
            let base = "\(itemCount) prompt\(itemCount == 1 ? "" : "s"); longest "
                + "≈\(longestEstimatedTokens) tokens against a \(promptBudget)-token "
                + "budget. ESTIMATED from character length (≈"
                + "\(charactersPerToken.formatted(.number.precision(.fractionLength(1)))) "
                + "chars/token) — only the pinned revision's tokenizer can "
                + "settle it"
            guard itemsOverEstimate > 0 else {
                return base + "; run the server's token preflight to confirm."
            }
            return base + ". \(itemsOverEstimate) item"
                + "\(itemsOverEstimate == 1 ? "" : "s") may not fit — run the "
                + "server's token preflight before submitting."
        }
    }

    /// The declared context window for a model id, from the app's own model
    /// catalog. Nil for anything not listed — including every model a
    /// cluster might host that this Mac has never seen, which is exactly why
    /// the exact check lives on the server.
    @MainActor
    public static func declaredContextWindow(modelID: String) -> Int? {
        ChatService.availableModels
            .first { $0.id == modelID }?.contextWindow
    }

    public static func estimate(
        texts: [String], contextWindow: Int?, maxTokens: Int
    ) -> Estimate? {
        guard let contextWindow, contextWindow > 0, !texts.isEmpty else {
            return nil
        }
        let budget = contextWindow - maxTokens - contextBudgetReserve
        guard budget > 0 else {
            return Estimate(
                itemCount: texts.count,
                longestEstimatedTokens: texts.map(estimatedTokens).max() ?? 0,
                promptBudget: max(0, budget),
                itemsOverEstimate: texts.count)
        }
        let counts = texts.map(estimatedTokens)
        return Estimate(
            itemCount: texts.count,
            longestEstimatedTokens: counts.max() ?? 0,
            promptBudget: budget,
            itemsOverEstimate: counts.count { $0 > budget })
    }
}
