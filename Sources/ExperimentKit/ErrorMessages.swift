import Foundation

/// Retroactive `LocalizedError` conformance for every error type in
/// ExperimentKit that carries its own message.
///
/// Swift's `Error.localizedDescription` does NOT consult
/// `CustomStringConvertible`. A type that describes itself perfectly through
/// `description` still renders as "The operation couldn't be completed.
/// (ExperimentKit.ChatServiceError error 1.)" the moment any caller reaches
/// for `localizedDescription` — which SwiftUI, Foundation, and several of
/// our own catch-all handlers do by default. The message is not degraded,
/// it is DESTROYED: the reason string never appears.
///
/// Observed live 2026-07-26: a cluster study submission failed and the
/// researcher was shown the placeholder above, with the real cause — which
/// the throwing code had written out in full — discarded at the last hop
/// (`SubstrateRouting`'s catch-all uses `localizedDescription`).
///
/// Conformances are collected here rather than added to 25 declarations so
/// the invariant is checkable in one place, and so a new error type that
/// forgets it is visible as an absence from this list.

extension ChatServiceError: LocalizedError {
    public var errorDescription: String? { description }
}

extension ClusterClient.MissingVariantDependency: LocalizedError {
    public var errorDescription: String? { description }
}

extension CodeResourceError: LocalizedError {
    public var errorDescription: String? { description }
}

extension ExperimentError: LocalizedError {
    public var errorDescription: String? { description }
}

extension ExperimentTasks.ContextBudgetError: LocalizedError {
    public var errorDescription: String? { description }
}

extension FactorialDesign.Problem: LocalizedError {
    public var errorDescription: String? { description }
}

extension FineTuneTrainingData.Problem: LocalizedError {
    public var errorDescription: String? { description }
}

extension FineTuneTrainingError: LocalizedError {
    public var errorDescription: String? { description }
}

extension GemmaScopeAnalysisError: LocalizedError {
    public var errorDescription: String? { description }
}

extension GemmaScopeReportImportError: LocalizedError {
    public var errorDescription: String? { description }
}

extension GeometryAnalysisError: LocalizedError {
    public var errorDescription: String? { description }
}

extension NormBackfill.BackfillError: LocalizedError {
    public var errorDescription: String? { description }
}

extension PairedJudgeError: LocalizedError {
    public var errorDescription: String? { description }
}

extension ScenarioDiagnostics.UnequalInputs: LocalizedError {
    public var errorDescription: String? { description }
}

extension SmokeTestFailure: LocalizedError {
    public var errorDescription: String? { description }
}

extension StimulusGeneratorError: LocalizedError {
    public var errorDescription: String? { description }
}

extension TabularImport.Problem: LocalizedError {
    public var errorDescription: String? { description }
}

extension TaskPromptsDocument.ParseError: LocalizedError {
    public var errorDescription: String? { description }
}

extension TaskPromptsImport.ImportError: LocalizedError {
    public var errorDescription: String? { description }
}

extension ToyConceptFailure: LocalizedError {
    public var errorDescription: String? { description }
}

extension VectorParity.ParityError: LocalizedError {
    public var errorDescription: String? { description }
}
