import Foundation

/// Retroactive `LocalizedError` conformance for every error type in
/// SteeringKit that carries its own message.
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

extension ConceptExtractorError: LocalizedError {
    public var errorDescription: String? { description }
}

extension RepEReader.ReaderError: LocalizedError {
    public var errorDescription: String? { description }
}

extension SteeringVectorStore.NonFiniteArtifactError: LocalizedError {
    public var errorDescription: String? { description }
}

extension StimulusSetError: LocalizedError {
    public var errorDescription: String? { description }
}
