import Foundation

/// A row keeps its typed refusal and repair as well as the human-readable
/// failure. Operational failures remain distinguishable from admission gates.
public struct StudyBatchIssue: Sendable, Equatable, Encodable {
    public let state: SteerLabCLIState
    public let code: String
    public let reason: String
    public let repairAction: String

    init(_ error: Error) {
        if let design = error as? StudyDesignAuthoringError {
            state = .refused; code = design.code; reason = design.reason; repairAction = design.repairAction
        } else if let experiment = error as? ExperimentError {
            reason = experiment.reason
            if let gate = experiment.lifecycleRefusal {
                state = .refused; code = gate.gate.rawValue; repairAction = gate.repairAction
            } else if let gate = experiment.freezeRefusal {
                state = .refused; code = gate.gate.rawValue; repairAction = gate.repairAction
            } else if let malformed = experiment.malformedInvocation {
                state = .blocked; code = "usage"; repairAction = malformed.repairAction
            } else {
                state = .failed; code = "batchRowFailed"
                repairAction = "Inspect this row's inputs and the recorded successful studies before retrying only the failed rows."
            }
        } else if (error as? CocoaError)?.code == .fileReadNoSuchFile {
            state = .notFound; code = "batchInputNotFound"; reason = error.localizedDescription
            repairAction = "Inspect this row's design and artifact paths in the originating workspace."
        } else {
            state = .failed; code = "batchRowFailed"; reason = error.localizedDescription
            repairAction = "Inspect this row's inputs and file access before retrying only the failed rows."
        }
    }
}
