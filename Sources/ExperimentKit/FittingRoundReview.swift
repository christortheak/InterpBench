import Foundation

/// A retained review is historical context. Only its current approval hash can
/// authorize an action, and the server still rechecks that hash at submission.
public struct FittingRoundReview: Sendable {
    public enum Kind: String, Sendable { case queue, merge }

    public struct Snapshot: Sendable {
        public let jobID: String
        public let kind: Kind
        public let document: JSONValue
        public let lines: [String]

        public var label: String { "Last reviewed \(kind.rawValue) plan for round \(jobID) — not live status" }
    }

    public private(set) var snapshot: Snapshot?
    public private(set) var planSHA256: String?
    public var confirmed = false

    public init() {}

    public mutating func record(_ document: JSONValue, jobID: String, kind: Kind) {
        snapshot = Snapshot(jobID: jobID, kind: kind, document: document,
                            lines: FittingReviewSummary.capacityLines(document) + ScientificGPUPlacement.reviewLines(document))
        if case .object(let object) = document, case .string(let hash) = object["planSHA256"], !hash.isEmpty {
            planSHA256 = hash
        } else { planSHA256 = nil }
        confirmed = false
    }

    public func hash(for kind: Kind) -> String? {
        snapshot?.kind == kind ? planSHA256 : nil
    }

    /// Selection changes, attempted mutations, and errors retire approval, but
    /// leave the review available to explain the researcher's last decision.
    public mutating func invalidate() {
        planSHA256 = nil
        confirmed = false
    }

    public mutating func changeJob() { self = Self() }
}
