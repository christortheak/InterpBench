import Foundation

/// The study page's custom-code notice: what the selected study carries, and
/// the acknowledgement its button records. The owner is `CustomCodeNotice`,
/// the same one both command lines use, so the app and the command lines read
/// and write one record.
extension ExperimentPanel {
    /// What the page shows for a study: the review (nil when the study carries
    /// no custom code) or the problem that kept it from being read.
    public struct CustomCodeState: Sendable, Equatable {
        public let review: CustomCodeNotice.Review?
        public let problem: String?
    }

    public func customCodeState(for name: String) -> CustomCodeState {
        customCodeState(for: name, workspaceRoot: ExperimentStore.workspaceRoot)
    }

    func customCodeState(for name: String, workspaceRoot: URL) -> CustomCodeState {
        do {
            return CustomCodeState(
                review: try CustomCodeNotice.review(study: name, workspaceRoot: workspaceRoot), problem: nil)
        } catch let error as ExperimentError {
            // A damaged record still shows the notice, for every provider.
            let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(name)
            let document = (try? JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))) ?? .null
            let found = CustomCodeNotice.providers(in: document, workspaceRoot: workspaceRoot)
            return CustomCodeState(
                review: found.isEmpty ? nil : CustomCodeNotice.Review(study: name, providers: found.map {
                    .init(sha256: $0.sha256, policyNames: $0.policyNames, sourceText: $0.sourceText,
                          acknowledgedAt: nil, acknowledgedBy: nil)
                }),
                problem: CustomCodeNotice.line(error))
        } catch {
            return CustomCodeState(review: nil, problem: nil)
        }
    }

    /// Record the acknowledgement for exactly the code the page showed. Code
    /// the study no longer carries is refused by the owner, so a page that went
    /// stale cannot acknowledge something nobody looked at.
    @discardableResult
    public func acknowledgeCustomCode(_ review: CustomCodeNotice.Review) -> Bool {
        acknowledgeCustomCode(review, workspaceRoot: ExperimentStore.workspaceRoot)
    }

    @discardableResult
    func acknowledgeCustomCode(_ review: CustomCodeNotice.Review, workspaceRoot: URL) -> Bool {
        do {
            let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(review.study)
            let document = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
            try CustomCodeNotice.acknowledge(
                review.pending.map(\.sha256), in: document, study: review.study,
                workspaceRoot: workspaceRoot, client: "SteerLab app")
            note(
                "Recorded your acknowledgement of the custom code in '\(review.study)' "
                    + "in \(CustomCodeNotice.fileName).",
                severity: .success)
            return true
        } catch {
            note(
                "Couldn't record the acknowledgement, so the study still won't be sent to run. "
                    + "Details: \(error)",
                severity: .error)
            return false
        }
    }
}
