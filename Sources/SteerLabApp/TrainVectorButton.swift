import ExperimentKit
import Foundation
import SwiftUI

/// All entry points author through the same maintained interview and client owner.
struct TrainVectorButton: View {
    let service: ChatService
    var operation = "optvec-train"
    var title = "Train a vector…"
    var root: URL? = nil

    /// One presentation value captures the selected operation and its origin.
    /// A Boolean plus an optional workflow can present before the workflow is
    /// visible to the sheet closure, producing an empty first presentation.
    private struct Request: Identifiable {
        let workflow: ScienceCatalog.Workflow
        let root: URL
        let client: ClusterClient?
        var id: String { workflow.id }
    }
    @State private var request: Request?
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading) {
            Button(title) {
                do {
                    guard let workflow = try ScienceCatalog.workflows().first(where: { $0.id == operation }) else {
                        failure = "Training form unavailable in this build."
                        return
                    }
                    failure = nil
                    request = Request(workflow: workflow, root: root ?? ExperimentStore.workspaceRoot,
                        client: service.cluster.computeTarget == .server ? service.cluster.client : nil)
                } catch { failure = error.localizedDescription }
            }
            if let failure { Text(failure).font(.caption).foregroundStyle(.red) }
        }
        .sheet(item: $request) { context in
            MethodAuthoringSheet(workflow: context.workflow, root: context.root, client: context.client)
        }
    }
}
