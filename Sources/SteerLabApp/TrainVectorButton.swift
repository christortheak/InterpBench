import ExperimentKit
import SwiftUI

/// All entry points author through the same maintained interview and client owner.
struct TrainVectorButton: View {
    let service: ChatService
    var operation = "optvec-train"
    var title = "Train a vector…"
    @State private var workflow: ScienceCatalog.Workflow?
    @State private var failure: String?
    @State private var presented = false
    @State private var root = ExperimentStore.workspaceRoot
    @State private var client: ClusterClient?

    var body: some View {
        VStack(alignment: .leading) {
            Button(title) {
                do {
                    workflow = try ScienceCatalog.workflows().first { $0.id == operation }
                    guard workflow != nil else { failure = "Training form unavailable in this build."; return }
                    root = ExperimentStore.workspaceRoot
                    client = service.cluster.client
                    presented = true
                } catch { failure = error.localizedDescription }
            }
            if let failure { Text(failure).font(.caption).foregroundStyle(.red) }
        }
        .sheet(isPresented: $presented) {
            if let workflow { MethodAuthoringSheet(workflow: workflow, root: root, client: client) }
        }
    }
}
