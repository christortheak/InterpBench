import ExperimentKit
import SwiftUI

struct ServerNeutralBasisControls: View {
    @Bindable var service: ChatService
    @State private var showingBuild = false
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Projection basis", selection: Binding(
                get: { service.serverNeutralPCBasisPath },
                set: { service.selectServerNeutralBasis(path: $0) })) {
                Text("None — no component removal").tag(String?.none)
                ForEach(service.serverNeutralCatalog?.bases.filter { $0.modelID == service.workspaceSelectedModelID } ?? []) { basis in
                    Text(basis.label).tag(String?.some(basis.id))
                }
                if let path = service.serverNeutralPCBasisPath,
                   service.serverNeutralCatalog?.bases.contains(where: { $0.id == path }) != true {
                    Text("Selected basis: \(service.serverNeutralPCBasisLabel ?? path)").tag(String?.some(path))
                }
            }
            Text("Optionally remove reference components from active vectors before steering. None keeps the original directions. This changes subsequent generation, not saved vector files.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Refresh bases") { Task { await service.refreshServerNeutralBases() } }
                Button("Build projection basis…") { showingBuild = true }
            }
            if let problem = service.serverNeutralBasisProblem {
                Text(problem).font(.caption).foregroundStyle(.orange)
            }
            if let error = service.serverNeutralCatalogError {
                Text("Could not list reference artifacts: " + error).font(.caption).foregroundStyle(.secondary)
            }
        }
        .task(id: service.cluster.activeWorkspace) { await service.refreshServerNeutralBases() }
        .sheet(isPresented: $showingBuild) {
            if let client = service.cluster.client {
                ServerNeutralBuildSheet(service: service, client: client, modelID: service.workspaceSelectedModelID ?? "")
            }
        }
    }
}

struct ServerNeutralBuildSheet: View {
    let service: ChatService
    let client: ClusterClient
    let modelID: String
    @Environment(\.dismiss) private var dismiss
    @State private var corpora: [RemoteNeutralCatalog.Corpus] = []
    @State private var corpus = "norm"
    @State private var status = ""
    @State private var jobID: String?
    @State private var busy = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Build a projection basis").font(.title2)
            Text("Model: " + modelID)
            Text("Run this model on reference examples already present in this execution workspace. The result contains recurring activation directions; removal is optional and can remove useful signal. Keep your original data in the authoring workspace.")
            Picker("Reference dataset", selection: $corpus) {
                ForEach(corpora) { item in Text("\(item.name) · \(item.count) examples").tag(item.name) }
            }.disabled(busy)
            Text("Capture: all layers. Select components explaining 50% of variation, up to the engine's component cap per layer. This can take substantial memory and time.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Build on this engine") { build() }
                    .disabled(busy || modelID.isEmpty || (corpora.first(where: { $0.name == corpus })?.count ?? 0) < 4)
                if let jobID, busy {
                    Button("Request cancellation") { Task {
                        do { try await client.cancelJob(jobID); status = "Cancellation requested; waiting for the engine." }
                        catch { status = error.localizedDescription }
                    } }
                }
                Button("Done") { dismiss() }.disabled(busy)
            }
            Text(status).font(.caption).textSelection(.enabled)
        }.padding().frame(minWidth: 600, minHeight: 330)
        .interactiveDismissDisabled(busy)
        .task {
            do { corpora = try await client.neutralCatalog().corpora }
            catch { status = error.localizedDescription }
        }
    }
    private func build() {
        busy = true; status = "Submitting projection build…"
        Task {
            defer { busy = false }
            do {
                let id = try await client.buildNeutralBasis(corpus: corpus, modelID: modelID)
                jobID = id
                let job = await service.followServerJobInActivity(jobID: id, client: client, title: "Projection basis")
                status = job.map { $0.status == "succeeded"
                    ? "Basis saved. Select it in Playground to test component removal."
                    : "Build \($0.status). \($0.error ?? "See the job log in Compute.")" }
                    ?? "The job is still running; follow it in Compute."
                await service.refreshServerNeutralBases()
            } catch { status = error.localizedDescription }
        }
    }
}
