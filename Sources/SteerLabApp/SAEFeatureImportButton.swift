import ExperimentKit
import SwiftUI

struct SAEFeatureImportButton: View {
    let service: ChatService
    @State private var presented = false
    var body: some View {
        Button("Import an SAE feature…") { presented = true }
            .sheet(isPresented: $presented) {
                if service.cluster.computeTarget == .server, let client = service.cluster.client {
                    SAEFeatureImportSheet(service: service, client: client,
                        modelID: service.workspaceSelectedModelID ?? "")
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Import an SAE feature").font(.title2)
                        Text("Direct feature import currently uses the Python engine. Connect one in Compute; it can run on this Mac using MPS. Select the matching model there, then return to import your feature. Existing local Gemma Scope reports also offer their own feature-import action.")
                        Button("Done") { presented = false }
                    }.padding().frame(width: 540)
                }
            }
    }
}

private struct SAEFeatureImportSheet: View {
    let service: ChatService
    let client: ClusterClient
    let modelID: String
    @Environment(\.dismiss) private var dismiss
    @State private var link = ""
    @State private var release = ""
    @State private var dictionary = ""
    @State private var feature = ""
    @State private var label = ""
    @State private var donor = ""
    @State private var donors: [RemoteVectorRecord] = []
    @State private var status = ""
    @State private var busy = false
    @State private var jobID: String?
    @State private var review: ClusterClient.SAEFeatureImport?
    @State private var confirming = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add an SAE feature to the vector library").font(.title2)
            Text("Model: " + modelID)
            Text("Use a feature's decoder direction as an intervention. Its description is a hypothesis to test; importing it does not validate its behavioral effect.")
                .font(.caption)
            Form {
                TextField("Neuronpedia feature link (optional)", text: $link)
                Button("Look up feature identifiers") { lookup() }.disabled(link.isEmpty)
                TextField("SAE release", text: $release)
                TextField("SAE dictionary ID", text: $dictionary)
                TextField("Feature number", text: $feature)
                TextField("Your label", text: $label)
                Picker("Steering-scale calibration", selection: $donor) {
                    Text("Choose compatible measured calibration…").tag("")
                    ForEach(donors.filter { $0.modelID == modelID && $0.residualNormPerLayer != nil }) { vector in
                        Text(vector.concept).tag(vector.id)
                    }
                }
                Text("Calibration supplies this model's typical activation size, not a semantic reference vector. If none is listed, measure norms for a compatible artifact in Concepts & Vectors first. The import owner checks layer, dimensions and model identity.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(busy)
            HStack {
                Button("Review import") {
                    guard let id = Int(feature), id >= 0, !modelID.isEmpty,
                          !release.isEmpty, !dictionary.isEmpty, !label.isEmpty, !donor.isEmpty else {
                        status = "Choose a model, release, dictionary, nonnegative feature number, label and calibration."; return
                    }
                    review = .init(model: modelID, release: release, saeID: dictionary,
                        feature: id, label: label, residualNormArtifact: donor,
                        neuronpediaURL: link.isEmpty ? nil : link)
                    confirming = true
                }.disabled(busy)
                Button("Done") { dismiss() }.disabled(busy)
                if busy { ProgressView().controlSize(.small) }
                if busy, let jobID {
                    Button("Request cancellation") { Task {
                        do { try await client.cancelJob(jobID); status = "Cancellation requested; waiting for the engine." }
                        catch { status = error.localizedDescription }
                    } }
                }
            }
            Text(status).font(.caption).textSelection(.enabled)
        }.padding().frame(minWidth: 680, minHeight: 500)
        .interactiveDismissDisabled(busy)
        .confirmationDialog("Import this feature?", isPresented: $confirming, titleVisibility: .visible, presenting: review) { reviewed in
            Button("Import feature") { submit(reviewed) }
            Button("Cancel", role: .cancel) {}
        } message: { review in
                Text("\(review.model)\n\(review.release) / \(review.saeID)\nFeature \(review.feature), label: \(review.label)\nCalibration: \(review.residualNormArtifact)\nThis may download SAE weights. The engine saves a new vector scaled to the measured residual norm.")
        }
        .task {
            do { donors = try await client.vectorArtifacts() }
            catch { status = "Could not list calibration artifacts: " + error.localizedDescription }
        }
    }
    private func lookup() {
        busy = true; status = "Looking up the installed SAE directory…"
        Task {
            defer { busy = false }
            do {
                let found = try await client.resolveSAEFeature(url: link)
                release = found.release; dictionary = found.saeID; feature = String(found.feature)
                status = "Directory model: \(found.model). Confirm it corresponds to the selected base model. No weights downloaded."
            } catch { status = error.localizedDescription }
        }
    }
    private func submit(_ request: ClusterClient.SAEFeatureImport) {
        let origin = service.cluster.activeWorkspace
        busy = true; status = "Submitting feature import…"
        Task {
            defer { busy = false }
            do {
                let id = try await client.importSAEFeature(request)
                jobID = id
                let job = await service.followServerJobInActivity(jobID: id, client: client, title: "SAE feature import")
                guard let job else { status = "Job \(id) is still running. Follow it in Compute."; return }
                guard job.status == "succeeded" else { status = "Import \(job.status): \(job.error ?? "See the job log in Compute.")"; return }
                let refreshed = service.cluster.activeWorkspace == origin
                    ? await service.catalog.refreshRemoteVectors() : false
                status = "Feature saved on \(client.profile.baseURL.absoluteString). "
                    + (refreshed ? "Find '\(request.label)' in Data → Concepts & Vectors or Playground → Steering Vectors for \(request.model)."
                        : "Return to that execution workspace and refresh its vector library. \(service.catalog.remoteVectorsError ?? "")")
            } catch { status = error.localizedDescription }
        }
    }
}
