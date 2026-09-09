import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

struct ArtifactImportButton: View {
    let service: ChatService
    let kind: String
    var onImported: () -> Void = {}
    @State private var presented = false

    var body: some View {
        Button(kind == "jlens" ? "Import my own lens files…" : "Import my own SAE decoder files…") { presented = true }
            .sheet(isPresented: $presented) {
                ArtifactImportSheet(service: service, kind: kind, onImported: onImported)
            }
    }
}

private struct ArtifactImportSheet: View {
    let service: ChatService
    let kind: String
    let onImported: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var selection: ArtifactImportSelection?
    @State private var plan: ArtifactImportPlan?
    @State private var stagedDescription: String?
    @State private var reviewedClient: ClusterClient?
    @State private var reviewedWorkspace: ClusterConnectionStore.Workspace?
    @State private var status = ""
    @State private var busy = false
    @State private var jobID: String?

    private var isLens: Bool { kind == "jlens" }
    var body: some View {
        ScrollView { VStack(alignment: .leading, spacing: 16) {
            Text(isLens ? "Import a fitted J-lens" : "Import an SAE decoder feature").font(.title2)
            Text(isLens
                 ? "Add a lens fitted by you or another researcher. It becomes available in the J-lens library for readouts at its fitted layers. A full-depth lens can also create steering token vectors. No model training runs during import."
                 : "Add one feature’s decoder direction to the steering-vector library. This needs the fitted decoder and measured calibration for its model; it does not import a complete latent SAE or train one.")
                .fixedSize(horizontal: false, vertical: true)
            GroupBox("1. Choose source files") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Choose an import description (.json) supplied with the artifact or prepared with your coding agent. It identifies the model, layer mapping, tensor file, and source. Keep the tensor and any source configuration beside it, at the relative paths it names.")
                        .fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Button("Choose description…", action: choose).disabled(busy)
                        Text(selection?.description.lastPathComponent ?? "No description selected").foregroundStyle(.secondary)
                    }
                    Button("Copy instructions for my coding agent", action: copyInstructions)
                    if let selection {
                        Text("Declared model: \(selection.modelID)")
                        ForEach(selection.files, id: \.relativePath) { file in
                            Text(file.relativePath).font(.caption).textSelection(.enabled)
                        }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
            }
            GroupBox("2. Stage files and review") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("The selected files will be copied to the connected Python workbench for inspection. Your local originals remain in place. This may transfer a large tensor file; it does not download model weights or publish a vector or lens yet.")
                        .fixedSize(horizontal: false, vertical: true)
                    if let client = service.cluster.client, service.cluster.computeTarget == .server {
                        Text("Destination: \(client.profile.baseURL.absoluteString)").font(.caption)
                    } else {
                        Text("Connect a Python workbench in Compute to import here. It can run on this Mac. Agents can also import locally with the app-free client.").font(.caption)
                    }
                    Button("Stage selected files and review", action: stageAndReview)
                        .disabled(selection == nil || service.cluster.computeTarget != .server || busy)
                    if let plan {
                        Text("Model: \(plan.modelID)")
                        Text("Fit revision: \(plan.modelRevision ?? "Unknown — retained as unknown")")
                        Text("Workspace: \(plan.workspaceRoot)").font(.caption).textSelection(.enabled)
                        if isLens, case .string(let tier) = plan.details["tier"] {
                            Text("Intended use: \(tier == "evidence" ? "Study evidence" : "Testing and rehearsal")")
                            Text("This choice comes from lens.tier in the description and defaults to testing. Study use still needs qualification for the exact model runtime; import does not establish validity.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        DisclosureGroup("Layer mapping and conversion details") {
                            ForEach(plan.details.keys.sorted(), id: \.self) { key in
                                if let value = plan.details[key], let data = try? JSONEncoder().encode(value) {
                                    Text("\(detailLabel(key)): \(String(decoding: data, as: UTF8.self))").font(.caption).textSelection(.enabled)
                                }
                            }
                        }
                        ForEach(plan.warnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
            }
            HStack {
                Button(isLens ? "Import reviewed lens" : "Import reviewed feature", action: publish)
                    .disabled(plan == nil || busy)
                Button("Done") { dismiss() }.disabled(busy)
                if busy { ProgressView().controlSize(.small) }
            }
            Text(status).font(.callout).textSelection(.enabled)
        }
        .padding(24) }.frame(width: 760, height: 760)
        .interactiveDismissDisabled(busy)
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]; panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            selection = try ArtifactImportSelection.read(url, expectedKind: kind)
            plan = nil; stagedDescription = nil; reviewedClient = nil; reviewedWorkspace = nil
            status = "Source selected. Review the destination before transferring files."
        } catch { status = error.localizedDescription }
    }

    private func stageAndReview() {
        guard let selection, let client = service.cluster.client else { return }
        let origin = service.cluster.activeWorkspace
        busy = true; plan = nil; stagedDescription = nil; jobID = nil
        Task {
            defer { busy = false }
            do {
                let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
                for file in selection.files {
                    status = "Copying \(file.relativePath)…"
                    let path = try await client.stageArtifactSource(file, sourceID: id)
                    if file.relativePath == "description.json" { stagedDescription = path }
                }
                guard let stagedDescription else { throw ExperimentError(reason: "The source description was not staged.") }
                status = "Checking tensor geometry and source hashes…"
                plan = try await client.artifactImportPlan(descriptionFile: stagedDescription)
                reviewedClient = client
                reviewedWorkspace = origin
                status = "Review the model, layer mapping, and scaling above, then import when ready."
            } catch { status = error.localizedDescription }
        }
    }

    private func publish() {
        guard let plan, let client = reviewedClient, let path = stagedDescription else { return }
        let origin = reviewedWorkspace
        busy = true
        Task {
            defer { busy = false }
            do {
                let id = try await client.importReviewedArtifact(descriptionFile: path, planSHA256: plan.planSHA256)
                jobID = id; status = "Importing reviewed files…"
                guard let job = await service.followServerJobInActivity(jobID: id, client: client, title: "Custom artifact import") else {
                    status = "Import is still running. Follow its job in Compute."; return
                }
                guard job.status == "succeeded" else {
                    status = "Import \(job.status): \(job.error ?? "See the job log.")"; return
                }
                if service.cluster.activeWorkspace == origin {
                    _ = await service.catalog.refreshRemoteVectors()
                    onImported()
                }
                self.plan = nil
                status = "Imported into \(plan.workspaceRoot). " + (isLens
                    ? "Refresh the J-lens library for \(plan.modelID) and select the new lens."
                    : "Refresh Data → Concepts & Vectors or Playground → Steering Vectors for \(plan.modelID).")
            } catch { status = error.localizedDescription }
        }
    }

    private func copyInstructions() {
        let text = """
        Help me prepare a SteerLab \(kind) artifact import description from my existing files and their documentation.
        Read the repository's docs/GENERAL-ARTIFACT-IMPORTS.md for the exact JSON schema and examples.
        Identify the actual model, layer mapping, tensor key/layout, and any known fit revision. Ask me about missing metadata; do not guess it.
        Keep original files, use relative paths beside the description, and leave unknown modelRevision null.
        \(isLens ? "The lens must map declared source layers to the final block before final normalization. Ask whether I intend testing or study evidence, and set lens.tier to testing or evidence accordingly; this is separate from qualification." : "Import one residual-post SAE decoder feature, with explicit rows/columns orientation and a compatible measured calibrationArtifact from the destination workbench.")
        Do not train anything, generate data, download weights, or transfer files until we agree to those actions.
        """
        let guide = (try? ScienceCatalog.guide(isLens ? "jlens" : "sae").text) ?? ""
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text + "\n\n" + guide, forType: .string)
        status = "Instructions copied. Give them to your coding agent with the artifact files and source documentation."
    }

    private func detailLabel(_ key: String) -> String {
        ["tier": "Intended use", "sourceLayers": "Fitted source layers", "targetLayer": "Final target layer",
         "hiddenSize": "Residual-stream width", "promptsFitted": "Prompts used for fitting",
         "conversion": "Conversion", "layer": "Injection layer", "feature": "Feature number",
         "label": "Feature label", "rawDecoderNorm": "Original decoder length",
         "targetNorm": "Calibrated vector length", "calibrationModelRevision": "Calibration checkpoint"][key] ?? key
    }
}
