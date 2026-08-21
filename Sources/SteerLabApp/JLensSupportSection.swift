import ExperimentKit
import SwiftUI

/// "What vocabulary is this vector made of?" — the J-lens support readout.
///
/// A concept vector is a direction with a label a researcher chose. This reads
/// it back as a non-negative sparse combination of lens token directions, so the
/// vector is described in the model's own words. It applies to vectors already
/// in the library and needs no generation run.
///
/// Server-only and Gemma-only by rule (CLAUDE.md), like every other J-lens
/// surface: the atoms are PyTorch/HF-native and an MLX-extracted direction lives
/// in a different space.
///
/// **The energy figure is deliberately never shown alone.** A matched-norm
/// random direction reconstructs comparably in this dictionary — measured 3–17%
/// for real concept vectors against 11–32% for nulls on gemma-3-4b-it — so a
/// bare percentage invites a conclusion it cannot support. Every layer shows its
/// own control beside it, and the tokens are presented as the finding.
@MainActor
struct JLensSupportSection: View {
    @Bindable var service: ChatService

    @State private var catalog: JLensCatalog?
    @State private var selectedVectorID: String = ""
    @State private var budget: Int = 25
    @State private var layerText: String = ""
    @State private var readout: JLensSupportReadout?
    @State private var status: String?
    @State private var isBusy = false

    var body: some View {
        Section("Vector support — read a vector as tokens") {
            explanation
            if service.cluster.client == nil {
                Text("""
                     Needs a server connection. The lens is PyTorch/HF-native, so \
                     there is no local path — rather than degrade to something \
                     that looks like it worked.
                     """)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                controls
                if let status {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .task { await refreshCatalog() }

        if let readout {
            Section("Support of \(readout.vector.name)") {
                readoutHeader(readout)
                ForEach(readout.layers) { layer in
                    layerRow(layer, modelID: readout.modelID)
                }
            }
        }
    }

    // MARK: Explanation

    private var explanation: some View {
        Text("""
             Expresses a stored vector as a non-negative combination of lens token \
             directions and lists the tokens carrying it. Also a validity check: a \
             vector labelled "impartiality" whose support is affect words has a \
             problem no cross-concept cosine would surface.
             """)
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    // MARK: Controls

    /// Vectors this can actually decompose: on the server substrate, and on a
    /// model some imported lens was fitted on. Filtering here rather than
    /// failing later is the difference between an empty picker that explains
    /// itself and a job that refuses after submission.
    private var eligibleVectors: [RemoteVectorRecord] {
        let fitted = Set((catalog?.lenses ?? []).compactMap { $0.fit?.modelID })
        return service.catalog.remoteVectors
            .filter { fitted.contains($0.modelID) }
            // An MLX artifact in a shared tree carries "swift-mlx"; its
            // activations never met these atoms. The server refuses it too, but
            // an option that cannot work should not be offered.
            .filter { $0.substrate == nil || $0.substrate == "python-hf-transformers" }
            .sorted { $0.id < $1.id }
    }

    private func lensID(for vector: RemoteVectorRecord) -> String? {
        (catalog?.lenses ?? []).first { $0.fit?.modelID == vector.modelID }?.lensID
    }

    @ViewBuilder
    private var controls: some View {
        if eligibleVectors.isEmpty {
            Text("""
                 No server vector matches an imported lens. Import a lens for the \
                 model a vector was extracted on (J-Space → lens library), or \
                 extract a vector on a model that already has one.
                 """)
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Picker("Vector", selection: $selectedVectorID) {
                Text("Choose…").tag("")
                ForEach(eligibleVectors) { vector in
                    Text("\(vector.name) — \(vector.method ?? "?") "
                         + "(\(shortModel(vector.modelID)))")
                        .tag(vector.id)
                }
            }
            Stepper("Budget: \(budget) tokens", value: $budget, in: 1...200, step: 5)
                .help("how many atoms the reconstruction may use. 25 is the "
                      + "occupancy the workspace paper reports")
            TextField("Layers (blank = every fitted layer)", text: $layerText)
                .help("comma-separated, e.g. 11,17,29. Blank reads every fitted "
                      + "source layer: readability varies by layer and choosing "
                      + "one for you would hide that")
            HStack {
                Button("Read support") { Task { await read() } }
                    .disabled(isBusy || selectedVectorID.isEmpty)
                if isBusy { ProgressView().controlSize(.small) }
            }
        }
    }

    // MARK: Readout

    private func readoutHeader(_ readout: JLensSupportReadout) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            LabeledContent("Lens", value: readout.lensID)
            LabeledContent("Fitted on",
                           value: "\(readout.lensFitPrompts.map(String.init) ?? "?") prompts"
                           + (readout.lensFitCorpus.map { ", \($0)" } ?? ""))
            LabeledContent("Budget", value: "\(readout.budget) tokens")
            if let directory = readout.runDirectory {
                LabeledContent("Saved to", value: (directory as NSString).lastPathComponent)
                    .help(directory)
            }
            Text("""
                 Percentages below are shares of the reconstructed direction's \
                 length. The reconstructed fraction is shown against a \
                 matched-norm random direction through the same solver — on its \
                 own it separates nothing, so read the tokens.
                 """)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func layerRow(_ layer: JLensSupportLayer, modelID: String) -> some View {
        DisclosureGroup(isExpanded: .constant(true)) {
            VStack(alignment: .leading, spacing: 3) {
                ForEach(layer.support) { atom in
                    HStack(spacing: 8) {
                        Text(atom.piece)
                            .font(.system(.caption, design: .monospaced))
                            .frame(minWidth: 150, alignment: .leading)
                            .textSelection(.enabled)
                        ProgressView(value: min(max(atom.share, 0), 1))
                            .frame(width: 90)
                        Text(atom.share.formatted(.percent.precision(.fractionLength(1))))
                            .font(.caption2)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                        Text("id \(atom.tokenID)")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                if layer.support.isEmpty {
                    Text("no atom pointed into this direction at all")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.top, 2)
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("Layer \(layer.layer)").font(.caption).fontWeight(.semibold)
                Text(reconstructionSummary(layer))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if let exhausted = layer.coneExhaustedAt {
                    Text("cone exhausted at k=\(exhausted)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .help("no unused atom pointed into the residual. The atoms "
                              + "share a strong common component so their "
                              + "non-negative cone is narrow — a real ceiling, "
                              + "not a failure")
                }
            }
        }
    }

    /// Never a bare percentage: the control travels with it in the same string,
    /// so it cannot be read or screenshotted without its null.
    private func reconstructionSummary(_ layer: JLensSupportLayer) -> String {
        let fraction = layer.energyFraction.formatted(
            .percent.precision(.fractionLength(1)))
        let null = layer.nullEnergyFraction.formatted(
            .percent.precision(.fractionLength(1)))
        let sign = layer.beatsNull ? "+" : ""
        let margin = layer.energyOverNull.formatted(
            .percent.precision(.fractionLength(1)))
        return "reconstructed \(fraction) vs \(null) random (\(sign)\(margin))"
    }

    private func shortModel(_ modelID: String) -> String {
        modelID.split(separator: "/").last.map(String.init) ?? modelID
    }

    // MARK: Work

    private func refreshCatalog() async {
        guard let client = service.cluster.client else { return }
        catalog = try? await client.jlensCatalog()
    }

    private func read() async {
        guard let client = service.cluster.client,
              let vector = eligibleVectors.first(where: { $0.id == selectedVectorID }),
              let lens = lensID(for: vector)
        else { return }
        isBusy = true
        status = "submitting…"
        readout = nil
        defer { isBusy = false }

        let layers = layerText
            .split(whereSeparator: { ", ".contains($0) })
            .compactMap { Int($0) }
        do {
            let jobID = try await client.jlensSupport(
                lensID: lens, vectorID: vector.id,
                layers: layers.isEmpty ? nil : layers, budget: budget)
            readout = try await awaitReadout(client: client, jobID: jobID)
            status = readout == nil ? "job finished with no readout" : nil
        } catch {
            status = "failed: \(error.localizedDescription)"
        }
    }

    /// Poll the durable job and decode its result. The readout is the job's
    /// return value, so there is nothing to fetch separately.
    private func awaitReadout(client: ClusterClient,
                              jobID: String) async throws -> JLensSupportReadout? {
        while true {
            let job = try await client.job(jobID)
            switch job.status {
            case "succeeded", "finished", "completed":
                guard let result = job.result else { return nil }
                let data = try JSONEncoder().encode(result)
                return try JSONDecoder().decode(JLensSupportReadout.self, from: data)
            case "failed", "cancelled":
                throw NSError(
                    domain: "JLensSupport", code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                                job.error ?? "job \(job.status)"])
            default:
                status = job.logTail.last ?? "running…"
                try await Task.sleep(for: .milliseconds(600))
            }
        }
    }
}
