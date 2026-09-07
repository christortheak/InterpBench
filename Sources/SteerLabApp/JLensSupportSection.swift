import ExperimentKit
import SwiftUI

/// "What vocabulary is this vector made of?" — the J-lens support readout.
///
/// A concept vector is a direction with a label a researcher chose. This reads
/// it back as a non-negative sparse combination of lens token directions, so the
/// vector is described in the model's own words. It applies to vectors already
/// in the library and needs no generation run.
///
/// Server-only by rule, like every other J-lens
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
    /// Catalog fetch state, kept apart from the readout's `status`: a failed
    /// fetch used to be swallowed by `try?` and then rendered as "no server
    /// vector matches an imported lens", which is a different claim.
    @State private var catalogStatus: String?
    @State private var isLoadingCatalog = false

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
                catalogRow
                controls
                if let status {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
        }
        // A change of server used to leave this section telling the user to
        // import a lens forever: the fetch ran once and never again.
        .task(id: service.cluster.client?.profile.baseURL) { await refreshCatalog() }

        if let readout {
            Section("Support of \(readout.vector.name)") {
                readoutHeader(readout)
                ForEach(readout.layers) { layer in
                    JLensSupportLayerRow(
                        layer: layer, summary: reconstructionSummary(layer))
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

    /// Refresh + the catalog's own failure. Rendered above the controls, not
    /// inside their non-empty branch: the moment a researcher most needs to
    /// re-ask the server is when the list came back empty.
    @ViewBuilder
    private var catalogRow: some View {
        HStack(spacing: 8) {
            Button(isLoadingCatalog ? "Refreshing…" : "Refresh Lenses") {
                Task { await refreshCatalog() }
            }
            .controlSize(.small)
            .disabled(isLoadingCatalog)
            .help(
                "re-ask the server which lenses are imported — do this after "
                    + "importing one in the J-Space section above")
            if isLoadingCatalog { ProgressView().controlSize(.small) }
            Spacer()
        }
        if let catalogStatus {
            Label(catalogStatus, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private var controls: some View {
        if eligibleVectors.isEmpty {
            // Only an honest claim when the fetch actually succeeded; the
            // failure case says so above instead.
            Text("""
                 No server vector matches an imported lens. Import a lens for the \
                 model a vector was extracted on — the J-Space section above: \
                 Acquire, then Import — or extract a vector on a model that \
                 already has one.
                 """)
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Picker("Vector", selection: $selectedVectorID) {
                Text("select…").tag("")
                ForEach(eligibleVectors) { vector in
                    Text("\(vector.name) — \(vector.method ?? "?") "
                         + "(\(shortModel(vector.modelID)))")
                        .tag(vector.id)
                }
            }
            .help(
                "which stored vector is decomposed — only vectors on a model "
                    + "some imported lens was fitted on are listed")
            Stepper("Budget: \(budget) tokens", value: $budget, in: 1...200, step: 5)
                .help("how many atoms the reconstruction may use; 25 is the "
                      + "default and is enough for the token list to be read "
                      + "as a description")
            TextField("Layers (blank = every fitted layer)", text: $layerText)
                .help("comma-separated, e.g. 11,17,29. Blank reads every fitted "
                      + "source layer: readability varies by layer and choosing "
                      + "one for you would hide that")
            layerEcho
            HStack {
                Button("Read support") {
                    guard !isBusy else { return }
                    Task { await read() }
                }
                .disabled(isBusy || selectedVectorID.isEmpty || layerSelection.isInvalid)
                .help(readSupportHelp)
                if isBusy { ProgressView().controlSize(.small) }
            }
        }
    }

    private var readSupportHelp: String {
        if layerSelection.isInvalid {
            return "fix the Layers field first — it is not a list of layer "
                + "numbers"
        }
        if selectedVectorID.isEmpty { return "pick a vector first" }
        return "queue a durable job on the server that solves this vector "
            + "against the lens atoms — followed in Activity, and the readout "
            + "lands below when it finishes"
    }

    /// What the Layers field will actually mean, echoed before the submit.
    /// `compactMap { Int($0) }` used to drop every unparseable token, so
    /// "L11" or "11;17" submitted an EMPTY list — which the server reads as
    /// "every fitted layer", the opposite of what was typed.
    private enum LayerSelection {
        case everyFittedLayer
        case layers([Int])
        case invalid(String)

        var isInvalid: Bool { if case .invalid = self { true } else { false } }
    }

    private var layerSelection: LayerSelection {
        let tokens = layerText
            .split(whereSeparator: { ", ".contains($0) })
            .map(String.init)
        guard !tokens.isEmpty else { return .everyFittedLayer }
        var parsed: [Int] = []
        for token in tokens {
            guard let value = Int(token), value >= 0 else { return .invalid(token) }
            parsed.append(value)
        }
        return .layers(parsed)
    }

    @ViewBuilder
    private var layerEcho: some View {
        switch layerSelection {
        case .everyFittedLayer:
            Text("blank — reads every fitted source layer")
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .layers(let layers):
            Text("reads layer\(layers.count == 1 ? "" : "s") "
                 + layers.map(String.init).joined(separator: ", "))
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .invalid(let token):
            Label(
                "not a layer number: \"\(token)\" — use comma-separated "
                    + "integers, e.g. 11,17,29 (blank reads every fitted layer)",
                systemImage: "exclamationmark.triangle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
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
                // The full path, selectable: the last component alone cannot
                // be copied into a shell or a note.
                LabeledContent("Saved to") {
                    Text(directory)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                        .help(directory)
                }
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

    /// A failed fetch is a message, not an empty catalog: `try?` turned a
    /// server error into "no server vector matches an imported lens", which
    /// sent researchers off to import a lens they already had.
    private func refreshCatalog() async {
        guard let client = service.cluster.client else { return }
        guard !isLoadingCatalog else { return }
        isLoadingCatalog = true
        catalogStatus = nil
        defer { isLoadingCatalog = false }
        do {
            catalog = try await client.jlensCatalog()
        } catch {
            catalogStatus =
                "could not list the server's lenses: \(error.localizedDescription)"
        }
    }

    private func read() async {
        guard !isBusy,
              let client = service.cluster.client,
              let vector = eligibleVectors.first(where: { $0.id == selectedVectorID }),
              let lens = lensID(for: vector)
        else { return }
        // Validated, never coerced: an unparseable Layers field used to
        // submit "every fitted layer".
        let layers: [Int]?
        switch layerSelection {
        case .everyFittedLayer: layers = nil
        case .layers(let parsed): layers = parsed
        case .invalid(let token):
            status = "Layers: \"\(token)\" is not a layer number — nothing was "
                + "submitted"
            return
        }
        isBusy = true
        status = "submitting…"
        readout = nil
        defer { isBusy = false }

        do {
            let jobID = try await client.jlensSupport(
                lensID: lens, vectorID: vector.id,
                layers: layers, budget: budget)
            status = "server job \(jobID) — live log in Activity"
            // The same hand-off the server Gemma Scope run uses: the live log
            // lives in Activity, where the job is visible and cancellable, and
            // the poll is no longer an untracked Task of this view's.
            guard let job = await service.followServerJobInActivity(
                jobID: jobID, client: client,
                title: "J-lens support: \(vector.name)")
            else {
                status = "job \(jobID) is still running server-side — open "
                    + "Compute to keep watching"
                return
            }
            switch job.status {
            case "succeeded", "finished", "completed":
                readout = try decodeReadout(job)
                status = readout == nil ? "job finished with no readout" : nil
            default:
                status = "job \(jobID) \(job.status): "
                    + (job.error ?? job.logTail.last ?? "no detail")
            }
        } catch {
            status = "failed: \(error.localizedDescription)"
        }
    }

    /// The readout is the job's return value, so there is nothing to fetch
    /// separately.
    private func decodeReadout(_ job: RemoteJobRecord) throws -> JLensSupportReadout? {
        guard let result = job.result else { return nil }
        let data = try JSONEncoder().encode(result)
        return try JSONDecoder().decode(JLensSupportReadout.self, from: data)
    }
}

/// One layer's atoms. Its own view so the disclosure has REAL state: the
/// chevron used to be bound to `.constant(true)` and did nothing when clicked.
private struct JLensSupportLayerRow: View {
    let layer: JLensSupportLayer
    let summary: String

    @State private var isExpanded = true

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
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
                Text(summary)
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
}
