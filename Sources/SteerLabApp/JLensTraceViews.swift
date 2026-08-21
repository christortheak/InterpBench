import ExperimentKit
import SwiftUI

/// Trace viewer: the layer × prediction-step heatmap and the prediction-aligned
/// table, over a SERVER-produced `jlens-readout.jsonl`.
///
/// The grid arithmetic lives in `ExperimentKit.JLensHeatmap` rather than in a
/// View body, because the interesting part of a heatmap is its normalization and
/// a bug there produces a picture that looks plausible and is wrong. Here we
/// only draw what that type computed.
@MainActor
struct JLensTraceSection: View {
    @Bindable var service: ChatService

    @State private var runID: String = ""
    @State private var trace: JLensTrace?
    @State private var malformed = 0
    @State private var selectedRowID: JLensTraceRow.ID?
    @State private var watchIndex = 0
    @State private var pieces: [String: String] = [:]
    @State private var status: String?
    @State private var isLoading = false

    private var selectedRow: JLensTraceRow? {
        guard let trace else { return nil }
        return trace.rows.first { $0.id == selectedRowID } ?? trace.rows.first
    }

    var body: some View {
        Section("Readout trace") {
            explanation
            loader
            if let trace, !trace.rows.isEmpty {
                completeness(trace)
                if trace.rows.count > 1 { generationPicker(trace) }
                if let row = selectedRow {
                    watchedTokenPicker(row)
                    JLensHeatmapView(
                        heatmap: JLensHeatmap.build(row: row, watchlistIndex: watchIndex),
                        pieces: pieces)
                    JLensObservationTable(row: row, pieces: pieces)
                }
            }
        }
    }

    private var explanation: some View {
        Text("""
             Prediction-aligned: each row is the activation that predicted that \
             generated token, read during the same forward pass. Not a replay — \
             a post-hoc reconstruction would read a residual the model never had \
             at that step.
             """)
            .font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var loader: some View {
        HStack {
            TextField("run id, e.g. 20260729T…-exp-…-run", text: $runID)
                .onSubmit { load() }
            Button("Load trace") { load() }
                .disabled(isLoading || runID.isEmpty || service.cluster.client == nil)
            if isLoading { ProgressView().controlSize(.small) }
        }
        .overlay(alignment: .bottomLeading) {
            if let status {
                Text(status).font(.caption).foregroundStyle(.orange).offset(y: 16)
            }
        }
    }

    /// Completeness is stated before any number is shown. An incomplete trace is
    /// not a smaller finding — it is not a readout, and reading its cells as one
    /// is the mistake this line exists to prevent.
    private func completeness(_ trace: JLensTrace) -> some View {
        HStack(spacing: 10) {
            if trace.isComplete {
                Label("complete", systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
            } else {
                Label("\(trace.incompleteCount) incomplete — NOT usable as a readout",
                      systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
            Text("\(trace.rows.count) generation(s), \(trace.totalObservations) observation(s)")
                .foregroundStyle(.secondary)
            if malformed > 0 {
                Text("\(malformed) unparseable line(s)").foregroundStyle(.orange)
            }
            // The LENS's tier, and — separately — whether THIS row may be
            // cited. A qualified lens over an agent with an unpinned
            // adapter configuration is still an exploratory measurement, so
            // the tier badge alone overstated reportability (round 7).
            if let tier = selectedRow?.evidenceTier {
                TierBadge(tier: tier, expanded: false)
            }
            if let row = selectedRow {
                ClaimBadge(claim: row.conditionClaim,
                           reason: row.unreportableReason)
            }
        }
        .font(.caption)
    }

    private func generationPicker(_ trace: JLensTrace) -> some View {
        Picker("Generation", selection: Binding(
            get: { selectedRowID ?? trace.rows.first?.id },
            set: { selectedRowID = $0 })) {
            ForEach(trace.rows) { row in
                Text(row.label).tag(Optional(row.id))
            }
        }
    }

    private func watchedTokenPicker(_ row: JLensTraceRow) -> some View {
        let watchlist = row.watchlistTokenIDs ?? []
        return VStack(alignment: .leading, spacing: 4) {
            if watchlist.count > 1 {
                Picker("Colour by", selection: $watchIndex) {
                    ForEach(Array(watchlist.enumerated()), id: \.offset) { index, id in
                        Text(pieces[String(id)].map { "\"\($0)\"" } ?? "token \(id)")
                            .tag(index)
                    }
                }
            }
            if let steering = row.steering, !steering.isEmpty {
                // Without the arming, a steered row and a baseline row are
                // indistinguishable after the fact.
                Text("arming: " + steering.map(\.summary).joined(separator: ", "))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            } else {
                Text("arming: baseline (no injection)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
        }
    }

    private func load() {
        guard let client = service.cluster.client else { return }
        isLoading = true
        status = nil
        Task {
            defer { isLoading = false }
            do {
                let (loaded, bad) = try await client.jlensTrace(runID: runID)
                trace = loaded
                malformed = bad
                selectedRowID = loaded.rows.first?.id
                watchIndex = 0
                await decodeVisibleTokens(loaded, client: client)
            } catch {
                trace = nil
                status = error.localizedDescription
            }
        }
    }

    /// Token IDs are what the trace stores; a table of bare integers cannot be
    /// read. Decoded on demand rather than at write time, so no tokenizer's
    /// answer is baked into the durable record.
    private func decodeVisibleTokens(_ trace: JLensTrace,
                                     client: ClusterClient) async {
        guard let row = trace.rows.first(where: { $0.id == selectedRowID })
                ?? trace.rows.first,
              let modelID = row.modelID else { return }
        var ids = Set(row.watchlistTokenIDs ?? [])
        for observation in (row.observations ?? []).prefix(200) {
            if let predicted = observation.predictedTokenID { ids.insert(predicted) }
            ids.formUnion((observation.topKIDs ?? []).prefix(3))
            ids.formUnion((observation.topKIDsLogitLens ?? []).prefix(3))
        }
        guard !ids.isEmpty else { return }
        if let decoded = try? await client.jlensDecodeTokens(
            modelID: modelID, tokenIDs: Array(ids)) {
            pieces = decoded.pieces
        }
    }
}

/// Layer × prediction-step grid. Colour encodes ONE watched token's score, so
/// the colour means one thing; a blended metric would be unreadable.
struct JLensHeatmapView: View {
    let heatmap: JLensHeatmap
    let pieces: [String: String]

    private var caption: String {
        let token = heatmap.watchedTokenID.map { id in
            pieces[String(id)].map { "\"\($0)\"" } ?? "token \(id)"
        } ?? "watched token"
        guard let low = heatmap.minimum, let high = heatmap.maximum else {
            return "layer × prediction step — \(token)"
        }
        return String(format: "layer × prediction step — %@, range %.1f…%.1f",
                      token, low, high)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(caption).font(.caption).foregroundStyle(.secondary)
            Text("dashed outline = a watched token was already mentioned at that step (primed)")
                .font(.caption2).foregroundStyle(.tertiary)
            ScrollView(.horizontal) {
                Grid(horizontalSpacing: 1, verticalSpacing: 1) {
                    GridRow {
                        Text("layer").font(.caption2.monospaced())
                            .foregroundStyle(.secondary).frame(width: 38)
                        ForEach(heatmap.steps, id: \.self) { step in
                            Text("\(step)").font(.caption2.monospaced())
                                .foregroundStyle(.secondary).frame(width: 22)
                        }
                    }
                    ForEach(heatmap.layers, id: \.self) { layer in
                        GridRow {
                            Text("\(layer)").font(.caption2.monospaced())
                                .frame(width: 38)
                            ForEach(heatmap.steps, id: \.self) { step in
                                cellView(heatmap.cell(layer: layer, step: step))
                            }
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    @ViewBuilder
    private func cellView(_ cell: JLensHeatmap.Cell?) -> some View {
        let intensity = cell?.intensity
        RoundedRectangle(cornerRadius: 2)
            .fill(Color.accentColor.opacity((intensity ?? 0) * 0.85 + (intensity == nil ? 0 : 0.10)))
            .frame(width: 22, height: 18)
            .overlay {
                if intensity == nil {
                    Text("·").font(.caption2).foregroundStyle(.tertiary)
                }
                if cell?.mentionPrimed == true {
                    RoundedRectangle(cornerRadius: 2)
                        .strokeBorder(Color.orange, style: StrokeStyle(lineWidth: 1, dash: [2, 2]))
                }
            }
            .help(tooltip(cell))
    }

    private func tooltip(_ cell: JLensHeatmap.Cell?) -> String {
        guard let cell, let value = cell.value else { return "no observation" }
        var text = String(format: "layer %d, step %d: %.2f", cell.layer, cell.step, value)
        if let id = cell.predictedTokenID {
            text += " — predicted \(pieces[String(id)].map { "\"\($0)\"" } ?? "\(id)")"
        }
        if cell.mentionPrimed { text += " (mention-primed)" }
        return text
    }
}

/// The prediction-aligned table. Every row names the token that activation
/// predicted, and the logit-lens companion sits beside the J-lens reading so
/// "did transport do any work" is answerable by eye.
struct JLensObservationTable: View {
    let row: JLensTraceRow
    let pieces: [String: String]

    private var shown: [JLensObservation] {
        Array((row.observations ?? []).prefix(200))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Table(shown) {
                TableColumn("layer") { Text("\($0.layer)").font(.caption.monospaced()) }
                    .width(46)
                TableColumn("pass") { Text($0.passKind ?? "—").font(.caption) }
                    .width(58)
                TableColumn("step") { Text("\($0.predictedIndex)").font(.caption.monospaced()) }
                    .width(42)
                TableColumn("predicted") { observation in
                    if observation.isUnaligned {
                        Text("unaligned").font(.caption).foregroundStyle(.orange)
                    } else {
                        Text(piece(observation.predictedTokenID)).font(.caption.monospaced())
                    }
                }
                TableColumn("watched (J)") { Text(numbers($0.watched)).font(.caption.monospaced()) }
                TableColumn("watched (logit lens)") {
                    Text(numbers($0.watchedLogitLens)).font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                TableColumn("top-k (J)") { Text(tokens($0.topKIDs)).font(.caption.monospaced()) }
                TableColumn("top-k (logit lens)") {
                    Text(tokens($0.topKIDsLogitLens)).font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .frame(minHeight: 220)
            if (row.observations ?? []).count > shown.count {
                Text("showing \(shown.count) of \((row.observations ?? []).count) observations")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let reason = row.traceFailureReason {
                Text("trace failure: \(reason)").font(.caption).foregroundStyle(.orange)
            }
        }
    }

    private func piece(_ id: Int?) -> String {
        guard let id else { return "—" }
        return pieces[String(id)].map { "\"\($0)\"" } ?? "\(id)"
    }

    private func numbers(_ values: [Double]?) -> String {
        guard let values, !values.isEmpty else { return "—" }
        return values.map { String(format: "%.1f", $0) }.joined(separator: ", ")
    }

    private func tokens(_ ids: [Int]?) -> String {
        guard let ids, !ids.isEmpty else { return "—" }
        return ids.prefix(3).map(Optional.some).map(piece).joined(separator: " ")
    }
}
