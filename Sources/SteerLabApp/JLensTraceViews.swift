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
    /// Server runs that actually hold a readout, so the id can be picked
    /// rather than typed from memory — the same `client.runs()` filter the
    /// Gemma Scope Reports section two sections up already uses.
    @State private var runs: [RemoteRunRecord] = []
    @State private var isLoadingRuns = false
    @State private var runsStatus: String?

    private var selectedRow: JLensTraceRow? {
        guard let trace else { return nil }
        return trace.rows.first { $0.id == selectedRowID } ?? trace.rows.first
    }

    /// A pasted id arrives with trailing whitespace often enough that not
    /// trimming it is a bug report waiting to happen.
    private var trimmedRunID: String {
        runID.trimmingCharacters(in: .whitespacesAndNewlines)
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
        .task(id: service.cluster.client?.profile.baseURL) { await loadRuns() }
        // Tokens were decoded once, for the FIRST generation only: every other
        // generation then showed raw integer ids in its predicted / top-k
        // columns and in the "Color by" picker.
        .onChange(of: selectedRowID) {
            watchIndex = 0
            guard let row = selectedRow else { return }
            Task { await decodeTokens(for: row) }
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

    /// Run picker first, free-text second. The status lines are normal rows:
    /// they used to be an `.overlay(.offset(y: 16))`, i.e. outside the
    /// HStack's layout bounds, so "run not found" painted over the next row
    /// instead of taking space.
    @ViewBuilder
    private var loader: some View {
        if !runs.isEmpty {
            HStack(spacing: 8) {
                // A typed id that is not in the list shows as "select…"
                // rather than as an invalid selection: the field below is the
                // fallback, and the two share one value.
                Picker("Run", selection: Binding(
                    get: { runs.contains { $0.id == runID } ? runID : "" },
                    set: { if !$0.isEmpty { runID = $0 } })) {
                    Text("select…").tag("")
                    ForEach(runs) { run in
                        Text(run.id).tag(run.id)
                    }
                }
                .help(
                    "server runs whose directory holds a jlens-readout.jsonl "
                        + "— picking one fills the field below")
                Button(isLoadingRuns ? "Refreshing…" : "Refresh") {
                    Task { await loadRuns() }
                }
                .controlSize(.small)
                .disabled(isLoadingRuns || service.cluster.client == nil)
                .help("re-ask the server which runs hold a readout")
            }
        }

        HStack {
            TextField("run id, e.g. 20260729T…-exp-…-run", text: $runID)
                .onSubmit { load() }
                .help(
                    "the SERVER run directory whose jlens-readout.jsonl is "
                        + "read — surrounding whitespace is ignored")
            Button("Load trace") { load() }
                .disabled(isLoading || trimmedRunID.isEmpty || service.cluster.client == nil)
                .help(
                    service.cluster.client == nil
                        ? "needs a server connection — the readout lives in the "
                            + "server's runs/"
                        : "fetch and parse that run's jlens-readout.jsonl; "
                            + "unparseable lines are counted, never dropped")
            if isLoading { ProgressView().controlSize(.small) }
        }

        if runs.isEmpty, !isLoadingRuns, service.cluster.client != nil {
            Text("no server run lists a jlens-readout.jsonl — type an id if "
                + "you have one")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let runsStatus {
            Label(runsStatus, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        if let status {
            Label(status, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    /// Completeness is stated before any number is shown. An incomplete trace is
    /// not a smaller finding — it is not a readout, and reading its cells as one
    /// is the mistake this line exists to prevent.
    ///
    /// Up to five items, so it wraps rather than truncating at the pane's
    /// 560 pt minimum.
    private func completeness(_ trace: JLensTrace) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                completenessVerdict(trace)
                completenessCounts(trace)
                completenessBadges
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    completenessVerdict(trace)
                    completenessBadges
                }
                completenessCounts(trace)
            }
        }
        .font(.caption)
    }

    @ViewBuilder
    private func completenessVerdict(_ trace: JLensTrace) -> some View {
        if trace.isComplete {
            Label("complete", systemImage: "checkmark.seal.fill")
                .foregroundStyle(.green)
        } else {
            Label("\(trace.incompleteCount) incomplete — NOT usable as a readout",
                  systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func completenessCounts(_ trace: JLensTrace) -> some View {
        HStack(spacing: 10) {
            Text(Self.plural(trace.rows.count, "generation")
                 + ", " + Self.plural(trace.totalObservations, "observation"))
                .foregroundStyle(.secondary)
            if malformed > 0 {
                Text(Self.plural(malformed, "unparseable line"))
                    .foregroundStyle(.orange)
            }
        }
    }

    /// The LENS's tier, and — separately — whether THIS row may be cited. A
    /// qualified lens over an agent with an unpinned adapter configuration is
    /// still an exploratory measurement, so the tier badge alone overstated
    /// reportability (round 7).
    @ViewBuilder
    private var completenessBadges: some View {
        if let tier = selectedRow?.evidenceTier {
            TierBadge(tier: tier, expanded: false)
        }
        if let row = selectedRow {
            ClaimBadge(claim: row.conditionClaim,
                       reason: row.unreportableReason)
        }
    }

    private static func plural(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }

    private func generationPicker(_ trace: JLensTrace) -> some View {
        Picker("Generation", selection: Binding(
            get: { selectedRowID ?? trace.rows.first?.id },
            set: { selectedRowID = $0 })) {
            ForEach(trace.rows) { row in
                Text(row.label).tag(Optional(row.id))
            }
        }
        .help(
            "which traced generation the heatmap and table below describe — "
                + "condition · prompt · sample")
    }

    private func watchedTokenPicker(_ row: JLensTraceRow) -> some View {
        let watchlist = row.watchlistTokenIDs ?? []
        return VStack(alignment: .leading, spacing: 4) {
            if watchlist.count > 1 {
                Picker("Color by", selection: $watchIndex) {
                    ForEach(Array(watchlist.enumerated()), id: \.offset) { index, id in
                        Text(pieces[String(id)].map { "\"\($0)\"" } ?? "token \(id)")
                            .tag(index)
                    }
                }
                .help(
                    "which watched token the heatmap's colour encodes — one "
                        + "token, so the colour means one thing")
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
        guard !isLoading else { return }
        let requested = trimmedRunID
        guard !requested.isEmpty else { return }
        isLoading = true
        status = nil
        Task {
            defer { isLoading = false }
            do {
                let (loaded, bad) = try await client.jlensTrace(runID: requested)
                trace = loaded
                malformed = bad
                pieces = [:]
                selectedRowID = loaded.rows.first?.id
                watchIndex = 0
                if let row = loaded.rows.first {
                    await decodeTokens(for: row)
                }
            } catch {
                trace = nil
                status = error.localizedDescription
            }
        }
    }

    /// Which server runs hold a readout. Same filter shape as the Gemma Scope
    /// Reports section; a failure is a message, not an empty picker.
    private func loadRuns() async {
        guard let client = service.cluster.client else { return }
        guard !isLoadingRuns else { return }
        isLoadingRuns = true
        runsStatus = nil
        defer { isLoadingRuns = false }
        do {
            runs = try await client.runs()
                .filter { $0.files.contains("jlens-readout.jsonl") }
        } catch {
            runsStatus =
                "could not list server runs: \(error.localizedDescription)"
        }
    }

    /// Token IDs are what the trace stores; a table of bare integers cannot be
    /// read. Decoded on demand rather than at write time, so no tokenizer's
    /// answer is baked into the durable record — and decoded again whenever
    /// the selected generation changes, merging into what is already known so
    /// switching back is free.
    private func decodeTokens(for row: JLensTraceRow) async {
        guard let client = service.cluster.client,
              let modelID = row.modelID else { return }
        var ids = Set(row.watchlistTokenIDs ?? [])
        for observation in (row.observations ?? []).prefix(200) {
            if let predicted = observation.predictedTokenID { ids.insert(predicted) }
            ids.formUnion((observation.topKIDs ?? []).prefix(3))
            ids.formUnion((observation.topKIDsLogitLens ?? []).prefix(3))
        }
        let known = Set(pieces.keys.compactMap(Int.init))
        let missing = ids.subtracting(known)
        guard !missing.isEmpty else { return }
        if let decoded = try? await client.jlensDecodeTokens(
            modelID: modelID, tokenIDs: Array(missing)) {
            pieces.merge(decoded.pieces) { _, new in new }
        }
    }
}

/// Layer × prediction-step grid. Colour encodes ONE watched token's score, so
/// the colour means one thing; a blended metric would be unreadable.
struct JLensHeatmapView: View {
    let heatmap: JLensHeatmap
    let pieces: [String: String]

    /// The grid is one row per layer, so a deep model makes it arbitrarily
    /// tall. It scrolls inside a box whose minimum AND maximum are both
    /// constants: neither moves with the data, which is what the macOS 27
    /// split-view minimum-height hazard is about (a two-axis ScrollView with
    /// no minimum collapses, so "no minimum at all" is not an option here).
    private static let minimumBoxHeight: CGFloat = 120
    private static let maximumBoxHeight: CGFloat = 220

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
                .fixedSize(horizontal: false, vertical: true)
            ScrollView([.horizontal, .vertical]) {
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
            .frame(
                minHeight: Self.minimumBoxHeight,
                maxHeight: Self.maximumBoxHeight)
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
///
/// Hand-laid rows rather than a `Table`, for two reasons the audit of
/// 2026-09-06 named: eight columns do not fit the Analysis pane's 560 pt
/// minimum (the logit-lens companions now hide behind a toggle), and a
/// `Table(minHeight:)` that appears with async state inside a split-view
/// column is the macOS 27 hazard this project keeps clear of. The rows scroll
/// inside a box whose height bounds are both constants.
struct JLensObservationTable: View {
    let row: JLensTraceRow
    let pieces: [String: String]

    @State private var showsLogitLens = false

    /// Constant minimum AND maximum: the box never resizes with the row
    /// count, which is the property the split-view hazard cares about, and
    /// the old `Table(minHeight: 220)` did not have.
    private static let minimumBoxHeight: CGFloat = 120
    private static let maximumBoxHeight: CGFloat = 240
    private static let shownLimit = 200

    private var shown: [JLensObservation] {
        Array((row.observations ?? []).prefix(Self.shownLimit))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            header
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    headerRow
                    Divider()
                    ForEach(shown) { observation in
                        dataRow(observation)
                    }
                }
                .padding(.vertical, 2)
            }
            .frame(
                minHeight: Self.minimumBoxHeight,
                maxHeight: Self.maximumBoxHeight)
            if (row.observations ?? []).count > shown.count {
                Text("showing \(shown.count) of \((row.observations ?? []).count) observations")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let reason = row.traceFailureReason {
                Text("trace failure: \(reason)")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Toggle("Logit-lens columns", isOn: $showsLogitLens)
                .toggleStyle(.checkbox)
                .font(.caption)
                .help(
                    "show the logit-lens companion beside each J-lens reading "
                        + "— hidden by default because eight columns do not fit "
                        + "this pane")
            CopyButton(
                "Copy as TSV",
                help: "copy the rows shown as tab-separated text (every "
                    + "column, including the logit-lens companions) for notes "
                    + "or a spreadsheet",
                text: { tsv() })
                .buttonStyle(.link)
                .font(.caption)
            Spacer()
        }
    }

    private var headerRow: some View {
        HStack(spacing: 6) {
            Text("layer").frame(width: 42, alignment: .trailing)
            Text("pass").frame(width: 56, alignment: .leading)
            Text("step").frame(width: 40, alignment: .trailing)
            Text("predicted").frame(minWidth: 76, maxWidth: .infinity, alignment: .leading)
            Text("watched (J)").frame(minWidth: 76, maxWidth: .infinity, alignment: .leading)
            Text("top-k (J)").frame(minWidth: 90, maxWidth: .infinity, alignment: .leading)
            if showsLogitLens {
                Text("watched (LL)")
                    .frame(minWidth: 76, maxWidth: .infinity, alignment: .leading)
                    .help("watched-token scores read through the logit lens")
                Text("top-k (LL)")
                    .frame(minWidth: 90, maxWidth: .infinity, alignment: .leading)
                    .help("top-k tokens read through the logit lens")
            }
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }

    private func dataRow(_ observation: JLensObservation) -> some View {
        HStack(spacing: 6) {
            Text("\(observation.layer)")
                .font(.caption.monospacedDigit())
                .frame(width: 42, alignment: .trailing)
            Text(observation.passKind ?? "—")
                .font(.caption)
                .frame(width: 56, alignment: .leading)
            Text("\(observation.predictedIndex)")
                .font(.caption.monospacedDigit())
                .frame(width: 40, alignment: .trailing)
            predictedCell(observation)
            valueCell(numbers(observation.watched), width: 76)
            valueCell(tokens(observation.topKIDs), width: 90)
            if showsLogitLens {
                valueCell(numbers(observation.watchedLogitLens),
                          width: 76, secondary: true)
                valueCell(tokens(observation.topKIDsLogitLens),
                          width: 90, secondary: true)
            }
        }
        .lineLimit(1)
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func predictedCell(_ observation: JLensObservation) -> some View {
        if observation.isUnaligned {
            Text("unaligned")
                .font(.caption)
                .foregroundStyle(.orange)
                .frame(minWidth: 76, maxWidth: .infinity, alignment: .leading)
                .help("the trace could not name the token this activation "
                      + "predicted, so the row is uninterpretable")
        } else {
            let text = piece(observation.predictedTokenID)
            Text(text)
                .font(.caption.monospaced())
                .frame(minWidth: 76, maxWidth: .infinity, alignment: .leading)
                .help(text)
        }
    }

    @ViewBuilder
    private func valueCell(
        _ text: String, width: CGFloat, secondary: Bool = false
    ) -> some View {
        let cell = Text(text)
            .font(.caption.monospaced())
        if secondary {
            cell.foregroundStyle(.secondary)
                .frame(minWidth: width, maxWidth: .infinity, alignment: .leading)
                .help(text)
        } else {
            cell.frame(minWidth: width, maxWidth: .infinity, alignment: .leading)
                .help(text)
        }
    }

    /// Every column, whether or not the logit-lens ones are on screen: a copy
    /// that silently dropped half the measurement would be worse than none.
    private func tsv() -> String {
        var lines = [[
            "layer", "pass", "step", "predicted", "watched (J)",
            "top-k (J)", "watched (logit lens)", "top-k (logit lens)",
        ].joined(separator: "\t")]
        for observation in shown {
            lines.append([
                "\(observation.layer)",
                observation.passKind ?? "",
                "\(observation.predictedIndex)",
                observation.isUnaligned
                    ? "unaligned" : piece(observation.predictedTokenID),
                numbers(observation.watched),
                tokens(observation.topKIDs),
                numbers(observation.watchedLogitLens),
                tokens(observation.topKIDsLogitLens),
            ].joined(separator: "\t"))
        }
        return lines.joined(separator: "\n")
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
