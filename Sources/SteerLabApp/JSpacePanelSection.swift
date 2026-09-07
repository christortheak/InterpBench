import ExperimentKit
import SwiftUI

/// J-Space: the app's control and rendering surface over SERVER lens artifacts.
///
/// Server-only by rule (hard requirement). Imported
/// lens artifacts are PyTorch/HF-native and activations do not transfer across
/// substrates, so this panel deliberately offers no local/MLX path: on a
/// workspace that does not compute on a server it says so and stops, rather
/// than degrading to something that would look like it worked.
///
/// Lives inside Analysis ("vector geometry and mechanistic analysis") rather
/// than as its own section — a J-lens readout is a mechanistic instrument, and
/// the app's own taxonomy already has a home for those.
@MainActor
struct JSpacePanelSection: View {
    @Bindable var service: ChatService

    @State private var catalog: JLensCatalog?
    @State private var selectedModelID: String = ""
    @State private var selectedLensID: String?
    @State private var detail: JLensRecord?
    @State private var status: String?
    @State private var isBusy = false

    @State private var tokenQuery: String = ""
    @State private var includeCaseVariants = false
    @State private var tokenOptions: JLensTokenOptions?
    @State private var selectedToken: JLensTokenCandidate?
    @State private var deriveName: String = ""
    @State private var deriveResult: String?

    /// The gate is the WORKSPACE's compute target, not `cluster.client`:
    /// the store always builds a client (it falls back to loopback), so the
    /// old `client == nil` test could never fire and a Local workspace
    /// rendered the whole server surface and then failed against 127.0.0.1
    /// (audit 2026-09-06, headline 10). Same test `FineTuningPanelView` uses
    /// for its server route.
    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    /// Rendered as Form sections so it composes into the Analysis panel rather
    /// than becoming a second layout language beside it.
    var body: some View {
        Section("J-Space — Jacobian lens") {
            header
            if isServerWorkspace {
                lensLibrary
            } else {
                noServerNotice
            }
        }
        .task { await refresh() }
        if isServerWorkspace {
            Section("Token → direction") {
                tokenDirectionBuilder
            }
            JLensSupportSection(service: service)
            JLensTraceSection(service: service)
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("""
                 Reads what the model is poised to verbalize, and derives steering \
                 directions from a lens. Server-only: lens artifacts \
                 are PyTorch/HF-native and activations do not transfer across \
                 substrates, so everything here runs on the connected server. \
                 Scores are estimated verbalizable representations — never access \
                 to beliefs or reasoning.
                 """)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let status {
                Text(status).font(.caption.monospaced()).foregroundStyle(.secondary)
            }
        }
    }

    /// No local fallback on purpose: a J-lens direction derived on this machine
    /// would be meaningless steering that no existing check would catch.
    private var noServerNotice: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                Label("Needs a server workspace", systemImage: "bolt.horizontal.circle")
                    .font(.callout.bold())
                Text(
                    "This workspace computes on \(service.cluster.substrateLabel), "
                        + "and J-Space has no local equivalent by design. Point "
                        + "the workspace at a server in Compute — the lens, its "
                        + "derivations, and its readouts all live on the "
                        + "substrate that can actually run them.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
        }
    }

    // MARK: Lens library

    private var lensLibrary: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let supported = catalog?.supported, !supported.isEmpty {
                HStack(spacing: 10) {
                    Picker("Model", selection: $selectedModelID) {
                        ForEach(supported) { model in
                            Text("\(model.modelID) — \(model.tier)").tag(model.modelID)
                        }
                    }
                    .frame(maxWidth: 380)
                    .help(
                        "which model's published lens the two buttons act on — "
                        + "a lens only reads the model it was fitted for")
                    Button("Acquire lens") { run(.acquire) }
                        .disabled(isBusy || selectedModelID.isEmpty)
                        .help("fetch the published lens bytes into the server's HF cache")
                    Button("Import lens") { run(.importLens) }
                        .disabled(isBusy || selectedModelID.isEmpty)
                        .help("convert a cached lens into the server workspace")
                    if isBusy { ProgressView().controlSize(.small) }
                }
                if let tier = supported.first(where: { $0.modelID == selectedModelID }),
                   !tier.isEvidenceTier {
                    TierBadge(tier: tier.tier, expanded: true)
                }
            } else if catalog != nil {
                // Empty `supported` used to hide the picker and leave every
                // button permanently disabled with no explanation.
                Text(
                    "this server publishes no supported lens models, so there "
                        + "is nothing to acquire, import, or derive from here — "
                        + "the catalog's supported list is empty")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            let lenses = catalog?.lenses ?? []
            if lenses.isEmpty {
                Text("No lenses imported on this server yet — Acquire, then Import.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(lenses) { lens in
                    lensRow(lens)
                }
            }
            if let detail { provenance(detail) }
        }
    }

    /// A real Button, not a tap gesture: the row is focusable, keyboard
    /// reachable, and announced as a selectable control (audit 2026-09-06).
    private func lensRow(_ lens: JLensRecord) -> some View {
        let isSelected = selectedLensID == lens.lensID
        let tier = catalog?.supported
            .first(where: { $0.modelID == lens.fit?.modelID })?.tier ?? "testing"
        return Button {
            select(lens)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(lens.lensID).font(.callout.monospaced())
                    HStack(spacing: 8) {
                        Text("layers \(lens.layerSpan)")
                        if let converted = lens.converted?.dtype {
                            Text("converted \(converted)").foregroundStyle(.green)
                        } else {
                            Text("not converted").foregroundStyle(.orange)
                        }
                        let passing = lens.passingQualifications.count
                        Text(passing > 0 ? "\(passing) qualification(s)" : "unqualified")
                            .foregroundStyle(passing > 0 ? .green : .secondary)
                    }
                    .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                TierBadge(tier: tier, expanded: false)
            }
            .padding(8)
            .background(isSelected ? Color.accentColor.opacity(0.10) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(
            "select this lens — its full provenance loads below, and Derive "
                + "uses the model it was fitted for")
        .accessibilityLabel("Lens \(lens.lensID)")
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    /// Full provenance, not a summary. The fit-time revision is UNKNOWN for the
    /// published artifacts and has to stay visibly unknown wherever it appears —
    /// substituting the runtime's would relabel an absence of evidence as
    /// evidence.
    private func provenance(_ lens: JLensRecord) -> some View {
        GroupBox("Provenance") {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                row("source", "\(lens.source?.repo ?? "?")/\(lens.source?.folder ?? "?")")
                row("upstream commit", lens.source?.commit ?? "—")
                row("tensor sha256", String((lens.source?.tensorSHA256 ?? "—").prefix(16)))
                row("fit model", lens.fit?.modelID ?? "—")
                row("fit revision",
                    (lens.fit?.revisionKnown ?? false)
                        ? (lens.fit?.revision ?? "—")
                        : "unknown — the published configs pin none")
                row("fit corpus", lens.fit?.corpus ?? "—")
                row("prompts fitted", lens.fit?.promptsFitted.map(String.init) ?? "—")
                row("converted", lens.converted.map {
                    "\($0.layerCount ?? 0) layers, \($0.dtype ?? "?")" } ?? "—")
                row("reference", "\(lens.referencePackage ?? "?") @ "
                    + String((lens.referenceCommit ?? "?").prefix(12)))
                row("direction", lens.directionConvention ?? "—")
                row("readout", lens.readoutConvention ?? "—")
                row("substrate", lens.substrate ?? "—")
            }
            .font(.caption.monospaced())
            .padding(6)
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
    }

    // MARK: Token → direction

    private var tokenDirectionBuilder: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("""
                 A direction is indexed by ONE exact vocabulary token. Nothing here \
                 picks for you: a word that is not a single token shows its \
                 components, and choosing one derives a direction for that component.
                 """)
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                TextField("word or string, e.g. courage", text: $tokenQuery)
                    .frame(maxWidth: 260)
                    .onSubmit { lookUpTokens() }
                    .help(
                        "the word or string to look up in the model's "
                            + "vocabulary — the server tokenizes it and lists "
                            + "every candidate")
                    .accessibilityLabel("Word or string to tokenize")
                Toggle("case variants", isOn: $includeCaseVariants)
                    .toggleStyle(.checkbox)
                    .help(
                        "also list the capitalized and lower-cased forms — they "
                            + "are different vocabulary entries with different "
                            + "directions")
                Button("Show token options") { lookUpTokens() }
                    .disabled(isBusy || tokenQuery.isEmpty || selectedModelID.isEmpty)
                    .help("ask the server for every vocabulary entry this string produces")
            }
            if let options = tokenOptions {
                ForEach(options.candidates) { candidate in
                    candidateRow(candidate)
                }
            }
            HStack {
                TextField("name (optional)", text: $deriveName)
                    .frame(maxWidth: 240)
                    .help(
                        "what the derived direction is called in the vector "
                            + "catalog — left empty, the server names it from "
                            + "the token")
                    .accessibilityLabel("Derived direction name")
                Button("Derive direction on server") { run(.derive) }
                    .disabled(
                        isBusy || selectedToken == nil || selectedLensID == nil
                            || deriveModelMismatch != nil)
                    .help(
                        "runs on the connected server against the selected "
                            + "lens's own model; the result appears in the "
                            + "ordinary vector catalog")
                if isBusy { ProgressView().controlSize(.small) }
            }
            // The token IDs are looked up against the picker's model; a lens
            // fitted for another model cannot use them (audit 2026-09-06).
            if let mismatch = deriveModelMismatch {
                Text(mismatch)
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let deriveResult {
                Text(deriveResult).font(.caption.monospaced())
                    .foregroundStyle(.green).textSelection(.enabled)
            }
        }
    }

    /// A real Button for the same reason the lens rows are: focusable,
    /// keyboard reachable, announced as selectable (audit 2026-09-06).
    private func candidateRow(_ candidate: JLensTokenCandidate) -> some View {
        let isSelected = selectedToken?.tokenID == candidate.tokenID
            && selectedToken?.form == candidate.form
        return Button {
            selectedToken = candidate
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 10) {
                    Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                        .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                    Text(String(candidate.tokenID)).font(.callout.monospaced())
                    Text(candidate.form).font(.caption).foregroundStyle(.secondary)
                    if candidate.singleToken {
                        Label("single token", systemImage: "checkmark.circle")
                            .font(.caption).foregroundStyle(.green).labelStyle(.titleAndIcon)
                    } else {
                        Label("multi-token", systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange).labelStyle(.titleAndIcon)
                    }
                    Text(candidate.decoded.map { "\"\($0)\"" } ?? candidate.piece)
                        .font(.callout.monospaced())
                    Spacer()
                }
                // Bytes are always shown: a vocabulary entry need not be printable,
                // and two entries can render identically.
                Text("bytes \(candidate.decodedBytes)")
                    .font(.caption2.monospaced()).foregroundStyle(.tertiary)
                if let note = candidate.note {
                    Text(note).font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(8)
            .background(isSelected ? Color.accentColor.opacity(0.10) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(
            "derive the direction for THIS vocabulary entry — a direction is "
                + "indexed by one exact token, never by the word")
        .accessibilityLabel("Token \(candidate.tokenID), \(candidate.form)")
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    // MARK: Derive target

    /// The model the SELECTED LENS was fitted for. A lens only reads the
    /// model it was fitted against, so this — not the Model picker — is what
    /// a derive runs on (audit 2026-09-06).
    private var selectedLensFitModelID: String? {
        guard let selectedLensID else { return nil }
        let record: JLensRecord? =
            detail?.lensID == selectedLensID
            ? detail
            : catalog?.lenses.first(where: { $0.lensID == selectedLensID })
        let fitted = record?.fit?.modelID ?? ""
        return fitted.isEmpty ? nil : fitted
    }

    private var deriveModelID: String {
        selectedLensFitModelID ?? selectedModelID
    }

    /// The token candidates were resolved against the Model picker's
    /// tokenizer; applying those ids to a lens fitted for another model is
    /// meaningless, so the derive is refused with the repair.
    private var deriveModelMismatch: String? {
        guard let fitted = selectedLensFitModelID, !selectedModelID.isEmpty,
            fitted != selectedModelID, selectedToken != nil
        else { return nil }
        return
            "the selected lens was fitted for \(fitted), but these token "
            + "options came from \(selectedModelID)'s tokenizer — switch the "
            + "Model picker to \(fitted) and look the token up again"
    }

    // MARK: Actions

    private enum Action { case acquire, importLens, derive }

    private func select(_ lens: JLensRecord) {
        selectedLensID = lens.lensID
        detail = lens
        Task {
            guard let client = service.cluster.client else { return }
            do {
                detail = try await client.jlensLens(id: lens.lensID)
            } catch {
                // The Provenance box would otherwise show the catalog's
                // summary record — fit revision, corpus, prompts fitted all
                // "—" — with nothing saying the full record failed to load.
                status =
                    "showing the catalog summary only — the full record for "
                    + "\(lens.lensID) did not load: \(error.localizedDescription)"
            }
        }
    }

    private func refresh() async {
        // Never fetch a lens catalog from a Local workspace's loopback URL:
        // the panel is server-only, and the notice above says so.
        guard isServerWorkspace, let client = service.cluster.client else { return }
        do {
            let fetched = try await client.jlensCatalog()
            catalog = fetched
            if selectedModelID.isEmpty {
                // Default to the EVIDENCE-tier model when one is offered: the
                // testing tier is for exercising the path, not for work.
                selectedModelID = fetched.supported.first(where: \.isEvidenceTier)?.modelID
                    ?? fetched.supported.first?.modelID ?? ""
            }
            if selectedLensID == nil, fetched.lenses.count == 1 {
                select(fetched.lenses[0])
            }
        } catch {
            status = "could not read the server's lens catalog: \(error.localizedDescription)"
        }
    }

    private func lookUpTokens() {
        guard isServerWorkspace, !isBusy, let client = service.cluster.client
        else { return }
        isBusy = true
        status = "resolving tokens on the server…"
        Task {
            defer { isBusy = false }
            do {
                tokenOptions = try await client.jlensTokenOptions(
                    modelID: selectedModelID, text: tokenQuery,
                    includeCaseVariants: includeCaseVariants)
                selectedToken = nil
                status = nil
            } catch {
                status = "token lookup failed: \(error.localizedDescription)"
            }
        }
    }

    private func run(_ action: Action) {
        guard isServerWorkspace, !isBusy, let client = service.cluster.client
        else { return }
        isBusy = true
        // A slow submit used to look like a dead click: the buttons dimmed
        // and nothing was said until the server answered with a job id.
        status = submitStatus(action)
        if action == .derive {
            // Stale success from an earlier derive must not sit under a new
            // one's error (audit 2026-09-06).
            deriveResult = nil
        }
        let modelID = action == .derive ? deriveModelID : selectedModelID
        Task {
            defer { isBusy = false }
            do {
                let (jobID, title): (String, String)
                switch action {
                case .acquire:
                    jobID = try await client.jlensAcquire(modelID: modelID)
                    title = "J-lens acquire: \(modelID)"
                case .importLens:
                    jobID = try await client.jlensImport(modelID: modelID)
                    title = "J-lens import: \(modelID)"
                case .derive:
                    guard let lensID = selectedLensID, let token = selectedToken else { return }
                    jobID = try await client.jlensDerive(
                        lensID: lensID, modelID: modelID,
                        tokenID: token.tokenID, piece: token.decoded ?? token.piece,
                        name: deriveName.isEmpty ? nil : deriveName)
                    title = "J-lens derive: token \(token.tokenID)"
                }
                status = "server job \(jobID) — live log in Activity"
                // Same job-following path every other server verb uses, so the
                // log lands in the Activity pane instead of a bespoke spinner.
                let job = await service.followServerJobInActivity(
                    jobID: jobID, client: client, title: title)
                guard let job else {
                    status = "\(title) is still running server-side — open Compute to watch"
                    return
                }
                if job.status == "succeeded" {
                    status = "\(title): done"
                    if action == .derive {
                        deriveResult = "derived — appears in the ordinary vector "
                            + "catalog; choose layer and alpha as usual"
                    }
                    await refresh()
                } else {
                    status = "\(title) \(job.status)"
                }
            } catch {
                status = "\(error.localizedDescription)"
            }
        }
    }

    private func submitStatus(_ action: Action) -> String {
        switch action {
        case .acquire: "asking the server to fetch the lens bytes…"
        case .importLens: "asking the server to convert the cached lens…"
        case .derive: "asking the server to derive the direction…"
        }
    }
}

/// The evidence tier, rendered wherever a lens or a derived artifact appears.
///
/// Not decoration. An evidence-tier and a testing-tier artifact differ in one
/// field, and this app is where the wrong one would be mistaken for the right
/// one by eye.
///
/// The tier records THIS PROJECT'S scope — which model its evidence comes from —
/// not what a model can do. A testing-tier run is out of scope for citation
/// here, not scientifically worthless.
/// Reportability of one trace row — the weaker of the lens's tier and the
/// identity of the condition that produced it. Distinct from ``TierBadge``,
/// which describes only whether the READOUT can be believed (round 7).
struct ClaimBadge: View {
    let claim: String?
    let reason: String?

    private var isQualified: Bool { claim == "qualified" }
    private var tint: Color { isQualified ? .green : .orange }

    var body: some View {
        Text(claim ?? "unstamped")
            .font(.caption2.monospaced())
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(tint.opacity(0.18))
            .foregroundStyle(tint)
            .clipShape(Capsule())
            .help(reason ?? "this condition is pinned and the lens is "
                  + "qualified — the row may be cited")
    }
}

struct TierBadge: View {
    let tier: String
    let expanded: Bool

    private var isEvidence: Bool { tier == "evidence" }

    var body: some View {
        Group {
            if expanded {
                Label(
                    "\(tier) tier — outside this study's evidence scope",
                    systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
            } else {
                Text(tier).font(.caption2.monospaced())
            }
        }
        .padding(.horizontal, 7).padding(.vertical, 2)
        .background((isEvidence ? Color.green : Color.orange).opacity(0.18))
        .foregroundStyle(isEvidence ? Color.green : Color.orange)
        .clipShape(Capsule())
        .help(isEvidence
              ? "evidence tier: this study's chosen model — may be qualified and cited"
              : "testing tier: fully usable, but outside this study's evidence "
                + "scope, so it cannot be qualified or cited here")
    }
}
