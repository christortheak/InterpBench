import AppKit
import ExperimentKit
import QuickLook
import SteeringKit
import SwiftUI

/// Data section: concepts/corpora/vector builders (the former Concept Lab)
/// plus Adapter Training (the former Fine-Tuning tab) as a build tool inside
/// Data — the design brief's "Build Tools can remain inside Data initially".
/// OptVec sits beside Adapter Training as the third build tool (2026-08-10):
/// its bundles are workspace DATA and its trained vectors are instrument
/// artifacts, the same family as adapters.
///
/// Inventory leads (WP-Data phase 1, 2026-08-19): the section is reorganized
/// around "Dataset → Check → Derive", so the landing view is what this
/// workspace HOLDS rather than a builder form.
///
/// Creation is role-first EVERYWHERE as of phase 4: the New Dataset flow
/// (`DatasetCreationSheet`) is the one entry, reachable from the Inventory
/// header, its empty state, and the Concepts & Vectors tool's own concept
/// pickers. The build tools are exactly that — editors and derivers. Each
/// still owns its rows, its recipe options, and its build gates; none of them
/// invents a dataset from a name typed into a field any more.
struct DataSectionView: View {
    @Bindable var service: ChatService
    /// The active tool, owned by `ChatView` (as the Agents section's region
    /// already is): the display pane follows it — Inventory shows the
    /// selected row's detail, the builders show the shared Activity feed —
    /// and the pane is rendered outside this view.
    @Binding var tool: Tool

    enum Tool: String, CaseIterable, Identifiable {
        case inventory = "Inventory"
        case concepts = "Concepts & Vectors"
        case adapterTraining = "Adapter Training"
        case optvec = "OptVec"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tool) {
                ForEach(Tool.allCases) { tool in
                    Text(tool.rawValue).tag(tool)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(8)

            switch tool {
            case .inventory:
                DatasetInventoryView(
                    service: service, openInConceptBuilder: openInConceptBuilder)
            case .concepts:
                ConceptsPanelView(service: service)
            case .adapterTraining:
                FineTuningPanelView(service: service)
            case .optvec:
                OptVecPanelView(service: service)
            }
        }
    }

    /// The New Dataset flow's one in-section routing action, performed by the
    /// shared router (`DataSectionRouting`) so this section and the display
    /// pane's inventory detail name the same destinations. The inventory's
    /// other routes are performed from the display pane, where the selected
    /// row's actions now live.
    private func openInConceptBuilder(_ concept: String?) {
        DataSectionRouting.openInConceptBuilder(
            concept, grandMeanRecipe: false, service: service, tool: $tool)
    }
}

/// The Data section's routing, in ONE place because two views perform it: the
/// section itself and the display pane's inventory detail
/// (`DataInventoryDetailColumn`), which renders the selected row outside this
/// section but names the same destinations. It knows both this section's
/// tools and the workbench's sections, which is why the destinations are
/// resolved here rather than in either view.
///
/// Every branch only navigates and SELECTS — no build is started, so each
/// builder's own gates still stand between the researcher and a forward pass.
///
/// The two out-of-section routes preselect through their panel's own model
/// seam (phase 4). Neither refuses when the selection no longer resolves —
/// the seams re-scan and answer false, and the destination's own empty state
/// takes it from there.
@MainActor
enum DataSectionRouting {

    /// `ConceptBuilder.selectedExisting` is the same state the Concepts
    /// panel's own picker binds to, so setting it drives the real selection
    /// (its `didSet` loads the concept's files) rather than only changing
    /// tabs. The index is refreshed first so a concept authored outside this
    /// app session is present in the picker's options before it is selected.
    static func openInConceptBuilder(
        _ concept: String?, grandMeanRecipe: Bool, service: ChatService,
        tool: Binding<DataSectionView.Tool>
    ) {
        if let concept {
            // The builder's OWN seam (refresh the index, then set the dataset
            // selection, whose didSet loads the files and moves the build
            // target). Open-coding it here is what let the two selections
            // drift apart before 2026-08-19.
            service.concepts.selectConcept(concept)
        }
        // ORDER MATTERS: selecting a concept loads its files, and that load
        // resolves the recipe FROM DISK (`ConceptBuilder.loadSelectedExisting`
        // → `pairedRecipeFamilyOnDisk`, or `.emotionGrandMean` when only
        // story rows exist). Setting the recipe first would be overwritten a
        // line later. `recipeFamily`'s own didSet performs the whole switch,
        // so this is the builder's real seam, not a shadow copy of it.
        if grandMeanRecipe {
            service.concepts.recipeFamily = .emotionGrandMean
        }
        tool.wrappedValue = .concepts
    }

    static func route(
        _ request: DatasetRouteRequest, service: ChatService,
        tool: Binding<DataSectionView.Tool>,
        navigate: (WorkbenchSection) -> Void,
        openAgentsLibrary: () -> Void
    ) {
        switch request {
        case .conceptBuilder(let concept, let grandMeanRecipe):
            openInConceptBuilder(
                concept, grandMeanRecipe: grandMeanRecipe, service: service,
                tool: tool)
        case .derived(.conceptsAndVectors, _):
            openInConceptBuilder(
                nil, grandMeanRecipe: false, service: service, tool: tool)
        case .derived(.adapterTraining, _):
            tool.wrappedValue = .adapterTraining
        case .derived(.analysis, let selection):
            if let selection { service.geometry.select(vectorIDs: [selection]) }
            navigate(.analysis)
        case .derived(.agents, let selection):
            if let selection { service.fineTuning.selectAgent(id: selection) }
            openAgentsLibrary()
        }
    }
}

/// Compute section: a thin global-state header (compute target, connection,
/// installed models) above the existing server jobs/logs panel. Connection
/// editing stays in the window-toolbar Compute selector.
struct ComputeSectionView: View {
    @Bindable var service: ChatService

    // SPLIT-VIEW MINIMUM HEIGHT (UI audit 2026-09-06, headline 8). This view
    // sits DIRECTLY in the HSplitView column (`ChatView.detailSplit`) with no
    // ScrollView, so every row above the Divider contributes to the column's
    // reported minimum — and a minimum that moves mid-display-cycle is fatal
    // on macOS 27 (the 2026-08-05 crash class; `ServerJobsPanelView.jobsRegion`
    // carries the incident write-up).
    //
    // Of the two repairs the audit offered, this file takes CONSTANT-HEIGHT
    // SLOTS rather than wrapping the rows in a fixed-frame ScrollView: the
    // three credential rows must stay visible without scrolling, and the slot
    // idiom is already the one the jobs panel below uses. So every row here
    // renders the same number of lines in every state — one-line captions
    // with the full text in `.help` or behind an ⓘ, per-model detail in a
    // popover, and conditional messages as always-present slots that go
    // transparent instead of disappearing.
    var body: some View {
        VStack(spacing: 0) {
            header
            ClaudeAPIKeyRow()
            ExternalJudgeKeyRow(service: service)
            HuggingFaceTokenRow()
            ModelCapabilitiesRow(service: service)
            pairingWarningRow
            Divider()
            ServerJobsPanelView(service: service)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Compute: \(service.cluster.substrateLabel)")
                    .font(.headline)
                Text(statusLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    // The status can be a long failure sentence; the group
                    // help below explains the section, so without this the
                    // truncated text has no full-text carrier.
                    .help(statusLine)
            }
            Spacer()
            if service.cluster.computeTarget == .server {
                InstallModelButton(cluster: service.cluster)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .help(headerHelp)
    }

    /// One name for the control that switches this — "Compute", the toolbar's
    /// own word — and an honest sentence for the Local target, which has no
    /// server jobs to show.
    private var headerHelp: String {
        switch service.cluster.computeTarget {
        case .local:
            return "switch where this workspace computes with the Compute "
                + "selector in the window toolbar — on Local (MLX) everything "
                + "runs inside this app, so the jobs list below stays empty"
        case .server:
            return "switch where this workspace computes with the Compute "
                + "selector in the window toolbar — the list below shows the "
                + "active target's jobs and logs"
        }
    }

    /// Standing unpaired-server warning (the root-incident guard): when the
    /// active server's artifact root is not this app's data workspace, every
    /// server-side authoring/build/run write lands elsewhere — say so HERE,
    /// permanently, instead of letting a run refusal be the first hint.
    ///
    /// One constant slot, three states (UI audit 2026-09-06): the mismatch
    /// warning, the remote-root description, and nothing used to be three
    /// different heights, arriving with async cluster state — see the
    /// split-view note on `body`. Same font, same line count, same padding in
    /// every branch; the full text lives in `.help`.
    private var pairingWarningRow: some View {
        let warning = service.cluster.activeServerPairingWarning
        let description = service.cluster.activeServerPairingDescription
        let text = warning ?? description ?? ""
        let isWarning = warning != nil
        let tint: Color = isWarning ? .orange : .secondary
        let background: Color = isWarning ? Color.orange.opacity(0.10) : .clear
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            Label(
                text.isEmpty ? " " : text,
                systemImage: isWarning
                    ? "exclamationmark.triangle.fill"
                    : "externaldrive.connected.to.line.below")
                .font(.callout)
                .foregroundStyle(tint)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                .opacity(text.isEmpty ? 0 : 1)
                .accessibilityHidden(text.isEmpty)
                .help(isWarning ? Self.pairingMismatchHelp : text)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(background)
    }

    private static let pairingMismatchHelp: String =
        "the server's /api/info root differs from this app's data workspace — "
        + "restart it with serve --root <workspace> (or STEERLAB_ROOT) so both "
        + "engines share one artifact tree"

    private var statusLine: String {
        let models = service.workspaceModelOptions.count
        let modelText = "\(models) model\(models == 1 ? "" : "s") available"
        switch service.cluster.computeTarget {
        case .local:
            return "in-process MLX · \(modelText)"
        case .server:
            return "\(service.cluster.status ?? "not connected") · \(modelText)"
        }
    }
}

/// The workspace's chat-template capability records (2026-09-05): what each
/// probed model's template does with a system turn, whether it has a thinking
/// switch, which reasoning-effort levels it accepts, with any human override
/// shown beside the detected value. Read from `prompts/models/` of the active
/// workspace, which both engines share, so the row is the same whichever
/// substrate computed the record.
///
/// ONE LINE, ALWAYS (UI audit 2026-09-06, headline 8): this used to render one
/// two-line `Text` per probed model, growing by up to 2 × N lines at the
/// moment the server's model list landed — an async change to the split-view
/// column's minimum height, which is the documented crash shape. The per-model
/// detail moved into a popover and the tooltip.
private struct ModelCapabilitiesRow: View {
    @Bindable var service: ChatService
    @State private var records: [(path: String, record: ModelCapabilities.Record)] = []
    @State private var showingRecords = false

    var body: some View {
        HStack(spacing: 8) {
            Text("Model capabilities")
                .font(.callout)
            Text(summaryLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .help(detailText)
            Spacer()
            Button("Show records") { showingRecords = true }
                .controlSize(.small)
                .disabled(records.isEmpty)
                .help(records.isEmpty
                    ? "nothing has been probed in this workspace yet, so there "
                        + "is no record to show"
                    : "list every probed model's record — what its chat "
                        + "template does with a system turn, its thinking "
                        + "switch, and the effort levels it accepts")
                .popover(isPresented: $showingRecords) { recordsPopover }
            Button("Re-read records") { refresh() }
                .controlSize(.small)
                .help("re-read prompts/models/ in this workspace — it reads "
                    + "what is already on disk; nothing is probed or downloaded")
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .onAppear { refresh() }
        .onChange(of: service.workspaceModelOptions.count) { _, _ in refresh() }
    }

    private var summaryLine: String {
        guard !records.isEmpty else {
            return "nothing probed yet — support is read from the model name"
        }
        return "\(records.count) probed model\(records.count == 1 ? "" : "s")"
    }

    private var detailText: String {
        guard !records.isEmpty else { return Self.emptyExplanation }
        return records
            .map { Self.line(for: $0.record.effective(path: $0.path)) }
            .joined(separator: "\n")
    }

    private static let emptyExplanation: String =
        "No chat-template record has been probed in this workspace yet. One is "
        + "written under prompts/models/ when a model is installed or first "
        + "loaded here; until then the app decides what a model supports from "
        + "its name alone, and each declaration says so."

    /// Fixed frame: a popover is outside the column's layout, but a popover
    /// that resizes as records land is its own kind of jumpy — it scrolls.
    private var recordsPopover: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(records, id: \.path) { entry in
                    let view = entry.record.effective(path: entry.path)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(Self.line(for: view))
                            .font(.caption)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(Self.detail(for: view))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(width: 460, height: 260)
    }

    /// One line per record: the detected facts, an override marked beside
    /// the value it replaces.
    static func line(for view: ModelCapabilities) -> String {
        func shown(_ field: String, _ detected: String) -> String {
            guard let override = view.overrides[field] else { return detected }
            return "\(override.value.description) (override; detected \(detected))"
        }
        let record = view
        var parts: [String] = [
            "\(record.modelID) @ \((record.revision ?? "unpinned").prefix(12))",
            "system role \(shown("systemRole", record.systemRole.rawValue))",
            "thinking switch \(shown("thinkingSwitch", record.thinkingSwitch.rawValue))",
        ]
        let accepted = record.acceptedEfforts
        if record.hasThinkingSwitch {
            parts.append(
                accepted.isEmpty
                    ? "effort levels: none accepted"
                    : "effort levels: " + accepted.joined(separator: ", "))
        }
        if record.source == .heuristic { parts.append("HEURISTIC — not probed") }
        return parts.joined(separator: " · ")
    }

    /// The record's own summary, with the Markdown emphasis SwiftUI would not
    /// parse here stripped out.
    static func detail(for view: ModelCapabilities) -> String {
        view.summaryLines.joined(separator: "\n")
            .replacingOccurrences(of: "**", with: "")
    }

    private func refresh() {
        records = ModelCapabilitiesStore.list(root: ExperimentStore.workspaceRoot)
            .filter { $0.record.source == .probe }
    }
}

/// Green "a key is stored" badge beside each credential field. The previous
/// signal was that Clear happened to be enabled — legible only to someone
/// who already knew the answer.
private struct KeyStoredBadge: View {
    let isStored: Bool

    var body: some View {
        Image(systemName: isStored ? "checkmark.circle.fill" : "circle.dotted")
            .foregroundStyle(isStored ? Color.green : Color.secondary.opacity(0.5))
            .imageScale(.medium)
            .help(isStored ? "a key is stored in the macOS Keychain"
                           : "no key stored")
            .accessibilityLabel(isStored ? "key stored" : "no key stored")
    }
}

/// The credential rows' ONE caption line: the current notice when there is
/// one (a Keychain failure, a cluster sync outcome), otherwise the resting
/// status, with the custody policy behind the ⓘ.
///
/// One line in every state (UI audit 2026-09-06, headline 8). These rows sit
/// directly in the Compute split-view column, so both shapes they used to
/// have were hazards: a 40-to-70-word `.caption2` paragraph that reflowed to
/// a different number of lines with the window width, and a message row that
/// appeared and disappeared with an async Keychain write or cluster sync.
/// Either moves the column's reported minimum height mid-display-cycle — the
/// 2026-08-05 crash class. Plain `Text`, never a `Label`, so an icon's
/// metrics cannot move it either; a problem is orange AND says so in words,
/// never colour alone.
private struct CredentialStatusRow: View {
    let status: String
    var notice: String?
    var noticeIsProblem: Bool = true
    let policy: String

    var body: some View {
        HStack(spacing: 6) {
            Text(notice ?? status)
                .font(.caption2)
                .foregroundStyle(
                    notice != nil && noticeIsProblem ? Color.orange : Color.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(notice ?? status)
            InfoButton(text: policy)
            Spacer()
        }
    }
}

/// The sentence every credential row shows when the Keychain refuses a write.
/// Before the 2026-09-06 audit all three rows cleared the field and reported
/// "no key stored", which is what the row also says when nothing was ever
/// typed.
private enum CredentialWriteFailure {
    static let keychain: String =
        "could not write to the Keychain — nothing was stored; unlock the "
        + "login keychain (or grant this app access to it) and Save again"
}

/// The ONE place the app takes the researcher's Anthropic API key. Writes
/// go to the macOS Keychain through `AnthropicKeyStore` (never plaintext
/// UserDefaults); the stored secret is never echoed back into the field.
private struct ClaudeAPIKeyRow: View {
    @State private var draft = ""
    @State private var hasStoredKey = false
    @State private var writeFailed = false
    @State private var confirmingClear = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            fieldRow
            CredentialStatusRow(
                status: statusCaption,
                notice: writeFailed ? CredentialWriteFailure.keychain : nil,
                policy: Self.custodyPolicy)
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .onAppear { refresh() }
    }

    private var fieldRow: some View {
        HStack(spacing: 8) {
            KeyStoredBadge(isStored: hasStoredKey)
            Text("Claude API key")
                .font(.callout)
            SecureField("sk-ant-…", text: $draft)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 280)
                .accessibilityLabel("Claude API key")
                .help("paste an Anthropic API key (console.anthropic.com → API "
                    + "keys); Save puts it in this Mac's Keychain and it is "
                    + "never read back into this field")
            Button("Save") { save() }
                .controlSize(.small)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("store the key in the macOS Keychain")
            Button("Clear") { confirmingClear = true }
                .controlSize(.small)
                .disabled(!hasStoredKey)
                .help("delete the stored key from the macOS Keychain")
                .confirmationDialog(
                    "Delete the stored Claude API key?",
                    isPresented: $confirmingClear,
                    titleVisibility: .visible
                ) {
                    Button("Delete key", role: .destructive) { clear() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Claude judging and stimulus generation stop working "
                        + "on this Mac until a key is entered again, and the "
                        + "key cannot be read back out of the Keychain first.")
                }
            Spacer()
        }
    }

    private var statusCaption: String {
        var text = hasStoredKey ? "a key is stored" : "no key stored"
        text += " · Keychain only, never sent to a server"
        if AnthropicKeyStore.environmentOverrides() {
            text += " · ANTHROPIC_API_KEY in this app's environment wins"
        }
        return text
    }

    // Key-custody policy (2026-07-18): the key lives in this Mac's Keychain
    // and NEVER goes to the cluster — Claude judging always runs here, against
    // downloaded run artifacts; cluster-side judging uses local-model judges.
    private static let custodyPolicy: String =
        "The key is stored in this Mac's Keychain and is never sent to a "
        + "server. Claude judging, stimulus generation, and sweep credential "
        + "checks all run here: generations produced on the cluster are judged "
        + "on this Mac after they are downloaded. To judge on the cluster "
        + "instead, pin a local judge model. ANTHROPIC_API_KEY set in this "
        + "app's environment wins over the stored key."

    private func save() {
        let stored = ClaudeStimulusGenerator.saveAPIKey(draft)
        draft = ""
        refresh()
        writeFailed = !stored
    }

    private func clear() {
        _ = ClaudeStimulusGenerator.saveAPIKey("")
        draft = ""
        refresh()
        writeFailed = false
    }

    private func refresh() {
        hasStoredKey = AnthropicKeyStore.hasStoredKey()
    }
}

/// The EXTERNAL judge key (key-custody design, seamless-pipeline extension
/// 2026-07-19): a dedicated, ideally spend-capped OpenRouter or Anthropic
/// key that enables INLINE external judging on the cluster. Distinct from
/// the personal Claude key above, which never leaves this Mac. Stored in
/// the Keychain (`JudgeKeyStore`); synced to `~/.steerlab/judge-key` (mode
/// 600) at every cluster connect — and REMOVED from the cluster at the
/// next connect after Clear, so deletion propagates.
private struct ExternalJudgeKeyRow: View {
    @Bindable var service: ChatService
    @State private var draft = ""
    @State private var kind = "openrouter"
    @State private var hasStoredKey = false
    @State private var writeFailed = false
    @State private var syncNote = ""
    @State private var isSyncing = false
    @State private var confirmingClear = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            fieldRow
            CredentialStatusRow(
                status: statusCaption,
                notice: notice,
                noticeIsProblem: noticeIsProblem,
                policy: Self.custodyPolicy)
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .onAppear { refresh() }
    }

    private var fieldRow: some View {
        HStack(spacing: 8) {
            KeyStoredBadge(isStored: hasStoredKey)
            Text("External judge key")
                .font(.callout)
            Picker("Judge key provider", selection: $kind) {
                Text("OpenRouter").tag("openrouter")
                Text("Anthropic").tag("anthropic")
            }
            .labelsHidden()
            .frame(width: 110)
            .accessibilityLabel("Judge key provider")
            .help("which service issued the key — decides how the cluster's "
                + "inline judge authenticates with it")
            SecureField(kind == "openrouter" ? "sk-or-…" : "sk-ant-…",
                        text: $draft)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 220)
                .accessibilityLabel("External judge key")
                .help("paste the dedicated judging key; Save stores it in this "
                    + "Mac's Keychain and pushes it to the cluster, and it is "
                    + "never read back into this field")
            Button("Save") { save() }
                .controlSize(.small)
                .disabled(isSyncing
                    || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("store in the macOS Keychain and push to the cluster "
                    + "(~/.steerlab/judge-key, mode 600) at every connect")
            Button("Clear") { confirmingClear = true }
                .controlSize(.small)
                .disabled(isSyncing || !hasStoredKey)
                .help("delete from the Keychain AND remove from the cluster "
                    + "at the next sync")
                .confirmationDialog(
                    "Delete the external judge key?",
                    isPresented: $confirmingClear,
                    titleVisibility: .visible
                ) {
                    Button("Delete key", role: .destructive) { clear() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("It is removed from this Mac's Keychain now and from "
                        + "the cluster at the next sync, so unattended "
                        + "pipelines lose inline external judging until a key "
                        + "is entered again.")
                }
            Spacer()
        }
    }

    /// The sync outcome, when there is one: "syncing…" while the push runs,
    /// then either the cluster's warning or a positive "pushed at <time>".
    /// A successful push sets `judgeKeySyncResult` to the EMPTY string, which
    /// rendered as nothing at all before the 2026-09-06 audit — silence and
    /// success looked identical.
    private var notice: String? {
        if writeFailed { return CredentialWriteFailure.keychain }
        if isSyncing {
            return "syncing the key to \(service.cluster.substrateLabel)…"
        }
        let warning = service.judgeKeySyncResult ?? ""
        if !warning.isEmpty { return warning }
        return syncNote.isEmpty ? nil : syncNote
    }

    /// Only a real warning is orange: "syncing…" and "pushed at …" are news.
    private var noticeIsProblem: Bool {
        if writeFailed { return true }
        if isSyncing { return false }
        return !(service.judgeKeySyncResult ?? "").isEmpty
    }

    private var statusCaption: String {
        guard hasStoredKey else {
            return "no key stored · cluster-side external judging defers to this Mac"
        }
        let storedKind = JudgeKeyStore.stored()?.kind ?? ""
        return "a \(storedKind) key is stored · pushed to the cluster at every connect"
    }

    private static let custodyPolicy: String =
        "Use a dedicated, spend-capped key here, never a personal one: unlike "
        + "the Claude key above, this one is pushed to the cluster (mode 600 "
        + "in $HOME) at every connect, so unattended pipelines can judge "
        + "inline. Clearing it here also removes it from the cluster at the "
        + "next sync."

    private func save() {
        let stored = JudgeKeyStore.save(kind: kind, key: draft)
        draft = ""
        refresh()
        writeFailed = !stored
        guard stored else { return }
        sync(outcome: "pushed")
    }

    private func clear() {
        JudgeKeyStore.delete()
        refresh()
        writeFailed = false
        sync(outcome: "removed")
    }

    /// Push (or removal) with a busy state and a positive confirmation — both
    /// missing before the 2026-09-06 audit.
    private func sync(outcome: String) {
        guard !isSyncing else { return }
        isSyncing = true
        syncNote = ""
        Task {
            await service.syncJudgeKeyNow()
            isSyncing = false
            if (service.judgeKeySyncResult ?? "").isEmpty {
                syncNote = "\(outcome) to \(service.cluster.substrateLabel) at "
                    + Self.clock.string(from: Date())
            }
        }
    }

    private static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    private func refresh() {
        hasStoredKey = JudgeKeyStore.stored() != nil
        if let stored = JudgeKeyStore.stored() { kind = stored.kind }
    }
}

/// The Hugging Face READ token, beside the other credentials instead of
/// buried in the cluster connection menu. Saving stores it in the Keychain
/// AND materializes `~/.cache/huggingface/token` (the hub's native
/// location, mode 600) — the copy this Mac's server, CLI, and calibration
/// scripts actually read, which the cluster-only install sheet never wrote.
/// The cluster's own copy still travels via the connection menu's
/// per-site install.
private struct HuggingFaceTokenRow: View {
    @State private var draft = ""
    @State private var hasStoredToken = false
    @State private var saveError: String?
    @State private var confirmingClear = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            fieldRow
            CredentialStatusRow(
                status: statusCaption, notice: saveError,
                policy: Self.tokenPolicy)
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .onAppear { refresh() }
    }

    private var fieldRow: some View {
        HStack(spacing: 8) {
            KeyStoredBadge(isStored: hasStoredToken)
            Text("Hugging Face token")
                .font(.callout)
            SecureField("hf_…", text: $draft)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 280)
                .accessibilityLabel("Hugging Face token")
                .help("paste a read token (huggingface.co → Settings → Access "
                    + "Tokens); Save stores it in this Mac's Keychain and "
                    + "writes the hub's own token file, and it is never read "
                    + "back into this field")
            Button("Save") { save() }
                .controlSize(.small)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("store in the macOS Keychain and write "
                    + "~/.cache/huggingface/token for local downloads")
            Button("Clear") { confirmingClear = true }
                .controlSize(.small)
                .disabled(!hasStoredToken)
                .help("delete from the Keychain; the hub token file is "
                    + "removed only if it holds this same token")
                .confirmationDialog(
                    "Delete the stored Hugging Face token?",
                    isPresented: $confirmingClear,
                    titleVisibility: .visible
                ) {
                    Button("Delete token", role: .destructive) { clear() }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Downloads of gated models stop authenticating on "
                        + "this Mac until a token is entered again. The "
                        + "cluster's own copy is unaffected.")
                }
            Spacer()
        }
    }

    private var statusCaption: String {
        var text = hasStoredToken ? "a token is stored" : "no token stored"
        if HuggingFaceTokenStore.environmentOverrides() {
            text += " · HF_TOKEN in this app's environment wins"
        } else if !hasStoredToken, HuggingFaceTokenStore.hubFileExists() {
            // hf auth login was run independently — downloads work; the
            // badge is honest about the Keychain, this line about reality.
            text += " · but ~/.cache/huggingface/token exists, so downloads "
                + "still authenticate"
        }
        return text
    }

    private static let tokenPolicy: String =
        "A read token from huggingface.co → Settings → Access Tokens. Gated "
        + "models (Gemma, for instance) also need their licence accepted by "
        + "the same account. Saving writes both copies this Mac owns — the "
        + "Keychain and ~/.cache/huggingface/token — so the local engine can "
        + "download; the cluster's copy is installed from the connection "
        + "menu. HF_TOKEN set in this app's environment wins over both."

    private func save() {
        saveError = HuggingFaceTokenStore.save(draft)
        draft = ""
        refresh()
    }

    private func clear() {
        saveError = HuggingFaceTokenStore.save("")
        draft = ""
        refresh()
    }

    private func refresh() {
        hasStoredToken = HuggingFaceTokenStore.hasStoredToken()
    }
}

/// Results section: a read-only browser over immutable `runs/` directories.
/// Substrate-aware source handling (`ExperimentPanel.resultsSource`):
/// - Local compute: scan this workspace's runs/ (unchanged).
/// - Paired server: same local list — server runs land in the shared tree —
///   plus a caption saying so instead of a duplicate remote list.
/// - Unpaired server: browse the SERVER's runs read-only over the API,
///   source-labeled like Optimizations, with the same detail layout and the same
///   preview rendering (`RunBrowser`'s pure parsers on bounded fetches).
/// Study-scoped result review stays in Studies; this is the global browse.
struct ResultsPanelView: View {
    @Bindable var service: ChatService
    @State private var items: [RunBrowser.Item] = []
    @State private var filterText = ""
    @State private var runTypeFilter: String?
    @State private var selectedID: String?
    /// Unpaired-server workspaces (the normal CLUSTER shape: compute on
    /// the cluster, data local) get a source toggle instead of a
    /// remote-only view. LOCAL IS THE DEFAULT (field report 2026-08-03:
    /// "Results is empty when I'm not connected — but results are
    /// imported and everything is held locally"): the workspace's own
    /// runs/ tree browses with no connection at all; "On server" lists
    /// the server-resident runs when connected.
    @State private var remoteWorkspaceSource: RemoteWorkspaceSource = .local

    private enum RemoteWorkspaceSource: String, CaseIterable, Identifiable {
        case local = "This workspace (imported)"
        case server = "On server"
        var id: String { rawValue }
    }

    var body: some View {
        Group {
            if service.experiments.resultsSource == .remoteServer {
                VStack(spacing: 0) {
                    Picker("", selection: $remoteWorkspaceSource) {
                        ForEach(RemoteWorkspaceSource.allCases) { source in
                            Text(source.rawValue).tag(source)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .padding(8)
                    .help(
                        "cluster workspace: runs imported into this "
                            + "workspace's tree browse offline; On server "
                            + "lists the server-resident runs and needs the "
                            + "connection")
                    if remoteWorkspaceSource == .server {
                        RemoteResultsBrowserView(service: service)
                    } else {
                        localBrowser
                    }
                }
                .onChange(of: remoteWorkspaceSource) {
                    // Switching to the local list retires the remote
                    // selection so the viewer column follows the browser
                    // the researcher is actually looking at.
                    if remoteWorkspaceSource == .local {
                        service.experiments.results.selectedRemoteResultsRun = nil
                    }
                }
            } else {
                localBrowser
            }
        }
    }

    private var localBrowser: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if service.experiments.resultsSource == .pairedServer {
                pairedCaptionRow
            }
            Divider()
            if items.isEmpty {
                emptyState
            } else {
                browser
            }
        }
        .onAppear { rescan() }
        .onChange(of: selectedID) { syncSelection() }
        // Selection intentionally survives navigation: the "Selected Run"
        // viewer can be pinned while browsing other sections.
    }

    /// Paired-server clarification: the server writes into THIS workspace's
    /// tree, so its runs are already in the local list below — no duplicate
    /// remote listing.
    private var pairedCaptionRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "link")
                .imageScale(.small)
                .foregroundStyle(.secondary)
            Text(
                "paired server — runs shown from the shared workspace "
                    + "(server runs land in this same tree)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 6)
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Run directories")
                    .font(.headline)
                Text(
                    "\(filtered.count) of \(items.count) shown · immutable, "
                        + "newest first")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            // Workspace-level entry: the explorer's own run picker over
            // every run in the workspace — no selection required.
            ResultsExplorerButton(runName: nil)
            runTypePicker
            TextField("filter (name, type, experiment, model)", text: $filterText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 220)
            Button {
                rescan()
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    /// Run types actually present in the scan (config.json runType stamps,
    /// with sweep runs reading as "optimization (screen)").
    private var runTypes: [String] {
        Array(Set(items.compactMap(\.displayRunType))).sorted()
    }

    private var runTypePicker: some View {
        Picker("Type", selection: $runTypeFilter) {
            Text("all types").tag(String?.none)
            ForEach(runTypes, id: \.self) { runType in
                Text(runType).tag(String?.some(runType))
            }
        }
        .frame(maxWidth: 180)
        .help("filter by the config.json runType stamp")
    }

    private var filtered: [RunBrowser.Item] {
        var result = items
        if let runTypeFilter {
            result = result.filter { $0.displayRunType == runTypeFilter }
        }
        let needle = filterText.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return result }
        return result.filter { item in
            item.name.lowercased().contains(needle)
                || (item.displayRunType?.lowercased().contains(needle) ?? false)
                || (item.experiment?.lowercased().contains(needle) ?? false)
                || (item.modelID?.lowercased().contains(needle) ?? false)
        }
    }

    private var browser: some View {
        VSplitView {
            runList
                .frame(minHeight: 140, idealHeight: 220)
            detailPane
                .frame(minHeight: 200, maxHeight: .infinity)
        }
    }

    private var runList: some View {
        List(filtered, selection: $selectedID) { item in
            RunDirectoryRow(item: item)
                .tag(item.id)
        }
    }

    @ViewBuilder
    private var detailPane: some View {
        if let item = selectedItem {
            RunDetailView(service: service, item: item)
        } else {
            ContentUnavailableView {
                Label("Select a run", systemImage: "cursorarrow.click")
            } description: {
                Text(
                    "Click a run directory above to see its stamps and files; "
                        + "the focused file's contents render in the viewer.")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var selectedItem: RunBrowser.Item? {
        guard let selectedID else { return nil }
        return items.first { $0.id == selectedID }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No runs yet", systemImage: "archivebox")
        } description: {
            Text(
                "Every extraction, validation, sweep, study, and multi-agent "
                    + "run writes an immutable directory under this workspace's "
                    + "runs/ folder.")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func rescan() {
        items = RunBrowser.list()
        if let selectedID, !items.contains(where: { $0.id == selectedID }) {
            self.selectedID = nil
        }
        syncSelection()
    }

    /// Mirror the selection into the shared state the activity pane reads.
    /// Deselecting the run also drops the focused file — the viewer must not
    /// keep previewing a file of a run that is no longer selected.
    private func syncSelection() {
        service.selectedResultsRun = selectedItem
        if selectedItem == nil {
            service.experiments.results.selectedResultsFile = nil
        }
    }
}

private struct RunDirectoryRow: View {
    let item: RunBrowser.Item

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.callout.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(detailLine)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([item.url])
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
            .help("reveal this run directory in Finder")
        }
        .padding(.vertical, 2)
    }

    private var detailLine: String {
        // Shared with the remote row (WS6.1): run-type + ENGINE badges, so
        // local and server runs read identically with the substrate badge as
        // the only difference. Stampless runs render exactly as before.
        RunBrowser.rowDetailLine(
            runType: item.displayRunType, substrate: item.substrate,
            experiment: item.experiment, modelID: item.modelID)
    }
}

/// Detail for one selected run: config.json stamps and the run's FILE LIST —
/// previewable files (reports, metrics CSV, generations/judgments JSONL,
/// recommendations) as selectable rows whose contents render in the activity
/// viewer's Results mode, plus a name+size list for everything else. The
/// listing stays here; the contents go to the viewer (live-testing finding).
/// Previewability checks are bounded (`RunBrowser.preview`) — big files
/// degrade to name+size+Finder, never a stall.
private struct RunDetailView: View {
    @Bindable var service: ChatService
    let item: RunBrowser.Item
    @State private var previewable: [RunBrowser.FileEntry] = []
    @State private var unpreviewed: [RunBrowser.FileEntry] = []
    /// System QuickLook target (field request 2026-08-03: a file row should
    /// quick-look, with an option to open in the default app).
    @State private var quickLookURL: URL?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                stampHeader
                // Semantic layer (A3/A6/A13, P3/P4): evidence-status
                // classification, categorical study view, analyze action,
                // statistics tables, structured validation report — all
                // derived read-only from the run's own artifacts.
                RunSemanticSectionsView(service: service, item: item)
                if !previewable.isEmpty {
                    previewableFilesBox
                }
                if !unpreviewed.isEmpty {
                    otherFilesBox
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear { load() }
        .onChange(of: item.id) { load() }
        .quickLookPreview($quickLookURL)
    }

    private var previewableFilesBox: some View {
        GroupBox("Files — click one to preview it in the viewer") {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(previewable) { file in
                    SelectableRunFileRow(
                        file: file,
                        isSelected: service.experiments.results.selectedResultsFile?.id == file.id,
                        select: {
                            service.experiments.results.selectedResultsFile = file
                        },
                        quickLook: { quickLookURL = file.url })
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var stampHeader: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(item.name)
                    .font(.callout.monospaced().weight(.semibold))
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([item.url])
                } label: {
                    Label("Open in Finder", systemImage: "folder")
                }
                .controlSize(.small)
                ResultsExplorerButton(runName: item.name)
            }
            Text(stampLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var stampLine: String {
        var parts: [String] = []
        if let runType = item.displayRunType { parts.append(runType) }
        if let createdAt = item.createdAt { parts.append(createdAt) }
        if let model = item.modelID { parts.append(model) }
        if let revision = item.revision { parts.append("rev \(revision.prefix(12))") }
        if let experiment = item.experiment { parts.append("exp \(experiment)") }
        if let substrate = item.substrate { parts.append(substrate) }
        if parts.isEmpty { parts.append("no config.json stamp (legacy run type)") }
        return parts.joined(separator: " · ")
    }

    private var otherFilesBox: some View {
        GroupBox("Other files") {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(unpreviewed) { file in
                    OtherFileRow(file: file) { quickLookURL = file.url }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func load() {
        let files = RunBrowser.files(in: item.url)
        var shown: [RunBrowser.FileEntry] = []
        var rest: [RunBrowser.FileEntry] = []
        for file in files {
            if case .unavailable = RunBrowser.preview(for: file) {
                rest.append(file)
            } else {
                shown.append(file)
            }
        }
        shown.sort { runFilePreviewPriority($0.name) < runFilePreviewPriority($1.name) }
        previewable = shown
        unpreviewed = rest
        // Keep a still-valid focused file across reloads; otherwise focus the
        // top-priority file so the viewer shows content as soon as a run is
        // selected (never an empty viewer next to a populated list).
        let currentID = service.experiments.results.selectedResultsFile?.id
        if currentID == nil || !shown.contains(where: { $0.id == currentID }) {
            service.experiments.results.selectedResultsFile = shown.first
        }
    }
}

/// One previewable file in the run detail's list: selecting it focuses the
/// file, and the activity viewer's Results mode renders its bounded preview.
private struct SelectableRunFileRow: View {
    let file: RunBrowser.FileEntry
    let isSelected: Bool
    let select: () -> Void
    let quickLook: () -> Void

    // The icon controls are SIBLINGS of the selection button, not children
    // (review 2026-08-03, P2): nested buttons give ambiguous click targets
    // and a wrong accessibility tree.
    var body: some View {
        HStack(spacing: 6) {
            Button(action: select) {
                HStack(spacing: 6) {
                    Image(systemName: "doc.text")
                        .foregroundStyle(isSelected ? .primary : .secondary)
                        .imageScale(.small)
                    Text(file.name)
                        .font(.caption.monospaced().weight(isSelected ? .semibold : .regular))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Text(ByteCountFormatter.string(
                        fromByteCount: Int64(file.size), countStyle: .file))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("preview \(file.name) in the viewer pane")
            Button(action: quickLook) {
                Image(systemName: "eye")
            }
            .buttonStyle(.borderless)
            .help("Quick Look \(file.name)")
            Button {
                NSWorkspace.shared.open(file.url)
            } label: {
                Image(systemName: "arrow.up.forward.square")
            }
            .buttonStyle(.borderless)
            .help("open \(file.name) in its default app")
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(
            isSelected ? AnyShapeStyle(.tint.opacity(0.15)) : AnyShapeStyle(.clear),
            in: RoundedRectangle(cornerRadius: 5))
    }
}

/// Researcher-priority ordering for preview boxes (local AND remote detail
/// panes): reports first, then metrics, then raw generations/judgments,
/// then everything else alphabetically.
private func runFilePreviewPriority(_ name: String) -> String {
    switch name {
    case "report.json": "0"
    case "validation-report.json": "1"
    case "recommendations.json": "2"
    case _ where name.hasSuffix(".csv"): "3-\(name)"
    case "generations.jsonl": "4"
    case "judgments.jsonl": "5"
    case _ where name.hasSuffix(".jsonl"): "6-\(name)"
    default: "7-\(name)"
    }
}

private struct OtherFileRow: View {
    let file: RunBrowser.FileEntry
    var quickLook: (() -> Void)?

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: file.isDirectory ? "folder" : "doc")
                .foregroundStyle(.secondary)
                .imageScale(.small)
            Text(file.name)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            if !file.isDirectory {
                Text(ByteCountFormatter.string(
                    fromByteCount: Int64(file.size), countStyle: .file))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            if !file.isDirectory, let quickLook {
                Button(action: quickLook) {
                    Image(systemName: "eye")
                }
                .buttonStyle(.borderless)
                .help("Quick Look \(file.name)")
                Button {
                    NSWorkspace.shared.open(file.url)
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("open \(file.name) in its default app")
            }
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([file.url])
            } label: {
                Image(systemName: "arrow.up.forward.square")
            }
            .buttonStyle(.borderless)
            .help("reveal in Finder")
        }
    }
}

/// One previewed file: a GroupBox whose content matches the preview kind.
/// Takes plain name+size so the SAME box renders local files and remote
/// (server-fetched) previews. Internal (not private): the activity viewer's
/// Results mode (`ResultsRunSummaryColumn`) renders the focused file with
/// this exact box — one renderer, one parser (`RunBrowser`).
struct RunFilePreviewBox: View {
    let name: String
    let size: Int
    let preview: RunBrowser.FilePreview

    init(name: String, size: Int, preview: RunBrowser.FilePreview) {
        self.name = name
        self.size = size
        self.preview = preview
    }

    init(file: RunBrowser.FileEntry, preview: RunBrowser.FilePreview) {
        self.init(name: file.name, size: file.size, preview: preview)
    }

    var body: some View {
        GroupBox {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack(spacing: 6) {
                Text(name)
                    .font(.caption.monospaced().weight(.semibold))
                Text(ByteCountFormatter.string(
                    fromByteCount: Int64(size), countStyle: .file))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch preview {
        case .keyValues(let rows):
            KeyValuePreviewGrid(rows: rows)
        case .table(let header, let rows, let truncated):
            CSVPreviewTable(header: header, rows: rows, truncated: truncated)
        case .records(let records, let truncated):
            JSONLPreviewList(records: records, truncated: truncated)
        case .text(let text, let truncated):
            TextPreview(text: text, truncated: truncated)
        case .unavailable(let reason):
            Text(reason)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct KeyValuePreviewGrid: View {
    let rows: [RunBrowser.KeyValueRow]

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
            ForEach(rows) { row in
                GridRow {
                    Text(row.key)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .gridColumnAlignment(.trailing)
                    Text(row.value)
                        .font(.caption)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }
            }
        }
        .padding(.vertical, 2)
    }
}

private struct CSVPreviewTable: View {
    let header: [String]
    let rows: [[String]]
    let truncated: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                    headerRow
                    ForEach(rows.indices, id: \.self) { index in
                        dataRow(rows[index])
                    }
                }
                .font(.caption.monospaced())
                .padding(.vertical, 2)
            }
            if truncated {
                Text("first \(rows.count) rows — open in Finder for the full table")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var headerRow: some View {
        GridRow {
            ForEach(header.indices, id: \.self) { index in
                Text(header[index])
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func dataRow(_ row: [String]) -> some View {
        GridRow {
            ForEach(row.indices, id: \.self) { index in
                Text(row[index])
                    .textSelection(.enabled)
            }
        }
    }
}

private struct JSONLPreviewList: View {
    let records: [RunBrowser.RecordExcerpt]
    let truncated: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(records) { record in
                recordView(record)
            }
            if truncated {
                Text("first \(records.count) records — open in Finder for the full file")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func recordView(_ record: RunBrowser.RecordExcerpt) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let condition = record.condition {
                Text(condition)
                    .font(.caption.monospaced().weight(.semibold))
            }
            if let prompt = record.prompt {
                labeled("prompt", prompt)
            }
            if let output = record.output {
                labeled("output", output)
            }
            if let choice = record.choiceSummary {
                labeled("choice", choice)
            }
            if let fallback = record.fallback {
                Text(fallback)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 5))
    }

    private func labeled(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .trailing)
            Text(value)
                .font(.caption)
                .textSelection(.enabled)
        }
    }
}

private struct TextPreview: View {
    let text: String
    let truncated: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(text)
                .font(.caption.monospaced())
                .textSelection(.enabled)
            if truncated {
                Text("head only — open in Finder for the full file")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Remote (unpaired-server) Results browsing

/// Read-only browser over an UNPAIRED server's `runs/` tree: the server
/// provides the listing (stamps + file sizes) and bounded file reads; every
/// preview is parsed client-side by the same `RunBrowser` pure parsers as
/// local browsing. Import Evidence remains the way a remote run becomes
/// durable in this workspace.
private struct RemoteResultsBrowserView: View {
    @Bindable var service: ChatService
    @State private var filterText = ""
    @State private var runTypeFilter: String?

    private var runs: [RemoteStampedRunRecord] {
        service.experiments.results.remoteResultsRuns
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            content
        }
        .task { await service.experiments.refreshRemoteResultsRuns() }
    }

    @ViewBuilder
    private var content: some View {
        if runs.isEmpty {
            emptyState
        } else {
            browser
        }
    }

    private var serverLabel: String {
        service.cluster.substrateLabel
    }

    private var header: some View {
        HStack(spacing: 8) {
            headerTitle
            Spacer()
            runTypePicker
            TextField("filter (name, type, experiment, model)", text: $filterText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 220)
            refreshButton
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var headerTitle: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label("Server runs — \(serverLabel)", systemImage: "server.rack")
                .font(.headline)
            Text(headerCaption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private var headerCaption: String {
        let counts = "\(filtered.count) of \(runs.count) shown · read-only, newest first"
        guard let status = service.experiments.results.remoteResultsStatus else { return counts }
        return "\(counts) · \(status)"
    }

    private var refreshButton: some View {
        Button {
            Task { await service.experiments.refreshRemoteResultsRuns() }
        } label: {
            Label("Refresh", systemImage: "arrow.clockwise")
        }
        .disabled(service.experiments.results.isLoadingRemoteResults)
    }

    /// Run types present in the remote stamps — same filter mechanics as the
    /// local list, possible because the listing now carries the stamps. The
    /// same display mapping applies (server sweep runs are stamped "sweep"
    /// and read as optimizations here too).
    private var runTypes: [String] {
        Array(Set(runs.compactMap { RunBrowser.displayRunType(stamped: $0.runType) }))
            .sorted()
    }

    private var runTypePicker: some View {
        Picker("Type", selection: $runTypeFilter) {
            Text("all types").tag(String?.none)
            ForEach(runTypes, id: \.self) { runType in
                Text(runType).tag(String?.some(runType))
            }
        }
        .frame(maxWidth: 180)
        .help("filter by the server run's config.json runType stamp")
    }

    private var filtered: [RemoteStampedRunRecord] {
        var result = runs
        if let runTypeFilter {
            result = result.filter {
                RunBrowser.displayRunType(stamped: $0.runType) == runTypeFilter
            }
        }
        let needle = filterText.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return result }
        return result.filter { run in
            let displayType = RunBrowser.displayRunType(stamped: run.runType)
            return run.id.lowercased().contains(needle)
                || (displayType?.lowercased().contains(needle) ?? false)
                || (run.experiment?.lowercased().contains(needle) ?? false)
                || (run.modelID?.lowercased().contains(needle) ?? false)
        }
    }

    private var selectionBinding: Binding<String?> {
        Binding(
            get: { service.experiments.results.selectedRemoteResultsRun?.id },
            set: { newID in
                service.experiments.results.selectedRemoteResultsRun =
                    runs.first { $0.id == newID }
            })
    }

    private var browser: some View {
        VSplitView {
            runList
                .frame(minHeight: 140, idealHeight: 220)
            detailPane
                .frame(minHeight: 200, maxHeight: .infinity)
        }
    }

    private var runList: some View {
        List(filtered, selection: selectionBinding) { run in
            RemoteRunDirectoryRow(run: run)
                .tag(run.id)
        }
    }

    @ViewBuilder
    private var detailPane: some View {
        if let run = service.experiments.results.selectedRemoteResultsRun {
            RemoteRunDetailView(service: service, run: run)
        } else {
            ContentUnavailableView {
                Label("Select a server run", systemImage: "cursorarrow.click")
            } description: {
                Text("Click a run above to fetch its stamps and bounded previews from \(serverLabel).")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No server runs listed", systemImage: "archivebox")
        } description: {
            Text(emptyDescription)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyDescription: String {
        if let status = service.experiments.results.remoteResultsStatus {
            return "This unpaired server reported nothing to browse. \(status)"
        }
        return "Connect to \(serverLabel) to browse its immutable runs/ tree read-only."
    }
}

private struct RemoteRunDirectoryRow: View {
    let run: RemoteStampedRunRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(run.id)
                .font(.callout.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Text(detailLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    private var detailLine: String {
        // Same shared line as the local row (WS6.1) — the substrate badge is
        // what distinguishes a python-hf run from a swift-mlx one.
        RunBrowser.rowDetailLine(
            runType: RunBrowser.displayRunType(stamped: run.runType),
            substrate: run.substrate,
            experiment: run.experiment, modelID: run.modelID)
    }
}

/// Detail for one SERVER run: the same layout as the local detail pane —
/// stamp header, priority-ordered bounded previews, name+size list for the
/// rest — with fetches bounded by the `head=` param and JSON size-gated
/// from the listed size BEFORE any bytes move.
private struct RemoteRunDetailView: View {
    @Bindable var service: ChatService
    let run: RemoteStampedRunRecord
    @State private var previewed: [RemoteRunFilePreviewItem] = []
    @State private var other: [RemoteRunFileEntry] = []
    @State private var semanticModel: RunResults.Model?
    @State private var isLoading = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                stampHeader
                if isLoading {
                    loadingRow
                }
                if let semanticModel {
                    // F9: the FULL shared section stack — identical to the
                    // local surfaces, absent-artifact tolerant. The analyze
                    // action lives in the stamp header (RemoteRunAnalyzeRow).
                    RunSemanticSectionsContent(model: semanticModel)
                }
                ForEach(previewed) { item in
                    RunFilePreviewBox(
                        name: item.file.name, size: item.file.size,
                        preview: item.preview)
                }
                if !other.isEmpty {
                    otherFilesBox
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task(id: run.id) { await load() }
    }

    private var stampHeader: some View {
        VStack(alignment: .leading, spacing: 2) {
            stampTitleRow
            Text(stampLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            importRow
            // A3, remote half: python-hf study runs are analyzed on the
            // server that produced them (per-engine epoch guard).
            RemoteRunAnalyzeRow(
                service: service, runType: run.runType,
                substrate: run.substrate, experiment: run.experiment)
        }
    }

    private var stampTitleRow: some View {
        HStack(spacing: 8) {
            Text(run.id)
                .font(.callout.monospaced().weight(.semibold))
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            Text("on \(service.cluster.substrateLabel)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            RemoteResultsExplorerButton(runID: run.id)
        }
    }

    /// Making a remote run durable locally goes through Import Evidence —
    /// direct when the run directory carries its evidence bundle, otherwise
    /// via the producing job's row in the Compute section.
    @ViewBuilder
    private var importRow: some View {
        if ExperimentPanel.evidenceBundleFileName(in: run.files) != nil {
            HStack(spacing: 8) {
                Button {
                    Task { await service.experiments.importEvidence(fromServerRun: run) }
                } label: {
                    Label("Import Evidence", systemImage: "square.and.arrow.down")
                }
                .controlSize(.small)
                .help(
                    "download this run's evidence bundle, verify its hashes, "
                        + "and land it under this workspace's runs/ as an "
                        + "immutable imported run")
                importStatusText
            }
            .padding(.top, 4)
        } else {
            Text(
                "read-only view of the server's run — use Import Evidence on "
                    + "the producing job (Compute section) to make it durable "
                    + "in this workspace")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.top, 2)
        }
    }

    @ViewBuilder
    private var importStatusText: some View {
        if let status = service.experiments.results.remoteResultsStatus,
            status.contains("evidence")
        {
            Text(status)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private var stampLine: String {
        var parts: [String] = []
        if let runType = RunBrowser.displayRunType(stamped: run.runType) {
            parts.append(runType)
        }
        if let createdAt = run.createdAt { parts.append(createdAt) }
        if let model = run.modelID { parts.append(model) }
        if let revision = run.revision { parts.append("rev \(revision.prefix(12))") }
        if let experiment = run.experiment { parts.append("exp \(experiment)") }
        if let substrate = run.substrate { parts.append(substrate) }
        if parts.isEmpty { parts.append("no config.json stamp (legacy run type)") }
        return parts.joined(separator: " · ")
    }

    private var loadingRow: some View {
        HStack(spacing: 6) {
            ProgressView()
                .controlSize(.small)
            Text("fetching bounded previews from the server...")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var otherFilesBox: some View {
        GroupBox("Other files") {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(other) { file in
                    RemoteOtherFileRow(file: file)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func load() async {
        isLoading = true
        previewed = []
        other = []
        semanticModel = nil
        // ONE fetch pass feeds both the previews and the semantic model —
        // no double download; failures surface via remoteResultsStatus.
        let detail = await service.experiments.loadRemoteRunDetail(run: run)
        var shown = detail.previewed
        shown.sort { runFilePreviewPriority($0.file.name) < runFilePreviewPriority($1.file.name) }
        previewed = shown
        other = detail.other
        semanticModel = detail.model
        isLoading = false
    }
}

private struct RemoteOtherFileRow: View {
    let file: RemoteRunFileEntry

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "doc")
                .foregroundStyle(.secondary)
                .imageScale(.small)
            Text(file.name)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Text(ByteCountFormatter.string(
                fromByteCount: Int64(file.size), countStyle: .file))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}
