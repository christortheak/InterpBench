import AppKit
import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    @Bindable var service: ChatService
    let workspace: WorkspaceStore
    @State private var draft = ""
    @State private var stagedLongPrompt: String?
    /// The selected sidebar section. Launches to Home (design brief: the app
    /// should open on a workspace dashboard, not an empty transcript).
    @State private var section: WorkbenchSection = .home
    /// The Agents section's active region (Library / New Agent /
    /// Optimizations) — owned here so cross-section links can land on a
    /// specific region (e.g. Home's "Optimize" → Agents → Optimizations).
    @State private var agentsRegion: AgentsRegion = .library
    /// The Data section's active tool (Inventory / the three builders) —
    /// owned here for the same reason as `agentsRegion`: the display pane
    /// follows it (Inventory renders the selected row's detail; the builders
    /// render the shared Activity feed), and the pane lives outside the
    /// section.
    @State private var dataTool: DataSectionView.Tool = .inventory
    /// Pinned display-pane viewer: while set, the right-hand pane stays put
    /// as the user navigates (watch a long job while browsing elsewhere).
    /// The pin carries its ORIGIN section, so a pinned viewer showing one
    /// section's own data under another section says whose data it is
    /// (`WorkbenchViewerPin`, ExperimentKit — the decision logic is there so
    /// it can be unit-tested). Nothing else survives a section switch.
    @State private var viewerPin = WorkbenchViewerPin<ActivityViewerMode>()
    /// One-time hint for the send-as-assistant instrument ("the model will
    /// treat this as its own prior turn") — shown until the first seed.
    @AppStorage("SteerLabSeededTurnHintShown") private var seededTurnHintShown = false
    /// In-place transcript editing (context-menu "Edit turn…"): the turn
    /// being edited plus its provenance consequence, shown in a small sheet.
    @State private var editingTurn: EditingTurn?

    private struct EditingTurn: Identifiable {
        let id: UUID
        let role: ChatService.ChatMessage.Role
        let original: String
        let provenanceNote: String
        let branchUnavailableReason: String?
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detailSplit
        }
        .navigationTitle("SteerLab")
        .navigationSubtitle(subtitle)
        // Form's platform-default borderless fields disappear into our light
        // panel backgrounds and often inherit trailing alignment from
        // LabeledContent. Give every ordinary SwiftUI TextField in the
        // workbench a visible, left-aligned editing surface. Purpose-built
        // controls (the AppKit long-prompt editor, sliders, pickers) are
        // unaffected.
        .textFieldStyle(.roundedBorder)
        .multilineTextAlignment(.leading)
    }

    private var sidebar: some View {
        List(selection: sidebarSelection) {
            ForEach(WorkbenchSection.allCases) { item in
                Label(item.rawValue, systemImage: item.systemImage)
                    .tag(item)
                    .help(item.help)
                    // No type-select: typing a section's first letter while
                    // keyboard focus is NOT in a Playground field yanked the
                    // user to another tab mid-thought (live 2026-07-18).
                    .typeSelectEquivalent("")
            }
        }
        // ideal ≥ the widest label ("Multi-Agent" + icon) so section names
        // never truncate at launch (live-testing finding).
        .navigationSplitViewColumnWidth(min: 160, ideal: 185, max: 240)
    }

    /// List selection is optional; the workbench always has a section.
    private var sidebarSelection: Binding<WorkbenchSection?> {
        Binding(
            get: { section },
            set: { section = $0 ?? .home })
    }

    /// Pane order (live-testing finding): the CONTROLS sit next to the
    /// sidebar navigation; the VIEWER — the thing you watch (chat transcript,
    /// live logs, run summaries) — is on the right. The swap is positional
    /// only: modes, pinning, and chat-only-in-Playground are unchanged.
    ///
    /// Two floors have to hold here at once, and before the 2026-09-06 audit
    /// neither did (headline 1: the controls pane rendered at ≈444 pt in a
    /// 560 pt section, and the Results header wrapped to "Run directo-ries").
    ///
    /// 1. **The controls pane may never render below the section's floor.**
    ///    `HSplitView` keeps ONE divider position for the whole split and a
    ///    new minimum only constrains DRAGGING, not the position it is
    ///    already holding: a divider left at Playground's 340 pt floor stayed
    ///    at 340–450 pt after a switch to a 560 pt section, and stayed there
    ///    even when the window was zoomed wider. Re-identifying the split on
    ///    the floor itself makes SwiftUI build a fresh split view whenever the
    ///    floor changes, so the divider is recomputed against the new minimum
    ///    instead of being carried across. The cost is deliberate and small: a
    ///    section switch that crosses a floor boundary resets a divider the
    ///    researcher had dragged (and re-mounts a pinned viewer, whose content
    ///    lives in the stores, not in the view). Switches inside one floor —
    ///    every 560 pt section to every other — keep the divider.
    /// 2. **The viewer still takes the surplus.** It used to get that from
    ///    `.layoutPriority(1)` alone, which also made the CONTROLS pane the
    ///    one that yields whenever the two minimums did not both fit — the
    ///    other half of the same bug, because the window could restore under
    ///    its own floor (fixed with `.windowResizability(.contentMinSize)` and
    ///    a floor that fits the sidebar at its maximum, `SteerLabApp`). The
    ///    priority stays, but the controls pane now asks for exactly its floor
    ///    as its IDEAL width, so the surplus has one obvious home and the
    ///    stack offers the viewer `total − floor` rather than squeezing the
    ///    controls to make room for a viewer that wanted more.
    private var detailSplit: some View {
        HSplitView {
            sectionContent
                .frame(
                    minWidth: section.minimumContentWidth,
                    idealWidth: section.minimumContentWidth,
                    maxWidth: .infinity)
            activityViewerColumn
                .frame(minWidth: 420, maxWidth: .infinity)
                // The viewer takes the surplus at first layout; the divider
                // stays user-draggable.
                .layoutPriority(1)
        }
        .id(section.minimumContentWidth)
    }

    @ViewBuilder
    private var sectionContent: some View {
        switch section {
        case .home:
            HomeDashboardView(
                service: service, workspace: workspace, navigate: navigate,
                openOptimizations: openOptimizations,
                openAdapterTraining: openAdapterTraining,
                openAgent: openAgent)
        case .agents:
            ModelVariantsPanelView(
                service: service, region: $agentsRegion, navigate: navigate)
        case .playground:
            steeringPanel
        case .data:
            DataSectionView(service: service, tool: $dataTool)
        case .templates:
            TemplatesPanelView(service: service, navigate: navigate)
        case .studies:
            ExperimentsPanelView(
                service: service, openOptimizations: openOptimizations,
                openTemplates: { section = .templates },
                openCompute: { section = .compute })
        case .multiAgent:
            MultiAgentPanelView(service: service)
        case .results:
            ResultsPanelView(service: service)
        case .analysis:
            GeometryPanelView(service: service)
        case .compute:
            ComputeSectionView(service: service)
        }
    }

    private func navigate(_ target: WorkbenchSection) {
        section = target
    }

    /// Cross-section landing on the Agents section's Optimizations region —
    /// the returning-user path to declared/in-flight optimization runs.
    private func openOptimizations() {
        agentsRegion = .optimizations
        section = .agents
    }

    /// Cross-section landing on the Agents section's LIBRARY region — where
    /// the Data section's derived scope sends an agent row after preselecting
    /// it (`FineTuningPanel.selectAgent`). The region matters: the Library is
    /// the only region whose browser shows that selection.
    private func openAgentsLibrary() {
        agentsRegion = .library
        section = .agents
    }

    /// Cross-section landing on ONE agent: preselect it, then open the
    /// Library — the same two-step the Data section's derived scope performs.
    /// Home's agent rows use it so a listed agent is reachable from the
    /// dashboard (2026-09-06 audit: the rows were inert while study rows,
    /// which look identical, opened their study).
    private func openAgent(_ id: ModelVariantRecord.ID) {
        service.fineTuning.selectAgent(id: id)
        openAgentsLibrary()
    }

    /// Cross-section landing on Data › Adapter Training. `navigate(.data)`
    /// alone lands on whichever tool was last shown (2026-09-06 audit), so a
    /// button that names Adapter Training has to set the tool as well — the
    /// same shape as `openOptimizations` for Agents › Optimizations.
    private func openAdapterTraining() {
        dataTool = .adapterTraining
        section = .data
    }

    // MARK: Activity viewer (the contextual right-hand pane)

    /// The selected section's OWN viewer — region-aware within Agents
    /// (Library keeps the robustness viewer; New Agent / Optimizations
    /// stream the live Activity feed, where sweep logs land) and tool-aware
    /// within Data (Inventory renders the selected row; the builders stream
    /// that same Activity feed). Recomputed from the selection every time,
    /// which is what makes a section switch reset the pane: it holds no mode
    /// of its own to go stale.
    private var sectionViewerMode: ActivityViewerMode {
        ActivityViewerMode.mode(
            for: section, agentsRegion: agentsRegion, dataTool: dataTool)
    }

    /// Identity of the selected section's own viewer (title + owning
    /// section + whether its content belongs to that section alone).
    private var sectionViewerIdentity: WorkbenchViewerIdentity {
        sectionViewerMode.identity(ownedBy: section)
    }

    /// What the pane actually renders: the section's viewer, unless the
    /// researcher pinned one.
    private var effectiveViewerMode: ActivityViewerMode {
        viewerPin.resolvedMode(section: sectionViewerMode)
    }

    private var activityViewerColumn: some View {
        VStack(spacing: 0) {
            viewerHeaderBar
            Divider()
            viewerContent
        }
    }

    @ViewBuilder
    private var viewerContent: some View {
        switch effectiveViewerMode {
        case .chat:
            chatColumn
        case .multiAgentRun:
            multiAgentRunColumn
        case .agentRobustness:
            variantRobustnessColumn
        case .resultsRun:
            ResultsRunSummaryColumn(service: service)
        case .analysisGeometry:
            GeometryViewerColumn(service: service)
        case .dataInventory:
            DataInventoryDetailColumn(
                service: service,
                route: { request in
                    DataSectionRouting.route(
                        request, service: service, tool: $dataTool,
                        navigate: navigate,
                        openAgentsLibrary: openAgentsLibrary)
                })
        case .studyOverview:
            StudyViewerColumn(service: service)
        case .activity(let title):
            ActivityFeedColumn(service: service, title: title)
        }
    }

    private var viewerHeaderBar: some View {
        let identity = sectionViewerIdentity
        return HStack(spacing: 8) {
            Text(effectiveViewerMode.title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                // Says out loud whether this pane is workspace-wide
                // (Activity) or this section's own — the ambiguity that made
                // a shared feed read as one tab leaking into another.
                .help(viewerPin.scopeDescription(section: identity))
            // A pinned viewer showing a DIFFERENT section's own data names
            // that section; same-section and workspace-wide pins read just
            // "pinned" rather than claiming a false origin.
            if let badge = viewerPin.badge(section: identity) {
                Text(badge)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(1)
                    .help(viewerPin.scopeDescription(section: identity))
            }
            Spacer()
            Button {
                viewerPin.toggle(mode: effectiveViewerMode, identity: identity)
            } label: {
                Label(
                    viewerPin.isPinned ? "Unpin" : "Pin viewer",
                    systemImage: viewerPin.isPinned ? "pin.fill" : "pin")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help(viewerPin.pinControlHelp(section: identity))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.quaternary.opacity(0.5))
    }

    /// Substrate-first window subtitle (live-testing finding: which
    /// substrate is active must be unmistakable): "MLX (local) — …" or
    /// "Server: <label> — …" ahead of the model/connection detail.
    private var subtitle: String {
        switch service.cluster.computeTarget {
        case .local:
            return "MLX (local) — " + localSubtitleDetail
        case .server:
            // Connection line ONLY: activity is a full sentence and lives
            // in the transcript status card + the Model section, where it
            // can wrap — appended here it rendered truncated in the title
            // bar (live 2026-07-18).
            return "Server: " + service.cluster.substrateLabel + " — "
                + (service.cluster.status ?? "not connected")
        }
    }

    private var localSubtitleDetail: String {
        switch service.state {
        case .unloaded: return "no model loaded"
        case .loading(let percent): return "loading… \(percent)%"
        case .ready: return service.loadedModelID ?? "model ready"
        }
    }

    // MARK: Model loading (Steering pane — the window toolbar carries only
    // workspace concerns: the workspace switcher, connection status/popover,
    // and install-model; model selection is a property of the task at hand)

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    private var isLoading: Bool {
        if case .loading = service.state { return true }
        return false
    }

    /// The confirm button restates the size, so the destructive-ish choice
    /// names what it commits to.
    private func downloadConfirmTitle(
        _ pending: ChatService.PendingRemoteDownload
    ) -> String {
        if let bytes = pending.downloadBytes {
            return String(
                format: "Download ~%.1f GB and load", Double(bytes) / 1e9)
        }
        return "Download and load"
    }

    private var loadButtonTitle: String {
        switch service.cluster.computeTarget {
        case .local:
            switch service.localModelButtonAction {
            // Never "Load" for weights that are not here: a button named Load
            // must not start a multi-gigabyte download.
            case .download: "Download…"
            case .load, .busy: service.state == .ready ? "Reload" : "Load"
            }
        case .server: "Load on Server"
        }
    }

    private var loadButtonHelp: String {
        switch service.cluster.computeTarget {
        case .local:
            switch service.localModelButtonAction {
            case .download(let model):
                return "\(model) is not on this Mac — download its weights "
                    + "(several GB) into the model cache; Load enables when it "
                    + "finishes"
            case .load(let model):
                return "load \(model) from the local model cache; clears the chat"
            case .busy(let reason):
                return reason
            }
        case .server:
            return "load the selected model on \(service.cluster.substrateLabel)"
        }
    }

    /// Local: Load loads, Download downloads — one button, two verbs, never
    /// the second disguised as the first.
    private func runLoadButtonAction() {
        guard service.cluster.computeTarget == .local else {
            Task { await service.loadWorkspaceModel() }
            return
        }
        switch service.localModelButtonAction {
        case .download(let model):
            Task { await service.installWorkspaceModel(model) }
        case .load:
            Task { await service.loadWorkspaceModel() }
        case .busy:
            break
        }
    }

    private var loadButtonDisabled: Bool {
        switch service.cluster.computeTarget {
        case .local: isLoading || service.modelInstaller.isInstalling
        // Also disabled mid-load: repeated clicks launched overlapping load
        // streams, and switching models made whichever finished last win
        // (engineer review 2026-07-17). And disabled when the CURRENT
        // selection cannot fit the live session's GPU — a disabled picker
        // row alone does not clear an already-selected oversized model
        // (engineer review 2026-07-18).
        case .server:
            service.selectedRemoteModelID == nil
                || service.isRemoteModelLoading
                || SessionModelFit.tooBigNote(
                    cluster: service.cluster,
                    model: service.selectedRemoteModelID) != nil
        }
    }

    private var canSendPrompt: Bool {
        if service.isGenerating { return false }
        switch service.cluster.computeTarget {
        case .local:
            return service.state == .ready
        case .server:
            return service.serverHasLoadedModel
                || service.selectedRemoteVariantPath != nil
        }
    }

    // MARK: Chat column

    /// "Reset Chat" is irreversible, so it asks first whenever there is a
    /// transcript to lose.
    @State private var confirmingResetChat = false
    /// Save Agent's own outcome + busy flag: `FineTuningPanel.status` is a
    /// shared single slot (training progress writes it too), so Playground
    /// snapshots the line THIS action produced instead of mirroring the
    /// panel and showing an unrelated message.
    @State private var isSavingAgent = false
    @State private var saveAgentStatus: String?

    private var chatColumn: some View {
        VStack(spacing: 0) {
            if let error = service.errorMessage {
                errorBanner(error)
            }

            transcriptToolbar

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        // CONVERSATION TURNS ONLY (viewer-state hygiene,
                        // 2026-08-19). Live activity logs share this array
                        // but belong to the workspace-wide Activity feed:
                        // rendering them here put a vector build started in
                        // Data — or a study run started in Studies — into
                        // Playground as assistant bubbles, where they stayed
                        // until the next Send happened to clear them. The
                        // mirror of the fixed "Compute still shows my chat"
                        // bug; `ChatService` owns the split.
                        ForEach(service.conversationTranscript) { message in
                            MessageBubble(
                                message: message,
                                copyText: {
                                    service.transcriptTurnText(message)
                                },
                                editAction: service.canEditTranscriptMessage(message)
                                    ? { beginEditingTurn(message) }
                                    : nil,
                                removeAction: service.canRemoveTranscriptMessage(message)
                                    ? { service.removeLastTranscriptMessage() }
                                    : nil)
                                .id(message.id)
                        }
                        // Server activity gets the ROOM the chrome lacks: the
                        // heartbeat captions are full sentences ("staging
                        // 'google/gemma-3-4b-it' (8.0 GiB) to node-local disk
                        // (75s elapsed)") that render truncated in the title
                        // bar — during a load, the transcript pane is empty
                        // at exactly the minutes there is something to read.
                        // A rendered card, never a transcript entry.
                        if service.isRemoteModelLoading || service.isGenerating,
                            let activity = service.cluster.activity
                        {
                            HStack(alignment: .top, spacing: 8) {
                                ProgressView()
                                    .controlSize(.small)
                                Text(activity)
                                    .font(.callout)
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                            }
                            .padding(10)
                            .background(
                                .quaternary.opacity(0.5),
                                in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(12)
                }
                // Follow the rendered turns, not the raw array: scrolling to
                // a live-log id that this pane no longer renders is a no-op
                // that leaves the newest reply off-screen.
                .onChange(of: service.conversationTranscript.last?.text) {
                    if let last = service.conversationTranscript.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            Divider()

            VStack(spacing: 4) {
                if let stagedLongPrompt {
                    StagedLongPromptCard(text: stagedLongPrompt) {
                        self.stagedLongPrompt = nil
                    }
                }
                HStack(spacing: 8) {
                    // Send-as role: User sends+generates; Assistant SEEDS the
                    // text into the transcript as the model's own prior turn
                    // (the metacognition instrument) without generating.
                    Picker("Send as", selection: $service.composerRole) {
                        Text("User").tag(ChatService.ComposerRole.user)
                        Text("Assistant").tag(ChatService.ComposerRole.assistant)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    .help(
                        "send as User generates a reply (or Seed appends the "
                            + "user turn without generating); send as Assistant "
                            + "seeds the text into the transcript as the "
                            + "model's own prior turn (no generation)")

                    PromptInputBox(
                        text: $draft,
                        placeholder: inputPlaceholder,
                        disabled: !canSendPrompt || stagedLongPrompt != nil,
                        onSubmit: sendDraft,
                        onLongPaste: { pasted in
                            stagedLongPrompt = pasted.trimmingCharacters(in: .whitespacesAndNewlines)
                            draft = ""
                        }
                    )
                    .frame(minHeight: 34, maxHeight: 92)
                        .disabled(!canSendPrompt)
                        .accessibilityLabel("Message")
                        .accessibilityValue(activeDraftText)
                        .help(composerBoxHelp)

                    if service.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                        Button("Stop", role: .destructive) { service.stopGeneration() }
                            .keyboardShortcut(".", modifiers: .command)
                            .help(
                                "stop generating (⌘.) — the text produced so "
                                    + "far stays in the transcript and in the "
                                    + "session's history")
                    } else if service.composerRole == .assistant {
                        Button("Seed", action: sendDraft)
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(
                                service.seedUnavailableReason != nil
                                    || activeDraftText.isEmpty)
                            .help(
                                service.seedUnavailableReason
                                    ?? "append this text as an assistant turn "
                                    + "without generating — the model treats it "
                                    + "as its own prior turn")
                        Button("Continue") {
                            submitDraft {
                                seededTurnHintShown = true
                                service.continueSeededAssistantTurn($0)
                            }
                        }
                        .disabled(
                            service.continuationUnavailableReason != nil
                                || activeDraftText.isEmpty)
                        .help(
                            service.continuationUnavailableReason
                                ?? "seed this text as an INCOMPLETE assistant "
                                + "turn and have the model continue it mid-turn "
                                + "(prefill)")
                    } else {
                        Button("Send", action: sendDraft)
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(
                                !canSendPrompt || activeDraftText.isEmpty
                                    || service.turnConstraintReason(for: .user) != nil)
                            .help(
                                service.turnConstraintReason(for: .user)
                                    ?? composerUnavailableReason
                                    ?? "send this text as a user turn and "
                                    + "generate a reply")
                        // Seed works for BOTH roles: in User mode it appends
                        // the turn WITHOUT generating — scripted transcripts,
                        // and the canonical first move on user-first templates
                        // (gemma-3) before seeding an assistant reply.
                        Button("Seed") {
                            submitDraft {
                                seededTurnHintShown = true
                                service.seedUserTurn($0)
                            }
                        }
                        .disabled(
                            service.seedUserUnavailableReason != nil
                                || activeDraftText.isEmpty)
                        .help(
                            service.seedUserUnavailableReason
                                ?? "append this text as a user turn without "
                                + "generating (build a scripted transcript)")
                    }
                }
                composerCaptions
            }
            .padding(10)
        }
        .sheet(item: $editingTurn) { target in
            TranscriptTurnEditor(
                role: target.role,
                provenanceNote: target.provenanceNote,
                branchUnavailableReason: target.branchUnavailableReason,
                text: target.original,
                onSave: { service.editTranscriptMessage(id: target.id, newText: $0) },
                onContinueAssistant: {
                    service.continueGenerationFromAssistant(id: target.id, newText: $0)
                },
                onReturnControl: {
                    service.returnControlToUserFromAssistant(id: target.id, newText: $0)
                },
                onRestartUser: {
                    service.restartConversationFromUser(id: target.id, newText: $0)
                })
        }
    }

    /// The composer's conditional captions — the turn-constraint refusal,
    /// the one-time seeding hint, the seed refusal, the draft-size line —
    /// in ONE line-capped container. They sit directly in a split-view
    /// column, so their height must not grow with the length of a server
    /// message (macOS 27 split-view rule); each line caps at two lines and
    /// keeps its full text in the tooltip.
    @ViewBuilder
    private var composerCaptions: some View {
        VStack(alignment: .leading, spacing: 2) {
            if service.composerRole == .user,
                let reason = service.turnConstraintReason(for: .user)
            {
                composerCaption(reason, tone: .orange)
            }
            if service.composerRole == .assistant, !seededTurnHintShown {
                composerCaption(seedingHint, tone: .secondary)
            }
            if service.composerRole == .assistant,
                let reason = service.seedUnavailableReason
            {
                composerCaption(reason, tone: .orange)
            }
            if !activeDraftText.isEmpty {
                composerCaption(draftSizeText, tone: draftSizeTone)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func composerCaption(_ text: String, tone: Color) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(tone)
            .lineLimit(2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .help(text)
    }

    private var seedingHint: String {
        "Seeding puts words in the model's mouth: the model will treat this "
            + "text as its own prior turn (rendered through its real chat "
            + "template). Seeded turns are badged in the transcript."
    }

    private var draftSizeTone: Color {
        activeDraftText.count >= ChatService.longPromptStudyPathCharacterThreshold
            ? .orange : .secondary
    }

    /// What the box itself does — Return sends, Shift-Return keeps typing,
    /// and a very long paste is staged as a whole prompt rather than pasted
    /// into the field.
    private var composerBoxHelp: String {
        if stagedLongPrompt != nil {
            return "a long prompt is staged below — Discard staged prompt to "
                + "type here again"
        }
        return "Return sends, Shift-Return starts a new line; pasting more "
            + "than \(ChatService.longPromptStudyPathCharacterThreshold.formatted()) "
            + "characters stages the paste as a long prompt instead of "
            + "filling the box"
    }

    /// Send / seed / edit / download failures. Selectable (a refusal is
    /// usually something to paste into a note or a bug report) and
    /// dismissable — it used to persist until the next successful action
    /// with no way to clear it. Line-capped so this column's minimum height
    /// cannot grow with the message length (split-view rule); the whole text
    /// stays reachable by selection and in the tooltip.
    private func errorBanner(_ error: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .accessibilityHidden(true)
            Text(error)
                .textSelection(.enabled)
                .lineLimit(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button {
                service.errorMessage = nil
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .help("dismiss this message — nothing is undone or retried")
            .accessibilityLabel("Dismiss message")
        }
        .foregroundStyle(.white)
        .padding(6)
        .frame(maxWidth: .infinity)
        .background(.red.opacity(0.8))
        .help(error)
    }

    private func beginEditingTurn(_ message: ChatService.ChatMessage) {
        editingTurn = EditingTurn(
            id: message.id,
            role: message.role,
            original: message.text,
            provenanceNote: Self.editProvenanceNote(for: message),
            branchUnavailableReason: message.role == .assistant
                ? service.continuationFromEditUnavailableReason
                : service.restartFromEditUnavailableReason)
    }

    /// What saving an edit does to the turn's provenance record — shown in
    /// the editor so the researcher consents to the flag, never surprised
    /// by it (rules live in `ChatService.editTranscriptMessage`).
    private static func editProvenanceNote(for message: ChatService.ChatMessage) -> String {
        guard message.role == .assistant else {
            return "User turns are always researcher-authored — editing adds "
                + "no provenance flag."
        }
        if message.seededPrefixLength != nil {
            return "This turn was seeded, then continued by the model. Saving "
                + "keeps \u{201C}seeded\u{201D}, adds \u{201C}edited\u{201D}, and drops the prefix "
                + "marker (it would no longer describe the text)."
        }
        if message.seeded {
            return "This seeded turn stays \u{201C}seeded\u{201D} — it remains wholly "
                + "researcher-authored."
        }
        return "This turn was generated. Saving marks it \u{201C}edited\u{201D} "
            + "(generated, then altered by researcher) — never again "
            + "indistinguishable from a pure generation."
    }

    private var multiAgentRunColumn: some View {
        let panel = service.multiAgent
        return VStack(spacing: 0) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Multi-Agent Run")
                        .font(.headline)
                    Text(panel.name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .help(panel.name)
                }
                Spacer()
                if panel.isRunning {
                    ProgressView()
                        .controlSize(.small)
                    Button(role: .destructive) {
                        panel.stopScenarioRun()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .help(
                        "stop this multi-agent run after the current "
                            + "generation finishes — turns already written "
                            + "stay in the run directory")
                }
                CopyButton(
                    "Copy Run",
                    help: "copy the visible multi-agent run — its turns, "
                        + "warnings, and failure — to the clipboard as Markdown"
                ) { multiAgentRunMarkdown(panel) }
                    .disabled(
                        panel.liveTurnResults.isEmpty && panel.liveActiveTurn == nil
                            && panel.liveRunWarnings.isEmpty
                            && panel.liveRunFailure == nil)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary.opacity(0.35))

            if let liveRunDirectory = panel.liveRunDirectory {
                Text(liveRunDirectory)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(liveRunDirectory)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        // Failure and warnings live INSIDE the scroll view:
                        // above it they were conditional, unbounded rows
                        // directly in a split-view column, whose minimum
                        // height then moved with async state (macOS 27
                        // split-view rule).
                        if let failure = panel.liveRunFailure {
                            Text(failure)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    .red.opacity(0.08),
                                    in: RoundedRectangle(cornerRadius: 8))
                        }
                        if !panel.liveRunWarnings.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                ForEach(
                                    Array(panel.liveRunWarnings.enumerated()),
                                    id: \.offset
                                ) { _, warning in
                                    Text(warning.message)
                                        .font(.caption)
                                        .foregroundStyle(.orange)
                                        .textSelection(.enabled)
                                }
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                .orange.opacity(0.08),
                                in: RoundedRectangle(cornerRadius: 8))
                        }
                        if panel.liveTurnResults.isEmpty, panel.liveActiveTurn == nil,
                            panel.liveRunFailure == nil
                        {
                            ContentUnavailableView(
                                "No run yet",
                                systemImage: "person.3.sequence",
                                description: Text("Run the selected scenario to watch turns appear here."))
                                .frame(maxWidth: .infinity, minHeight: 360)
                        }
                        ForEach(panel.liveTurnResults) { result in
                            MultiAgentTurnBubble(result: result)
                                .id("result-\(result.id)")
                        }
                        if let active = panel.liveActiveTurn {
                            MultiAgentActiveTurnBubble(turn: active)
                                .id("active-turn")
                        }
                    }
                    .padding(12)
                }
                .onChange(of: panel.liveTurnResults.count) {
                    if let last = panel.liveTurnResults.last {
                        proxy.scrollTo("result-\(last.id)", anchor: .bottom)
                    }
                }
                .onChange(of: panel.liveActiveTurn?.output) {
                    if panel.liveActiveTurn != nil {
                        proxy.scrollTo("active-turn", anchor: .bottom)
                    }
                }
            }
        }
    }

    private var variantRobustnessColumn: some View {
        let panel = service.fineTuning
        return VStack(spacing: 0) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Agent Robustness")
                        .font(.headline)
                    Text(panel.robustnessTargetName ?? "No agent chosen")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .help(panel.robustnessTargetName ?? "no agent chosen yet")
                }
                Spacer()
                if panel.isRobustnessRunning {
                    ProgressView()
                        .controlSize(.small)
                }
                CopyButton(
                    "Copy Outputs",
                    help: "copy the visible robustness outputs, judge results, "
                        + "and summary to the clipboard as Markdown"
                ) { variantRobustnessMarkdown(panel) }
                    .disabled(
                        panel.liveRobustnessOutputs.isEmpty
                            && panel.liveRobustnessJudgments.isEmpty
                            && panel.robustnessReport == nil
                            && panel.liveRobustnessFailure == nil)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary.opacity(0.35))

            if let path = panel.lastRobustnessDirectory {
                Text(path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        // Inside the scroll view, not above it: as
                        // conditional multi-line rows in a split-view column
                        // these moved the column's minimum height with async
                        // state (macOS 27 split-view rule).
                        if let failure = panel.liveRobustnessFailure {
                            Text(failure)
                                .font(.caption)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    .red.opacity(0.08),
                                    in: RoundedRectangle(cornerRadius: 8))
                        }
                        if let judgeStatus = panel.liveRobustnessJudgeStatus {
                            Text(judgeStatus)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if panel.liveRobustnessOutputs.isEmpty,
                            panel.robustnessReport == nil,
                            panel.liveRobustnessFailure == nil
                        {
                            ContentUnavailableView(
                                "No robustness output yet",
                                systemImage: "checklist.checked",
                                description: Text("Run a robustness check in the Agents section to watch baseline and agent outputs here."))
                                .frame(maxWidth: .infinity, minHeight: 360)
                        }
                        ForEach(panel.liveRobustnessOutputs) { output in
                            VariantRobustnessOutputBubble(output: output)
                                .id(output.id)
                        }
                        ForEach(panel.liveRobustnessJudgments) { judgment in
                            VariantRobustnessJudgeBubble(judgment: judgment)
                                .id(judgment.id)
                        }
                        if let report = panel.robustnessReport {
                            VariantRobustnessSummaryCard(report: report)
                                .id("robustness-summary")
                        }
                    }
                    .padding(12)
                }
                .onChange(of: panel.liveRobustnessOutputs.count) {
                    if let last = panel.liveRobustnessOutputs.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .onChange(of: panel.liveRobustnessOutputs.last?.output) {
                    if let last = panel.liveRobustnessOutputs.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .onChange(of: panel.liveRobustnessJudgments.count) {
                    if let last = panel.liveRobustnessJudgments.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .onChange(of: panel.robustnessReport != nil) {
                    if panel.robustnessReport != nil {
                        proxy.scrollTo("robustness-summary", anchor: .bottom)
                    }
                }
            }
        }
    }

    private var transcriptToolbar: some View {
        HStack(spacing: 8) {
            Text("Transcript")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Reset Chat") {
                if service.conversationTranscript.isEmpty {
                    resetChat()
                } else {
                    confirmingResetChat = true
                }
            }
            // Same predicate as Send: server-mode chats (remote model or
            // agent, no local model loaded) must be resettable too.
            .disabled(!canSendPrompt)
            .help(
                "clear the conversation and its KV cache and dismiss the "
                    + "error banner; steering settings are kept. Asks first "
                    + "when the transcript has turns")
            // The export carries conversation turns only, so an activity log
            // alone must not enable it.
            CopyButton(
                "Copy Transcript",
                help: "copy the full Playground transcript, with the current "
                    + "model, adapter, vector, and probe settings, to the "
                    + "clipboard"
            ) { service.transcriptMarkdown() }
                .disabled(service.conversationTranscript.isEmpty)
            Button {
                downloadTranscript()
            } label: {
                Label("Download Transcript", systemImage: "square.and.arrow.down")
            }
            .disabled(service.conversationTranscript.isEmpty)
            .help("save the full Playground transcript as a Markdown file")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.quaternary.opacity(0.35))
        .confirmationDialog(
            "Clear this chat transcript?",
            isPresented: $confirmingResetChat,
            titleVisibility: .visible
        ) {
            Button("Clear Transcript", role: .destructive) { resetChat() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(resetChatConsequence)
        }
    }

    private var resetChatConsequence: String {
        let count = service.conversationTranscript.count
        return "\(count) turn\(count == 1 ? "" : "s") and the session's KV "
            + "cache are discarded — this cannot be undone. The loaded model, "
            + "the selected agent, and every steering setting are kept. Copy "
            + "or download the transcript first if you need it."
    }

    /// Reset clears the error banner too: a refusal from the conversation
    /// just discarded survived the reset and read as a fresh failure.
    private func resetChat() {
        service.errorMessage = nil
        service.resetChat()
    }

    private var inputPlaceholder: String {
        // A staged long prompt makes the box read-only: say THAT, instead of
        // inviting a message the box will not accept.
        if stagedLongPrompt != nil {
            return "long prompt staged — Discard staged prompt to type here"
        }
        // Server mode generates remotely: local model state is irrelevant.
        if service.cluster.computeTarget == .server {
            let serverReady =
                service.serverHasLoadedModel
                || service.selectedRemoteVariantPath != nil
            return serverReady
                ? "message (server: \(service.cluster.serverHostLabel))"
                : "load a server model or select a server agent to start chatting"
        }
        switch service.state {
        case .unloaded: return "load a model to start chatting"
        case .loading: return "loading model…"
        case .ready:
            return service.steeringActive
                ? "message (steered: \(service.activeSlotCount) vector\(service.activeSlotCount == 1 ? "" : "s"))"
                : "message"
        }
    }

    /// Return in the composer, and the Send/Seed buttons, run through here.
    ///
    /// It used to clear `draft` BEFORE calling the service, and the key
    /// handler submitted without consulting the buttons' disabled predicate
    /// — so a refused send (consecutive user turns on a user-first template,
    /// a server without `chat.seededTurns`) destroyed the typed text and
    /// left only the red banner. Now the same predicate the button reads
    /// gates the submit, and the draft is cleared only once the service has
    /// actually accepted the turn.
    private func sendDraft() {
        guard !activeDraftText.isEmpty else { return }
        switch service.composerRole {
        case .user:
            if let reason = composerUnavailableReason
                ?? service.turnConstraintReason(for: .user)
            {
                service.errorMessage = reason
                return
            }
            submitDraft { service.send($0) }
        case .assistant:
            if let reason = service.seedUnavailableReason {
                service.errorMessage = reason
                return
            }
            // Seed only — the whole point: the turn enters the transcript
            // without generating; the NEXT user turn replays it as history.
            submitDraft {
                service.seedAssistantTurn($0)
                seededTurnHintShown = true
            }
        }
    }

    /// Runs one composer action and clears the draft ONLY if the transcript
    /// grew, i.e. the service accepted the turn. Every send/seed entry point
    /// in `ChatService` early-returns with `errorMessage` set on a refusal;
    /// the typed text must survive that.
    private func submitDraft(_ action: (String) -> Void) {
        let text = activeDraftText
        let before = service.conversationTranscript.count
        action(text)
        guard service.conversationTranscript.count > before else { return }
        draft = ""
        stagedLongPrompt = nil
    }

    /// Why the composer cannot send at all right now (nil = it can) — the
    /// same rule `canSendPrompt` encodes, said in words so Return does not
    /// fail silently.
    private var composerUnavailableReason: String? {
        if service.isGenerating { return "wait for the current generation to finish" }
        switch service.cluster.computeTarget {
        case .local:
            return service.state == .ready
                ? nil : "load a model before sending a turn"
        case .server:
            return service.serverHasLoadedModel
                || service.selectedRemoteVariantPath != nil
                ? nil
                : "load a server model or select a server agent before "
                    + "sending a turn"
        }
    }

    private var draftSizeText: String {
        let text = activeDraftText
        let words = text.split(whereSeparator: \.isWhitespace).count
        let route =
            text.count >= ChatService.longPromptStudyPathCharacterThreshold
            ? " · sent in chunks, like a study prompt"
            : ""
        return "\(text.count.formatted()) chars · ~\(words.formatted()) words\(route)"
    }

    private var activeDraftText: String {
        stagedLongPrompt ?? draft
    }

    private func multiAgentRunMarkdown(_ panel: MultiAgentPanel) -> String {
        var lines: [String] = [
            "# \(panel.name)",
            "",
        ]
        if let path = panel.liveRunDirectory {
            lines.append("Run directory: \(path)")
            lines.append("")
        }
        if let failure = panel.liveRunFailure {
            lines.append("Failure: \(failure)")
            lines.append("")
        }
        for warning in panel.liveRunWarnings {
            lines.append("Warning: \(warning.message)")
            lines.append("")
        }
        for result in panel.liveTurnResults {
            lines.append("## \(result.turnIndex). \(result.title)")
            lines.append("")
            lines.append("Speaker: \(result.speakerName)")
            lines.append("Output label: \(result.outputLabel)")
            lines.append("")
            lines.append(result.output)
            lines.append("")
        }
        if let active = panel.liveActiveTurn {
            lines.append("## \(active.index). \(active.title)")
            lines.append("")
            lines.append("Speaker: \(active.speaker)")
            lines.append("")
            lines.append(active.output)
        }
        return lines.joined(separator: "\n")
    }

    private func variantRobustnessMarkdown(_ panel: FineTuningPanel) -> String {
        var lines: [String] = [
            "# Agent Robustness",
            "",
            "Agent: \(panel.selectedVariant?.artifact.name ?? "none")",
            "",
        ]
        if let path = panel.lastRobustnessDirectory {
            lines.append("Run directory: \(path)")
            lines.append("")
        }
        if let failure = panel.liveRobustnessFailure {
            lines.append("Failure: \(failure)")
            lines.append("")
        }
        for output in panel.liveRobustnessOutputs {
            lines.append("## \(output.title)")
            lines.append("")
            lines.append("Prompt:")
            lines.append(output.prompt)
            lines.append("")
            lines.append("Output:")
            lines.append(output.output)
            lines.append("")
        }
        for judgment in panel.liveRobustnessJudgments {
            lines.append("## Judge \(judgment.index): \(judgment.result)")
            lines.append("")
            lines.append("Prompt:")
            lines.append(judgment.prompt)
            lines.append("")
            lines.append("Winner: \(judgment.response.winner)")
            lines.append("Confidence: \(judgment.response.confidence)")
            lines.append("Reason: \(judgment.response.briefReason)")
            lines.append("")
        }
        if let report = panel.robustnessReport {
            lines.append("## Summary")
            lines.append("")
            lines.append("Capability: \(percent(report.variantBatteryAccuracy)) variant · \(percent(report.baselineBatteryAccuracy)) baseline")
            lines.append("Distinct-2: \(report.meanVariantDistinct2) variant · \(report.meanBaselineDistinct2) baseline")
            if !report.warnings.isEmpty {
                lines.append("Warnings: \(report.warnings.joined(separator: "; "))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private func percent(_ value: Float) -> String {
        value.formatted(.percent.precision(.fractionLength(0)))
    }

    private func downloadTranscript() {
        let panel = NSSavePanel()
        if let markdownType = UTType(filenameExtension: "md") {
            panel.allowedContentTypes = [markdownType, .plainText]
        } else {
            panel.allowedContentTypes = [.plainText]
        }
        panel.nameFieldStringValue = "steerlab-transcript-\(Int(Date().timeIntervalSince1970)).md"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try service.transcriptMarkdown().write(to: url, atomically: true, encoding: .utf8)
        } catch {
            service.errorMessage =
                "could not save the transcript to \(url.lastPathComponent): "
                + error.localizedDescription
        }
    }

    /// Local install/load progress. The install is a network fetch measured in
    /// gigabytes and the load is a cache read; both now say so here rather
    /// than leaving the section mute while the button sits disabled.
    @ViewBuilder
    private var localModelProgressRows: some View {
        if !isServerWorkspace {
            if service.modelInstaller.isInstalling {
                ProgressView(value: localInstallFraction)
            }
            if let status = service.modelInstaller.statusLine {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(
                        service.modelInstaller.isFailed ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if case .loading(let percent) = service.state {
                Text("loading from the local model cache — \(percent)%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var localInstallFraction: Double {
        if case .installing(_, let percent) = service.modelInstaller.phase {
            return Double(percent) / 100
        }
        return 0
    }

    // MARK: Steering panel

    private var steeringPanel: some View {
        Form {
            // Model selection is a property of whatever this pane is about to
            // do, not a whole-app setting: one workspace-scoped picker — the
            // local tiers in Local, the active server's installed models in a
            // server workspace — with the load action beside it. Workspace
            // switching itself stays in the window toolbar.
            Section("Model") {
                // Frozen mid-load: switching the selection during an active
                // remote load invited a second overlapping load (the
                // single-flight guard refuses it, but a disabled picker says
                // WHY the selection cannot change until Cancel Load).
                WorkspaceModelPicker(service: service)
                    .disabled(service.isRemoteModelLoading)
                HStack(spacing: 8) {
                    Button(loadButtonTitle) { runLoadButtonAction() }
                        .disabled(loadButtonDisabled)
                        .help(loadButtonHelp)
                    if isServerWorkspace {
                        InstallModelButton(cluster: service.cluster)
                    } else {
                        AddLocalModelButton(service: service)
                    }
                    if service.isRemoteModelLoading {
                        Button("Cancel Load") { service.cancelRemoteLoad() }
                            .help(
                                service.cluster.capabilities?.supportsLoadCancel
                                    == true
                                    ? "stop this load on the engine too — a "
                                        + "download stops within seconds, a "
                                        + "weight copy at its next phase "
                                        + "boundary, and the resident slot "
                                        + "frees"
                                    : "stop waiting for this load — the server "
                                        + "may still finish it and keep the "
                                        + "model resident")
                    }
                    if service.modelInstaller.isInstalling {
                        Button("Cancel Download") { service.modelInstaller.cancel() }
                            .help(
                                "stop the download — files already fetched stay "
                                    + "in the cache and a re-run resumes")
                    }
                    Spacer()
                }
                // The local half of the load story, which used to be invisible:
                // download progress and, when it fails, the reason — in the
                // section, not only in the window subtitle.
                localModelProgressRows
                // The selected model's memory-fit refusal, named where the
                // disabled Load button would otherwise be mute.
                if let note = SessionModelFit.tooBigNote(
                    cluster: service.cluster,
                    model: service.selectedRemoteModelID)
                {
                    Text("\(service.selectedRemoteModelID ?? "model") is \(note)"
                        + " — pick a smaller model or a bigger session GPU")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                // Live load progress, WRAPPED — the server's heartbeat
                // captions are full sentences and truncated everywhere else.
                if service.isRemoteModelLoading,
                    let activity = service.cluster.activity
                {
                    Text(activity)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack(spacing: 6) {
                    Image(
                        systemName: isServerWorkspace
                            ? "server.rack" : "desktopcomputer")
                    Text(computeStatusLine)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                // Carries the full, untruncated line: the model id in the
                // middle is exactly what the truncation eats.
                .help(
                    computeStatusLine + " — the workspace is global; switch "
                        + "it with the Compute selector in the window toolbar")
            }
            // A server load that would really be a multi-GB download asks
            // first (field incident 2026-08-29: an uncached 27B silently
            // began a ~55 GB hub download that held the engine's only
            // resident-model slot). Same rule the local tier enforces with
            // its separate Download… verb.
            .confirmationDialog(
                "Download model weights first?",
                isPresented: Binding(
                    get: { service.pendingRemoteDownload != nil },
                    set: { if !$0 { service.dismissPendingRemoteDownload() } }),
                titleVisibility: .visible,
                presenting: service.pendingRemoteDownload
            ) { pending in
                Button(downloadConfirmTitle(pending)) {
                    service.beginConfirmedRemoteLoad()
                }
                Button("Cancel", role: .cancel) {
                    service.dismissPendingRemoteDownload()
                }
            } message: { pending in
                Text(pending.confirmationText)
            }

            // GPU session (plan §2.7): conspicuous, adjacent to the model
            // picker the session is sized to. Capability-gated — older
            // servers without chat.gpuSession never show the controls.
            if isServerWorkspace,
                service.cluster.capabilities?.supportsGPUSession == true
            {
                GPUSessionSection(service: service)
            }

            // Broken into small @ViewBuilder chunks: one flat Section body
            // with this many conditional controls exceeds the type-checker's
            // budget (the classic SwiftUI too-complex-expression failure).
            Section("Steering controls") {
                variantPickerControls
                injectionMasterControls
                adapterControls
                neutralDirectionControls
                probeControls
                generationControls
            }
            // Dimmed + disabled while the master switch is off: the boxes
            // hold the configured mix, but nothing applies or edits until
            // injection is on (one unambiguous "is this chat steered?").
            // Live in BOTH workspaces: server boxes drive the composed
            // variant spec exactly as local boxes drive the MLX injectors.
            ForEach($service.slots) { $slot in
                slotSection($slot)
                    .disabled(!service.steeringEnabled)
                    .opacity(service.steeringEnabled ? 1 : 0.45)
            }

            Section {
                Button("Add Vector") { service.addSlot() }
                    .disabled(workspaceVectorsEmpty || !service.steeringEnabled)
                    .help(
                        "add another steering box (inherits this one's settings). All "
                            + "enabled boxes inject simultaneously as a linear combination — "
                            + "composition (fear+authority), counter-vectors, or purification "
                            + "via a negative α on a control concept")
            }

            Section("Save Agent") {
                TextField(
                    "agent name",
                    text: Binding(
                        get: { service.fineTuning.variantName },
                        set: { service.fineTuning.variantName = $0 }))
                    .help(
                        "the name this agent gets in Agents › Library; it also "
                            + "names the definition file, so keep it short and "
                            + "distinct")
                Button(isSavingAgent ? "Saving…" : "Save Agent") {
                    guard !isSavingAgent else { return }
                    // Async: the server-workspace path may refresh the vector
                    // catalog once before deciding to save or refuse.
                    isSavingAgent = true
                    saveAgentStatus = nil
                    Task {
                        await service.fineTuning.captureVariant()
                        // The panel's status slot is shared; snapshot the
                        // line THIS action just wrote.
                        saveAgentStatus = service.fineTuning.status
                        isSavingAgent = false
                    }
                }
                // Workspace-runnable gate, not local-only state: in a server
                // workspace the button is live exactly when chatting is.
                .disabled(!service.workspaceHasRunnableModel || isSavingAgent)
                .help(
                    isServerWorkspace
                        ? "save the Playground's current configuration as an "
                            + "agent definition (git-versioned recipe, stored "
                            + "locally) recording the server base model and server "
                            + "vector/adapter refs"
                        : "save the Playground's current configuration as an "
                            + "agent definition (stored locally as a variant "
                            + "artifact); it appears in Agents › Library")
                // The gate, visible rather than only in the tooltip.
                if !service.workspaceHasRunnableModel {
                    Text(
                        isServerWorkspace
                            ? "load a server model, or select a server agent, "
                                + "before saving — the definition records what "
                                + "it was built on"
                            : "load a model before saving — the definition "
                                + "records the base model it was built on")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
                // The outcome, where the action is. It used to land only in
                // the Agents section's status line and the bell.
                if let saveAgentStatus {
                    Text(saveAgentStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }

            if workspaceVectorsEmpty {
                Section {
                    Text(
                        isServerWorkspace
                            ? "No vectors on \(service.cluster.substrateLabel) for the "
                                + "selected model. Extract one in Data (it runs as "
                                + "a server job), then Refresh artifacts."
                            : service.state == .ready
                                ? "No vectors for this model in runs/. Extract one with:\n"
                                    + "steerlab-cli --config prompts/configs/toy-french.json\n"
                                    + "then Refresh artifacts."
                                : "Load a model to see its vectors."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    /// Vector availability for the ACTIVE workspace — strict per-substrate:
    /// the server catalog in a server workspace, never a merged view.
    private var workspaceVectorsEmpty: Bool {
        isServerWorkspace
            ? service.compatibleServerVectors.isEmpty
            : service.compatibleVectors.isEmpty
    }

    // MARK: Steering section chunks (extracted so the Form type-checks)

    /// ONE agent picker, workspace-scoped (rows from the tested rule
    /// `ChatService.agentPickerRows`): a server workspace lists the server's
    /// stored agents PLUS local definitions (inline specs) — manually-created
    /// agents included; Local lists local definitions. Excluded rows stay
    /// VISIBLE but disabled, naming why. Selection binds to
    /// `playgroundAgentSelection`, the same state the Agents tab's Chat
    /// handoff sets — so a handoff both seeds the controls AND selects here.
    @ViewBuilder
    private var variantPickerControls: some View {
        Picker(
            "Agent",
            selection: Binding<ChatService.PlaygroundAgentSelection?>(
                get: { service.playgroundAgentSelection },
                set: { selection in
                    Task { await service.selectPlaygroundAgent(selection) }
                }))
        {
            Text("None").tag(ChatService.PlaygroundAgentSelection?.none)
            // A selection whose row vanished (stale server listing, deleted
            // record) stays rendered — SwiftUI must not silently blank the
            // binding — but cannot be re-picked.
            if let current = service.playgroundAgentSelection,
                !service.playgroundAgentPickerRows.contains(where: {
                    $0.selection == current
                })
            {
                Text(currentAgentFallbackLabel(current))
                    .tag(Optional(current))
                    .selectionDisabled()
            }
            ForEach(service.playgroundAgentPickerRows) { row in
                if let reason = row.exclusionReason {
                    Text("\(row.title) (\(reason))")
                        .tag(Optional(row.selection))
                        .selectionDisabled()
                } else {
                    Text(row.title).tag(Optional(row.selection))
                }
            }
        }
        .help(agentPickerHelp)
        if isServerWorkspace {
            // Live stored-vs-inline answer: an edited seed visibly
            // stops claiming the stored variant before anything sends.
            Text(service.serverSendModeCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Toggle(
                "Strip agent interventions",
                isOn: $service.stripRemoteVariantInterventions
            )
            .disabled(service.selectedRemoteVariantPath == nil)
            .help(
                "ask the server to run the selected agent's base model "
                    + "without LoRA adapters or steering vectors")
            Button("Upload Local Agent") {
                Task { await service.uploadSelectedVariantToServer() }
            }
            .disabled(
                service.selectedServerLocalDefinitionID == nil
                    && service.selectedSteeringVariant == nil)
            .help(
                "store the selected local agent definition on the server "
                    + "(sends then carry the stored path + hash identity); "
                    + "model weights stay server-side")
        }
    }

    private var agentPickerHelp: String {
        isServerWorkspace
            ? "agents stored on \(service.cluster.substrateLabel) (base model "
                + "+ adapters + injections applied server-side) plus local "
                + "agent definitions runnable there as inline specs. Selecting "
                + "one seeds the live controls below — edit them and the send "
                + "switches from the stored agent to an inline spec. Rows the "
                + "server cannot run are disabled with the reason"
            : "saved agents (definitions); picking one loads its configuration "
                + "into the live steering controls. Agents built for a "
                + "different base model stay listed but disabled until that "
                + "model is loaded"
    }

    private func currentAgentFallbackLabel(
        _ selection: ChatService.PlaygroundAgentSelection
    ) -> String {
        switch selection {
        case .server(let path):
            return "\(URL(filePath: path).lastPathComponent) "
                + "(not in the server's current listing)"
        case .localDefinition(let id):
            return "\(URL(filePath: id).deletingLastPathComponent().lastPathComponent) "
                + "(definition not found)"
        }
    }

    /// Master injection switch + the settings shared by every vector box.
    /// Help strings live in typed String properties below — inline
    /// conditional concatenations blow the view type-checker's budget.
    @ViewBuilder
    private var injectionMasterControls: some View {
        Toggle("Inject vectors", isOn: $service.steeringEnabled)
            .help(injectVectorsHelp)

        Stepper(
            "Layer band: \(service.layerBandWidth)",
            value: $service.layerBandWidth, in: 1 ... 11, step: 2)
        .help(layerBandHelp)

        // Unavailability may only block turning norm units ON. Turning them
        // OFF must always work: sessions default to norm units, so a blanket
        // disable locked the toggle ON for a norms-less vector with no path
        // to raw α (field report 2026-08-06) — the refusal said "switch α to
        // raw" and the switch was dead.
        Toggle("Alpha in residual-norm units", isOn: $service.alphaInNormUnits)
            .disabled(!service.normUnitsAvailable && !service.alphaInNormUnits)
            .help(normUnitsHelp)
    }

    private var injectVectorsHelp: String {
        var help =
            "master switch: every enabled vector box below adds its α·v to "
            + "the residual stream on each generated token — multiple boxes "
            + "inject the linear combination h + Σ αᵢ·vᵢ. Applies from the "
            + "next message, so it can be toggled mid-conversation"
        if isServerWorkspace {
            help +=
                ". In a server workspace the injection runs server-side "
                + "through an agent spec composed from these controls"
        }
        return help
    }

    private var layerBandHelp: String {
        var help =
            "shared by all boxes: inject across N consecutive layers centered "
            + "on each box's layer (1–11, odd values only — a band is "
            + "symmetric around its center)"
        if isServerWorkspace {
            help += " (the agent spec's bandWidth — applied server-side)"
        }
        return help
    }

    private var normUnitsHelp: String {
        if service.normUnitsAvailable {
            return "shared by all boxes: the injected perturbation's L2 norm "
                + "becomes alpha × the layer's typical residual norm "
                + "exactly (the vector's own norm is folded out, so alphas "
                + "compare across layers and concepts). On the 4B shakedown "
                + "expression appears ~0.2–0.4, collapse above ~1. Re-enter "
                + "alphas when flipping this — the numbers change meaning"
        }
        if isServerWorkspace {
            return "turning norm units ON needs every enabled steering box's "
                + "server vector to carry a residual-norm table (the catalog's "
                + "hasResidualNorms) — ablation boxes don't need one (λ never "
                + "converts). Turning OFF to raw α always works; the "
                + "conversion runs server-side"
        }
        return "turning norm units ON needs every enabled steering box's "
            + "vector extracted (or backfilled) with residual norms — "
            + "ablation boxes don't need one (λ never converts). Turning OFF "
            + "to raw α always works"
    }

    private var useAdapterHelp: String {
        isServerWorkspace
            ? "includes the selected server-side LoRA adapter in the "
                + "composed agent spec (applied on the server)"
            : "loads the selected LoRA/DoRA adapter for the next generation and "
                + "unloads it afterward; adapter artifacts are registered in "
                + "Data › Adapter Training"
    }

    /// Adapter toggle + workspace-scoped adapter picker (strict availability:
    /// the ACTIVE SERVER's adapters in a server workspace, never local ones).
    @ViewBuilder
    private var adapterControls: some View {
        Toggle("Use adapter", isOn: $service.adaptersEnabled)
            .help(useAdapterHelp)

        if isServerWorkspace {
            Picker(
                "Adapter",
                selection: Binding<String?>(
                    get: { service.selectedServerAdapterID },
                    set: { service.selectedServerAdapterID = $0 }))
            {
                Text("None").tag(String?.none)
                ForEach(service.serverAdapterOptions) { adapter in
                    Text(adapter.name).tag(String?.some(adapter.id))
                }
            }
            .disabled(!service.adaptersEnabled || service.serverAdapterOptions.isEmpty)
            .help("LoRA adapters found on \(service.cluster.substrateLabel)")
            if service.adaptersEnabled, service.serverAdapterOptions.isEmpty {
                Text(
                    "No adapters on the server. Train one in Data › Adapter "
                        + "Training (it runs as a server job), or upload an "
                        + "agent that carries one."
                )
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        } else {
            Picker(
                "Adapter",
                selection: Binding<String?>(
                    get: { service.selectedRuntimeAdapterID },
                    set: { service.selectedRuntimeAdapterID = $0 }))
            {
                Text("None").tag(String?.none)
                ForEach(service.compatibleAdapters) { adapter in
                    Text(adapter.label).tag(String?.some(adapter.id))
                }
            }
            .disabled(!service.adaptersEnabled || service.compatibleAdapters.isEmpty)
            .help("only adapters registered for the loaded model are shown")

            if service.adaptersEnabled, service.compatibleAdapters.isEmpty {
                Text("No registered adapters for the loaded model.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Neutral-direction removal — an MLX-local projection at injection time.
    /// Genuinely unbacked on server workspaces (no server neutral-basis
    /// catalog yet); a seeded variant's own basis pin is preserved verbatim
    /// in composed inline specs, so nothing is silently dropped.
    @ViewBuilder
    private var neutralDirectionControls: some View {
        Toggle(
            "Remove neutral directions",
            isOn: $service.removeNeutralDirectionsAtSteering
        )
        .disabled(isServerWorkspace)
        .help(
            "projects each active steering vector away from the selected neutral-corpus "
                + "principal components immediately before injection. The saved vector "
                + "artifact is unchanged, so this can be toggled for A/B checks")

        Picker(
            "Neutral basis",
            selection: Binding<String?>(
                get: { service.selectedNeutralPCBasisID },
                set: { service.selectedNeutralPCBasisID = $0 }))
        {
            Text("None").tag(String?.none)
            ForEach(service.compatibleNeutralPCBases) { record in
                Text(record.label).tag(String?.some(record.id))
            }
        }
        .disabled(
            !service.removeNeutralDirectionsAtSteering
                || service.compatibleNeutralPCBases.isEmpty
                || isServerWorkspace
        )
        .help(
            "model-specific neutral PC artifacts built from prompts/neutral/corpus.jsonl")

        if service.removeNeutralDirectionsAtSteering,
            service.compatibleNeutralPCBases.isEmpty
        {
            Text("No neutral PC basis for the loaded model. Build one in Data.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }

        serverAvailabilityCaption(
            "neutral-direction removal",
            reason: "no server neutral-basis catalog yet; a seeded agent's "
                + "basis pin is kept in composed specs")

        Toggle(
            "Center ablations on the neutral mean",
            isOn: $service.centerAblationAgainstNeutralMean
        )
        .help(
            "projects the neutral-corpus residual MEAN out of every ablating "
                + "vector before removal (v − (v·m̂)m̂). Extracted vectors share a "
                + "large component with that mean, and ablating it raw at λ=1 "
                + "collapses generation into single-token repetition; centering "
                + "removes only the concept-specific part and keeps the model "
                + "coherent. Uses the mean stored in the vector artifact "
                + "(re-extract older concepts to add it); the artifact is unchanged")

        if let advisory = service.ablationAdvisory {
            Label(advisory, systemImage: "exclamationmark.triangle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    /// Probe highlighting — an MLX-local instrument (reads residual
    /// activations during local generation), so it stays disabled with a
    /// reason caption in server workspaces.
    @ViewBuilder
    private var probeControls: some View {
        Toggle("Probe-highlight generated text", isOn: $service.probeHighlightsEnabled)
            .disabled(isServerWorkspace)
            .help(
                "records residual activations while the assistant generates and "
                    + "highlights streamed token chunks using the selected trained "
                    + "reading probe, or vector cosine if no trained probe is "
                    + "selected — blue with a solid underline is aligned, red "
                    + "with a dashed underline is opposite. Chat-mode "
                    + "generations only")
        Picker(
            "Probe",
            selection: Binding<String?>(
                get: { service.selectedProbeID },
                set: { service.selectedProbeID = $0 }))
        {
            Text("Active vectors (cosine)").tag(String?.none)
            ForEach(service.compatibleReadingProbes) { probe in
                Text(probe.label).tag(String?.some(probe.id))
            }
        }
        .disabled(!service.probeHighlightsEnabled || isServerWorkspace)
        .help(
            "trained probes are model-specific artifacts from Data. "
                + "The fallback uses cosine against enabled steering vectors and is less calibrated")
        if let probeStatus = service.probeStatus {
            Text(probeStatus)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }

        serverAvailabilityCaption(
            "probe highlighting",
            reason: "an MLX-local instrument reading residual activations "
                + "during local generation")
    }

    /// Prompt mode, system prompt, sampling, and the artifact refresh —
    /// live in both workspaces (server chat honors all of them).
    @ViewBuilder
    private var generationControls: some View {
        Picker("Prompt mode", selection: $service.promptMode) {
            ForEach(ExperimentManifest.PromptMode.allCases, id: \.self) { mode in
                Text(mode.label).tag(mode)
            }
        }
        .help(
            "chat assistant uses the model chat template and session history; raw "
                + "completion sends literal text through the study-style generator")

        TextField("system prompt", text: $service.systemPrompt, axis: .vertical)
            .lineLimit(1 ... 4)
            .help("optional system instruction for chat mode, or text prepended in raw mode")

        // Temperature stays enabled on server workspaces: server chat
        // honors it (/api/generate/stream and agent chat both take
        // `temperature`), so an "unavailable" caption would be false.
        // The shared row, not a bare slider: 0 and 0.1 are one step apart
        // and looked identical with no readout (finding 7a).
        TemperatureRow(
            value: $service.temperature,
            help: "sampling temperature for this Playground chat only — it "
                + "does not change any study's declared temperature. 0 is "
                + "greedy decoding; higher values sample more freely")

        Toggle("Qwen thinking mode", isOn: $service.qwenThinkingEnabled)
            .disabled(!service.selectedModelID.lowercased().contains("qwen"))
            .help(
                "a Qwen-only chat-template option; studies default to "
                    + "no-think. Toggling this resets the chat, because the "
                    + "rendered prompt format changes")

        Button("Refresh artifacts") {
            if isServerWorkspace {
                Task { await service.refreshServerSteeringArtifacts() }
            } else {
                service.refreshVectors()
            }
        }
        .help(refreshArtifactsHelp)
    }

    private var refreshArtifactsHelp: String {
        isServerWorkspace
            ? "re-fetch the server's vector catalog and adapter listing"
            : "rescan runs/ for vectors, probes, neutral bases, adapters, "
                + "and agents created by this app or by steerlab-cli"
    }

    /// Honest inline disabling (design correction): controls with no server
    /// backing stay in place, disabled, with this short reason caption —
    /// never quarantined in a separate "server" section.
    @ViewBuilder
    private func serverAvailabilityCaption(_ what: String, reason: String) -> some View {
        if isServerWorkspace {
            Text("\(what) — unavailable on server workspaces: \(reason)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// One-line compute-target + connection status for the Steering pane's
    /// read-only status row (the selector lives in the window toolbar).
    private var computeStatusLine: String {
        switch service.cluster.computeTarget {
        case .local:
            let state =
                switch service.state {
                case .unloaded: "no model loaded"
                case .loading(let percent): "loading… \(percent)%"
                case .ready: service.loadedModelID ?? "ready"
                }
            return "Compute: Local (MLX) — \(state) · change in toolbar"
        case .server:
            return "Compute: Server (\(service.cluster.serverHostLabel)) — "
                + "\(service.cluster.status ?? "not connected") · change in toolbar"
        }
    }

    /// Steer / Ablate, above the controls each mode needs. The controls are
    /// SWAPPED rather than greyed out: ablation takes no layer and its
    /// strength is not an alpha, so leaving those on screen disabled would
    /// suggest they mean something here.
    @ViewBuilder
    private func slotModePicker(
        _ slot: Binding<ChatService.SteerSlot>
    ) -> some View {
        Picker("Mode", selection: slot.mode) {
            Text("Steer").tag(InterventionPlan.Mode.add)
            Text("Ablate").tag(InterventionPlan.Mode.ablate)
        }
        .pickerStyle(.segmented)
        .onChange(of: slot.wrappedValue.mode) { _, new in
            // Never carry a steering alpha across as a lambda: alpha is
            // typically 1–3, and lambda 2 is already a REFLECTION.
            //
            // Returning to Steer re-asks the ARTIFACT what its strength
            // should be rather than restoring a flat literal — the same
            // decision a fresh selection makes (SlotAlphaDefault).
            if new == .ablate {
                slot.alpha.wrappedValue = ChatService.SteerSlot.defaultAblationStrength
            } else {
                service.applyDefaultsForSelectedVector(slotID: slot.wrappedValue.id)
            }
        }
        .help(InjectionModeCopy.pickerHelp)
    }

    /// Ablation's controls: a strength, no layer, and the consequences said
    /// out loud rather than left in a tooltip.
    @ViewBuilder
    private func slotAblationControls(
        _ slot: Binding<ChatService.SteerSlot>
    ) -> some View {
        let lambda = slot.wrappedValue.alpha
        HStack {
            Text("Strength λ")
            Slider(value: slot.alpha, in: 0 ... 2, step: 0.05)
            TextField(
                "", value: slot.alpha,
                format: .number.precision(.fractionLength(0 ... 2))
            )
            .frame(width: 56)
        }
        .help(InjectionModeCopy.lambdaHelp)
        Text(InjectionModeCopy.lambdaLabel(lambda))
            .font(.caption)
            .foregroundStyle(
                InjectionModeCopy.lambdaIsUnusual(lambda) ? .orange : .secondary)
        Label(
            "Every layer, every token — including the whole prompt",
            systemImage: "square.stack.3d.up")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    /// One full steering box per vector — every slot has identical controls.
    /// Workspace-scoped like every other artifact selector: local vector
    /// artifacts in Local, the ACTIVE SERVER's vector catalog in a server
    /// workspace (strict availability — never a merged list).
    @ViewBuilder
    private func slotSection(_ slot: Binding<ChatService.SteerSlot>) -> some View {
        if isServerWorkspace {
            serverSlotSection(slot)
        } else {
            localSlotSection(slot)
        }
    }

    /// Server-workspace steering box: same layout as local, backed by the
    /// server's vector artifacts (referenced by server path/id) — the slots
    /// feed the composed inline variant spec, applied server-side.
    @ViewBuilder
    private func serverSlotSection(_ slot: Binding<ChatService.SteerSlot>) -> some View {
        let value = slot.wrappedValue
        let record = service.serverVectorRecord(for: value)
        Section(service.serverSlotConcept(for: value) ?? "vector") {
            HStack(spacing: 6) {
                Toggle("Include this vector", isOn: slot.enabled)
                    .labelsHidden()
                    .help(
                        "include this vector in the composed agent spec the "
                            + "server applies")
                Picker("Vector", selection: slot.vectorID) {
                    Text("none").tag(String?.none)
                    // A seeded ref missing from the catalog listing stays
                    // selectable by its server path, so an untouched seed
                    // round-trips byte-identically.
                    if let id = value.vectorID, record == nil {
                        Text("\(service.serverSlotConcept(for: value) ?? "vector") · \(id)")
                            .tag(String?.some(id))
                    }
                    ForEach(service.compatibleServerVectorsGrouped, id: \.concept) { group in
                        Section(group.concept) {
                            ForEach(group.items) { candidate in
                                Text(serverVectorLabel(candidate))
                                    .tag(String?.some(candidate.id))
                            }
                        }
                    }
                }
                .labelsHidden()
                .help(
                    "which of the server's vectors this box injects; picking "
                        + "one adopts the artifact's own layer, α, and units")
                .onChange(of: value.vectorID) {
                    // Layer AND alpha AND units: selecting a vector adopts what
                    // the artifact knows about itself (SlotAlphaDefault).
                    service.applyDefaultsForSelectedVector(slotID: value.id)
                }
                Button {
                    service.removeSlot(value.id)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("remove this steering box (the last box clears instead)")
                .accessibilityLabel("Remove this steering box")
            }

            slotModePicker(slot)

            if value.mode == .add {
                LabeledContent("Layer: \(Int(value.layer))") {
                    Slider(
                        value: slot.layer,
                        in: 0 ... Double(max(1, service.maxLayerIndex(for: value))),
                        step: 1)
                }
                .help(
                    "band center for this vector. Mid-network layers usually steer "
                        + "best — see the norm-by-layer curve in Data › "
                        + "Concepts & Vectors")

                HStack {
                    Text(alphaFieldLabel)
                        .foregroundStyle(service.alphaInNormUnits ? Color.primary : Color.orange)
                        .fontWeight(service.alphaInNormUnits ? .regular : .semibold)
                    Slider(value: slot.alpha, in: alphaRange, step: alphaStep)
                    TextField(
                        "", value: slot.alpha,
                        format: .number.precision(.fractionLength(0 ... 3))
                    )
                    .frame(width: 64)
                    .multilineTextAlignment(.leading)
                }
                .help(alphaFieldHelp)
            } else {
                slotAblationControls(slot)
            }

            if record == nil, value.vectorID != nil {
                HStack(spacing: 6) {
                    Text(
                        "vector not in the fetched server catalog — layer bounds "
                            + "fall back to the loaded model's")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Button("Refresh") {
                        Task { await service.catalog.refreshRemoteVectors() }
                    }
                    .controlSize(.mini)
                    .help("re-fetch the server's vector catalog so this reference resolves")
                }
            }

            if let record {
                Text(serverVectorProvenanceLine(record))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help(
                        "provenance from the server catalog — the injection itself "
                            + "runs server-side through the composed agent spec")
                if let preview = service.serverInjectionPreview(for: value) {
                    Text(injectionPreviewLine(preview))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(injectionPreviewColor(preview))
                        .help(
                            "estimated perturbation the server-side injection adds "
                                + "on each generated token for this slot (from the "
                                + "catalog's per-layer norm tables; the server "
                                + "converts from its own sidecar)")
                }
                if record.residualNormPerLayer == nil,
                    record.hasResidualNorms != true
                {
                    normBackfillOffer {
                        Task {
                            await service.concepts.backfillNormsOnActiveServer(record)
                        }
                    }
                }
            } else if value.vectorID != nil {
                Text("not in the server catalog listing — kept by path")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            slotNotAppliedCaption(for: value)
        }
    }

    /// The standing "this row will NOT be applied, because …" caption —
    /// rendered whenever a row that looks active would not actually inject
    /// (unresolved vector, missing residual norms under norm-unit α, …).
    /// The rule lives in ChatService (`slotNotAppliedCaption`), shared with
    /// the send-time refusal so caption and behavior cannot drift.
    @ViewBuilder
    private func slotNotAppliedCaption(for slot: ChatService.SteerSlot) -> some View {
        if let reason = service.slotNotAppliedCaption(for: slot) {
            Text("⚠︎ \(reason)")
                .font(.caption2)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
    }

    private func serverVectorLabel(_ record: RemoteVectorRecord) -> String {
        // Concept is the SECTION header now — the row carries what differs
        // within the concept: name, recipe, extraction time (to minutes).
        var parts = [record.name]
        if let method = record.resolvedMethod { parts.append(method) }
        if let date = record.resolvedExtractionDate {
            parts.append(
                String(date.prefix(16)).replacingOccurrences(of: "T", with: " "))
        }
        return parts.joined(separator: " · ")
    }

    private func serverVectorProvenanceLine(_ record: RemoteVectorRecord) -> String {
        var parts = [
            record.resolvedMethod ?? "method?",
            record.resolvedReadingPosition ?? "last token*",
            record.resolvedExtractionDate ?? "date?",
            "\(record.layerCount) layers",
        ]
        parts.append(
            record.hasResidualNorms == true || record.residualNormPerLayer != nil
                ? "norms: \(record.residualNormSource ?? "recorded")"
                : "no residual norms")
        // Substrate stamp: which engine extracted this artifact. Unstamped
        // server-workspace records are honestly unknown, not assumed.
        parts.append(record.substrate ?? "engine unknown")
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func localSlotSection(_ slot: Binding<ChatService.SteerSlot>) -> some View {
        let value = slot.wrappedValue
        let artifact = service.artifact(for: value)
        Section(artifact?.sidecar.concept ?? "vector") {
            HStack(spacing: 6) {
                Toggle("Include this vector", isOn: slot.enabled)
                    .labelsHidden()
                    .help("include this vector in the injected sum")
                Picker("Vector", selection: slot.vectorID) {
                    Text("none").tag(VectorArtifact.ID?.none)
                    ForEach(service.compatibleVectorsGrouped, id: \.concept) { group in
                        Section(group.concept) {
                            ForEach(group.items) { artifact in
                                Text(
                                    artifact.label
                                        + (service.staleVectorIDs.contains(artifact.id)
                                            ? "  ⚠︎ stale" : "")
                                )
                                .tag(VectorArtifact.ID?.some(artifact.id))
                            }
                        }
                    }
                }
                .labelsHidden()
                .help(
                    "which vector artifact this box injects; picking one "
                        + "adopts the artifact's own layer, α, and units")
                .onChange(of: value.vectorID) {
                    // Layer AND alpha AND units: selecting a vector adopts what
                    // the artifact knows about itself (SlotAlphaDefault).
                    service.applyDefaultsForSelectedVector(slotID: value.id)
                }
                Button {
                    service.removeSlot(value.id)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("remove this steering box (the last box clears instead)")
                .accessibilityLabel("Remove this steering box")
            }

            slotModePicker(slot)

            if value.mode == .add {
                let fixedLayer = artifact?.fixedSteeringLayer
                LabeledContent(
                    fixedLayer.map { "Layer: \($0) fixed" } ?? "Layer: \(Int(value.layer))"
                ) {
                    Slider(
                        value: slot.layer,
                        in: 0 ... Double(max(1, service.maxLayerIndex(for: value))),
                        step: 1)
                    .disabled(fixedLayer != nil)
                }
                .help(
                    fixedLayer.map {
                        "fixed: Gemma Scope SAE feature vectors are tied to "
                            + "their source residual-stream layer L\($0), and "
                            + "band width is ignored for this box"
                    }
                        ?? "band center for this vector. Mid-network layers usually steer "
                        + "best — see the norm-by-layer curve in Data › "
                        + "Concepts & Vectors")

                HStack {
                    Text(alphaFieldLabel)
                        .foregroundStyle(service.alphaInNormUnits ? Color.primary : Color.orange)
                        .fontWeight(service.alphaInNormUnits ? .regular : .semibold)
                    Slider(value: slot.alpha, in: alphaRange, step: alphaStep)
                    TextField(
                        "", value: slot.alpha,
                        format: .number.precision(.fractionLength(0 ... 3))
                    )
                    .frame(width: 64)
                    .multilineTextAlignment(.leading)
                }
                .help(alphaFieldHelp)
            } else {
                slotAblationControls(slot)
            }

            if let artifact {
                Text(provenanceLine(artifact, layer: artifact.fixedSteeringLayer ?? Int(value.layer)))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help(provenanceHelp(artifact))
                if let preview = service.injectionPreview(for: value) {
                    Text(injectionPreviewLine(preview))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(injectionPreviewColor(preview))
                        .help("effective perturbation that will be added on each generated token for this slot")
                }
                if artifact.sidecar.residualNormPerLayer == nil {
                    normBackfillOffer {
                        Task {
                            await service.concepts.backfillNormsLocally(artifact)
                        }
                    }
                }
            }

            slotNotAppliedCaption(for: value)
        }
    }

    /// Inline backfill affordance rendered directly on a norms-less vector
    /// row — the norm-units refusal's remedy, offered where the vector was
    /// just picked instead of only in
    /// Data › Concept Index › "Server vectors missing norms". Ablation rows
    /// run fine without norms; the measurement is what norm-unit STEERING α
    /// needs.
    @ViewBuilder
    private func normBackfillOffer(action: @escaping () -> Void) -> some View {
        HStack(spacing: 6) {
            Text("no residual norms — norm-unit α needs a measured denominator "
                + "(ablation λ does not)")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Button("Measure norms") { action() }
                .controlSize(.mini)
                .disabled(service.concepts.isWorking)
                .help(normBackfillOfferHelp)
            if service.concepts.isWorking {
                Text("another concept job is running — wait for it to finish")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }

    /// Names the corpus AND its hash before the button is pressed: the
    /// measurement becomes the new artifact's residualNormSource /
    /// neutralCorpusHash provenance, so which bytes denominate α must be
    /// visible at the point of action.
    private var normBackfillOfferHelp: String {
        let (path, hash) = ConceptBuilder.neutralCorpusProvenance()
        let corpus = hash.map { "\(path) (sha256 \(String($0.prefix(12)))…)" }
            ?? "\(path) — not found in this workspace; seed the neutral corpus first"
        var help = "measure per-layer residual norms on the workspace's pinned "
            + "neutral corpus \(corpus) and write a NEW artifact — the original "
            + "is never modified; the corpus hash is stamped as "
            + "residualNormSource: neutral-corpus provenance"
        if isServerWorkspace {
            help += ". Runs as a durable server job on the server's checkout "
                + "of this corpus; the new artifact appears in the server catalog"
        }
        return help
    }

    private func provenanceLine(_ artifact: VectorArtifact, layer: Int) -> String {
        let method = methodDescription(artifact)
        let reading = artifact.sidecar.readingPosition ?? "last token*"
        let date = String(artifact.sidecar.extractionDate.prefix(10))
        let normText = norm(of: artifact, at: layer)
        var line = "\(method) · \(reading) · \(date) · norm@L\(layer) \(normText)"
        // A raw rendering stays unmentioned, exactly like an absent stamp —
        // they are the same rendering, and naming one would read as a
        // difference.
        if let rendering = artifact.sidecar.extractionRendering, !rendering.isRaw {
            line += " · \(rendering.label)"
        }
        if let substrate = artifact.sidecar.substrate {
            line += " · \(substrate)"
        }
        // A mirrored pole is the NEGATION of another artifact's direction, and
        // its method/date/norm fragments are all inherited from that source —
        // so without this fragment the caption of a mirror and the caption of
        // its parent read identically while the two point opposite ways. The
        // one caption that has to say so is this one.
        if artifact.sidecar.negatedFrom != nil {
            line += " · mirrored pole"
        }
        return line
    }

    private func provenanceHelp(_ artifact: VectorArtifact) -> String {
        let hash = String(artifact.sidecar.stimulusSetHash.prefix(12))
        let norms = artifact.sidecar.residualNormSource ?? "extraction-stimuli*"
        var line = "provenance — concept: \(artifact.sidecar.concept); model: "
            + "\(artifact.sidecar.modelID); stimuli: \(hash)…; norm-unit "
            + "denominator: \(norms)"
        // Which averaging rule stands behind that denominator. Legacy
        // artifacts say "legacy (pre-stamp)" rather than borrowing today's
        // rule — a pre-stamp artifact's rule is genuinely unknown.
        if let convention = ResidualNormConvention.displayLabel(
            residualNormPerLayer: artifact.sidecar.residualNormPerLayer,
            stamp: artifact.sidecar.residualNormConvention)
        {
            line += " (convention: \(convention))"
        }
        line += "; method and reading position are the extraction choices "
            + "recorded in this vector's sidecar"
        // The mirror's whole story, because every other fragment above is the
        // SOURCE's: the stimulus hash is the source's two files with their
        // poles swapped, and the direction is the source's negated.
        if let negation = artifact.sidecar.negatedFrom {
            line += ". Mirrored pole: every layer × −1 from concept "
                + "'\(negation.concept)' (\(negation.path)), so the stimulus "
                + "hash above is that concept's two files with the positive "
                + "and negative roles swapped"
        }
        // A pre-scaled OptVec vector must announce itself wherever it can be
        // selected for steering — scaling it like an ordinary direction is an
        // orders-of-magnitude dose error (open-issues §24).
        if let advisory = OptVecPackaging.advisory(for: artifact.sidecar) {
            line += ". \(advisory)"
        }
        return line
    }

    /// The slot alpha field's denomination, spelled inline at the point of
    /// entry: a 0.4 in norm units and a 0.4 raw differ by roughly the
    /// residual-norm/vector-norm ratio (~15× on the shakedown vectors), so
    /// the number's meaning must be visible where it is typed. Shared by the
    /// local and server steering boxes — the mode is session-wide.
    private var alphaFieldLabel: String {
        // The engine's own label when it has one — it names the LAYER too
        // ("α in residual-norm units at L18"), which is the half that was
        // missing when α silently meant a fraction of an unnamed denominator.
        if let decision = service.alphaDefaultDecision,
            decision.isNormUnits == service.alphaInNormUnits
        {
            return decision.alphaLabel
        }
        return service.alphaInNormUnits ? "α (norm units)" : "α (raw)"
    }

    private var alphaFieldHelp: String {
        (service.alphaInNormUnits
            ? "α is a fraction of the layer's typical residual-stream norm "
                + "(injected Δnorm = α × residual norm). "
            : "α is the literal coefficient on the vector (injected Δnorm = "
                + "α × ‖v‖). ")
            + "This vector's coefficient in the injected sum (negative steers "
            + "away / subtracts). Too high collapses coherence; type an exact "
            + "value to map fine-grained dose-response"
            + alphaDefaultExplanation
    }

    /// Where the current default came from, which denominator convention
    /// stands behind it, and — in RAW mode — the one command that gives this
    /// artifact a denominator. Rendered, never recomputed: every string comes
    /// from `SlotAlphaDefault.Decision`.
    private var alphaDefaultExplanation: String {
        guard let decision = service.alphaDefaultDecision else { return "" }
        var parts = [". \(decision.rationale)"]
        if let convention = decision.conventionNote {
            parts.append("denominator convention: \(convention)")
        }
        if let hint = decision.backfillHint {
            parts.append("give this vector a denominator with: \(hint)")
        }
        return parts.joined(separator: "; ")
    }

    /// Norm-unit strengths live in a smaller range than raw alphas: with the
    /// exact conversion (injected norm = α·r), the 4B shakedown shows
    /// expression from ~0.2 and capability collapse above ~1.
    private var alphaRange: ClosedRange<Double> {
        service.alphaInNormUnits ? -1 ... 1 : -10 ... 10
    }

    private var alphaStep: Double {
        service.alphaInNormUnits ? 0.01 : 0.1
    }

    private func norm(of artifact: VectorArtifact, at layer: Int) -> String {
        let norms = artifact.sidecar.normsPerLayer
        guard !norms.isEmpty else { return "—" }
        let index = min(max(0, layer), norms.count - 1)
        return norms[index].formatted(.number.precision(.significantDigits(4)))
    }

    private func methodDescription(_ artifact: VectorArtifact) -> String {
        if let recipeName = artifact.sidecar.recipeName,
            recipeName.lowercased().contains("gemma scope")
        {
            return "Gemma Scope SAE"
        }
        return switch artifact.sidecar.extractionMethod {
        case ExtractionMethod.pairedDifferencePCA.rawValue: "Paired-difference PCA (RepE-inspired)"
        case ExtractionMethod.meanDifference.rawValue: "Mean difference"
        default: "mean difference*"  // pre-options artifact; it was the only method
        }
    }

    /// Amber when the norm-units label would otherwise lie (missing
    /// denominator → the raw α is what's shown), blue for a real norm-unit
    /// conversion, secondary for plain raw mode.
    private func injectionPreviewColor(_ preview: ChatService.InjectionPreview) -> Color {
        if preview.normUnitsFallback { return .orange }
        return preview.isNormUnits ? .blue : .secondary
    }

    private func injectionPreviewLine(_ preview: ChatService.InjectionPreview) -> String {
        // A missing denominator is a REFUSAL at send time (both substrates
        // refuse norm-unit injection without residual norms) — the label
        // must not suggest a raw-α fallback will quietly inject instead.
        let mode = preview.normUnitsFallback
            ? "norm α — norms missing (send refuses; number shown is raw α)"
            : preview.isNormUnits ? "norm α" : "raw α"
        let residual =
            preview.residualNorm.map {
                " · residual \($0.formatted(.number.precision(.significantDigits(4))))"
            } ?? ""
        let fixed = preview.fixedLayer ? " · fixed layer" : ""
        // The modernization bridge: what this raw α amounts to in norm
        // units, so converting a legacy variant is copy-a-number.
        let equivalent =
            preview.equivalentNormUnits.map {
                " · ≈ \($0.formatted(.number.precision(.significantDigits(3)))) norm units"
            } ?? ""
        return "\(mode) · inject L\(preview.layer) · scale "
            + preview.effectiveAlpha.formatted(.number.precision(.significantDigits(4)))
            + " · Δnorm "
            + preview.injectedNorm.formatted(.number.precision(.significantDigits(4)))
            + residual
            + equivalent
            + fixed
    }
}

struct MessageBubble: View {
    let message: ChatService.ChatMessage
    /// What the copy affordance puts on the clipboard.
    let copyText: () -> String
    /// What this bubble IS, for the copy affordance's wording: transcript
    /// bubbles are turns, the Activity feed's bubbles are log entries — and
    /// offering "Copy this turn" on a build log is exactly the chat/feed
    /// confusion the split of the two panes exists to prevent.
    var copyNoun: String = "turn"
    /// Context-menu "Edit turn…" (nil = not editable, e.g. live logs).
    var editAction: (() -> Void)? = nil
    /// Context-menu "Remove turn" — only offered for the LAST transcript
    /// turn (the caller gates via `canRemoveTranscriptMessage`).
    var removeAction: (() -> Void)? = nil
    @State private var showingOriginal = false

    /// Provenance badge text for researcher-authored or researcher-altered
    /// assistant turns; nil for ordinary turns. A seeded or edited turn must
    /// be visibly distinct from a real generation everywhere it is rendered.
    private var provenanceBadge: String? {
        guard message.role == .assistant else { return nil }
        if message.seededPrefixLength != nil {
            return message.edited ? "seeded prefix · edited" : "seeded prefix · continued"
        }
        if message.continuationPrefixLength != nil {
            return message.edited ? "edited · continued" : "generated · continued"
        }
        if message.seeded { return message.edited ? "seeded · edited" : "seeded" }
        return message.edited ? "edited" : nil
    }

    private var provenanceBadgeHelp: String {
        if let prefixLength = message.seededPrefixLength {
            return "Researcher-authored prefix (first \(prefixLength) "
                + "characters); the rest was generated mid-turn"
                + (message.edited ? "; the prefix was edited from the original" : "")
        }
        if let prefixLength = message.continuationPrefixLength {
            return "The model continued after the first \(prefixLength) characters; "
                + (message.edited
                    ? "that prefix was generated, then edited by the researcher"
                    : "that prefix was generated previously")
        }
        if message.seeded, message.edited {
            return "Seeded (researcher-authored), then edited afterwards — "
                + "not a pure generation"
        }
        if message.seeded {
            return "Researcher-authored turn — the model treats it as its own "
                + "prior message; it was not generated"
        }
        return "Generated, then edited by the researcher — no longer a pure "
            + "generation"
    }

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                HStack(spacing: 6) {
                    if let editAction {
                        Button(action: editAction) {
                            Image(systemName: "pencil")
                        }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                        .help(
                            "edit this \(copyNoun)'s text — the original stays "
                                + "available under the bubble and the turn is "
                                + "badged as edited")
                        .accessibilityLabel("Edit this \(copyNoun)")
                    }
                    CopyButton(
                        help: "copy this \(copyNoun)'s text to the clipboard",
                        text: { copyText() }
                    ) {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    if let provenanceBadge {
                        Text(provenanceBadge)
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.orange.opacity(0.18), in: Capsule())
                            .foregroundStyle(.orange)
                            .help(provenanceBadgeHelp)
                    }
                }
                content
                    .textSelection(.enabled)
                    .padding(10)
                    .background(
                        message.role == .user
                            ? AnyShapeStyle(.tint.opacity(0.25))
                            : AnyShapeStyle(.quaternary),
                        in: RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        provenanceBadge == nil
                            ? nil
                            : RoundedRectangle(cornerRadius: 10)
                                .strokeBorder(
                                    .orange.opacity(0.55),
                                    style: StrokeStyle(lineWidth: 1, dash: [5, 3])))
                    .contextMenu {
                        Button("Copy \(copyNoun)") { _ = Clipboard.copy(copyText()) }
                        if let editAction {
                            Button("Edit \(copyNoun)…", action: editAction)
                        }
                        if let removeAction {
                            Button("Remove \(copyNoun)", role: .destructive, action: removeAction)
                        }
                    }
                if let original = message.originalText {
                    DisclosureGroup(isExpanded: $showingOriginal) {
                        Text(original)
                            .font(.caption)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 4)
                    } label: {
                        // Not "not in context": that read as if the text had
                        // gone missing. It is the pre-edit version, kept for
                        // provenance and no longer sent to the model.
                        Label(
                            "Before edit (the model no longer sees this)",
                            systemImage: "clock.arrow.circlepath")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .help(
                        "the text of this \(copyNoun) before it was edited — "
                            + "kept as provenance and included in copies and "
                            + "exports, but not sent to the model")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 4)
                }
            }
            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }

    @ViewBuilder
    private var content: some View {
        if message.role == .assistant {
            if !message.probeSpans.isEmpty {
                ProbeHighlightedText(spans: message.probeSpans)
            } else {
                // Models reply in markdown; render it. User text stays verbatim.
                MarkdownMessageText(text: message.text.isEmpty ? "…" : message.text)
            }
        } else {
            userContent
        }
    }

    @ViewBuilder
    private var userContent: some View {
        if message.text.count >= ChatService.longPromptStudyPathCharacterThreshold {
            DisclosureGroup {
                Text(message.text)
                    .font(.body.monospaced())
                    .frame(maxWidth: .infinity, alignment: .leading)
            } label: {
                VStack(alignment: .trailing, spacing: 3) {
                    Text("Long prompt")
                        .font(.body.weight(.semibold))
                    Text(
                        "\(message.text.count.formatted()) chars · "
                            + "~\(message.text.split(whereSeparator: \.isWhitespace).count.formatted()) words")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(String(message.text.prefix(700)) + (message.text.count > 700 ? "…" : ""))
                        .font(.caption)
                        .lineLimit(6)
                        .foregroundStyle(.secondary)
                }
            }
            .help(
                "show the whole prompt — long prompts are collapsed so one "
                    + "of them cannot bury the rest of the transcript")
        } else {
            Text(message.text)
        }
    }
}

/// Small modal editor for one transcript turn (context-menu "Edit turn…").
/// The provenance consequence is stated up front; saving routes through
/// `ChatService.editTranscriptMessage` — the single choke point for the
/// provenance rules and the session-replay invalidation.
private struct TranscriptTurnEditor: View {
    let role: ChatService.ChatMessage.Role
    let provenanceNote: String
    let branchUnavailableReason: String?
    @State var text: String
    let onSave: (String) -> Void
    let onContinueAssistant: (String) -> Void
    let onReturnControl: (String) -> Void
    let onRestartUser: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    private var validText: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Edit turn")
                .font(.headline)
            Text(provenanceNote)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextEditor(text: $text)
                .font(.body)
                .frame(minWidth: 620, minHeight: 240)
                .padding(4)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 1))
                .help(
                    "the text of this turn as the model will see it; the "
                        + "version before the edit is kept under the turn in "
                        + "the transcript")
                .accessibilityLabel("Turn text")
            // The branching buttons truncate the conversation, and that is
            // not something to discover afterwards: say it in the sheet, not
            // only in a tooltip.
            Text(truncationConsequence)
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
            if let branchUnavailableReason {
                Label(branchUnavailableReason, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without changing this turn")
                Spacer()
                // ⌘↩ belongs on the SAFE action: it used to fire the most
                // consequential branch (truncate everything after this turn
                // and regenerate) on a reflex Return.
                Button("Save as is") {
                    onSave(text)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!validText)
                .help(
                    "save the edited text and leave the rest of the "
                        + "conversation exactly as it is — later turns stay "
                        + "in context and nothing regenerates")
                switch role {
                case .assistant:
                    Button("Return control to user") {
                        onReturnControl(text)
                        dismiss()
                    }
                    .disabled(!validText)
                    .help(
                        "save this as the model's last word, DELETE every "
                            + "later turn, and hand the composer back to you "
                            + "as User — nothing regenerates")
                    Button("Continue generation from here") {
                        onContinueAssistant(text)
                        dismiss()
                    }
                    .disabled(!validText || branchUnavailableReason != nil)
                    .help(
                        branchUnavailableReason
                            ?? "save this text as an unfinished assistant turn, "
                            + "DELETE every later turn, and have the model "
                            + "carry on mid-turn from here (prefill)")
                case .user:
                    Button("Restart conversation from here") {
                        onRestartUser(text)
                        dismiss()
                    }
                    .disabled(!validText || branchUnavailableReason != nil)
                    .help(
                        branchUnavailableReason
                            ?? "save this user turn, DELETE every later turn, "
                            + "and generate a fresh reply to it")
                }
            }
        }
        .padding(18)
    }

    /// What the branching buttons do to the turns AFTER this one — the
    /// consequence the sheet never stated. They delete those turns outright,
    /// so the sheet says "deleted", not "dropped from context".
    private var truncationConsequence: String {
        let actions =
            role == .assistant
            ? "Return control to user and Continue generation from here"
            : "Restart conversation from here"
        return "\(actions) delete every turn after this one from the "
            + "transcript — they cannot be recovered. Copy or download the "
            + "transcript first if you need them. Save as is keeps the whole "
            + "conversation and changes only this turn's text (the version "
            + "before the edit stays under the turn)."
    }
}

private struct MultiAgentTurnBubble: View {
    let result: MultiAgentTurnResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(result.turnIndex). \(result.title)")
                        .font(.headline)
                    Text(result.speakerName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(result.outputLabel)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Text(result.output)
                .textSelection(.enabled)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct MultiAgentActiveTurnBubble: View {
    let turn: MultiAgentPanel.LiveTurn

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(turn.index). \(turn.title)")
                        .font(.headline)
                    Text(turn.speaker)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                ProgressView()
                    .controlSize(.small)
            }
            Text(turn.output.isEmpty ? "…" : turn.output)
                .textSelection(.enabled)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct VariantRobustnessOutputBubble: View {
    let output: FineTuningPanel.LiveRobustnessOutput

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(output.title)
                        .font(.headline)
                    Text(output.prompt)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .textSelection(.enabled)
                }
                Spacer()
                if output.isComplete {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .help("this output finished generating")
                        .accessibilityLabel("Finished")
                } else {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            Text(output.output.isEmpty ? "waiting for first tokens…" : output.output)
                .font(.body.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(backgroundColor, in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
    }

    private var backgroundColor: Color {
        output.side.lowercased() == "variant" ? .blue.opacity(0.08) : .gray.opacity(0.12)
    }
}

private struct VariantRobustnessSummaryCard: View {
    let report: VariantRobustnessReport

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Robustness Summary")
                .font(.headline)
            LabeledContent(
                "Capability",
                value:
                    percent(report.variantBatteryAccuracy) + " variant · "
                    + percent(report.baselineBatteryAccuracy) + " baseline")
            LabeledContent(
                "Distinct-2",
                value:
                    report.meanVariantDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " variant · "
                    + report.meanBaselineDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " baseline")
            if report.judgeModel != nil {
                let counts = Dictionary(grouping: report.coherenceItems.compactMap(\.judgeResult)) { $0 }
                    .mapValues(\.count)
                Text(
                    "Judge: baseline \(counts["baseline"] ?? 0) · variant \(counts["variant"] ?? 0) · ties \(counts["tie"] ?? 0)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(report.coherenceItems.filter { $0.judge != nil }, id: \.index) { item in
                    if let judge = item.judge {
                        DisclosureGroup("Judge \(item.index): \(item.judgeResult ?? judge.winner)") {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("confidence \(judge.confidence.formatted(.number.precision(.fractionLength(2))))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(judge.briefReason)
                                    .textSelection(.enabled)
                                if let structured = structuredFieldsSummary(judge) {
                                    Text(structured)
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                        .textSelection(.enabled)
                                }
                            }
                            .padding(.vertical, 4)
                        }
                        .help(
                            "this judgment's confidence, its stated reason, "
                                + "and any structured fields the judge returned")
                    }
                }
            }
            if report.warnings.isEmpty {
                Label("No robustness warnings", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                // Indexed, not `id: \.self`: two identical warnings are two
                // findings and both must render.
                ForEach(Array(report.warnings.enumerated()), id: \.offset) { _, warning in
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .help(warning)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func percent(_ value: Float) -> String {
        value.formatted(.percent.precision(.fractionLength(0)))
    }

    private func structuredFieldsSummary(_ judge: PairedJudgeResponse) -> String? {
        guard let fields = judge.structuredFields, !fields.isEmpty else { return nil }
        return fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
    }
}

private struct VariantRobustnessJudgeBubble: View {
    let judgment: FineTuningPanel.LiveRobustnessJudgment

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Judge \(judgment.index): \(judgment.result)")
                        .font(.headline)
                    Text(judgment.prompt)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .textSelection(.enabled)
                }
                Spacer()
                Text("confidence \(judgment.response.confidence.formatted(.number.precision(.fractionLength(2))))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(judgment.response.briefReason)
                .textSelection(.enabled)
            if let scoreSummary {
                Text(scoreSummary)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let structuredSummary {
                Text(structuredSummary)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.purple.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private var scoreSummary: String? {
        let a = (judgment.response.aScores ?? [:])
            .map { "\($0.key): \($0.value)" }
            .sorted()
            .joined(separator: ", ")
        let b = (judgment.response.bScores ?? [:])
            .map { "\($0.key): \($0.value)" }
            .sorted()
            .joined(separator: ", ")
        guard !a.isEmpty || !b.isEmpty else { return nil }
        return "A [\(a.isEmpty ? "no scores" : a)] · B [\(b.isEmpty ? "no scores" : b)]"
    }

    private var structuredSummary: String? {
        guard let fields = judgment.response.structuredFields, !fields.isEmpty else { return nil }
        return fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
    }
}

private struct StagedLongPromptCard: View {
    let text: String
    let clear: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "doc.text")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 3) {
                Text("Long prompt staged")
                    .font(.callout.weight(.semibold))
                Text(
                    "\(text.count.formatted()) chars · "
                        + "~\(text.split(whereSeparator: \.isWhitespace).count.formatted()) words")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(preview)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
            }
            Spacer()
            // Not "Clear": that is Reset Chat's verb, and this throws away
            // one staged paste, not the conversation.
            Button("Discard staged prompt", action: clear)
                .buttonStyle(.borderless)
                .help(
                    "throw away this staged paste and type in the box again — "
                        + "the conversation is untouched")
        }
        .padding(8)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private var preview: String {
        String(text.prefix(700)) + (text.count > 700 ? "…" : "")
    }
}

private struct PromptInputBox: NSViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let disabled: Bool
    let onSubmit: () -> Void
    let onLongPaste: (String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .bezelBorder
        scrollView.drawsBackground = true

        let textView = PromptInputTextView()
        textView.isRichText = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        textView.textContainerInset = NSSize(width: 6, height: 6)
        textView.minSize = NSSize(width: 0, height: 34)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.delegate = context.coordinator
        textView.onSubmit = onSubmit
        textView.onLongPaste = onLongPaste
        textView.placeholder = placeholder
        textView.isEditable = !disabled
        textView.string = text

        scrollView.documentView = textView
        context.coordinator.textView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? PromptInputTextView else { return }
        context.coordinator.parent = self
        textView.onSubmit = onSubmit
        textView.onLongPaste = onLongPaste
        textView.placeholder = placeholder
        textView.isEditable = !disabled
        if textView.string != text {
            textView.string = text
        }
        textView.needsDisplay = true
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptInputBox
        weak var textView: PromptInputTextView?

        init(_ parent: PromptInputBox) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }
    }
}

private final class PromptInputTextView: NSTextView {
    var onSubmit: (() -> Void)?
    var onLongPaste: ((String) -> Void)?
    var placeholder = ""

    override func paste(_ sender: Any?) {
        if let pasted = NSPasteboard.general.string(forType: .string),
            pasted.count >= ChatService.longPromptStudyPathCharacterThreshold
        {
            onLongPaste?(pasted)
            return
        }
        super.paste(sender)
    }

    override func keyDown(with event: NSEvent) {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if event.charactersIgnoringModifiers == "\r" {
            // Return sends; Shift-Return inserts a newline (⌘-Return still
            // sends for muscle memory). Other modifier combos keep the
            // system behavior.
            if modifiers.isEmpty || modifiers == .command {
                onSubmit?()
                return
            }
            if modifiers == .shift {
                insertNewlineIgnoringFieldEditor(nil)
                return
            }
        }
        super.keyDown(with: event)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholder.isEmpty else { return }
        let attributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.placeholderTextColor,
            .font: font ?? NSFont.systemFont(ofSize: NSFont.systemFontSize)
        ]
        NSString(string: placeholder).draw(
            at: NSPoint(x: textContainerInset.width + 2, y: textContainerInset.height),
            withAttributes: attributes)
    }
}

private struct ProbeHighlightedText: View {
    let spans: [ChatService.ProbeSpan]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(legend)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(attributed)
        }
    }

    private var attributed: AttributedString {
        var result = AttributedString()
        for span in spans {
            var part = AttributedString(span.text)
            let opacity = span.opacity
            let aligned = (span.score ?? 0) >= 0
            if opacity > 0 {
                part.backgroundColor =
                    aligned
                    ? Color.blue.opacity(opacity)
                    : Color.red.opacity(opacity)
            }
            part.foregroundColor = .primary
            // The SIGN must not be colour-only: a solid rule under aligned
            // text, a dashed one under opposite text, so the two are
            // distinguishable without colour vision.
            if opacity > 0.28 {
                part.underlineStyle =
                    aligned
                    ? Text.LineStyle.single
                    : Text.LineStyle(pattern: .dash, color: nil)
            }
            result += part
        }
        if result.characters.isEmpty {
            return AttributedString("…")
        }
        return result
    }

    private var maxMagnitude: Float {
        max(spans.compactMap(\.score).map { abs($0) }.max() ?? 0, 0.001)
    }

    private var legend: String {
        "Probe highlight: blue with a solid underline = aligned, red with a "
            + "dashed underline = opposite; fixed scale, max |score| "
            + maxMagnitude.formatted(.number.precision(.fractionLength(3)))
    }
}
