import ExperimentKit
import SwiftUI

// MARK: - Shared start machinery (both surfaces + the model-job warning)

/// The start-sheet minting and model-first sizing shared by the Playground
/// section, the toolbar control (cluster-testing item 1), and the
/// no-GPU-session warning's one-click start (item 2) — one defaults
/// machinery, never three copies.
@MainActor
enum GPUSessionStartFlow {

    static func slurmSiteData(
        cluster: ClusterConnectionStore
    ) -> ClusterSiteProfile.SlurmSiteData? {
        guard case .slurm(let data)? = cluster.activeSite?.scheduler else {
            return nil
        }
        return data
    }

    /// The pre-filled start sheet for the currently selected workspace model
    /// (nil when no model is picked — the session is sized to the model).
    static func sheetModel(service: ChatService) -> GPUSessionStartSheetModel? {
        guard let modelID = service.workspaceSelectedModelID else { return nil }
        let slurm = slurmSiteData(cluster: service.cluster)
        let suggestion = GPUSessionSizing.suggest(modelID: modelID, slurm: slurm)
        return GPUSessionStartSheetModel(
            modelID: modelID,
            gpuTypes: slurm.map { data in
                data.gpuTypes.isEmpty ? data.gpuVRAMGB.keys.sorted() : data.gpuTypes
            } ?? [],
            gpuVRAMGB: slurm?.gpuVRAMGB ?? [:],
            partition: slurm?.resolvedDefaultPartition,
            suggestion: suggestion)
    }

    /// One-click start with exactly the defaults the sheet would pre-fill
    /// (the warning dialog's "Start GPU session, then submit"). With no
    /// model selected the sizing falls back to the site's declared largest
    /// GPU / default gres — the safe over-ask, stated in the suggestion's
    /// rationale.
    static func startWithDefaults(service: ChatService) async {
        let slurm = slurmSiteData(cluster: service.cluster)
        let suggestion = GPUSessionSizing.suggest(
            modelID: service.workspaceSelectedModelID ?? "", slurm: slurm)
        let request = GPUSessionStartRequest(
            gres: suggestion.gres,
            partition: slurm?.resolvedDefaultPartition,
            walltime: suggestion.walltime,
            idleMinutes: suggestion.idleMinutes)
        await service.cluster.gpuSession.start(request: request)
    }
}

/// The stop gate BOTH GPU surfaces use (UI audit 2026-09-06, headline 7).
///
/// The Playground button called `controller.stop()` outright while the
/// toolbar counted unfinished server jobs and confirmed first — two surfaces,
/// one session, two different safeties. The rule itself lives in
/// `GPUSessionStopCheck` (ExperimentKit, unit-tested); this is the glue.
@MainActor
enum GPUSessionStopFlow {
    /// Returns the confirmation message when the researcher must be asked;
    /// nil means nothing was running and the session has already been stopped.
    static func requestStop(cluster: ClusterConnectionStore) async -> String? {
        var count: Int?
        if let client = cluster.client, let jobs = try? await client.jobs() {
            count = GPUSessionStopCheck.unfinishedJobCount(jobs)
        }
        if let message = GPUSessionStopCheck.confirmationMessage(
            unfinishedJobCount: count)
        {
            return message
        }
        await cluster.gpuSession.stop()
        return nil
    }
}

/// The walltime field's shape check, shared with its disabled-reason caption.
///
/// Before the 2026-09-06 audit the only gate was "not empty": a malformed
/// walltime was sent, the sheet had already dismissed, and the failure landed
/// in `lastActionError`, which renders in exactly one place — the Playground.
enum GPUSessionWalltime {
    /// nil when `text` is a usable HH:MM:SS; otherwise the plain-words reason.
    static func problem(in text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return "enter a walltime as HH:MM:SS" }
        let parts = trimmed.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 3 else {
            return "walltime must be HH:MM:SS — three numbers separated by colons"
        }
        let numbers = parts.compactMap { Int($0) }
        guard numbers.count == 3, numbers.allSatisfy({ $0 >= 0 }) else {
            return "walltime must be HH:MM:SS with whole, non-negative numbers"
        }
        guard numbers[1] < 60, numbers[2] < 60 else {
            return "minutes and seconds must each be under 60"
        }
        guard numbers[0] + numbers[1] + numbers[2] > 0 else {
            return "walltime must be longer than zero"
        }
        return nil
    }
}

extension GPUSessionDisplayState {
    /// One traffic-light mapping for every surface rendering the state.
    var uiColor: Color {
        switch self {
        case .ready, .busy: .green
        case .queued, .starting, .ending: .orange
        case .idle: .yellow
        case .failed: .red
        case .unknownState: .orange
        case .off, .ended, .other: .secondary
        }
    }
}

/// Playground GPU-session control (GPU-session plan §2.7): one conspicuous
/// section next to the workspace model picker — this is a running meter on
/// an institutional allocation and must never be ambient. The view is glue
/// only: every decision (states, polling, ended-detection, sizing) lives in
/// `GPUSessionController` / `GPUSessionSizing` (ExperimentKit).
///
/// Callers gate on `capabilities.supportsGPUSession` — on older servers this
/// section never appears.
struct GPUSessionSection: View {
    @Bindable var service: ChatService
    @State private var startSheet: GPUSessionStartSheetModel?
    @State private var confirmingRelease = false
    /// Non-nil presents the stop confirmation — the SAME `GPUSessionStopCheck`
    /// gate the toolbar Stop uses (UI audit 2026-09-06, headline 7).
    @State private var stopConfirmationMessage: String?
    @State private var isCheckingJobs = false

    private var cluster: ClusterConnectionStore { service.cluster }
    private var controller: GPUSessionController { service.cluster.gpuSession }

    var body: some View {
        Section("GPU Session") {
            HStack(spacing: 8) {
                Image(systemName: "bolt.fill")
                    .foregroundStyle(stateColor)
                // The Section header already says "GPU Session"; the row says
                // what it is doing.
                Text(controller.displayState.label)
                    .fontWeight(.medium)
                if let walltime = controller.remainingWalltimeDescription {
                    Text("· \(walltime)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if controller.isActive {
                    if case .idle = controller.displayState {
                        Button("Keep session") {
                            Task { await controller.keepSession() }
                        }
                        .help("reset the worker's idle countdown — the one "
                            + "deliberate keep-alive; ordinary status polling "
                            + "never holds the GPU")
                    }
                    if releaseOffered {
                        // The server refuses to guess whether the allocation
                        // is gone; the stateDetail below names the sacct
                        // check. This releases the slot AFTER that check —
                        // behind a confirmation naming the job, because a
                        // wrong click over a live allocation keeps billing.
                        Button("Release…") {
                            confirmingRelease = true
                        }
                        .help("only after confirming by hand (sacct -j <job>) "
                            + "that the allocation ended — releases the "
                            + "session slot; a live job would keep billing")
                        .confirmationDialog(
                            "Release GPU session"
                                + (controller.record?.slurmJobID.map { " (Slurm job \($0))" } ?? "")
                                + "?",
                            isPresented: $confirmingRelease,
                            titleVisibility: .visible
                        ) {
                            Button("I verified it ended — release", role: .destructive) {
                                Task { await controller.releaseVerifiedGone() }
                            }
                            Button("Cancel", role: .cancel) {}
                        } message: {
                            Text("Confirm with sacct -j "
                                + "\(controller.record?.slurmJobID ?? "<job>") first: "
                                + "releasing while the allocation still runs leaves it "
                                + "billing with nothing tracking it.")
                        }
                    }
                    Button(isCheckingJobs ? "Checking…" : "Stop") {
                        requestStop()
                    }
                    .disabled(isCheckingJobs)
                    .help("end the GPU session — asks first if server jobs are "
                        + "still running; the worker job is cancelled and the "
                        + "controller connection stays up")
                } else {
                    Button("Start…") { presentStartSheet() }
                        .disabled(startDisabled)
                        .help(startHelp)
                }
            }
            // Attached to the row, not the Section: a modified Section can
            // lose its Form grouping.
            .sheet(item: $startSheet) { model in
                GPUSessionStartSheet(service: service, model: model)
            }
            .confirmationDialog(
                "Stop GPU session?",
                isPresented: stopConfirmationPresented,
                titleVisibility: .visible
            ) {
                Button("Stop session", role: .destructive) {
                    Task { await controller.stop() }
                }
                Button("Keep running", role: .cancel) {}
            } message: {
                Text(stopConfirmationMessage ?? "")
            }
            // Why Start is dim, in the row rather than only on hover.
            if !controller.isActive, selectedModelID == nil {
                Text("Pick a server model above first — the session is sized "
                    + "to the model it will hold.")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let node = controller.record?.node, controller.isActive {
                Text(sessionDetailLine(node: node))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let notice = controller.sessionNotice {
                HStack(spacing: 6) {
                    Label(notice, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    // A bordered small control, not a 12pt plain caption: the
                    // old hit target was roughly 12×40 pt (UI audit
                    // 2026-09-06).
                    Button("Dismiss") { controller.clearSessionNotice() }
                        .controlSize(.small)
                        .help("clear this session notice — it says nothing "
                            + "about the session itself, which keeps running")
                }
            }
            if let detail = controller.record?.stateDetail, !detail.isEmpty {
                // The server's own words — e.g. a past-grace failure naming
                // the Slurm job id so the operator can check it by hand.
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let error = controller.lastActionError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: Derived presentation

    /// When the confirmed force-release is offered: `unknown` always, and an
    /// `ending` session whose stateDetail says the scheduler cannot confirm
    /// the cancellation — the server accepts force for both, and an app-only
    /// user must never be stranded in Ending with curl as the only exit.
    private var releaseOffered: Bool {
        if controller.displayState == .unknownState { return true }
        if controller.record?.state == "ending",
            let detail = controller.record?.stateDetail, !detail.isEmpty
        {
            return true
        }
        return false
    }

    private var stateColor: Color {
        controller.displayState.uiColor
    }

    private var startDisabled: Bool {
        selectedModelID == nil || controller.isStartInFlight
    }

    private var startHelp: String {
        selectedModelID == nil
            ? "pick a server model first — the session is sized to the model"
            : "request a GPU worker job sized to \(selectedModelID ?? "")"
    }

    private var selectedModelID: String? {
        service.workspaceSelectedModelID
    }

    private func sessionDetailLine(node: String) -> String {
        var parts = ["on \(node)"]
        if let gres = controller.record?.gres { parts.append(gres) }
        if let jobID = controller.record?.slurmJobID { parts.append("job \(jobID)") }
        return parts.joined(separator: " · ")
    }

    /// Model-first sizing (plan §2.7): the model is already picked in the
    /// workspace model selector; the sheet opens pre-filled from the site's
    /// GPU inventory. Shared machinery — the toolbar control mints the same
    /// sheet.
    private func presentStartSheet() {
        startSheet = GPUSessionStartFlow.sheetModel(service: service)
    }

    /// Exactly what the toolbar Stop does, through the one shared gate.
    private func requestStop() {
        guard !isCheckingJobs else { return }
        isCheckingJobs = true
        Task {
            let message = await GPUSessionStopFlow.requestStop(cluster: cluster)
            isCheckingJobs = false
            stopConfirmationMessage = message
        }
    }

    private var stopConfirmationPresented: Binding<Bool> {
        Binding(
            get: { stopConfirmationMessage != nil },
            set: { if !$0 { stopConfirmationMessage = nil } })
    }
}

// MARK: - Toolbar control (cluster-testing item 1)

/// Compact GPU-session start/stop in the window TOP BAR, beside the
/// connection dot — visible whenever the active workspace is a server whose
/// profile supports GPU sessions. The researcher forgot to start a session
/// because the only control lived in the Playground; the session state must
/// be glanceable (and startable) from every panel. Renders the SAME
/// `GPUSessionController` state as the Playground section — two surfaces,
/// one truth.
struct GPUSessionToolbarControl: View {
    @Bindable var service: ChatService
    @State private var startSheet: GPUSessionStartSheetModel?
    /// Non-nil presents the stop confirmation (jobs running, or the job
    /// list could not be checked). Message text from `GPUSessionStopCheck`.
    @State private var stopConfirmationMessage: String?
    @State private var isCheckingJobs = false

    private var cluster: ClusterConnectionStore { service.cluster }
    private var controller: GPUSessionController { cluster.gpuSession }

    var body: some View {
        if case .server = cluster.activeWorkspace,
            cluster.capabilities?.supportsGPUSession == true
        {
            HStack(spacing: 5) {
                Image(systemName: controller.isActive ? "bolt.fill" : "bolt")
                    .foregroundStyle(controller.displayState.uiColor)
                    .imageScale(.small)
                Text(stateLabel)
                    .font(.caption)
                    .monospacedDigit()
                if controller.isStartInFlight || isCheckingJobs
                    || controller.displayState == .queued
                    || controller.displayState == .starting
                    || controller.displayState == .ending
                {
                    ProgressView()
                        .controlSize(.mini)
                }
                // The start sheet dismisses before its request is sent, and
                // its failure lands in `lastActionError`, which used to render
                // ONLY in the Playground's GPU Session section — invisible
                // from every other panel (UI audit 2026-09-06). The toolbar
                // now at least says a start failed and carries the words.
                if let error = controller.lastActionError, !error.isEmpty {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .imageScale(.small)
                        .help("last GPU session action failed: \(error)")
                        .accessibilityLabel("GPU session action failed")
                }
                if controller.isActive {
                    Button(isCheckingJobs ? "Checking…" : "Stop") { requestStop() }
                        .controlSize(.small)
                        .disabled(stopDisabled)
                        .help(stopHelp)
                } else if !controller.isStartInFlight {
                    Button("Start…") { startSheet = GPUSessionStartFlow.sheetModel(service: service) }
                        .controlSize(.small)
                        .disabled(service.workspaceSelectedModelID == nil)
                        .help(startHelp)
                }
            }
            .help("GPU session on \(cluster.substrateLabel): "
                + "\(controller.displayState.label) — the same session the "
                + "Playground's GPU Session section controls")
            .sheet(item: $startSheet) { model in
                GPUSessionStartSheet(service: service, model: model)
            }
            .confirmationDialog(
                "Stop GPU session?",
                isPresented: stopConfirmationPresented,
                titleVisibility: .visible
            ) {
                Button("Stop session", role: .destructive) {
                    Task { await controller.stop() }
                }
                Button("Keep running", role: .cancel) {}
            } message: {
                Text(stopConfirmationMessage ?? "")
            }
        }
    }

    /// "GPU Ready · 1h 42m" — state (Idle carries its own countdown) plus
    /// remaining walltime, straight from the controller's tested mapping.
    private var stateLabel: String {
        var label = "GPU \(controller.displayState.label)"
        if let walltime = controller.remainingWalltimeDescription {
            label += " · \(walltime)"
        }
        return label
    }

    private var stopDisabled: Bool {
        isCheckingJobs
            || controller.displayState == .ending
            || controller.displayState == .unknownState
    }

    private var stopHelp: String {
        if controller.displayState == .unknownState {
            return "the scheduler conversation broke — resolve this session "
                + "from the Playground's GPU Session section (verify on the "
                + "cluster, then release)"
        }
        if controller.displayState == .ending {
            return "already ending — the worker job has been cancelled and "
                + "the session is winding down"
        }
        if isCheckingJobs {
            return "checking whether any server job is still running before "
                + "asking"
        }
        return "end the GPU session — asks first if server jobs are still "
            + "running; the controller connection stays up"
    }

    private var startHelp: String {
        service.workspaceSelectedModelID == nil
            ? "pick a server model first — the session is sized to the model"
            : "request a GPU worker job sized to "
                + (service.workspaceSelectedModelID ?? "")
    }

    /// Stop is one click when nothing is running; with unfinished server
    /// jobs (or an uncheckable job list) it confirms first. Shared with the
    /// Playground section through `GPUSessionStopFlow`; the rule itself lives
    /// in `GPUSessionStopCheck` (ExperimentKit, unit-tested).
    private func requestStop() {
        guard !isCheckingJobs else { return }
        isCheckingJobs = true
        Task {
            let message = await GPUSessionStopFlow.requestStop(cluster: cluster)
            isCheckingJobs = false
            stopConfirmationMessage = message
        }
    }

    private var stopConfirmationPresented: Binding<Bool> {
        Binding(
            get: { stopConfirmationMessage != nil },
            set: { if !$0 { stopConfirmationMessage = nil } })
    }
}

/// Sheet identity + pre-filled form state. `Identifiable` so presentation
/// uses `.sheet(item:)` — the sheet always opens against the model it was
/// minted with, never a blank first render.
struct GPUSessionStartSheetModel: Identifiable {
    let id = UUID()
    var modelID: String
    var gpuTypes: [String]
    var gpuVRAMGB: [String: Int]
    var partition: String?
    var suggestion: GPUSessionSizing.Suggestion
}

/// The small start form: GPU type (from the site's declared vocabulary),
/// walltime, idle timeout — pre-filled by `GPUSessionSizing`, editable
/// before submitting.
struct GPUSessionStartSheet: View {
    @Bindable var service: ChatService
    let model: GPUSessionStartSheetModel

    @Environment(\.dismiss) private var dismiss
    @State private var gpuType: String?
    @State private var walltime: String
    @State private var idleMinutes: Int

    init(service: ChatService, model: GPUSessionStartSheetModel) {
        self.service = service
        self.model = model
        // Never leave the Picker on a nil selection with no highlighted row:
        // a site can declare gpuTypes with no VRAM table, in which case the
        // sizing suggests no type (UI audit 2026-09-06).
        _gpuType = State(
            initialValue: model.suggestion.gpuType ?? model.gpuTypes.first)
        _walltime = State(initialValue: model.suggestion.walltime)
        _idleMinutes = State(initialValue: model.suggestion.idleMinutes)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Start GPU Session")
                .font(.headline)
            Text("Model: \(model.modelID)")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Text(sizingCaption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Form {
                if !model.gpuTypes.isEmpty {
                    Picker("GPU type", selection: $gpuType) {
                        ForEach(model.gpuTypes, id: \.self) { type in
                            Text(gpuTypeLabel(type)).tag(String?.some(type))
                        }
                    }
                    .help("which GPU the worker job asks the scheduler for, "
                        + "from this site's declared inventory — the number "
                        + "beside each is its VRAM")
                }
                TextField("Walltime (HH:MM:SS)", text: $walltime)
                    .help("the hard backstop the scheduler enforces: the "
                        + "worker job is killed at this age even mid-"
                        + "generation, so size it to the work, not the day")
                Stepper(
                    "Idle timeout: \(idleMinutes) min",
                    value: $idleMinutes, in: 5 ... 240, step: 5)
                    .help("how long the worker waits with nothing to do before "
                        + "it exits and gives the allocation back")
            }
            .formStyle(.columns)
            Text("The worker self-exits after \(idleMinutes) idle minutes; "
                + "only real work (loads, generation, probes, Keep session) "
                + "resets the countdown. Walltime is the hard backstop.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // Always-present slot, so accepting a walltime does not resize the
            // sheet under the pointer.
            Text(walltimeProblem ?? " ")
                .font(.caption2)
                .foregroundStyle(.orange)
                .lineLimit(1)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without requesting anything")
                Button("Start") { start() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(walltimeProblem != nil)
                    .help("submit the worker job — it queues like any other "
                        + "job, and the toolbar's bolt reports its state")
            }
        }
        .padding(16)
        .frame(width: 420)
    }

    private var walltimeProblem: String? {
        GPUSessionWalltime.problem(in: walltime)
    }

    private var sizingCaption: String {
        var line = model.suggestion.rationale
        if let need = model.suggestion.estimatedVRAMGB {
            line += String(format: " (≈%.0f GB estimated)", need.rounded(.up))
        }
        return line
    }

    private func gpuTypeLabel(_ type: String) -> String {
        guard let vram = model.gpuVRAMGB[type] else { return type }
        return "\(type) (\(vram) GB)"
    }

    private func start() {
        // With no GPU type to name, fall back to the suggestion's own gres —
        // which on a site with no VRAM table IS the site default the caption
        // above promises. Sending nil dropped that promise (UI audit
        // 2026-09-06); `startWithDefaults` always sent it.
        let request = GPUSessionStartRequest(
            gres: gpuType.map { "gpu:\($0):1" } ?? model.suggestion.gres,
            partition: model.partition,
            walltime: walltime.trimmingCharacters(in: .whitespaces),
            idleMinutes: idleMinutes)
        let controller = service.cluster.gpuSession
        dismiss()
        Task { await controller.start(request: request) }
    }
}
