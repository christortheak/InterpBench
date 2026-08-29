import ExperimentKit
import SwiftUI

// =============================================================================
// WP3 — the Local Engine sheet: "bootstrap, but localhost", as the researcher
// sees it.
//
// The rules this view obeys, and why:
//
//   * TEXT, NOT CHROME. Every step renders its precondition and what it
//     decided, in sentences. A row that says "skipped" is useless; a row that
//     says "skipped — Server/.venv.nosync already exists and matches
//     requirements-macos-arm64.lock" is a report.
//   * NOTHING DOWNLOADS BEFORE IT IS NAMED. The download preamble sits above
//     the button that starts the work, not in a log the researcher reads
//     afterwards.
//   * macOS 27 LAYOUT RULE (see MEMORY: split-view minimum crash). This is a
//     plain sheet — no split view, no NavigationSplitView, no column minimum
//     that changes mid-cycle. The one variable-height region (the step list
//     plus the qualification rows) lives inside a ScrollView with a FIXED
//     frame, so the sheet's own height never moves as the flow progresses.
// =============================================================================

struct LocalEngineSetupSheet: View {
    @Bindable var engine: LocalEngineProvisioner
    let service: ChatService
    /// The running server's lifecycle owner — Restart terminates through its
    /// identity-gated stop, never a bare kill.
    let server: LocalServerController

    @Environment(\.dismiss) private var dismiss

    /// The engine's live model registry (`GET /api/state` on loopback),
    /// polled while the sheet is open; nil while the engine is not
    /// answering, which hides the controls section entirely.
    @State private var engineState: RemoteState?
    @State private var engineCapabilities: ClusterCapabilities?
    /// Outcome sentence of the last engine control action.
    @State private var engineControlNote: String?
    @State private var engineActionInFlight = false
    @State private var confirmingRestart = false
    @State private var restarting = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if engineState != nil { engineControlsSection }
                    if !engine.downloadPreamble.isEmpty { downloadNotice }
                    stepList
                    if let report = engine.qualification {
                        qualificationSection(report)
                    }
                    if let skew = engine.skewAdvisory { skewNotice(skew) }
                }
                .padding(.vertical, 2)
            }
            // Fixed, not content-derived: the list grows by many rows as the
            // flow runs, and a sheet that resizes under the pointer is the
            // exact shape macOS 27 punishes.
            .frame(height: 380)
            Divider()
            controls
        }
        .padding(18)
        .frame(width: 620)
        .task { if engine.phase == .unknown { await engine.refreshPlan() } }
        .task { await pollEngine() }
        .confirmationDialog(
            "Restart the local engine?",
            isPresented: $confirmingRestart,
            titleVisibility: .visible
        ) {
            Button("Stop and restart", role: .destructive) {
                Task { await restartEngine() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "Terminates the engine process (unloading every resident "
                    + "model and aborting any in-flight load or generation), "
                    + "then provisions and starts it again. Nothing installed "
                    + "is re-downloaded.")
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Local Python Engine")
                .font(.headline)
            Text(engine.statusLine)
                .font(.subheadline)
                .foregroundStyle(statusColor)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Text(
                "The app provisions its own Python engine the same way a "
                    + "cluster bootstrap provisions a remote one: every step "
                    + "checks what is already there before it does anything, so "
                    + "stopping and re-running continues rather than restarting.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var statusColor: Color {
        switch engine.phase {
        case .failed: return .red
        case .cancelled: return .orange
        case .ready: return .green
        default: return .primary
        }
    }

    // MARK: Engine controls (field incident 2026-08-29)

    // The pane used to be report-only: when a silent 55 GB download held
    // the engine's only resident-model slot, nothing here could unload,
    // cancel, or restart — the researcher SIGTERMed the process by hand.
    // These controls are the pane-side repair; each speaks the engine's
    // typed routes and reports the engine's own answer.

    private var engineClient: ClusterClient? {
        guard let url = URL(string: "http://127.0.0.1:\(engine.port)") else {
            return nil
        }
        return ClusterClient(profile: ClusterConnectionProfile(baseURL: url))
    }

    private var engineControlsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Running engine — resident models")
                .font(.subheadline.weight(.semibold))
            let loaded = engineState?.loadedModels ?? []
            if loaded.isEmpty {
                Text("no models resident — the slot is free")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(loaded) { model in
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(model.modelID)
                            .font(.caption.weight(.medium))
                        Text(residencyLabel(model))
                            .font(.caption2)
                            .foregroundStyle(
                                model.loading == true ? .orange : .secondary)
                    }
                    // The load's live phase — download progress included —
                    // for the researcher who missed the load stream.
                    if model.loading == true, let phase = model.loadPhase {
                        Text(phase)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            HStack(spacing: 8) {
                Button("Unload Models") { Task { await unloadModels() } }
                    .disabled(
                        engineActionInFlight
                            || !loaded.contains { $0.loading != true })
                    .help(
                        "release every idle resident model slot "
                            + "(POST /api/models/unload) — a model mid-"
                            + "generation or mid-load is never touched")
                if loaded.contains(where: { $0.loading == true }),
                    engineCapabilities?.supportsLoadCancel == true
                {
                    Button("Cancel Load") { Task { await cancelEngineLoad() } }
                        .disabled(engineActionInFlight)
                        .help(
                            "interrupt the load in flight and free its slot "
                                + "(POST /api/models/load/cancel) — a download "
                                + "stops within seconds, a weight copy at its "
                                + "next phase boundary")
                }
                Button(restarting ? "Restarting…" : "Restart Engine…") {
                    confirmingRestart = true
                }
                .disabled(restarting || engine.phase.isRunning)
                .help(
                    "terminate the engine process and provision + start it "
                        + "again — the recovery of last resort when a slot "
                        + "is wedged")
                Spacer()
            }
            if let note = engineControlNote {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
    }

    private func residencyLabel(_ model: RemoteLoadedModel) -> String {
        if model.loading == true {
            return model.cancelRequested == true
                ? "loading — cancel requested" : "loading…"
        }
        var parts: [String] = []
        if let device = model.device { parts.append(device) }
        if let dtype = model.dtype { parts.append(dtype) }
        let where_ = parts.isEmpty ? "" : " (\(parts.joined(separator: ", ")))"
        return (model.busy == true ? "busy" : "resident") + where_
    }

    private func pollEngine() async {
        while !Task.isCancelled {
            await refreshEngine()
            try? await Task.sleep(for: .seconds(2))
        }
    }

    private func refreshEngine() async {
        guard let client = engineClient else { return }
        if engineCapabilities == nil {
            engineCapabilities = try? await client.capabilities()
        }
        engineState = try? await client.state()
    }

    private func unloadModels() async {
        guard let client = engineClient else { return }
        engineActionInFlight = true
        defer { engineActionInFlight = false }
        do {
            let result = try await client.unloadModels()
            // The engine's own hint wins: "unloaded: 0" alone is the dead
            // end the incident hit — the hint names the cancel verb.
            engineControlNote = result.hint
                ?? "released \(result.unloaded) resident model "
                + "slot\(result.unloaded == 1 ? "" : "s")"
        } catch {
            engineControlNote = "unload failed: \(error)"
        }
        await refreshEngine()
    }

    private func cancelEngineLoad() async {
        guard let client = engineClient else { return }
        engineActionInFlight = true
        defer { engineActionInFlight = false }
        do {
            let result = try await client.cancelModelLoad()
            engineControlNote = result.note
                ?? "cancel requested for \(result.cancelRequested.count) load(s)"
        } catch {
            engineControlNote = "cancel failed: \(error)"
        }
        await refreshEngine()
    }

    private func restartEngine() async {
        restarting = true
        defer { restarting = false }
        // The provisioner does not retain the server process; the
        // controller's identity-gated stop is the one safe terminate. An
        // engine started AFTER app launch is adopted from its pidfile first.
        server.adoptIfRunning()
        server.stop()
        for _ in 0 ..< 240 {
            if server.phase == .idle,
                !LocalServerPidfile.endpointIsSteerLab(port: engine.port)
            {
                break
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
        if LocalServerPidfile.endpointIsSteerLab(port: engine.port) {
            engineControlNote = "the engine did not stop "
                + "(\(server.statusLine)) — stop it from the terminal that "
                + "started it, then Re-verify"
            return
        }
        engineState = nil
        engineControlNote = "engine stopped — starting it again"
        engine.run(host: service)
    }

    // MARK: Downloads

    private var downloadNotice: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("This will download", systemImage: "arrow.down.circle")
                .font(.subheadline.weight(.semibold))
            ForEach(engine.downloadPreamble, id: \.self) { line in
                Text("• " + line)
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(
                "uv and CPython come from pinned, sha256-verified artifacts; the "
                    + "wheels come from PyPI as pinned by the committed "
                    + "platform lock. Nothing is fetched until you press Set Up.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
    }

    // MARK: Steps

    private var stepList: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(engine.plan) { entry in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: icon(for: entry))
                        .foregroundStyle(color(for: entry))
                        .frame(width: 16)
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(entry.step.title)
                                .font(.subheadline.weight(.semibold))
                            if let note = engine.progress[entry.step] {
                                Text(note)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Text("Checks: " + entry.step.precondition)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(entry.state.summary)
                            .font(.caption)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                }
            }
        }
    }

    private func icon(for entry: LocalEnginePlanEntry) -> String {
        if case .running(let step) = engine.phase, step == entry.step {
            return "circle.dotted"
        }
        if entry.state.isBlocked { return "xmark.octagon.fill" }
        if entry.state.advisory != nil { return "exclamationmark.circle.fill" }
        if entry.state.actions.isEmpty { return "checkmark.circle.fill" }
        return "arrow.right.circle"
    }

    private func color(for entry: LocalEnginePlanEntry) -> Color {
        if entry.state.isBlocked { return .red }
        if entry.state.advisory != nil { return .orange }
        if entry.state.actions.isEmpty { return .green }
        return .secondary
    }

    // MARK: Acceptance

    private func qualificationSection(_ report: SiteQualificationReport) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("site qualify — \(report.generatedBy) on \(report.platform)")
                .font(.subheadline.weight(.semibold))
            Text(report.summaryLine)
                .font(.caption)
                .foregroundStyle(.secondary)
            // Rows, not a verdict: a report that only said "passed" would hide
            // the skip, which is the number that says how much was verified.
            ForEach(report.checks) { check in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(check.status.uppercased())
                        .font(.caption2.monospaced().weight(.semibold))
                        .foregroundStyle(statusColor(check.status))
                        .frame(width: 40, alignment: .leading)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(check.id)
                            .font(.caption.weight(.medium))
                        Text(check.observed)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            if let version = engine.engineVersion {
                Text("engineVersion (from /api/info): \(version)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
    }

    private func statusColor(_ status: String) -> Color {
        switch status {
        case "pass": return .green
        case "warn": return .orange
        case "fail": return .red
        default: return .secondary
        }
    }

    private func skewNotice(_ text: String) -> some View {
        Label(text, systemImage: "info.circle")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Controls

    private var controls: some View {
        HStack {
            Button("Re-check") { Task { await engine.refreshPlan() } }
                .disabled(engine.phase.isRunning)
            Spacer()
            if engine.phase.isRunning {
                Button("Cancel") { engine.cancel() }
                    .help(
                        "stops after the step in flight — nothing already "
                            + "installed is undone, and re-running continues")
            }
            Button("Close") { dismiss() }
            Button(startTitle) {
                engine.run(host: service)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(engine.phase.isRunning || startIsBlocked)
        }
    }

    private var startTitle: String {
        if case .ready = engine.phase { return "Re-verify" }
        return "Set Up"
    }

    private var startIsBlocked: Bool {
        engine.plan.contains { $0.state.isBlocked }
    }
}
