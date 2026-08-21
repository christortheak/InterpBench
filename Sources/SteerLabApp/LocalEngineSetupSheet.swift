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

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
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
