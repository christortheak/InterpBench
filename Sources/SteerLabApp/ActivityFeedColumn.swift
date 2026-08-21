import AppKit
import ExperimentKit
import SwiftUI

/// The display pane's non-chat mode: a read-only activity view for sections
/// that are not Playground. Long-running local work (vector builds, LoRA
/// training, study runs) mirrors its progress lines into the shared
/// transcript via `ChatService.startLiveLog`; this column renders ONLY those
/// live-log messages — never the Playground chat turns that share the same
/// transcript (the "Compute still shows my chat" bug: the mode switched but
/// the feed re-rendered the whole transcript, chat included).
///
/// This feed is ONE workspace-wide log, shared by every section that resolves
/// to `.activity` (Home, Data, Templates, Compute, and the Agents authoring
/// regions). Its content surviving a section switch is correct — it is a log,
/// not that section's own state — and the caption says so, because nothing
/// saying so is what made it read as one tab leaking into another.
struct ActivityFeedColumn: View {
    @Bindable var service: ChatService
    let title: String

    var body: some View {
        VStack(spacing: 0) {
            scopeCaption
            if !runningStates.isEmpty {
                runningBar
            }
            if activityMessages.isEmpty {
                emptyState
            } else {
                feedScroll
            }
        }
    }

    /// The transcript's live-log messages only, from `ChatService`'s
    /// authoritative membership set. It used to be re-derived here by
    /// sniffing the rendered markdown (a bold title line plus a ```text
    /// fence) because the ID set was private; it is exposed now, so the
    /// chat pane and this one split the same array by the same fact.
    private var activityMessages: [ChatService.ChatMessage] {
        service.activityLogTranscript
    }

    /// One fixed-height line (no mode-dependent minimums — macOS 27 beta
    /// split-view hazard) naming the feed's scope.
    private var scopeCaption: some View {
        Text(
            "workspace-wide — jobs, builds, training, and runs from every "
                + "section stream into this one feed")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.tail)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .help(
                "the Activity log is not any one section's display: it is the "
                    + "workspace's log, and the same entries appear in every "
                    + "section that shows it")
    }

    // MARK: Running-state chips (real state only)

    private struct RunningState: Identifiable {
        let id: String
        let label: String
        let systemImage: String
    }

    private var runningStates: [RunningState] {
        var states: [RunningState] = []
        if service.isGenerating {
            states.append(
                .init(id: "chat", label: "generating", systemImage: "text.cursor"))
        }
        if service.fineTuning.isTraining {
            states.append(
                .init(
                    id: "train",
                    label: service.fineTuning.trainingProgress ?? "training adapter",
                    systemImage: "slider.horizontal.2.square.on.square"))
        }
        if service.fineTuning.isRobustnessRunning {
            states.append(
                .init(
                    id: "robustness", label: "robustness check",
                    systemImage: "checklist.checked"))
        }
        if service.experiments.isValidating {
            states.append(
                .init(id: "validate", label: "validating study", systemImage: "seal"))
        }
        if service.experiments.isRunning {
            states.append(
                .init(id: "study", label: "study run", systemImage: "play.circle"))
        }
        if service.experiments.isEvaluating {
            states.append(
                .init(id: "judge", label: "evaluating", systemImage: "scale.3d"))
        }
        if service.multiAgent.isRunning {
            states.append(
                .init(
                    id: "multi-agent", label: "multi-agent run",
                    systemImage: "person.3.sequence"))
        }
        if let job = service.experiments.activeServerJob {
            states.append(
                .init(
                    id: "server-job", label: "server \(job.verb) job \(job.id)",
                    systemImage: "server.rack"))
        }
        return states
    }

    private var runningBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(runningStates) { state in
                    HStack(spacing: 4) {
                        ProgressView()
                            .controlSize(.mini)
                        Label(state.label, systemImage: state.systemImage)
                            .font(.caption2)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(.tint.opacity(0.12)))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .background(.quaternary.opacity(0.35))
    }

    // MARK: Feed

    private var feedScroll: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(activityMessages) { message in
                        MessageBubble(message: message) {
                            copyToClipboard(service.transcriptTurnText(message))
                        }
                        .id(message.id)
                    }
                }
                .padding(12)
            }
            .onAppear {
                if let last = activityMessages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .onChange(of: activityMessages.last?.text) {
                if let last = activityMessages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No activity yet", systemImage: "clock.arrow.circlehalf.clockwise")
        } description: {
            Text(
                "Job, training, build, and study progress appears here as it "
                    + "runs. Chat lives in Playground.")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// The display pane's Results mode: the CONTENTS of the run selected in the
/// Results main pane — its stamps plus the focused file's bounded preview
/// (the file LIST and filters live in the main pane; contents render here —
/// live-testing finding). Read-only. Follows the panel's source: in a remote
/// (unpaired-server) browse it mirrors the remote selection instead, from
/// the listing's stamps + file sizes (no filesystem access — the run lives
/// on the server; remote previews stay in the main pane).
struct ResultsRunSummaryColumn: View {
    @Bindable var service: ChatService
    /// Bounded preview of the focused file (`RunBrowser.preview` reads a
    /// capped head only — big files degrade to a reason line, never a stall).
    @State private var preview: RunBrowser.FilePreview?

    var body: some View {
        Group {
            // Follow the live selection, not the source alone: an unpaired
            // cluster workspace can browse its LOCAL (imported) runs with
            // the source toggle (2026-08-03), so a remote-only column here
            // would show an empty pane beside a populated local browser.
            if service.experiments.resultsSource == .remoteServer,
                service.experiments.selectedRemoteResultsRun != nil
            {
                remoteBody
            } else if let run = service.selectedResultsRun {
                summary(for: run)
            } else if service.experiments.resultsSource == .remoteServer {
                remoteBody
            } else {
                emptyState
            }
        }
        .onAppear { refreshPreview() }
        .onChange(of: service.selectedResultsRun?.id) { refreshPreview() }
        .onChange(of: service.experiments.selectedResultsFile?.id) { refreshPreview() }
    }

    @ViewBuilder
    private var remoteBody: some View {
        if let run = service.experiments.selectedRemoteResultsRun {
            RemoteRunSummaryPane(service: service, run: run)
        } else {
            emptyState
        }
    }

    private func summary(for run: RunBrowser.Item) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header(for: run)
                stampGrid(for: run)
                Divider()
                filePreview
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// The focused file's contents — same box + parsers as everywhere else
    /// (`RunFilePreviewBox` over `RunBrowser.preview`), never a second
    /// parser. The Results detail pane owns the file list and selection.
    @ViewBuilder
    private var filePreview: some View {
        if let file = service.experiments.selectedResultsFile, let preview {
            RunFilePreviewBox(name: file.name, size: file.size, preview: preview)
        } else {
            Text("Select a file in the Results pane to preview its contents here.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func header(for run: RunBrowser.Item) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(run.name)
                .font(.callout.monospaced().weight(.semibold))
                .textSelection(.enabled)
                .lineLimit(2)
                .truncationMode(.middle)
            Spacer()
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([run.url])
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
            .help("reveal this run directory in Finder")
        }
    }

    private func stampGrid(for run: RunBrowser.Item) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 3) {
            stampRow("type", run.runType)
            stampRow("created", run.createdAt)
            stampRow("model", run.modelID)
            stampRow("revision", run.revision)
            stampRow("experiment", run.experiment)
            stampRow("substrate", run.substrate)
            stampRow("engine", run.appVersion)
            if run.runType == nil {
                stampRow("stamp", "no config.json (legacy run type)")
            }
        }
        .font(.caption)
    }

    @ViewBuilder
    private func stampRow(_ label: String, _ value: String?) -> some View {
        if let value {
            GridRow {
                Text(label)
                    .foregroundStyle(.secondary)
                    .gridColumnAlignment(.trailing)
                Text(value)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No run selected", systemImage: "archivebox")
        } description: {
            Text(
                "Select a run directory in the Results pane; its stamps and "
                    + "the focused file's contents render here.")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func refreshPreview() {
        guard service.selectedResultsRun != nil,
            let file = service.experiments.selectedResultsFile
        else {
            preview = nil
            return
        }
        preview = RunBrowser.preview(for: file)
    }
}

/// Remote sibling of the local summary: stamps + file inventory straight
/// from the enriched `/api/runs` listing (name+size entries), source-labeled
/// with the server. No Finder affordances — nothing local exists until an
/// evidence import lands it in this workspace.
private struct RemoteRunSummaryPane: View {
    @Bindable var service: ChatService
    let run: RemoteStampedRunRecord

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                stampGrid
                Divider()
                fileList
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(run.id)
                .font(.callout.monospaced().weight(.semibold))
                .textSelection(.enabled)
                .lineLimit(2)
                .truncationMode(.middle)
            Label("on \(service.cluster.substrateLabel)", systemImage: "server.rack")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var stampGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 3) {
            stampRow("type", run.runType)
            stampRow("created", run.createdAt)
            stampRow("model", run.modelID)
            stampRow("revision", run.revision)
            stampRow("experiment", run.experiment)
            stampRow("substrate", run.substrate)
            stampRow("engine", run.appVersion)
            if run.runType == nil {
                stampRow("stamp", "no config.json (legacy run type)")
            }
        }
        .font(.caption)
    }

    @ViewBuilder
    private func stampRow(_ label: String, _ value: String?) -> some View {
        if let value {
            GridRow {
                Text(label)
                    .foregroundStyle(.secondary)
                    .gridColumnAlignment(.trailing)
                Text(value)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    private var fileList: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(fileCountLabel)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(run.previewFileEntries) { file in
                fileRow(file)
            }
        }
    }

    private var fileCountLabel: String {
        let count = run.previewFileEntries.count
        return "\(count) file\(count == 1 ? "" : "s") on the server"
    }

    private func fileRow(_ file: RemoteRunFileEntry) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text")
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
