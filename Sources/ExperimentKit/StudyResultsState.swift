import Foundation
import Observation
import SteeringKit

/// Result selection and bounded reads, independent of draft edits and job lifetimes.
@Observable @MainActor
public final class StudyResultsState {
    @ObservationIgnored private var detailGeneration = UUID()
    @ObservationIgnored private var listingGeneration = UUID()

    public internal(set) var resultRuns: [StudyRunListItem] = []
    public var selectedResultID: String? {
        didSet {
            if oldValue != selectedResultID {
                loadSelectedResult()
            }
        }
    }
    public internal(set) var selectedResult: StudyRunDetail?
    /// The selected run's Results-browser item, built ONCE per selection
    /// (F10): `RunBrowser.item(at:)` reads config.json and probes sweep.csv
    /// synchronously, and the study-detail body re-evaluates on every live
    /// progress note — that read must never sit inline in a SwiftUI body.
    public internal(set) var selectedResultBrowserItem: RunBrowser.Item?
    /// The active server's enriched `runs/` listing (config.json stamps +
    /// file sizes) for the remote Results browser. Cleared when no server
    /// workspace is active.
    public internal(set) var remoteResultsRuns: [RemoteStampedRunRecord] = []
    public internal(set) var remoteResultsStatus: String?
    /// Whether the last LISTING attempt failed (or could not be made at
    /// all: not connected, invalid URL). A typed signal beside the
    /// free-text status so the browser can tell "the server has no runs"
    /// from "we could not ask it" without substring-matching prose.
    /// Per-file detail fetch failures do NOT set this — they are attributed
    /// per file (`RemoteRunDetailPayload.otherReasons`).
    public internal(set) var remoteResultsFailed = false
    public internal(set) var isLoadingRemoteResults = false
    /// The remote run selected in the Results pane — the remote sibling of
    /// `ChatService.selectedResultsRun` (which stays a LOCAL `RunBrowser.Item`
    /// and is owned elsewhere). The activity pane's summary column mirrors
    /// whichever selection matches the active results source.
    public var selectedRemoteResultsRun: RemoteStampedRunRecord? {
        didSet {
            if oldValue?.id != selectedRemoteResultsRun?.id { detailGeneration = UUID() }
        }
    }
    /// The FILE focused in the LOCAL Results run detail (lives beside
    /// `ChatService.selectedResultsRun`): the Results detail pane lists the
    /// run's files and sets this; the activity viewer's Results mode renders
    /// this file's bounded preview. Pure UI selection state — no run data.
    public var selectedResultsFile: RunBrowser.FileEntry?
    @ObservationIgnored private var browserItemMemo = RunBrowser.MemoizedItem()
    @ObservationIgnored private var repository: StudyResultRepository?
    public init() {}

    public func clearSelectionAndRuns() {
        resultRuns = []
        selectedResultID = nil
        selectedResult = nil
        selectedResultBrowserItem = nil
    }

    public func refresh(
        experimentName: String?, repository: StudyResultRepository,
        selecting preferredID: String? = nil
    ) {
        if self.repository?.workspaceRoot != repository.workspaceRoot { clearSelectionAndRuns() }
        self.repository = repository
        guard let name = experimentName else {
            resultRuns = []
            selectedResultID = nil
            selectedResult = nil
            selectedResultBrowserItem = nil
            return
        }
        resultRuns = repository.list(experimentName: name)
        if let preferredID, resultRuns.contains(where: { $0.id == preferredID }) {
            selectedResultID = preferredID
            loadSelectedResult()
            return
        }
        if let selectedResultID, !resultRuns.contains(where: { $0.id == selectedResultID }) {
            self.selectedResultID = nil
        }
        if selectedResultID == nil {
            selectedResultID =
                resultRuns.first(where: { $0.kind == .run })?.id
                ?? resultRuns.first?.id
        } else {
            loadSelectedResult()
        }
    }
    private func loadSelectedResult() {
        guard let id = selectedResultID,
            let item = resultRuns.first(where: { $0.id == id })
        else {
            selectedResult = nil
            selectedResultBrowserItem = nil
            return
        }
        selectedResult = repository?.detail(for: item)
        // F10: build the browser item here, once per selection — never in a
        // view body. Runs are immutable, so the memo may serve repeats.
        selectedResultBrowserItem = browserItemMemo.item(
            at: URL(filePath: item.path))
    }
    public func refreshRemoteResultsRuns(cluster: ClusterConnectionStore?) async {
        let generation = UUID()
        listingGeneration = generation
        isLoadingRemoteResults = false
        guard let cluster, case .server = cluster.activeWorkspace else {
            remoteResultsRuns = []
            selectedRemoteResultsRun = nil
            remoteResultsStatus = nil
            remoteResultsFailed = false
            return
        }
        // Not connected yet (fresh launch, server workspace persisted from a
        // prior session): a friendly empty state, never an attempted fetch
        // that surfaces a raw transport error.
        guard cluster.remoteState != nil else {
            remoteResultsRuns = []
            selectedRemoteResultsRun = nil
            remoteResultsStatus =
                "not connected to \(cluster.substrateLabel) "
                + "— connect in Compute to browse its runs"
            remoteResultsFailed = true
            return
        }
        cluster.loadStoredToken()
        guard let client = cluster.client else {
            remoteResultsRuns = []
            remoteResultsStatus = "invalid server URL"
            remoteResultsFailed = true
            return
        }
        isLoadingRemoteResults = true
        defer { if listingGeneration == generation { isLoadingRemoteResults = false } }
        do {
            let runs = try await client.stampedRuns()
            guard listingGeneration == generation else { return }
            remoteResultsRuns = runs
            if let selected = selectedRemoteResultsRun {
                selectedRemoteResultsRun = runs.first { $0.id == selected.id }
            }
            let substrate = cluster.substrateLabel
            remoteResultsFailed = false
            remoteResultsStatus =
                "\(runs.count) run\(runs.count == 1 ? "" : "s") "
                + "on \(substrate)"
        } catch {
            guard listingGeneration == generation else { return }
            remoteResultsRuns = []
            remoteResultsFailed = true
            // Human-sized reason, not a raw Swift error dump.
            remoteResultsStatus =
                "could not reach \(cluster.substrateLabel) — "
                + "check the connection in Compute "
                + "(\(error.localizedDescription))"
        }
    }
    public struct RemoteRunDetailPayload: Sendable {
        public var previewed: [RemoteRunFilePreviewItem] = []
        public var other: [RemoteRunFileEntry] = []
        /// Why each `other` file has no preview, keyed by file name — the
        /// fetch failure or the plan's refusal. Without it a file that
        /// FAILED to download is indistinguishable from one that simply has
        /// no preview renderer.
        public var otherReasons: [String: String] = [:]
        public var model: RunResults.Model?

        public init() {}
    }
    public static let remoteReportByteLimit = 4_194_304

    /// Head-fetch caps for the files the semantic model reads. Superset of
    /// the preview needs for the same names, so ONE fetch serves both.
    static let remoteSemanticCaps: [String: Int] = [
        "generations.jsonl": RunBrowser.jsonPreviewByteLimit,
        "report.json": remoteReportByteLimit,
        "experiment.json": remoteReportByteLimit,
        "validation-report.json": remoteReportByteLimit,
        "promoted-movers.json": RunBrowser.jsonPreviewByteLimit,
        "summaries.csv": RunBrowser.jsonPreviewByteLimit,
        "effect-sizes.csv": RunBrowser.jsonPreviewByteLimit,
        "alien-residuals.csv": RunBrowser.jsonPreviewByteLimit,
        "cosine-matrix.csv": RunBrowser.jsonPreviewByteLimit,
        "panel-effects.csv": RunBrowser.jsonPreviewByteLimit,
    ]

    public func loadRemoteRunDetail(
        run: RemoteStampedRunRecord, client: ClusterClient?,
        fetcher: (@Sendable (_ name: String, _ maxBytes: Int) async throws -> RemoteRunFileHead)? =
            nil
    ) async -> RemoteRunDetailPayload {
        // A stale status from a previous run's failed load must not caption
        // THIS load — it re-appears below only if this load itself fails.
        let generation = UUID()
        detailGeneration = generation
        remoteResultsStatus = nil
        var payload = RemoteRunDetailPayload()
        let fetch: @Sendable (String, Int) async throws -> RemoteRunFileHead
        if let fetcher {
            fetch = fetcher
        } else {
            guard let client else {
                payload.other = run.previewFileEntries
                remoteResultsStatus = "invalid server URL"
                return payload
            }
            let runID = run.id
            fetch = { name, maxBytes in
                try await client.runFileHead(
                    runID: runID, name: name, maxBytes: maxBytes)
            }
        }

        // One bounded fetch per listed file, concurrent across files. The
        // request size is the semantic cap for model-feeding files (already
        // ≥ the preview parser's need for that type), else the preview
        // plan's bytes; files with no plan and no semantic role move no
        // bytes at all.
        let entries = run.previewFileEntries
        var fetched: [String: (head: RemoteRunFileHead, requested: Int)] = [:]
        var failures: [String: String] = [:]
        await withTaskGroup(
            of: (name: String, requested: Int, result: Result<RemoteRunFileHead, any Error>).self
        ) { group in
            for file in entries {
                let requested: Int
                if let semanticCap = Self.remoteSemanticCaps[file.name] {
                    requested = semanticCap
                } else if let planBytes = RunBrowser.remoteFetchPlan(
                    name: file.name, size: file.size
                ).requestBytes {
                    requested = planBytes
                } else {
                    continue
                }
                group.addTask {
                    do {
                        return (
                            file.name, requested,
                            .success(try await fetch(file.name, requested))
                        )
                    } catch {
                        return (file.name, requested, .failure(error))
                    }
                }
            }
            for await outcome in group {
                switch outcome.result {
                case .success(let head):
                    fetched[outcome.name] = (head, outcome.requested)
                case .failure(let error):
                    failures[outcome.name] = "\(error)"
                }
            }
        }

        // Previews, in listing order, from the shared bytes (pure parsers,
        // identical caps to local browsing).
        var artifacts = RunResults.ArtifactBytes()
        for file in entries {
            let preview: RunBrowser.FilePreview
            if let (fetchedHead, requested) = fetched[file.name] {
                // The server's actual file size (when stamped) supersedes the
                // listing for truncation captions — 0 there means "unknown".
                preview = RunBrowser.remotePreview(
                    name: file.name,
                    size: fetchedHead.fileSize ?? file.size,
                    data: fetchedHead.data)
                if RunResults.ArtifactBytes.fileNames.contains(file.name) {
                    let head = RunBrowser.remoteHead(
                        data: fetchedHead.data, listedSize: file.size,
                        requestedBytes: requested,
                        serverFileSize: fetchedHead.fileSize,
                        serverTruncated: fetchedHead.truncated)
                    artifacts.assign(
                        name: file.name, data: head.data,
                        truncated: head.truncated)
                }
            } else if let failure = failures[file.name] {
                preview = .unavailable(reason: "fetch failed: \(failure)")
            } else if case .none(let reason) = RunBrowser.remoteFetchPlan(
                name: file.name, size: file.size)
            {
                preview = .unavailable(reason: reason)
            } else {
                preview = .unavailable(reason: "no preview")
            }
            if case .unavailable(let reason) = preview {
                payload.other.append(file)
                payload.otherReasons[file.name] = reason
            } else {
                payload.previewed.append(
                    RemoteRunFilePreviewItem(file: file, preview: preview))
            }
        }

        // Semantic model: pure parsing off the main actor (mirrors the
        // local detail's Task.detached load).
        if !artifacts.isEmpty {
            let runID = run.id
            let built = artifacts
            payload.model = await Task.detached(priority: .userInitiated) {
                RunResults.remoteModel(runID: runID, artifacts: built)
            }.value
        }

        // Surface fetch failures through the existing status line — a
        // missing semantic section must be attributable, never a bare nil.
        if !failures.isEmpty, detailGeneration == generation {
            let names = failures.keys.sorted()
            let shown = names.prefix(3).map {
                "\($0) (\(failures[$0] ?? "error"))"
            }
            remoteResultsStatus =
                "run \(run.id): could not fetch "
                + shown.joined(separator: "; ")
                + (names.count > 3 ? " — and \(names.count - 3) more" : "")
        }
        return payload
    }
}
