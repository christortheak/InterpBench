import Foundation
import Observation

// MARK: - Evidence auto-import (WS3.1 — results come home by default)

/// One evidence bundle eligible for import: assembled from a succeeded
/// server job's result payload (the RICHER source — it carries the
/// server-stamped bundle SHA-256) or from the housekeeping status's
/// `evidence.bundles` list (path + run id only).
public struct EvidenceCandidate: Sendable, Equatable, Identifiable {
    public var bundlePath: String
    public var runId: String?
    public var jobId: String?
    /// Server-stamped bundle hash when known — cross-checked before
    /// extraction by the existing verified importer.
    public var sha256: String?
    /// True when this bundle is a FAILURE RECORD packaged from a job that
    /// did not complete (retention 2026-07-24). It imports and verifies
    /// exactly like any other bundle — the data in it is real — but it is
    /// never a completed result, and every surface that shows it says so.
    public var isPartial: Bool
    /// The server's error string for a partial bundle, so a row can explain
    /// itself without another round trip.
    public var failureSummary: String?

    public var id: String { bundlePath }

    public init(bundlePath: String, runId: String? = nil, jobId: String? = nil,
                sha256: String? = nil, isPartial: Bool = false,
                failureSummary: String? = nil) {
        self.bundlePath = bundlePath
        self.runId = runId
        self.jobId = jobId
        self.sha256 = sha256
        self.isPartial = isPartial
        self.failureSummary = failureSummary
    }
}

/// Auto-import service: while a cluster site is connected (and the per-site
/// flag allows it), poll the durable-job list on a modest interval, download
/// every succeeded run job's evidence bundle that is not yet in the local
/// ledger, run it through the EXISTING hash-verified importer
/// (`EvidenceBundleImporter` — per-file manifest check, unlisted files
/// refuse), and record the import in
/// `<workspace>/.steerlab/imported-remote.json`.
///
/// Failures are never silent: each bundle keeps its last error + attempt
/// count, retries back off exponentially, and retries cap out visibly
/// ("use Import now / the manual path") instead of looping forever. The
/// `events` feed is the observable record for the Activity surfaces and the
/// health card badge.
@Observable @MainActor
public final class EvidenceAutoImportService {

    // MARK: Ledger

    /// One imported bundle. `sha256` is the server-stamped bundle hash when
    /// it was known at import time (per-file verification happened either
    /// way — this is provenance, not the proof).
    public struct LedgerEntry: Codable, Sendable, Equatable {
        public var bundlePath: String
        public var runId: String?
        public var importedAt: String
        public var sha256: String?
        /// True when this entry recorded a FAILURE record rather than a
        /// result (retention 2026-07-24). Defaults false so ledgers written
        /// before partial import existed decode unchanged — every entry
        /// they hold was, correctly, a completed run.
        public var isPartial: Bool = false

        public init(bundlePath: String, runId: String?, importedAt: String,
                    sha256: String?, isPartial: Bool = false) {
            self.bundlePath = bundlePath
            self.runId = runId
            self.importedAt = importedAt
            self.sha256 = sha256
            self.isPartial = isPartial
        }

        /// Hand-written so a LEGACY ledger still decodes. Swift's
        /// synthesized `init(from:)` does not fall back to a property's
        /// default value for a missing key — it throws — so relying on
        /// `isPartial`'s default would have made every pre-2026-07-24
        /// ledger fail to decode, silently reset to empty, and re-import
        /// every bundle it already knew about.
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            bundlePath = try container.decode(String.self, forKey: .bundlePath)
            runId = try container.decodeIfPresent(String.self, forKey: .runId)
            importedAt = try container.decode(String.self, forKey: .importedAt)
            sha256 = try container.decodeIfPresent(String.self, forKey: .sha256)
            isPartial = try container.decodeIfPresent(
                Bool.self, forKey: .isPartial) ?? false
        }
    }

    // MARK: Events

    public enum Outcome: Sendable, Equatable {
        case imported(runDirectory: String)
        /// The run already exists locally (manual import, paired tree) —
        /// recorded in the ledger so it is never fetched again.
        case skippedAlreadyPresent
        /// The server answered a structured skip: the run directory is a
        /// failure record with nothing to bundle (a refused continuation's
        /// ledger-only pipeline dir, 2026-08-11). Noted with the server's
        /// reason; never a failure, never a ledger entry — the chain may
        /// still be resumed and produce evidence later.
        case skippedUnbundleable(note: String)
        case failed(String)
    }

    public struct ImportEvent: Sendable, Equatable, Identifiable {
        public let id: UUID
        public var date: Date
        public var bundlePath: String
        public var runId: String?
        public var outcome: Outcome

        public init(
            id: UUID = UUID(), date: Date, bundlePath: String, runId: String?,
            outcome: Outcome
        ) {
            self.id = id
            self.date = date
            self.bundlePath = bundlePath
            self.runId = runId
            self.outcome = outcome
        }
    }

    public struct FailureRecord: Sendable, Equatable {
        public var message: String
        public var attempts: Int
        public var nextRetryAt: Date
        public var exhausted: Bool
    }

    public struct Configuration: Sendable {
        public var pollInterval: Duration = .seconds(60)
        /// First-retry delay; doubles per attempt up to `retryCap`.
        public var retryBase: TimeInterval = 120
        public var retryCap: TimeInterval = 3600
        /// After this many failed attempts the bundle stops auto-retrying
        /// (still visible, still manually importable).
        public var maxAttempts: Int = 6
        public var maxEvents: Int = 200

        public init() {}
    }

    // MARK: Observable state

    public private(set) var events: [ImportEvent] = []
    public private(set) var failures: [String: FailureRecord] = [:]
    public private(set) var ledgerEntries: [LedgerEntry] = []
    public private(set) var isImporting = false
    public private(set) var lastCheckedAt: Date?
    /// One-line summary of the last pass, for status rows.
    public private(set) var lastSummary: String?

    // MARK: Wiring

    public let workspaceRoot: URL
    public var configuration = Configuration()
    /// The shared connection registry; the service is inert without it
    /// unless test seams are injected.
    @ObservationIgnored private weak var cluster: ClusterConnectionStore?
    /// Test seams: when set, they replace the live client calls.
    @ObservationIgnored private let fetchJobsOverride:
        (@Sendable () async throws -> [RemoteJobRecord])?
    @ObservationIgnored private let performImportOverride:
        (@Sendable (EvidenceCandidate) async throws -> URL)?
    @ObservationIgnored private let packageEvidenceOverride:
        (@Sendable (String) async throws
            -> ClusterClient.EvidencePackageReceipt)?
    @ObservationIgnored private let now: @Sendable () -> Date
    @ObservationIgnored private var pollTask: Task<Void, Never>?

    public init(
        workspaceRoot: URL,
        cluster: ClusterConnectionStore? = nil,
        fetchJobs: (@Sendable () async throws -> [RemoteJobRecord])? = nil,
        performImport: (@Sendable (EvidenceCandidate) async throws -> URL)? = nil,
        packageEvidence: (@Sendable (String) async throws
            -> ClusterClient.EvidencePackageReceipt)? = nil,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.workspaceRoot = workspaceRoot
        self.cluster = cluster
        self.fetchJobsOverride = fetchJobs
        self.performImportOverride = performImport
        self.packageEvidenceOverride = packageEvidence
        self.now = now
        self.ledgerEntries = Self.loadLedger(at: Self.ledgerURL(workspaceRoot: workspaceRoot))
    }

    // MARK: Ledger IO (static, unit-tested round-trip)

    public static func ledgerURL(workspaceRoot: URL) -> URL {
        workspaceRoot.appending(components: ".steerlab", "imported-remote.json")
    }

    /// Absent or unreadable file → empty ledger (create-on-first-write).
    public static func loadLedger(at url: URL) -> [LedgerEntry] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([LedgerEntry].self, from: data)) ?? []
    }

    public static func saveLedger(_ entries: [LedgerEntry], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(entries)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    // MARK: Candidate extraction (pure, unit-tested)

    /// Candidate from a durable job record: succeeded, evidence-capable kind
    /// (same rule as the manual Import buttons), and a bundle path in the
    /// result payload. Carries the server-stamped bundle SHA-256 when
    /// present.
    public static func candidate(fromJob job: RemoteJobRecord) -> EvidenceCandidate? {
        // A job qualifies as a RESULT (succeeded, evidence-capable kind) or
        // as a FAILURE RECORD (retention 2026-07-24) — both come home; only
        // the first is ever citable, and `isPartial` is what carries that
        // distinction into the ledger and the UI.
        let isPartial: Bool
        if ExperimentPanel.jobOffersEvidenceImport(kind: job.kind, state: job.status) {
            isPartial = false
        } else if ExperimentPanel.jobOffersPartialEvidenceImport(job) {
            isPartial = true
        } else {
            return nil
        }
        guard let result = job.result else { return nil }
        let value = JSONValue.object(result)
        guard
            let bundlePath = string(in: value, at: ["runResult", "evidenceBundle", "bundlePath"])
                ?? string(in: value, at: ["evidenceBundle", "bundlePath"])
        else { return nil }
        let sha = string(in: value, at: ["runResult", "evidenceBundle", "bundleSha256"])
            ?? string(in: value, at: ["evidenceBundle", "bundleSha256"])
        let runDirectory = string(in: value, at: ["runResult", "runDirectory"])
            ?? string(in: value, at: ["runDirectory"])
        let runId = runDirectory.map { URL(filePath: $0).lastPathComponent }
            ?? runID(fromBundlePath: bundlePath)
        return EvidenceCandidate(
            bundlePath: bundlePath, runId: runId, jobId: job.id, sha256: sha,
            isPartial: isPartial, failureSummary: job.failureSummary)
    }

    public static func candidates(fromJobs jobs: [RemoteJobRecord]) -> [EvidenceCandidate] {
        var seen = Set<String>()
        return jobs.compactMap(candidate(fromJob:)).filter { seen.insert($0.bundlePath).inserted }
    }

    /// Candidates from the housekeeping status (no hash — the per-file
    /// manifest verification inside the importer still applies).
    public static func candidates(
        fromHousekeepingBundles bundles: [HousekeepingEvidenceBundle]
    ) -> [EvidenceCandidate] {
        var seen = Set<String>()
        return bundles.compactMap { bundle -> EvidenceCandidate? in
            guard let path = bundle.path, !path.isEmpty else { return nil }
            return EvidenceCandidate(
                bundlePath: path,
                runId: bundle.runId ?? runID(fromBundlePath: path),
                jobId: bundle.jobId,
                sha256: nil)
        }
        .filter { seen.insert($0.bundlePath).inserted }
    }

    /// `<run_id>.evidence-bundle.tar.gz` → `<run_id>`; nil for other names.
    public static func runID(fromBundlePath path: String) -> String? {
        let name = URL(filePath: path).lastPathComponent
        let suffix = ".evidence-bundle.tar.gz"
        guard name.hasSuffix(suffix), name.count > suffix.count else { return nil }
        return String(name.dropLast(suffix.count))
    }

    private static func string(in value: JSONValue, at keyPath: [String]) -> String? {
        guard let first = keyPath.first else {
            if case .string(let string) = value { return string }
            return nil
        }
        guard case .object(let object) = value, let child = object[first] else { return nil }
        return string(in: child, at: Array(keyPath.dropFirst()))
    }

    // MARK: Queries for the UI

    public func isImported(bundlePath: String) -> Bool {
        ledgerEntries.contains { $0.bundlePath == bundlePath }
    }

    public var importedRunIDs: Set<String> {
        Set(ledgerEntries.compactMap(\.runId))
    }

    /// Server-listed bundles not yet in the local ledger — the health card's
    /// "evidence pending" number and Import-now payload.
    public func pendingCandidates(
        fromHousekeepingBundles bundles: [HousekeepingEvidenceBundle]
    ) -> [EvidenceCandidate] {
        Self.candidates(fromHousekeepingBundles: bundles)
            .filter { !isImported(bundlePath: $0.bundlePath) }
    }

    public func failure(forBundlePath path: String) -> FailureRecord? {
        failures[path]
    }

    // MARK: Polling lifecycle

    /// Start the background poll loop (idempotent). Each tick is a no-op
    /// unless a cluster site is connected and its auto-import flag is on.
    public func startPolling() {
        guard pollTask == nil else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                // Strong only for the iteration: the loop ends when the
                // owning store releases the service.
                guard let self else { return }
                await self.runOnce()
                let interval = self.configuration.pollInterval
                try? await Task.sleep(for: interval)
            }
        }
    }

    public func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    /// Whether an automatic tick may do work right now.
    private var isEligibleTick: Bool {
        guard let cluster else { return false }
        guard case .server = cluster.activeWorkspace else { return false }
        guard cluster.capabilities != nil else { return false }  // connected
        return cluster.activeAutoImportEnabled
    }

    /// One pass: fetch jobs, extract candidates, import what the ledger does
    /// not know. `force` bypasses the enabled/connected ELIGIBILITY gate (a
    /// "check now" — it still needs a client unless test seams are
    /// injected); per-bundle retry backoff is ALWAYS honored here — only an
    /// explicit `importNow` retries a backing-off bundle early. Returns the
    /// events this pass produced.
    @discardableResult
    public func runOnce(force: Bool = false) async -> [ImportEvent] {
        guard !isImporting else { return [] }
        guard force || isEligibleTick else { return [] }
        guard fetchJobsOverride != nil || cluster?.client != nil else { return [] }
        isImporting = true
        defer { isImporting = false }
        lastCheckedAt = now()
        let jobs: [RemoteJobRecord]
        do {
            jobs = try await fetchJobs()
        } catch {
            lastSummary = "could not list server jobs: \(error.localizedDescription)"
            return []
        }
        let produced = await importAll(
            candidates: Self.candidates(fromJobs: jobs), bypassBackoff: false)
        summarize(produced)
        return produced
    }

    /// Import an explicit candidate list (the health card's Import now with
    /// housekeeping bundles, or a single job row's import button). A direct
    /// user action retries immediately, backoff or not.
    @discardableResult
    public func importNow(candidates: [EvidenceCandidate]) async -> [ImportEvent] {
        guard !isImporting else { return [] }
        isImporting = true
        defer { isImporting = false }
        let produced = await importAll(candidates: candidates, bypassBackoff: true)
        summarize(produced)
        return produced
    }

    @discardableResult
    public func importNow(job: RemoteJobRecord) async -> ImportEvent? {
        guard let candidate = Self.candidate(fromJob: job) else { return nil }
        return await importNow(candidates: [candidate]).first
    }

    /// One-click import for a dead/parked pipeline chain (2026-08-06, a
    /// replication-run incident): package the chain's evidence ON the server —
    /// the packager walks `pipeline.json` to every completed stage run —
    /// then run the bundle through the exact verified path the auto-import
    /// uses: download, server-stamped SHA-256, `EvidenceBundleImporter`,
    /// and `EvidenceRevisionAdoption` (the reconciliation the incident's
    /// manual recovery skipped, which made analyze refuse on the
    /// auto-pinned revision's epoch diff). The runs of a chain the server
    /// could not finish still come home this way; a bundle stamped
    /// incomplete imports as a partial, never as a completed result.
    @discardableResult
    public func importPipeline(runID: String) async -> ImportEvent? {
        do {
            let receipt = try await packageEvidence(runDirectory: runID)
            if receipt.skipped == true {
                // A failure record with nothing to bundle — noted, never
                // failed, never ledgered (a later resume can still produce
                // evidence, and the ledger would suppress its import).
                let note = receipt.reason
                    ?? "the server reported a failure record with nothing to bundle"
                lastSummary = "pipeline '\(runID)' skipped — \(note)"
                return publish(
                    candidate: EvidenceCandidate(
                        bundlePath: "", runId: receipt.runID ?? runID),
                    outcome: .skippedUnbundleable(note: note))
            }
            guard let bundlePath = receipt.bundlePath else {
                lastSummary = "server packaged pipeline '\(runID)' but "
                    + "returned no bundle path — nothing imported"
                return nil
            }
            let candidate = EvidenceCandidate(
                bundlePath: bundlePath,
                runId: receipt.runID ?? runID,
                sha256: receipt.bundleSha256,
                isPartial: receipt.evidenceComplete != true)
            return await importNow(candidates: [candidate]).first
        } catch {
            lastSummary = "could not package pipeline '\(runID)' on the "
                + "server: \(String(describing: error))"
            return nil
        }
    }

    private func packageEvidence(
        runDirectory: String
    ) async throws -> ClusterClient.EvidencePackageReceipt {
        if let packageEvidenceOverride {
            return try await packageEvidenceOverride(runDirectory)
        }
        guard let client = cluster?.client else {
            throw ChatServiceError(
                reason: "no connected server client for evidence packaging")
        }
        return try await client.packageEvidence(runDirectory: runDirectory)
    }

    // MARK: Internals

    private func fetchJobs() async throws -> [RemoteJobRecord] {
        if let fetchJobsOverride { return try await fetchJobsOverride() }
        guard let client = cluster?.client else { return [] }
        return try await client.jobs()
    }

    private func importAll(
        candidates: [EvidenceCandidate], bypassBackoff: Bool
    ) async -> [ImportEvent] {
        var produced: [ImportEvent] = []
        for candidate in candidates {
            guard !isImported(bundlePath: candidate.bundlePath) else { continue }
            if !bypassBackoff, let failure = failures[candidate.bundlePath],
                failure.exhausted || now() < failure.nextRetryAt
            {
                continue  // backing off (or capped out) — not silent: `failures` shows it
            }
            // A run already present locally (manual import, paired tree)
            // needs no download: record it so it is never fetched again.
            if let runId = candidate.runId, localRunExists(runId) {
                appendLedger(for: candidate)
                produced.append(publish(candidate: candidate, outcome: .skippedAlreadyPresent))
                continue
            }
            do {
                let imported = try await performImport(candidate)
                appendLedger(for: candidate)
                failures[candidate.bundlePath] = nil
                produced.append(
                    publish(candidate: candidate, outcome: .imported(runDirectory: imported.path)))
            } catch {
                let message = String(describing: error)
                if message.contains("refusing to overwrite existing run") {
                    // The verified importer's collision refusal: the run is
                    // already durable locally — ledger it and move on.
                    appendLedger(for: candidate)
                    failures[candidate.bundlePath] = nil
                    produced.append(
                        publish(candidate: candidate, outcome: .skippedAlreadyPresent))
                } else {
                    recordFailure(candidate: candidate, message: message)
                    produced.append(publish(candidate: candidate, outcome: .failed(message)))
                }
            }
        }
        return produced
    }

    private func performImport(_ candidate: EvidenceCandidate) async throws -> URL {
        if let performImportOverride { return try await performImportOverride(candidate) }
        guard let client = cluster?.client else {
            throw ChatServiceError(reason: "no connected server client for evidence import")
        }
        let downloads = workspaceRoot.appending(components: ".steerlab", "downloads")
        let localBundle = try await client.downloadArtifact(
            path: candidate.bundlePath, to: downloads)
        let expected = candidate.sha256
        // Extraction + per-file hashing off the main actor (same rule as the
        // manual import path).
        let imported = try await Task.detached {
            try EvidenceBundleImporter.importEvidenceBundle(
                localBundle, expectedSHA256: expected)
        }.value
        // Server-manifest-mutation family (2026-08-04): the AUTO path was
        // how runs arrived WITHOUT revision adoption — only the manual
        // Import Evidence button reconciled. Every imported run's snapshot
        // now reconciles here too; outcomes surface in the import summary.
        let outcome = EvidenceRevisionAdoption.adoptModelRevision(
            fromImportedRun: imported)
        if let notice = EvidenceRevisionAdoption.notice(for: outcome) {
            lastSummary = notice.message
        }
        return imported
    }

    /// Whether a run directory of this id is already in the workspace —
    /// public so the pipeline awaiting-import triage can share the exact
    /// same notion of "already here".
    public static func localRunExists(_ runId: String) -> Bool {
        var isDirectory: ObjCBool = false
        let url = ExperimentStore.runsDirectory.appending(component: runId)
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private func localRunExists(_ runId: String) -> Bool {
        Self.localRunExists(runId)
    }

    private func appendLedger(for candidate: EvidenceCandidate) {
        guard !isImported(bundlePath: candidate.bundlePath) else { return }
        let entry = LedgerEntry(
            bundlePath: candidate.bundlePath,
            runId: candidate.runId,
            importedAt: HousekeepingDates.format(now()),
            sha256: candidate.sha256,
            isPartial: candidate.isPartial)
        ledgerEntries.append(entry)
        do {
            try Self.saveLedger(
                ledgerEntries, to: Self.ledgerURL(workspaceRoot: workspaceRoot))
        } catch {
            lastSummary = "imported, but could not write the ledger: "
                + error.localizedDescription
        }
    }

    private func recordFailure(candidate: EvidenceCandidate, message: String) {
        let attempts = (failures[candidate.bundlePath]?.attempts ?? 0) + 1
        let exhausted = attempts >= configuration.maxAttempts
        let delay = min(
            configuration.retryBase * pow(2, Double(attempts - 1)), configuration.retryCap)
        failures[candidate.bundlePath] = FailureRecord(
            message: message,
            attempts: attempts,
            nextRetryAt: exhausted ? .distantFuture : now().addingTimeInterval(delay),
            exhausted: exhausted)
    }

    private func publish(candidate: EvidenceCandidate, outcome: Outcome) -> ImportEvent {
        let event = ImportEvent(
            date: now(), bundlePath: candidate.bundlePath, runId: candidate.runId,
            outcome: outcome)
        events.insert(event, at: 0)
        if events.count > configuration.maxEvents {
            events.removeLast(events.count - configuration.maxEvents)
        }
        return event
    }

    private func summarize(_ produced: [ImportEvent]) {
        guard !produced.isEmpty else { return }
        var imported = 0
        var skipped = 0
        var unbundleable = 0
        var failed = 0
        for event in produced {
            switch event.outcome {
            case .imported: imported += 1
            case .skippedAlreadyPresent: skipped += 1
            case .skippedUnbundleable: unbundleable += 1
            case .failed: failed += 1
            }
        }
        var parts: [String] = []
        if imported > 0 { parts.append("imported \(imported)") }
        if skipped > 0 { parts.append("already present \(skipped)") }
        if unbundleable > 0 {
            parts.append("skipped \(unbundleable) failure "
                + "record\(unbundleable == 1 ? "" : "s") (nothing to bundle)")
        }
        if failed > 0 { parts.append("FAILED \(failed) (see failures)") }
        lastSummary = "evidence auto-import: " + parts.joined(separator: ", ")
    }
}
