import Foundation

/// Process-wide bound on concurrent evidence transfers against a controller
/// (2026-09-13 incident: a launch-check app began fetching every succeeded
/// job's bundle the local ledger had never seen, was killed mid-transfer, and
/// the controller behind the tunnel stopped answering for half an hour).
///
/// Two independent pools:
///
/// - **downloads** — `GET /api/bundles/download` bodies. At most
///   `maxConcurrentDownloads` (2) in flight from this process, whoever asks:
///   the auto-import service, a job row's Import button, a chain import.
/// - **exports** — `POST /api/science/jobs/{id}/export`, which tars and
///   hashes a job's whole output on the controller (12 GB in the incident).
///   One at a time, and only ever from an EXPLICIT fetch (the Fetch and
///   verify evidence button, `remote science-fetch`); the auto-import
///   service has no path to it.
///
/// Waiters queue in arrival order. The counters are observable so a test can
/// assert the bound held under a burst.
public actor EvidenceTransferGate {

    public static let shared = EvidenceTransferGate()

    nonisolated public let maxConcurrentDownloads: Int
    nonisolated public let maxConcurrentExports: Int

    public private(set) var activeDownloads = 0
    public private(set) var activeExports = 0
    /// Highest concurrency ever observed, per pool — the number a test
    /// asserts against.
    public private(set) var peakDownloads = 0
    public private(set) var peakExports = 0

    private var downloadWaiters: [CheckedContinuation<Void, Never>] = []
    private var exportWaiters: [CheckedContinuation<Void, Never>] = []

    public init(maxConcurrentDownloads: Int = 2, maxConcurrentExports: Int = 1) {
        self.maxConcurrentDownloads = max(1, maxConcurrentDownloads)
        self.maxConcurrentExports = max(1, maxConcurrentExports)
    }

    /// Run `body` holding one download slot; waits for a slot when the pool
    /// is full. The slot is released however `body` exits.
    public func withDownloadSlot<T: Sendable>(
        _ body: @Sendable () async throws -> T
    ) async throws -> T {
        await acquireDownload()
        defer { releaseDownload() }
        return try await body()
    }

    /// Run `body` holding THE export slot (exports serialize).
    public func withExportSlot<T: Sendable>(
        _ body: @Sendable () async throws -> T
    ) async throws -> T {
        await acquireExport()
        defer { releaseExport() }
        return try await body()
    }

    private func acquireDownload() async {
        if activeDownloads < maxConcurrentDownloads {
            activeDownloads += 1
            peakDownloads = max(peakDownloads, activeDownloads)
            return
        }
        await withCheckedContinuation { continuation in
            downloadWaiters.append(continuation)
        }
        // The releaser incremented on our behalf before resuming us.
    }

    private func releaseDownload() {
        if downloadWaiters.isEmpty {
            activeDownloads -= 1
        } else {
            // Hand the slot straight to the next waiter: the count stays
            // put, the peak cannot exceed the bound.
            downloadWaiters.removeFirst().resume()
        }
    }

    private func acquireExport() async {
        if activeExports < maxConcurrentExports {
            activeExports += 1
            peakExports = max(peakExports, activeExports)
            return
        }
        await withCheckedContinuation { continuation in
            exportWaiters.append(continuation)
        }
    }

    private func releaseExport() {
        if exportWaiters.isEmpty {
            activeExports -= 1
        } else {
            exportWaiters.removeFirst().resume()
        }
    }
}
