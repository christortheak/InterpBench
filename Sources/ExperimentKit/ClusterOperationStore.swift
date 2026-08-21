import Foundation
import Synchronization
#if canImport(Darwin)
import Darwin
#endif

// =============================================================================
// Durable cluster-lifecycle operation records (CLUSTER-CLI-LIFECYCLE-PLAN §7.4).
//
// The wizard's observable step records are fine for a window that stays open.
// They are useless for a CLI that submits a Slurm job and EXITS while the job
// queues. Operation records are therefore written atomically to
//
//     ~/Library/Application Support/SteerLab/cluster-operations/<site>/<op>.json
//
// so a later invocation (or the app) recovers the bootstrap plan hash, the
// scheduler job ids, the last observed daemon host, and the forward identity
// instead of starting the work again.
//
// Two hard rules:
//   * NEVER persist credentials or remote token contents. The record carries
//     token PRESENCE and SOURCE, and nothing else. Transcript references are
//     redacted lines only.
//   * A per-site interprocess lock (flock) means the app and a CLI process
//     cannot both bootstrap, submit a controller, or move the same forward.
//     The second caller OBSERVES the running operation and reports it.
// =============================================================================

/// One step's durable status inside an operation record.
public struct ClusterOperationStepRecord: Codable, Sendable, Equatable {
    public enum Status: String, Codable, Sendable {
        case pending
        case running
        case succeeded
        case failed
        case skipped
        /// Submitted and queued — the state that must never decay into a
        /// timeout failure.
        case awaitingScheduler
    }

    public var step: ClusterLifecycleStep
    public var status: Status
    public var startedAt: Date
    public var finishedAt: Date?
    /// Plain-language, redacted. Never a command containing a secret.
    public var message: String

    public init(
        step: ClusterLifecycleStep, status: Status, startedAt: Date,
        finishedAt: Date? = nil, message: String = ""
    ) {
        self.step = step
        self.status = status
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.message = message
    }
}

/// The durable record of one `ensure` / individual operation.
///
/// Every field here is non-secret by construction — see the type's tests,
/// which serialize a record produced against a fake token store and assert the
/// token material appears nowhere in the bytes.
public struct ClusterOperationRecord: Codable, Sendable, Equatable, Identifiable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var operationID: String
    public var siteID: String
    /// SHA-256 of the site profile this operation acted on, so a later reader
    /// can tell whether the profile drifted underneath it.
    public var siteProfileHash: String
    public var target: ClusterLifecycleTarget
    /// The mutation set the caller authorized, by stable identifier.
    public var authorizedMutations: [String]
    public var state: ClusterLifecycleState
    public var startedAt: Date
    public var updatedAt: Date
    public var steps: [ClusterOperationStepRecord]

    /// The reviewed bootstrap plan's hash — the durable half of the
    /// dry-run-before-real gate (§7.9), so the review survives process exit.
    public var bootstrapPlanHash: String?
    public var bootstrapJobID: String?
    /// The cluster-side file the bootstrap job writes its exit code into.
    /// Recorded WITH the job id, because polling a job whose status file we
    /// have forgotten falls back to the scheduler alone — and the scheduler
    /// forgets a finished job long before we would stop caring about it.
    public var bootstrapStatusFile: String?
    public var controllerJobID: String?
    public var lastDaemonHost: String?
    public var lastServerBuild: String?
    public var localPort: Int?
    /// Site + remote identity + target host + remote port + local port
    /// (§7.7) — what makes a forward adoptable rather than duplicated.
    public var forwardIdentity: String?
    /// Whether a bearer token was available, and from where. PRESENCE ONLY.
    public var tokenAvailable: Bool
    public var tokenSource: String?
    /// Stable failure code + concrete repair action (plan acceptance §9).
    public var failureCode: String?
    public var repairAction: String?
    /// Redacted transcript lines only.
    public var transcript: [String]

    public var id: String { operationID }

    public init(
        operationID: String,
        siteID: String,
        siteProfileHash: String,
        target: ClusterLifecycleTarget,
        authorizedMutations: [String] = [],
        state: ClusterLifecycleState = .running,
        startedAt: Date = Date(),
        updatedAt: Date = Date(),
        steps: [ClusterOperationStepRecord] = [],
        bootstrapPlanHash: String? = nil,
        bootstrapJobID: String? = nil,
        bootstrapStatusFile: String? = nil,
        controllerJobID: String? = nil,
        lastDaemonHost: String? = nil,
        lastServerBuild: String? = nil,
        localPort: Int? = nil,
        forwardIdentity: String? = nil,
        tokenAvailable: Bool = false,
        tokenSource: String? = nil,
        failureCode: String? = nil,
        repairAction: String? = nil,
        transcript: [String] = []
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.operationID = operationID
        self.siteID = siteID
        self.siteProfileHash = siteProfileHash
        self.target = target
        self.authorizedMutations = authorizedMutations
        self.state = state
        self.startedAt = startedAt
        self.updatedAt = updatedAt
        self.steps = steps
        self.bootstrapPlanHash = bootstrapPlanHash
        self.bootstrapJobID = bootstrapJobID
        self.bootstrapStatusFile = bootstrapStatusFile
        self.controllerJobID = controllerJobID
        self.lastDaemonHost = lastDaemonHost
        self.lastServerBuild = lastServerBuild
        self.localPort = localPort
        self.forwardIdentity = forwardIdentity
        self.tokenAvailable = tokenAvailable
        self.tokenSource = tokenSource
        self.failureCode = failureCode
        self.repairAction = repairAction
        self.transcript = transcript
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        guard version <= Self.currentSchemaVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .schemaVersion, in: container,
                debugDescription:
                    "operation record schemaVersion \(version) is newer than this build "
                    + "understands (\(Self.currentSchemaVersion))")
        }
        schemaVersion = version
        operationID = try container.decode(String.self, forKey: .operationID)
        siteID = try container.decode(String.self, forKey: .siteID)
        siteProfileHash =
            try container.decodeIfPresent(String.self, forKey: .siteProfileHash) ?? ""
        target =
            try container.decodeIfPresent(ClusterLifecycleTarget.self, forKey: .target)
            ?? .connected
        authorizedMutations =
            try container.decodeIfPresent([String].self, forKey: .authorizedMutations) ?? []
        state =
            try container.decodeIfPresent(ClusterLifecycleState.self, forKey: .state)
            ?? .running
        startedAt = try container.decode(Date.self, forKey: .startedAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? startedAt
        steps =
            try container.decodeIfPresent(
                [ClusterOperationStepRecord].self, forKey: .steps) ?? []
        bootstrapPlanHash = try container.decodeIfPresent(
            String.self, forKey: .bootstrapPlanHash)
        bootstrapJobID = try container.decodeIfPresent(String.self, forKey: .bootstrapJobID)
        bootstrapStatusFile = try container.decodeIfPresent(
            String.self, forKey: .bootstrapStatusFile)
        controllerJobID = try container.decodeIfPresent(String.self, forKey: .controllerJobID)
        lastDaemonHost = try container.decodeIfPresent(String.self, forKey: .lastDaemonHost)
        lastServerBuild = try container.decodeIfPresent(String.self, forKey: .lastServerBuild)
        localPort = try container.decodeIfPresent(Int.self, forKey: .localPort)
        forwardIdentity = try container.decodeIfPresent(String.self, forKey: .forwardIdentity)
        tokenAvailable =
            try container.decodeIfPresent(Bool.self, forKey: .tokenAvailable) ?? false
        tokenSource = try container.decodeIfPresent(String.self, forKey: .tokenSource)
        failureCode = try container.decodeIfPresent(String.self, forKey: .failureCode)
        repairAction = try container.decodeIfPresent(String.self, forKey: .repairAction)
        transcript = try container.decodeIfPresent([String].self, forKey: .transcript) ?? []
    }

    /// Whether this record describes work that is still in flight (a later
    /// invocation should adopt it rather than start a second one).
    public var isTerminal: Bool {
        switch state {
        case .ready, .failed, .blocked: true
        case .planned, .running, .pending, .needsHumanAuthentication,
            .needsApproval, .degraded:
            false
        }
    }

    /// Update-or-append one step, preserving its original start time.
    public mutating func note(
        step: ClusterLifecycleStep, status: ClusterOperationStepRecord.Status,
        message: String = "", now: Date = Date()
    ) {
        if let index = steps.firstIndex(where: { $0.step == step }) {
            steps[index].status = status
            steps[index].message = message
            if status != .running { steps[index].finishedAt = now }
        } else {
            steps.append(
                ClusterOperationStepRecord(
                    step: step, status: status, startedAt: now,
                    finishedAt: status == .running ? nil : now,
                    message: message))
        }
        updatedAt = now
    }
}

// MARK: - Pending bootstrap jobs (review finding 5)

/// A bootstrap job that was SUBMITTED and has not yet been observed to
/// finish. Durable by design: the whole point is that it outlives the process
/// (and the SSH session) that submitted it.
public struct ClusterPendingBootstrapJob: Sendable, Equatable {
    public var siteID: String
    public var jobID: String
    /// The cluster-side file the job writes its exit code into, when the
    /// submitter reported one.
    public var statusFile: String?
    /// The plan hash this job was submitted against, carried so a resumed
    /// poll still names the review it belongs to.
    public var planHash: String?
    public var submittedAt: Date

    public init(
        siteID: String, jobID: String, statusFile: String? = nil,
        planHash: String? = nil, submittedAt: Date = Date()
    ) {
        self.siteID = siteID
        self.jobID = jobID
        self.statusFile = statusFile
        self.planHash = planHash
        self.submittedAt = submittedAt
    }
}

/// The durable seam for in-flight bootstrap jobs.
///
/// This is deliberately a VIEW over the existing operation records rather than
/// a second store: a bootstrap job id already had a home
/// (`ClusterOperationRecord.bootstrapJobID`), and two places to look for the
/// same fact is how a recovery path goes stale.
public protocol ClusterBootstrapJobJournal: Sendable {
    /// The newest submitted-but-unfinished bootstrap job for a site, or nil.
    func pendingBootstrapJob(forSite siteID: String) -> ClusterPendingBootstrapJob?
    /// Record a job as in flight. Called BEFORE the first status poll, so a
    /// process that dies mid-poll still leaves the job adoptable.
    func recordPendingBootstrapJob(_ job: ClusterPendingBootstrapJob) throws
    /// Close a pending job out once it has a verdict, so a later invocation
    /// stops resuming it.
    func finishPendingBootstrapJob(
        forSite siteID: String, jobID: String, succeeded: Bool, message: String)
}

extension ClusterOperationStore: ClusterBootstrapJobJournal {

    /// The last bootstrap job id this site is known to have submitted.
    public func lastBootstrapJobID(forSite siteID: String) -> String? {
        records(forSite: siteID).compactMap(\.bootstrapJobID).first
    }

    public func pendingBootstrapJob(
        forSite siteID: String
    ) -> ClusterPendingBootstrapJob? {
        guard let record = records(forSite: siteID).first(where: {
            !$0.isTerminal && ($0.bootstrapJobID?.isEmpty == false)
        }), let jobID = record.bootstrapJobID else { return nil }
        return ClusterPendingBootstrapJob(
            siteID: siteID, jobID: jobID, statusFile: record.bootstrapStatusFile,
            planHash: record.bootstrapPlanHash, submittedAt: record.startedAt)
    }

    public func recordPendingBootstrapJob(_ job: ClusterPendingBootstrapJob) throws {
        var record = ClusterOperationRecord(
            operationID: Self.newOperationID(now: job.submittedAt),
            siteID: job.siteID, siteProfileHash: "", target: .bootstrapped,
            state: .pending, startedAt: job.submittedAt, updatedAt: job.submittedAt,
            bootstrapPlanHash: job.planHash, bootstrapJobID: job.jobID,
            bootstrapStatusFile: job.statusFile)
        record.note(
            step: .bootstrapApply, status: .awaitingScheduler,
            message: "bootstrap job \(job.jobID) submitted — polled, never resubmitted",
            now: job.submittedAt)
        try save(record)
    }

    /// Closes EVERY non-terminal record naming this job, not just the newest:
    /// one apply can leave both the journal's pending record and the CLI's own
    /// step record open, and a single survivor would keep the next invocation
    /// resuming a job that has already reported.
    public func finishPendingBootstrapJob(
        forSite siteID: String, jobID: String, succeeded: Bool, message: String
    ) {
        for var record in records(forSite: siteID)
        where !record.isTerminal && record.bootstrapJobID == jobID {
            record.state = succeeded ? .ready : .failed
            record.note(
                step: .bootstrapApply, status: succeeded ? .succeeded : .failed,
                message: message)
            try? save(record)
        }
    }
}

// MARK: - Per-site interprocess lock

/// An exclusive, advisory, INTERPROCESS lock on one site's lifecycle, held for
/// as long as the object lives.
///
/// `flock` is per open-file-description, so two `open()`s — in one process or
/// two — genuinely contend. Acquisition is non-blocking by design: a caller
/// that cannot take the lock must OBSERVE the running operation and report it,
/// never queue behind it (a blocked CLI invocation is exactly the fragile
/// foreground wait this plan removes, and a blocking wait inside Swift
/// Testing's cooperative pool would starve it).
///
/// `@unchecked Sendable` is justified: the descriptor is immutable after init,
/// and the only mutable state (the released flag) lives behind a `Mutex`.
public final class ClusterSiteLock: @unchecked Sendable {
    private let descriptor: Int32
    private let released = Mutex<Bool>(false)
    public let siteID: String
    public let lockURL: URL

    fileprivate init(descriptor: Int32, siteID: String, lockURL: URL) {
        self.descriptor = descriptor
        self.siteID = siteID
        self.lockURL = lockURL
    }

    /// Release early; idempotent. `deinit` releases too, so a thrown error
    /// mid-operation cannot strand the lock.
    public func release() {
        let alreadyReleased = released.withLock { flag -> Bool in
            if flag { return true }
            flag = true
            return false
        }
        guard !alreadyReleased else { return }
        #if canImport(Darwin)
        flock(descriptor, LOCK_UN)
        close(descriptor)
        #endif
    }

    deinit { release() }
}

// MARK: - Store

/// Reads and writes operation records, and hands out the per-site lock.
public struct ClusterOperationStore: Sendable {

    public let rootDirectory: URL

    public init(rootDirectory: URL? = nil) {
        self.rootDirectory = rootDirectory ?? ClusterSupportPaths.operationsDirectory
    }

    public func directory(forSite siteID: String) -> URL {
        rootDirectory.appending(component: Self.safeComponent(siteID))
    }

    public func url(forSite siteID: String, operationID: String) -> URL {
        directory(forSite: siteID)
            .appending(component: Self.safeComponent(operationID) + ".json")
    }

    // MARK: Record IO

    /// Write a record atomically. The caller owns the site lock; this method
    /// does not take one, so a single operation can checkpoint repeatedly.
    public func save(_ record: ClusterOperationRecord) throws {
        let data: Data
        do {
            data = try ClusterSupportPaths.encoder().encode(record)
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "operation \(record.operationID): \(error.localizedDescription)")
        }
        try ClusterSupportPaths.writeAtomically(
            data, to: url(forSite: record.siteID, operationID: record.operationID))
    }

    public func record(
        siteID: String, operationID: String
    ) -> ClusterOperationRecord? {
        guard let data = try? Data(contentsOf: url(forSite: siteID, operationID: operationID))
        else { return nil }
        return try? ClusterSupportPaths.decoder().decode(
            ClusterOperationRecord.self, from: data)
    }

    /// Every record for a site, newest first. A record that no longer decodes
    /// is skipped, never fatal — the same per-entry leniency as the registry.
    public func records(forSite siteID: String) -> [ClusterOperationRecord] {
        let directory = directory(forSite: siteID)
        guard let names = try? FileManager.default.contentsOfDirectory(
            atPath: directory.path)
        else { return [] }
        let decoder = ClusterSupportPaths.decoder()
        return names
            .filter { $0.hasSuffix(".json") }
            .compactMap { name -> ClusterOperationRecord? in
                guard let data = try? Data(contentsOf: directory.appending(component: name))
                else { return nil }
                return try? decoder.decode(ClusterOperationRecord.self, from: data)
            }
            .sorted { $0.startedAt > $1.startedAt }
    }

    public func latestRecord(forSite siteID: String) -> ClusterOperationRecord? {
        records(forSite: siteID).first
    }

    /// The newest record that is still in flight — what a second caller
    /// reports instead of starting duplicate work.
    public func activeRecord(forSite siteID: String) -> ClusterOperationRecord? {
        records(forSite: siteID).first { !$0.isTerminal }
    }

    /// The most recently REVIEWED bootstrap plan hash for a site — the
    /// durable half of the dry-run-before-real gate across processes.
    public func lastBootstrapPlanHash(forSite siteID: String) -> String? {
        records(forSite: siteID).compactMap(\.bootstrapPlanHash).first
    }

    /// The last controller job id this site is known to have submitted.
    /// Controller inspection reconciles from here (§7.10) rather than from a
    /// host file, so a stale `serverd.host` can never masquerade as proof.
    public func lastControllerJobID(forSite siteID: String) -> String? {
        records(forSite: siteID).compactMap(\.controllerJobID).first
    }

    // MARK: Locking

    /// Take the site's exclusive lifecycle lock, or return nil when another
    /// process (or another task in this one) already holds it.
    public func acquireLock(siteID: String) throws -> ClusterSiteLock? {
        let directory = directory(forSite: siteID)
        try ClusterSupportPaths.ensureDirectory(directory)
        let lockURL = directory.appending(component: "lifecycle.lock")
        #if canImport(Darwin)
        let descriptor = open(lockURL.path, O_RDWR | O_CREAT, 0o600)
        guard descriptor >= 0 else {
            throw ClusterLifecycleError.storeUnwritable(
                "\(lockURL.path): could not open the site lock (errno \(errno))")
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            return nil
        }
        return ClusterSiteLock(descriptor: descriptor, siteID: siteID, lockURL: lockURL)
        #else
        return ClusterSiteLock(descriptor: -1, siteID: siteID, lockURL: lockURL)
        #endif
    }

    // MARK: Identifiers

    /// `cluster-<yyyyMMddHHmmss>-<random>` — sortable, non-secret, unique.
    public static func newOperationID(now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let suffix = String(UInt32.random(in: 0..<0xFFFF), radix: 16)
        return "cluster-\(formatter.string(from: now))-\(suffix)"
    }

    /// Keep caller-supplied ids inside one directory component — a site id is
    /// DATA (it can arrive from a shared profile or a CLI flag), and `../` in
    /// a path component is a traversal. Dots are folded to hyphens rather
    /// than allowed: no id this store mints needs one, and `..` is the whole
    /// attack.
    static func safeComponent(_ value: String) -> String {
        let cleaned = value.map { character -> Character in
            character.isLetter || character.isNumber || character == "-"
                || character == "_"
                ? character : "-"
        }
        let text = String(cleaned)
        return text.isEmpty ? "site" : text
    }
}
