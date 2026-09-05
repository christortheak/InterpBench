import Foundation
import Observation
import SteeringKit

/// Durable job identity, recent jobs and bounded log following, independent of study selection.
@Observable @MainActor
public final class StudyRemoteJobController {
    public internal(set) var remoteStatus: String?
    public internal(set) var remoteProfileSummary: String?
    // Persisted so a researcher can reconnect to a running Slurm job after an
    // app restart (Phase C exit criterion).
    public internal(set) var remoteJobID: String? = UserDefaults.standard.string(
        forKey: "SteerLabRemoteJobID")
    {
        didSet { UserDefaults.standard.set(remoteJobID, forKey: "SteerLabRemoteJobID") }
    }
    public internal(set) var remoteLogLines: [String] = []
    public internal(set) var remoteLastUploadedBundle: String?
    public internal(set) var remoteImportedRunDirectory: String?
    public internal(set) var recentServerJobs: [RecentServerJob] = []
    public internal(set) var activeServerJob: ActiveServerJob?
    /// The in-flight server SWEEP job, tracked in its own slot: the
    /// Optimizations Cancel button must never cancel a study-run or Submit
    /// Bundle job that happens to occupy `activeServerJob`/`remoteJobID`.
    /// Same lifecycle as `activeServerJob` (cleared on terminal state; a
    /// timed-out follow keeps it set — the job is still cancellable).
    public internal(set) var activeSweepJob: ActiveServerJob?
    /// The server-side run directory produced by the last completed
    /// run-on-active-server job (a path in the SERVER's tree, not local).
    public internal(set) var lastServerRunDirectory: String?
    public struct RecentServerJob: Identifiable, Sendable, Equatable {
        public let id: String
        public let verb: String
        public let study: String
        public var state: String

        public init(id: String, verb: String, study: String, state: String) {
            self.id = id
            self.verb = verb
            self.study = study
            self.state = state
        }
    }
    public struct ActiveServerJob: Sendable, Equatable {
        public let id: String
        public let verb: String
        public let study: String
    }
    @ObservationIgnored var presentation = StudyJobPresentation()
    @ObservationIgnored private var remoteLogTask: Task<Void, Never>?
    @ObservationIgnored private var streamGeneration = UUID()
    public init() {}
    private func note(_ text: String, severity: PanelNotice.Severity) {
        presentation.note(text, severity)
    }
    private var status: String? { didSet { presentation.status(status) } }

    func noteRecentServerJob(id: String, verb: String, study: String, state: String) {
        if let index = recentServerJobs.firstIndex(where: { $0.id == id }) {
            recentServerJobs[index].state = state
            return
        }
        recentServerJobs.insert(
            RecentServerJob(id: id, verb: verb, study: study, state: state), at: 0)
        if recentServerJobs.count > 15 {
            recentServerJobs.removeLast(recentServerJobs.count - 15)
        }
    }
    public func refreshRecentServerJobs(client: ClusterClient?) async {
        guard let client else { return }
        guard let jobs = try? await client.jobs() else { return }
        for job in jobs
        where job.kind.hasPrefix("experiment:") || job.kind.hasPrefix("study-submit") {
            if let index = recentServerJobs.firstIndex(where: { $0.id == job.id }) {
                recentServerJobs[index].state = job.status
            } else {
                let verb = job.kind.split(separator: ":").last.map(String.init) ?? job.kind
                noteRecentServerJob(id: job.id, verb: verb, study: "—", state: job.status)
            }
        }
    }
    public func reconnectRemoteJob(_ id: String, client: ClusterClient?) async {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            remoteStatus = "enter a job id to reconnect"
            return
        }
        remoteJobID = trimmed
        remoteLogLines = []
        stopRemoteLogStream()
        remoteLogTask = Task { [weak self] in
            await self?.streamRemoteJobLog(jobID: trimmed, client: client)
        }
    }
    func follow(_ operation: @escaping @MainActor () async -> Void) {
        stopRemoteLogStream()
        remoteLogTask = Task { await operation() }
    }

    public func stopRemoteLogStream() {
        streamGeneration = UUID()
        remoteLogTask?.cancel()
        remoteLogTask = nil
    }
    func appendRemoteLogLine(_ line: String) {
        remoteLogLines.append(line)
        if remoteLogLines.count > 400 {
            remoteLogLines.removeFirst(remoteLogLines.count - 400)
        }
    }
    public func streamRemoteJobLog(jobID: String? = nil, client: ClusterClient?) async {
        guard let client else { return }
        await streamRemoteJobLog(
            jobID: jobID,
            stream: { id, receive in try await client.streamJobLog(jobID: id, onLine: receive) },
            job: { id in try await client.job(id) })
    }

    func streamRemoteJobLog(
        jobID: String?,
        stream: @Sendable (String, @escaping @Sendable (String) async -> Void) async throws -> Void,
        job: @Sendable (String) async throws -> RemoteJobRecord
    ) async {
        guard let id = jobID ?? remoteJobID else {
            remoteStatus = "no remote job selected"
            return
        }
        let generation = UUID()
        streamGeneration = generation
        do {
            try await stream(id) { [weak self] line in
                await MainActor.run {
                    guard let self, self.streamGeneration == generation else { return }
                    self.appendRemoteLogLine(line)
                }
            }
            if let job = try? await job(id), streamGeneration == generation {
                remoteStatus = "job \(id): \(job.status)"
            }
        } catch is CancellationError {
            return
        } catch {
            guard streamGeneration == generation else { return }
            remoteStatus = "remote log stream failed: \(error)"
        }
    }
    func followServerJobInDisplay(
        jobID: String,
        client: ClusterClient,
        title: String,
        label: String,
        maxLines: Int = 400,
        mirrorToRemoteLog: Bool = false
    ) async -> RemoteJobRecord? {
        var lines = ["queued job \(jobID)"]
        guard let logID = presentation.startLog(title, lines[0]) else { return nil }

        do {
            try await client.streamJobLog(jobID: jobID) { line in
                await MainActor.run {
                    lines.append(line)
                    if lines.count > maxLines {
                        lines.removeFirst(lines.count - maxLines)
                    }
                    self.presentation.updateLog(logID, title, lines)
                    self.status = "\(label) job \(jobID): \(line)"
                    if mirrorToRemoteLog {
                        self.appendRemoteLogLine(line)
                    }
                }
            }
        } catch is CancellationError {
            return nil
        } catch {
            lines.append("log stream ended: \(error.localizedDescription)")
            self.presentation.updateLog(logID, title, lines)
        }

        if let job = try? await client.job(jobID),
            Self.terminalJobStatuses.contains(job.status) || job.finishedAt != nil
        {
            lines.append("job \(job.status)")
            if let error = job.error, !error.isEmpty {
                lines.append(error)
            }
            self.presentation.updateLog(logID, title, lines)
            return job
        }
        // Stream gone but the job is unresolved (transient fetch error, or a
        // non-terminal status like "cancelling"/"running" after a broken
        // stream): poll until terminal, mirroring ConceptBuilder's fallback.
        let deadline = Date().addingTimeInterval(600)
        while !Task.isCancelled, Date() < deadline {
            if let job = try? await client.job(jobID) {
                if Self.terminalJobStatuses.contains(job.status)
                    || job.finishedAt != nil
                {
                    return job
                }
                if RemoteJobStatusClass.classify(status: job.status) == .resumable {
                    // Actionable, not a dead end (2026-07-22 incident): the
                    // checkpoint line names the Resume button and the
                    // auto-resume toggle.
                    note(
                        RemoteJobStatusClass.checkpointGuidance(jobID: jobID),
                        severity: .warning)
                } else {
                    note(
                        "\(label) job \(jobID) \(job.status)"
                            + (job.logTail.last.map { ": \($0)" } ?? "…"), severity: .info)
                }
            }
            try? await Task.sleep(for: .seconds(2))
        }
        return nil
    }
    static let terminalJobStatuses: Set<String> = [
        "succeeded", "failed", "cancelled", "prepared", "parked",
    ]
}
