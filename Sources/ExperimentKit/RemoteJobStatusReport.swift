import Foundation

// =============================================================================
// `remote status` and `remote logs --tail` (issue #7).
//
// The server answers both in fixed fields (`api/job_status.py`). The JSON
// envelope carries the server's document as it arrived (`result.response`),
// so a field this client does not know still reaches an agent; the types
// below read only what the plain-text summary shows a person. Every field is
// optional: a summary never refuses because the server left something out.
// =============================================================================

/// `GET /api/jobs/{id}/status`, as far as the plain-text summary reads it.
public struct RemoteJobStatusReport: Codable, Sendable {

    /// The scheduler's view of the job: fixed `sacct`/`squeue` columns.
    public struct SchedulerView: Codable, Sendable {
        public var queryOk: Bool?
        public var rawState: String?
        public var state: String?
        public var reason: String?
        public var expectedStartAt: Double?
        public var startedAt: Double?
        public var endedAt: Double?
        public var nodeList: String?
        public var elapsed: String?
        public var timeLimit: String?
        public var walltimeRemainingSeconds: Double?
        public var exitCode: String?
        public var schedulerOutcome: String?
        public var detail: String?
    }

    /// Records written so far, counted the way a resume counts them.
    public struct Progress: Codable, Sendable {
        public var unit: String?
        public var recordsWritten: Int?
        public var recordsExpected: Int?
        /// `shardStamp` (exact) or `preflightEstimate` (the submission's
        /// planning figure).
        public var recordsExpectedSource: String?
        public var lastRecordAt: Double?
        public var tornTail: Bool?
        public var runID: String?
        public var stage: String?
        public var detail: String?
    }

    public var schemaVersion: Int?
    public var jobId: String?
    public var kind: String?
    public var status: String?
    public var executor: String?
    public var executorJobID: String?
    public var createdAt: Double?
    public var startedAt: Double?
    public var finishedAt: Double?
    public var continuedAs: String?
    public var scheduler: SchedulerView?
    /// Set when the scheduler, not the job, decided how it ended or was
    /// interrupted (`TIMEOUT`, `OUT_OF_MEMORY`, `PREEMPTED`, `NODE_FAIL`,
    /// `CANCELLED`, …). Beside the status, never a status value.
    public var schedulerOutcome: String?
    public var deathDetail: String?
    public var progress: Progress?
    public var notes: [String]?
    /// A split run: one entry per part (its current continuation, when the
    /// part was resumed).
    public var shards: [RemoteJobStatusReport]?
    public var shardIndex: Int?
    public var shardCount: Int?
    /// The part's original job record, when `jobId` is its continuation.
    public var shardJob: String?
    public var found: Bool?
    public var detail: String?

    /// Decoded leniently from the server's document: anything it cannot
    /// read is simply absent from the summary.
    public static func from(_ value: JSONValue) -> RemoteJobStatusReport? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return try? JSONDecoder().decode(RemoteJobStatusReport.self, from: data)
    }
}

/// `GET /api/jobs/{id}/log-tail`, as far as the plain-text output reads it.
public struct RemoteJobLogTail: Codable, Sendable {
    public struct OutputFile: Codable, Sendable {
        public var name: String?
        public var exists: Bool?
        public var lines: [String]?
        public var truncated: Bool?
        public var sizeBytes: Int?
        public var bytesRead: Int?
        public var detail: String?
    }

    public struct SchedulerOutput: Codable, Sendable {
        public var available: Bool?
        public var stdout: OutputFile?
        public var stderr: OutputFile?
        public var detail: String?
    }

    public var jobId: String?
    public var status: String?
    public var lines: [String]?
    public var truncated: Bool?
    public var schedulerOutput: SchedulerOutput?
    public var notes: [String]?

    public static func from(_ value: JSONValue) -> RemoteJobLogTail? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return try? JSONDecoder().decode(RemoteJobLogTail.self, from: data)
    }
}

extension ExperimentCLIRunner {

    /// The job id a status read may put in a request path, or a malformed
    /// refusal (64) that names the shape.
    static func remoteJobID(_ value: String, verb: String) throws -> String {
        guard RemoteJobStatusText.isJobID(value) else {
            throw ExperimentError.malformed(
                "'\(value)' is not a job id: a job id is letters, digits, "
                    + "'.', '_', and '-' only",
                repair: "steerlab-cli remote jobs --json — list the server's "
                    + "job records, then run remote \(verb) with an id from it")
        }
        return value
    }

    /// Refuses, with an update repair, when the server predates the read:
    /// its route would answer 404, which reads like "no such job".
    static func requireJobRead(
        _ client: ClusterClient, supported: (ClusterCapabilities) -> Bool,
        verb: String, jobID: String
    ) async throws {
        let capabilities = try await client.capabilities()
        guard supported(capabilities) else {
            throw ExperimentCLIStop(
                exitCode: 1, state: .refused, code: "jobStatusUnsupported",
                reason: "this server's engine predates remote \(verb): it "
                    + "cannot report a job's scheduler view or log tail",
                repairAction: "update the engine on the server (`steerlab-cli "
                    + "cluster push --site <id>`, then restart its controller), "
                    + "or meanwhile follow the job with `steerlab-cli remote "
                    + "jobs --json`",
                payload: ["jobID": .string(jobID)])
        }
    }

    /// The not-found refusal (66) both reads share.
    static func remoteJobNotFound(_ jobID: String) -> ExperimentCLIStop {
        ExperimentCLIStop(
            exitCode: 1, state: .notFound, code: "notFound",
            reason: "no job '\(jobID)' on this server: status and log reads "
                + "cover only jobs this server submitted, named by their "
                + "SteerLab job id (not the scheduler's number)",
            repairAction: "steerlab-cli remote jobs --json — list the server's "
                + "job records, then re-run with an id from it",
            payload: ["jobID": .string(jobID)])
    }
}

/// The plain-text forms of both answers, and the one rule both verbs share
/// about what may name a job.
public enum RemoteJobStatusText {

    /// The most lines `remote logs --tail` asks for; the server's own cap.
    public static let maximumTailLines = 1000

    /// Whether `value` can name a job on the server: letters, digits, `.`,
    /// `_`, and `-`, at most 128 characters, never `.` or `..`. Checked
    /// before the id becomes part of a request path, so it can only ever
    /// name a job, never another route.
    public static func isJobID(_ value: String) -> Bool {
        guard !value.isEmpty, value.count <= 128, value != ".", value != ".."
        else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) && $0.isASCII
                || $0 == "." || $0 == "_" || $0 == "-"
        }
    }

    /// The one-sentence envelope message for a status report.
    public static func message(for report: RemoteJobStatusReport) -> String {
        let id = report.jobId ?? "?"
        var text = "job \(id) is \(report.status ?? "unknown")"
        if let outcome = report.schedulerOutcome {
            text += " (ended by the scheduler: \(outcome))"
        }
        if let records = recordsPhrase(report.progress) {
            text += "; \(records)"
        }
        return text
    }

    /// The summary a person reads: the job, the scheduler's view, its
    /// records, how it ended, and the server's notes; then each part of a
    /// split run, indented.
    public static func lines(
        for report: RemoteJobStatusReport, timeZone: TimeZone = .current
    ) -> [String] {
        var lines: [String] = []
        var head = "job \(report.jobId ?? "?") [\(report.status ?? "unknown")]"
        if let slurm = report.executorJobID { head += " — scheduler job \(slurm)" }
        lines.append(head)
        lines.append(contentsOf: body(for: report, indent: "  ", timeZone: timeZone))
        for shard in report.shards ?? [] {
            let index = shard.shardIndex.map { "\($0 + 1)" } ?? "?"
            let count = shard.shardCount.map { " of \($0)" } ?? ""
            var line = "  part \(index)\(count): job \(shard.jobId ?? "?")"
            if shard.found == false {
                lines.append(line + " — " + (shard.detail ?? "not found"))
                continue
            }
            line += " [\(shard.status ?? "unknown")]"
            if let slurm = shard.executorJobID { line += " — scheduler job \(slurm)" }
            if let original = shard.shardJob { line += " (continues job \(original))" }
            lines.append(line)
            lines.append(contentsOf: body(for: shard, indent: "    ", timeZone: timeZone))
        }
        return lines
    }

    /// The plain-text log tail: the controller's lines, then each of the
    /// job's own scheduler output files under a heading, then the notes.
    public static func lines(for tail: RemoteJobLogTail) -> [String] {
        var lines = tail.lines ?? []
        if tail.truncated == true {
            lines.insert("(earlier controller lines not shown)", at: 0)
        }
        if let output = tail.schedulerOutput {
            if output.available == false {
                lines.append("--- scheduler output: \(output.detail ?? "not available") ---")
            }
            for file in [output.stdout, output.stderr].compactMap({ $0 }) {
                let name = file.name ?? "scheduler output"
                guard file.exists == true else {
                    lines.append("--- \(name): \(file.detail ?? "does not exist") ---")
                    continue
                }
                let shown = file.lines ?? []
                var heading = "--- \(name): last \(shown.count) line(s)"
                if file.truncated == true { heading += ", earlier lines not shown" }
                lines.append(heading + " ---")
                lines.append(contentsOf: shown)
                if let detail = file.detail { lines.append("(\(detail))") }
            }
        }
        lines.append(contentsOf: (tail.notes ?? []).map { "note: \($0)" })
        return lines
    }

    // MARK: - Pieces

    private static func body(
        for report: RemoteJobStatusReport, indent: String, timeZone: TimeZone
    ) -> [String] {
        var lines: [String] = []
        if let view = report.scheduler {
            lines.append(indent + "scheduler: " + schedulerPhrase(view, timeZone: timeZone))
        }
        if let outcome = report.schedulerOutcome {
            lines.append(indent + "ended by the scheduler: \(outcome)")
        }
        if let records = recordsPhrase(report.progress) {
            var line = indent + "records: " + records
            if let last = report.progress?.lastRecordAt {
                line += "; last written \(stamp(last, timeZone: timeZone))"
            }
            lines.append(line)
        } else if let detail = report.progress?.detail {
            lines.append(indent + "records: \(detail)")
        }
        if let death = report.deathDetail {
            lines.append(indent + death)
        }
        if let continued = report.continuedAs {
            lines.append(indent + "continued as job \(continued)")
        }
        for note in report.notes ?? [] where !note.hasPrefix("this job's run continues as") {
            lines.append(indent + "note: \(note)")
        }
        return lines
    }

    private static func schedulerPhrase(
        _ view: RemoteJobStatusReport.SchedulerView, timeZone: TimeZone
    ) -> String {
        guard view.queryOk != false else {
            return "did not answer" + (view.detail.map { " (\($0))" } ?? "")
        }
        guard let raw = view.rawState else {
            return view.detail ?? "has no record of this job"
        }
        var text = raw
        if let reason = view.reason { text += " (\(reason))" }
        if let expected = view.expectedStartAt {
            text += "; expected to start \(stamp(expected, timeZone: timeZone))"
        }
        if let node = view.nodeList { text += " on \(node)" }
        if let started = view.startedAt {
            text += "; started \(stamp(started, timeZone: timeZone))"
        }
        if let elapsed = view.elapsed {
            text += "; elapsed \(elapsed)"
            if let limit = view.timeLimit { text += " of \(limit)" }
        } else if let limit = view.timeLimit {
            text += "; time limit \(limit)"
        }
        if let remaining = view.walltimeRemainingSeconds {
            text += " (\(duration(remaining)) left)"
        }
        if let ended = view.endedAt {
            text += "; ended \(stamp(ended, timeZone: timeZone))"
        }
        if let exit = view.exitCode, view.endedAt != nil { text += "; exit \(exit)" }
        return text
    }

    private static func recordsPhrase(_ progress: RemoteJobStatusReport.Progress?) -> String? {
        guard let progress, let written = progress.recordsWritten else { return nil }
        let unit = progress.unit == "turn" ? "turn" : "record"
        var text: String
        if let expected = progress.recordsExpected {
            text = "\(written) of \(expected) \(unit)s written"
            if progress.recordsExpectedSource == "preflightEstimate" {
                text += " (the total is the submission's planning estimate)"
            }
        } else {
            text = "\(written) \(unit)\(written == 1 ? "" : "s") written"
        }
        if progress.tornTail == true {
            text += ", plus one incomplete line that is not counted"
        }
        return text
    }

    private static func stamp(_ epoch: Double, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd HH:mm zzz"
        return formatter.string(from: Date(timeIntervalSince1970: epoch))
    }

    private static func duration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        if hours > 0 { return "\(hours)h \(String(format: "%02d", minutes))m" }
        if minutes > 0 { return "\(minutes)m" }
        return "\(total)s"
    }
}
