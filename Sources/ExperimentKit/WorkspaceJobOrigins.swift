import Foundation

/// Where one remote job came from, recorded INSIDE the workspace.
///
/// The app used to keep job origins only in its own preferences, so a job a
/// coding assistant submitted through a command line was unknown to it:
/// Import Evidence refused it as "missing or ambiguous origin" until the
/// researcher reconnected by job ID. Both command lines now write this record
/// when they submit, and the app reads it before its own preferences.
///
/// The format is shared with the Python client
/// (`steerlab_server/client/job_origins.py`):
///
///     <workspace>/.steerlab/job-origins/origins.json
///     {"schemaVersion": 1, "jobs": {"<job id>": [<origin>, ...]}}
///
/// A job ID maps to a LIST because job IDs are only unique per server: two
/// servers can hand out the same one, and a client must then refuse rather
/// than guess. Only references and paths live here — never a token, never a
/// Keychain value.
public struct WorkspaceJobOrigin: Codable, Sendable, Equatable {
    /// The durable server identity, spelled exactly as
    /// `ClusterConnectionStore.registryKey` spells it: `ssh://<host>:<port>`
    /// for an SSH site (never the tunnel's local port), the normalized
    /// `scheme://host:port` otherwise.
    public var serverIdentity: String
    /// The URL the submitting client actually spoke to.
    public var endpoint: String
    /// The saved site the submission named, when it named one.
    public var siteID: String?
    /// The server's artifact root (`serve --root`) when the client knew it.
    public var servingRoot: String?
    /// The local workspace the submission was made from.
    public var workspaceRoot: String
    /// Which client wrote the record: `steerlab-cli` or `steerlab`.
    public var submittedBy: String
    /// ISO 8601, UTC.
    public var recordedAt: String
    public var experiment: String?
    public var verb: String?
    /// What kind of submission made the job (`submit-bundle`, `run`,
    /// `resubmit`, `science-submit`).
    public var operation: String?

    public init(
        serverIdentity: String, endpoint: String, siteID: String? = nil,
        servingRoot: String? = nil, workspaceRoot: String, submittedBy: String,
        recordedAt: String = WorkspaceJobOrigins.timestamp(), experiment: String? = nil,
        verb: String? = nil, operation: String? = nil
    ) {
        self.serverIdentity = serverIdentity
        self.endpoint = endpoint
        self.siteID = siteID
        self.servingRoot = servingRoot
        self.workspaceRoot = workspaceRoot
        self.submittedBy = submittedBy
        self.recordedAt = recordedAt
        self.experiment = experiment
        self.verb = verb
        self.operation = operation
    }

    /// The Mac command line's name in `submittedBy`.
    public static let macCommandLine = "steerlab-cli"
    /// The cross-platform Python client's name in `submittedBy`.
    public static let pythonClient = "steerlab"

    /// True when a command line, not this app, submitted the job.
    public var isCommandLine: Bool {
        submittedBy == Self.macCommandLine || submittedBy == Self.pythonClient
    }

    /// Which command line, in a researcher's words.
    public var submitterDescription: String {
        switch submittedBy {
        case Self.macCommandLine: "the Mac command line (steerlab-cli)"
        case Self.pythonClient: "the SteerLab client (steerlab)"
        default: submittedBy
        }
    }

    /// One sentence for a job row's tooltip: who submitted it, for what, and
    /// that the app can act on it.
    public var commandLineSummary: String {
        var text = "Submitted by \(submitterDescription) from this workspace"
        if let experiment, !experiment.isEmpty { text += " for the study “\(experiment)”" }
        if let verb, !verb.isEmpty { text += " (\(verb))" }
        text += ", recorded \(recordedAt). Import evidence and the other job actions work "
            + "here just as they do for jobs this app submitted."
        return text
    }

    /// The display name a refusal uses for the server.
    public var serverDisplayName: String {
        if let siteID, !siteID.isEmpty { return siteID }
        return ClusterConnectionStore.hostLabel(forURLString: endpoint)
    }
}

/// Read and write `.steerlab/job-origins/origins.json`.
///
/// Writers hold an advisory `flock` on a sidecar lock file for the whole
/// read-modify-write and publish with an atomic rename, so two processes —
/// the Mac command line and the Python client, or two assistants at once —
/// never leave a torn or half-merged file. Rows a writer cannot read (a newer
/// client's) are carried through verbatim: a merge only ever replaces the row
/// for the same job on the same server.
public enum WorkspaceJobOrigins {
    public static let schemaVersion = 1

    /// The folder, relative to the workspace root. Its own `.gitignore`
    /// keeps the record out of a workspace's git history: it names servers
    /// and paths that mean something only on this machine.
    public static let relativeDirectory = [".steerlab", "job-origins"]

    public static func directory(workspaceRoot: URL) -> URL {
        relativeDirectory.reduce(workspaceRoot) { $0.appending(component: $1) }
    }

    public static func fileURL(workspaceRoot: URL) -> URL {
        directory(workspaceRoot: workspaceRoot).appending(component: "origins.json")
    }

    static func lockURL(workspaceRoot: URL) -> URL {
        directory(workspaceRoot: workspaceRoot).appending(component: "origins.lock")
    }

    /// The identity of a server reached by URL — the same normalization the
    /// app's registry uses for a direct server.
    public static func serverIdentity(forEndpoint url: URL) -> String {
        ClusterConnectionStore.normalizedEndpointKey(url.absoluteString)
    }

    public static func timestamp(_ date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: date)
    }

    // MARK: Reading

    /// Every readable origin, by job ID. An absent or unreadable file is an
    /// empty record (never an error): the app then falls back to its own
    /// preferences, exactly as before this file existed.
    public static func load(workspaceRoot: URL) -> [String: [WorkspaceJobOrigin]] {
        guard let data = try? Data(contentsOf: fileURL(workspaceRoot: workspaceRoot)),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let jobs = object["jobs"] as? [String: Any]
        else { return [:] }
        var out: [String: [WorkspaceJobOrigin]] = [:]
        for (jobID, rows) in jobs {
            guard let rows = rows as? [Any] else { continue }
            let decoded = rows.compactMap(decodeRow)
            if !decoded.isEmpty { out[jobID] = decoded }
        }
        return out
    }

    public static func origins(forJob jobID: String, workspaceRoot: URL) -> [WorkspaceJobOrigin] {
        load(workspaceRoot: workspaceRoot)[jobID] ?? []
    }

    /// The jobs a command line submitted to one server from this workspace,
    /// by job ID — what Server Jobs marks "from the command line". A record
    /// for another server, or one naming a different serving root when both
    /// roots are known, does not count.
    public static func commandLineOrigins(
        workspaceRoot: URL, serverIdentity: String, servingRoot: String?
    ) -> [String: WorkspaceJobOrigin] {
        var out: [String: WorkspaceJobOrigin] = [:]
        for (jobID, rows) in load(workspaceRoot: workspaceRoot) {
            let matching = rows.filter { row in
                guard row.isCommandLine, row.serverIdentity == serverIdentity else { return false }
                guard let servingRoot, let recorded = row.servingRoot else { return true }
                return recorded == servingRoot
            }
            if matching.count == 1 { out[jobID] = matching[0] }
        }
        return out
    }

    private static func decodeRow(_ row: Any) -> WorkspaceJobOrigin? {
        guard JSONSerialization.isValidJSONObject(row),
            let data = try? JSONSerialization.data(withJSONObject: row),
            let origin = try? JSONDecoder().decode(WorkspaceJobOrigin.self, from: data),
            !origin.serverIdentity.isEmpty
        else { return nil }
        return origin
    }

    // MARK: Writing

    /// Record (or refresh) one job's origin. A row for the same job on the
    /// same server is replaced; a row from another server is kept beside it,
    /// so a colliding ID stays visibly ambiguous rather than silently won.
    public static func record(_ origin: WorkspaceJobOrigin, jobID: String, workspaceRoot: URL) throws {
        let trimmedID = jobID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedID.isEmpty else {
            throw ExperimentError(reason: "a job origin needs a job ID")
        }
        let folder = directory(workspaceRoot: workspaceRoot)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeIgnoreRule(in: folder)
        try withLock(workspaceRoot: workspaceRoot) {
            let url = fileURL(workspaceRoot: workspaceRoot)
            var document: [String: Any] = [:]
            if let data = try? Data(contentsOf: url) {
                if let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    parsed["jobs"] == nil || parsed["jobs"] is [String: Any]
                {
                    document = parsed
                } else {
                    // Never destroy what cannot be read: set it aside, then
                    // start a fresh record beside it.
                    let aside = url.deletingLastPathComponent().appending(
                        component: "origins.json.unreadable-\(Int(Date().timeIntervalSince1970))")
                    try? FileManager.default.moveItem(at: url, to: aside)
                }
            }
            var jobs = document["jobs"] as? [String: Any] ?? [:]
            var rows = (jobs[trimmedID] as? [Any]) ?? []
            rows.removeAll { row in
                (row as? [String: Any])?["serverIdentity"] as? String == origin.serverIdentity
            }
            let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(origin))
            rows.append(encoded)
            jobs[trimmedID] = rows
            document["jobs"] = jobs
            let existingVersion = document["schemaVersion"] as? Int ?? 0
            document["schemaVersion"] = max(existingVersion, schemaVersion)
            let data = try JSONSerialization.data(
                withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url, options: .atomic)
        }
    }

    private static func writeIgnoreRule(in folder: URL) throws {
        let path = folder.appending(component: ".gitignore").path
        let descriptor = Darwin.open(path, O_CREAT | O_EXCL | O_WRONLY, mode_t(0o644))
        if descriptor >= 0 {
            defer { Darwin.close(descriptor) }
            let rule = Data("*\n".utf8)
            _ = rule.withUnsafeBytes { Darwin.write(descriptor, $0.baseAddress, $0.count) }
        } else if errno != EEXIST {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    private static func withLock<T>(workspaceRoot: URL, _ body: () throws -> T) throws -> T {
        let descriptor = Darwin.open(
            lockURL(workspaceRoot: workspaceRoot).path, O_CREAT | O_RDWR, mode_t(0o600))
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { flock(descriptor, LOCK_UN) }
        return try body()
    }
}
