import CryptoKit
import Foundation

// =============================================================================
// A study submission sent twice lands once (public issue #11)
// =============================================================================
//
// A large study's submission is answered only after the server has measured
// every task prompt — about eight minutes for 400 long transcripts. The client
// gave up at URLSession's 60-second default, reported a failure, and the
// server went on to create the job; submitted again, the study became a second
// job. Three pieces close that:
//
// 1. the study-submission requests wait long enough
//    (`ClusterClient.studySubmissionTimeout`);
// 2. a request that was sent and never answered is reported as
//    `SubmissionOutcomeUnknown` — `pending`, exit 12, never a failure — with
//    the repair "look with `remote jobs`, and submit again only if no job
//    appears";
// 3. every submission carries a key derived from what is submitted
//    (`SubmissionKey`), and a server that advertises `cluster.submissionKey`
//    answers a repeat with the job the first request created.
//
// Python twin: `steerlab_server.client.runner` (`SUBMIT_TIMEOUT`,
// `SubmitOutcomeUnknown`, `submission_key`).

/// The submission key: the SHA-256 of the request body itself, as canonical
/// JSON (sorted keys), with the staged path replaced by what the bundle
/// CONTAINS (`bundleContentDigest`), never by its packaging time or its
/// archive's digest, which every repackaging of an unchanged study changes.
/// The same command with the same flags, from this client, carries the same
/// key however often the study is packaged and uploaded; any change to what
/// is submitted — the verb, the executor, the resources, the number of
/// parts, the target root, a dry run — changes it. The other client sends a
/// different body, so its key differs.
public enum SubmissionKey {
    /// The version of this derivation, hashed into every key. 2: the bundle
    /// is named by its content, not by its archive digest.
    public static let version = 2

    /// What a run bundle contains, as one digest: its study's name and
    /// content hash, and every packaged file's path and SHA-256 in path
    /// order (the `entries` of `steerlab-bundle.json`, as the server's
    /// inspection returns them). Nil when the metadata lists no files.
    /// Python twin: `runner.bundle_content_digest`, pinned to the same
    /// hand-computed digest.
    public static func bundleContentDigest(_ metadata: JSONValue) -> String? {
        guard case .object(let fields) = metadata,
            case .array(let entries)? = fields["entries"], !entries.isEmpty
        else { return nil }
        func text(_ value: JSONValue?) -> String {
            if case .string(let string)? = value { return string }
            return ""
        }
        var rows: [[String: JSONValue]] = []
        for entry in entries {
            guard case .object(let row) = entry else { continue }
            rows.append(["path": .string(text(row["path"])),
                         "sha256": .string(text(row["sha256"]))])
        }
        rows.sort { text($0["path"]) < text($1["path"]) }
        let document: [String: JSONValue] = [
            "entries": .array(rows.map { .object($0) }),
            "experiment": .string(text(fields["experiment"])),
            "experimentContentHash": .string(text(fields["experimentContentHash"])),
        ]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let canonical = try? encoder.encode(document) else { return nil }
        return SHA256.hash(data: canonical).map { String(format: "%02x", $0) }.joined()
    }

    /// The key for `body` (encoded without a key of its own). `identity`
    /// replaces the `pathKey` field: the bundle's digest on the bundle route,
    /// the server manifest's digest on the server-resident route.
    public static func derive<Body: Encodable>(
        body: Body, pathKey: String? = nil, identityKey: String, identity: String
    ) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var document = try JSONDecoder().decode(
            [String: JSONValue].self, from: encoder.encode(body))
        if let pathKey { document.removeValue(forKey: pathKey) }
        document.removeValue(forKey: "submissionKey")
        document[identityKey] = .string(identity)
        document["submissionKeyVersion"] = .number(Double(version))
        let canonical = try encoder.encode(document)
        return SHA256.hash(data: canonical).map { String(format: "%02x", $0) }
            .joined()
    }
}

/// A study submission that was sent and never answered.
///
/// Not a failure: on a cluster the server is usually still preparing the job.
/// The CLI reports it as `pending` (exit 12, "valid asynchronous work is in
/// flight"): the request most likely created a job that is still being
/// prepared, and on a server that recognises a repeat
/// (`repeatIsSafe`), sending the same command again is answered with that job
/// rather than a second one.
public struct SubmissionOutcomeUnknown: Error, CustomStringConvertible, LocalizedError {
    public static let code = "submitOutcomeUnknown"

    /// Where the request went.
    public let endpoint: String
    /// What happened to the request, in a few words.
    public let cause: String
    /// The server advertises `cluster.submissionKey`: a repeat is answered
    /// with the job the first request created.
    public let repeatIsSafe: Bool
    /// How long this client waited, in seconds.
    public let waitedSeconds: TimeInterval

    public init(endpoint: String, cause: String, repeatIsSafe: Bool,
                waitedSeconds: TimeInterval) {
        self.endpoint = endpoint
        self.cause = cause
        self.repeatIsSafe = repeatIsSafe
        self.waitedSeconds = waitedSeconds
    }

    public var reason: String {
        "The submission to \(endpoint) was sent and not answered (\(cause)). "
            + "This is not a failure: the server may still be preparing the job, "
            + "and nothing was sent again."
    }

    public var repairAction: String {
        repair(jobsCommand: "steerlab-cli remote jobs --json")
    }

    /// The repair, naming the exact command that lists this server's jobs.
    public func repair(jobsCommand: String) -> String {
        if repeatIsSafe {
            return "\(jobsCommand)  (the job usually appears within a few "
                + "minutes). If none appears once the server has finished "
                + "measuring, run the same command again with the same flags: "
                + "this server answers that repeat with the job the first "
                + "request created. A command with any flag changed is a new "
                + "submission"
        }
        return "\(jobsCommand)  (look before you submit again: this server "
            + "does not recognise a repeated submission, so a second submit is "
            + "a second study). Submit again only if no job for this study "
            + "appears within a few minutes"
    }

    public var description: String { reason + " To repair: " + repairAction }
    public var errorDescription: String? { description }

    /// Whether `error`, thrown by the submission request itself, means the
    /// request may have reached the server: the wait ran out, the connection
    /// dropped mid-answer, the answer was unreadable (including a success
    /// answer this client could not decode: the job may well have landed),
    /// or a proxy in front of the server stopped waiting (502, 504). A
    /// request that cannot have been sent (no host, no route, refused
    /// connection) is not one.
    public static func cause(of error: Error) -> String? {
        if error is DecodingError {
            return "the server's answer could not be read"
        }
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut: return "the wait for an answer ran out"
            case .networkConnectionLost: return "the connection dropped"
            case .badServerResponse, .cannotParseResponse, .zeroByteResource,
                .cannotDecodeContentData, .cannotDecodeRawData:
                return "the answer could not be read"
            default: return nil
            }
        }
        if case ClusterClient.ClientError.badResponse(let status, _) = error,
            status == 502 || status == 504
        {
            return "a proxy in front of the server answered \(status)"
        }
        return nil
    }
}

/// A resume request (`POST /api/jobs/{id}/resubmit`) that was sent and never
/// answered (review round 4). A resume can run sbatch inside the request, so
/// the server may have resumed the job: `pending`, exit 12, and the repair is
/// to check the job's status before resuming again. Python twin:
/// `runner.ResumeOutcomeUnknown`.
public struct ResumeOutcomeUnknown: Error, CustomStringConvertible, LocalizedError {
    public static let code = "resumeOutcomeUnknown"

    public let endpoint: String
    public let jobID: String
    public let cause: String
    public let waitedSeconds: TimeInterval

    public init(endpoint: String, jobID: String, cause: String, waitedSeconds: TimeInterval) {
        self.endpoint = endpoint
        self.jobID = jobID
        self.cause = cause
        self.waitedSeconds = waitedSeconds
    }

    public var reason: String {
        "The resume request for job \(jobID) to \(endpoint) was sent and not "
            + "answered (\(cause)). This is not a failure: the server may have "
            + "resumed it, and nothing was sent again."
    }

    /// The repair, naming the exact command that reads the job's status.
    public func repair(statusCommand: String) -> String {
        "\(statusCommand)  (check the job's status: a resumed job names the "
            + "job that continues it). Resume again only if it still reads "
            + "checkpointed or cancelled with nothing continuing it"
    }

    public var repairAction: String {
        repair(statusCommand: "steerlab-cli remote status \(jobID) --json")
    }

    /// The app's words for it.
    public var appSentence: String {
        "The resume request for job \(jobID) was sent, but the server did not "
            + "answer (\(cause)). The outcome is unknown: the server may have "
            + "resumed it. Check the job's status in Compute › Server Jobs before "
            + "resuming again."
    }

    public var description: String { reason + " To repair: " + repairAction }
    public var errorDescription: String? { description }
}

/// Watches one submission request's network metrics, to tell a request that
/// was never sent (its connection was never made) from one that may have
/// reached the server. URLError's `.timedOut` covers both; only the second is
/// "unknown". With no metrics (a stubbed session), it says nothing, and the
/// request is treated as possibly sent.
final class SubmissionSendProbe: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var transactions: [SubmissionSendProbe.Transaction] = []

    /// What one attempt's metrics say about its connection and its request.
    struct Transaction: Equatable {
        var lookupStarted = false
        var lookupEnded = false
        var connectStarted = false
        var connectEnded = false
        var requestStarted = false
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        didFinishCollecting metrics: URLSessionTaskMetrics
    ) {
        let seen = metrics.transactionMetrics.map {
            Transaction(
                lookupStarted: $0.domainLookupStartDate != nil,
                lookupEnded: $0.domainLookupEndDate != nil,
                connectStarted: $0.connectStartDate != nil,
                connectEnded: $0.connectEndDate != nil,
                requestStarted: $0.requestStartDate != nil)
        }
        lock.lock()
        transactions = seen
        lock.unlock()
    }

    /// The request certainly never left this machine.
    var neverSent: Bool {
        lock.lock()
        defer { lock.unlock() }
        return Self.neverSent(transactions)
    }

    /// No attempt began its request, and one stopped while looking up the
    /// server's name or making its connection. Anything else, including no
    /// metrics at all, may have been sent.
    static func neverSent(_ transactions: [Transaction]) -> Bool {
        guard !transactions.isEmpty, !transactions.contains(where: \.requestStarted)
        else { return false }
        return transactions.contains {
            ($0.lookupStarted && !$0.lookupEnded)
                || ($0.connectStarted && !$0.connectEnded)
        }
    }
}

extension SubmissionOutcomeUnknown {
    /// The app's words for it, the same as the command line's: the outcome
    /// is unknown, check the server's jobs first, and whether a repeat of
    /// the same study is recognised here.
    public var appSentence: String {
        "The submission was sent, but the server did not answer (\(cause)). "
            + "The outcome is unknown: the server may still be preparing the "
            + "job. Check the server's jobs in Compute › Server Jobs before "
            + "submitting again. "
            + (repeatIsSafe
                ? "This server recognises a repeat of the same study with the "
                    + "same settings and answers it with the job it already "
                    + "made."
                : "This server does not recognise a repeat, so submitting "
                    + "again would run the study twice; submit again only if "
                    + "no job appears within a few minutes.")
    }
}

extension RemoteStudySubmission {
    /// The app's status line for an answer the server marked
    /// `alreadySubmitted` (issue #11), or nil for a first submission.
    public var alreadySubmittedStatus: String? {
        guard alreadySubmitted == true else { return nil }
        return alreadySubmittedNote
            ?? "Already submitted as job \(jobId); nothing was submitted again."
    }
}

extension ClusterCapabilities {
    /// `cluster.submissionKey` (issue #11): a repeated study submission is
    /// answered with the job the first one created while that job is being
    /// prepared or has not ended. Absent on older servers.
    public var supportsSubmissionKey: Bool {
        if case .bool(let value)? = cluster?["submissionKey"] { return value }
        if case .bool(let value)? = features?["submissionKey"] { return value }
        return false
    }
}

extension RemoteStudySubmission {
    /// The walltime check's plain-language advice (issue #11): that the job
    /// is estimated to need longer than the requested walltime, with the
    /// walltime to ask for (`walltimeAdvice`), and that the history it priced
    /// from came from items of a very different length, or does not say how
    /// long its items were (`promptLengthNotice`). Both, when both are said;
    /// nil when there is neither. Python twin: `runner.walltime_notice`.
    public var walltimeNotice: String? {
        guard let check = preflight?.checks.first(where: { $0.id == "walltime" })
        else { return nil }
        let said = ["walltimeAdvice", "promptLengthNotice"].compactMap { key -> String? in
            guard case .string(let text)? = check.data?[key], !text.isEmpty else { return nil }
            return text
        }
        return said.isEmpty ? nil : said.joined(separator: " ")
    }
}
