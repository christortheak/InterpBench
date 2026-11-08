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
/// JSON (sorted keys), with the staged path replaced by the bundle's own
/// digest. The same request for the same bundle always carries the same key
/// (a re-upload stages the same bytes at a new path); any change to what is
/// submitted — the verb, the resources, the number of parts, the target root,
/// a dry run — changes it.
public enum SubmissionKey {
    /// The version of this derivation, hashed into every key.
    public static let version = 1

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
                + "minutes). Sending the same submission again is safe: this "
                + "server answers a repeat with the job the first request "
                + "created, and never submits it twice"
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
    /// dropped mid-answer, the answer was unreadable, or a proxy in front of
    /// the server stopped waiting (502, 504). A request that cannot have been
    /// sent (no host, no route, refused connection) is not one.
    public static func cause(of error: Error) -> String? {
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
    /// The walltime check's plain-language notice: the throughput history it
    /// priced from came from much shorter items than this submission's, or
    /// does not say how long its items were. Nil when there is none.
    public var walltimeNotice: String? {
        guard let check = preflight?.checks.first(where: { $0.id == "walltime" }),
            case .string(let notice)? = check.data?["promptLengthNotice"],
            !notice.isEmpty
        else { return nil }
        return notice
    }
}
