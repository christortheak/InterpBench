import Foundation
import Testing

@testable import ExperimentKit

// Public issue #11 on the Mac client: a large study's submission is answered
// only after the server has measured every task prompt (about eight minutes
// for 400 long transcripts). `remote submit-bundle` gave up at URLSession's
// 60-second default, reported `failed`/70, and the server went on to create
// the job; submitted again, the study became a second job. Pinned here,
// against a server stubbed below the client:
//
// - the submission waits an hour, so a slow answer is waited for;
// - a submission that was sent and never answered is `pending`/12 with the
//   repair "look with `remote jobs`", never a failure;
// - the request carries a submission key derived from what is submitted, to
//   a server that recognises one, and an `alreadySubmitted` answer says so;
// - the walltime check's notice reaches stderr and the envelope.
@Suite(.serialized) struct SubmissionRepeatCLITests {

    struct Outcome {
        var envelope: SteerLabCLIEnvelope
        var exitCode: Int32
        var stderr: String
        var submitBodies: [[String: JSONValue]]
        var submitTimeouts: [TimeInterval]
    }

    enum SubmitBehaviour: @unchecked Sendable {
        /// Answer after `seconds` of (simulated) server work: a request whose
        /// own wait is shorter times out, as URLSession would time it out.
        case answerAfter(seconds: TimeInterval, extra: [String: Any] = [:])
        case fail(URLError.Code)
        case status(Int)
    }

    func submit(
        _ args: [String] = ["--verb", "run", "--executor", "slurm"],
        behaviour: SubmitBehaviour = .answerAfter(seconds: 0),
        advertisesKey: Bool = true
    ) async throws -> Outcome {
        let bodies = WiringLocked<[[String: JSONValue]]>([])
        let timeouts = WiringLocked<[TimeInterval]>([])
        RepeatURLProtocol.handler = { request in
            switch request.url?.path ?? "" {
            case "/api/capabilities":
                let body: [String: Any] = [
                    "apiSchemaVersion": "2026-07-01",
                    "engine": "python-hf-transformers",
                    "instrumentation": [
                        "policy-evidence-v2", "policy-v1", "probe-readings-v1",
                    ],
                    "profile": ["profile": "cluster", "executor": "slurm"],
                    "cluster": advertisesKey ? ["submissionKey": true] : [:],
                ]
                return (try JSONSerialization.data(withJSONObject: body), 200)
            case "/api/bundles/inspect":
                return (Data(#"{"kind":"runBundle","experiment":"demo","bundleSha256":"\#(String(repeating: "d", count: 64))"}"#.utf8), 200)
            case "/api/studies/submit-bundle":
                let data = SubmitBundleExecutorWiringTests.bodyData(from: request) ?? Data()
                bodies.withLock {
                    $0.append((try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:])
                }
                timeouts.withLock { $0.append(request.timeoutInterval) }
                switch behaviour {
                case .fail(let code): throw URLError(code)
                case .status(let status):
                    return (Data(#"{"detail":"gateway"}"#.utf8), status)
                case .answerAfter(let seconds, let extra):
                    if seconds > request.timeoutInterval { throw URLError(.timedOut) }
                    var reply: [String: Any] = [
                        "jobId": "stub0001", "experiment": "demo", "verb": "run",
                        "executor": "slurm", "dryRun": false, "runBundle": [:],
                        "command": ["python"], "recordsDirectory": "/stub/records",
                        "submissionDirectory": "/stub",
                    ]
                    for (key, value) in extra { reply[key] = value }
                    return (try JSONSerialization.data(withJSONObject: reply), 200)
                }
            default:
                return (Data(#"{"detail":"no such route"}"#.utf8), 404)
            }
        }
        defer { RepeatURLProtocol.handler = nil }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RepeatURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let stderr = WiringLocked("")
        let sink = ExperimentCLISink(
            out: { _ in }, outRaw: { _ in },
            err: { text in stderr.withLock { $0 += text } })
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "submit-repeat-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let result = await ExperimentCLIRunner.$remoteSessionOverride.withValue(session) {
            await ExperimentCLIRunner(sink: sink).run(
                namespace: "remote",
                ["submit-bundle", "/srv/b.tar.gz", "--url", "http://127.0.0.1:1",
                 "--token", "stub", "--json"] + args)
        }
        return Outcome(
            envelope: result.envelope, exitCode: result.exitCode,
            stderr: stderr.withLock { $0 },
            submitBodies: bodies.withLock { $0 }, submitTimeouts: timeouts.withLock { $0 })
    }

    // MARK: - Waiting long enough

    /// THE issue: a submission the server answers after eight minutes of
    /// measuring. Before, the request carried URLSession's 60 s and timed out.
    @Test func aSlowSubmissionIsWaitedFor() async throws {
        let outcome = try await submit(behaviour: .answerAfter(seconds: 8 * 60))
        #expect(outcome.envelope.exitCode == 0, "\(outcome.envelope.error?.reason ?? "")")
        #expect(outcome.submitTimeouts == [ClusterClient.studySubmissionTimeout])
        #expect(ClusterClient.studySubmissionTimeout == 3600)
        guard case .object(let response)? = outcome.envelope.result?["response"] else {
            Issue.record("no response echoed")
            return
        }
        #expect(response["jobId"] == .string("stub0001"))
    }

    @Test func aDryRunWaitsJustAsLong() async throws {
        let outcome = try await submit(
            ["--verb", "run", "--executor", "slurm", "--dry-run"],
            behaviour: .answerAfter(seconds: 8 * 60))
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.submitTimeouts == [3600])
    }

    // MARK: - Never a failure

    @Test func aTimeoutIsPendingWithTheRepairToLook() async throws {
        let outcome = try await submit(behaviour: .fail(.timedOut))
        #expect(outcome.envelope.state == .pending)
        #expect(outcome.envelope.exitCode == 12)
        #expect(outcome.exitCode == 12)
        #expect(outcome.envelope.error?.code == "submitOutcomeUnknown")
        let repair = outcome.envelope.error?.repairAction ?? ""
        #expect(repair.hasPrefix("steerlab-cli remote jobs --url http://127.0.0.1:1 --json"))
        #expect(repair.contains("Sending the same submission again is safe"))
        #expect(!repair.contains("stub"), "the token never reaches a repair")
        #expect(outcome.envelope.error?.reason.contains("not a failure") == true)
        #expect(outcome.envelope.result?["repeatIsSafe"] == .bool(true))
        #expect(outcome.envelope.result?["retried"] == .bool(false))
        #expect(outcome.submitBodies.count == 1, "nothing is sent again")
        #expect(outcome.stderr.contains("submission outcome unknown"))
    }

    @Test func anOlderServerIsToldToLookBeforeSubmittingAgain() async throws {
        let outcome = try await submit(behaviour: .fail(.timedOut), advertisesKey: false)
        #expect(outcome.envelope.exitCode == 12)
        #expect(outcome.envelope.result?["repeatIsSafe"] == .bool(false))
        #expect(outcome.envelope.error?.repairAction
            .contains("a second submit is a second study") == true)
    }

    @Test func aDroppedConnectionOrAGatewayIsUnknownToo() async throws {
        for behaviour in [SubmitBehaviour.fail(.networkConnectionLost), .status(504), .status(502)] {
            let outcome = try await submit(behaviour: behaviour)
            #expect(outcome.envelope.exitCode == 12)
            #expect(outcome.envelope.error?.code == "submitOutcomeUnknown")
        }
    }

    /// A request that cannot have reached the server stays what it is.
    @Test func aRefusedConnectionIsStillAFailure() async throws {
        let outcome = try await submit(behaviour: .fail(.cannotConnectToHost))
        #expect(outcome.envelope.state == .failed)
        #expect(outcome.envelope.error?.code != "submitOutcomeUnknown")
    }

    // MARK: - The submission key

    @Test func theKeyIsDerivedFromWhatIsSubmitted() async throws {
        let first = try await submit()
        let again = try await submit()
        let parallel = try await submit(["--verb", "run", "--executor", "slurm", "--parallel", "4"])
        guard case .string(let key)? = first.submitBodies.first?["submissionKey"] else {
            Issue.record("no submissionKey sent")
            return
        }
        #expect(key.count == 64)
        #expect(again.submitBodies.first?["submissionKey"] == .string(key))
        #expect(parallel.submitBodies.first?["submissionKey"] != .string(key))
    }

    /// An older server ignores the field, so it is sent to every server: the
    /// capability matters only to the repair after a submission that was
    /// never answered.
    @Test func theKeyIsSentToAnOlderServerToo() async throws {
        let outcome = try await submit(advertisesKey: false)
        #expect(outcome.envelope.exitCode == 0)
        guard case .string(let key)? = outcome.submitBodies.first?["submissionKey"] else {
            Issue.record("no submissionKey sent")
            return
        }
        #expect(key.count == 64)
    }

    /// Independent of the code under test: a hand-written canonical document
    /// (`printf … | shasum -a 256`). The Python client's derivation hashes
    /// the same document to the same digest.
    @Test func theKeyMatchesItsHandWrittenDerivation() throws {
        struct Body: Encodable {
            var bundlePath: String
            var verb: String
            var dryRun: Bool
        }
        let key = try SubmissionKey.derive(
            body: Body(bundlePath: "/x.tar.gz", verb: "run", dryRun: false),
            pathKey: "bundlePath", identityKey: "bundleSha256",
            identity: String(repeating: "f", count: 64))
        #expect(key == "7d123640164d76866a737e457c97322da19275130de077cab5230977acfec7f2")
    }

    @Test func anAlreadySubmittedAnswerSaysSoAndChangesNothing() async throws {
        let note = "Already submitted: this request was submitted before as job "
            + "stub0001, which is running. Nothing was submitted again."
        let outcome = try await submit(behaviour: .answerAfter(
            seconds: 0,
            extra: ["alreadySubmitted": true, "alreadySubmittedNote": note,
                    "currentJobID": "stub0001", "currentStatus": "running"]))
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.envelope.changed == false)
        #expect(outcome.envelope.message
            == "already submitted as job stub0001; nothing was submitted again")
        #expect(outcome.envelope.result?["alreadySubmitted"] == .bool(true))
        #expect(outcome.stderr.contains("note: \(note)"))
    }

    // MARK: - The walltime notice

    @Test func theWalltimeNoticeReachesStderrAndTheEnvelope() async throws {
        let notice = "The throughput history this estimate uses came from items "
            + "of about 1,000 prompt tokens; these average 150,000."
        let preflight: [String: Any] = [
            "verdict": "warn",
            "checks": [["id": "walltime", "status": "warn", "message": "m",
                        "data": ["promptLengthNotice": notice]]],
        ]
        let outcome = try await submit(behaviour: .answerAfter(
            seconds: 0, extra: ["preflight": preflight]))
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.stderr.contains("warning: \(notice)"))
        #expect(outcome.envelope.result?["walltimeNotice"] == .string(notice))
    }
}

private final class RepeatURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) throws -> (Data, Int))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        do {
            let (data, status) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
