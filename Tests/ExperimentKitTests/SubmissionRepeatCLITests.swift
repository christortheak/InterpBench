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
        /// A success answer this client cannot decode: the job may well have
        /// landed.
        case undecodable
    }

    /// The server's inspection of a staged bundle: a study's files by path
    /// and digest, packaged at `createdAt` into an archive `archive`.
    static func inspection(createdAt: Double, archive: String, promptsDigest: String = "2") -> String {
        let one = String(repeating: "1", count: 64)
        let two = String(repeating: promptsDigest, count: 64)
        return #"{"kind":"runBundle","experiment":"demo","experimentContentHash":"cafe","createdAt":"#
            + "\(createdAt)" + #","bundleSha256":""# + archive + #"","entries":["#
            + #"{"path":"prompts/tasks/t.jsonl","sha256":""# + two + #"","bytes":9},"#
            + #"{"path":"experiments/demo/experiment.json","sha256":""# + one + #"","bytes":5}]}"#
    }

    func submit(
        _ args: [String] = ["--verb", "run", "--executor", "slurm"],
        behaviour: SubmitBehaviour = .answerAfter(seconds: 0),
        advertisesKey: Bool = true,
        inspected: String? = nil
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
                if let inspected { return (Data(inspected.utf8), 200) }
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
                case .undecodable:
                    return (Data(#"{"jobId":5,"experiment":"demo"}"#.utf8), 200)
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
        #expect(repair.contains("run the same command again with the same flags"))
        #expect(repair.contains("A command with any flag changed is a new submission"))
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

    /// A success answer this client could not read: the job may have landed,
    /// so it is unknown (12), never a failure (70).
    @Test func anUndecodableSuccessAnswerIsUnknownToo() async throws {
        let outcome = try await submit(behaviour: .undecodable)
        #expect(outcome.envelope.exitCode == 12)
        #expect(outcome.envelope.error?.code == "submitOutcomeUnknown")
        #expect(outcome.envelope.error?.reason.contains("could not be read") == true)
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
            pathKey: "bundlePath", identityKey: "bundleContent",
            identity: String(repeating: "f", count: 64))
        #expect(key == "e5a4afb57cef35974fb494d3a5ababbeb528e81a8082c0f0a682a27b5cc195e1")
    }

    /// What a bundle contains, by hand (`printf … | shasum -a 256`): the
    /// files in path order, never the packaging time or the archive digest.
    /// The Python client is pinned to the same digest.
    @Test func theContentDigestMatchesItsHandWrittenDerivation() throws {
        let first = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(Self.inspection(createdAt: 1, archive: String(repeating: "f", count: 64)).utf8))
        let repackaged = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(Self.inspection(createdAt: 2, archive: String(repeating: "e", count: 64)).utf8))
        let expected = "06d989f81b0723e471b1241772721085cbf85a14a628ae82fcbba5393b78cb1f"
        #expect(SubmissionKey.bundleContentDigest(first) == expected)
        #expect(SubmissionKey.bundleContentDigest(repackaged) == expected)
        #expect(SubmissionKey.bundleContentDigest(.object(["experiment": .string("demo")])) == nil)
    }

    /// THE review's probe on this client: the same command, `--parallel 4`,
    /// against a study packaged twice (a new `createdAt`, a new archive
    /// digest, the same files) carries one key; a different study does not.
    @Test func aRepackagedStudyCarriesTheSameKey() async throws {
        let args = ["--verb", "run", "--executor", "slurm", "--parallel", "4"]
        let first = try await submit(
            args, inspected: Self.inspection(createdAt: 1, archive: String(repeating: "a", count: 64)))
        let again = try await submit(
            args, inspected: Self.inspection(createdAt: 2, archive: String(repeating: "b", count: 64)))
        let other = try await submit(
            args, inspected: Self.inspection(
                createdAt: 3, archive: String(repeating: "c", count: 64), promptsDigest: "3"))
        let fewer = try await submit(
            ["--verb", "run", "--executor", "slurm", "--parallel", "2"],
            inspected: Self.inspection(createdAt: 4, archive: String(repeating: "d", count: 64)))
        guard case .string(let key)? = first.submitBodies.first?["submissionKey"] else {
            Issue.record("no submissionKey sent")
            return
        }
        #expect(first.submitBodies.first?["parallelJobs"] == .number(4))
        #expect(again.submitBodies.first?["submissionKey"] == .string(key))
        #expect(other.submitBodies.first?["submissionKey"] != .string(key))
        #expect(fewer.submitBodies.first?["submissionKey"] != .string(key))
    }

    /// The app repackages on every submit: two packagings of one unchanged
    /// study differ in their time and archive, and agree in their content.
    @Test func theAppsTwoPackagingsHaveOneContentDigest() throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "repack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        WorkspaceRoot.programmaticOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let manifest = try ExperimentStore.create(
            name: "repack", description: "", modelID: "test/model", modelRevision: "abc123")
        func metadata(_ bundle: URL) throws -> (JSONValue, Data) {
            let destination = temp.appending(component: "x-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            let process = Process()
            process.executableURL = URL(filePath: "/usr/bin/tar")
            process.arguments = ["-xzf", bundle.path, "-C", destination.path]
            try process.run()
            process.waitUntilExit()
            let data = try Data(contentsOf: destination.appending(component: "steerlab-bundle.json"))
            return (try JSONDecoder().decode(JSONValue.self, from: data), try Data(contentsOf: bundle))
        }
        let (first, firstBytes) = try metadata(try RunBundlePackager.packageExperiment(manifest))
        let (second, secondBytes) = try metadata(try RunBundlePackager.packageExperiment(manifest))
        #expect(firstBytes != secondBytes, "each packaging stamps its own time")
        let digest = try #require(SubmissionKey.bundleContentDigest(first))
        #expect(SubmissionKey.bundleContentDigest(second) == digest)
    }

    @Test func theAppSaysTheOutcomeIsUnknownInTheCommandLinesWords() {
        let safe = SubmissionOutcomeUnknown(
            endpoint: "http://server.test", cause: "the wait for an answer ran out",
            repeatIsSafe: true, waitedSeconds: 3600)
        #expect(safe.appSentence.contains("The outcome is unknown"))
        #expect(safe.appSentence.contains("Check the server's jobs"))
        #expect(safe.appSentence.contains("recognises a repeat of the same study"))
        #expect(!safe.appSentence.contains("nothing is running"))
        let older = SubmissionOutcomeUnknown(
            endpoint: "http://server.test", cause: "the connection dropped",
            repeatIsSafe: false, waitedSeconds: 3600)
        #expect(older.appSentence.contains("would run the study twice"))
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

    /// The server-resident submission (the app's Server Jobs panel) waits
    /// as long, and carries a key derived from the server's own manifest.
    @Test func theServerResidentSubmissionWaitsAndCarriesAKey() async throws {
        let seen = WiringLocked<[(TimeInterval, [String: JSONValue])]>([])
        RepeatURLProtocol.handler = { request in
            switch request.url?.path ?? "" {
            case "/api/experiment/demo/manifest":
                return (Data(#"{"name":"demo"}"#.utf8), 200)
            case "/api/studies/submit":
                let data = SubmitBundleExecutorWiringTests.bodyData(from: request) ?? Data()
                let body = (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
                seen.withLock { $0.append((request.timeoutInterval, body)) }
                return (Data(#"{"jobId":"id","experiment":"demo","verb":"verify","executor":"local","dryRun":true,"runBundle":{},"command":[],"recordsDirectory":"/r","submissionDirectory":"/s"}"#.utf8), 200)
            default:
                return (Data(#"{"detail":"no such route"}"#.utf8), 404)
            }
        }
        defer { RepeatURLProtocol.handler = nil }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RepeatURLProtocol.self]
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: URLSession(configuration: configuration))
        _ = try await client.submitStudy(
            experiment: "demo", verb: "verify", executor: "local", dryRun: true)
        let sent = seen.withLock { $0 }
        #expect(sent.count == 1)
        #expect(sent.first?.0 == 3600)
        // By hand (printf … | shasum -a 256): the body without its key, plus
        // the digest of the manifest bytes the server returned
        // (d7d234f7…0602 for {"name":"demo"}) and the derivation's version.
        #expect(sent.first?.1["submissionKey"] == .string(
            "6cf379cc132b7b9b48b48f6f5f0474dacb5edea4c144ea76134089d6b04fc4e4"))
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
