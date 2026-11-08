import Foundation
import Testing

@testable import ExperimentKit

// Issue #12 on the Mac client: `remote submit-bundle --parallel` when the
// cluster's job limit is reached, and `remote status` of a split run whose
// parts SteerLab holds. Against a server stubbed below the client; the
// server side has its own suite (`Server/tests/test_held_shards.py`).
@Suite(.serialized) struct HeldShardsCLITests {

    struct Outcome {
        var envelope: SteerLabCLIEnvelope
        var stdout: String
        var stderr: String
    }

    static let heldMessage =
        "2 of 4 parts were submitted now; 2 are held by SteerLab because the "
        + "cluster's job limit is reached. The controller submits each held "
        + "part, in order, when a slot frees; while the controller is not "
        + "running they wait. The run is merged when every part has finished."

    static func submission(shardsHeld: Any) -> [String: Any] {
        [
            "jobId": "parent01", "experiment": "demo", "verb": "run",
            "executor": "slurm", "dryRun": false, "runBundle": [:] as [String: Any],
            "command": ["python"], "recordsDirectory": "/stub/records",
            "submissionDirectory": "/stub",
            "shardJobIDs": ["p0", "p1", "p2", "p3"], "shardsHeld": shardsHeld,
        ]
    }

    func run(_ args: [String], routes: [String: (Int, Any)]) async throws -> Outcome {
        var capabilityBody: [String: Any] = [
            "apiSchemaVersion": "2026-07-01", "engine": "python-hf-transformers",
            "instrumentation": ["policy-evidence-v2", "policy-v1", "probe-readings-v1"],
            "profile": ["profile": "cluster", "executor": "slurm"],
            "cluster": ["jobStatus": true, "jobLogTail": true],
        ]
        capabilityBody["remoteStudy"] = [:] as [String: Any]
        let capabilityData = try JSONSerialization.data(withJSONObject: capabilityBody)
        var routeData: [String: (Int, Data)] = [:]
        for (path, (status, body)) in routes {
            routeData[path] = (status, try JSONSerialization.data(withJSONObject: body))
        }
        let served = routeData
        HeldShardsStubURLProtocol.handler = { request in
            let path = request.url?.path ?? ""
            if path == "/api/capabilities" { return (capabilityData, 200) }
            if path == "/api/bundles/inspect" {
                return (Data(#"{"kind":"runBundle","experiment":"demo"}"#.utf8), 200)
            }
            if let (status, data) = served[path] { return (data, status) }
            return (Data(#"{"detail":"no such route"}"#.utf8), 404)
        }
        defer { HeldShardsStubURLProtocol.handler = nil }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [HeldShardsStubURLProtocol.self]
        let session = URLSession(configuration: configuration)

        let stdout = WiringLocked("")
        let stderr = WiringLocked("")
        let sink = ExperimentCLISink(
            out: { text in stdout.withLock { $0 += text } }, outRaw: { _ in },
            err: { text in stderr.withLock { $0 += text } })
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "held-shards-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let result = await ExperimentCLIRunner.$remoteSessionOverride.withValue(session) {
            await ExperimentCLIRunner(sink: sink).run(
                namespace: "remote",
                args + ["--url", "http://127.0.0.1:1", "--token", "stub"])
        }
        return Outcome(
            envelope: result.envelope, stdout: stdout.withLock { $0 },
            stderr: stderr.withLock { $0 })
    }

    let submit = ["submit-bundle", "/srv/b.tar.gz", "--verb", "run",
                  "--executor", "slurm", "--parallel", "4"]

    // MARK: - remote submit-bundle

    @Test func aSplitRunWithHeldPartsIsASuccessWithANote() async throws {
        let held: [String: Any] = [
            "reason": "clusterJobLimit", "count": 2, "partCount": 4,
            "submittedCount": 2, "jobIDs": ["p2", "p3"], "since": 1_700_000_000.0,
            "schedulerMessage": "sbatch: error: QOSMaxSubmitJobPerUserLimit",
            "message": Self.heldMessage,
        ]
        let outcome = try await run(
            submit,
            routes: ["/api/studies/submit-bundle": (200, Self.submission(shardsHeld: held))])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.envelope.message
            == "submitted /srv/b.tar.gz; 2 of 4 parts held by SteerLab until slots free")
        #expect(outcome.stderr.contains("note: \(Self.heldMessage)\n"))
        guard case .object(let response)? = outcome.envelope.result?["response"],
            case .object(let echoed)? = response["shardsHeld"]
        else {
            Issue.record("the envelope does not carry shardsHeld")
            return
        }
        #expect(echoed["count"] == .number(2))
        #expect(echoed["jobIDs"] == .array([.string("p2"), .string("p3")]))
    }

    @Test func aSplitRunWithEveryPartSubmittedSaysNothingOfHolding() async throws {
        let outcome = try await run(
            submit,
            routes: ["/api/studies/submit-bundle": (200, Self.submission(shardsHeld: NSNull()))])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.envelope.message == "submitted /srv/b.tar.gz")
        #expect(!outcome.stderr.contains("held by SteerLab"))
    }

    /// The server answers a split run it could not start as an error, so
    /// the submit exits nonzero with the server's own sentence.
    @Test func aSplitRunRefusedForAnotherReasonExitsNonzero() async throws {
        let detail =
            "the split run was not started, and its record, job parent01, is "
            + "failed: shard 2 of 4 failed to submit: sbatch: error: Invalid "
            + "partition name specified — shard job(s) p0 was cancelled "
            + "(confirmed). Correct what the scheduler said, then submit the "
            + "study again"
        let outcome = try await run(
            submit, routes: ["/api/studies/submit-bundle": (400, ["detail": detail])])
        #expect(outcome.envelope.exitCode != 0)
        #expect(outcome.envelope.state != .ready)
        #expect(outcome.envelope.error?.reason.contains("the split run was not started") == true)
        #expect(outcome.envelope.error?.reason.contains("Invalid partition name") == true)
    }

    // MARK: - remote status

    @Test func statusShowsHeldPartsInPlainWords() async throws {
        let parent: [String: Any] = [
            "schemaVersion": 1, "jobId": "parent01", "status": "running",
            "executorJobID": NSNull(), "scheduler": NSNull(),
            "notes": ["1 of 2 parts is held by SteerLab: the cluster's job limit is reached."],
            "held": NSNull(), "heldNote": NSNull(),
            "shards": [
                ["jobId": "p0", "status": "running", "executorJobID": "5401",
                 "shardIndex": 0, "shardCount": 2, "found": true,
                 "held": NSNull(), "heldNote": NSNull()] as [String: Any],
                ["jobId": "p1", "status": "pending", "executorJobID": NSNull(),
                 "shardIndex": 1, "shardCount": 2, "found": true,
                 "held": ["reason": "clusterJobLimit", "since": 1_700_000_000.0]
                    as [String: Any],
                 "heldNote": "held by SteerLab: the cluster's job limit is reached; "
                    + "it is submitted when a slot frees"] as [String: Any],
            ] as [Any],
        ]
        let outcome = try await run(
            ["status", "parent01"], routes: ["/api/jobs/parent01/status": (200, parent)])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.stdout.contains("part 2 of 2: job p1 [pending]"))
        #expect(outcome.stdout.contains(
            "    held by SteerLab: the cluster's job limit is reached; it is "
                + "submitted when a slot frees (held since "))
        // Only the held part says so (the run's note says it once more).
        #expect(outcome.stdout.components(
            separatedBy: "it is submitted when a slot frees").count == 2)
        #expect(outcome.stdout.contains("note: 1 of 2 parts is held by SteerLab"))
        #expect(outcome.envelope.message
            == "job parent01 is running; 1 of 2 parts held by SteerLab until slots free")
    }

    @Test func aHeldPartWithoutTheServersSentenceStillReadsPlainly() throws {
        let document: JSONValue = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(#"{"jobId":"p1","status":"pending","held":{"reason":"clusterJobLimit"}}"#.utf8))
        let report = try #require(RemoteJobStatusReport.from(document))
        #expect(RemoteJobStatusText.lines(for: report)
            == ["job p1 [pending]", "  " + RemoteJobStatusText.heldPartNote])
        #expect(RemoteJobStatusText.message(for: report)
            == "job p1 is pending (held by SteerLab until the cluster's job limit frees a slot)")
    }

    // MARK: - remote resubmit (review round 6)

    static func resumed(heldUnexplained: Bool) -> [String: Any] {
        var answer: [String: Any] = [
            "ok": true, "resubmitOf": "parent01", "manualResubmit": true,
            "resumedAfterCancel": true,
            "resumedShards": [
                ["ok": true, "jobId": "p2b", "resubmitOf": "p2", "held": true]
                    as [String: Any],
            ] as [Any],
            "message": "Run parent01 was cancelled; 1 of its parts now continues.",
        ]
        if heldUnexplained {
            answer["heldUnexplained"] = true
            answer["schedulerMessage"] =
                "sbatch: error: Batch job submission failed: Job violates "
                + "accounting/QOS policy"
            answer["message"] =
                "Run parent01 was resumed, but its 1 part is held, not "
                + "running: the scheduler refused the first without saying why."
        }
        return answer
    }

    @Test func aResumeHeldWithoutAReasonIsPendingWithTheRepair() async throws {
        let outcome = try await run(
            ["resubmit", "parent01", "--walltime", "02:00:00", "--json"],
            routes: [
                "/api/jobs/parent01": (200, ["id": "parent01", "status": "cancelled"]),
                "/api/jobs/parent01/resubmit": (200, Self.resumed(heldUnexplained: true)),
            ])
        #expect(outcome.envelope.exitCode == 12)
        #expect(outcome.envelope.state == .pending)
        #expect(outcome.envelope.error == nil)
        #expect(outcome.envelope.message.contains("held, not running"))
        #expect(!outcome.envelope.message.contains("now continue"))
        #expect(outcome.envelope.nextAction?.verb
            == "steerlab-cli remote status parent01 --url http://127.0.0.1:1 --json")
        #expect(outcome.envelope.nextAction?.detail
            == "if its parts stay held, check the walltime against the cluster's "
            + "limit for one job")
        guard case .object(let response)? = outcome.envelope.result?["response"] else {
            Issue.record("the envelope does not carry the server's answer")
            return
        }
        #expect(response["heldUnexplained"] == .bool(true))
        #expect(outcome.stderr.contains("note: Run parent01 was resumed"))
    }

    @Test func aResumeWhosePartsContinueIsReady() async throws {
        let outcome = try await run(
            ["resubmit", "parent01", "--json"],
            routes: [
                "/api/jobs/parent01": (200, ["id": "parent01", "status": "cancelled"]),
                "/api/jobs/parent01/resubmit": (200, Self.resumed(heldUnexplained: false)),
            ])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.envelope.state == .ready)
    }

    // MARK: - the rule

    @Test func heldPartsIsNilWhenNothingIsHeld() {
        #expect(ShardedSubmission.heldParts(nil) == nil)
        #expect(ShardedSubmission.heldParts(HeldShards(count: 0, partCount: 4)) == nil)
        let fallback = ShardedSubmission.heldParts(HeldShards(count: 1, partCount: 3))
        #expect(fallback?.phrase == "1 of 3 parts held by SteerLab until slots free")
        #expect(fallback?.note.hasPrefix("1 of 3 parts held by SteerLab") == true)
    }
}

private final class HeldShardsStubURLProtocol: URLProtocol, @unchecked Sendable {
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
