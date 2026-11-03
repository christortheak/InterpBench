import Foundation
import Testing

@testable import ExperimentKit

// `steerlab-cli remote status` and `remote logs --tail` (issue #7), against a
// server stubbed below the client: what goes on the wire, what a person
// reads, and what an agent gets in the envelope. The server side has its own
// suite (`Server/tests/test_job_status.py`).
@Suite(.serialized) struct RemoteJobStatusCLITests {

    struct Outcome {
        var envelope: SteerLabCLIEnvelope
        var stdout: String
        var paths: [String]
        var queries: [String]
    }

    /// A pending job's status document, as the engine writes it.
    static var pendingStatus: [String: Any] { [
        "schemaVersion": 1, "jobId": "a1b2c3d4e5f6", "kind": "study-submit",
        "status": "submitted", "executor": "slurm", "executorJobID": "5101",
        "createdAt": 1_700_000_000.0, "startedAt": NSNull(),
        "scheduler": [
            "queryOk": true, "rawState": "PENDING", "state": "submitted",
            "reason": "Priority", "expectedStartAt": 1_700_003_600.0,
            "timeLimit": "08:00:00",
        ] as [String: Any],
        "schedulerOutcome": NSNull(), "deathDetail": NSNull(),
        "progress": [
            "unit": "response", "recordsWritten": 3, "recordsExpected": 50,
            "recordsExpectedSource": "preflightEstimate",
        ] as [String: Any],
        "notes": [String](),
        "aFieldThisClientDoesNotKnow": "kept",
    ] }

    func run(
        _ args: [String], capabilities: [String: Any]? = ["jobStatus": true, "jobLogTail": true],
        routes: [String: (Int, Any)] = [:]
    ) async throws -> Outcome {
        let paths = WiringLocked<[String]>([])
        let queries = WiringLocked<[String]>([])
        // Serialized up front, so the stub captures only Sendable bytes.
        var capabilityBody: [String: Any] = [
            "apiSchemaVersion": "2026-07-01", "engine": "python-hf-transformers",
        ]
        if let capabilities { capabilityBody["cluster"] = capabilities }
        let capabilityData = try JSONSerialization.data(withJSONObject: capabilityBody)
        var routeData: [String: (Int, Data)] = [:]
        for (path, (status, body)) in routes {
            routeData[path] = (status, try JSONSerialization.data(withJSONObject: body))
        }
        let served = routeData
        StatusStubURLProtocol.handler = { request in
            let path = request.url?.path ?? ""
            paths.withLock { $0.append(path) }
            queries.withLock { $0.append(request.url?.query ?? "") }
            if path == "/api/capabilities" { return (capabilityData, 200) }
            if let (status, data) = served[path] { return (data, status) }
            return (Data(#"{"detail":"no such job"}"#.utf8), 404)
        }
        defer { StatusStubURLProtocol.handler = nil }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StatusStubURLProtocol.self]
        let session = URLSession(configuration: configuration)

        let stdout = WiringLocked("")
        let sink = ExperimentCLISink(
            out: { text in stdout.withLock { $0 += text } }, outRaw: { _ in }, err: { _ in })
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "remote-status-\(UUID().uuidString)")
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
            paths: paths.withLock { $0 }, queries: queries.withLock { $0 })
    }

    // MARK: - remote status

    @Test func statusSaysWhyAJobIsWaitingAndHowManyRecordsItWrote() async throws {
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, Self.pendingStatus)])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.stdout.contains("job a1b2c3d4e5f6 [submitted] — scheduler job 5101"))
        #expect(outcome.stdout.contains("scheduler: PENDING (Priority); expected to start"))
        #expect(outcome.stdout.contains(
            "records: 3 of 50 records written (the total is the submission's planning estimate)"))
        #expect(outcome.envelope.message == "job a1b2c3d4e5f6 is submitted; 3 of 50 records "
            + "written (the total is the submission's planning estimate)")
    }

    /// Seen live 2026-11-03: the scheduler said RUNNING one controller check
    /// before SteerLab's record did, and the message said only `submitted`.
    @Test func aRecordBehindTheSchedulerSaysWhatTheSchedulerReports() async throws {
        var started = Self.pendingStatus
        started["scheduler"] = [
            "queryOk": true, "rawState": "RUNNING", "state": "running",
            "nodeList": "n7", "startedAt": 1_700_000_500.0, "elapsed": "00:00:58",
            "timeLimit": "01:30:00",
        ] as [String: Any]
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, started)])
        #expect(outcome.envelope.message == "job a1b2c3d4e5f6 is submitted (the scheduler "
            + "already reports RUNNING; SteerLab's record has not caught up yet); 3 of 50 "
            + "records written (the total is the submission's planning estimate)")
        #expect(outcome.stdout.contains(
            "  SteerLab's record still says submitted; the controller updates it when its "
                + "own check of the scheduler sees the change."))

        // The scheduler's state names an outcome it decided once, not twice.
        var timedOut = started
        timedOut["status"] = "running"
        timedOut["schedulerOutcome"] = "TIMEOUT"
        timedOut["scheduler"] = [
            "queryOk": true, "rawState": "TIMEOUT", "state": "failed",
        ] as [String: Any]
        let ended = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, timedOut)])
        #expect(ended.envelope.message.hasPrefix("job a1b2c3d4e5f6 is running (the scheduler "
            + "already reports TIMEOUT; SteerLab's record has not caught up yet); "))
        #expect(!ended.envelope.message.contains("(the scheduler recorded"))

        func ahead(_ status: String, _ view: [String: Any]) -> String? {
            var document = Self.pendingStatus
            document["status"] = status
            document["scheduler"] = view
            let data = try? JSONSerialization.data(withJSONObject: document)
            let value = data.flatMap { try? JSONDecoder().decode(JSONValue.self, from: $0) }
            return value.flatMap(RemoteJobStatusReport.from).flatMap {
                RemoteJobStatusText.schedulerAhead(of: $0)
            }
        }
        #expect(ahead("running", ["rawState": "COMPLETED", "state": "succeeded"]) == "COMPLETED")
        // Queued again: the controller's next check turns `running` back
        // into `submitted`, so the record is behind here too.
        #expect(ahead("running", ["rawState": "PREEMPTED", "state": "submitted"]) == "PREEMPTED")
        // A checkpoint exit is FAILED to the scheduler but parks a run to resume.
        #expect(ahead("running", ["rawState": "FAILED", "state": "checkpointed"])
            == "FAILED with the checkpoint exit, which parks the run to resume")
        // An answer from the queue alone, after accounting timed out, is live.
        #expect(ahead("submitted", [
            "queryOk": true, "timedOut": true, "rawState": "RUNNING", "state": "running",
        ]) == "RUNNING")

        // Agreement, an ended record, a scheduler that did not answer, and a
        // field of an unexpected type add nothing.
        for (status, view) in [
            ("running", ["queryOk": true, "rawState": "RUNNING", "state": "running"]),
            ("failed", ["queryOk": true, "rawState": "COMPLETED", "state": "succeeded"]),
            ("checkpointed", ["queryOk": true, "rawState": "FAILED", "state": "checkpointed"]),
            ("submitted", ["queryOk": false, "rawState": "RUNNING", "state": "running"]),
            ("submitted", ["queryOk": true, "rawState": "", "state": "running"]),
            ("submitted", ["queryOk": true, "rawState": "RUNNING", "state": ["running"]]),
        ] as [(String, [String: Any])] {
            #expect(ahead(status, view) == nil, "\(status) \(view)")
            var document = Self.pendingStatus
            document["status"] = status
            document["scheduler"] = view
            let quiet = try await run(
                ["status", "a1b2c3d4e5f6"],
                routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, document)])
            #expect(!quiet.envelope.message.contains("already reports"), "\(status)")
            #expect(!quiet.stdout.contains("SteerLab's record still says"), "\(status)")
        }
    }

    @Test func theEnvelopeCarriesTheServersDocumentAsItArrived() async throws {
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6", "--json"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, Self.pendingStatus)])
        // JSON mode: one envelope, no human lines.
        #expect(outcome.stdout.isEmpty)
        guard case .object(let response)? = outcome.envelope.result?["response"] else {
            Issue.record("no response in the envelope")
            return
        }
        #expect(response["aFieldThisClientDoesNotKnow"] == .string("kept"))
        guard case .object(let scheduler)? = response["scheduler"] else {
            Issue.record("no scheduler view")
            return
        }
        #expect(scheduler["reason"] == .string("Priority"))
    }

    @Test func aTimedOutJobSaysWhatTheSchedulerRecorded() async throws {
        var ended = Self.pendingStatus
        ended["status"] = "failed"
        ended["schedulerOutcome"] = "TIMEOUT"
        ended["schedulerOutcomeNote"] =
            "The scheduler recorded TIMEOUT: the job reached its time limit."
        ended["deathDetail"] =
            "Slurm reported TIMEOUT after 08:00:00 (walltime limit 08:00:00, exit 0:1)"
        ended["scheduler"] = [
            "queryOk": true, "rawState": "TIMEOUT", "state": "failed",
            "startedAt": 1_700_000_000.0, "endedAt": 1_700_028_800.0,
            "elapsed": "08:00:00", "timeLimit": "08:00:00", "exitCode": "0:1",
            "schedulerOutcome": "TIMEOUT",
        ] as [String: Any]
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, ended)])
        #expect(outcome.stdout.contains(
            "  The scheduler recorded TIMEOUT: the job reached its time limit."))
        #expect(outcome.stdout.contains(
            "Slurm reported TIMEOUT after 08:00:00 (walltime limit 08:00:00, exit 0:1)"))
        #expect(outcome.envelope.message.contains("(the scheduler recorded TIMEOUT)"))
        #expect(!outcome.stdout.contains("ended by the scheduler"))
    }

    /// One field of an unexpected type costs that field, never the summary
    /// (review 2026-11-02: a whole-document decode dropped every line).
    @Test func anUnexpectedFieldShapeCostsOnlyThatField() async throws {
        var odd = Self.pendingStatus
        odd["kind"] = 7
        odd["notes"] = "not a list"
        odd["progress"] = [
            "unit": "response", "recordsWritten": "3", "recordsExpected": 50,
        ] as [String: Any]
        odd["scheduler"] = [
            "queryOk": true, "rawState": "PENDING", "reason": "Priority",
            "elapsed": 0, "expectedStartAt": "soon",
        ] as [String: Any]
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, odd)])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.stdout.contains("job a1b2c3d4e5f6 [submitted] — scheduler job 5101"))
        #expect(outcome.stdout.contains("scheduler: PENDING (Priority)"))
        #expect(!outcome.stdout.contains("expected to start"))
        #expect(outcome.envelope.message == "job a1b2c3d4e5f6 is submitted")
    }

    @Test func aSplitRunShowsEachPart() async throws {
        var parent = Self.pendingStatus
        parent["status"] = "running"
        parent["executorJobID"] = NSNull()
        parent["scheduler"] = NSNull()
        parent["progress"] = ["recordsWritten": 27, "recordsExpected": 50,
                              "recordsExpectedSource": "shardStamp"] as [String: Any]
        parent["shards"] = [
            ["jobId": "part0", "status": "running", "executorJobID": "5401",
             "shardIndex": 0, "shardCount": 2, "found": true,
             "progress": ["recordsWritten": 20, "recordsExpected": 25,
                          "recordsExpectedSource": "shardStamp"] as [String: Any]]
                as [String: Any],
            ["jobId": "part1b", "status": "running", "executorJobID": "5403",
             "shardIndex": 1, "shardCount": 2, "found": true, "shardJob": "part1",
             "progress": ["recordsWritten": 7, "recordsExpected": 25,
                          "recordsExpectedSource": "shardStamp"] as [String: Any]]
                as [String: Any],
        ]
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, parent)])
        #expect(outcome.stdout.contains("records: 27 of 50 records written"))
        #expect(outcome.stdout.contains("part 1 of 2: job part0 [running] — scheduler job 5401"))
        #expect(outcome.stdout.contains(
            "part 2 of 2: job part1b [running] — scheduler job 5403 (continues job part1)"))
        #expect(outcome.stdout.contains("    records: 7 of 25 records written"))
    }

    @Test func anOlderServerIsRefusedWithAnUpdateRepairAndNeverAsked() async throws {
        let outcome = try await run(
            ["status", "a1b2c3d4e5f6"], capabilities: ["manualResubmit": true],
            routes: ["/api/jobs/a1b2c3d4e5f6/status": (200, Self.pendingStatus)])
        #expect(outcome.envelope.state == .refused)
        #expect(outcome.envelope.error?.code == "jobStatusUnsupported")
        #expect(outcome.envelope.error?.repairAction.contains("cluster push") == true)
        #expect(!outcome.paths.contains("/api/jobs/a1b2c3d4e5f6/status"))
    }

    @Test func anUnknownJobIsNotFound() async throws {
        let outcome = try await run(["status", "nosuchjob123"])
        #expect(outcome.envelope.state == .notFound)
        #expect(outcome.envelope.exitCode == 66)
        #expect(outcome.envelope.error?.repairAction.contains("remote jobs") == true)
    }

    @Test func aPathShapedIDNeverReachesTheServer() async throws {
        for bad in ["../capabilities", "a/b", "..", "5101 --all", ""] {
            let outcome = try await run(["status", bad])
            #expect(outcome.envelope.exitCode == 64, "\(bad)")
            #expect(!outcome.paths.contains { $0.hasPrefix("/api/jobs") }, "\(bad)")
        }
    }

    // MARK: - remote logs --tail

    static var tail: [String: Any] { [
        "schemaVersion": 1, "jobId": "a1b2c3d4e5f6", "status": "running",
        "lines": ["slurm 5101 → running", "fold pending"], "lineCount": 2,
        "truncated": true,
        "schedulerOutput": [
            "available": true,
            "stdout": ["name": "slurm-5101.out", "exists": true,
                       "lines": ["record 8", "record 9"], "truncated": true] as [String: Any],
            "stderr": ["name": "slurm-5101.err", "exists": false, "lines": [String](),
                       "detail": "slurm-5101.err does not exist (the job may not have "
                           + "started yet, or the file was removed)"] as [String: Any],
        ] as [String: Any],
        "notes": [String](),
    ] }

    @Test func aTailPrintsTheLastLinesAndTheJobsOwnOutputThenReturns() async throws {
        let outcome = try await run(
            ["logs", "a1b2c3d4e5f6", "--tail", "2"],
            routes: ["/api/jobs/a1b2c3d4e5f6/log-tail": (200, Self.tail)])
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.queries.contains("lines=2"))
        // The streaming route is never opened.
        #expect(!outcome.paths.contains("/api/jobs/a1b2c3d4e5f6/stream"))
        let printed = outcome.stdout.split(separator: "\n").map(String.init)
        #expect(printed == [
            "(earlier controller lines not shown)",
            "slurm 5101 → running", "fold pending",
            "--- slurm-5101.out: last 2 line(s), earlier lines not shown ---",
            "record 8", "record 9",
            "--- slurm-5101.err: slurm-5101.err does not exist (the job may not have "
                + "started yet, or the file was removed) ---",
        ])
        #expect(outcome.envelope.result?["tailLines"] == .number(2))
    }

    @Test func aTailOutOfRangeIsMalformedBeforeAnyRequest() async throws {
        for bad in ["0", "1001", "ten"] {
            let outcome = try await run(["logs", "a1b2c3d4e5f6", "--tail", bad])
            #expect(outcome.envelope.exitCode == 64, "\(bad)")
            #expect(outcome.paths.isEmpty, "\(bad)")
        }
    }

    @Test func aTailOnAnOlderServerIsRefused() async throws {
        let outcome = try await run(
            ["logs", "a1b2c3d4e5f6", "--tail", "5"], capabilities: ["jobStatus": true],
            routes: ["/api/jobs/a1b2c3d4e5f6/log-tail": (200, Self.tail)])
        #expect(outcome.envelope.state == .refused)
        #expect(!outcome.paths.contains("/api/jobs/a1b2c3d4e5f6/log-tail"))
    }

    // MARK: - the id rule

    @Test func jobIDsAreLettersDigitsAndThreeMarks() {
        #expect(RemoteJobStatusText.isJobID("a1b2c3d4e5f6"))
        #expect(RemoteJobStatusText.isJobID("study_run-2.part"))
        for bad in ["", ".", "..", "a/b", "a b", "a?b", "a#b", "é", String(repeating: "a", count: 129)] {
            #expect(!RemoteJobStatusText.isJobID(bad), "\(bad)")
        }
    }
}

private final class StatusStubURLProtocol: URLProtocol, @unchecked Sendable {
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
