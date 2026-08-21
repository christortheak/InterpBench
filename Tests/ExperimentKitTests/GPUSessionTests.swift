import Foundation
import Testing

@testable import ExperimentKit

// MARK: - Wire contract (record decode + capability gate)

@Suite struct GPUSessionRecordTests {
    @Test func fullRecordDecodesEveryContractKey() throws {
        let json = """
            {
              "sessionGeneration": "0e8f8f2a-1111-2222-3333-444455556666",
              "slurmJobID": "3141592",
              "node": "ra3-1",
              "port": 8081,
              "state": "idle",
              "workspaceRoot": "/scratch/me/workspace",
              "gpuType": "A100",
              "gres": "gpu:A100:1",
              "partition": "gpu_p",
              "walltime": "02:00:00",
              "idleMinutes": 30,
              "startedAt": "2026-07-16T09:00:00Z",
              "expiresAt": "2026-07-16T11:00:00Z",
              "serverVersion": "0.9.0",
              "role": "gpu-session",
              "idleRemainingSeconds": 1080,
              "busy": false
            }
            """
        let record = try JSONDecoder().decode(GPUSessionRecord.self, from: Data(json.utf8))
        #expect(record.sessionGeneration == "0e8f8f2a-1111-2222-3333-444455556666")
        #expect(record.slurmJobID == "3141592")
        #expect(record.node == "ra3-1")
        #expect(record.port == 8081)
        #expect(record.state == "idle")
        #expect(record.workspaceRoot == "/scratch/me/workspace")
        #expect(record.gpuType == "A100")
        #expect(record.gres == "gpu:A100:1")
        #expect(record.partition == "gpu_p")
        #expect(record.walltime == "02:00:00")
        #expect(record.idleMinutes == 30)
        #expect(record.startedAt == "2026-07-16T09:00:00Z")
        #expect(record.expiresAt == "2026-07-16T11:00:00Z")
        #expect(record.serverVersion == "0.9.0")
        #expect(record.role == "gpu-session")
        #expect(record.idleRemainingSeconds == 1080)
        #expect(record.busy == false)
        #expect(!record.isTerminal)
    }

    @Test func minimalRecordDecodesLeniently() throws {
        // Everything except sessionGeneration/state is optional on the wire.
        let json = #"{"sessionGeneration": "g-1", "state": "queued"}"#
        let record = try JSONDecoder().decode(GPUSessionRecord.self, from: Data(json.utf8))
        #expect(record.sessionGeneration == "g-1")
        #expect(record.state == "queued")
        #expect(record.slurmJobID == nil)
        #expect(record.port == nil)
        #expect(record.idleRemainingSeconds == nil)
        #expect(record.busy == nil)
    }

    @Test func recordWithoutSessionGenerationFailsToDecode() {
        let json = #"{"state": "ready"}"#
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(GPUSessionRecord.self, from: Data(json.utf8))
        }
    }

    @Test func recordMinedFrom409ConflictBody() throws {
        // Double-start: the refusal body carries the existing record next to
        // the FastAPI detail — the client adopts it instead of dead-ending.
        let body = """
            {"detail": "a GPU session is already running",
             "session": {"sessionGeneration": "g-2", "state": "ready", "node": "ra3-7"}}
            """
        let record = try #require(GPUSessionRecord.record(fromResponseBody: body))
        #expect(record.sessionGeneration == "g-2")
        #expect(record.state == "ready")
        #expect(record.node == "ra3-7")
        #expect(GPUSessionRecord.record(fromResponseBody: #"{"detail": "nope"}"#) == nil)
    }

    @Test func terminalStates() {
        #expect(GPUSessionRecord(sessionGeneration: "g", state: "ended").isTerminal)
        #expect(GPUSessionRecord(sessionGeneration: "g", state: "failed").isTerminal)
        #expect(!GPUSessionRecord(sessionGeneration: "g", state: "ending").isTerminal)
        #expect(!GPUSessionRecord(sessionGeneration: "g", state: "ready").isTerminal)
    }

    @Test func remainingWalltimeFormatsFromExpiresAt() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let expires = HousekeepingDates.format(now.addingTimeInterval(6120))  // 1h42m
        let record = GPUSessionRecord(
            sessionGeneration: "g", state: "ready", expiresAt: expires)
        #expect(record.remainingWalltimeDescription(now: now) == "1h 42m walltime left")
        let past = GPUSessionRecord(
            sessionGeneration: "g", state: "ready",
            expiresAt: HousekeepingDates.format(now.addingTimeInterval(-60)))
        #expect(past.remainingWalltimeDescription(now: now) == nil)
        let unknown = GPUSessionRecord(sessionGeneration: "g", state: "ready")
        #expect(unknown.remainingWalltimeDescription(now: now) == nil)
    }
}

@Suite struct GPUSessionCapabilityTests {
    @Test func capabilityGateReadsChatBlockAndServerRole() throws {
        let json = """
            {"serverVersion": "1.0", "serverRole": "controller",
             "chat": {"gpuSession": true, "seededTurns": true}}
            """
        let caps = try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
        #expect(caps.supportsGPUSession)
        #expect(caps.serverRole == "controller")
    }

    @Test func capabilityAbsentMeansUnsupported() throws {
        // Older servers: the whole session surface must stay hidden.
        let older = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion": "0.8"}"#.utf8))
        #expect(!older.supportsGPUSession)
        #expect(older.serverRole == nil)
        // A chat block WITHOUT the flag is also unsupported — never inferred.
        let chatOnly = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"chat": {"seededTurns": true}}"#.utf8))
        #expect(!chatOnly.supportsGPUSession)
    }

    @Test func capabilityGateHonorsDottedFeaturesSpelling() throws {
        let json = #"{"features": {"chat.gpuSession": true}}"#
        let caps = try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
        #expect(caps.supportsGPUSession)
    }

    @Test func streamingLoadGateReadsChatLoadStream() throws {
        // POST /api/load/stream (2026-07-17): only advertised servers get the
        // SSE load path — older servers keep the sync POST, never a 404.
        let caps = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"chat": {"loadStream": true}}"#.utf8))
        #expect(caps.supportsStreamingLoad)
        let older = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion": "0.8"}"#.utf8))
        #expect(!older.supportsStreamingLoad)
    }
}

// MARK: - Client routes

@Suite(.serialized) struct GPUSessionClientRouteTests {
    @Test func startPostsRequestAndDecodesEnvelope() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/session/start")
            #expect(request.httpMethod == "POST")
            let body = try #require(Self.bodyData(from: request))
            let object = try #require(
                JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(object["gres"] as? String == "gpu:A100:1")
            #expect(object["partition"] as? String == "gpu_p")
            #expect(object["walltime"] as? String == "02:00:00")
            #expect(object["idleMinutes"] as? Int == 30)
            // Nil fields stay off the wire.
            #expect(object["memory"] == nil)
            #expect(object["cpus"] == nil)
            return (
                Data(#"{"session": {"sessionGeneration": "g-1", "state": "queued"}}"#.utf8),
                200
            )
        }
        let record = try await client.startGPUSession(
            GPUSessionStartRequest(
                gres: "gpu:A100:1", partition: "gpu_p", walltime: "02:00:00",
                idleMinutes: 30))
        #expect(record.sessionGeneration == "g-1")
        #expect(record.state == "queued")
    }

    @Test func statusDecodesNullSessionAsNil() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/session")
            #expect(request.httpMethod == "GET")
            return (Data(#"{"session": null}"#.utf8), 200)
        }
        #expect(try await client.gpuSessionStatus() == nil)
    }

    @Test func stopSendsDELETEAndReturnsRecord() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/session")
            #expect(request.httpMethod == "DELETE")
            return (
                Data(
                    #"{"ok": true, "session": {"sessionGeneration": "g-1", "state": "ending"}}"#
                        .utf8),
                200
            )
        }
        let record = try await client.stopGPUSession()
        #expect(record?.state == "ending")
    }

    @Test func keepalive409SurfacesAsClientError() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/session/keepalive")
            return (Data(#"{"detail": "no GPU session"}"#.utf8), 409)
        }
        await #expect(throws: ClusterClient.ClientError.self) {
            try await client.gpuSessionKeepalive()
        }
    }

    private static func client(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> ClusterClient {
        MockGPUSessionURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockGPUSessionURLProtocol.self]
        return ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: URLSession(configuration: configuration))
    }

    private static func bodyData(from request: URLRequest) -> Data? {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let count = stream.read(buffer, maxLength: bufferSize)
            if count > 0 { data.append(buffer, count: count) } else { break }
        }
        return data
    }
}

private final class MockGPUSessionURLProtocol: URLProtocol, @unchecked Sendable {
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
                url: request.url ?? URL(string: "http://server.test")!,
                statusCode: status, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

// MARK: - Controller state machine

@MainActor
@Suite struct GPUSessionControllerTests {

    private func makeController(
        capability: Bool = true,
        status: @escaping @Sendable () async throws -> GPUSessionRecord? = { nil },
        start: @escaping @Sendable (GPUSessionStartRequest) async throws -> GPUSessionRecord = {
            _ in GPUSessionRecord(sessionGeneration: "g", state: "queued")
        },
        stop: @escaping @Sendable () async throws -> GPUSessionRecord? = { nil },
        keepalive: @escaping @Sendable () async throws -> Void = {}
    ) -> GPUSessionController {
        let controller = GPUSessionController()
        controller.capabilityProvider = { capability }
        controller.transportProvider = {
            GPUSessionTransport(
                status: status, start: start, stop: stop, keepalive: keepalive)
        }
        return controller
    }

    private func record(
        state: String, idleRemainingSeconds: Int? = nil, busy: Bool? = nil
    ) -> GPUSessionRecord {
        GPUSessionRecord(
            sessionGeneration: "g-1", state: state,
            idleRemainingSeconds: idleRemainingSeconds, busy: busy)
    }

    @Test func unknownStateBlocksAndOffersOnlyVerifiedRelease() async {
        // "unknown" = the scheduler conversation broke; the server refuses to
        // guess whether the allocation is gone. NONTERMINAL: it must read as
        // an active session (no Start over a possible billed orphan), never
        // fire the became-reachable hook, and clear only via the operator's
        // release — which must use the force verb, not plain stop.
        // Sendable-safe call counters (the transport closures are @Sendable).
        final class Calls: @unchecked Sendable {
            private let lock = NSLock()
            private var counts: [String: Int] = [:]
            func bump(_ key: String) { lock.lock(); counts[key, default: 0] += 1; lock.unlock() }
            func count(_ key: String) -> Int {
                lock.lock(); defer { lock.unlock() }; return counts[key, default: 0]
            }
        }
        let calls = Calls()
        let controller = GPUSessionController()
        controller.capabilityProvider = { true }
        controller.transportProvider = {
            GPUSessionTransport(
                status: { nil },
                start: { _ in GPUSessionRecord(sessionGeneration: "g", state: "queued") },
                stop: {
                    calls.bump("stop")
                    return GPUSessionRecord(sessionGeneration: "g", state: "ending")
                },
                keepalive: {},
                forceClear: {
                    calls.bump("force")
                    return GPUSessionRecord(
                        sessionGeneration: "g", state: "ended",
                        stateDetail: "cleared by operator after manual verification")
                })
        }
        var readyFires = 0
        controller.onBecameReady = { readyFires += 1 }

        controller.ingest(record(state: "unknown"))
        #expect(controller.displayState == .unknownState)
        #expect(controller.displayState.label == "Unknown — verify on cluster")
        #expect(!controller.displayState.isTerminal)
        #expect(controller.isActive, "unknown must keep the session slot occupied")
        #expect(readyFires == 0, "unknown is not reachability")
        #expect(controller.sessionNotice == nil, "unknown is not ended")

        await controller.releaseVerifiedGone()
        #expect(calls.count("force") == 1)
        #expect(calls.count("stop") == 0, "release must use the force verb, not plain stop")
        #expect(controller.displayState == .ended)

        // Recovery path: an unknown session whose worker reappears goes back
        // to ready — and THAT counts as becoming reachable.
        controller.clearSessionNotice()
        controller.ingest(record(state: "unknown"))
        controller.ingest(record(state: "ready"))
        #expect(readyFires == 1)
        #expect(controller.displayState == .ready)
    }

    @Test func becameReadyFiresOncePerReachabilityTransition() {
        // Live shakedown: a model load raced the session's startup, the
        // pre-session /api/state answer (no loaded model) went stale, and
        // the chat composer stayed locked. The store hangs a state refresh
        // on this hook — it must fire exactly once per became-reachable
        // transition, not on every poll.
        let controller = makeController()
        var fires = 0
        controller.onBecameReady = { fires += 1 }

        controller.ingest(record(state: "queued"))
        controller.ingest(record(state: "starting"))
        #expect(fires == 0)
        controller.ingest(record(state: "ready"))
        #expect(fires == 1)
        controller.ingest(record(state: "busy"))
        controller.ingest(record(state: "idle", idleRemainingSeconds: 300))
        #expect(fires == 1, "staying reachable must not refire")
        controller.ingest(record(state: "ended"))
        #expect(fires == 1)
        // A NEW session becoming reachable fires again.
        controller.clearSessionNotice()
        controller.ingest(record(state: "queued"))
        controller.ingest(record(state: "ready"))
        #expect(fires == 2)
        // Connection change resets the edge detector.
        controller.resetForConnectionChange()
        controller.ingest(record(state: "ready"))
        #expect(fires == 3)
    }

    @Test func interruptedStreamNamesTheTruncation() {
        // EOF without a terminal `done` event used to render truncation as a
        // clean completion (live: an assistant turn that never answered, no
        // error). The error text must say the output is truncated and name
        // the likely causes.
        let message = ClusterClient.ClientError.interruptedStream.description
        #expect(message.contains("interrupted"))
        #expect(message.contains("truncated"))
        #expect(
            ClusterClient.ClientError.interruptedStream.localizedDescription == message)
    }

    @Test func serverSentEventDecodesTheFullKeyVocabulary() throws {
        // The SSE contract with the Python server (`_locked_sse` +
        // `variant_generate_stream`): chunk, done, error, variant, and the
        // 2026-07-17 `status` preamble/heartbeat ("model X is not loaded yet
        // — loading it now… (45s elapsed)") that captions an in-stream cold
        // load. A status event must decode alongside — not instead of — the
        // other fields, and unknown keys must not break decoding.
        let decoder = JSONDecoder()
        let status = try decoder.decode(
            ServerSentEvent.self,
            from: Data(#"{"status": "model g is not loaded yet (45s elapsed)"}"#.utf8))
        #expect(status.status == "model g is not loaded yet (45s elapsed)")
        #expect(status.chunk == nil && status.done == nil && status.error == nil)

        let chunk = try decoder.decode(
            ServerSentEvent.self, from: Data(#"{"chunk": "hel"}"#.utf8))
        #expect(chunk.chunk == "hel" && chunk.status == nil)

        let done = try decoder.decode(
            ServerSentEvent.self,
            from: Data(#"{"done": true, "modelID": "org/m", "variant": {"source": "inline"}}"#.utf8))
        #expect(done.done == true)
        #expect(done.variant?["source"] == .string("inline"))

        let error = try decoder.decode(
            ServerSentEvent.self, from: Data(#"{"error": "boom"}"#.utf8))
        #expect(error.error == "boom")
    }

    @Test func displayStateMapsThePlanStates() {
        let controller = makeController()
        #expect(controller.displayState == .off)
        #expect(controller.displayState.label == "Off")

        controller.ingest(record(state: "queued"))
        #expect(controller.displayState.label == "Queued")
        controller.ingest(record(state: "starting"))
        #expect(controller.displayState.label == "Starting")
        controller.ingest(record(state: "ready"))
        #expect(controller.displayState.label == "Ready")
        controller.ingest(record(state: "busy"))
        #expect(controller.displayState.label == "Busy")
        controller.ingest(record(state: "idle", idleRemainingSeconds: 1080))
        #expect(controller.displayState.label == "Idle 18m")
        controller.ingest(record(state: "ending"))
        #expect(controller.displayState.label == "Ending")
        controller.resetForConnectionChange()
    }

    @Test func busyFlagOutranksTheStateString() {
        // A generation in flight reads Busy even while the stored state
        // string lags (the worker refuses to expire mid-request anyway).
        let controller = makeController()
        controller.ingest(record(state: "idle", idleRemainingSeconds: 600, busy: true))
        #expect(controller.displayState == .busy)
        controller.resetForConnectionChange()
    }

    @Test func unknownStateDisplaysVerbatimNotAsFailure() {
        let controller = makeController()
        controller.ingest(record(state: "draining"))
        #expect(controller.displayState == .other("draining"))
        #expect(controller.displayState.label == "draining")
        #expect(controller.sessionNotice == nil)
        controller.resetForConnectionChange()
    }

    @Test func idleLabelFormatting() {
        #expect(GPUSessionDisplayState.idleLabel(remainingSeconds: 1080) == "Idle 18m")
        #expect(GPUSessionDisplayState.idleLabel(remainingSeconds: 1050) == "Idle 18m")
        #expect(GPUSessionDisplayState.idleLabel(remainingSeconds: 30) == "Idle 1m")
        #expect(GPUSessionDisplayState.idleLabel(remainingSeconds: nil) == "Idle")
        #expect(GPUSessionDisplayState.idleLabel(remainingSeconds: 0) == "Idle")
    }

    @Test func nullAfterSessionProducesControllerStillConnectedMessage() {
        // Session end (any cause) must read as the plan's exact message and
        // NEVER as a lost server connection — the notice lives on the
        // session controller; nothing here touches connection status.
        let controller = makeController()
        controller.ingest(record(state: "ready"))
        #expect(controller.sessionNotice == nil)
        controller.ingest(nil)
        #expect(controller.sessionNotice == "GPU session ended — controller still connected")
        #expect(controller.displayState == .ended)
        #expect(!controller.isActive)
        #expect(!controller.isPolling)
        controller.resetForConnectionChange()
    }

    @Test func endedAndFailedStatesProduceTheNoticeAndTerminalStates() {
        let controller = makeController()
        controller.ingest(record(state: "ready"))
        controller.ingest(record(state: "ended"))
        #expect(controller.sessionNotice == "GPU session ended — controller still connected")
        #expect(controller.displayState == .ended)

        controller.resetForConnectionChange()
        #expect(controller.sessionNotice == nil)

        controller.ingest(record(state: "ready"))
        controller.ingest(record(state: "failed"))
        #expect(controller.sessionNotice == "GPU session ended — controller still connected")
        #expect(controller.displayState == .failed)
        controller.resetForConnectionChange()
    }

    @Test func nullWithoutPriorSessionIsJustOff() {
        // First status check on connect finding nothing is not an ending.
        let controller = makeController()
        controller.ingest(nil)
        #expect(controller.sessionNotice == nil)
        #expect(controller.displayState == .off)
    }

    @Test func noPollingWithoutCapability() async {
        // Older servers 404 on the session routes — the controller must not
        // poll (or fetch at all) when the capability is absent.
        let controller = makeController(
            capability: false,
            status: {
                Issue.record("status must never be fetched without the capability")
                return nil
            })
        controller.refreshAfterConnect()
        await controller.syncOnce()
        #expect(!controller.isPolling)
        // Even a record placed by hand must not start the poll loop.
        controller.ingest(record(state: "ready"))
        #expect(!controller.isPolling)
        controller.resetForConnectionChange()
    }

    @Test func liveSessionPollsAndEndedSessionStops() {
        let controller = makeController()
        controller.ingest(record(state: "ready"))
        #expect(controller.isPolling)
        controller.ingest(record(state: "ended"))
        #expect(!controller.isPolling)
        controller.resetForConnectionChange()
        #expect(!controller.isPolling)
    }

    @Test func startAdoptsExistingSessionOn409() async {
        // Concurrent start is idempotent-or-409 (plan §2.5): the refusal
        // body carries the running record, which the controller adopts.
        let body = """
            {"detail": "a GPU session is already running",
             "session": {"sessionGeneration": "g-live", "state": "ready"}}
            """
        let controller = makeController(start: { _ in
            throw ClusterClient.ClientError.badResponse(409, body)
        })
        await controller.start(request: GPUSessionStartRequest())
        #expect(controller.record?.sessionGeneration == "g-live")
        #expect(controller.displayState == .ready)
        #expect(controller.lastActionError?.contains("already running") == true)
        controller.resetForConnectionChange()
    }

    @Test func startFailureSurfacesTheDetailWithoutTouchingRecord() async {
        let controller = makeController(start: { _ in
            throw ClusterClient.ClientError.badResponse(
                400, #"{"detail": "maintenance window imminent"}"#)
        })
        await controller.start(request: GPUSessionStartRequest())
        #expect(controller.record == nil)
        #expect(controller.lastActionError?.contains("maintenance window imminent") == true)
        #expect(controller.displayState == .off)
    }

    @Test func transientPollFailureDoesNotEndTheSession() async {
        // A network blip on GET /api/session is not "the session ended" —
        // only a successful answer with a null session counts.
        let controller = makeController(status: { throw URLError(.timedOut) })
        controller.ingest(record(state: "ready"))
        await controller.syncOnce()
        #expect(controller.record?.state == "ready")
        #expect(controller.sessionNotice == nil)
        #expect(controller.isActive)
        controller.resetForConnectionChange()
    }

    @Test func stopIngestsTheEndingRecord() async {
        let controller = makeController(stop: {
            GPUSessionRecord(sessionGeneration: "g-1", state: "ending")
        })
        controller.ingest(record(state: "ready"))
        await controller.stop()
        #expect(controller.displayState == .ending)
        // Still watching: "ending" is not terminal; polling continues until
        // the record lands ended/null.
        #expect(controller.isPolling)
        controller.resetForConnectionChange()
    }

    @Test func connectionChangeResetSilencesEverything() {
        let controller = makeController()
        controller.ingest(record(state: "ready"))
        controller.resetForConnectionChange()
        #expect(controller.record == nil)
        #expect(controller.displayState == .off)
        #expect(controller.sessionNotice == nil)
        #expect(!controller.isPolling)
        // The reset is not ended-detection: a later null stays silent.
        controller.ingest(nil)
        #expect(controller.sessionNotice == nil)
    }
}

// MARK: - Model-first sizing

@Suite struct GPUSessionSizingTests {

    /// A fictional site inventory (see ClusterSiteTestFixtures).
    private var exampleInventory: ClusterSiteProfile.SlurmSiteData {
        ClusterSiteProfile.SlurmSiteData(
            partitions: [.init(name: "gpu_p", maxWalltimeHours: 168)],
            gpuTypes: ["A100", "H100", "L4", "P100"],
            gpuVRAMGB: ["A100": 80, "H100": 80, "L4": 24, "P100": 16],
            defaultGres: "gpu:A100:1",
            defaultPartition: "gpu_p")
    }

    @Test func parameterCountParsesOutOfModelIDs() {
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "mlx-community/gemma-3-27b-it-8bit") == 27)
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "Qwen/Qwen3-14B-MLX-8bit") == 14)
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "Qwen/Qwen3-4B") == 4)
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "Qwen/Qwen3-0.6B") == 0.6)
        // "4bit"/"8bit" quantization suffixes never read as sizes, and the
        // family version digit ("3") loses to the actual size.
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "mlx-community/gemma-3-4b-it-4bit") == 4)
        #expect(GPUSessionSizing.parameterBillions(fromModelID: "org/mystery-model") == nil)
    }

    @Test func largeModelPicksAn80GBGPU() {
        // 27B → 27 × 2 × 1.3 ≈ 70 GB: L4/P100 are out; smallest fitting is
        // the 80 GB tier (A100 before H100, deterministic tie-break).
        let suggestion = GPUSessionSizing.suggest(
            modelID: "mlx-community/gemma-3-27b-it-8bit", slurm: exampleInventory)
        #expect(suggestion.gpuType == "A100")
        #expect(suggestion.gres == "gpu:A100:1")
        #expect(suggestion.walltime == "02:00:00")
        #expect(suggestion.idleMinutes == 30)
    }

    @Test func smallModelPicksTheSmallestFittingGPU() {
        // 4B ≈ 10.4 GB → P100 (16 GB) is the smallest that fits.
        let suggestion = GPUSessionSizing.suggest(
            modelID: "Qwen/Qwen3-4B", slurm: exampleInventory)
        #expect(suggestion.gpuType == "P100")
        #expect(suggestion.gres == "gpu:P100:1")
    }

    @Test func midModelSkipsTheTooSmallTier() {
        // 14B ≈ 36.4 GB → L4 (24) too small, next fitting is 80 GB.
        let suggestion = GPUSessionSizing.suggest(
            modelID: "Qwen/Qwen3-14B-MLX-8bit", slurm: exampleInventory)
        #expect(suggestion.gpuType == "A100")
    }

    @Test func unparseableModelDefaultsToTheLargestGPU() {
        let suggestion = GPUSessionSizing.suggest(
            modelID: "org/mystery-model", slurm: exampleInventory)
        #expect(suggestion.gpuType == "H100")  // largest by (vram, name)
        #expect(suggestion.estimatedVRAMGB == nil)
        #expect(suggestion.rationale.contains("could not parse"))
    }

    @Test func singleGPUTypeSiteAlwaysSuggestsIt() {
        let site = ClusterSiteProfile.SlurmSiteData(
            gpuTypes: ["L4"], gpuVRAMGB: ["L4": 24])
        #expect(GPUSessionSizing.suggest(modelID: "Qwen/Qwen3-4B", slurm: site).gpuType == "L4")
        // Even a model that cannot fit: the one type is suggested, loudly.
        let big = GPUSessionSizing.suggest(
            modelID: "mlx-community/gemma-3-27b-it-8bit", slurm: site)
        #expect(big.gpuType == "L4")
        #expect(big.rationale.contains("may not fit"))
    }

    @Test func noInventoryFallsBackToSiteDefaults() {
        let bare = ClusterSiteProfile.SlurmSiteData(defaultGres: "gpu:1")
        let suggestion = GPUSessionSizing.suggest(modelID: "Qwen/Qwen3-4B", slurm: bare)
        #expect(suggestion.gpuType == nil)
        #expect(suggestion.gres == "gpu:1")
        let none = GPUSessionSizing.suggest(modelID: "Qwen/Qwen3-4B", slurm: nil)
        #expect(none.gres == nil)
        #expect(none.walltime == "02:00:00")
        #expect(none.idleMinutes == 30)
    }
}

// MARK: - "No GPU session" 409 hint

@Suite struct GPUSessionRefusalTests {
    @Test func noSessionRefusalGetsThePlaygroundPointer() throws {
        let hint = try #require(
            GPUSessionRefusal.hint(
                code: 409,
                body: #"{"detail": "no GPU session — start one to load models"}"#))
        #expect(hint.contains("no GPU session"))
        #expect(hint.contains("GPU Session control"))
    }

    @Test func matchIsTolerantOfPhrasingAndCase() {
        #expect(
            GPUSessionRefusal.hint(
                code: 409, body: #"{"detail": "No GPU Session is running"}"#) != nil)
        // Unstructured body still matches on the string.
        #expect(GPUSessionRefusal.hint(code: 409, body: "no gpu session here") != nil)
    }

    @Test func otherErrorsPassThroughUntouched() {
        // A 409 about something else (e.g. busy model slot) is not this.
        #expect(GPUSessionRefusal.hint(code: 409, body: #"{"detail": "model is busy"}"#) == nil)
        // The right words on the wrong status are not this either.
        #expect(
            GPUSessionRefusal.hint(
                code: 400, body: #"{"detail": "no GPU session"}"#) == nil)
    }
}
