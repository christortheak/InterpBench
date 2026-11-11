import Foundation
import Testing

@testable import ExperimentKit

/// The Mac's half of an engine's refusal on a science route.
///
/// The engine's science routes (`/api/science/…`: the stage, plan, and submit
/// calls, the diagnostic transport calls, and the catalogued actions those
/// routes own) refuse with `{"detail": {"code", "reason", "repairAction"}}`
/// and no `message`. A refusal there that names a code is the engine's own
/// typed refusal: the Mac throws it with the engine's code, sentence, and
/// repair, the command line reports it as `refused` (65) with the engine's
/// code as `error.code`, and the app shows the sentence and the repair. Every
/// other answer (a refusal that names no code, a 401, a failure) is thrown as
/// the bad response it always was.
///
/// The server here is a stand-in that answers what a test tells it to, so
/// every expectation is about what this CLIENT says. The answers and what
/// both clients make of them are held to one file,
/// `prompts/fixtures/engine-refusals/science-routes.json`. Python twin:
/// `Server/tests/test_science_refusals.py`.
@Suite(.serialized) struct EngineRefusalTests {

    // MARK: - The fixture

    private struct Fixture: Decodable {
        struct Call: Decodable {
            var name: String
            var path: String
            var operation: String?
            var action: String?
        }
        struct AnotherRoute: Decodable {
            var path: String
            var operation: String
            var action: String
            var status: Int
            var body: String
        }
        struct Fallback: Decodable {
            var status: Int
            var reason: String
            var repairAction: String
            var textLimit: Int
        }
        struct Answer: Decodable {
            var note: String
            var status: Int
            var body: String
            var engineRefusal: Bool
            var code: String?
            var reason: String?
            var repairAction: String?
        }
        var server: String
        var jobID: String
        var calls: [Call]
        var anotherRoute: AnotherRoute
        var fallback: Fallback
        var bodies: [Answer]
    }

    private static var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func fixture() throws -> Fixture {
        let url = repository.appending(path: "prompts/fixtures/engine-refusals/science-routes.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private static let staged = String(repeating: "a", count: 64)
    private static let plan = String(repeating: "b", count: 64)
    /// Every call this client makes to a science route, by the fixture's name.
    private static let calls = [
        "stage", "plan", "submit", "export", "cleanup-plan", "cleanup-apply", "fitting-round", "campaign",
    ]

    // MARK: - A stand-in engine

    /// Answers every request with one status and one body, and keeps the
    /// paths it was asked for. The one exception is the capabilities read
    /// that `science-fetch` makes before it exports: that is answered as an
    /// engine that permits HTTP transfer, and is not kept.
    private final class Engine: @unchecked Sendable {
        private let lock = NSLock()
        private var status = 200
        private var body = Data(#"{"jobId": "job-1"}"#.utf8)
        private var asked: [String] = []
        let server: String

        init(server: String) { self.server = server }

        func answer(_ status: Int, _ body: String) {
            lock.lock()
            defer { lock.unlock() }
            self.status = status
            self.body = Data(body.utf8)
            asked = []
        }

        var paths: [String] {
            lock.lock()
            defer { lock.unlock() }
            return asked
        }

        func session() -> URLSession {
            EngineRefusalStubURLProtocol.handler = { [self] request in
                lock.lock()
                defer { lock.unlock() }
                let path = request.url?.path ?? ""
                if path == "/api/capabilities" {
                    return (Data(#"{"remoteStudy": {"httpTransfer": true, "externalTransferRequired": false}}"#.utf8), 200)
                }
                asked.append(path)
                return (body, status)
            }
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [EngineRefusalStubURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        func client() -> ClusterClient {
            ClusterClient(profile: .init(baseURL: URL(string: server)!), session: session())
        }
    }

    /// A request document for one catalogued action: its path parameters, no
    /// query, and an empty body.
    private static func catalogued(job: String, parameters: Bool = true) -> Data {
        let path = parameters ? #"{"job_id": "\#(job)", "action": "status"}"# : "{}"
        return Data(#"{"path": \#(path), "query": {}, "body": {}}"#.utf8)
    }

    private static func invoke(_ call: Fixture.Call, _ client: ClusterClient, job: String) async throws {
        let request = JSONValue.object(["inputBundleSHA256": .string(staged)])
        let custody = JSONValue.object(["receiptSHA256": .string(staged)])
        switch call.name {
        case "stage": _ = try await client.stageDiagnostic(path: "/staged/request.tar.gz", sha256: staged)
        case "plan": _ = try await client.scientificPlan(request)
        case "submit": _ = try await client.scientificSubmit(request, planSHA256: plan)
        case "export": _ = try await client.exportDiagnostic(job)
        case "cleanup-plan": _ = try await client.planDiagnosticCleanup(job, custody: custody)
        case "cleanup-apply": _ = try await client.applyDiagnosticCleanup(job, custody: custody, planSHA256: plan)
        default:
            _ = try await client.callScientificAction(
                operation: try #require(call.operation), actionID: try #require(call.action),
                document: catalogued(job: job))
        }
    }

    // MARK: - What the client throws

    @Test func theFixtureNamesEveryScienceCallThisClientMakes() throws {
        let fixture = try Self.fixture()
        #expect(fixture.calls.map(\.name).sorted() == Self.calls.sorted())
        #expect(fixture.calls.allSatisfy { $0.path.hasPrefix("/api/science/") })
        #expect(fixture.bodies.contains { $0.engineRefusal } && fixture.bodies.contains { !$0.engineRefusal })
    }

    /// Read from the source, so that a call added later cannot go without
    /// the relay. Five files hold a science path as text. Two make the
    /// calls, and every call in them is relayed. `ClusterClient` forwards a
    /// catalogued action and relays it when its path is a science route.
    /// The other two make no call: the catalog's own text, and the routes
    /// the Mac's local server answers itself.
    @Test func everyCallThisClientMakesToAScienceRouteIsRelayed() throws {
        let sources = Self.repository.appending(component: "Sources")
        func count(_ needle: String, in text: String) -> Int { text.components(separatedBy: needle).count - 1 }
        var naming: [String: String] = [:]
        let files = try #require(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        for case let file as URL in files where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            if text.contains(#""/api/science/"#) { naming[file.lastPathComponent] = text }
        }
        #expect(naming.keys.sorted() == [
            "ClusterClient.swift", "DiagnosticRemote.swift", "RemoteScientificWorkflows.swift",
            "ScienceResourceText.swift", "WebServer.swift",
        ])
        // Every request `DiagnosticRemote` sends goes to a science route.
        let transport = try #require(naming["DiagnosticRemote.swift"])
        #expect(count("try await post(", in: transport) == 4)
        #expect(count("relayingEngineRefusal {", in: transport) == count("try await post(", in: transport))
        let workflows = try #require(naming["RemoteScientificWorkflows.swift"])
        #expect(count(#"post("/api/science/"#, in: workflows) == 2)
        #expect(count("relayingEngineRefusal {", in: workflows) == count(#"post("/api/science/"#, in: workflows))
        let client = try #require(naming["ClusterClient.swift"])
        #expect(client.contains(
            #"if resolved.path.hasPrefix("/api/science/"), let refusal = Self.engineRefusal(in: error) { throw refusal }"#))
        #expect(!client.contains(#"post("/api/science/"#) && !client.contains(#"get("/api/science/"#))
    }

    /// Each answer in the fixture, on each science call. A refusal that
    /// names a code is thrown with the engine's code, sentence, and repair.
    /// Every other answer is the bad response it always was, with the
    /// engine's body untouched.
    @Test(arguments: EngineRefusalTests.calls)
    func aCodedRefusalOnAScienceRouteIsTheEnginesOwnAndEveryOtherAnswerReadsAsItDid(_ name: String) async throws {
        let fixture = try Self.fixture()
        let call = try #require(fixture.calls.first { $0.name == name })
        let engine = Engine(server: fixture.server)
        let client = engine.client()

        // A success is untouched.
        try await Self.invoke(call, client, job: fixture.jobID)
        #expect(engine.paths == [call.path])

        for answer in fixture.bodies {
            engine.answer(answer.status, answer.body)
            do {
                try await Self.invoke(call, client, job: fixture.jobID)
                Issue.record("\(answer.note): an answer that is not a success must throw")
            } catch let refusal as ClusterClient.EngineRefusal {
                #expect(answer.engineRefusal, "\(answer.note): thrown as the engine's own refusal")
                #expect(refusal.status == answer.status, "\(answer.note)")
                #expect(refusal.code == answer.code, "\(answer.note)")
                #expect(refusal.reason == answer.reason, "\(answer.note)")
                #expect(refusal.repairAction == answer.repairAction, "\(answer.note)")
                // The app shows an error's description: the sentence and the repair.
                #expect(refusal.localizedDescription == refusal.reason + " " + refusal.repairAction)
            } catch let error as ClusterClient.ClientError {
                #expect(!answer.engineRefusal, "\(answer.note): thrown as a bad response")
                guard case .badResponse(let status, let text) = error else {
                    Issue.record("\(answer.note): not the bad response it always was: \(error)")
                    continue
                }
                #expect(status == answer.status && text == answer.body, "\(answer.note)")
            } catch {
                Issue.record("\(answer.note): thrown as \(type(of: error))")
            }
            #expect(engine.paths == [call.path], "\(answer.note)")
        }
    }

    @Test func whatTheEngineLeavesOutIsOneSentenceAndOneRepairAndNoMoreTextThanTheLimit() async throws {
        let fixture = try Self.fixture()
        #expect(ClusterClient.EngineRefusal.fallbackReason(status: fixture.fallback.status) == fixture.fallback.reason)
        // The repair is the one a lens-library refusal gets: one sentence
        // for every refusal that names none.
        #expect(LensLibrary.refusalFallbackRepair == fixture.fallback.repairAction)
        #expect(LensLibrary.refusalTextLimit == fixture.fallback.textLimit)

        let limit = fixture.fallback.textLimit
        let long = #"{"detail": {"code": "scientificExecutionRefused", "reason": "\#(String(repeating: "r", count: limit + 50))", "#
            + #""repairAction": "\#(String(repeating: "p", count: limit + 50))"}}"#
        let refusal = try #require(ClusterClient.engineRefusal(status: 409, data: Data(long.utf8)))
        #expect(refusal.reason == String(repeating: "r", count: limit))
        #expect(refusal.repairAction == String(repeating: "p", count: limit))
    }

    /// `science-call` forwards a catalogued action to the route that owns
    /// it. Only a science route's refusal is relayed: the same body from
    /// another route is the bad response it always was.
    @Test func aCataloguedActionOnARouteOutsideTheScienceRoutesReadsAsItDid() async throws {
        let fixture = try Self.fixture()
        let other = fixture.anotherRoute
        #expect(!other.path.hasPrefix("/api/science/"))
        #expect(ClusterClient.engineRefusal(status: other.status, data: Data(other.body.utf8)) != nil)
        let engine = Engine(server: fixture.server)
        let client = engine.client()
        engine.answer(other.status, other.body)
        do {
            _ = try await client.callScientificAction(
                operation: other.operation, actionID: other.action,
                document: Self.catalogued(job: fixture.jobID, parameters: false))
            Issue.record("a refused action must throw")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(let status, let text) = error else {
                Issue.record("not the bad response it always was: \(error)")
                return
            }
            #expect(status == other.status && text == other.body)
        }
        #expect(engine.paths == [other.path])
    }

    // MARK: - On the command line

    /// The science verbs that need nothing but the engine. `science-stage`
    /// confirms the Python client's files first, and the cleanup verbs
    /// verify a custody receipt first, so those are held at the client
    /// (`calls`) and, for a stage, in `aStageTheEngineRefusesIsReportedTheSameWay`.
    private static let verbs = ["science-call", "science-export", "science-fetch", "science-plan", "science-submit"]

    /// One command line per science verb that needs nothing but the engine.
    private static func commandLine(_ verb: String, fixture: Fixture, in directory: URL) throws -> [String] {
        let request = directory.appending(component: "request.json")
        try Data(#"{"inputBundleSHA256": "\#(staged)"}"#.utf8).write(to: request)
        let action = directory.appending(component: "action.json")
        try catalogued(job: fixture.jobID).write(to: action)
        let round = try #require(fixture.calls.first { $0.name == "fitting-round" })
        let arguments: [String]
        switch verb {
        case "science-plan": arguments = [verb, request.path]
        case "science-submit": arguments = [verb, request.path, "--plan-sha256", plan]
        // A fetch reads the engine's capabilities, then asks for the
        // export: the refusal is the export's.
        case "science-export", "science-fetch": arguments = [verb, fixture.jobID]
        default:
            arguments = [
                verb, try #require(round.operation), "--action", try #require(round.action), "--request", action.path,
            ]
        }
        return arguments + ["--url", fixture.server]
    }

    private static func run(_ arguments: [String], engine: Engine, in directory: URL) async -> ExperimentCLIOutcome {
        let session = engine.session()
        ExperimentRootOverrideLock.acquire()
        ExperimentStore.rootOverride = directory
        defer {
            ExperimentStore.rootOverride = nil
            ExperimentRootOverrideLock.release()
        }
        return await ExperimentCLIRunner.$remoteSessionOverride.withValue(session) {
            await ExperimentCLIRunner(sink: .discarding).run(namespace: "remote", arguments)
        }
    }

    private static func temporary() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "engine-refusal-\(UUID().uuidString)").resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// The whole command line, from its arguments to its envelope: `refused`
    /// (65 in JSON mode, 1 in human mode), with the engine's code as
    /// `error.code`, the engine's sentence, and the engine's repair.
    @Test(arguments: EngineRefusalTests.verbs)
    func theCommandLineReportsTheEnginesRefusalAsRefusedWithItsCodeSentenceAndRepair(_ verb: String) async throws {
        let fixture = try Self.fixture()
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let arguments = try Self.commandLine(verb, fixture: fixture, in: directory)
        let engine = Engine(server: fixture.server)
        for answer in fixture.bodies where answer.engineRefusal {
            engine.answer(answer.status, answer.body)
            let outcome = await Self.run(arguments, engine: engine, in: directory)
            #expect(outcome.envelope.state == .refused, "\(answer.note)")
            #expect(outcome.exitCode(json: true) == 65 && outcome.exitCode(json: false) == 1, "\(answer.note)")
            #expect(outcome.envelope.verb == "remote \(verb)")
            #expect(outcome.envelope.error?.code == answer.code, "\(answer.note)")
            #expect(outcome.envelope.error?.reason == answer.reason, "\(answer.note)")
            #expect(outcome.envelope.error?.repairAction == answer.repairAction, "\(answer.note)")
            // A person at a terminal reads the sentence, then the repair.
            #expect(outcome.failure?.reason == answer.reason && outcome.failure?.repairAction == answer.repairAction)
            #expect(engine.paths.count == 1 && engine.paths[0].hasPrefix("/api/science/"), "\(answer.note)")
        }
    }

    /// Every other answer keeps the report it had: a failed verb that quotes
    /// the engine's status and body.
    @Test(arguments: EngineRefusalTests.verbs)
    func theCommandLineReportsEveryOtherAnswerAsItDid(_ verb: String) async throws {
        let fixture = try Self.fixture()
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let arguments = try Self.commandLine(verb, fixture: fixture, in: directory)
        let engine = Engine(server: fixture.server)
        for answer in fixture.bodies where !answer.engineRefusal {
            engine.answer(answer.status, answer.body)
            let outcome = await Self.run(arguments, engine: engine, in: directory)
            #expect(outcome.envelope.state == .failed && outcome.exitCode(json: true) == 70, "\(answer.note)")
            #expect(outcome.envelope.error?.code == "verbFailed", "\(answer.note)")
            #expect(
                outcome.envelope.error?.reason == "server returned \(answer.status): \(answer.body)",
                "\(answer.note)")
        }
    }

    /// `science-stage` confirms the Python client's files first and records
    /// the staged request last. Between them, the engine's refusal passes
    /// through as it is, and the runner reports it the same way.
    @Test func aStageTheEngineRefusesIsReportedTheSameWay() async throws {
        let fixture = try Self.fixture()
        let answer = try #require(fixture.bodies.first { $0.engineRefusal })
        let engine = Engine(server: fixture.server)
        let client = engine.client()
        engine.answer(answer.status, answer.body)
        do {
            _ = try await DiagnosticRemote.stage(
                path: "/staged/request.tar.gz", sha256: Self.staged, client: client,
                root: URL(filePath: "/workspace"), confirm: {},
                record: { _ in
                    Issue.record("nothing is recorded for a stage the engine refused")
                    return .object([:])
                })
            Issue.record("a stage the engine refuses must throw")
        } catch let refusal as ClusterClient.EngineRefusal {
            let outcome = ExperimentCLIRunner(sink: .discarding).outcome(
                for: refusal, namespace: "remote", verb: "remote science-stage")
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.exitCode(json: true) == 65 && outcome.exitCode(json: false) == 1)
            #expect(outcome.envelope.error?.code == answer.code)
            #expect(outcome.envelope.error?.reason == answer.reason)
            #expect(outcome.envelope.error?.repairAction == answer.repairAction)
        }
        #expect(engine.paths == ["/api/science/stage"])
    }
}

/// The stub transport for this suite alone (its own handler, so it shares
/// nothing with another suite's stub).
private final class EngineRefusalStubURLProtocol: URLProtocol, @unchecked Sendable {
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
