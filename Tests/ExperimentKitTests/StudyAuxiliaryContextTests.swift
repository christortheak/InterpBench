import Foundation
import Testing
@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct StudyAuxiliaryContextTests {
    @Test(arguments: ["runs", "optimizations", "residency", "judgments", "sweep-detail"], [false, true])
    func auxiliaryObservationDiscardsChangedOrigin(operation: String, changed: Bool) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(component: "auxiliary-context-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        let suite = "auxiliary-context-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            defaults.removePersistentDomain(forName: suite)
            ExperimentRootOverrideLock.release()
        }
        _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
        let store = clusterStore(defaults: defaults)
        let first = store.addServer(name: "First", urlString: "http://first.invalid:19322")
        store.activeWorkspace = .server(first.id)
        store.applyServerWorkspaceSwitch(info: .init(root: "/remote/first"))
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [AuxiliaryReplyProtocol.self]
        let session = URLSession(configuration: configuration)
        store.clientSessionOverride = session
        let host = ChatService(cluster: store)
        let panel = host.experiments
        panel.management.selectedName = "study"
        AuxiliaryReplyProtocol.setHook {
            if changed { store.remoteInfo = .init(root: "/remote/second") }
        }
        defer { AuxiliaryReplyProtocol.setHook(nil); session.invalidateAndCancel() }
        switch operation {
        case "runs":
            await panel.refreshRemoteRuns()
            #expect(panel.remoteRuns.isEmpty == changed)
        case "optimizations":
            await panel.refreshRemoteOptimizations()
            #expect(panel.remoteOptimizations.isEmpty == changed)
        case "residency":
            await panel.refreshServerResidency()
            #expect(panel.serverHasSelectedStudy == (changed ? nil : true))
        case "judgments":
            await panel.refreshAwaitingSweepJudgments(study: "study")
            #expect(panel.awaitingSweepJudgments.isEmpty == changed)
        default:
            let sweep = await panel.loadRemoteSweepRun(experiment: "study")
            if changed { #expect(sweep == nil) }
            // The unchanged detail parser is covered by OptimizationsRemoteTests.
        }
    }
}

private final class AuxiliaryReplyProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var hook: (@MainActor @Sendable () -> Void)?
    static func setHook(_ value: (@MainActor @Sendable () -> Void)?) { lock.withLock { hook = value } }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let hook = Self.lock.withLock { Self.hook }
        let reply = PendingReply(handler: self, url: request.url!)
        Task.detached { await hook?(); reply.send() }
    }
    private struct PendingReply: @unchecked Sendable {
        let handler: AuxiliaryReplyProtocol
        let url: URL
        func send() {
            let text: String
            if url.path == "/api/experiments" {
                text = #"{"experiments":[{"name":"study","status":"draft","conditions":[]}]}"#
            } else if url.path == "/api/runs" {
                text = #"{"runs":[{"id":"20260101T000000000-exp-study-sweep","path":"runs/20260101T000000000-exp-study-sweep","hasReport":false,"hasGenerations":false,"hasCosineMatrix":false,"vectorNames":[],"files":["sweep.csv"]}]}"#
            } else if url.path.hasSuffix("/awaiting") {
                text = #"{"awaiting":[{"run":"awaiting-run"}]}"#
            } else { text = "concept,layer\n" }
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
            handler.client?.urlProtocol(handler, didReceive: response, cacheStoragePolicy: .notAllowed)
            handler.client?.urlProtocol(handler, didLoad: Data(text.utf8))
            handler.client?.urlProtocolDidFinishLoading(handler)
        }
    }
    override func stopLoading() {}
}
