import Foundation
import Testing

@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct StudyOperationEnvironmentTests {
    private func context(
        client: ClusterClient?, name: String? = "study", draft: Bool = true,
        server: Bool = true, root: URL = URL(filePath: "/private/tmp/operation-context"),
        remoteRoot: String = "/remote/workspace"
    ) -> StudyOperationContext {
        StudyOperationContext(
            workspaceRoot: root, selectedName: name, selectedIsDraft: draft,
            serverURL: "http://runner.invalid", isServer: server, pairing: .paired,
            substrate: "server", capabilities: nil, client: client, hasDisplay: false,
            serverOrigin: .init(serverIdentity: "fixture", remoteRoot: remoteRoot, workspaceRoot: root))
    }

    private func withClient(_ body: (ClusterClient) async throws -> Void) async rethrows {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OperationReplyProtocol.self]
        let session = URLSession(configuration: configuration)
        OperationReplyProtocol.reset()
        defer { session.invalidateAndCancel(); OperationReplyProtocol.reset() }
        let client = ClusterClient(
            profile: .init(baseURL: URL(string: "http://runner.invalid")!, tokenKey: "fixture"),
            token: nil, session: session)
        try await body(client)
    }

    @Test func freezeAdmissionDoesNotResolveCredentialsWithoutSelectionOrServer() async {
        let owner = StudyFreezeController()
        var connections = 0
        var notes: [String] = []
        owner.presentation.note = { text, _ in notes.append(text) }
        for context in [context(client: nil, name: nil), context(client: nil, server: false)] {
            await owner.freezeOnServer(in: .init(current: { context }, connect: {
                connections += 1
                return nil
            }))
        }
        #expect(connections == 0)
        #expect(notes.count == 2)
        #expect(notes[0].contains("select a study"))
        #expect(notes[1].contains("no server workspace"))
    }

    @Test func draftSyncAdmissionDoesNotResolveCredentialsForFrozenOrLocalStudy() async {
        let owner = StudyFreezeController()
        var connections = 0
        for context in [context(client: nil, draft: false), context(client: nil, server: false)] {
            await owner.pushManifest(in: .init(current: { context }, connect: {
                connections += 1
                return nil
            }))
        }
        #expect(connections == 0)
        #expect(!owner.isSyncingServerDraft)
    }

    @Test(arguments: ["selection", "workspace", "serving-root", "connection"])
    func freezeRefusesContextChangedDuringCredentials(change: String) async {
        await withClient { client in
            var current = context(client: client)
            var notes: [String] = []
            let owner = StudyFreezeController()
            owner.presentation.note = { text, _ in notes.append(text) }
            await owner.freezeOnServer(in: .init(current: { current }, connect: {
                switch change {
                case "selection": current = context(client: client, name: "other")
                case "workspace": current = context(client: client, root: URL(filePath: "/private/tmp/other"))
                case "serving-root": current = context(client: client, remoteRoot: "/remote/other")
                default: current = context(client: nil)
                }
                return client
            }))
            #expect(OperationReplyProtocol.requests.isEmpty)
            #expect(notes.last?.contains("changed") == true)
        }
    }

    @Test func runAndBundleSubmissionRefuseCredentialRetargeting() async {
        await withClient { client in
            let jobs = StudyRemoteJobController()
            let execution = StudyServerJobCoordinator(jobs: jobs)
            let submission = StudyBundleSubmissionController(jobs: jobs)
            var current = context(client: client)
            let environment = StudyOperationEnvironment(current: { current }, connect: {
                current = context(client: client, remoteRoot: "/remote/other")
                return client
            })
            await execution.run(experimentName: "study", verb: "run", in: environment)
            current = context(client: client)
            let manifest = ExperimentManifest(name: "study", description: "", modelID: "test/model")
            let result = await submission.submit(
                manifest, request: StudySubmissionOptions().snapshot, followLog: false,
                execution: execution, in: environment)
            guard case .failure(let failure) = result else {
                Issue.record("changed destination must refuse")
                return
            }
            #expect(failure.reason.contains("changed"))
            #expect(OperationReplyProtocol.requests.isEmpty)
            #expect(jobs.recentServerJobs.isEmpty)
        }
    }

    @Test func connectionResolverCannotReturnAnUnreviewedProfile() async {
        await withClient { client in
            let jobs = StudyRemoteJobController()
            let execution = StudyServerJobCoordinator(jobs: jobs)
            var notes: [String] = []
            jobs.presentation.note = { text, _ in notes.append(text) }
            // The visible context remains unchanged but the resolver supplies another profile.
            let current = context(client: nil)
            await execution.run(experimentName: "study", verb: "run", in: .init(
                current: { current }, connect: { client }))
            #expect(notes.last?.contains("changed") == true)
            #expect(OperationReplyProtocol.requests.isEmpty)
        }
    }

    @Test func pipelineObservationUsesCapturedRootWithoutCredentials() async throws {
        try await withClient { client in
            let root = FileManager.default.temporaryDirectory.appending(component: "pipeline-context-\(UUID())")
            defer { try? FileManager.default.removeItem(at: root) }
            let run = root.appending(components: "runs", "local-run")
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            try Data(#"{"experiment":"study","stages":[]}"#.utf8)
                .write(to: run.appending(component: "pipeline-portable.json"))
            let current = context(client: client, root: root)
            var connections = 0
            let owner = StudyPipelineController()
            await owner.refresh(in: .init(current: { current }, connect: {
                connections += 1
                return client
            }))
            #expect(connections == 0)
            #expect(owner.localPipelineRuns.map(\.run) == ["local-run"])
            #expect(owner.pipelineRuns.map(\.run) == ["remote-run"])
            #expect(OperationReplyProtocol.requests == ["/api/experiment/study/pipelines"])
        }
    }

    @Test func pipelineObservationDiscardsAResponseAfterSelectionChanges() async {
        await withClient { client in
            var current = context(client: client)
            OperationReplyProtocol.setHook { current = context(client: client, name: "other") }
            let owner = StudyPipelineController()
            await owner.refresh(in: .init(current: { current }, connect: {
                Issue.record("observation must not load credentials")
                return nil
            }))
            #expect(owner.pipelineRuns.isEmpty)
        }
    }

    @Test(arguments: ["before-selection", "during-selection", "during-bytes", "missing-file"])
    func delayedPipelineKeepsReviewThroughCredentialResolution(change: String) async throws {
        try await withClient { client in
            let root = FileManager.default.temporaryDirectory.appending(component: "delayed-pipeline-\(UUID())")
            defer { try? FileManager.default.removeItem(at: root) }
            let directory = root.appending(components: "experiments", "study")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let file = directory.appending(component: "experiment.json")
            let manifest = ExperimentManifest(name: "study", description: "", modelID: "test/model")
            if change != "missing-file" { try JSONEncoder().encode(manifest).write(to: file) }
            let jobs = StudyRemoteJobController()
            let owner = StudyBundleSubmissionController(jobs: jobs)
            let execution = StudyServerJobCoordinator(jobs: jobs)
            let options = StudySubmissionOptions()
            var notes: [String] = []
            jobs.presentation.note = { text, _ in notes.append(text) }
            var current = context(client: client, root: root)
            var connections = 0
            let environment = StudyOperationEnvironment(current: { current }, connect: {
                connections += 1
                if change == "during-selection" {
                    current = context(client: client, name: "other", root: root)
                } else if change == "during-bytes" {
                    // Preserve valid JSON and scientific content; even formatting changes stale the review.
                    var data = try! Data(contentsOf: file)
                    data.append(contentsOf: [32])
                    try! data.write(to: file)
                }
                return client
            })
            let submit = owner.pipelineSubmissionAction(
                manifest: manifest, request: options.snapshot, options: options,
                execution: execution, in: environment)
            if change == "before-selection" {
                current = context(client: client, name: "other", root: root)
            }
            await submit()
            #expect(connections == (change.hasPrefix("during") ? 1 : 0))
            #expect(notes.last?.contains("changed") == true)
            #expect(OperationReplyProtocol.requests.isEmpty)
            #expect(jobs.recentServerJobs.isEmpty)
        }
    }

    @Test func vanishedSurfaceDoesNotConnect() async {
        let owner = StudyFreezeController()
        let jobs = StudyRemoteJobController()
        let execution = StudyServerJobCoordinator(jobs: jobs)
        var connections = 0
        let environment = StudyOperationEnvironment(current: { nil }, connect: {
            connections += 1
            return nil
        })
        await owner.freezeOnServer(in: environment)
        await execution.run(experimentName: "study", verb: "run", in: environment)
        #expect(connections == 0)
    }
}

/// Every request stays in this protocol; tests never contact an actual runner.
private final class OperationReplyProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var paths: [String] = []
    nonisolated(unsafe) private static var hook: (@MainActor @Sendable () -> Void)?
    static var requests: [String] { lock.withLock { paths } }
    static func reset() { lock.withLock { paths = []; hook = nil } }
    static func setHook(_ value: @escaping @MainActor @Sendable () -> Void) {
        lock.withLock { hook = value }
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let hook = Self.lock.withLock {
            Self.paths.append(request.url!.path)
            return Self.hook
        }
        let reply = PendingReply(handler: self, url: request.url!)
        Task.detached { [hook, reply] in
            await hook?()
            reply.send()
        }
    }
    private struct PendingReply: @unchecked Sendable {
        let handler: OperationReplyProtocol
        let url: URL
        func send() {
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
            handler.client?.urlProtocol(handler, didReceive: response, cacheStoragePolicy: .notAllowed)
            handler.client?.urlProtocol(handler, didLoad: Data(#"{"pipelines":[{"run":"remote-run","stages":[]}]}"#.utf8))
            handler.client?.urlProtocolDidFinishLoading(handler)
        }
    }
    override func stopLoading() {}
}
