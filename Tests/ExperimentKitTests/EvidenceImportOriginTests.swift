import Foundation
import Testing

@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct EvidenceImportOriginTests {
    private func store() throws -> ClusterConnectionStore {
        let defaults = try #require(UserDefaults(suiteName: "evidence-origin-\(UUID().uuidString)"))
        return clusterStore(defaults: defaults)
    }

    @Test func sshIdentitySurvivesTunnelChangesButDistinguishesRemoteEndpoints() throws {
        let store = try store()
        var first = ClusterSiteProfile.exampleCluster
        first.transport = .ssh(host: "researcher@first.invalid", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        let entry = store.addSite(first)
        store.activeWorkspace = .server(entry.id)
        store.applyServerWorkspaceSwitch(info: .init(root: "/remote"))
        let before = try #require(store.evidenceImportOrigin)
        let connection = store.connectionProfile
        store.noteTunnelLocalPort(19321)
        // Updating transport metadata clears connection observations until refresh.
        store.applyServerWorkspaceSwitch(info: .init(root: "/remote"))
        #expect(store.evidenceImportOrigin == before)
        #expect(store.connectionProfile != connection)
        var second = first
        second.transport = .ssh(host: "researcher@second.invalid", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        let other = store.addSite(second)
        store.activeWorkspace = .server(other.id)
        store.noteTunnelLocalPort(19321)
        store.applyServerWorkspaceSwitch(info: .init(root: "/remote"))
        #expect(store.connectionProfile?.baseURL == URL(string: "http://127.0.0.1:19321"))
        #expect(store.evidenceImportOrigin != before)
        let encoded = try JSONEncoder().encode(before)
        #expect(try JSONDecoder().decode(EvidenceImportOrigin.self, from: encoded) == before)
        #expect(!String(decoding: encoded, as: UTF8.self).contains("19321"))
    }

    @Test func importerRegistrationFollowsTheLocalWorkspace() throws {
        let store = try store()
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = store.registerEvidenceAutoImport(workspaceRoot: root.appending(component: "first"))
        let second = store.registerEvidenceAutoImport(workspaceRoot: root.appending(component: "second"))
        defer { first.stopPolling(); second.stopPolling() }
        #expect(first !== second)
        #expect(first.workspaceRoot.lastPathComponent == "first")
        #expect(second.workspaceRoot.lastPathComponent == "second")
        #expect(store.registerEvidenceAutoImport(workspaceRoot: second.workspaceRoot) === second)
    }

    @Test(arguments: [false, true])
    func lateConnectionResponsesCannotPopulateAnotherServer(handshake: Bool) async throws {
        let store = try store()
        let first = store.addServer(name: "First", urlString: "http://first.invalid:19322")
        let second = store.addServer(name: "Second", urlString: "http://second.invalid:19323")
        store.activeWorkspace = .server(first.id)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OriginReplyProtocol.self]
        let session = URLSession(configuration: configuration)
        store.clientSessionOverride = session
        OriginReplyProtocol.setHook { store.activeWorkspace = .server(second.id) }
        defer { OriginReplyProtocol.setHook(nil); session.invalidateAndCancel() }
        if handshake {
            #expect(await store.connect() == false)
        } else {
            await store.refreshRemoteState()
        }
        #expect(store.activeWorkspace == .server(second.id))
        #expect(store.remoteInfo == nil)
        #expect(store.remoteState == nil)
        #expect(store.capabilities == nil)
        #expect(store.remoteVariants.isEmpty)
    }
}

private final class OriginReplyProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var hook: (@MainActor @Sendable () -> Void)?

    static func setHook(_ value: (@MainActor @Sendable () -> Void)?) {
        lock.withLock { hook = value }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let hook = Self.lock.withLock { Self.hook }
        let reply = PendingReply(handler: self, url: request.url!)
        Task.detached { [hook, reply] in
            await hook?()
            reply.send()
        }
    }

    /// A single response owns this test-only callback reference. It crosses
    /// to one detached task after loading starts; no other code sends a reply.
    private struct PendingReply: @unchecked Sendable {
        let handler: OriginReplyProtocol
        let url: URL

        func send() {
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
            let data = Data(#"{"root":"/old-root","serverVersion":"fixture","models":[],"isBusy":false,"loadedModels":[],"jobs":[],"variants":[]}"#.utf8)
            handler.client?.urlProtocol(handler, didReceive: response, cacheStoragePolicy: .notAllowed)
            handler.client?.urlProtocol(handler, didLoad: data)
            handler.client?.urlProtocolDidFinishLoading(handler)
        }
    }
    override func stopLoading() {}
}
