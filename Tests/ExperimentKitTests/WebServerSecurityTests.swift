import Foundation
import Testing

@testable import ExperimentKit

/// Guards for the unauthenticated loopback web server: the Origin/Host CSRF
/// check and the project-root containment used for API-writable paths.
struct WebServerSecurityTests {

    @Test func sameOriginLoopbackIsAllowed() {
        #expect(SteerLabWebServer.isLoopbackContext(
            host: "localhost:8080", origin: "http://localhost:8080"))
        #expect(SteerLabWebServer.isLoopbackContext(
            host: "127.0.0.1:8080", origin: "http://127.0.0.1:8080"))
        #expect(SteerLabWebServer.isLoopbackContext(
            host: "[::1]:8080", origin: "http://[::1]:8080"))
    }

    @Test func crossOriginIsRejected() {
        #expect(!SteerLabWebServer.isLoopbackContext(
            host: "localhost:8080", origin: "http://evil.example"))
        // A non-loopback Host (DNS rebinding) is rejected even with no Origin.
        #expect(!SteerLabWebServer.isLoopbackContext(
            host: "steerlab.attacker.test", origin: nil))
    }

    @Test func unparseableOriginIsUnsafe() {
        #expect(!SteerLabWebServer.isLoopbackContext(host: nil, origin: "not a url"))
    }

    @Test func loopbackOriginOnWrongPortIsRejected() {
        // A page served from another local port is still cross-origin even
        // though its host is loopback (shared-node process isolation).
        #expect(!SteerLabWebServer.isLoopbackContext(
            host: "localhost:8080", origin: "http://localhost:9999"))
        #expect(SteerLabWebServer.portOnly("localhost:8080") == 8080)
        #expect(SteerLabWebServer.portOnly("127.0.0.1") == nil)
        #expect(SteerLabWebServer.portOnly("[::1]:8080") == 8080)
    }

    /// Exercises the exact refusal decision `route` makes, so the browser-gate
    /// wiring (not just the pure helper) is covered.
    @Test func routeRefusesBrowserCrossOriginButNotClients() {
        // Browser cross-origin → refused.
        #expect(SteerLabWebServer.isRequestRefused(
            isBrowserRequest: true, host: "localhost:8080", origin: "http://evil.example"))
        // Browser same-origin loopback → allowed.
        #expect(!SteerLabWebServer.isRequestRefused(
            isBrowserRequest: true, host: "localhost:8080", origin: "http://localhost:8080"))
        // Non-browser client (no Origin/Sec-Fetch) → never refused, even with a
        // non-loopback Host, because it can't be a CSRF vector.
        #expect(!SteerLabWebServer.isRequestRefused(
            isBrowserRequest: false, host: "steerlab.attacker.test", origin: nil))
    }

    @Test func hostOnlyStripsPortsAndBrackets() {
        #expect(SteerLabWebServer.hostOnly("localhost:8080") == "localhost")
        #expect(SteerLabWebServer.hostOnly("127.0.0.1") == "127.0.0.1")
        #expect(SteerLabWebServer.hostOnly("[::1]:8080") == "::1")
    }

    @Test func projectFileRejectsTraversalAndAbsolutePaths() {
        // Contained relative paths resolve under the project root.
        let ok = try? VectorCatalog.projectFile("prompts/dev/dev-prompts.jsonl")
        #expect(ok != nil)
        #expect(ok?.path.hasPrefix(VectorCatalog.projectRoot.standardizedFileURL.path) == true)

        #expect((try? VectorCatalog.projectFile("../../../etc/passwd")) == nil)
        #expect((try? VectorCatalog.projectFile("../outside.jsonl")) == nil)
    }
}
