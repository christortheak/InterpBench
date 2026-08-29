import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// Cancelling an in-flight model load (field incident 2026-08-29).
//
// Selecting an uncached 27B on the local engine silently began a ~55 GB hub
// download that held the only resident-model slot: unload answered
// `unloaded: 0`, no cancel existed anywhere, and the engine had to be
// SIGTERMed. These tests pin the app half of the repair: the capability
// gates, the preflight-before-download confirmation, the new client routes,
// and the qualify argv's root pin (the `metadataRoot: /.steerlab` ghost).
// =============================================================================

@Suite struct ModelLoadCancelCapabilityTests {

    @Test func loadCancelGateReadsChatLoadCancel() throws {
        let caps = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"chat": {"loadCancel": true}}"#.utf8))
        #expect(caps.supportsLoadCancel)
        // Older servers never grow the affordance by inference.
        let older = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion": "0.9"}"#.utf8))
        #expect(!older.supportsLoadCancel)
    }

    @Test func loadPreflightGateReadsChatLoadPreflight() throws {
        let caps = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"chat": {"loadPreflight": true}}"#.utf8))
        #expect(caps.supportsLoadPreflight)
        let older = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"chat": {"loadStream": true}}"#.utf8))
        #expect(!older.supportsLoadPreflight)
    }

    @Test func loadedModelRowsDecodeTheLoadingSlotShape() throws {
        // The registry's loading placeholder: busy, loading, with the live
        // phase text (download progress) and the cancel flag.
        let json = #"""
        {"modelID": "google/gemma-3-27b-it", "revision": null,
         "device": "mps", "dtype": "auto", "busy": true, "loading": true,
         "loadPhase": "downloading 'google/gemma-3-27b-it' from the hub — cache holds 5.2 of ~55.4 GB (9%)",
         "cancelRequested": false, "numLayers": null}
        """#
        let row = try JSONDecoder().decode(
            RemoteLoadedModel.self, from: Data(json.utf8))
        #expect(row.loading == true)
        #expect(row.cancelRequested == false)
        #expect(row.loadPhase?.contains("cache holds 5.2 of ~55.4 GB") == true)
        // An older server's row (no loading fields) still decodes.
        let ready = try JSONDecoder().decode(
            RemoteLoadedModel.self,
            from: Data(#"{"modelID": "org/tiny", "busy": false}"#.utf8))
        #expect(ready.loading == nil && ready.loadPhase == nil)
    }
}

@Suite struct DownloadConfirmationTextTests {

    @Test func confirmationNamesTheModelAndTheSize() {
        let text = ChatService.PendingRemoteDownload.confirmationText(
            model: "google/gemma-3-27b-it", bytes: 55_400_000_000)
        #expect(text.contains("google/gemma-3-27b-it"))
        #expect(text.contains("~55.4 GB"))
        #expect(text.contains("resident-model slot"))
    }

    @Test func unknownSizeIsSpokenNeverRoundedToReassurance() {
        let text = ChatService.PendingRemoteDownload.confirmationText(
            model: "org/mystery", bytes: nil)
        #expect(text.contains("an unknown amount"))
        #expect(!text.contains("GB —"))
    }
}

@Suite struct QualifyArgvRootPinTests {

    /// The 2026-08-29 ghost check: the qualify subprocess inherited the
    /// app's cwd (`/` from Finder) and no STEERLAB_ROOT, so its profile
    /// check probed `/.steerlab` — a verdict about a deployment nobody was
    /// running. The argv now pins the SAME root the serve step passes.
    @Test func qualifyArgvPinsTheServedWorkspaceRoot() throws {
        let engine = CodeResources.EngineRoot(
            root: URL(filePath: "/fixture/Engine"), stamp: nil)
        let argv = LocalEngineProvisioner.qualifyArgv(
            source: .engineRoot(engine),
            workspaceRoot: URL(filePath: "/data/ws"),
            reportFile: URL(filePath: "/tmp/report.json"))
        #expect(argv[0] == "/fixture/Engine/Server/.venv.nosync/bin/python")
        #expect(argv.contains("site") && argv.contains("qualify"))
        let rootIndex = try #require(argv.firstIndex(of: "--root"))
        #expect(argv[rootIndex + 1] == "/data/ws")
        let jsonIndex = try #require(argv.firstIndex(of: "--json"))
        #expect(argv[jsonIndex + 1] == "/tmp/report.json")
    }
}

@Suite(.serialized) struct ModelLoadCancelRouteTests {

    @Test func preflightSendsModelAndDecodesTheAnswer() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/models/preflight")
            #expect(request.httpMethod == "GET")
            let query = request.url?.query ?? ""
            #expect(query.contains("model=google%2Fgemma-3-27b-it")
                || query.contains("model=google/gemma-3-27b-it"))
            return (
                Data(#"""
                {"modelID": "google/gemma-3-27b-it", "revision": null,
                 "cached": false, "residency": null,
                 "downloadBytes": 55400000000}
                """#.utf8),
                200
            )
        }
        let preflight = try await client.modelLoadPreflight("google/gemma-3-27b-it")
        #expect(!preflight.cached)
        #expect(preflight.downloadBytes == 55_400_000_000)
    }

    @Test func cancelPostsTheRouteAndDecodesTheIdentities() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/models/load/cancel")
            #expect(request.httpMethod == "POST")
            return (
                Data(#"""
                {"ok": true,
                 "cancelRequested": [{"modelID": "google/gemma-3-27b-it",
                                      "revision": null, "device": "mps"}],
                 "note": "cancellation is cooperative: a download stops within seconds"}
                """#.utf8),
                200
            )
        }
        let result = try await client.cancelModelLoad()
        #expect(result.cancelRequested.map(\.modelID) == ["google/gemma-3-27b-it"])
        #expect(result.note?.contains("cooperative") == true)
    }

    @Test func unloadSurfacesTheLoadingHintTheIncidentLacked() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/models/unload")
            return (
                Data(#"""
                {"ok": true, "unloaded": 0,
                 "loading": ["google/gemma-3-27b-it"],
                 "hint": "a model load in flight holds its slot and cannot be unloaded — cancel it with POST /api/models/load/cancel"}
                """#.utf8),
                200
            )
        }
        let result = try await client.unloadModels()
        #expect(result.unloaded == 0)
        #expect(result.loading == ["google/gemma-3-27b-it"])
        #expect(result.hint?.contains("POST /api/models/load/cancel") == true)
    }

    @Test func unloadWithoutHintStillDecodes() async throws {
        // Older-server shape: {"ok": true, "unloaded": 1} and nothing else.
        let client = Self.client { _ in
            (Data(#"{"ok": true, "unloaded": 1}"#.utf8), 200)
        }
        let result = try await client.unloadModels("org/tiny")
        #expect(result.unloaded == 1)
        #expect(result.loading == nil && result.hint == nil)
    }

    private static func client(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> ClusterClient {
        MockModelRoutesURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockModelRoutesURLProtocol.self]
        return ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: URL(string: "http://server.test")!),
            session: URLSession(configuration: configuration))
    }
}

private final class MockModelRoutesURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler:
        (@Sendable (URLRequest) throws -> (Data, Int))?

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else { return }
        do {
            let (data, status) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response,
                                cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
