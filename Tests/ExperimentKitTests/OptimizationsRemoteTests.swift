import Foundation
import Testing

@testable import ExperimentKit

/// Server-workspace Optimizations plumbing: remote sweep-run discovery reuses the
/// SAME name matcher and parsers as local discovery; the server promote route
/// surfaces its self-naming 400 refusals verbatim.
@Suite(.serialized) struct OptimizationsRemoteTests {

    // MARK: Remote sweep-run discovery

    private func record(
        id: String, files: [String] = ["sweep.csv", "recommendations.json"]
    ) -> RemoteRunRecord {
        RemoteRunRecord(
            id: id, path: "/srv/ws/runs/\(id)", task: nil,
            hasReport: false, hasGenerations: false, hasCosineMatrix: false,
            vectorNames: [], files: files)
    }

    @Test func newestRemoteSweepRunRecordPicksNewestMatch() {
        let records = [
            record(id: "20260101T000000000-exp-demo-sweep"),
            record(id: "20260103T000000000-exp-demo-sweep-2"),
            record(id: "20260102T000000000-exp-demo-sweep"),
            record(id: "20260104T000000000-exp-other-sweep"),
            record(id: "20260105T000000000-exp-demo-run"),
        ]
        let newest = SweepRunCatalog.newestRemoteSweepRunRecord(
            experiment: "demo", in: records)
        #expect(newest?.id == "20260103T000000000-exp-demo-sweep-2")
    }

    @Test func newestRemoteSweepRunRecordRequiresSweepCSV() {
        let records = [
            record(id: "20260101T000000000-exp-demo-sweep", files: ["report.json"])
        ]
        #expect(
            SweepRunCatalog.newestRemoteSweepRunRecord(
                experiment: "demo", in: records) == nil)
    }

    @Test func newestRemoteSweepRunRecordNeverCapturesNameShapedSiblings() {
        // Experiment "x" must not capture experiment "x-sweep"'s runs and
        // vice versa — same strictness as the local matcher (shared code).
        let records = [record(id: "20260101T000000000-exp-x-sweep-sweep")]
        #expect(
            SweepRunCatalog.newestRemoteSweepRunRecord(
                experiment: "x", in: records) == nil)
        #expect(
            SweepRunCatalog.newestRemoteSweepRunRecord(
                experiment: "x-sweep", in: records)?.id
                == "20260101T000000000-exp-x-sweep-sweep")
    }

    @Test func remoteSweepRunParsesThroughSharedEntryPoints() throws {
        let csv = """
            concept,layer,alpha,markerDensity,distinct2,batteryAccuracy
            fear,-1,0,0.01,0.8,1.0
            fear,12,0.4,0.31,0.7,0.9
            """
        let recommendations = Data(#"{"fear": "no cell passed the constraints"}"#.utf8)
        let run = try SweepRunCatalog.remoteSweepRun(
            runPath: "/srv/ws/runs/20260101T000000000-exp-demo-sweep",
            csvText: csv,
            recommendationsData: recommendations)
        #expect(run.runName == "20260101T000000000-exp-demo-sweep")
        #expect(run.rows.count == 2)
        #expect(run.rows[0].isBaseline)
        #expect(run.rows[1].layer == 12)
        #expect(run.recommendations["fear"] == .failure("no cell passed the constraints"))
    }

    @Test func remoteSweepRunToleratesMissingRecommendations() throws {
        let csv = """
            concept,layer,alpha,markerDensity,distinct2,batteryAccuracy
            fear,12,0.4,0.31,0.7,0.9
            """
        let run = try SweepRunCatalog.remoteSweepRun(
            runPath: "/srv/ws/runs/20260101T000000000-exp-demo-sweep",
            csvText: csv,
            recommendationsData: nil)
        #expect(run.recommendations.isEmpty)
    }

    // MARK: Server experiment summaries (selection provenance decode)

    @Test func remoteExperimentRecordDecodesConditionSelection() throws {
        let json = """
            {"experiments": [{
                "name": "demo", "status": "frozen", "modelID": "org/m",
                "concepts": [{"name": "fear", "hash": "abc123def456"}],
                "conditions": [
                    {"name": "baseline", "bandWidth": 1, "normUnits": true, "slots": []},
                    {"name": "fear-recommended", "bandWidth": 1, "normUnits": true,
                     "slots": [{"concept": "fear", "layer": 12, "alpha": 0.4}],
                     "selection": {
                        "sweepRun": "20260101T000000000-exp-demo-sweep",
                        "criterion": {"objective": {"metric": "markerDensity"}},
                        "devPromptsHash": "feedfacefeedface",
                        "winningCell": {"layer": 12, "alpha": 0.4},
                        "metrics": {"markerDensity": 0.31}}},
                    {"name": "fear-control", "bandWidth": 1, "normUnits": true,
                     "slots": [], "controlType": "randomMatchedNorm"}
                ]},
                {"name": "broken", "status": "unreadable", "error": "boom"}]}
            """
        struct Response: Decodable { var experiments: [RemoteExperimentRecord] }
        let response = try JSONDecoder().decode(Response.self, from: Data(json.utf8))
        #expect(response.experiments.count == 2)
        let demo = response.experiments[0]
        let selections = Dictionary(
            (demo.conditions ?? []).compactMap { c in c.selection.map { (c.name, $0) } },
            uniquingKeysWith: { first, _ in first })
        #expect(selections.count == 1)
        let provenance = try #require(selections["fear-recommended"])
        #expect(provenance.winningCell.layer == 12)
        #expect(provenance.sweepRun == "20260101T000000000-exp-demo-sweep")
        #expect(provenance.criterion.objective?.metric == "markerDensity")
        #expect(demo.conditions?.last?.controlType == "randomMatchedNorm")
        // Unreadable manifests still list (name + error only).
        #expect(response.experiments[1].conditions == nil)
        #expect(response.experiments[1].error == "boom")
    }

    // MARK: Declared sweep spec decode (top-level "sweep" key, verbatim
    // manifest SweepSpec JSON — pinned cross-engine contract)

    @Test func remoteExperimentRecordDecodesDeclaredSweepSpec() throws {
        let json = """
            {"experiments": [
              {"name": "declared", "status": "draft", "modelID": "org/m",
               "concepts": [{"name": "fear"}],
               "conditions": [],
               "sweep": {
                 "layerFractions": [0.35, 0.5, 0.65],
                 "alphas": [0.1, 0.2, 0.4],
                 "devPromptsFile": "prompts/dev/dev-prompts.jsonl",
                 "batteryFile": "prompts/batteries/basic.jsonl",
                 "maxTokens": 80,
                 "selection": {
                   "objective": {"metric": "markerDensity"},
                   "constraints": {"capabilityTolerance": 0.2,
                                   "coherenceFloor": 0.5},
                   "controls": {"matchedNormRandomMargin": 0.05}}}},
              {"name": "undeclared", "status": "frozen", "modelID": "org/m"}]}
            """
        struct Response: Decodable { var experiments: [RemoteExperimentRecord] }
        let response = try JSONDecoder().decode(Response.self, from: Data(json.utf8))
        let declared = try #require(response.experiments.first)
        let sweep = try #require(declared.sweep)
        #expect(sweep.layerFractions == [0.35, 0.5, 0.65])
        #expect(sweep.alphas == [0.1, 0.2, 0.4])
        #expect(sweep.devPromptsFile == "prompts/dev/dev-prompts.jsonl")
        #expect(sweep.batteryFile == "prompts/batteries/basic.jsonl")
        #expect(sweep.maxTokens == 80)
        #expect(sweep.selection?.objective?.metric == "markerDensity")
        #expect(sweep.selection?.constraints?.capabilityTolerance == 0.2)
        #expect(sweep.selection?.constraints?.coherenceFloor == 0.5)
        #expect(sweep.selection?.controls?.matchedNormRandomMargin == 0.05)
        // Older servers omit the key entirely: nil, never a decode failure.
        #expect(response.experiments.last?.sweep == nil)
    }

    @Test func remoteExperimentRecordDecodesSweepWithoutSelection() throws {
        // The selection block is optional inside the spec too — a declared
        // grid with the historical default criterion travels as sweep-only.
        let json = """
            {"name": "grid-only", "status": "draft",
             "sweep": {"layerFractions": [0.5], "alphas": [0.2],
                       "devPromptsFile": "d.jsonl", "batteryFile": "b.jsonl",
                       "maxTokens": 64}}
            """
        let record = try JSONDecoder().decode(
            RemoteExperimentRecord.self, from: Data(json.utf8))
        let sweep = try #require(record.sweep)
        #expect(sweep.selection == nil)
        #expect(sweep.layerFractions == [0.5])
        #expect(sweep.maxTokens == 64)
    }

    // MARK: Evidence-import eligibility (recent-jobs rows)

    @Test @MainActor func evidenceImportOfferedOnlyForCompletedRunJobs() {
        // Completed run-verb jobs qualify — direct, bundled, or listed by
        // the server as a study-submit kind.
        #expect(ExperimentPanel.jobOffersEvidenceImport(verb: "run", state: "succeeded"))
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            verb: "run (bundle)", state: "succeeded"))
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            verb: "study-submit", state: "succeeded"))
        // Non-run verbs never offer it, whatever their state.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(verb: "sweep", state: "succeeded"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            verb: "verify (bundle)", state: "succeeded"))
        // Unfinished, failed, and dry-run ("prepared") jobs never offer it.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(verb: "run", state: "running"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(verb: "run", state: "failed"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            verb: "run (bundle)", state: "prepared"))
    }

    @Test @MainActor func evidenceImportKindTwinMatchesServerJobKinds() {
        // The Compute panel's rows carry the server's raw RemoteJobRecord
        // kinds: direct run verbs list as "experiment:run"; study
        // submissions as "study-submit" / "study-submit-bundle".
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            kind: "experiment:run", state: "succeeded"))
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            kind: "study-submit", state: "succeeded"))
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            kind: "study-submit-bundle", state: "succeeded"))
        // The daemon's startup resume of an orphaned chain packages the
        // chain's evidence on success (2026-08-06) — its results flow home
        // through the same auto-import.
        #expect(ExperimentPanel.jobOffersEvidenceImport(
            kind: "pipeline-orphan-reconcile", state: "succeeded"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "pipeline-orphan-reconcile", state: "failed"))
        // Non-run experiment verbs and non-study kinds never qualify.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "experiment:sweep", state: "succeeded"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "experiment:validate", state: "succeeded"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "concept-extract:fear", state: "succeeded"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "gemmascope", state: "succeeded"))
        // Unfinished, failed, and dry-run ("prepared") jobs never offer it.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "experiment:run", state: "running"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "study-submit", state: "failed"))
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "study-submit-bundle", state: "prepared"))
    }

    // MARK: Server promote

    @Test func promoteExperimentEncodesBodyAndDecodesMintedAgent() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/experiment/demo/promote")
                #expect(request.httpMethod == "POST")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["concept"] as? String == "fear")
                let cell = try #require(object["cell"] as? [String: Any])
                #expect(cell["layer"] as? Int == 12)
                #expect(object["overrideReason"] as? String == "pilot follow-up")
                let payload = """
                    {"variant": {"name": "demo-fear-agent", "baseModelID": "org/m",
                     "promotion": {"experiment": "demo", "experimentHash": "h",
                       "promotedAt": "2026-07-07T00:00:00Z",
                       "promotedBy": "manualOverride",
                       "overrideReason": "pilot follow-up",
                       "substrate": "python-hf-transformers",
                       "appVersion": "steerlab-server test"}},
                     "path": "runs/x/variant.json", "runDirectory": "runs/x"}
                    """
                return (Data(payload.utf8), 200)
            })
        let minted = try await client.promoteExperiment(
            name: "demo", concept: "fear",
            cell: (layer: 12, alpha: 0.4), overrideReason: "pilot follow-up")
        #expect(minted.variant.name == "demo-fear-agent")
        #expect(minted.variant.promotion?.promotedBy == "manualOverride")
        #expect(minted.variant.promotion?.overrideReason == "pilot follow-up")
    }

    @Test func promoteExperimentSurfaces400DetailVerbatim() async throws {
        let refusal = "no sweep-selected recommendation for 'fear' in 'demo' "
            + "— run 'experiment sweep' first"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(#"{"detail": "\#(refusal)"}"#.utf8), 400)
            })
        do {
            _ = try await client.promoteExperiment(name: "demo", concept: "fear")
            Issue.record("expected a thrown refusal")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(let code, let text) = error else {
                Issue.record("unexpected error case: \(error)")
                return
            }
            #expect(code == 400)
            #expect(text == refusal)  // the self-naming refusal, not raw JSON
        }
    }

    @Test func unwrappingDetailPassesThroughNonJSONBodies() {
        let unwrapped = ClusterClient.unwrappingDetail(
            .badResponse(500, "Internal Server Error"))
        guard case .badResponse(let code, let text) = unwrapped else {
            Issue.record("unexpected case")
            return
        }
        #expect(code == 500)
        #expect(text == "Internal Server Error")
    }

    @Test func unwrappingDetailFormatsStructuredConstraintDetails() {
        // Chat-template constraint 400s carry an OBJECT detail
        // ({message, constraint, modelID}) — surfaced as one readable line,
        // never raw JSON.
        let body = #"{"detail": {"message": "Conversation roles must alternate"#
            + #" user/assistant/user/assistant/...", "constraint": "roleOrder","#
            + #" "modelID": "mlx-community/gemma-3-4b-it-4bit"}}"#
        let unwrapped = ClusterClient.unwrappingDetail(.badResponse(400, body))
        guard case .badResponse(let code, let text) = unwrapped else {
            Issue.record("unexpected case")
            return
        }
        #expect(code == 400)
        #expect(text.contains("Conversation roles must alternate"))
        #expect(text.contains("roleOrder"))
        #expect(text.contains("gemma-3-4b-it-4bit"))
        #expect(!text.contains("{"), "no raw JSON in the surfaced refusal")
    }

    @Test func runFilePercentEncodesAndReturnsRawBytes() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/runs/20260101T000000000-exp-demo-sweep/file")
                #expect(request.url?.query()?.contains("name=sweep.csv") == true)
                return (Data("concept,layer".utf8), 200)
            })
        let data = try await client.runFile(
            runID: "20260101T000000000-exp-demo-sweep", name: "sweep.csv")
        #expect(String(decoding: data, as: UTF8.self) == "concept,layer")
    }

    // MARK: Mock transport (file-private twin of ClusterClientTests')

    private static func session(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> URLSession {
        OptimizationsMockURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OptimizationsMockURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private static func bodyData(from request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }
        guard let stream = request.httpBodyStream else {
            return nil
        }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let count = stream.read(buffer, maxLength: bufferSize)
            if count > 0 {
                data.append(buffer, count: count)
            } else {
                break
            }
        }
        return data
    }
}

private final class OptimizationsMockURLProtocol: URLProtocol, @unchecked Sendable {
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
                url: request.url!,
                statusCode: status,
                httpVersion: "HTTP/1.1",
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
