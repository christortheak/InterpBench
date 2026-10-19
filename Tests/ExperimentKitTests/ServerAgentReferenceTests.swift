import Foundation
import Testing

@testable import ExperimentKit
import SteeringKit

/// Saved agents name their vectors workspace-relative
/// (`ModelVariantStore.save` normalizes, and the Python engine's promote
/// writes `relpath`), while a server catalog row's `id` is the server's
/// absolute path; the row also advertises `workspaceRelativeID`. Every place
/// a server workspace matches a catalog row against an agent reference must
/// answer both forms (`RemoteVectorRecord.matches(reference:)`). A literal
/// compare blocked every saved definition as "not in this server's catalog"
/// and left a seeded agent's rows without layer bounds or norm checks.
///
/// The catalog rows here have the Python engine's shape: `id` is
/// `runDirectory/name`, `workspaceRelativeID` is `runs/<run>/<name>`
/// (`catalog.list_vectors`).
@Suite("Server workspace agent references")
struct ServerAgentReferenceTests {

    static let relativeFear = "runs/2026-10-01-sweep/fear"
    static let absoluteFear = "/srv/ws/runs/2026-10-01-sweep/fear"
    static let relativeMLX = "runs/2026-10-02-mlx/joy"
    static let unlisted = "runs/2026-10-01-sweep/anger"
    static let baseModel = "org/model"

    /// A row as `GET /api/vectors` returns it, decoded the way the client
    /// decodes it.
    static func record(
        run: String, name: String, substrate: String = "python-hf-transformers",
        advertisesRelativeID: Bool = true
    ) throws -> RemoteVectorRecord {
        let directory = "/srv/ws/runs/\(run)"
        var row: [String: Any] = [
            "id": "\(directory)/\(name)", "runDirectory": directory,
            "name": name, "concept": name, "modelID": baseModel,
            "layerCount": 8, "hiddenSize": 16, "substrate": substrate,
            "hasResidualNorms": true,
            "normsPerLayer": Array(repeating: 2.0, count: 8),
            "residualNormPerLayer": Array(repeating: 40.0, count: 8),
        ]
        if advertisesRelativeID {
            row["workspaceRelativeID"] = "runs/\(run)/\(name)"
        }
        return try JSONDecoder().decode(
            RemoteVectorRecord.self,
            from: JSONSerialization.data(withJSONObject: row))
    }

    static func agent(
        _ name: String, reference: String, alpha: Double = 0.12,
        mode: InterventionPlan.Mode? = nil
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name, baseModelID: baseModel,
            injections: [
                .init(
                    concept: "fear", vectorArtifactID: reference, layer: 4,
                    alpha: alpha, mode: mode)
            ],
            alphaInNormUnits: true, promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
    }

    // MARK: - Pure rules

    @Test("a catalog row is found by either reference form, and only by its own")
    func catalogLookupAcceptsBothForms() throws {
        let fear = try Self.record(run: "2026-10-01-sweep", name: "fear")
        let anger = try Self.record(run: "2026-10-01-sweep", name: "anger-other")
        let catalog = [anger, fear]

        #expect(ChatService.serverVectorRecord(
            matching: Self.relativeFear, in: catalog)?.id == Self.absoluteFear)
        #expect(ChatService.serverVectorRecord(
            matching: Self.absoluteFear, in: catalog)?.id == Self.absoluteFear)
        // Negative control: a reference no row names.
        #expect(ChatService.serverVectorRecord(matching: Self.unlisted, in: catalog) == nil)
        // An older server that does not advertise the relative form cannot be
        // matched by it; the row stays "kept by path", which is honest.
        let legacy = try Self.record(
            run: "2026-10-01-sweep", name: "fear", advertisesRelativeID: false)
        #expect(ChatService.serverVectorRecord(
            matching: Self.relativeFear, in: [legacy]) == nil)
        #expect(ChatService.serverVectorRecord(
            matching: Self.absoluteFear, in: [legacy]) != nil)
    }

    @Test("the picker tags a seeded row with the slot's own reference")
    func pickerTagKeepsTheSeededReference() throws {
        let fear = try Self.record(run: "2026-10-01-sweep", name: "fear")
        let other = try Self.record(run: "2026-10-01-sweep", name: "calm")
        // The row the seeded slot names carries the slot's spelling, so the
        // picker shows it selected and leaves the reference as it is.
        #expect(ChatService.serverVectorPickerTag(
            for: fear, slotReference: Self.relativeFear) == Self.relativeFear)
        #expect(ChatService.serverVectorPickerTag(
            for: fear, slotReference: Self.absoluteFear) == Self.absoluteFear)
        // Every other row, and every row of an empty slot, is tagged by id.
        #expect(ChatService.serverVectorPickerTag(
            for: other, slotReference: Self.relativeFear) == other.id)
        #expect(ChatService.serverVectorPickerTag(for: fear, slotReference: nil) == fear.id)
    }

    @Test("a promoted strength is recommended for the server row its agent names")
    func serverRecommendationMatchesRelativeReferences() throws {
        let fear = try Self.record(run: "2026-10-01-sweep", name: "fear")
        let agents = [
            Self.agent("fear-promoted", reference: Self.relativeFear, alpha: 0.12),
            // Negative controls: another vector, and an ablation of this one
            // (λ is not an α).
            Self.agent("anger", reference: Self.unlisted, alpha: 0.30),
            Self.agent("fear-ablate", reference: Self.relativeFear, alpha: 1, mode: .ablate),
        ]

        let recommendation = SlotAlphaDefault.recommendation(
            forServerVector: fear, among: agents)
        #expect(recommendation?.alpha == 0.12)
        #expect(recommendation?.agent == "fear-promoted")

        // The local lookup resolves against this Mac's workspace root, which
        // says nothing about the server's tree: it misses the agent.
        #expect(SlotAlphaDefault.recommendation(
            forVectorArtifactID: Self.absoluteFear, among: agents) == nil)

        let unrelated = try Self.record(run: "2026-10-01-sweep", name: "calm")
        #expect(SlotAlphaDefault.recommendation(
            forServerVector: unrelated, among: agents) == nil)
    }

    @Test("the Agent Library finds a saved agent's vector in the server catalog")
    func agentLibraryCountsRelativeReferencesAsPresent() throws {
        let catalog = [
            try Self.record(run: "2026-10-01-sweep", name: "fear")
        ]
        let availability = AgentLibrary.Availability(
            serverVectorIDs: Set(catalog.flatMap(\.referenceForms)))

        #expect(AgentLibrary.missingVectorRefs(
            Self.agent("fear", reference: Self.relativeFear),
            availability: availability).isEmpty)
        // Negative control: the chip still names a vector the server lacks.
        #expect(AgentLibrary.missingVectorRefs(
            Self.agent("anger", reference: Self.unlisted),
            availability: availability) == [Self.unlisted])
    }

    // MARK: - Through ChatService, against a stubbed server

    /// Answers the routes these tests drive. Constant, so concurrently running
    /// tests cannot disturb each other.
    private final class StubServer: URLProtocol, @unchecked Sendable {
        static let catalog = #"""
            {"vectors": [
              {"id": "/srv/ws/runs/2026-10-01-sweep/fear",
               "runDirectory": "/srv/ws/runs/2026-10-01-sweep",
               "name": "fear", "concept": "fear", "modelID": "org/model",
               "layerCount": 8, "hiddenSize": 16,
               "substrate": "python-hf-transformers", "hasResidualNorms": true,
               "normsPerLayer": [2, 2, 2, 2, 2, 2, 2, 2],
               "residualNormPerLayer": [40, 40, 40, 40, 40, 40, 40, 40],
               "workspaceRelativeID": "runs/2026-10-01-sweep/fear"},
              {"id": "/srv/ws/runs/2026-10-01-sweep/calm",
               "runDirectory": "/srv/ws/runs/2026-10-01-sweep",
               "name": "calm", "concept": "calm", "modelID": "org/model",
               "layerCount": 8, "hiddenSize": 16,
               "substrate": "python-hf-transformers",
               "workspaceRelativeID": "runs/2026-10-01-sweep/calm"},
              {"id": "/srv/ws/runs/2026-10-02-mlx/joy",
               "runDirectory": "/srv/ws/runs/2026-10-02-mlx",
               "name": "joy", "concept": "joy", "modelID": "org/model",
               "layerCount": 8, "hiddenSize": 16, "substrate": "swift-mlx",
               "workspaceRelativeID": "runs/2026-10-02-mlx/joy"}
            ]}
            """#

        static func detail(reference: String, concept: String = "fear") -> String {
            """
            {"hash": "sha256-agent", "variant": {
              "name": "server-fear", "baseModelID": "org/model",
              "alphaInNormUnits": true, "promptMode": "chatAssistant",
              "qwenThinkingEnabled": false, "temperature": 0, "systemPrompt": "",
              "injections": [{"concept": "\(concept)", "vectorArtifactID": "\(reference)",
                              "layer": 4, "alpha": 0.12}]}}
            """
        }

        static func body(for url: URL) -> String? {
            switch url.path {
            case "/api/vectors":
                return catalog
            case "/api/variant/detail":
                let path = URLComponents(url: url, resolvingAgainstBaseURL: false)?
                    .queryItems?.first { $0.name == "path" }?.value
                switch path {
                case "agents/listed.json": return detail(reference: relativeFear)
                case "agents/unlisted.json": return detail(reference: unlisted)
                // A promotion names the experiment's concept, which need not
                // be the catalog row's.
                case "agents/renamed.json":
                    return detail(reference: relativeFear, concept: "fear-experiment")
                default: return nil
                }
            default:
                return nil
            }
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for r: URLRequest) -> URLRequest { r }
        override func startLoading() {
            let body = request.url.flatMap(Self.body(for:))
            let response = HTTPURLResponse(
                url: request.url!, statusCode: body == nil ? 404 : 200,
                httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data((body ?? "{}").utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    /// A server workspace whose catalog has been fetched and whose inventory
    /// holds the agents' base model.
    @MainActor
    private func serverWorkspace(
        _ name: String
    ) async throws -> (ChatService, ClusterConnectionStore, () -> Void) {
        let suite = "ServerAgentReference.\(name).\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        let store = clusterStore(defaults: defaults)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubServer.self]
        store.clientSessionOverride = URLSession(configuration: configuration)
        let entry = store.addServer(name: "stub", urlString: "http://server.test")
        store.activeWorkspace = .server(entry.id)
        store.remoteState = RemoteState(
            models: [Self.baseModel], loadedModel: nil, loadedRevision: nil,
            device: nil, isBusy: false, loadedModels: [], jobs: [])
        let chat = ChatService(cluster: store)
        #expect(await chat.catalog.refreshRemoteVectors())
        #expect(chat.catalog.remoteVectors.count == 3)
        return (chat, store, { defaults.removePersistentDomain(forName: suite) })
    }

    @MainActor
    @Test("a saved definition with relative refs applies, steers, and recommends on a server")
    func savedDefinitionAppliesOnServer() async throws {
        let raw = FileManager.default.temporaryDirectory
            .appending(component: "server-agent-refs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
        let temp = raw.resolvingSymlinksInPath()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }

        let (chat, _, cleanup) = try await serverWorkspace("definition")
        defer { cleanup() }
        try ModelVariantStore.save(Self.agent("fear-agent", reference: Self.relativeFear))
        try ModelVariantStore.save(Self.agent("anger-agent", reference: Self.unlisted))
        try ModelVariantStore.save(Self.agent("joy-mlx-agent", reference: Self.relativeMLX))
        chat.fineTuning.refresh()
        func saved(_ name: String) throws -> ModelVariantRecord {
            try #require(chat.fineTuning.variants.first { $0.artifact.name == name })
        }
        let fear = try saved("fear-agent")
        // What the save path stores, so the test exercises the real shape.
        #expect(fear.artifact.injections.first?.vectorArtifactID == Self.relativeFear)

        // Applicability: the Playground picker, the Agent Library chips and
        // filter, and the robustness check all read this.
        #expect(chat.serverApplicability(of: fear.artifact) == .applicable)
        #expect(chat.playgroundAgentPickerRows.first {
            $0.selection == .localDefinition(id: fear.id)
        }?.isSelectable == true)
        // Negative controls: a vector the server lacks, and a vector the
        // server lists but cannot steer with (an MLX extraction).
        #expect(chat.serverApplicability(of: try saved("anger-agent").artifact)
            == .blocked(reasons: ["1 vector ref not in this server's catalog"]))
        #expect(chat.serverApplicability(of: try saved("joy-mlx-agent").artifact)
            == .blocked(reasons: ["1 vector ref not in this server's catalog"]))

        chat.applyLocalDefinitionToServerSteering(fear)
        #expect(chat.playgroundAgentSelection == .localDefinition(id: fear.id))
        let slot = try #require(chat.slots.first)
        // The row keeps the agent's own reference; the server resolves it.
        #expect(slot.vectorID == Self.relativeFear)
        #expect(chat.serverVectorRecord(for: slot)?.id == Self.absoluteFear)
        #expect(chat.serverControlState().slots.map(\.vectorPath) == [Self.relativeFear])
        #expect(chat.maxLayerIndex(for: slot) == 7)
        #expect(chat.normUnitsAvailable)

        // The α default reads the agent library through the server row.
        let facts = try #require(chat.alphaDefaultFacts(slotID: slot.id))
        #expect(facts.recommendedAlpha == 0.12)
        #expect(facts.recommendedFromAgent == "fear-agent")
    }

    @MainActor
    @Test("a seeded server agent's rows resolve against the catalog")
    func seededServerAgentResolves() async throws {
        let (chat, _, cleanup) = try await serverWorkspace("seeded")
        defer { cleanup() }

        await chat.applyRemoteVariantToSteering(path: "agents/listed.json")
        let slot = try #require(chat.slots.first)
        #expect(slot.vectorID == Self.relativeFear)
        #expect(chat.serverVectorRecord(for: slot)?.id == Self.absoluteFear)
        #expect(chat.serverSlotConcept(for: slot) == "fear")
        #expect(chat.errorMessage?.contains("kept by path") != true)
        #expect(chat.maxLayerIndex(for: slot) == 7)
        #expect(chat.normUnitsAvailable)
        #expect(chat.serverInjectionPreview(for: slot) != nil)
        // Sending is unchanged: the untouched seed still sends the stored
        // agent, and a composed spec carries the agent's own reference.
        #expect(chat.serverSendModeCaption.contains("controls match the seeded agent"))
        #expect(chat.serverControlResolution().unresolvedVectorIDs.isEmpty)
        #expect(chat.serverControlState().slots.map(\.vectorPath) == [Self.relativeFear])

        // A new box defaults to the next vector the seeded rows do not use.
        chat.addSlot()
        #expect(chat.slots.last?.vectorID == "/srv/ws/runs/2026-10-01-sweep/calm")
    }

    @MainActor
    @Test("a seeded row keeps the concept its agent declares")
    func seededConceptIsTheAgents() async throws {
        let (chat, _, cleanup) = try await serverWorkspace("renamed")
        defer { cleanup() }

        await chat.applyRemoteVariantToSteering(path: "agents/renamed.json")
        let slot = try #require(chat.slots.first)
        // The row resolves against the catalog for its bounds and norms...
        #expect(chat.serverVectorRecord(for: slot)?.concept == "fear")
        // ...but composes with the agent's own concept, so the composed spec
        // does not depend on whether the listing is current.
        #expect(chat.serverControlState().slots.map(\.concept) == ["fear-experiment"])
        #expect(chat.serverSendModeCaption.contains("controls match the seeded agent"))
    }

    @MainActor
    @Test("a seeded server agent naming an unlisted vector is still reported")
    func seededServerAgentWithUnlistedVectorIsReported() async throws {
        let (chat, _, cleanup) = try await serverWorkspace("unlisted")
        defer { cleanup() }

        await chat.applyRemoteVariantToSteering(path: "agents/unlisted.json")
        let slot = try #require(chat.slots.first)
        #expect(slot.vectorID == Self.unlisted)
        #expect(chat.serverVectorRecord(for: slot) == nil)
        #expect(chat.errorMessage?.contains(
            "1 vector not in the server catalog listing (kept by path)") == true)
        // Kept through the seeded concept, so the spec still round-trips.
        #expect(chat.serverControlState().slots.map(\.vectorPath) == [Self.unlisted])
    }
}
