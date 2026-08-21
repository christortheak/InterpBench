import Foundation
import Testing

@testable import ExperimentKit

/// The Playground agent selector's contract (bug fix, 2026-07-13):
///
/// 1. SELECTOR SOURCE RULE — a server workspace lists the ACTIVE server's
///    stored agents PLUS every local agent definition as an inline-spec
///    entry (manually-created agents are local definitions, and the
///    historical picker listed only `cluster.remoteVariants`, so a manual
///    agent was visible in the Agents tab but silently absent from the
///    Playground selector). A Local workspace lists local definitions.
/// 2. VISIBLE EXCLUSION — an agent the workspace cannot run right now stays
///    LISTED as a disabled row that names why ("built for X — load it to
///    use"), never silently filtered out.
/// 3. HANDOFF SELECTS — the Agents tab's Chat action routes through the same
///    apply functions the picker uses, so it both seeds the controls AND
///    shows as selected (`playgroundAgentSelection`); the historical path
///    seeded the controls while the selector showed "None".
struct PlaygroundAgentPickerTests {

    // MARK: - Pure exclusion rules

    @Test func serverExclusionNilWhenBaseInstalled() {
        #expect(
            ChatService.serverAgentExclusion(
                baseModelID: "org/base", installedServerModels: ["org/base", "org/other"])
                == nil)
    }

    @Test func serverExclusionNilWhenInventoryUnknown() {
        // Before connect the inventory is empty — unknown must never invent
        // exclusions (the row stays selectable; the fetch is the backstop).
        #expect(
            ChatService.serverAgentExclusion(
                baseModelID: "org/base", installedServerModels: []) == nil)
    }

    @Test func serverExclusionNamesTheMissingModel() {
        let reason = ChatService.serverAgentExclusion(
            baseModelID: "org/base", installedServerModels: ["org/other"])
        #expect(reason?.contains("org/base") == true)
        #expect(reason?.contains("not installed") == true)
    }

    @Test func localExclusionNilWhenModelLoaded() {
        #expect(
            ChatService.localAgentExclusion(
                baseModelID: "org/base", loadedModelID: "org/base") == nil)
    }

    @Test func localExclusionNamesTheModelToLoad() {
        for loaded in [nil, "org/other"] {
            let reason = ChatService.localAgentExclusion(
                baseModelID: "org/base", loadedModelID: loaded)
            #expect(reason?.contains("org/base") == true)
            #expect(reason?.contains("load it to use") == true)
        }
    }

    // MARK: - Selector source rule (pure rows rule)

    private func remoteAgent(name: String, base: String, path: String) -> RemoteVariantRecord {
        RemoteVariantRecord(
            name: name, baseModelID: base, path: path, injections: 0, adapters: 0)
    }

    @Test func serverWorkspaceListsServerAgentsAndLocalDefinitions() {
        let rows = ChatService.agentPickerRows(
            workspaceIsServer: true,
            serverAgents: [
                remoteAgent(name: "stored-ok", base: "org/base", path: "/srv/a.json"),
                remoteAgent(name: "stored-foreign", base: "org/gone", path: "/srv/b.json"),
            ],
            installedServerModels: ["org/base"],
            localDefinitions: [
                .init(id: "runs/mv-1", name: "manual-agent", baseModelID: "org/base"),
                .init(
                    id: "runs/mv-2", name: "blocked-agent", baseModelID: "org/other",
                    serverBlockReasons: ["base model 'org/other' is not in this server's inventory"]),
            ],
            localLoadedModelID: nil)

        #expect(rows.count == 4)
        // Stored agent whose base is installed: selectable.
        #expect(rows[0].selection == .server(path: "/srv/a.json"))
        #expect(rows[0].isSelectable)
        // Stored agent whose base is NOT installed: visible but disabled,
        // naming the model — never silently absent.
        #expect(rows[1].selection == .server(path: "/srv/b.json"))
        #expect(!rows[1].isSelectable)
        #expect(rows[1].exclusionReason?.contains("org/gone") == true)
        // THE reported bug: a manually-created agent is a LOCAL definition —
        // it must be listed (as an inline-spec entry) in a server workspace.
        #expect(rows[2].selection == .localDefinition(id: "runs/mv-1"))
        #expect(rows[2].isSelectable)
        #expect(rows[2].title.contains("manual-agent"))
        #expect(rows[2].title.contains("inline"))
        // A definition the server can't run stays visible with the reasons.
        #expect(!rows[3].isSelectable)
        #expect(rows[3].exclusionReason?.contains("org/other") == true)
    }

    @Test func localWorkspaceListsLocalDefinitionsOnly() {
        let rows = ChatService.agentPickerRows(
            workspaceIsServer: false,
            serverAgents: [
                remoteAgent(name: "stored", base: "org/base", path: "/srv/a.json")
            ],
            installedServerModels: ["org/base"],
            localDefinitions: [
                .init(id: "runs/mv-1", name: "for-loaded", baseModelID: "org/base"),
                .init(id: "runs/mv-2", name: "for-other", baseModelID: "org/other"),
            ],
            localLoadedModelID: "org/base")

        // Strict per-substrate scoping: never server agents in Local.
        #expect(rows.count == 2)
        #expect(rows.allSatisfy {
            if case .localDefinition = $0.selection { return true }
            return false
        })
        #expect(rows[0].isSelectable)
        // Built for a different model: visible, disabled, names the remedy —
        // the historical rule hid EVERY agent until its model was loaded.
        #expect(!rows[1].isSelectable)
        #expect(rows[1].exclusionReason == "built for org/other — load it to use")
    }

    @Test func localWorkspaceWithNoLoadedModelStillListsAgents() {
        let rows = ChatService.agentPickerRows(
            workspaceIsServer: false,
            serverAgents: [],
            installedServerModels: [],
            localDefinitions: [
                .init(id: "runs/mv-1", name: "agent", baseModelID: "org/base")
            ],
            localLoadedModelID: nil)
        #expect(rows.count == 1)
        #expect(!rows[0].isSelectable)
        #expect(rows[0].exclusionReason?.contains("org/base") == true)
    }
}

/// Selection-state transitions on a live ChatService (no disk, no model):
/// the Chat-from-Agents handoff must land SELECTED in the Playground picker.
@MainActor
struct PlaygroundAgentSelectionTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.playground-agent-picker.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    /// A ChatService in an ACTIVE server workspace whose (stubbed) inventory
    /// installs `models` — no network involved until a stored path is fetched.
    private func makeServerWorkspaceService(
        _ name: String, models: [String]
    ) throws -> ChatService {
        let store = clusterStore(defaults: try freshDefaults(name))
        let entry = store.addServer(name: "stub", urlString: "http://127.0.0.1:9")
        store.activeWorkspace = .server(entry.id)
        store.remoteState = RemoteState(
            models: models, loadedModel: nil, loadedRevision: nil,
            device: nil, isBusy: false, loadedModels: [], jobs: [])
        return ChatService(cluster: store)
    }

    private func makeRecord(
        name: String = "manual-agent", base: String = "org/base"
    ) -> ModelVariantRecord {
        ModelVariantRecord(
            url: URL(filePath: "/tmp/\(name)/model-variant.json"),
            artifact: ModelVariantArtifact(
                name: name,
                baseModelID: base,
                alphaInNormUnits: true,
                promptMode: "chatAssistant",
                qwenThinkingEnabled: false,
                temperature: 0,
                systemPrompt: ""))
    }

    @Test func chatHandoffSelectsTheLocalDefinition() throws {
        let service = try makeServerWorkspaceService("handoff", models: ["org/base"])
        let record = makeRecord()

        // The Agents tab's Chat action (server workspace) — must seed the
        // controls AND select the agent in the Playground picker.
        service.applyLocalDefinitionToServerSteering(record)

        #expect(service.selectedServerLocalDefinitionID == record.id)
        #expect(service.playgroundAgentSelection == .localDefinition(id: record.id))
        // The definition pins its base model, exactly like a stored agent.
        #expect(service.selectedRemoteModelID == "org/base")
        // No stored-variant identity is claimed for an inline seed.
        #expect(service.selectedRemoteVariantPath == nil)
    }

    @Test func blockedDefinitionDoesNotSelect() throws {
        let service = try makeServerWorkspaceService("blocked", models: ["org/base"])
        let record = makeRecord(name: "foreign", base: "org/not-installed")

        service.applyLocalDefinitionToServerSteering(record)

        #expect(service.selectedServerLocalDefinitionID == nil)
        #expect(service.playgroundAgentSelection == nil)
        #expect(service.errorMessage?.contains("cannot apply") == true)
    }

    @Test func selectingNoneClearsTheLocalDefinition() async throws {
        let service = try makeServerWorkspaceService("clear", models: ["org/base"])
        service.applyLocalDefinitionToServerSteering(makeRecord())
        #expect(service.playgroundAgentSelection != nil)

        // Picker → None (path nil short-circuits before any network).
        await service.selectPlaygroundAgent(nil)

        #expect(service.selectedServerLocalDefinitionID == nil)
        #expect(service.selectedRemoteVariantPath == nil)
        #expect(service.playgroundAgentSelection == nil)
    }

    @Test func storedAgentSelectionClearsTheLocalDefinition() async throws {
        // Stored path and seeded local definition are mutually exclusive
        // selector identities: applying either clears the other. The detail
        // fetch fails (nothing listens on the stub endpoint) — the stored
        // path is still the selection; only control seeding is skipped.
        let service = try makeServerWorkspaceService("exclusive", models: ["org/base"])
        service.applyLocalDefinitionToServerSteering(makeRecord())
        #expect(service.selectedServerLocalDefinitionID != nil)

        await service.applyRemoteVariantToSteering(path: "/srv/agents/a.json")

        #expect(service.selectedServerLocalDefinitionID == nil)
        #expect(
            service.playgroundAgentSelection == .server(path: "/srv/agents/a.json"))
    }

    @Test func localWorkspaceSelectionMirrorsSteeringVariantID() throws {
        let store = clusterStore(defaults: try freshDefaults("local"))
        let service = ChatService(cluster: store)
        let record = makeRecord()

        // The Agents tab's Chat action (Local workspace): selection is set
        // even when the base model is not loaded — the picker shows the
        // (disabled) row as current instead of a silent "None".
        service.applyModelVariantToSteering(record)

        #expect(service.selectedSteeringVariantID == record.id)
        #expect(service.playgroundAgentSelection == .localDefinition(id: record.id))
    }

    @Test func serverRowsComeFromClusterAndLocalStores() throws {
        // Instance wiring of the pure rule: rows read cluster.remoteVariants
        // (stored server agents) — a listing update is reflected without any
        // extra subscription because both are observable state. (Assertions
        // filter to .server rows: the service also lists whatever local
        // definitions exist in the dev-fallback workspace on this machine.)
        func storedRows(_ service: ChatService) -> [ChatService.AgentPickerRow] {
            service.playgroundAgentPickerRows.filter {
                if case .server = $0.selection { return true }
                return false
            }
        }
        let service = try makeServerWorkspaceService("rows", models: ["org/base"])
        #expect(storedRows(service).isEmpty)

        service.cluster.remoteVariants = [
            RemoteVariantRecord(
                name: "stored", baseModelID: "org/base", path: "/srv/a.json",
                injections: 0, adapters: 0)
        ]
        let rows = storedRows(service)
        #expect(rows.count == 1)
        #expect(rows[0].selection == .server(path: "/srv/a.json"))
        #expect(rows[0].isSelectable)
    }
}

/// Refresh-after-create: an agent saved by the Agents tab's manual-creation
/// path appears in the Playground selector's rows without an app restart —
/// in the Local workspace AND (as an inline-spec entry) in a server
/// workspace. Declared as an extension of the serialized
/// `ExperimentStoreTests` suite because it uses the process-global workspace
/// override (`WorkspaceRoot.programmaticOverride`), the same seam as
/// `OptimizationComposerTests`.
extension ExperimentStoreTests {

    @MainActor
    @Test func manualAgentCreationAppearsInPlaygroundSelector() throws {
        let raw = FileManager.default.temporaryDirectory
            .appending(component: "playground-picker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: raw, withIntermediateDirectories: true)
        // Resolve /var → /private/var: the store's directory enumerator
        // returns symlink-resolved paths, and record ids are URL paths.
        let temp = raw.resolvingSymlinksInPath()
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }

        let suite = "steerlab.tests.playground-picker.refresh"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        let store = clusterStore(defaults: defaults)
        let service = ChatService(cluster: store)
        #expect(service.playgroundAgentPickerRows.isEmpty)

        // The Agents tab's manual-creation path: ModelVariantStore.save,
        // then panel.refresh() — no other notification exists or is needed;
        // the selector reads the SAME observable store. (Row identity is the
        // SCANNED record's id — the id every panel reads; the URL `save`
        // returns can differ by a resolved /private symlink under /var.)
        try ModelVariantStore.save(
            ModelVariantArtifact(
                name: "manual-agent",
                baseModelID: "org/base",
                alphaInNormUnits: true,
                promptMode: "chatAssistant",
                qwenThinkingEnabled: false,
                temperature: 0,
                systemPrompt: ""))
        service.fineTuning.refresh()
        let record = try #require(
            service.fineTuning.variants.first { $0.artifact.name == "manual-agent" })

        // Local workspace: listed (disabled — no model loaded — but VISIBLE).
        let localRow = service.playgroundAgentPickerRows.first {
            $0.selection == .localDefinition(id: record.id)
        }
        #expect(localRow != nil)
        #expect(localRow?.isSelectable == false)
        #expect(localRow?.exclusionReason?.contains("org/base") == true)

        // Server workspace: the same definition is an inline-spec entry,
        // selectable because its base model is in the server's inventory.
        let entry = store.addServer(name: "stub", urlString: "http://127.0.0.1:9")
        store.activeWorkspace = .server(entry.id)
        store.remoteState = RemoteState(
            models: ["org/base"], loadedModel: nil, loadedRevision: nil,
            device: nil, isBusy: false, loadedModels: [], jobs: [])
        let serverRow = service.playgroundAgentPickerRows.first {
            $0.selection == .localDefinition(id: record.id)
        }
        #expect(serverRow != nil)
        #expect(serverRow?.isSelectable == true)
        #expect(serverRow?.title.contains("inline") == true)
    }
}
