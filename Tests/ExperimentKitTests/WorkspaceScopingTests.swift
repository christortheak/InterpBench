import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the workspace scoping rules: variant filtering by
/// base model and builder-action routing per workspace. No networking, no
/// filesystem — synthetic records and isolated UserDefaults only.
@MainActor
struct WorkspaceScopingTests {

    private struct FakeVariant: Equatable {
        var name: String
        var base: String
    }

    private let variants = [
        FakeVariant(name: "fear-mix", base: "Qwen/Qwen3-4B-MLX-4bit"),
        FakeVariant(name: "authority", base: "mlx-community/gemma-3-4b-it-4bit"),
        FakeVariant(name: "fear-strong", base: "Qwen/Qwen3-4B-MLX-4bit"),
    ]

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.workspace-scoping.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    // MARK: Variant filtering

    @Test func filtersVariantsByBaseModel() {
        let matched = WorkspaceScoping.variants(
            variants, baseModelID: "Qwen/Qwen3-4B-MLX-4bit",
            whenUnselected: .none, base: \.base)
        #expect(matched.map(\.name) == ["fear-mix", "fear-strong"])
    }

    @Test func nilBaseWithNonePolicyShowsNothing() {
        // Local steering variants: nothing can apply until a model is loaded.
        let matched = WorkspaceScoping.variants(
            variants, baseModelID: nil, whenUnselected: .none, base: \.base)
        #expect(matched.isEmpty)
    }

    @Test func nilBaseWithAllPolicyShowsEverything() {
        // Server variant list before a remote model is picked.
        let matched = WorkspaceScoping.variants(
            variants, baseModelID: nil, whenUnselected: .all, base: \.base)
        #expect(matched == variants)
    }

    @Test func emptyBaseBehavesLikeNil() {
        let matched = WorkspaceScoping.variants(
            variants, baseModelID: "", whenUnselected: .all, base: \.base)
        #expect(matched == variants)
    }

    @Test func unknownBaseMatchesNothing() {
        let matched = WorkspaceScoping.variants(
            variants, baseModelID: "some/other-model",
            whenUnselected: .all, base: \.base)
        #expect(matched.isEmpty)
    }

    // MARK: Runnable-model gating (the Save Variant gate)

    @Test func localWorkspaceIsRunnableOnlyWithALoadedModel() {
        #expect(
            WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: false,
                localLoadedModelID: "Qwen/Qwen3-4B-MLX-4bit",
                selectedServerModelID: nil, serverLoadedModelID: nil))
        // Selected-but-unloaded is not runnable locally (unchanged behavior).
        #expect(
            !WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: false,
                localLoadedModelID: nil,
                selectedServerModelID: nil, serverLoadedModelID: nil))
    }

    @Test func serverWorkspaceIgnoresLocalStateAndUsesServerModel() {
        // The original bug: gating on local loadedModelID left Save Variant
        // permanently gray in a server workspace, even mid-conversation.
        #expect(
            WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: true,
                localLoadedModelID: nil,
                selectedServerModelID: "Qwen/Qwen3-0.6B",
                serverLoadedModelID: nil))
        #expect(
            WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: true,
                localLoadedModelID: nil,
                selectedServerModelID: nil,
                serverLoadedModelID: "Qwen/Qwen3-0.6B"))
        // …and a locally loaded model does NOT make a server workspace
        // runnable: the definition must record a server base model.
        #expect(
            !WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: true,
                localLoadedModelID: "Qwen/Qwen3-4B-MLX-4bit",
                selectedServerModelID: nil, serverLoadedModelID: nil))
    }

    @Test func emptyStringsCountAsUnsetModels() {
        #expect(
            !WorkspaceScoping.hasRunnableModel(
                workspaceIsServer: true,
                localLoadedModelID: nil,
                selectedServerModelID: "  ", serverLoadedModelID: ""))
    }

    // MARK: Strict-picker "(not installed)" rendering

    @Test func selectionOutsideInventoryFlagsOnlyRealOutsiders() {
        let inventory = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
        #expect(
            WorkspaceScoping.selectionOutsideInventory(
                "Qwen/Qwen3-4B-MLX-4bit", inventory: inventory))
        #expect(
            !WorkspaceScoping.selectionOutsideInventory(
                "Qwen/Qwen3-0.6B", inventory: inventory))
        // Empty/absent selections need no rendered row.
        #expect(!WorkspaceScoping.selectionOutsideInventory("", inventory: inventory))
        #expect(!WorkspaceScoping.selectionOutsideInventory(nil, inventory: inventory))
    }

    // MARK: Builder routing

    @Test func localWorkspaceRoutesEveryBuilderLocally() {
        for builder in WorkspaceScoping.Builder.allCases {
            #expect(WorkspaceScoping.route(for: builder, workspace: .local) == .local)
        }
    }

    @Test func serverWorkspaceRoutesToExistingEndpoints() throws {
        let store = clusterStore(defaults: try freshDefaults("routes"))
        let entry = store.addServer(name: "gpu-a", urlString: "http://gpu-a:8080")
        let workspace = ClusterConnectionStore.Workspace.server(entry.id)

        #expect(
            WorkspaceScoping.route(for: .conceptExtraction, workspace: workspace)
                == .serverJob(endpoint: "/api/concept/{name}/extract"))
        #expect(
            WorkspaceScoping.route(for: .grandMeanExtraction, workspace: workspace)
                == .serverJob(endpoint: "/api/multiconcept/extract"))
        #expect(
            WorkspaceScoping.route(for: .probeTraining, workspace: workspace)
                == .serverJob(endpoint: "/api/concept/{name}/probe-train"))
        #expect(
            WorkspaceScoping.route(for: .fineTuning, workspace: workspace)
                == .serverJob(endpoint: "/api/finetune/train"))
        // Robustness is client-orchestrated over the inline variant-generate
        // route: batteries/scoring stay local, only generation is remote.
        #expect(
            WorkspaceScoping.route(for: .robustnessBattery, workspace: workspace)
                == .serverJob(endpoint: "/api/variant/generate"))
    }

    @Test func missingServerBackingIsHonestlyLocalOnly() throws {
        // Neutral-PC builds have no client-wired server backing: the route
        // must say so (disabled with a caption), never pretend to run remotely.
        let store = clusterStore(defaults: try freshDefaults("local-only"))
        let entry = store.addServer(name: "gpu-a", urlString: "http://gpu-a:8080")
        let workspace = ClusterConnectionStore.Workspace.server(entry.id)

        for builder in [WorkspaceScoping.Builder.neutralPCBasis] {
            guard case .localOnly(let caption) = WorkspaceScoping.route(
                for: builder, workspace: workspace)
            else {
                Issue.record("expected localOnly route for \(builder)")
                continue
            }
            #expect(caption.contains("Local"))
        }
    }

    // MARK: Substrate-stamp filtering (steering pickers)

    @Test func serverPickerExcludesSwiftMLXStampsOnly() {
        // A vector extracted on MLX must never be offered for server-side
        // steering; the server's own stamp and unstamped legacy stay visible.
        #expect(!WorkspaceScoping.offerableForServerSteering(substrate: "swift-mlx"))
        #expect(
            WorkspaceScoping.offerableForServerSteering(
                substrate: "python-hf-transformers"))
        #expect(WorkspaceScoping.offerableForServerSteering(substrate: nil))
    }

    @Test func localPickerExcludesPythonStampsOnly() {
        // Mirror rule: a python-hf-transformers sidecar never steers
        // in-process MLX; swift-mlx and unstamped legacy stay visible.
        #expect(
            !WorkspaceScoping.offerableForLocalSteering(
                substrate: "python-hf-transformers"))
        #expect(WorkspaceScoping.offerableForLocalSteering(substrate: "swift-mlx"))
        #expect(WorkspaceScoping.offerableForLocalSteering(substrate: nil))
    }

    @Test func serverSubstrateConstantMatchesThePinnedContract() {
        // The exact string the Python engine stamps (repe_reader.SUBSTRATE /
        // experiment_store._THIS_SUBSTRATE) — a typo here would silently
        // stop filtering anything.
        #expect(WorkspaceScoping.serverSubstrate == "python-hf-transformers")
    }

    // MARK: The one artifact-list scoping rule (server-scoped panels)

    @Test func localTargetNeverShowsServerArtifacts() {
        // The mental model: under a Local target, server artifacts never
        // appear — regardless of any lingering pairing verdict.
        for pairing: WorkspaceScoping.ServerPairing? in [
            nil, .unknown, .paired,
            .unpaired(serverRoot: "/srv/other", isSourceCheckout: false),
        ] {
            let presentation = WorkspaceScoping.artifactListPresentation(
                workspaceIsServer: false, pairing: pairing)
            #expect(presentation == .localOnly)
            #expect(!presentation.showsServerArtifacts)
            #expect(!presentation.showsMismatchBanner)
        }
    }

    @Test func pairedServerTargetSharesOneTreeWithoutABanner() {
        let presentation = WorkspaceScoping.artifactListPresentation(
            workspaceIsServer: true, pairing: .paired)
        #expect(presentation == .serverShared)
        #expect(presentation.showsServerArtifacts)
        #expect(!presentation.showsMismatchBanner)
    }

    @Test func unpairedServerTargetIsAuthoritativeAndBanners() {
        // The cross-workspace-leakage fix: the server's root is authoritative
        // for server-scoped lists, AND the mismatch is loudly visible.
        let presentation = WorkspaceScoping.artifactListPresentation(
            workspaceIsServer: true,
            pairing: .unpaired(serverRoot: "/srv/other-ws", isSourceCheckout: false))
        #expect(presentation == .serverAuthoritative(mismatch: true))
        #expect(presentation.showsServerArtifacts)
        #expect(presentation.showsMismatchBanner)
    }

    @Test func unknownPairingShowsServerListsWithoutAFalseAlarm() {
        for pairing: WorkspaceScoping.ServerPairing? in [nil, .unknown] {
            let presentation = WorkspaceScoping.artifactListPresentation(
                workspaceIsServer: true, pairing: pairing)
            #expect(presentation == .serverAuthoritative(mismatch: false))
            #expect(presentation.showsServerArtifacts)
            #expect(!presentation.showsMismatchBanner)
        }
    }

    @Test func mismatchBannerNamesTheServingRootAndBothRemedies() {
        let banner = WorkspaceScoping.workspaceMismatchBanner(
            pairing: .unpaired(serverRoot: "/srv/other-ws", isSourceCheckout: false))
        #expect(
            banner == "server is serving workspace /srv/other-ws — switch the "
                + "app to it, or restart the server with --root <your workspace>")
        #expect(WorkspaceScoping.workspaceMismatchBanner(pairing: .paired) == nil)
        #expect(WorkspaceScoping.workspaceMismatchBanner(pairing: .unknown) == nil)
        #expect(WorkspaceScoping.workspaceMismatchBanner(pairing: nil) == nil)
    }

    @Test func workspaceSwitchAffordanceRequiresConfirmedMismatchAndSupport() {
        let unpaired = WorkspaceScoping.ServerPairing.unpaired(
            serverRoot: "/srv/other-ws", isSourceCheckout: false)
        // Same-machine server: the app's own workspace path is the one-click
        // repair.
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: unpaired, supportsSwitch: true,
                sharesLocalFilesystem: true, localWorkspaceRoot: "/Users/x/ws",
                serverSideCandidates: ["/srv/a"])
                == .pointServerAtLocalWorkspace(localRoot: "/Users/x/ws"))
        // Remote server: server-side roots only — never the Mac path.
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: unpaired, supportsSwitch: true,
                sharesLocalFilesystem: false, localWorkspaceRoot: "/Users/x/ws",
                serverSideCandidates: ["/scratch/u/ws"])
                == .offerServerSideRoots(["/scratch/u/ws"]))
        // Remote with nothing known to offer: no affordance.
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: unpaired, supportsSwitch: true,
                sharesLocalFilesystem: false, localWorkspaceRoot: nil,
                serverSideCandidates: [])
                == .unavailable)
        // Unsupported server (capability absent): text-only banner survives.
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: unpaired, supportsSwitch: false,
                sharesLocalFilesystem: true, localWorkspaceRoot: "/Users/x/ws",
                serverSideCandidates: ["/srv/a"])
                == .unavailable)
        // Only a CONFIRMED mismatch offers repair — paired/unknown/local
        // show no banner and no button.
        for pairing: WorkspaceScoping.ServerPairing? in [.paired, .unknown, nil] {
            #expect(
                WorkspaceScoping.workspaceSwitchAffordance(
                    pairing: pairing, supportsSwitch: true,
                    sharesLocalFilesystem: true, localWorkspaceRoot: "/Users/x/ws",
                    serverSideCandidates: ["/srv/a"])
                    == .unavailable)
        }
    }

    @Test func serverListTitlesNameExactlyWhichWorkspace() {
        #expect(
            WorkspaceScoping.serverArtifactListTitle(
                kind: "Agents", serverName: "gpu-a", pairing: .paired)
                == "Agents — this workspace (on gpu-a)")
        #expect(
            WorkspaceScoping.serverArtifactListTitle(
                kind: "Agents", serverName: "gpu-a",
                pairing: .unpaired(serverRoot: "/srv/other-ws", isSourceCheckout: false))
                == "Agents — gpu-a, serving /srv/other-ws")
        #expect(
            WorkspaceScoping.serverArtifactListTitle(
                kind: "Optimization runs", serverName: "gpu-a", pairing: .unknown)
                == "Optimization runs — gpu-a (serving root unknown)")
        #expect(
            WorkspaceScoping.serverArtifactListTitle(
                kind: "Optimization runs", serverName: "gpu-a", pairing: nil)
                == "Optimization runs — gpu-a (serving root unknown)")
    }

    // MARK: Robustness target sources (the empty-selector fix)

    @Test func robustnessOffersServerAgentsOnlyUnderAServerTarget() {
        let locals = ["def-a", "def-b"]
        let serverAgents = ["/srv/runs/model-variants/fear.json"]

        // Server target: the SERVER's stored agents (same source as the
        // agents list) PLUS local definitions (runnable as inline specs).
        let server = WorkspaceScoping.robustnessTargetSources(
            workspaceIsServer: true, localRecords: locals, serverAgents: serverAgents)
        #expect(server.local == locals)
        #expect(server.server == serverAgents)

        // Local target: server artifacts never appear.
        let local = WorkspaceScoping.robustnessTargetSources(
            workspaceIsServer: false, localRecords: locals, serverAgents: serverAgents)
        #expect(local.local == locals)
        #expect(local.server.isEmpty)
    }

    // MARK: Study-builder model options (the local-MLX-models-on-a-server fix)

    @Test func studyBaselineOptionsFollowTheActiveWorkspace() {
        let localTiers = ["Qwen/Qwen3-4B-MLX-4bit", "mlx-community/gemma-3-4b-it-4bit"]
        let serverModels = ["Qwen/Qwen3-4B", "google/gemma-3-4b-it"]

        #expect(
            WorkspaceScoping.studyBaselineModelOptions(
                workspaceIsServer: true,
                localOptions: localTiers, serverOptions: serverModels)
                == serverModels)
        #expect(
            WorkspaceScoping.studyBaselineModelOptions(
                workspaceIsServer: false,
                localOptions: localTiers, serverOptions: serverModels)
                == localTiers)
        // An empty server inventory stays empty (the view shows the
        // Install model… caption) — it must NOT fall back to local MLX ids.
        #expect(
            WorkspaceScoping.studyBaselineModelOptions(
                workspaceIsServer: true,
                localOptions: localTiers, serverOptions: [])
                .isEmpty)
    }

    // MARK: Variant-editor injection picker source (server workspace)

    private struct FakeVector {
        var id: String
        var modelID: String
        var substrate: String?
    }

    @Test func serverInjectionOptionsApplyChatSteeringRules() {
        // The Variants tab's injection picker in a server workspace must use
        // the SAME strict rules as chat steering: active server catalog only,
        // swift-mlx-stamped excluded, filtered to the definition's base model.
        let records = [
            FakeVector(
                id: "/srv/runs/a/independent", modelID: "google/gemma-3-4b-it",
                substrate: "python-hf-transformers"),
            FakeVector(
                id: "/srv/runs/b/fear", modelID: "google/gemma-3-4b-it",
                substrate: "swift-mlx"),  // shared-tree MLX artifact: never offered
            FakeVector(
                id: "/srv/runs/c/legacy", modelID: "google/gemma-3-4b-it",
                substrate: nil),  // engine unknown: stays visible
            FakeVector(
                id: "/srv/runs/d/other-model", modelID: "Qwen/Qwen3-4B",
                substrate: "python-hf-transformers"),
        ]

        let scoped = WorkspaceScoping.serverInjectionVectorOptions(
            records, baseModelID: "google/gemma-3-4b-it",
            substrate: \.substrate, modelID: \.modelID)
        #expect(scoped.map(\.id) == ["/srv/runs/a/independent", "/srv/runs/c/legacy"])

        // No base model picked yet → all offerable records (any model),
        // mirroring compatibleServerVectors' unselected behavior.
        let unscoped = WorkspaceScoping.serverInjectionVectorOptions(
            records, baseModelID: "", substrate: \.substrate, modelID: \.modelID)
        #expect(
            unscoped.map(\.id) == [
                "/srv/runs/a/independent", "/srv/runs/c/legacy", "/srv/runs/d/other-model",
            ])
        let nilModel = WorkspaceScoping.serverInjectionVectorOptions(
            records, baseModelID: nil, substrate: \.substrate, modelID: \.modelID)
        #expect(nilModel.map(\.id) == unscoped.map(\.id))
    }
}

/// Pure seam tests for "can this LOCAL variant definition apply in a server
/// workspace?" — the Variants tab's "Apply in this workspace" gate and the
/// server robustness pre-flight share this one rule.
struct ServerDefinitionApplicabilityTests {

    private func definition(
        base: String = "google/gemma-3-4b-it",
        vectorIDs: [String] = [],
        adapters: [ModelVariantArtifact.AdapterRef] = []
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "def",
            baseModelID: base,
            adapters: adapters,
            injections: vectorIDs.map {
                ModelVariantArtifact.InjectionRef(
                    concept: "fear", vectorArtifactID: $0, layer: 12, alpha: 1)
            },
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "")
    }

    private let catalog = ["/srv/runs/a/fear": "fear", "/srv/runs/b/awe": "awe"]

    @Test func fullyResolvedDefinitionIsApplicable() {
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(vectorIDs: ["/srv/runs/a/fear", "/srv/runs/b/awe"]),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in true })
        #expect(result == .applicable)
        #expect(result.isApplicable)
        #expect(result.blockedReasons.isEmpty)
    }

    @Test func missingBaseModelBlocksAndNamesIt() {
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(base: "Qwen/Qwen3-4B-MLX-4bit"),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in true })
        #expect(
            result
                == .blocked(reasons: [
                    "base model 'Qwen/Qwen3-4B-MLX-4bit' is not in this server's inventory"
                ]))
    }

    @Test func unresolvedVectorRefsBlockWithACountedCaption() {
        // The user-facing rule: name what's missing, plural-correct.
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(
                vectorIDs: ["/local/only/one", "/srv/runs/a/fear", "/local/only/two"]),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in true })
        #expect(result == .blocked(reasons: ["2 vector refs not in this server's catalog"]))

        let single = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(vectorIDs: ["/local/only/one"]),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in true })
        #expect(single == .blocked(reasons: ["1 vector ref not in this server's catalog"]))
    }

    @Test func anyUnresolvableAdapterRefRefuses() {
        let adapter = ModelVariantArtifact.AdapterRef(
            name: "judicial-lora",
            artifactPath: "runs/fine-tunes/x/fine-tune.json",
            adapterDirectory: "runs/fine-tunes/x/adapter")
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(adapters: [adapter]),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in false })
        #expect(result == .blocked(reasons: ["adapter 'judicial-lora' not on this server"]))
    }

    @Test func everyMissingPieceIsNamedTogether() {
        let adapter = ModelVariantArtifact.AdapterRef(
            name: "judicial-lora",
            artifactPath: "runs/fine-tunes/x/fine-tune.json",
            adapterDirectory: "runs/fine-tunes/x/adapter")
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(
                base: "Qwen/Qwen3-4B-MLX-4bit",
                vectorIDs: ["/local/only/one"],
                adapters: [adapter]),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { catalog[$0] },
            resolveAdapter: { _ in false })
        #expect(
            result.blockedReasons == [
                "base model 'Qwen/Qwen3-4B-MLX-4bit' is not in this server's inventory",
                "1 vector ref not in this server's catalog",
                "adapter 'judicial-lora' not on this server",
            ])
    }

    @Test func noRefsJustNeedsTheBaseModel() {
        // A prompt-only definition (system prompt + sampling) applies
        // anywhere its base model exists.
        let result = WorkspaceScoping.serverDefinitionApplicability(
            definition: definition(),
            installedModels: ["google/gemma-3-4b-it"],
            resolveVectorConcept: { _ in nil },
            resolveAdapter: { _ in false })
        #expect(result == .applicable)
    }
}

/// Pure seam tests for the server grand-mean build's dataset sync: which
/// local story corpora get PUSHED through the stories API before queuing,
/// and which required concepts must already exist server-side (pre-flighted
/// against GET /api/multiconcept/concepts).
struct GrandMeanServerSyncPlanTests {

    @Test func includedConceptsPresentLocallyArePushed() {
        let plan = ConceptBuilder.grandMeanServerSyncPlan(
            included: ["fear", "awe"], targets: ["fear"],
            localStoryConcepts: ["fear", "awe", "calm"])
        #expect(plan.push == ["awe", "fear"])
        #expect(plan.requireOnServer.isEmpty)
    }

    @Test func requiredConceptsWithoutLocalStoriesMustExistOnServer() {
        // "awe" is included but has no local corpus — it cannot be pushed,
        // so the pre-flight must verify the server already has it.
        let plan = ConceptBuilder.grandMeanServerSyncPlan(
            included: ["fear", "awe"], targets: nil,
            localStoryConcepts: ["fear"])
        #expect(plan.push == ["fear"])
        #expect(plan.requireOnServer == ["awe"])
    }

    @Test func nilIncludedPushesEveryLocalCorpus() {
        // included == nil pools the server's WHOLE corpus, so every locally
        // present corpus syncs (the local UI's "all concepts" semantic).
        let plan = ConceptBuilder.grandMeanServerSyncPlan(
            included: nil, targets: nil,
            localStoryConcepts: ["awe", "fear"])
        #expect(plan.push == ["awe", "fear"])
        #expect(plan.requireOnServer.isEmpty)
    }

    @Test func targetsOutsideTheLocalCorpusAreRequiredOnServer() {
        let plan = ConceptBuilder.grandMeanServerSyncPlan(
            included: nil, targets: ["serenity"],
            localStoryConcepts: ["fear"])
        #expect(plan.push == ["fear"])
        #expect(plan.requireOnServer == ["serenity"])
    }

    @Test func emptyEverythingPlansNothing() {
        let plan = ConceptBuilder.grandMeanServerSyncPlan(
            included: nil, targets: nil, localStoryConcepts: [])
        #expect(plan.push.isEmpty)
        #expect(plan.requireOnServer.isEmpty)
    }

    // MARK: Server↔workspace pairing (the root-incident guard)

    @Test func pairingUnknownWithoutAServerRoot() {
        // Not connected / older server: no root, no verdict, no false alarm.
        #expect(
            WorkspaceScoping.serverPairing(
                localWorkspacePath: "/data/ws",
                serverRoot: nil, rootLooksLikeSourceCheckout: nil) == .unknown)
        #expect(
            WorkspaceScoping.serverPairing(
                localWorkspacePath: "/data/ws",
                serverRoot: "", rootLooksLikeSourceCheckout: nil) == .unknown)
    }

    @Test func pairedWhenRootsMatchModuloTrailingSlash() {
        #expect(
            WorkspaceScoping.serverPairing(
                localWorkspacePath: "/data/ws",
                serverRoot: "/data/ws", rootLooksLikeSourceCheckout: false) == .paired)
        #expect(
            WorkspaceScoping.serverPairing(
                localWorkspacePath: "/data/ws/",
                serverRoot: "/data/ws", rootLooksLikeSourceCheckout: false) == .paired)
    }

    @Test func unpairedCarriesTheServerRoot() {
        let pairing = WorkspaceScoping.serverPairing(
            localWorkspacePath: "/data/ws",
            serverRoot: "/code/SteerLab", rootLooksLikeSourceCheckout: nil)
        #expect(
            pairing == .unpaired(serverRoot: "/code/SteerLab", isSourceCheckout: false))
    }

    @Test func remoteServerIsAuthoritativeNotUnpaired() {
        // An SSH cluster's /scratch can never path-equal the Mac's workspace;
        // comparing them is meaningless, so a remote server's root is simply
        // the authoritative workspace over there — never a mismatch banner
        // advising an impossible repair.
        let pairing = WorkspaceScoping.serverPairing(
            localWorkspacePath: "/Users/me/workspace",
            serverRoot: "/scratch/me/steerlab-workspace",
            rootLooksLikeSourceCheckout: false,
            serverSharesLocalFilesystem: false)
        #expect(
            pairing
                == .remoteAuthoritative(
                    serverRoot: "/scratch/me/steerlab-workspace", isSourceCheckout: false))
        #expect(WorkspaceScoping.serverPairingWarning(pairing) == nil)
        #expect(WorkspaceScoping.serverPairingBadge(pairing) == nil)
        #expect(
            WorkspaceScoping.serverPairingDescription(pairing)?
                .contains("/scratch/me/steerlab-workspace") == true)
        #expect(WorkspaceScoping.workspaceMismatchBanner(pairing: pairing) == nil)
        // Artifact lists: the remote root is authoritative, no mismatch flag.
        #expect(
            WorkspaceScoping.artifactListPresentation(
                workspaceIsServer: true, pairing: pairing)
                == .serverAuthoritative(mismatch: false))
        // Even an identical-looking path is NOT paired across filesystems.
        #expect(
            WorkspaceScoping.serverPairing(
                localWorkspacePath: "/data/ws", serverRoot: "/data/ws",
                rootLooksLikeSourceCheckout: false,
                serverSharesLocalFilesystem: false)
                == .remoteAuthoritative(serverRoot: "/data/ws", isSourceCheckout: false))
    }

    @Test func remoteSourceCheckoutStillWarns() {
        // Writing into a source checkout is wrong on ANY machine — the one
        // remote condition that keeps a warning.
        let pairing = WorkspaceScoping.serverPairing(
            localWorkspacePath: "/Users/me/workspace",
            serverRoot: "/home/me/steerlab",
            rootLooksLikeSourceCheckout: true,
            serverSharesLocalFilesystem: false)
        #expect(WorkspaceScoping.serverPairingWarning(pairing)?.contains("SOURCE CHECKOUT") == true)
        #expect(WorkspaceScoping.serverPairingBadge(pairing) == "remote: source checkout")
        #expect(WorkspaceScoping.serverPairingDescription(pairing) == nil)
    }

    @Test func remoteAuthoritativeKeepsTheServerSideSwitchAffordance() {
        // The Round-3 runtime workspace switch must survive the pairing rule
        // getting honest: a remote server is still repointable among its own
        // server-side workspaces — just never at a Mac path.
        let pairing = WorkspaceScoping.ServerPairing.remoteAuthoritative(
            serverRoot: "/scratch/me/ws-a", isSourceCheckout: false)
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: pairing, supportsSwitch: true,
                sharesLocalFilesystem: false,
                localWorkspaceRoot: "/Users/me/workspace",
                serverSideCandidates: ["/scratch/me/ws-a", "/scratch/me/ws-b"])
                == .offerServerSideRoots(["/scratch/me/ws-a", "/scratch/me/ws-b"]))
        #expect(
            WorkspaceScoping.workspaceSwitchAffordance(
                pairing: pairing, supportsSwitch: true,
                sharesLocalFilesystem: false,
                localWorkspaceRoot: "/Users/me/workspace",
                serverSideCandidates: [])
                == .unavailable)
    }

    @Test func unpairedWarningNamesRootAndRemedy() {
        let pairing = WorkspaceScoping.serverPairing(
            localWorkspacePath: "/data/ws",
            serverRoot: "/tmp/elsewhere", rootLooksLikeSourceCheckout: false)
        let warning = WorkspaceScoping.serverPairingWarning(pairing)
        #expect(
            warning == "unpaired: server writes to /tmp/elsewhere — "
                + "pair with serve --root <workspace>")
        #expect(WorkspaceScoping.serverPairingBadge(pairing) == "unpaired: elsewhere")
    }

    @Test func sourceCheckoutStrengthensTheWarning() {
        let pairing = WorkspaceScoping.serverPairing(
            localWorkspacePath: "/data/ws",
            serverRoot: "/code/SteerLab", rootLooksLikeSourceCheckout: true)
        let warning = WorkspaceScoping.serverPairingWarning(pairing)
        #expect(
            warning == "unpaired: server is writing into the SteerLab SOURCE "
                + "CHECKOUT (/code/SteerLab) — pair with serve --root <workspace>")
        #expect(
            WorkspaceScoping.serverPairingBadge(pairing) == "unpaired: source checkout")
    }

    @Test func pairedAndUnknownShowNoWarning() {
        #expect(WorkspaceScoping.serverPairingWarning(.paired) == nil)
        #expect(WorkspaceScoping.serverPairingWarning(.unknown) == nil)
        #expect(WorkspaceScoping.serverPairingBadge(.paired) == nil)
        #expect(WorkspaceScoping.serverPairingBadge(.unknown) == nil)
    }
}

/// Concept Lab workspace scoping: drift verdicts and badges, the edit-notice
/// and sync-affordance presentation rules, and the server-action preflight
/// (`serverConceptBuildSyncPlan`). All pure — hashes and flags in, verdicts
/// out.
struct ConceptLabScopingTests {

    // MARK: Drift verdicts

    @Test func matchingContentHashesAreInSync() {
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: "abc") == .inSync)
    }

    @Test func differingContentHashesAreVisibleDrift() {
        // The finding's hazard — "operate on different data than the UI
        // shows" — becomes a visible verdict instead of a latent state.
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: "def") == .differs)
    }

    @Test func existenceCasesRouteToLocalOnlyAndServerOnly() {
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: true, existsOnServer: false,
                localContentHash: "abc", serverContentHash: nil) == .localOnly)
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: false, existsOnServer: true,
                localContentHash: nil, serverContentHash: "def") == .serverOnly)
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: false, existsOnServer: false,
                localContentHash: nil, serverContentHash: nil) == .unknown)
    }

    @Test func missingHashesAreUnknownNeverAFalseAlarm() {
        // Older server without the contentHash field: both sides exist but
        // cannot be compared — stay quiet rather than cry drift.
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: true, existsOnServer: true,
                localContentHash: nil, serverContentHash: "def") == .unknown)
        #expect(
            WorkspaceScoping.conceptDrift(
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: nil) == .unknown)
    }

    // MARK: Badges

    @Test func badgesLabelEveryVerdictAndStayQuietOnUnknown() throws {
        let inSync = try #require(WorkspaceScoping.conceptDriftBadge(.inSync))
        #expect(inSync.label == "in sync with server" && !inSync.isWarning)
        let differs = try #require(WorkspaceScoping.conceptDriftBadge(.differs))
        #expect(differs.label == "differs from server" && differs.isWarning)
        let localOnly = try #require(WorkspaceScoping.conceptDriftBadge(.localOnly))
        #expect(localOnly.label == "not on server" && localOnly.isWarning)
        let serverOnly = try #require(WorkspaceScoping.conceptDriftBadge(.serverOnly))
        #expect(serverOnly.label == "server only" && !serverOnly.isWarning)
        #expect(WorkspaceScoping.conceptDriftBadge(.unknown) == nil)
    }

    // MARK: Edit notice + sync affordance presentation

    @Test func editingNoticeAppearsOnlyUnderAConfirmedMismatch() {
        let notice = WorkspaceScoping.conceptEditingNotice(
            .serverAuthoritative(mismatch: true))
        #expect(notice?.contains("LOCAL copy") == true)
        #expect(notice?.contains("upload") == true)
        // Paired = one tree (the server sees every edit); unknown pairing
        // must not false-alarm; local target has no server to notify.
        #expect(WorkspaceScoping.conceptEditingNotice(.serverShared) == nil)
        #expect(
            WorkspaceScoping.conceptEditingNotice(
                .serverAuthoritative(mismatch: false)) == nil)
        #expect(WorkspaceScoping.conceptEditingNotice(.localOnly) == nil)
    }

    @Test func syncAffordancesOnlyWhereTreesMayDiffer() {
        #expect(
            WorkspaceScoping.conceptSyncAffordancesVisible(
                .serverAuthoritative(mismatch: true)))
        #expect(
            WorkspaceScoping.conceptSyncAffordancesVisible(
                .serverAuthoritative(mismatch: false)))
        // Paired server: same files — "syncing" would be noise.
        #expect(!WorkspaceScoping.conceptSyncAffordancesVisible(.serverShared))
        #expect(!WorkspaceScoping.conceptSyncAffordancesVisible(.localOnly))
    }

    // MARK: Server-action preflight (build sync plan)

    @Test func planPushesWhenServerHasNothingToClobber() {
        #expect(
            WorkspaceScoping.serverConceptBuildSyncPlan(
                concept: "fear", catalogKnown: true,
                existsLocally: true, existsOnServer: false,
                localContentHash: "abc", serverContentHash: nil) == .push)
    }

    @Test func planKeepsHistoricalPushWhenTheCatalogIsUnknown() {
        // Older server without GET /api/concepts content hashes reachable:
        // never refuse on ignorance — behave exactly as before this feature.
        #expect(
            WorkspaceScoping.serverConceptBuildSyncPlan(
                concept: "fear", catalogKnown: false,
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: "def") == .push)
    }

    @Test func planKeepsHistoricalPushWhenHashesAreUnavailable() {
        #expect(
            WorkspaceScoping.serverConceptBuildSyncPlan(
                concept: "fear", catalogKnown: true,
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: nil) == .push)
    }

    @Test func planSkipsTheRedundantWriteWhenCopiesMatch() {
        #expect(
            WorkspaceScoping.serverConceptBuildSyncPlan(
                concept: "fear", catalogKnown: true,
                existsLocally: true, existsOnServer: true,
                localContentHash: "abc", serverContentHash: "abc") == .skipPush)
    }

    @Test func planRefusesOnDriftAndNamesBothHashes() throws {
        let plan = WorkspaceScoping.serverConceptBuildSyncPlan(
            concept: "fear", catalogKnown: true,
            existsLocally: true, existsOnServer: true,
            localContentHash: "aaaaaaaaaaaaaaaa", serverContentHash: "bbbbbbbbbbbbbbbb")
        guard case .refuse(let reason) = plan else {
            Issue.record("expected refusal, got \(plan)")
            return
        }
        // Actionable refusal, not a downstream missing-file error: names the
        // concept, both hash prefixes, and both remedies.
        #expect(reason.contains("server's 'fear' differs from local"))
        #expect(reason.contains("aaaaaaaaaaaa") && reason.contains("bbbbbbbbbbbb"))
        #expect(reason.contains("Upload") && reason.contains("Fetch"))
    }

    @Test func planRefusesServerOnlyConceptsWithAFetchRemedy() throws {
        let plan = WorkspaceScoping.serverConceptBuildSyncPlan(
            concept: "awe", catalogKnown: true,
            existsLocally: false, existsOnServer: true,
            localContentHash: nil, serverContentHash: "def")
        guard case .refuse(let reason) = plan else {
            Issue.record("expected refusal, got \(plan)")
            return
        }
        #expect(reason.contains("'awe'") && reason.contains("Fetch from server"))
    }
}

/// The cross-engine stimulus CONTENT hash — the drift comparator. Raw file
/// hashes differ across engines for the same texts (each engine's JSONL
/// formatting differs), so drift comparison rides on this canonical hash;
/// the golden hex is shared verbatim with the server's
/// `tests/test_authoring.py` (GOLDEN_CONTENT_HASH).
struct ConceptContentHashTests {

    private let goldenPositive = [
        "quote \" backslash \\ slash / tab\tend",
        "ligne à accents é — dash 🎯",
    ]
    private let goldenNegative = ["plain negative", "newline\nsecond line"]
    private let goldenHash =
        "0670e182f7e4e6b622e2da1345d1ce87b237b92ccb4059e623c5acc7c6ff7649"

    @Test func matchesTheCrossEngineGoldenFixture() {
        #expect(
            ConceptBuilder.stimulusContentHash(
                positive: goldenPositive, negative: goldenNegative) == goldenHash)
    }

    @Test func orderIsDataAndEmptySetsHaveNoHash() {
        let forward = ConceptBuilder.stimulusContentHash(
            positive: ["x", "y"], negative: ["z"])
        let reversed = ConceptBuilder.stimulusContentHash(
            positive: ["y", "x"], negative: ["z"])
        #expect(forward != nil && reversed != nil && forward != reversed)
        #expect(ConceptBuilder.stimulusContentHash(positive: [], negative: []) == nil)
        // One-sided legacy data still hashes (drift stays comparable).
        #expect(
            ConceptBuilder.stimulusContentHash(positive: ["only"], negative: [])
                != nil)
    }

    @Test func sidesAreNotInterchangeable() {
        #expect(
            ConceptBuilder.stimulusContentHash(positive: ["a"], negative: ["b"])
                != ConceptBuilder.stimulusContentHash(positive: ["b"], negative: ["a"]))
    }
}
