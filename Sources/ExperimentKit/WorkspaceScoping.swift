import Foundation
import SteeringKit

/// Pure decision logic for the substrate-as-workspace UI: which artifacts a
/// picker shows for a given workspace + base model, and where a builder
/// action executes (in-process MLX, a server job, or nowhere yet).
///
/// This lives in ExperimentKit — not in a view — so the scoping rules are
/// unit-testable and identical across the app's surfaces. No networking, no
/// filesystem: callers pass in already-fetched records.
public enum WorkspaceScoping {

    // MARK: Substrate-stamp filtering (steering pickers)

    /// The Python server's engine stamp (its `repe_reader.SUBSTRATE` /
    /// experiment-store constant); the local engine's is
    /// `RepEReader.substrate` ("swift-mlx"). Vector sidecars carry the stamp
    /// in `substrate`; absent = legacy/unknown.
    public static let serverSubstrate = "python-hf-transformers"

    /// May a vector record be offered in a SERVER workspace's steering
    /// picker? An MLX-extracted vector must never steer a server generation
    /// (activations do not transfer between engines) — on shared-tree
    /// localhost setups the server catalog interleaves both engines'
    /// artifacts, so records stamped "swift-mlx" are excluded. Unstamped
    /// records stay visible: the engine is unknown, and hiding every legacy
    /// artifact would be worse than labeling it "engine unknown".
    public static func offerableForServerSteering(substrate: String?) -> Bool {
        substrate != RepEReader.substrate
    }

    /// Local mirror: never offer a "python-hf-transformers"-stamped sidecar
    /// for in-process MLX steering; unstamped stays visible.
    public static func offerableForLocalSteering(substrate: String?) -> Bool {
        substrate != serverSubstrate
    }

    /// Injection-vector picker source for a SERVER workspace's variant
    /// editor — the same strict rule as chat steering: the active server's
    /// catalog records, minus "swift-mlx"-stamped ones (an MLX vector must
    /// never enter a server-run spec), filtered to the definition's base
    /// model when one is set (vectors are model-specific artifacts; an empty
    /// base model shows everything, mirroring `compatibleServerVectors`).
    /// Generic over the record type so it is unit-testable with synthetic
    /// records.
    public static func serverInjectionVectorOptions<T>(
        _ records: [T],
        baseModelID: String?,
        substrate: (T) -> String?,
        modelID: (T) -> String
    ) -> [T] {
        let offered = records.filter { offerableForServerSteering(substrate: substrate($0)) }
        guard let baseModelID, !baseModelID.isEmpty else { return offered }
        return offered.filter { modelID($0) == baseModelID }
    }

    // MARK: Variant filtering

    /// How to treat a `nil` base model when filtering.
    public enum UnselectedModelPolicy: Sendable {
        /// No base model selected → show nothing (local steering variants:
        /// nothing can apply until a model is loaded).
        case none
        /// No base model selected → show everything (server variant list
        /// before a remote model is picked).
        case all
    }

    /// Records whose base model matches `baseModelID`, generic over the
    /// record type so both local `ModelVariantRecord`s and server
    /// `RemoteVariantRecord`s route through one tested rule.
    public static func variants<T>(
        _ records: [T],
        baseModelID: String?,
        whenUnselected policy: UnselectedModelPolicy,
        base: (T) -> String
    ) -> [T] {
        guard let baseModelID, !baseModelID.isEmpty else {
            switch policy {
            case .none: return []
            case .all: return records
            }
        }
        return records.filter { base($0) == baseModelID }
    }

    // MARK: Runnable-model gating

    /// "Does the active workspace have a runnable model?" — the Save Variant
    /// gate (and any other capture-what-is-running affordance).
    ///
    /// Local: a model must be loaded in-process — a definition captured from
    /// the live controls reflects the loaded model. Server: the selected
    /// server model is set, or one is already loaded on the server — the same
    /// readiness chatting uses, and it guarantees the captured definition can
    /// record a real server base-model id (local `loadedModelID` is
    /// deliberately irrelevant in a server workspace). Empty strings count as
    /// unset.
    public static func hasRunnableModel(
        workspaceIsServer: Bool,
        localLoadedModelID: String?,
        selectedServerModelID: String?,
        serverLoadedModelID: String?
    ) -> Bool {
        func isSet(_ value: String?) -> Bool {
            guard let value else { return false }
            return !value.trimmingCharacters(in: .whitespaces).isEmpty
        }
        if workspaceIsServer {
            return isSet(selectedServerModelID) || isSet(serverLoadedModelID)
        }
        return isSet(localLoadedModelID)
    }

    /// Strict-picker rendering rule: a current selection that is not in the
    /// active workspace's inventory is *rendered* (SwiftUI must not silently
    /// drop the binding) but non-pickable — "(not installed)". Empty/absent
    /// selections need no such row.
    public static func selectionOutsideInventory(
        _ selection: String?, inventory: [String]
    ) -> Bool {
        guard let selection, !selection.isEmpty else { return false }
        return !inventory.contains(selection)
    }

    // MARK: Local-definition applicability in a server workspace

    /// Can a LOCAL variant definition (a git-versioned recipe) be applied as
    /// an INLINE spec in the active server workspace? `blocked` carries
    /// user-facing reasons naming exactly what is missing — the caption rule
    /// is "non-destructive and specific", never a generic "needs Local".
    public enum DefinitionApplicability: Equatable, Sendable {
        case applicable
        case blocked(reasons: [String])

        public var isApplicable: Bool { self == .applicable }

        public var blockedReasons: [String] {
            if case .blocked(let reasons) = self { return reasons }
            return []
        }
    }

    /// The rule: a definition is applicable in a server workspace iff
    /// (a) its base model is in the server's inventory, (b) EVERY vector ref
    /// resolves in the server's catalog (via the same
    /// `InlineVariantComposer.resolveSlots` seam a composed send uses — the
    /// caller's `resolveVectorConcept` should apply the strict
    /// `offerableForServerSteering` filter), and (c) every adapter ref
    /// resolves server-side; ANY unresolvable adapter ref refuses (the inline
    /// spec sends the adapter directory the server must already have).
    /// Pure — inventory/catalog/adapters arrive as data.
    public static func serverDefinitionApplicability(
        definition: ModelVariantArtifact,
        installedModels: [String],
        resolveVectorConcept: (String) -> String?,
        resolveAdapter: (ModelVariantArtifact.AdapterRef) -> Bool
    ) -> DefinitionApplicability {
        var reasons: [String] = []
        if !installedModels.contains(definition.baseModelID) {
            reasons.append(
                "base model '\(definition.baseModelID)' is not in this server's inventory")
        }
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: definition.injections.map {
                InlineVariantComposer.SlotInput(
                    vectorID: $0.vectorArtifactID, layer: $0.layer,
                    alpha: $0.alpha, enabled: true)
            },
            concept: resolveVectorConcept)
        let unresolved = resolution.unresolvedVectorIDs.count
        if unresolved > 0 {
            reasons.append(
                "\(unresolved) vector ref\(unresolved == 1 ? "" : "s") "
                    + "not in this server's catalog")
        }
        for adapter in definition.adapters where !resolveAdapter(adapter) {
            reasons.append("adapter '\(adapter.name)' not on this server")
        }
        return reasons.isEmpty ? .applicable : .blocked(reasons: reasons)
    }

    // MARK: Builder routing

    /// The builder surfaces that pick a model and run a compute action.
    public enum Builder: Sendable, Equatable, CaseIterable {
        case conceptExtraction  // contrastive CAA / RepE-LAT
        case grandMeanExtraction
        case probeTraining
        case fineTuning
        case robustnessBattery
        case neutralPCBasis
        case multiAgentScenario
    }

    /// Where a builder action executes for the active workspace.
    public enum Route: Sendable, Equatable {
        /// In-process MLX, exactly today's path.
        case local
        /// Queue a durable job on the active server (endpoint recorded for
        /// help text / logging, not for the view to call directly).
        case serverJob(endpoint: String)
        /// The server has no backing for this builder yet: the control stays
        /// local-only and the caption says so (honest scoping — a disabled
        /// control with a reason beats a broken affordance).
        case localOnly(caption: String)
    }

    /// Server job endpoints that exist today (steerlab_server routes). A nil
    /// entry means the capability has no server-side implementation yet.
    static func serverEndpoint(for builder: Builder) -> String? {
        switch builder {
        case .conceptExtraction: "/api/concept/{name}/extract"
        case .grandMeanExtraction: "/api/multiconcept/extract"
        case .probeTraining: "/api/concept/{name}/probe-train"
        case .fineTuning: "/api/finetune/train"
        // Client-orchestrated, not a queued job: the battery/coherence items
        // and the report assembly stay local recipe/pure code; the model work
        // goes through the server's INLINE variant routes — coherence (and a
        // legacy battery's items) through variant-generate, a format-2
        // battery through `/api/variant/battery`, which scores it under its
        // own declared arming (open issues §23). The endpoint named here is
        // the always-used one; it is help text, not a dispatch table.
        case .robustnessBattery: "/api/variant/generate"
        case .neutralPCBasis: nil  // /api/neutral-pcs/build exists but is not wired client-side yet
        // Ad-hoc "play this panel now" is deliberately a LOCAL affordance,
        // not an unfinished one. POST /api/scenario/run exists, and since B1
        // both engines agree panels live at prompts/panels/ — but agreeing on
        // the path is not the same as the file being THERE. A cluster is
        // `remoteAuthoritative`: it serves its own filesystem, so a panel
        // authored on this Mac is not on the cluster until something carries
        // it over. The route that does carry it already exists and is already
        // pinned, hashed and gated — submitting the panel as a multi-agent
        // STUDY, which is what `localOnlyCaption` points at.
        //
        // Wiring this to `serverJob` would therefore only serve a PAIRED
        // server (same filesystem, no transfer) — real, but narrow, and worth
        // nothing for a remote cluster. See MULTI-AGENT-SUBSTRATE-PARITY-PLAN § A3.
        case .multiAgentScenario: nil
        }
    }

    /// Caption shown when a builder cannot run on the active server.
    static func localOnlyCaption(for builder: Builder) -> String {
        switch builder {
        case .robustnessBattery:
            "the robustness battery runs locally — switch the Compute workspace to Local (MLX)"
        case .neutralPCBasis:
            "neutral-PC bases build locally — switch the Compute workspace to Local (MLX)"
        case .multiAgentScenario:
            "scenario runs execute locally on this Mac — switch the Compute "
                + "workspace to Local (MLX) to run one, or run the scenario as a "
                + "multi-agent STUDY, which does execute on the server"
        default:
            "this action runs locally — switch the Compute workspace to Local (MLX)"
        }
    }

    /// Route a builder action for the active workspace. Local workspace is
    /// always the in-process path; a server workspace routes to the matching
    /// server job where one exists and is honestly local-only otherwise.
    public static func route(
        for builder: Builder,
        workspace: ClusterConnectionStore.Workspace
    ) -> Route {
        switch workspace {
        case .local:
            return .local
        case .server:
            if let endpoint = serverEndpoint(for: builder) {
                return .serverJob(endpoint: endpoint)
            }
            return .localOnly(caption: localOnlyCaption(for: builder))
        }
    }

    // MARK: Server↔workspace pairing

    /// Is the active server writing into the same data workspace the app is
    /// scoped to? Computed from `GET /api/info`'s realpath'd `root` against
    /// the app's (realpath'd) workspace root. `unknown` = no root reported
    /// yet (not connected, or an older server) — no warning, never a false
    /// alarm. `unpaired` carries the server's root so the UI can say exactly
    /// where server-side writes land.
    public enum ServerPairing: Equatable, Sendable {
        case unknown
        case paired
        case unpaired(serverRoot: String, isSourceCheckout: Bool)
        /// The server runs on a DIFFERENT machine (SSH cluster, remote LAN
        /// box): its `/scratch/…` can never path-equal the Mac's workspace,
        /// so path comparison is meaningless — the server's root simply IS
        /// the authoritative workspace over there. Not a warning state
        /// (except when that root is a source checkout, which is wrong on
        /// any machine).
        case remoteAuthoritative(serverRoot: String, isSourceCheckout: Bool)
    }

    /// Pure pairing rule (paths arrive as data; the caller realpaths the
    /// local side — the server already realpaths its reported root).
    /// `serverSharesLocalFilesystem` gates the comparison itself: comparing
    /// a Mac path against a remote filesystem's path is structurally
    /// meaningless, and pretending it produced "unpaired" made the standing
    /// banner advise an impossible repair.
    public static func serverPairing(
        localWorkspacePath: String?,
        serverRoot: String?,
        rootLooksLikeSourceCheckout: Bool?,
        serverSharesLocalFilesystem: Bool = true
    ) -> ServerPairing {
        func normalized(_ path: String) -> String {
            path.count > 1 && path.hasSuffix("/") ? String(path.dropLast()) : path
        }
        guard let serverRoot, !serverRoot.isEmpty else { return .unknown }
        guard serverSharesLocalFilesystem else {
            return .remoteAuthoritative(
                serverRoot: serverRoot,
                isSourceCheckout: rootLooksLikeSourceCheckout == true)
        }
        if let localWorkspacePath, !localWorkspacePath.isEmpty,
            normalized(localWorkspacePath) == normalized(serverRoot)
        {
            return .paired
        }
        return .unpaired(
            serverRoot: serverRoot,
            isSourceCheckout: rootLooksLikeSourceCheckout == true)
    }

    // MARK: The one artifact-list scoping rule

    /// How a panel that lists per-substrate ARTIFACTS (agents/variants, runs,
    /// robustness targets, optimization runs, vector catalogs) sources its
    /// content for the active compute target. A workspace is self-contained:
    /// under a server target every artifact list shows the SERVER's workspace
    /// content (its serving root is authoritative — never a mix), and under
    /// the local target server artifacts never appear.
    public enum ArtifactListPresentation: Equatable, Sendable {
        /// Local target: local workspace lists only.
        case localOnly
        /// Server target, paired: the server's serving root IS the app's
        /// selected workspace — one list, same files, no mismatch UI.
        case serverShared
        /// Server target, unpaired or pairing-unknown: server-scoped lists
        /// show the SERVER's tree, labeled with its serving root;
        /// `mismatch` is true only for a CONFIRMED root difference (unknown
        /// pairing shows no banner — no false alarm before /api/info lands).
        case serverAuthoritative(mismatch: Bool)

        /// Server-scoped lists are rendered at all (any server target).
        public var showsServerArtifacts: Bool { self != .localOnly }

        /// The prominent mismatch banner is warranted.
        public var showsMismatchBanner: Bool {
            self == .serverAuthoritative(mismatch: true)
        }
    }

    /// Resolve the presentation from the active target + pairing verdict
    /// (`pairing` is nil in the Local workspace).
    public static func artifactListPresentation(
        workspaceIsServer: Bool, pairing: ServerPairing?
    ) -> ArtifactListPresentation {
        guard workspaceIsServer else { return .localOnly }
        switch pairing {
        case .paired: return .serverShared
        case .unpaired: return .serverAuthoritative(mismatch: true)
        // A remote server's root IS the workspace — authoritative, no
        // mismatch flag (the mismatch banner would advise an impossible
        // same-path repair across filesystems).
        case .remoteAuthoritative: return .serverAuthoritative(mismatch: false)
        case .unknown, nil: return .serverAuthoritative(mismatch: false)
        }
    }

    /// Prominent banner for a confirmed workspace mismatch: the app's
    /// selected workspace is NOT the tree the active server serves, so
    /// server-scoped panels are showing (and server actions write to) the
    /// server's own workspace. Nil when paired/unknown/local.
    public static func workspaceMismatchBanner(pairing: ServerPairing?) -> String? {
        guard case .unpaired(let serverRoot, _)? = pairing else { return nil }
        return "server is serving workspace \(serverRoot) — switch the app to "
            + "it, or restart the server with --root <your workspace>"
    }

    // MARK: Workspace-switch affordance (mismatch banner / Compute menu)

    /// What the mismatch banner (and the Compute menu) may offer to REPAIR a
    /// confirmed mismatch by repointing the SERVER, when the server supports
    /// runtime workspace switching (capability-gated — older servers keep
    /// today's text-only banner).
    public enum WorkspaceSwitchAffordance: Equatable, Sendable {
        /// No switch affordance: paired/unknown pairing, unsupported server,
        /// or nothing safe to offer.
        case unavailable
        /// Same-machine server (direct loopback transport): the app's own
        /// workspace path is a valid server path — one click points the
        /// server at THIS workspace.
        case pointServerAtLocalWorkspace(localRoot: String)
        /// Remote server (SSH site, or direct to another host): a Mac path
        /// must never be sent — offer only SERVER-side roots (the site's
        /// declared workspace root, known recents).
        case offerServerSideRoots([String])
    }

    /// Pure affordance rule: only a CONFIRMED mismatch on a switch-capable
    /// server offers anything; the transport gate decides whether the app's
    /// local path is a legal server path or only server-side candidates are.
    public static func workspaceSwitchAffordance(
        pairing: ServerPairing?,
        supportsSwitch: Bool,
        sharesLocalFilesystem: Bool,
        localWorkspaceRoot: String?,
        serverSideCandidates: [String]
    ) -> WorkspaceSwitchAffordance {
        guard supportsSwitch else { return .unavailable }
        switch pairing {
        case .unpaired?:
            if sharesLocalFilesystem, let localWorkspaceRoot, !localWorkspaceRoot.isEmpty {
                return .pointServerAtLocalWorkspace(localRoot: localWorkspaceRoot)
            }
            return serverSideCandidates.isEmpty
                ? .unavailable
                : .offerServerSideRoots(serverSideCandidates)
        case .remoteAuthoritative?:
            // Not a mismatch to repair — but a remote server can still be
            // repointed among ITS OWN workspaces (the Round-3 runtime-switch
            // feature must survive the pairing rule getting honest).
            return serverSideCandidates.isEmpty
                ? .unavailable
                : .offerServerSideRoots(serverSideCandidates)
        case .paired?, .unknown?, nil:
            return .unavailable
        }
    }

    /// Section title for a server-scoped artifact list, naming exactly WHICH
    /// workspace the content comes from (the historical "This workspace
    /// (<server>)" label hid that a server may serve a different tree).
    public static func serverArtifactListTitle(
        kind: String, serverName: String, pairing: ServerPairing?
    ) -> String {
        switch pairing {
        case .paired:
            return "\(kind) — this workspace (on \(serverName))"
        case .unpaired(let serverRoot, _):
            return "\(kind) — \(serverName), serving \(serverRoot)"
        case .remoteAuthoritative(let serverRoot, _):
            return "\(kind) — \(serverName), workspace \(serverRoot)"
        case .unknown, nil:
            return "\(kind) — \(serverName) (serving root unknown)"
        }
    }

    // MARK: Concept-catalog scoping (Concept Lab)

    /// Local↔server relationship of one concept's paired stimulus dataset,
    /// computed from the cross-engine CONTENT hashes (raw byte hashes are
    /// engine-formatting-sensitive and would false-alarm — see
    /// `ConceptBuilder.stimulusContentHash`).
    public enum ConceptDrift: Equatable, Sendable {
        /// Both sides exist and their content hashes match.
        case inSync
        /// Both sides exist with DIFFERENT content — the "different data
        /// than the UI shows" hazard, now visible.
        case differs
        /// Paired stimuli exist locally but the server's catalog has no such
        /// concept (or an empty one).
        case localOnly
        /// The server has the concept; the local workspace has no paired
        /// stimuli for it.
        case serverOnly
        /// Neither side has comparable paired stimuli, or a hash is
        /// unavailable (older server without the contentHash field).
        case unknown
    }

    /// Pure drift rule: existence flags + content hashes in, verdict out.
    /// `existsLocally`/`existsOnServer` mean "has paired stimuli" (a hash
    /// exists exactly when at least one side of the pair files has rows).
    public static func conceptDrift(
        existsLocally: Bool, existsOnServer: Bool,
        localContentHash: String?, serverContentHash: String?
    ) -> ConceptDrift {
        switch (existsLocally, existsOnServer) {
        case (false, false): return .unknown
        case (true, false): return .localOnly
        case (false, true): return .serverOnly
        case (true, true):
            guard let localContentHash, let serverContentHash else { return .unknown }
            return localContentHash == serverContentHash ? .inSync : .differs
        }
    }

    /// Badge text + warning emphasis for a drift verdict (nil = show no
    /// badge; `unknown` stays quiet — no false alarm on an older server).
    public static func conceptDriftBadge(
        _ drift: ConceptDrift
    ) -> (label: String, isWarning: Bool)? {
        switch drift {
        case .inSync: ("in sync with server", false)
        case .differs: ("differs from server", true)
        case .localOnly: ("not on server", true)
        case .serverOnly: ("server only", false)
        case .unknown: nil
        }
    }

    /// Caption for edit affordances under an UNPAIRED server target: edits
    /// land in the LOCAL workspace while server jobs read the server's tree.
    /// Nil when paired (one tree — the server sees every edit), when the
    /// pairing is unknown (no false alarm), and under the local target.
    public static func conceptEditingNotice(
        _ presentation: ArtifactListPresentation
    ) -> String? {
        guard presentation == .serverAuthoritative(mismatch: true) else { return nil }
        return "editing the LOCAL copy — the server will not see this edit "
            + "until you upload it (Upload to server)"
    }

    /// Are the explicit per-concept sync affordances (Upload to server… /
    /// Fetch from server…) worth rendering? Only when the server may hold a
    /// DIFFERENT tree (unpaired or pairing-unknown): on a paired server the
    /// two copies are the same files and "syncing" would be noise.
    public static func conceptSyncAffordancesVisible(
        _ presentation: ArtifactListPresentation
    ) -> Bool {
        if case .serverAuthoritative = presentation { return true }
        return false
    }

    /// What a server-executing concept action (extraction/validation) must
    /// do about the dataset BEFORE queuing the job.
    public enum ServerConceptSyncPlan: Equatable, Sendable {
        /// Sync the local dataset to the server first (nothing server-side
        /// would be clobbered, or the catalog cannot say — historical push).
        case push
        /// The server's copy already matches the local dataset byte-for-text;
        /// skip the redundant write and run on its copy.
        case skipPush
        /// Refuse to dispatch: acting would either run on data other than
        /// what the panel shows, or silently overwrite a DIFFERENT server
        /// dataset. The reason is the user-facing message.
        case refuse(reason: String)
    }

    /// The server-action preflight (Concept Lab hard rule): a job that reads
    /// the SERVER's `prompts/concepts/<name>` must run on the data the panel
    /// shows. `catalogKnown` is false when the server's concept catalog
    /// could not be listed (older server) — historical push behavior, no
    /// false refusal.
    public static func serverConceptBuildSyncPlan(
        concept: String,
        catalogKnown: Bool,
        existsLocally: Bool,
        existsOnServer: Bool,
        localContentHash: String?,
        serverContentHash: String?
    ) -> ServerConceptSyncPlan {
        guard catalogKnown else { return .push }
        let drift = conceptDrift(
            existsLocally: existsLocally, existsOnServer: existsOnServer,
            localContentHash: localContentHash, serverContentHash: serverContentHash)
        switch drift {
        case .inSync:
            return .skipPush
        case .localOnly, .unknown:
            return .push
        case .serverOnly:
            return .refuse(
                reason: "concept '\(concept)' has no local paired stimuli — it "
                    + "exists only on the server; Fetch from server first (or "
                    + "author stimuli locally), so this panel shows the data "
                    + "the job would use")
        case .differs:
            return .refuse(
                reason: "server's '\(concept)' differs from local (content "
                    + "hashes local \(localContentHash?.prefix(12) ?? "?") vs "
                    + "server \(serverContentHash?.prefix(12) ?? "?")) — Upload "
                    + "to server first, or Fetch the server copy, so extraction "
                    + "runs on the data this panel shows")
        }
    }

    /// Robustness Check target sources for the active target: local records
    /// are always offered (definitions are recipes — runnable in-process
    /// locally, or as inline specs on a server), and the server's STORED
    /// agents are offered ONLY under a server target — the same source the
    /// agents list shows, so the two panels can never disagree.
    public static func robustnessTargetSources<Local, Server>(
        workspaceIsServer: Bool, localRecords: [Local], serverAgents: [Server]
    ) -> (local: [Local], server: [Server]) {
        (localRecords, workspaceIsServer ? serverAgents : [])
    }

    /// Study-builder baseline-model options — the ACTIVE workspace's
    /// inventory: a server target offers the SERVER's installed models (never
    /// local MLX repos); the local target offers the local list. A current
    /// selection outside the returned inventory is the view's
    /// "(not installed)" row, not an extra pickable option.
    public static func studyBaselineModelOptions(
        workspaceIsServer: Bool, localOptions: [String], serverOptions: [String]
    ) -> [String] {
        workspaceIsServer ? serverOptions : localOptions
    }

    /// Standing warning line for an unpaired server (nil = nothing to show).
    /// This is the FULL message (Compute section header, help text); the
    /// compact toolbar indicator shows `serverPairingBadge`.
    public static func serverPairingWarning(_ pairing: ServerPairing) -> String? {
        switch pairing {
        case .unpaired(let serverRoot, let isSourceCheckout):
            if isSourceCheckout {
                return "unpaired: server is writing into the SteerLab SOURCE "
                    + "CHECKOUT (\(serverRoot)) — pair with serve --root <workspace>"
            }
            return "unpaired: server writes to \(serverRoot) — "
                + "pair with serve --root <workspace>"
        case .remoteAuthoritative(let serverRoot, let isSourceCheckout):
            // A remote workspace is not a mismatch — warn only about the one
            // thing that is wrong on any machine.
            guard isSourceCheckout else { return nil }
            return "remote server is writing into a SOURCE CHECKOUT "
                + "(\(serverRoot)) — bootstrap/serve it with a dedicated workspace"
        case .unknown, .paired:
            return nil
        }
    }

    /// Compact indicator label next to the Compute selector/status area.
    public static func serverPairingBadge(_ pairing: ServerPairing) -> String? {
        switch pairing {
        case .unpaired(let serverRoot, let isSourceCheckout):
            if isSourceCheckout { return "unpaired: source checkout" }
            let basename = (serverRoot as NSString).lastPathComponent
            return basename.isEmpty ? "unpaired" : "unpaired: \(basename)"
        case .remoteAuthoritative(_, let isSourceCheckout):
            return isSourceCheckout ? "remote: source checkout" : nil
        case .unknown, .paired:
            return nil
        }
    }

    /// Informative (non-warning) line for a remote server's workspace, shown
    /// where the unpaired warning would have been: names the authoritative
    /// remote tree instead of crying mismatch about it.
    public static func serverPairingDescription(_ pairing: ServerPairing) -> String? {
        guard case .remoteAuthoritative(let serverRoot, false) = pairing else {
            return nil
        }
        return "server workspace: \(serverRoot) (authoritative on the remote host)"
    }
}
