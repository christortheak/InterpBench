import Foundation

// =============================================================================
// Per-machine runtime state for saved sites (maintainer ruling, 2026-08-21).
//
// Everything a CONNECTION produces — the endpoint a tunnel landed on, the
// server build seen there, the app registry's entry UUID, when we last
// connected — is true of this Mac and this Mac only. It used to ride along in
// `cluster-sites.json` beside the profile, which was harmless while that file
// was per-machine Application Support state. It stops being harmless the
// moment the registry becomes a git repository the researcher syncs: every
// connect would dirty the tree, and every pull would carry another machine's
// forward ports.
//
// So it moves here, keyed by site id, next to the operation records that were
// always per-machine. The contract the Sites directory depends on: a
// connect/disconnect cycle writes ONLY this file.
// =============================================================================

/// What this machine last observed about one site.
public struct ClusterSiteRuntimeState: Codable, Sendable, Equatable {

    /// Last endpoint the lifecycle registered (`http://127.0.0.1:<port>`).
    /// Non-secret, but ephemeral and local: the token lives in the Keychain
    /// and the port is this machine's forward.
    public var lastEndpoint: String?
    /// Last server build identity observed at that endpoint.
    public var lastServerBuild: String?
    /// The app registry's `ServerEntry.ID` for this site. Per machine because
    /// the app's persisted active-workspace selection names it, and that
    /// selection is a UI preference, not a shared fact.
    public var entryID: UUID?
    /// OTHER app-registry UUIDs that once named this site — the losers of a
    /// dedupe. Kept so the app's persisted active-workspace selection, which
    /// names a UUID, still resolves after two entries were collapsed into one.
    public var aliasEntryIDs: [UUID]
    /// Where this site sits in the registry's reading order.
    ///
    /// A DIRECTORY has no order, and a site file deliberately carries no
    /// sequence number — ordering is a display preference, not a fact about
    /// the cluster, and stamping one into a shared git repository would make
    /// two researchers' reorderings conflict. So it lives here, per machine,
    /// and a site that arrived from a pull with no slot yet simply sorts after
    /// the ordered ones, by id.
    public var order: Int?
    /// The `sourceRevision` of the payload THIS MACHINE last pushed to the
    /// site — the deploy INTENT, and the identity the staleness gate compares
    /// the deployed manifest against when one is recorded.
    ///
    /// Per machine, and never in `Sites/`: it is a fact about what this Mac
    /// did, not about the cluster. Recording it is what lets `cluster status`
    /// tell "the engine is older than my app bundle" (push) apart from "the
    /// engine is NEWER than my app bundle because I deployed it from a
    /// generated payload" (nothing is wrong) — the 2026-08-24 field report's
    /// highest-priority finding, where the bundle-only comparison offered
    /// `--allow-push` as the repair for a forward skew and would have rolled
    /// the engine back.
    public var pushedPayloadRevision: String?
    /// The `BUILD_COMMIT` stamp that same push wrote on the far side. The dev
    /// (no-manifest) comparison reads the stamp, so intent for THAT path is
    /// the stamp, not the manifest revision — they can differ when a payload
    /// carries a manifest and sits in a checkout.
    public var pushedBuildStamp: String?
    /// When that push landed. Reported, never compared: two revisions with no
    /// common repository cannot be ordered, which is why intent is recorded
    /// rather than inferred.
    public var pushedPayloadAt: Date?
    /// When this machine first saw the site, and when it last wrote its file.
    public var firstSeenAt: Date?
    public var updatedAt: Date?
    public var lastConnectedAt: Date?

    public init(
        lastEndpoint: String? = nil, lastServerBuild: String? = nil,
        entryID: UUID? = nil, aliasEntryIDs: [UUID] = [], order: Int? = nil,
        pushedPayloadRevision: String? = nil, pushedBuildStamp: String? = nil,
        pushedPayloadAt: Date? = nil,
        firstSeenAt: Date? = nil, updatedAt: Date? = nil,
        lastConnectedAt: Date? = nil
    ) {
        self.lastEndpoint = lastEndpoint
        self.lastServerBuild = lastServerBuild
        self.entryID = entryID
        self.aliasEntryIDs = aliasEntryIDs
        self.order = order
        self.pushedPayloadRevision = pushedPayloadRevision
        self.pushedBuildStamp = pushedBuildStamp
        self.pushedPayloadAt = pushedPayloadAt
        self.firstSeenAt = firstSeenAt
        self.updatedAt = updatedAt
        self.lastConnectedAt = lastConnectedAt
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        lastEndpoint = try container.decodeIfPresent(String.self, forKey: .lastEndpoint)
        lastServerBuild = try container.decodeIfPresent(
            String.self, forKey: .lastServerBuild)
        entryID = try container.decodeIfPresent(UUID.self, forKey: .entryID)
        aliasEntryIDs =
            try container.decodeIfPresent([UUID].self, forKey: .aliasEntryIDs) ?? []
        order = try container.decodeIfPresent(Int.self, forKey: .order)
        pushedPayloadRevision = try container.decodeIfPresent(
            String.self, forKey: .pushedPayloadRevision)
        pushedBuildStamp = try container.decodeIfPresent(
            String.self, forKey: .pushedBuildStamp)
        pushedPayloadAt = try container.decodeIfPresent(
            Date.self, forKey: .pushedPayloadAt)
        firstSeenAt = try container.decodeIfPresent(Date.self, forKey: .firstSeenAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt)
        lastConnectedAt = try container.decodeIfPresent(
            Date.self, forKey: .lastConnectedAt)
    }
}

/// The versioned per-machine document.
public struct ClusterSiteRuntimeDocument: Codable, Sendable, Equatable {

    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    /// Keyed by site id.
    public var sites: [String: ClusterSiteRuntimeState]
    /// Stamped once the legacy stores were absorbed into the canonical
    /// directory. PER MACHINE deliberately: the stamp cannot live in the
    /// shared registry, or the second Mac would skip migrating its own
    /// UserDefaults because the first Mac had already migrated hers.
    public var migratedLegacyStoresAt: Date?

    public init(
        schemaVersion: Int = ClusterSiteRuntimeDocument.currentSchemaVersion,
        sites: [String: ClusterSiteRuntimeState] = [:],
        migratedLegacyStoresAt: Date? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.sites = sites
        self.migratedLegacyStoresAt = migratedLegacyStoresAt
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion =
            try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        // Per-entry leniency, the same doctrine as every other store here: a
        // corrupt runtime entry costs THAT entry. Runtime state is a cache —
        // losing it costs a reconnect, never a site.
        sites =
            try container.decodeIfPresent(
                [String: LenientState].self, forKey: .sites)?
            .compactMapValues(\.state) ?? [:]
        migratedLegacyStoresAt = try container.decodeIfPresent(
            Date.self, forKey: .migratedLegacyStoresAt)
    }

    private struct LenientState: Decodable {
        let state: ClusterSiteRuntimeState?
        init(from decoder: any Decoder) {
            state = try? ClusterSiteRuntimeState(from: decoder)
        }
    }
}

/// One file, read-modify-write atomically — the same shape as the site
/// repository, so two processes cannot half-observe each other.
public struct ClusterSiteRuntimeStore: Sendable {

    public let fileURL: URL

    public init(fileURL: URL? = nil) {
        self.fileURL = fileURL ?? ClusterSupportPaths.runtimeStateFile
    }

    /// Never throws: a missing or unreadable runtime cache is an EMPTY cache,
    /// not a failure. Nothing here is authoritative about anything.
    public func load() -> ClusterSiteRuntimeDocument {
        guard let data = try? Data(contentsOf: fileURL),
            let document = try? ClusterSupportPaths.decoder().decode(
                ClusterSiteRuntimeDocument.self, from: data)
        else { return ClusterSiteRuntimeDocument() }
        return document
    }

    public func state(forSite siteID: String) -> ClusterSiteRuntimeState {
        load().sites[siteID] ?? ClusterSiteRuntimeState()
    }

    public func write(_ document: ClusterSiteRuntimeDocument) throws {
        var stamped = document
        stamped.schemaVersion = ClusterSiteRuntimeDocument.currentSchemaVersion
        let data: Data
        do {
            data = try ClusterSupportPaths.encoder().encode(stamped)
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "site-runtime.json: \(error.localizedDescription)")
        }
        try ClusterSupportPaths.writeAtomically(data, to: fileURL)
    }

    /// Read-modify-write one site's slot.
    @discardableResult
    public func update(
        siteID: String, _ mutate: (inout ClusterSiteRuntimeState) -> Void
    ) throws -> ClusterSiteRuntimeState {
        var document = load()
        var state = document.sites[siteID] ?? ClusterSiteRuntimeState()
        mutate(&state)
        document.sites[siteID] = state
        try write(document)
        return state
    }

    /// Record what a SUCCESSFUL push deployed — the site's intended engine
    /// identity, from this machine.
    ///
    /// Called only after the bytes landed, and only with identities read out
    /// of the payload that landed (its manifest's `sourceRevision`, and the
    /// `BUILD_COMMIT` the same push stamped). A push that deployed neither is
    /// recorded as neither: nothing here is ever invented, because a wrong
    /// intent would license exactly the silent rollback this record exists to
    /// prevent.
    @discardableResult
    public func recordPush(
        siteID: String, payloadRevision: String?, buildStamp: String?,
        at date: Date = Date()
    ) throws -> ClusterSiteRuntimeState {
        try update(siteID: siteID) { state in
            if let payloadRevision, !payloadRevision.isEmpty {
                state.pushedPayloadRevision = payloadRevision
            }
            if let buildStamp, !buildStamp.isEmpty {
                state.pushedBuildStamp = buildStamp
            }
            state.pushedPayloadAt = date
        }
    }

    public func remove(siteID: String) throws {
        var document = load()
        guard document.sites.removeValue(forKey: siteID) != nil else { return }
        try write(document)
    }
}
