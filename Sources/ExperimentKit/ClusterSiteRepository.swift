import CryptoKit
import Foundation

// =============================================================================
// Shared, file-backed cluster state (CLUSTER-CLI-LIFECYCLE-PLAN §7.5).
//
// Saved sites used to live in `UserDefaults.standard`, which a separately
// launched CLI may not share with the app (and certainly will not once either
// is signed and sandboxed differently). The canonical NON-SECRET site registry
// therefore moves to an explicit versioned file in the shared SteerLab
// application-support directory: atomic writes, a schema version, stable site
// IDs, dedup by the existing canonical remote identity, and a ONE-TIME
// migration from the UserDefaults representation.
//
// The app keeps its own UserDefaults preferences (active workspace, per-site
// toggles). Only the site DEFINITIONS automation needs move here.
// =============================================================================

/// Where shared, non-secret cluster state lives on this Mac.
public enum ClusterSupportPaths {

    /// Test seam: when set, the shared cluster state lives under this root
    /// instead of Application Support. `nonisolated(unsafe)` is justified the
    /// same way as `ExperimentStore.rootOverride`: it is written by tests
    /// only, and each test writes it before touching the stores it scopes.
    /// (Deliberately NOT guarded by a blocking semaphore — the suite runs
    /// serialized, and a blocking lock here would starve Swift Testing's
    /// cooperative pool exactly as `ExperimentRootOverrideLock` does.)
    nonisolated(unsafe) public static var rootOverride: URL?

    /// `~/Library/Application Support/SteerLab` (or the test override).
    public static var root: URL {
        if let rootOverride { return rootOverride }
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? URL(filePath: NSHomeDirectory())
            .appending(components: "Library", "Application Support")
        return base.appending(component: "SteerLab")
    }

    /// The canonical site registry file.
    public static var sitesFile: URL {
        root.appending(component: "cluster-sites.json")
    }

    /// `cluster-operations/<site-id>/<operation-id>.json`.
    public static var operationsDirectory: URL {
        root.appending(component: "cluster-operations")
    }

    /// Create `directory` if needed, throwing a typed error the run log can
    /// carry.
    static func ensureDirectory(_ directory: URL) throws {
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "\(directory.path): \(error.localizedDescription)")
        }
    }

    /// Atomic write with a typed failure. Every writer in this file uses it —
    /// a half-written registry or operation record is worse than none.
    static func writeAtomically(_ data: Data, to url: URL) throws {
        try ensureDirectory(url.deletingLastPathComponent())
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "\(url.path): \(error.localizedDescription)")
        }
    }

    /// The JSON coders both stores use: readable, diffable, stable.
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Records

/// One saved site, keyed by a STABLE, non-secret id. The display name is a
/// label and is never identity (plan §6.1).
public struct ClusterSiteRecord: Codable, Sendable, Equatable, Identifiable {
    /// Stable slug (`lab-cluster`). Chosen once, never derived from the name
    /// again — renaming a site keeps its id.
    public var id: String
    public var displayName: String
    public var profile: ClusterSiteProfile
    /// The app registry's `ServerEntry.ID` this record came from, when it was
    /// migrated. Lets the app and CLI correlate the same site while the app
    /// still keeps its own registry.
    public var legacyEntryID: UUID?
    /// Last endpoint the lifecycle registered for this site
    /// (`http://127.0.0.1:<port>`). Non-secret; the token lives in Keychain.
    public var lastEndpoint: String?
    /// Last server build identity observed at that endpoint.
    public var lastServerBuild: String?
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String,
        displayName: String,
        profile: ClusterSiteProfile,
        legacyEntryID: UUID? = nil,
        lastEndpoint: String? = nil,
        lastServerBuild: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.displayName = displayName
        self.profile = profile
        self.legacyEntryID = legacyEntryID
        self.lastEndpoint = lastEndpoint
        self.lastServerBuild = lastServerBuild
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// The canonical remote identity this record dedupes on — the SAME rule
    /// the app registry uses (`ssh://host:port` with any `user@` stripped, or
    /// the normalized direct URL), so app and CLI can never disagree about
    /// whether two entries are one site.
    public var canonicalIdentity: String {
        ClusterConnectionStore.canonicalKey(
            ClusterConnectionStore.registryKey(forProfile: profile))
    }

    /// SHA-256 of the profile's canonical JSON — what an operation record
    /// stamps so a later reader knows which profile the work ran against.
    public var profileHash: String {
        guard let data = try? profile.encoded() else { return "" }
        return ClusterSupportPaths.sha256Hex(data)
    }

    /// The Keychain account key for this site's bearer token — the site's
    /// REMOTE identity for SSH transport (so tunnel-local port changes never
    /// orphan a token), the historical URL key otherwise. Shared verbatim
    /// with the app so a token written by either is usable by the other.
    public var tokenKey: String {
        if let identity = profile.remoteTokenIdentity { return identity }
        return ClusterTokenStore.key(
            forURLString: profile.directURLString ?? "")
    }
}

/// The versioned document on disk.
public struct ClusterSiteRegistryDocument: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var sites: [ClusterSiteRecord]
    /// Stamped once when the UserDefaults registry was absorbed, so the
    /// migration provably runs at most once.
    public var migratedFromUserDefaultsAt: Date?

    public init(
        schemaVersion: Int = ClusterSiteRegistryDocument.currentSchemaVersion,
        sites: [ClusterSiteRecord] = [],
        migratedFromUserDefaultsAt: Date? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.sites = sites
        self.migratedFromUserDefaultsAt = migratedFromUserDefaultsAt
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        guard version <= Self.currentSchemaVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .schemaVersion, in: container,
                debugDescription:
                    "cluster-sites.json schemaVersion \(version) is newer than this "
                    + "build understands (\(Self.currentSchemaVersion)) — update SteerLab")
        }
        schemaVersion = version
        // Per-entry leniency, the same doctrine as the app registry: one
        // corrupt record must cost THAT record, never the whole file.
        sites = try container.decodeIfPresent(
            [LenientSite].self, forKey: .sites)?.compactMap(\.record) ?? []
        migratedFromUserDefaultsAt = try container.decodeIfPresent(
            Date.self, forKey: .migratedFromUserDefaultsAt)
    }

    private struct LenientSite: Decodable {
        let record: ClusterSiteRecord?
        init(from decoder: any Decoder) {
            record = try? ClusterSiteRecord(from: decoder)
        }
    }
}

// MARK: - Repository

/// The shared site registry. A value type over one file: every mutation
/// re-reads, applies, and writes atomically, so the app and a CLI process
/// cannot half-observe each other's writes.
public struct ClusterSiteRepository: Sendable {

    public let fileURL: URL
    /// Where the one-time migration reads the legacy registry from. A closure
    /// rather than a `UserDefaults` (which is not `Sendable`), so tests inject
    /// a fixture payload and the live path reads the app's own domain.
    private let legacyRegistryData: @Sendable () -> Data?

    public init(
        fileURL: URL? = nil,
        legacyRegistryData: @escaping @Sendable () -> Data? = {
            // `.standard` first (the app migrating its own registry), then
            // the app's domain BY NAME: a separately launched CLI has its own
            // defaults domain (`steerlab-cli`), and the saved servers it must
            // migrate live in the app's. The winner is the first source whose
            // payload actually DECODES to sites — not merely the first
            // non-nil bytes: the 2026-08-11 shakedown found a leftover empty
            // `[]` in the CLI's own domain shadowing the app's real registry,
            // which read as "migrated: nothing" while the wizard showed every
            // site.
            let candidates: [Data?] = [
                UserDefaults.standard.data(
                    forKey: ClusterConnectionStore.serversDefaultsKey),
                UserDefaults(
                    suiteName: ClusterConnectionStore.legacyAppDefaultsDomain)?
                    .data(forKey: ClusterConnectionStore.serversDefaultsKey),
            ]
            for case let data? in candidates
            where ClusterConnectionStore.legacyRegistryPayloadHasSites(data) {
                return data
            }
            return candidates.compactMap { $0 }.first
        }
    ) {
        self.fileURL = fileURL ?? ClusterSupportPaths.sitesFile
        self.legacyRegistryData = legacyRegistryData
    }

    /// Convenience for callers that know the defaults suite by name (tests
    /// use their own suite; the app passes nil for `.standard`).
    public init(fileURL: URL? = nil, legacyDefaultsSuiteName: String?) {
        self.init(
            fileURL: fileURL,
            legacyRegistryData: {
                let defaults =
                    legacyDefaultsSuiteName.flatMap(UserDefaults.init(suiteName:))
                    ?? .standard
                return defaults.data(
                    forKey: ClusterConnectionStore.serversDefaultsKey)
            })
    }

    // MARK: Load / save

    /// The document as stored, running the one-time UserDefaults migration
    /// first when the file does not exist yet.
    ///
    /// An EMPTY migration result is returned but never persisted: writing it
    /// would stamp `migratedFromUserDefaultsAt` on a file that migrated
    /// nothing, and every later load would trust that file instead of
    /// retrying — a process that simply could not see the legacy domain
    /// (the shakedown's first finding) would poison the registry for every
    /// process that could. Retrying an empty migration on each load is
    /// cheap and idempotent; the file is born at first successful migration
    /// or first upsert, whichever comes first.
    public func load() throws -> ClusterSiteRegistryDocument {
        if let document = try readDocument() { return document }
        let migrated = migratedDocument()
        if !migrated.sites.isEmpty { try write(migrated) }
        return migrated
    }

    public func sites() throws -> [ClusterSiteRecord] {
        try load().sites
    }

    public func site(id: String) throws -> ClusterSiteRecord? {
        try load().sites.first { $0.id == id }
    }

    /// Resolve a caller-supplied site reference: exact id first, then a unique
    /// case-insensitive display-name match. Names are labels, so an ambiguous
    /// name resolves to nothing rather than guessing.
    public func resolve(reference: String) throws -> ClusterSiteRecord? {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        let all = try load().sites
        if let exact = all.first(where: { $0.id == trimmed }) { return exact }
        let named = all.filter {
            $0.displayName.caseInsensitiveCompare(trimmed) == .orderedSame
        }
        return named.count == 1 ? named.first : nil
    }

    public func write(_ document: ClusterSiteRegistryDocument) throws {
        var stamped = document
        stamped.schemaVersion = ClusterSiteRegistryDocument.currentSchemaVersion
        let data: Data
        do {
            data = try ClusterSupportPaths.encoder().encode(stamped)
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "cluster-sites.json: \(error.localizedDescription)")
        }
        try ClusterSupportPaths.writeAtomically(data, to: fileURL)
    }

    private func readDocument() throws -> ClusterSiteRegistryDocument? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return try ClusterSupportPaths.decoder().decode(
            ClusterSiteRegistryDocument.self, from: data)
    }

    // MARK: Mutation

    /// Insert or refresh a site by canonical remote identity. An existing
    /// record for the same identity KEEPS its id (identity is stable across
    /// profile edits) and its custom display name unless the name was just
    /// the profile's own.
    ///
    /// Write-time login validation (open-issues §17): the canonical identity
    /// deliberately ignores the ssh `user@` half, so `host` and `user@host`
    /// are ONE site — which is what lets a re-offered preset dedupe against a
    /// researcher's edited entry, and also what lets a login-less profile
    /// REPLACE a login-carrying one without a word. `sshLoginFinding` decides:
    /// a known login that would be dropped refuses; a site with no login
    /// anywhere warns through `warn` (some sites legitimately rely on a `User`
    /// entry in `~/.ssh/config`, so that case must not hard-refuse).
    @discardableResult
    public func upsert(
        profile: ClusterSiteProfile, preferredID: String? = nil,
        legacyEntryID: UUID? = nil, now: Date = Date(),
        warn: (String) -> Void = { _ in }
    ) throws -> ClusterSiteRecord {
        var document = try load()
        let identity = ClusterConnectionStore.canonicalKey(
            ClusterConnectionStore.registryKey(forProfile: profile))
        if let index = document.sites.firstIndex(where: {
            $0.canonicalIdentity == identity
        }) {
            var record = document.sites[index]
            switch Self.sshLoginFinding(
                incoming: profile, existing: record.profile) {
            case .ok:
                break
            case .warn(let note):
                warn(note)
            case .refuse(let user, let host, let source):
                throw ClusterLifecycleError.sshLoginDropped(
                    siteID: record.id, host: host, expectedUser: user,
                    source: source)
            }
            let hadCustomName =
                !record.displayName.isEmpty
                && record.displayName != record.profile.name
            record.profile = profile
            if !hadCustomName, !profile.name.isEmpty {
                record.displayName = profile.name
            }
            if let legacyEntryID { record.legacyEntryID = legacyEntryID }
            record.updatedAt = now
            document.sites[index] = record
            try write(document)
            return record
        }
        let id = Self.uniqueID(
            preferred: preferredID, profile: profile,
            taken: Set(document.sites.map(\.id)))
        // A brand-new record has no stored login to lose, but the profile can
        // still contradict itself (a `user@` transfer host or proxy jump beside
        // a login-less ssh destination), and a login-less site is worth saying
        // out loud the first time it is stored.
        switch Self.sshLoginFinding(incoming: profile, existing: nil) {
        case .ok:
            break
        case .warn(let note):
            warn(note)
        case .refuse(let user, let host, let source):
            throw ClusterLifecycleError.sshLoginDropped(
                siteID: id, host: host, expectedUser: user, source: source)
        }
        let record = ClusterSiteRecord(
            id: id,
            displayName: profile.name.isEmpty ? id : profile.name,
            profile: profile,
            legacyEntryID: legacyEntryID,
            createdAt: now,
            updatedAt: now)
        document.sites.append(record)
        try write(document)
        return record
    }

    // MARK: SSH login validation (open-issues §17)

    /// What a registry write would do to a site's ssh LOGIN.
    public enum SSHLoginFinding: Sendable, Equatable {
        /// The destination carries a login, or the site is not ssh at all.
        case ok
        /// No login anywhere — legal (`~/.ssh/config` can supply `User`), and
        /// loud, because the same shape is what an accidental drop looks like.
        case warn(String)
        /// A login this site is KNOWN to use would be dropped.
        case refuse(user: String, host: String, source: String)
    }

    /// The login half of an ssh destination (`alice@host` → `alice`),
    /// or nil when the destination is bare. Split on the LAST `@`, matching
    /// `ClusterConnectionStore.canonicalKey`, so the two cannot disagree
    /// about where the login ends.
    public static func sshLogin(in destination: String) -> String? {
        let trimmed = destination.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let at = trimmed.lastIndex(of: "@") else { return nil }
        let user = String(trimmed[trimmed.startIndex..<at])
        return user.isEmpty ? nil : user
    }

    private static func sshDestination(_ profile: ClusterSiteProfile) -> String? {
        guard case .ssh(let host, _, _, _) = profile.transport else { return nil }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Decide whether writing `incoming` over `existing` (nil for a fresh
    /// record) would leave the site unable to authenticate as itself.
    ///
    /// Deliberately NOT a heuristic. Only two things count as evidence of the
    /// expected login, both exact strings the researcher typed:
    ///
    /// 1. the login on the STORED destination this write replaces — the
    ///    2026-08-18 mechanism, since `canonicalIdentity` matches `user@host`
    ///    with bare `host` and then replaces the whole profile; and
    /// 2. a login on another destination for the SAME hostname inside the
    ///    incoming profile itself (the transfer host, or a proxy jump into the
    ///    same machine) — a profile that contradicts itself.
    ///
    /// Home-directory-shaped storage roots are NOT read as usernames: a site
    /// whose scratch root is `/scratch/projects` would refuse forever, and a
    /// guess that refuses is worse than the silence it replaced.
    public static func sshLoginFinding(
        incoming: ClusterSiteProfile, existing: ClusterSiteProfile?
    ) -> SSHLoginFinding {
        guard let destination = sshDestination(incoming) else { return .ok }
        if sshLogin(in: destination) != nil { return .ok }
        let host = destination
        if let existing, let storedDestination = sshDestination(existing),
           let storedUser = sshLogin(in: storedDestination),
           sshHostname(storedDestination).caseInsensitiveCompare(host)
            == .orderedSame {
            return .refuse(
                user: storedUser, host: host,
                source: "the site it replaces is saved as "
                    + "'\(storedDestination)'")
        }
        for (label, candidate) in sshSiblingDestinations(incoming) {
            guard let user = sshLogin(in: candidate),
                  sshHostname(candidate).caseInsensitiveCompare(host)
                    == .orderedSame
            else { continue }
            return .refuse(
                user: user, host: host,
                source: "the same profile's \(label) is '\(candidate)'")
        }
        return .warn(
            "the ssh destination '\(host)' carries no `user@` login — ssh will "
            + "authenticate as the local account unless ~/.ssh/config supplies "
            + "a `User` for this host. If this site logs in under a cluster "
            + "username, set transport.ssh.host to 'user@\(host)'.")
    }

    /// The hostname half of a destination (`user@host` → `host`).
    private static func sshHostname(_ destination: String) -> String {
        let trimmed = destination.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let at = trimmed.lastIndex(of: "@") else { return trimmed }
        return String(trimmed[trimmed.index(after: at)...])
    }

    /// Other ssh destinations the same profile carries, labelled as the
    /// refusal names them.
    private static func sshSiblingDestinations(
        _ profile: ClusterSiteProfile
    ) -> [(String, String)] {
        var siblings: [(String, String)] = []
        if case .ssh(_, let proxyJump, _, _) = profile.transport,
           let proxyJump,
           !proxyJump.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            siblings.append(("proxy jump",
                             proxyJump.trimmingCharacters(
                                in: .whitespacesAndNewlines)))
        }
        if let transfer = profile.environment.transferHost?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !transfer.isEmpty {
            siblings.append(("transfer host", transfer))
        }
        return siblings
    }

    /// Record the endpoint + server build the lifecycle last reached. Both are
    /// non-secret; the bearer token never travels through this API.
    @discardableResult
    public func noteConnection(
        siteID: String, endpoint: String?, serverBuild: String?, now: Date = Date()
    ) throws -> ClusterSiteRecord? {
        var document = try load()
        guard let index = document.sites.firstIndex(where: { $0.id == siteID })
        else { return nil }
        document.sites[index].lastEndpoint = endpoint
        if let serverBuild { document.sites[index].lastServerBuild = serverBuild }
        document.sites[index].updatedAt = now
        try write(document)
        return document.sites[index]
    }

    public func remove(id: String) throws {
        var document = try load()
        document.sites.removeAll { $0.id == id }
        try write(document)
    }

    /// How many records share a site's canonical identity — the registry-fork
    /// check `ClusterRegistrationObservation.duplicate` reports.
    public func duplicateCount(forIdentity identity: String) throws -> Int {
        try load().sites.filter { $0.canonicalIdentity == identity }.count
    }

    // MARK: Migration

    /// The document a first run produces: whatever the legacy UserDefaults
    /// registry held, deduplicated by the same canonical identity rule the app
    /// uses, with stable slugs minted for each survivor.
    func migratedDocument(now: Date = Date()) -> ClusterSiteRegistryDocument {
        guard let data = legacyRegistryData(),
            let decoded = ClusterConnectionStore.decodeServersLeniently(from: data)
        else {
            return ClusterSiteRegistryDocument()
        }
        let deduplicated = ClusterConnectionStore.deduplicatedServers(decoded)
        var taken: Set<String> = []
        var records: [ClusterSiteRecord] = []
        for entry in deduplicated {
            let profile = ClusterConnectionStore.resolvedSite(for: entry)
            let id = Self.uniqueID(preferred: nil, profile: profile, taken: taken)
            taken.insert(id)
            let name = entry.displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            records.append(
                ClusterSiteRecord(
                    id: id,
                    displayName: name.isEmpty ? id : name,
                    profile: profile,
                    legacyEntryID: entry.id,
                    lastEndpoint: profile.isSSHTransport ? nil : entry.urlString,
                    createdAt: now,
                    updatedAt: now))
        }
        return ClusterSiteRegistryDocument(
            sites: records, migratedFromUserDefaultsAt: now)
    }

    // MARK: IDs

    /// Slugify a site into a stable, URL- and shell-safe id.
    static func slug(for profile: ClusterSiteProfile) -> String {
        let source = profile.name.isEmpty
            ? (profile.registryIdentity ?? profile.directURLString ?? "site")
            : profile.name
        var out = ""
        var lastWasSeparator = true  // suppress a leading hyphen
        for character in source.lowercased() {
            if character.isLetter || character.isNumber {
                out.append(character)
                lastWasSeparator = false
            } else if !lastWasSeparator {
                out.append("-")
                lastWasSeparator = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return out.isEmpty ? "site" : String(out.prefix(60))
    }

    static func uniqueID(
        preferred: String?, profile: ClusterSiteProfile, taken: Set<String>
    ) -> String {
        let base = preferred.map { candidate -> String in
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? slug(for: profile) : trimmed
        } ?? slug(for: profile)
        guard taken.contains(base) else { return base }
        var suffix = 2
        while taken.contains("\(base)-\(suffix)") { suffix += 1 }
        return "\(base)-\(suffix)"
    }
}
