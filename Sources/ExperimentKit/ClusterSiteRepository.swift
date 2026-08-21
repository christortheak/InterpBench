import CryptoKit
import Foundation

// =============================================================================
// THE canonical cluster-site registry (maintainer ruling, 2026-08-21).
//
// Saved sites lived in `UserDefaults.standard` (the app) and then in a single
// `cluster-sites.json` in Application Support (the CLI) — two stores, and
// neither of them anywhere a researcher could see, edit, or move between the
// machines they actually work on. Both are now LEGACY: read once, migrated
// once, never written again.
//
// The canonical registry is `<SteerLab home>/Sites/cluster-sites/`, one
// human-editable JSON file per site, named by the site's stable id:
//
//   * Both clients — the Mac app and `steerlab-cli` — read and write it. There
//     is one repository type behind both, not two directory scanners.
//   * It is a PLAIN DIRECTORY. `Sites/` is typically a private git repository,
//     because git is how a researcher's sites reach their other machines — but
//     SteerLab never runs git, never requires it, and never commits. Its
//     writes leave the tree dirty; committing is the researcher's act.
//   * Files are diff-friendly by construction: stable key order, pretty
//     printed, one trailing newline, and a write that would not change the
//     bytes does not happen at all (so a connect cycle leaves the tree clean).
//   * Only PROFILE facts go in. Secrets stay in the Keychain, per machine;
//     runtime state goes to `site-runtime.json` beside the operation records.
//     `ClusterSiteSanitizer` is the single enforcement point for both.
//
// Application Support keeps what was always per-machine: the operation
// records, and now the runtime cache.
// =============================================================================

/// Where cluster state lives: the shared registry in the SteerLab home, and
/// the per-machine caches in Application Support.
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

    /// THE canonical site registry: `<SteerLab home>/Sites/cluster-sites`.
    /// Resolved through `HomeLayout` — there is exactly one home-resolution
    /// path in this codebase and this is not a second one.
    public static var sitesDirectory: URL {
        HomeLayout.clusterSitesDirectory
    }

    /// The pre-2026-08-21 single-file registry. Read once by the migration,
    /// then a read-only fallback nothing writes to.
    public static var legacySitesFile: URL {
        root.appending(component: "cluster-sites.json")
    }

    /// Per-machine runtime cache: endpoints, server builds, entry ids.
    public static var runtimeStateFile: URL {
        root.appending(component: "site-runtime.json")
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

    /// ISO 8601 WITH fractional seconds. `createdAt` is not decoration in the
    /// site registry — it is the registry's reading ORDER, and whole-second
    /// resolution meant two sites saved in the same second read back
    /// reshuffled into alphabetical order.
    static let timestampStyle = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
    /// The whole-second spelling every file written before 2026-08-21 uses.
    static let legacyTimestampStyle = Date.ISO8601FormatStyle()

    /// The JSON coders every store here uses: readable, diffable, stable.
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(timestampStyle.format(date))
        }
        return encoder
    }

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        // Both spellings decode: an operation record or registry written by an
        // older build must not become unreadable because the resolution grew.
        decoder.dateDecodingStrategy = .custom { decoder in
            let text = try decoder.singleValueContainer().decode(String.self)
            if let date = try? timestampStyle.parse(text) { return date }
            if let date = try? legacyTimestampStyle.parse(text) { return date }
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "'\(text)' is not an ISO 8601 timestamp")
        }
        return decoder
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Records

/// One saved site, keyed by a STABLE, non-secret id. The display name is a
/// label and is never identity (plan §6.1).
///
/// This is the MERGED view — the profile half read from the shared registry
/// file, plus whatever this machine's runtime cache remembers about it. Only
/// the profile half is ever written back to `Sites/`; `ClusterSiteSanitizer`
/// owns that boundary, and a bare `ClusterSiteProfile` document is what it
/// writes — the same bytes `sites export` produces.
public struct ClusterSiteRecord: Codable, Sendable, Equatable, Identifiable {
    /// Stable slug (`lab-cluster`). Chosen once, never derived from the name
    /// again — renaming a site keeps its id.
    public var id: String
    public var displayName: String
    public var profile: ClusterSiteProfile
    /// RUNTIME (per machine, never written to `Sites/`): the app registry's
    /// `ServerEntry.ID` for this site, so the app's persisted
    /// active-workspace selection survives a migration and a relaunch.
    public var legacyEntryID: UUID?
    /// RUNTIME (per machine, never written to `Sites/`): last endpoint the
    /// lifecycle registered for this site (`http://127.0.0.1:<port>`).
    /// Non-secret, but it is this Mac's forward; the token lives in Keychain.
    public var lastEndpoint: String?
    /// RUNTIME (per machine, never written to `Sites/`): last server build
    /// identity observed at that endpoint.
    public var lastServerBuild: String?
    /// RUNTIME (per machine, never written to `Sites/`): app-registry UUIDs a
    /// dedupe collapsed into `legacyEntryID`, so a persisted active-workspace
    /// selection naming the loser still resolves.
    public var aliasEntryIDs: [UUID] = []
    public var createdAt: Date
    /// When the PROFILE last changed. Connecting does not move it.
    public var updatedAt: Date

    public init(
        id: String,
        displayName: String,
        profile: ClusterSiteProfile,
        legacyEntryID: UUID? = nil,
        lastEndpoint: String? = nil,
        lastServerBuild: String? = nil,
        aliasEntryIDs: [UUID] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.displayName = displayName
        self.profile = profile
        self.legacyEntryID = legacyEntryID
        self.lastEndpoint = lastEndpoint
        self.lastServerBuild = lastServerBuild
        self.aliasEntryIDs = aliasEntryIDs
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

/// The registry as a whole.
///
/// Since 2026-08-21 this is an in-memory AGGREGATE — the canonical store is a
/// directory of per-site files, and this type is what a caller gets when it
/// asks for all of them at once. It is still `Codable` because it is also,
/// exactly, the shape of the LEGACY `cluster-sites.json` the migration reads.
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

/// THE site registry, shared by the app and the CLI. A value type over one
/// DIRECTORY: every mutation re-reads, applies, and writes atomically, so two
/// processes cannot half-observe each other's writes, and a write that would
/// not change a file's bytes is skipped entirely.
public struct ClusterSiteRepository: Sendable {

    /// `<SteerLab home>/Sites/cluster-sites` (or a test's temp directory).
    public let directoryURL: URL
    /// Per-machine runtime cache: endpoints, server builds, entry ids, and the
    /// migration stamp.
    public let runtime: ClusterSiteRuntimeStore
    /// The pre-2026-08-21 single-file registry, read once by the migration.
    private let legacyDocumentURL: URL
    /// Where the one-time migration reads the legacy registry from. A closure
    /// rather than a `UserDefaults` (which is not `Sendable`), so tests inject
    /// a fixture payload and the live path reads the app's own domain.
    private let legacyRegistryData: @Sendable () -> Data?

    public init(
        directory: URL? = nil,
        runtimeStateURL: URL? = nil,
        legacyDocumentURL: URL? = nil,
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
        // A repository pointed at a NON-canonical directory keeps its
        // per-machine files beside that directory rather than in Application
        // Support. That is what makes a test hermetic by construction: naming
        // a temp directory is enough, and no test can accidentally read the
        // researcher's real runtime cache or legacy registry because it forgot
        // a second argument.
        let sidecar: (String) -> URL = { name in
            guard let directory else { return ClusterSupportPaths.root.appending(component: name) }
            return directory.deletingLastPathComponent().appending(component: name)
        }
        self.directoryURL = directory ?? ClusterSupportPaths.sitesDirectory
        self.runtime = ClusterSiteRuntimeStore(
            fileURL: runtimeStateURL ?? sidecar("site-runtime.json"))
        self.legacyDocumentURL =
            legacyDocumentURL
            ?? (directory == nil
                ? ClusterSupportPaths.legacySitesFile
                : sidecar("legacy-cluster-sites.json"))
        self.legacyRegistryData = legacyRegistryData
    }

    /// Convenience for callers that know the defaults suite by name (tests
    /// use their own suite; the app passes nil for `.standard`).
    public init(
        directory: URL? = nil, runtimeStateURL: URL? = nil,
        legacyDocumentURL: URL? = nil, legacyDefaultsSuiteName: String?
    ) {
        self.init(
            directory: directory, runtimeStateURL: runtimeStateURL,
            legacyDocumentURL: legacyDocumentURL,
            legacyRegistryData: {
                let defaults =
                    legacyDefaultsSuiteName.flatMap(UserDefaults.init(suiteName:))
                    ?? .standard
                return defaults.data(
                    forKey: ClusterConnectionStore.serversDefaultsKey)
            })
    }

    // MARK: Load / save

    /// Every site in the canonical directory, in stable id order, merged with
    /// this machine's runtime cache — after absorbing the legacy stores if
    /// that has not happened on this machine yet.
    ///
    /// An EMPTY migration is never stamped: stamping it would mean a process
    /// that simply could not see the legacy domain (the 2026-08-11 shakedown's
    /// first finding) poisons the registry for every process that could.
    /// Retrying an empty migration on each load is cheap and idempotent.
    public func load() throws -> ClusterSiteRegistryDocument {
        _ = try migrateLegacyStoresIfNeeded()
        let runtimeDocument = runtime.load()
        let records = try storedSites().map {
            record(id: $0.id, profile: $0.profile, runtime: runtimeDocument.sites[$0.id])
        }
        // A directory has no order, but the registry does. `order` is a
        // per-machine display preference (see `ClusterSiteRuntimeState`); a
        // site pulled from another machine has none yet and sorts after the
        // ordered ones, deterministically, by id.
        return ClusterSiteRegistryDocument(
            sites: records.sorted {
                let (a, b) = (
                    runtimeDocument.sites[$0.id]?.order ?? .max,
                    runtimeDocument.sites[$1.id]?.order ?? .max)
                return a == b ? $0.id < $1.id : a < b
            },
            migratedFromUserDefaultsAt: runtimeDocument.migratedLegacyStoresAt)
    }

    /// Merge one stored profile with this machine's runtime slot.
    ///
    /// The site's display name is the PROFILE's name — there is nowhere else
    /// for it to live now that a registry file is a bare profile, and that is
    /// the right answer anyway: renaming a site renames the thing, in the one
    /// place both clients read it from.
    private func record(
        id: String, profile: ClusterSiteProfile, runtime state: ClusterSiteRuntimeState?
    ) -> ClusterSiteRecord {
        let name = profile.name.trimmingCharacters(in: .whitespacesAndNewlines)
        return ClusterSiteRecord(
            id: id,
            displayName: name.isEmpty ? id : name,
            profile: profile,
            legacyEntryID: state?.entryID,
            lastEndpoint: state?.lastEndpoint,
            lastServerBuild: state?.lastServerBuild,
            aliasEntryIDs: state?.aliasEntryIDs ?? [],
            createdAt: state?.firstSeenAt ?? Date(),
            updatedAt: state?.updatedAt ?? state?.firstSeenAt ?? Date())
    }

    // MARK: The directory

    /// `<site-id>.json` for a site id.
    public func fileURL(forSite siteID: String) -> URL {
        directoryURL.appending(component: "\(siteID).json")
    }

    /// Every readable site file: `(id from the filename, profile from the
    /// bytes)`, sorted by id.
    ///
    /// Per-FILE leniency, the same doctrine the app registry has always used:
    /// one unreadable or foreign-schema file costs THAT site, never the
    /// registry. A researcher hand-editing JSON in a git repository WILL
    /// produce a broken file eventually, and losing every other site to it
    /// would be indefensible. `unreadableFiles()` is the other half — lenient,
    /// never silent.
    func storedSites() throws -> [(id: String, profile: ClusterSiteProfile)] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directoryURL, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
        else { return [] }
        return
            entries
            .filter { $0.pathExtension.lowercased() == "json" }
            .compactMap { url -> (id: String, profile: ClusterSiteProfile)? in
                guard let data = try? Data(contentsOf: url),
                    let profile = try? ClusterSiteProfile.decode(from: data)
                else { return nil }
                // The FILENAME is the id: a researcher who copies
                // `site-a.json` to `site-b.json` has made a second site, and
                // there is no second place for the id to be wrong.
                return (url.deletingPathExtension().lastPathComponent, profile)
            }
            .sorted { $0.id < $1.id }
    }

    /// Files in the registry that could not be read, as `<name>: <reason>`.
    ///
    /// `storedSites` is deliberately lenient — a researcher hand-editing
    /// JSON in a git repository will break a file eventually, and losing every
    /// other site to it would be indefensible. Leniency without a report would
    /// be SILENCE, though, and a site that vanished quietly is worse than one
    /// that failed loudly, so both clients surface this.
    public func unreadableFiles() -> [String] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directoryURL, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
        else { return [] }
        let decoder = ClusterSupportPaths.decoder()
        return
            entries
            .filter { $0.pathExtension.lowercased() == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                do {
                    _ = try ClusterSiteProfile.decode(from: try Data(contentsOf: url))
                    return nil
                } catch {
                    return "\(url.lastPathComponent): \(error.localizedDescription)"
                }
            }
    }

    /// Write one site's file — and only when the bytes would actually change.
    ///
    /// The skip is the contract behind "a connect cycle leaves `Sites/`
    /// byte-identical": the app re-persists its whole registry on any registry
    /// mutation (a tunnel relabelling a local URL, say), and a researcher
    /// whose git status went dirty every time they connected would stop
    /// trusting the directory within a day.
    @discardableResult
    func writeIfChanged(_ record: ClusterSiteRecord) throws -> Bool {
        let url = fileURL(forSite: record.id)
        let data = try ClusterSiteSanitizer.encodedSiteFile(for: record)
        if let existing = try? Data(contentsOf: url), existing == data {
            return false
        }
        try ClusterSupportPaths.writeAtomically(data, to: url)
        return true
    }

    /// Persist one record: its profile half to the shared registry, its
    /// runtime half to this machine's cache.
    private func persist(_ record: ClusterSiteRecord, now: Date = Date()) throws {
        let wrote = try writeIfChanged(record)
        var document = runtime.load()
        var state = document.sites[record.id] ?? ClusterSiteRuntimeState()
        state.lastEndpoint = record.lastEndpoint
        state.lastServerBuild = record.lastServerBuild
        if let legacyEntryID = record.legacyEntryID { state.entryID = legacyEntryID }
        if state.order == nil { state.order = Self.nextOrder(in: document) }
        if state.firstSeenAt == nil { state.firstSeenAt = now }
        if wrote { state.updatedAt = now }
        document.sites[record.id] = state
        try runtime.write(document)
    }

    /// The next free slot in this machine's display order.
    static func nextOrder(in document: ClusterSiteRuntimeDocument) -> Int {
        (document.sites.values.compactMap(\.order).max() ?? -1) + 1
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

    /// Make the directory hold exactly `document.sites`: every record written
    /// (when its bytes changed), every file for a site no longer present
    /// removed. Runtime facts on the records are routed to the per-machine
    /// cache, never into `Sites/`.
    public func write(_ document: ClusterSiteRegistryDocument) throws {
        let survivors = Set(document.sites.map(\.id))
        var wrote: Set<String> = []
        for record in document.sites where try writeIfChanged(record) {
            wrote.insert(record.id)
        }
        for stale in try storedSites() where !survivors.contains(stale.id) {
            try? FileManager.default.removeItem(at: fileURL(forSite: stale.id))
        }
        var runtimeDocument = runtime.load()
        let now = Date()
        // Display order follows the order the caller listed the sites in —
        // the app's own registry order — for sites this machine has not
        // ordered yet. Existing slots are left alone: reordering is the
        // researcher's business, not a side effect of saving.
        for record in document.sites {
            var state = runtimeDocument.sites[record.id] ?? ClusterSiteRuntimeState()
            state.lastEndpoint = record.lastEndpoint
            state.lastServerBuild = record.lastServerBuild
            if let legacyEntryID = record.legacyEntryID { state.entryID = legacyEntryID }
            if state.order == nil { state.order = Self.nextOrder(in: runtimeDocument) }
            if state.firstSeenAt == nil { state.firstSeenAt = now }
            if wrote.contains(record.id) { state.updatedAt = now }
            runtimeDocument.sites[record.id] = state
        }
        for id in runtimeDocument.sites.keys where !survivors.contains(id) {
            runtimeDocument.sites[id] = nil
        }
        try runtime.write(runtimeDocument)
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
            record.profile = profile
            // The display name IS the profile's name now (a registry file is a
            // bare profile — there is nowhere else for a label to live), so a
            // caller that wants to keep a custom name keeps it in the profile
            // it passes. `ClusterConnectionStore.addSite` does exactly that
            // when it re-registers a preset over a renamed entry.
            let name = profile.name.trimmingCharacters(in: .whitespacesAndNewlines)
            record.displayName = name.isEmpty ? record.id : name
            if let legacyEntryID { record.legacyEntryID = legacyEntryID }
            record.updatedAt = now
            document.sites[index] = record
            try persist(record, now: now)
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
        try persist(record, now: now)
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

    /// Record the endpoint + server build the lifecycle last reached.
    ///
    /// PER MACHINE ONLY. Both facts are non-secret and the bearer token never
    /// travels through this API — but a forward port and a build id are true
    /// of this Mac at this moment, so they go to the runtime cache and the
    /// site's file in `Sites/` is not touched at all. That is what makes a
    /// connect/disconnect cycle leave the shared registry byte-identical.
    @discardableResult
    public func noteConnection(
        siteID: String, endpoint: String?, serverBuild: String?, now: Date = Date()
    ) throws -> ClusterSiteRecord? {
        let document = try load()
        guard let existing = document.sites.first(where: { $0.id == siteID })
        else { return nil }
        let state = try runtime.update(siteID: siteID) { state in
            state.lastEndpoint = endpoint
            if let serverBuild { state.lastServerBuild = serverBuild }
            state.lastConnectedAt = now
        }
        var record = existing
        record.lastEndpoint = state.lastEndpoint
        record.lastServerBuild = state.lastServerBuild
        return record
    }

    /// Delete a site: its file leaves the registry (a deletion the researcher
    /// then commits) and its runtime slot leaves this machine's cache.
    public func remove(id: String) throws {
        try? FileManager.default.removeItem(at: fileURL(forSite: id))
        try runtime.remove(siteID: id)
    }

    /// How many records share a site's canonical identity — the registry-fork
    /// check `ClusterRegistrationObservation.duplicate` reports.
    public func duplicateCount(forIdentity identity: String) throws -> Int {
        try load().sites.filter { $0.canonicalIdentity == identity }.count
    }

    // MARK: Migration

    /// What one migration pass did, so the CLI can print it and the app can
    /// surface it. Silence is the one thing a migration may not be.
    public struct MigrationReport: Sendable, Equatable {

        /// Site ids materialized into the canonical directory.
        public var migrated: [String] = []
        /// Site ids skipped because a canonical file already claimed them —
        /// the existing file always wins.
        public var skipped: [String] = []
        /// Which legacy stores actually contributed, in reading order.
        public var sources: [String] = []

        public var didAnything: Bool { !migrated.isEmpty || !skipped.isEmpty }

        /// One line for a log, a status row, or CLI stderr.
        public var summary: String? {
            guard didAnything else { return nil }
            var parts = [
                "migrated \(migrated.count) cluster site(s) into the Sites "
                + "registry from \(sources.joined(separator: " + "))"
            ]
            if !migrated.isEmpty { parts.append("added: \(migrated.joined(separator: ", "))") }
            if !skipped.isEmpty {
                parts.append(
                    "kept the existing file for: \(skipped.joined(separator: ", "))")
            }
            parts.append(
                "review and commit \(HomeLayout.sitesDirectoryName)/ yourself — "
                + "SteerLab never runs git")
            return parts.joined(separator: " — ")
        }
    }

    /// Absorb BOTH legacy stores into the canonical directory, at most once per
    /// machine, and never over a file that is already there.
    ///
    /// Order is the precedence: an existing canonical file wins over
    /// everything, then the old `cluster-sites.json` (which already carried
    /// chosen ids), then the app's UserDefaults registry. Collisions are
    /// REPORTED, not resolved silently — the researcher is the one who knows
    /// whether the copy they synced or the copy this Mac remembers is right.
    ///
    /// The stamp is per machine (it lives in the runtime cache), because the
    /// canonical directory is shared: stamping it there would mean the second
    /// Mac never migrates its own UserDefaults.
    @discardableResult
    public func migrateLegacyStoresIfNeeded(
        now: Date = Date()
    ) throws -> MigrationReport {
        var runtimeDocument = runtime.load()
        if runtimeDocument.migratedLegacyStoresAt != nil { return MigrationReport() }

        var report = MigrationReport()
        var taken = Set(try storedSites().map(\.id))

        func absorb(
            _ records: [ClusterSiteRecord], source: String
        ) throws {
            guard !records.isEmpty else { return }
            report.sources.append(source)
            for record in records {
                guard !taken.contains(record.id) else {
                    report.skipped.append(record.id)
                    continue
                }
                // Same canonical identity under a different id is still the
                // same site: the file that is already there wins.
                if try storedSites().contains(where: {
                    ClusterSiteRecord(
                        id: $0.id, displayName: $0.profile.name, profile: $0.profile
                    ).canonicalIdentity == record.canonicalIdentity
                }) {
                    report.skipped.append(record.id)
                    continue
                }
                taken.insert(record.id)
                try writeIfChanged(record)
                var state =
                    runtimeDocument.sites[record.id] ?? ClusterSiteRuntimeState()
                state.lastEndpoint = record.lastEndpoint ?? state.lastEndpoint
                state.lastServerBuild = record.lastServerBuild ?? state.lastServerBuild
                state.entryID = record.legacyEntryID ?? state.entryID
                state.aliasEntryIDs = record.aliasEntryIDs
                // Reading order becomes this machine's display order, so a
                // migration never reshuffles the researcher's site list.
                if state.order == nil { state.order = Self.nextOrder(in: runtimeDocument) }
                state.firstSeenAt = state.firstSeenAt ?? record.createdAt
                state.updatedAt = now
                runtimeDocument.sites[record.id] = state
                report.migrated.append(record.id)
            }
        }

        try absorb(legacyFileRecords(), source: "cluster-sites.json")
        try absorb(
            legacyDefaultsRecords(now: now, taken: taken),
            source: "the app's saved-servers preference")

        // An EMPTY migration is never stamped (the 2026-08-11 finding): a
        // process that could not see the legacy stores must not decide for the
        // process that can.
        guard report.didAnything else { return report }
        runtimeDocument.migratedLegacyStoresAt = now
        try runtime.write(runtimeDocument)
        return report
    }

    /// The old single-file registry's records, or none.
    private func legacyFileRecords() -> [ClusterSiteRecord] {
        guard let data = try? Data(contentsOf: legacyDocumentURL),
            let document = try? ClusterSupportPaths.decoder().decode(
                ClusterSiteRegistryDocument.self, from: data)
        else { return [] }
        return document.sites
    }

    /// The app's UserDefaults registry as records: deduplicated by the same
    /// canonical identity rule the app uses, with stable slugs minted for each
    /// survivor.
    func legacyDefaultsRecords(
        now: Date = Date(), taken initiallyTaken: Set<String> = []
    ) -> [ClusterSiteRecord] {
        guard let data = legacyRegistryData(),
            let decoded = ClusterConnectionStore.decodeServersLeniently(from: data)
        else { return [] }
        let (deduplicated, aliases) =
            ClusterConnectionStore.deduplicatedServersWithAliases(decoded)
        var aliasesByOwner: [UUID: [UUID]] = [:]
        for (alias, owner) in aliases { aliasesByOwner[owner, default: []].append(alias) }
        var taken = initiallyTaken
        var records: [ClusterSiteRecord] = []
        for (index, entry) in deduplicated.enumerated() {
            let profile = ClusterConnectionStore.resolvedSite(for: entry)
            let id = Self.uniqueID(preferred: nil, profile: profile, taken: taken)
            taken.insert(id)
            let name = entry.displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            // Creation stamps ascend in READING order, because that is what
            // the registry's own ordering reads back — a researcher's site
            // list must not be reshuffled alphabetically by a migration.
            let created = now.addingTimeInterval(Double(index) / 1000)
            records.append(
                ClusterSiteRecord(
                    id: id,
                    displayName: name.isEmpty ? id : name,
                    profile: profile,
                    legacyEntryID: entry.id,
                    lastEndpoint: profile.isSSHTransport ? nil : entry.urlString,
                    aliasEntryIDs: (aliasesByOwner[entry.id] ?? []).sorted {
                        $0.uuidString < $1.uuidString
                    },
                    createdAt: created,
                    updatedAt: created))
        }
        return records
    }

    // MARK: Import

    /// Import a shared profile document into the canonical directory.
    ///
    /// The ONE import entry point: the CLI's `sites import`, the app's
    /// "Import Site JSON…", and the wizard's button all land here, so the
    /// ssh-login validation below cannot be true of one client and not
    /// another (live finding, 2026-08-21: the app's own import decoded the
    /// JSON directly and the researcher first learned the profile had no
    /// `user@` login when the cluster refused them at Duo).
    ///
    /// Refuses to clobber a site that is already in the registry unless
    /// `force` — the registry is a git repository the researcher syncs, and a
    /// silently replaced profile is a conflict discovered at connect time.
    @discardableResult
    public func importProfile(
        _ profile: ClusterSiteProfile, force: Bool = false, now: Date = Date(),
        warn: (String) -> Void = { _ in }
    ) throws -> ClusterSiteRecord {
        let document = try load()
        let identity = ClusterConnectionStore.canonicalKey(
            ClusterConnectionStore.registryKey(forProfile: profile))
        let existing = document.sites.first { $0.canonicalIdentity == identity }
        // The LOGIN verdict is decided first, and deliberately: "this profile
        // would strand you at Duo" is a more useful thing to be told than
        // "there is already a file here", and a --force that then hit the
        // login refusal would have taught the researcher to reach for --force.
        if case .refuse(let user, let host, let source) = Self.sshLoginFinding(
            incoming: profile, existing: existing?.profile) {
            throw ClusterLifecycleError.sshLoginDropped(
                siteID: existing?.id ?? (profile.name.isEmpty ? host : profile.name),
                host: host, expectedUser: user, source: source)
        }
        if !force, let existing {
            throw ClusterLifecycleError.siteFileExists(
                siteID: existing.id, path: fileURL(forSite: existing.id).path)
        }
        return try upsert(profile: profile, now: now, warn: warn)
    }

    /// Decode-then-import, for the callers that hold bytes rather than a
    /// profile (both file importers).
    @discardableResult
    public func importProfile(
        from data: Data, force: Bool = false, now: Date = Date(),
        warn: (String) -> Void = { _ in }
    ) throws -> ClusterSiteRecord {
        try importProfile(
            ClusterSiteProfile.decode(from: data), force: force, now: now, warn: warn)
    }

    // MARK: IDs

    /// Slugify a site into a stable, URL- and shell-safe id.
    static func slug(for profile: ClusterSiteProfile) -> String {
        let source = profile.name.isEmpty
            ? (profile.registryIdentity ?? profile.directURLString ?? "site")
            : profile.name
        let out = fileSafeID(source)
        return out.isEmpty ? "site" : out
    }

    /// A site id is also a FILENAME now, so it goes through the same
    /// slug alphabet whatever its source: no separators, no leading dot, no
    /// surprises for a researcher reading `ls Sites/cluster-sites`.
    static func fileSafeID(_ candidate: String) -> String {
        var out = ""
        var lastWasSeparator = true
        for character in candidate.lowercased() {
            if character.isLetter || character.isNumber {
                out.append(character)
                lastWasSeparator = false
            } else if !lastWasSeparator {
                out.append("-")
                lastWasSeparator = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return String(out.prefix(60))
    }

    static func uniqueID(
        preferred: String?, profile: ClusterSiteProfile, taken: Set<String>
    ) -> String {
        let base = preferred.map { candidate -> String in
            let sanitized = fileSafeID(candidate)
            return sanitized.isEmpty ? slug(for: profile) : sanitized
        } ?? slug(for: profile)
        guard taken.contains(base) else { return base }
        var suffix = 2
        while taken.contains("\(base)-\(suffix)") { suffix += 1 }
        return "\(base)-\(suffix)"
    }
}
