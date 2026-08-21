import CryptoKit
import Foundation

/// Localizes a server-side vector artifact into the LOCAL workspace so a
/// hand-composed variant stores a workspace-relative reference backed by
/// local bytes.
///
/// Design rule (Christian, 2026-08-05): the Mac workspace is the source of
/// truth; the server is a dumb running environment that caches big files
/// and can re-download what's missing. A variant that references a vector
/// by the server's absolute path (`/scratch/…`) violates that — bundle
/// packaging refuses ("pinned input missing") because the pinned input
/// exists on only one substrate. Selecting a server vector therefore
/// FETCHES the pair (`<name>.json` sidecar + `<name>.safetensors`, ~1.3 MB)
/// into the local workspace at the SAME run-relative path and stores the
/// relative ref, which both engines resolve: locally by direct join, on the
/// server via `paths.resolve_artifact`'s root rebase.
///
/// Three rules make the fetch trustworthy (2026-08-06 review, P2 — the
/// first version accepted an existing local pair on EXISTENCE alone, wrote
/// downloads straight to their final names, and checked containment
/// lexically):
///
/// 1. **Bytes are bound to the catalog row by hash.** The server advertises
///    `sidecarSha256`/`tensorSha256`; a local file that disagrees is not a
///    stale cache to be refreshed, it is a CONFLICT — the workspace holds
///    bytes the server does not have. We refuse and name both hashes rather
///    than overwrite local truth. A pre-field server advertises neither, and
///    the historical existence-only behaviour continues, loudly.
/// 2. **The fetch is transactional.** Downloads land in a staging directory
///    and are verified there; only then are they published under the
///    advertised stem. A failure, a refusal, or a crash mid-fetch leaves the
///    workspace exactly as it was — never a half-written pair that
///    `isLocal` would report as present.
/// 3. **Containment is checked on the REAL path.** A symlinked run directory
///    passes every string test and then writes outside the workspace, so the
///    destination is symlink-resolved before it is compared to `runs/`.
///
/// Round 2 of the same review added two more (2026-08-06):
///
/// 4. **What was verified is DISCLOSED, never inferred.** A catalog row may
///    advertise one digest and not the other, and the first version verified
///    the advertised half while warning only when BOTH were absent — through
///    a `print` nobody reads. Requiring both digests was considered and
///    REJECTED (Christian): a pre-field server would become an unexplainable
///    save failure. So localization verifies whatever is advertised and
///    RETURNS what that came to (`Verification`), the caller shows it, and
///    nothing is ever blocked on it.
/// 5. **Publication is serialized per target and rolls back.** Publishing is
///    two moves; a failure on the second — or a concurrent localization of
///    the same stem landing between the check and the publish — left a
///    tensor-only workspace, contradicting rule 2's promise. The whole
///    check→stage→publish section now runs under a per-stem lock, and a
///    failed second move undoes the first.
public enum RemoteVectorLocalization {

    /// How much of the artifact pair the catalog let us bind to its bytes.
    ///
    /// A three-state answer rather than a Bool, because the middle case is
    /// real and common on a partly-upgraded server: one digest advertised,
    /// one not. The caller SHOWS this; it never gates on it.
    public enum Verification: Sendable, Equatable {
        /// Every member of the pair was checked against an advertised digest.
        case fullyVerified
        /// Some members were checked and some had nothing to check against.
        /// `missing` holds the unbacked member suffixes, sorted.
        case partiallyVerified(missing: [String])
        /// The catalog advertised no digests at all (pre-field server): the
        /// bytes are in the workspace, but nothing binds them to the row.
        case unverified

        public var isFullyVerified: Bool { self == .fullyVerified }
    }

    /// What a localization came to: the reference to store, and how well the
    /// bytes behind it are bound to the catalog row.
    public struct Outcome: Sendable, Equatable {
        /// The workspace-relative reference to store in the definition.
        public let relativeID: String
        public let verification: Verification
        /// The disclosure line for a non-fully-verified localization — the
        /// exact text handed to `warn`, so a caller can surface it without
        /// re-wording it. Nil when everything was verified.
        public let notice: String?
    }

    public enum LocalizationError: Error, LocalizedError, Equatable {
        /// The server did not provide a workspace-relative id (pre-field
        /// server build) — the caller keeps the verbatim server reference.
        case noRelativeID
        /// The advertised relative id is not a safe `runs/…` subpath.
        case unsafeRelativeID(String)
        /// The relative id resolves — through symlinks — outside the
        /// workspace's `runs/` tree.
        case escapesRunsTree(id: String, resolved: String)
        /// A file already in the workspace disagrees with the hash the
        /// catalog advertises. Never resolved by overwriting: local bytes
        /// are the source of truth, and one of the two sides is wrong in a
        /// way only the researcher can settle.
        case localBytesDiffer(path: String, advertised: String, local: String)
        /// A freshly downloaded file disagrees with its advertised hash —
        /// truncated transfer, wrong artifact, or a server whose catalog and
        /// tree have drifted apart. Staged bytes are discarded.
        case downloadedBytesDiffer(path: String, advertised: String,
                                   downloaded: String)
        /// The download did not produce a readable file to publish (or, with
        /// nothing advertised to verify it by, produced one under a name
        /// that is not the advertised member).
        case downloadDidNotLand(String)
        /// A verified member could not be moved into place. Everything this
        /// call had already published was rolled back, so the workspace is
        /// as it was — the promise the transactional fetch makes.
        case publishFailed(member: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .noRelativeID:
                return "the server catalog did not advertise a workspace-relative id"
            case .unsafeRelativeID(let id):
                return "refusing to localize '\(id)': not a runs/-relative subpath"
            case .escapesRunsTree(let id, let resolved):
                return """
                    refusing to localize '\(id)': it resolves to \(resolved), \
                    outside this workspace's runs/ tree
                    """
            case .localBytesDiffer(let path, let advertised, let local):
                return """
                    refusing to overwrite \(path): the workspace's copy hashes \
                    \(local) but the server advertises \(advertised). The local \
                    bytes are the source of truth — inspect both, then delete \
                    the local pair only if you mean to take the server's copy.
                    """
            case .downloadedBytesDiffer(let path, let advertised, let downloaded):
                return """
                    discarded the download of \(path): it hashes \(downloaded) \
                    but the catalog advertises \(advertised) — nothing was \
                    written to the workspace
                    """
            case .downloadDidNotLand(let path):
                return "the download did not produce \(path)"
            case .publishFailed(let member, let reason):
                return """
                    could not publish \(member) into the workspace (\(reason)); \
                    everything this fetch had already written was rolled back, \
                    so the workspace is unchanged
                    """
            }
        }
    }

    /// True when both artifact files already exist locally. EXISTENCE ONLY —
    /// a caller deciding whether to trust those bytes must verify them (see
    /// `localize`); this stays a cheap presence check for callers that only
    /// need to know whether a fetch is pending.
    public static func isLocal(relativeID: String, workspaceRoot: URL) -> Bool {
        let stem = workspaceRoot.appending(path: relativeID)
        let fm = FileManager.default
        return fm.fileExists(atPath: stem.path + ".safetensors")
            && fm.fileExists(atPath: stem.path + ".json")
    }

    /// The two members of an artifact pair, in publish order.
    private static let members = [".safetensors", ".json"]

    /// Ensures the artifact pair for `record` exists in the local workspace,
    /// matches the hashes the catalog advertises, and returns the
    /// workspace-relative reference to store. `download` fetches one server
    /// path into a directory and returns the local file URL (the
    /// `ClusterClient.downloadArtifact` shape, injected for testability).
    ///
    /// `sidecarSha256`/`tensorSha256` come from the catalog row
    /// (`RemoteVectorRecord`). A missing digest is never a refusal — a
    /// pre-field server advertises neither, a partly-upgraded one may
    /// advertise just one — but what went unverified is REPORTED, both
    /// through `warn` and in the returned `Outcome.verification`, so the
    /// caller can put it in front of the researcher at save time.
    public static func localize(
        serverID: String,
        workspaceRelativeID: String?,
        sidecarSha256: String? = nil,
        tensorSha256: String? = nil,
        workspaceRoot: URL = VectorCatalog.projectRoot,
        warn: @Sendable (String) -> Void = { print("⚠︎ \($0)") },
        download: @Sendable (String, URL) async throws -> URL
    ) async throws -> Outcome {
        guard let relativeID = workspaceRelativeID, !relativeID.isEmpty else {
            throw LocalizationError.noRelativeID
        }
        // Containment, first lexically: the id names where WE will write
        // inside the local workspace — accept only a clean runs/ subpath
        // (server responses are data, not trusted paths).
        let components = relativeID.split(separator: "/").map(String.init)
        guard !relativeID.hasPrefix("/"), components.count >= 2,
            components.first == "runs",
            !components.contains(".."), !components.contains(".")
        else {
            throw LocalizationError.unsafeRelativeID(relativeID)
        }
        // …then on the REAL path. A lexically perfect `runs/<run>/<name>`
        // whose run directory is a symlink to /scratch would otherwise write
        // the "localized" bytes right back onto the substrate we are
        // localizing away from.
        let stem = try resolvedStem(relativeID: relativeID,
                                    workspaceRoot: workspaceRoot)
        let expected = [".safetensors": tensorSha256, ".json": sidecarSha256]
        let verification = verification(for: expected)

        // Everything from the presence check to the publish is one critical
        // section per TARGET STEM (2026-08-06 review round 2, P2). It is a
        // check-then-act on the filesystem: two saves localizing the same
        // vector concurrently would each see the pair missing, each stage a
        // copy, and interleave their two-move publishes into exactly the
        // tensor-only state rule 2 promises never to leave. Keyed by stem,
        // so unrelated vectors still localize in parallel.
        await StemLock.shared.lock(stem.path)
        // `defer` cannot await, so the release is an unstructured Task. It
        // does NOT inherit cancellation from this task, so the lock is
        // released on every exit path including a cancelled one; a waiter
        // resumed a moment later is a scheduling detail, not a race.
        defer { Task { await StemLock.shared.unlock(stem.path) } }

        // Verify EVERY file already present before deciding anything. A
        // half-present pair must not let the fetch quietly replace the half
        // that is there, and a full pair that verifies needs no fetch at all.
        var missing: [String] = []
        for member in members {
            let url = URL(filePath: stem.path + member)
            guard FileManager.default.fileExists(atPath: url.path) else {
                missing.append(member)
                continue
            }
            guard let advertised = expected[member] ?? nil else { continue }
            let local = try sha256(of: url)
            guard local == advertised else {
                throw LocalizationError.localBytesDiffer(
                    path: url.path, advertised: advertised, local: local)
            }
        }
        if missing.isEmpty {
            let notice = keptNotice(relativeID: relativeID,
                                    verification: verification)
            if let notice { warn(notice) }
            return Outcome(relativeID: relativeID,
                           verification: verification, notice: notice)
        }

        // Transactional fetch: stage, verify, then publish. The staging
        // directory is a sibling of the destination so the publish is a
        // same-volume rename (atomic), and it is removed on every exit path.
        let directory = stem.deletingLastPathComponent()
        let staging = directory.appending(
            component: ".steerlab-localize-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.createDirectory(
            at: staging, withIntermediateDirectories: true)

        var staged: [String: URL] = [:]
        for (index, member) in missing.enumerated() {
            // One staging directory PER member: the transfer names the file,
            // and two transfers that choose the same name (a server sending
            // the same Content-Disposition twice) would otherwise clobber
            // each other into a pair that is half one artifact.
            let slot = staging.appending(component: String(index))
            try FileManager.default.createDirectory(
                at: slot, withIntermediateDirectories: true)
            let remote = serverID + member
            let landed = try await download(remote, slot)
            guard isReadableFile(landed) else {
                throw LocalizationError.downloadDidNotLand(
                    stem.lastPathComponent + member)
            }
            if let advertised = expected[member] ?? nil {
                // The content is what identifies the file, so whatever name
                // the transfer chose is irrelevant — it is published under
                // the advertised stem below.
                let downloaded = try sha256(of: landed)
                guard downloaded == advertised else {
                    throw LocalizationError.downloadedBytesDiffer(
                        path: stem.lastPathComponent + member,
                        advertised: advertised, downloaded: downloaded)
                }
            } else if landed.lastPathComponent
                != stem.lastPathComponent + member {
                // Nothing to verify the content by, so the only remaining
                // evidence that the right artifact arrived is that the
                // transfer named it the right thing (the historical guard).
                throw LocalizationError.downloadDidNotLand(
                    stem.lastPathComponent + member)
            }
            staged[member] = landed
        }
        let notice = fetchedNotice(relativeID: relativeID,
                                   verification: verification)
        if let notice { warn(notice) }

        // Publish. Tensor first, sidecar last: BOTH catalogs key discovery on
        // the `.json` sidecar and require the tensor beside it, so a crash
        // between the two renames leaves an undiscoverable stray rather than
        // a half-formed artifact. Only MISSING members are published — a
        // member already on disk verified above and is left untouched.
        //
        // A failed move ROLLS BACK the ones before it. "The workspace exactly
        // as it was" has to hold for a partial publish too, or the promise is
        // only about the paths we happened to test.
        var published: [(destination: URL, origin: URL)] = []
        for member in members {
            guard let source = staged[member] else { continue }
            let destination = URL(filePath: stem.path + member)
            do {
                try FileManager.default.moveItem(at: source, to: destination)
            } catch {
                for move in published.reversed() {
                    // Back to staging where possible (the bytes survive for
                    // the `defer` to clean up); otherwise remove, because a
                    // published half-pair is the one state we must not leave.
                    do {
                        try FileManager.default.moveItem(
                            at: move.destination, to: move.origin)
                    } catch {
                        try? FileManager.default.removeItem(at: move.destination)
                    }
                }
                throw LocalizationError.publishFailed(
                    member: stem.lastPathComponent + member,
                    reason: (error as NSError).localizedDescription)
            }
            published.append((destination, source))
        }
        guard isLocal(relativeID: relativeID, workspaceRoot: workspaceRoot) else {
            throw LocalizationError.downloadDidNotLand(relativeID)
        }
        return Outcome(relativeID: relativeID,
                       verification: verification, notice: notice)
    }

    // MARK: - Disclosure

    /// What the advertised digests add up to. Members with no digest are the
    /// unverified ones; `members` order is stable, so the report is too.
    static func verification(for expected: [String: String?]) -> Verification {
        let unbacked = members.filter { (expected[$0] ?? nil) == nil }
        if unbacked.isEmpty { return .fullyVerified }
        if unbacked.count == members.count { return .unverified }
        return .partiallyVerified(missing: unbacked.sorted())
    }

    /// Disclosure for a pair that was already on disk (nothing fetched).
    static func keptNotice(relativeID: String,
                           verification: Verification) -> String? {
        switch verification {
        case .fullyVerified:
            return nil
        case .unverified:
            return "kept the existing local pair for '\(relativeID)' "
                + "unverified: this server's catalog advertises no artifact "
                + "hashes, so nothing binds it to the server's copy — update "
                + "the server to verify localized vectors"
        case .partiallyVerified(let missing):
            return "kept the existing local pair for '\(relativeID)' only "
                + "PARTIALLY verified: this server's catalog advertises no "
                + "hash for \(missing.joined(separator: ", ")), so that half "
                + "is not bound to the server's copy — update the server to "
                + "verify localized vectors"
        }
    }

    /// Disclosure for a pair this call fetched.
    static func fetchedNotice(relativeID: String,
                              verification: Verification) -> String? {
        switch verification {
        case .fullyVerified:
            return nil
        case .unverified:
            return "localized '\(relativeID)' UNVERIFIED: this server's "
                + "catalog advertises no artifact hashes, so nothing binds "
                + "the fetched bytes to the catalog row — update the server "
                + "to verify localized vectors"
        case .partiallyVerified(let missing):
            return "localized '\(relativeID)' only PARTIALLY verified: this "
                + "server's catalog advertises no hash for "
                + "\(missing.joined(separator: ", ")), so those bytes are not "
                + "bound to the catalog row — update the server to verify "
                + "localized vectors"
        }
    }

    /// The save-time disclosure for an agent whose vectors localized without
    /// full hash verification. Concept names, not paths — the researcher
    /// picked concepts, and the row they picked from is what could not vouch
    /// for the bytes.
    ///
    /// It leads with "saved" because the save DID happen: the alternative
    /// (refusing) was considered and rejected, so the line has to report a
    /// completed action with a caveat, never imply a failure.
    public static func saveDisclosure(concepts: [String]) -> String {
        let names = concepts.joined(separator: ", ")
        let subject = concepts.count == 1 ? "vector" : "vectors"
        return "saved — but the \(subject) \(names) localized WITHOUT full "
            + "hash verification: this server's catalog does not advertise "
            + "every artifact digest, so the workspace copy is not bound to "
            + "the server's. Update the server to verify localized vectors."
    }

    // MARK: - Internals

    /// One in-flight localization per target stem.
    ///
    /// Not a general file lock (nothing outside this process is coordinated,
    /// and nothing else writes these paths): it exists so this type's own
    /// check-then-publish cannot interleave with itself. FIFO by
    /// construction — a waiter is resumed by the holder's `unlock`, which
    /// hands ownership over rather than releasing it, so no waiter starves
    /// and no third caller can jump the queue.
    private actor StemLock {
        static let shared = StemLock()
        private var held: Set<String> = []
        private var waiters: [String: [CheckedContinuation<Void, Never>]] = [:]

        func lock(_ key: String) async {
            guard held.contains(key) else {
                held.insert(key)
                return
            }
            await withCheckedContinuation { continuation in
                waiters[key, default: []].append(continuation)
            }
        }

        func unlock(_ key: String) {
            guard var queue = waiters[key], !queue.isEmpty else {
                held.remove(key)
                return
            }
            let next = queue.removeFirst()
            waiters[key] = queue.isEmpty ? nil : queue
            next.resume()   // ownership transfers; `held` stays set
        }
    }

    /// The real (symlink-resolved) stem `relativeID` names, refusing anything
    /// that escapes the workspace's `runs/` tree. The leaf does not exist yet,
    /// so resolution runs on the deepest EXISTING ancestor and the remaining
    /// components are re-appended; comparison is by path COMPONENT, not string
    /// prefix (a sibling `runs-scratch/` must not pass a `runs` prefix test).
    static func resolvedStem(relativeID: String,
                             workspaceRoot: URL) throws -> URL {
        let base = workspaceRoot.resolvingSymlinksInPath().standardizedFileURL
        let runsRoot = base.appending(component: "runs")
            .resolvingSymlinksInPath().standardizedFileURL
        var existing = workspaceRoot.appending(path: relativeID)
            .standardizedFileURL
        var trailing: [String] = []
        while !FileManager.default.fileExists(atPath: existing.path) {
            let parent = existing.deletingLastPathComponent().standardizedFileURL
            guard parent.path != existing.path else { break }
            trailing.insert(existing.lastPathComponent, at: 0)
            existing = parent
        }
        var resolved = existing.resolvingSymlinksInPath().standardizedFileURL
        for component in trailing {
            resolved = resolved.appending(component: component)
        }
        let root = runsRoot.pathComponents
        guard resolved.pathComponents.count > root.count,
            Array(resolved.pathComponents.prefix(root.count)) == root
        else {
            throw LocalizationError.escapesRunsTree(
                id: relativeID, resolved: resolved.path)
        }
        return resolved
    }

    private static func isReadableFile(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: url.path, isDirectory: &isDirectory)
        return exists && !isDirectory.boolValue
    }

    /// Streamed so a large safetensors file is never fully resident.
    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
