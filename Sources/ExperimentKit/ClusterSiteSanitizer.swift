import Foundation

// =============================================================================
// The three-way split, enforced at the one boundary that matters (maintainer
// ruling, 2026-08-21).
//
// A saved cluster site is three different KINDS of fact, and they belong in
// three different places:
//
//   1. PROFILE — host, login, scheduler/partition/gres data, storage roots,
//      the site's id and display name. Non-secret, machine-independent, and
//      the whole point of the shared registry. Goes in
//      `Sites/cluster-sites/<site-id>.json`, which is a git repository the
//      researcher syncs between their machines.
//   2. SECRETS — bearer tokens, the Hugging Face token. Keychain only, per
//      machine (service `SteerLabCluster`, unchanged). A new machine
//      re-prompts once, and that is a feature: a credential that travelled in
//      a git repository would be a credential in every clone forever.
//   3. RUNTIME STATE — the last endpoint a tunnel produced, the last observed
//      server build, the app registry's entry UUID, when we last connected.
//      Per machine, mutated by connecting, and meaningless on another Mac.
//      Lives in `~/Library/Application Support/SteerLab/site-runtime.json`.
//
// `sites export` has always drawn line 1 by hand — it writes `site.profile`
// and nothing else. This type is that same rule promoted to THE sanitizer, so
// the export path and the registry-write path are one code path rather than
// two that can drift. Every byte that reaches a Sites file goes through
// `encodedSiteFile`; every byte that reaches an export goes through
// `encodedExport`; both `verify` first and REFUSE rather than write.
// =============================================================================

/// The one boundary between what SteerLab knows about a site and what lands in
/// a file anyone else can read.
public enum ClusterSiteSanitizer {

    // MARK: The forbidden vocabulary

    /// Key fragments that mark a value as a SECRET. Matched on the lowercased
    /// key name anywhere in the document tree.
    static let secretFragments = [
        "token", "password", "passphrase", "secret", "credential", "apikey",
        "privatekey",
    ]

    /// Suffixes that make a secret-shaped key a REFERENCE rather than a
    /// secret. `storage.tokenFilePath` is where the cluster keeps its bearer
    /// token — a path, and legitimate profile data; `token` would be the token
    /// itself. Without this the sanitizer would refuse every honest profile,
    /// and a guard that refuses everything gets deleted.
    static let referenceSuffixes = [
        "path", "file", "filepath", "identity", "url", "dir", "directory",
        "root", "name", "required", "prefix",
    ]

    /// Keys that belong to the PER-MACHINE runtime cache. A Sites file holding
    /// any of these would make a connect/disconnect cycle dirty the shared
    /// registry.
    static let runtimeKeys: Set<String> = [
        "lastendpoint", "lastserverbuild", "lastconnectedat", "lastusedat",
        "legacyentryid", "localport", "tunnelstate", "connectionstatus",
        "migratedfromuserdefaultsat",
    ]

    /// Whether a key name is a secret rather than a reference to one.
    static func isSecretKey(_ key: String) -> Bool {
        let lowered = key.lowercased()
        guard secretFragments.contains(where: { lowered.contains($0) }) else {
            return false
        }
        return !referenceSuffixes.contains { lowered.hasSuffix($0) }
    }

    /// Every key in `data`'s JSON tree that must never appear in a Sites file
    /// or an export, sorted and deduplicated so the refusal reads the same way
    /// twice.
    public static func offendingKeys(in data: Data) -> [String] {
        guard let object = try? JSONSerialization.jsonObject(with: data) else {
            return []
        }
        var found: Set<String> = []
        func walk(_ value: Any) {
            if let dictionary = value as? [String: Any] {
                for (key, nested) in dictionary {
                    if isSecretKey(key) || runtimeKeys.contains(key.lowercased()) {
                        found.insert(key)
                    }
                    walk(nested)
                }
            } else if let array = value as? [Any] {
                for nested in array { walk(nested) }
            }
        }
        walk(object)
        return found.sorted()
    }

    /// Refuse rather than write. The caller never gets bytes it could have
    /// persisted by mistake.
    static func verify(_ data: Data, siteID: String) throws {
        let offenders = offendingKeys(in: data)
        guard offenders.isEmpty else {
            throw ClusterLifecycleError.siteFileWouldLeak(
                siteID: siteID, keys: offenders)
        }
    }

    // MARK: Encoding

    /// The bytes of one canonical Sites file — which are EXACTLY the bytes of
    /// an export, and deliberately so.
    ///
    /// A registry file is a bare `ClusterSiteProfile` document: `sites export`
    /// produces one, `sites import` consumes one, and dropping an exported
    /// file into `Sites/cluster-sites/` is a legitimate way to add a site.
    /// Wrapping the profile in a per-file envelope would have made the two
    /// formats almost-but-not-quite the same, which is the shape of a bug a
    /// researcher discovers by hand-editing.
    ///
    /// The file's NAME carries the site id; nothing inside it duplicates
    /// that, so renaming the file renames the site and there is no second
    /// place for the id to be wrong. Everything else about a site is either a
    /// secret (Keychain) or runtime (the per-machine cache) and is checked for
    /// here rather than trusted not to appear.
    ///
    /// Stable key order, pretty printed, one trailing newline: a human edits
    /// these and git diffs them.
    public static func encodedSiteFile(for record: ClusterSiteRecord) throws -> Data {
        try encodedExport(for: record)
    }

    /// The bytes `sites export` and the app's Export… produce.
    public static func encodedExport(for record: ClusterSiteRecord) throws -> Data {
        var data: Data
        do {
            data = try record.profile.encoded()
        } catch {
            throw ClusterLifecycleError.storeUnwritable(
                "\(record.id): \(error.localizedDescription)")
        }
        try verify(data, siteID: record.id)
        // A trailing newline is not decoration: without it every editor and
        // every `git diff` reports "\ No newline at end of file" on a
        // registry the researcher is expected to read.
        if data.last != UInt8(ascii: "\n") { data.append(UInt8(ascii: "\n")) }
        return data
    }
}
