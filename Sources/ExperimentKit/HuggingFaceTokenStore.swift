import Foundation
#if canImport(Security)
import Security
#endif

/// Storage seam so the store's rules are testable with an in-memory backend —
/// Keychain access in a CLI test runner is prompt-bound/flaky, so tests never
/// touch the real Keychain (same seam as `AnthropicKeyStorage`).
public protocol HuggingFaceTokenStorage {
    func readKeychain() -> String?
    /// "Is a token stored?" answered without reading the value. On the real
    /// Keychain a DATA read is ACL-gated per binary identity and can raise a
    /// blocking password prompt; a badge that only says stored/not-stored must
    /// not be able to raise one. In-memory backends get the obvious default.
    func keychainPresence() -> Bool
    func writeKeychain(_ token: String) -> Bool
    func deleteKeychain()
}

extension HuggingFaceTokenStorage {
    /// Default for non-Keychain backends, where reading costs nothing and no
    /// ACL exists. `KeychainHuggingFaceTokenStorage` overrides it.
    public func keychainPresence() -> Bool {
        !(readKeychain() ?? "").isEmpty
    }
}

/// Keychain-backed Hugging Face token, MATERIALIZED to the hub's native
/// location so it actually authenticates something.
///
/// The gap this closes (2026-07-30): the cluster connection menu could
/// install a token on the CLUSTER's `$HF_HOME/token` over SSH, but nothing
/// wrote the local Mac's copy — so the local server and CLI could not
/// download gated repos (`google/gemma-3-*`) even though the researcher had
/// "entered their token in the app". Saving here writes BOTH at-rest copies
/// this Mac owns:
///
/// - the Keychain (rotation without re-pasting, and what the UI reports), and
/// - `$HF_HOME/token` (default `~/.cache/huggingface/token`, mode 600) — the
///   file `huggingface_hub` reads natively, so local downloads, the local
///   server, and the calibration scripts need no further plumbing.
///
/// Clearing deletes the Keychain copy and removes the materialized file ONLY
/// when its content matches the cleared token — a token installed
/// independently by `hf auth login` is not ours to delete.
///
/// The environment still wins at read time (`HF_TOKEN` /
/// `HUGGING_FACE_HUB_TOKEN`), matching `huggingface_hub`'s own resolution;
/// the settings UI states this rather than fighting it.
public enum HuggingFaceTokenStore {

    /// Generic-password service, sibling to "SteerLabAnthropic".
    public static let keychainService = "SteerLabHuggingFace"
    public static let keychainAccount = "huggingface.co"

    // MARK: - Resolution

    public static func environmentOverrides(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        !(environment["HF_TOKEN"] ?? "").isEmpty
            || !(environment["HUGGING_FACE_HUB_TOKEN"] ?? "").isEmpty
    }

    public static func storedToken(
        storage: any HuggingFaceTokenStorage = KeychainHuggingFaceTokenStorage()
    ) -> String? {
        guard let token = storage.readKeychain(), !token.isEmpty else { return nil }
        return token
    }

    /// Presence for the settings badge — routed through `keychainPresence()`
    /// so it never reads the secret. `save` refuses to store an empty value
    /// (empty means CLEAR), so on the real Keychain "an item exists" and "a
    /// usable token is stored" are the same claim.
    public static func hasStoredToken(
        storage: any HuggingFaceTokenStorage = KeychainHuggingFaceTokenStorage()
    ) -> Bool {
        storage.keychainPresence()
    }

    /// Where the hub reads its token: `$HF_HOME/token`, defaulting to
    /// `~/.cache/huggingface/token` — the same resolution `huggingface_hub`
    /// applies, so what this writes is what `from_pretrained` finds.
    public static func hubTokenURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        let hfHome = (environment["HF_HOME"] ?? "").trimmingCharacters(
            in: .whitespacesAndNewlines)
        let root = hfHome.isEmpty
            ? home.appending(components: ".cache", "huggingface")
            : URL(fileURLWithPath: hfHome)
        return root.appending(component: "token")
    }

    /// True when the hub token file exists (whoever wrote it).
    public static func hubFileExists(at url: URL? = nil) -> Bool {
        FileManager.default.fileExists(atPath: (url ?? hubTokenURL()).path)
    }

    // MARK: - Mutation

    /// Save to the Keychain AND materialize the hub file. Empty/whitespace
    /// means CLEAR (both copies, file only if it is ours — see above).
    /// Returns a non-nil error message when the FILE write failed: the
    /// Keychain save still stands, but a silent file failure would rebuild
    /// exactly the gap this store exists to close.
    @discardableResult
    public static func save(
        _ token: String,
        storage: any HuggingFaceTokenStorage = KeychainHuggingFaceTokenStorage(),
        hubFile: URL? = nil
    ) -> String? {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = hubFile ?? hubTokenURL()
        if trimmed.isEmpty {
            let previous = storedToken(storage: storage)
            storage.deleteKeychain()
            removeMaterialized(at: url, ifMatching: previous)
            return nil
        }
        _ = storage.writeKeychain(trimmed)
        do {
            try materialize(trimmed, at: url)
            return nil
        } catch {
            return "token saved to the Keychain, but writing \(url.path) "
                + "failed (\(error.localizedDescription)) — gated downloads "
                + "will not authenticate until it exists"
        }
    }

    /// Write the hub token file with owner-only permissions (mode 600 —
    /// what `hf auth login` itself sets).
    static func materialize(_ token: String, at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(token.utf8).write(to: url, options: .atomic)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    /// Remove the hub file only when its content matches `token` — a file
    /// written by `hf auth login` with a DIFFERENT token is not ours.
    static func removeMaterialized(at url: URL, ifMatching token: String?) {
        guard let token, !token.isEmpty,
            let data = try? Data(contentsOf: url),
            String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) == token
        else { return }
        try? FileManager.default.removeItem(at: url)
    }
}

/// The live backend: Security.framework generic passwords, same shape as
/// `KeychainAnthropicKeyStorage`.
public struct KeychainHuggingFaceTokenStorage: HuggingFaceTokenStorage {

    public init() {}

    private var query: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: HuggingFaceTokenStore.keychainService,
            kSecAttrAccount as String: HuggingFaceTokenStore.keychainAccount,
        ]
    }

    public func readKeychain() -> String? {
        #if canImport(Security)
        var read = query
        read[kSecReturnData as String] = true
        read[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(read as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
        #else
        return nil
        #endif
    }

    /// Attribute-only match: `kSecReturnAttributes`, never `kSecReturnData`,
    /// so the value's ACL is not consulted and no password dialog can appear.
    /// No `kSecUseAuthenticationUI` flag is involved — this asks a different
    /// question rather than suppressing an answer to the same one.
    public func keychainPresence() -> Bool {
        #if canImport(Security)
        var read = query
        read[kSecReturnAttributes as String] = true
        read[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        return SecItemCopyMatching(read as CFDictionary, &item) == errSecSuccess
        #else
        return false
        #endif
    }

    public func writeKeychain(_ token: String) -> Bool {
        #if canImport(Security)
        SecItemDelete(query as CFDictionary)
        var insert = query
        insert[kSecValueData as String] = Data(token.utf8)
        return SecItemAdd(insert as CFDictionary, nil) == errSecSuccess
        #else
        return false
        #endif
    }

    public func deleteKeychain() {
        #if canImport(Security)
        SecItemDelete(query as CFDictionary)
        #endif
    }
}
