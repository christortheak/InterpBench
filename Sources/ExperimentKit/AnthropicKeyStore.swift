import Foundation
#if canImport(Security)
import Security
#endif

/// Storage seam for `AnthropicKeyStore` so the resolution + migration RULE is
/// pure and testable with an in-memory backend — Keychain access in a CLI
/// test runner is prompt-bound/flaky, so tests never touch the real Keychain.
///
/// `writeKeychain` returns whether the write stuck; a failed migration write
/// keeps the legacy value in place so a later launch can retry.
public protocol AnthropicKeyStorage {
    func readKeychain() -> String?
    func writeKeychain(_ key: String) -> Bool
    func deleteKeychain()
    func readLegacyDefaults() -> String?
    func deleteLegacyDefaults()
}

/// Keychain-backed storage for the researcher's Anthropic API key, replacing
/// the pre-2026-07-08 plaintext UserDefaults slot. Mirrors the
/// `ClusterTokenStore` conventions (Security.framework generic-password
/// items, a fixed service string, a stable account key — no entitlement
/// assumptions beyond what the server-token store already makes).
///
/// Resolution order (`resolve`): `ANTHROPIC_API_KEY` from the environment
/// wins (unchanged behavior); otherwise the Keychain; otherwise a one-time
/// silent migration of the legacy UserDefaults value INTO the Keychain,
/// deleting the plaintext copy on success.
///
/// The key is read on THIS Mac only — by Claude judges, stimulus generation,
/// and sweep credential preflights. Nothing here travels, BY POLICY
/// (2026-07-18 key-custody decision): the key never goes to the cluster in
/// any form — not through the tunnel, not in a Slurm bundle (the server
/// refuses secret-shaped env keys at bundle creation). Cluster generations
/// are judged on this Mac after download; cluster-side judging uses
/// local-model judges.
public enum AnthropicKeyStore {

    /// Generic-password service, sibling to `ClusterTokenStore`'s
    /// "SteerLabCluster".
    public static let keychainService = "SteerLabAnthropic"
    /// Account key: the API host, mirroring the token store's host-keyed
    /// account convention.
    public static let keychainAccount = "api.anthropic.com"
    /// The legacy plaintext slot (`UserDefaults` "AnthropicAPIKey") —
    /// consulted only to migrate, deleted once the Keychain holds the key.
    public static let legacyDefaultsKey = "AnthropicAPIKey"

    // MARK: - Resolution

    /// The key generation/judging should use: environment → Keychain →
    /// one-time legacy migration. Nil when no key exists anywhere.
    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        storage: any AnthropicKeyStorage = KeychainAnthropicKeyStorage()
    ) -> String? {
        if let key = environment["ANTHROPIC_API_KEY"], !key.isEmpty {
            return key
        }
        return storedKey(storage: storage)
    }

    /// True when `ANTHROPIC_API_KEY` is set (non-empty) in the environment —
    /// it then wins over any stored key (settings UI states this).
    public static func environmentOverrides(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        !(environment["ANTHROPIC_API_KEY"] ?? "").isEmpty
    }

    /// The at-rest key (never the environment): the Keychain value, or the
    /// one-time silent migration of the legacy plaintext value. Migration
    /// deletes the UserDefaults copy only after the Keychain write succeeds;
    /// on a failed write the legacy value is kept for a retry next launch
    /// (the key is still returned for this session). An empty/whitespace
    /// legacy value is junk: deleted, nothing migrated.
    public static func storedKey(
        storage: any AnthropicKeyStorage = KeychainAnthropicKeyStorage()
    ) -> String? {
        if let key = storage.readKeychain(), !key.isEmpty {
            return key
        }
        guard let legacy = storage.readLegacyDefaults() else { return nil }
        let trimmed = legacy.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            storage.deleteLegacyDefaults()
            return nil
        }
        if storage.writeKeychain(trimmed) {
            storage.deleteLegacyDefaults()
        }
        return trimmed
    }

    /// Whether a key is stored at rest (Keychain, after any migration).
    public static func hasStoredKey(
        storage: any AnthropicKeyStorage = KeychainAnthropicKeyStorage()
    ) -> Bool {
        storedKey(storage: storage) != nil
    }

    // MARK: - Mutation

    /// Save the key to the Keychain; empty/whitespace-only means DELETE.
    /// Either way the legacy plaintext slot is cleared — after any explicit
    /// save, the Keychain is the only at-rest location.
    public static func save(
        _ key: String,
        storage: any AnthropicKeyStorage = KeychainAnthropicKeyStorage()
    ) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            storage.deleteKeychain()
        } else {
            _ = storage.writeKeychain(trimmed)
        }
        storage.deleteLegacyDefaults()
    }
}

/// The live backend: Security.framework generic passwords (same shape as
/// `ClusterTokenStore`) + `UserDefaults.standard` for the legacy slot.
public struct KeychainAnthropicKeyStorage: AnthropicKeyStorage {

    public init() {}

    public func readKeychain() -> String? {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: AnthropicKeyStore.keychainService,
            kSecAttrAccount as String: AnthropicKeyStore.keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
        #else
        return nil
        #endif
    }

    public func writeKeychain(_ key: String) -> Bool {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: AnthropicKeyStore.keychainService,
            kSecAttrAccount as String: AnthropicKeyStore.keychainAccount,
        ]
        SecItemDelete(query as CFDictionary)
        var insert = query
        insert[kSecValueData as String] = Data(key.utf8)
        return SecItemAdd(insert as CFDictionary, nil) == errSecSuccess
        #else
        return false
        #endif
    }

    public func deleteKeychain() {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: AnthropicKeyStore.keychainService,
            kSecAttrAccount as String: AnthropicKeyStore.keychainAccount,
        ]
        SecItemDelete(query as CFDictionary)
        #endif
    }

    public func readLegacyDefaults() -> String? {
        UserDefaults.standard.string(forKey: AnthropicKeyStore.legacyDefaultsKey)
    }

    public func deleteLegacyDefaults() {
        UserDefaults.standard.removeObject(forKey: AnthropicKeyStore.legacyDefaultsKey)
    }
}
