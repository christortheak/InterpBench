import Foundation
#if canImport(Security)
import Security
#endif

/// Storage seam so the store rule is pure and testable in-memory (Keychain
/// access in a CLI test runner is prompt-bound/flaky — same convention as
/// `AnthropicKeyStorage`).
public protocol JudgeKeyStorage {
    func read() -> Data?
    func write(_ data: Data) -> Bool
    func delete()
}

/// The EXTERNAL judge key (key-custody design, seamless-pipeline extension
/// 2026-07-19): one dedicated, ideally spend-CAPPED key — OpenRouter or
/// Anthropic — that enables inline external judging on the cluster.
///
/// Custody contract:
/// - At rest on this Mac: the Keychain only (this store), never UserDefaults.
/// - On the cluster: `~/.steerlab/judge-key` (mode 600), pushed over SSH
///   stdin at EVERY connect and REMOVED at connect when cleared here —
///   deletion propagates; a cleared key cannot stay live on the cluster.
/// - Never in Slurm bundles or job env: the server refuses secret-shaped
///   env keys; only the file PATH travels.
/// - This is distinct from the personal Claude key (`AnthropicKeyStore`),
///   which never leaves this Mac. Pushing THAT key is deliberately
///   unsupported; provision a separate capped key for the cluster.
public enum JudgeKeyStore {

    public static let keychainService = "SteerLabJudgeKey"
    public static let keychainAccount = "external-judge"
    /// Where the key lands on the cluster — the server reads the same path
    /// by default (`judge_credentials.DEFAULT_KEY_PATH`).
    public static let remotePath = "~/.steerlab/judge-key"
    public static let validKinds = ["openrouter", "anthropic"]

    /// What's stored (and what the remote file contains): kind + key.
    public struct StoredKey: Codable, Sendable, Equatable {
        public var kind: String  // "openrouter" | "anthropic"
        public var key: String

        public init(kind: String, key: String) {
            self.kind = kind
            self.key = key
        }
    }

    public static func stored(
        storage: any JudgeKeyStorage = KeychainJudgeKeyStorage()
    ) -> StoredKey? {
        guard let data = storage.read(),
            let decoded = try? JSONDecoder().decode(StoredKey.self, from: data),
            !decoded.key.isEmpty
        else { return nil }
        return decoded
    }

    /// Save (empty/whitespace key means DELETE — the caller's Clear).
    /// An unknown kind is refused by returning false; the UI constrains
    /// kinds, so this only guards programmatic misuse.
    @discardableResult
    public static func save(
        kind: String, key: String,
        storage: any JudgeKeyStorage = KeychainJudgeKeyStorage()
    ) -> Bool {
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            storage.delete()
            return true
        }
        let trimmedKind = kind.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard validKinds.contains(trimmedKind) else { return false }
        guard let data = try? JSONEncoder().encode(
            StoredKey(kind: trimmedKind, key: trimmedKey))
        else { return false }
        return storage.write(data)
    }

    public static func delete(
        storage: any JudgeKeyStorage = KeychainJudgeKeyStorage()
    ) {
        storage.delete()
    }

    /// The EXACT file content pushed to the cluster — the JSON contract the
    /// server's `judge_credentials._read_key_file` parses. Deterministic
    /// (sorted keys) + newline-terminated so repeated pushes of the same
    /// key are byte-identical.
    public static func remoteFileContents(_ stored: StoredKey) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(stored)) ?? Data()
        return String(decoding: data, as: UTF8.self) + "\n"
    }

    /// The key LOCAL judging should use for a judge kind: the matching env
    /// var wins (`OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`), else the
    /// stored key when its kind matches. The Mac's deferred sweep judging
    /// resolves OpenRouter judges here; Claude judges keep their historical
    /// path (`ClaudeStimulusGenerator.apiKey`) and fall back to this store.
    public static func resolveKey(
        kind: String,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        storage: any JudgeKeyStorage = KeychainJudgeKeyStorage()
    ) -> String? {
        let envVar = kind == "openrouter"
            ? "OPENROUTER_API_KEY" : "ANTHROPIC_API_KEY"
        if let key = environment[envVar], !key.isEmpty {
            return key
        }
        guard let stored = stored(storage: storage) else { return nil }
        let matches = kind == "openrouter"
            ? stored.kind == "openrouter" : stored.kind == "anthropic"
        return matches ? stored.key : nil
    }
}

/// Live backend: Security.framework generic passwords, same shape as
/// `ClusterTokenStore` / `KeychainAnthropicKeyStorage`.
public struct KeychainJudgeKeyStorage: JudgeKeyStorage {

    public init() {}

    public func read() -> Data? {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: JudgeKeyStore.keychainService,
            kSecAttrAccount as String: JudgeKeyStore.keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data
        else { return nil }
        return data
        #else
        return nil
        #endif
    }

    public func write(_ data: Data) -> Bool {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: JudgeKeyStore.keychainService,
            kSecAttrAccount as String: JudgeKeyStore.keychainAccount,
        ]
        SecItemDelete(query as CFDictionary)
        var insert = query
        insert[kSecValueData as String] = data
        return SecItemAdd(insert as CFDictionary, nil) == errSecSuccess
        #else
        return false
        #endif
    }

    public func delete() {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: JudgeKeyStore.keychainService,
            kSecAttrAccount as String: JudgeKeyStore.keychainAccount,
        ]
        SecItemDelete(query as CFDictionary)
        #endif
    }
}
