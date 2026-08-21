import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the Anthropic key store's resolution + migration RULE
/// against an in-memory backend — the real Keychain is prompt-bound/flaky in
/// a CLI test runner, so the Security.framework backend
/// (`KeychainAnthropicKeyStorage`) is exercised only by using the app; these
/// tests pin the logic every path shares:
/// - resolution order: ANTHROPIC_API_KEY env (wins) → Keychain → legacy
///   UserDefaults migration
/// - one-time silent migration: legacy value moves into the Keychain and the
///   plaintext copy is DELETED (kept for retry when the write fails)
/// - save semantics: trimmed write; empty/whitespace = delete; the legacy
///   slot is cleared on every explicit save
struct AnthropicKeyStoreTests {

    /// In-memory stand-in for Keychain + legacy UserDefaults. `writeSucceeds`
    /// simulates a failing Keychain so the migration-retry rule is testable.
    private final class MemoryStorage: AnthropicKeyStorage {
        var keychain: String?
        var legacy: String?
        var writeSucceeds = true
        var keychainDeletes = 0
        var legacyDeletes = 0

        init(keychain: String? = nil, legacy: String? = nil) {
            self.keychain = keychain
            self.legacy = legacy
        }

        func readKeychain() -> String? { keychain }
        func writeKeychain(_ key: String) -> Bool {
            guard writeSucceeds else { return false }
            keychain = key
            return true
        }
        func deleteKeychain() {
            keychain = nil
            keychainDeletes += 1
        }
        func readLegacyDefaults() -> String? { legacy }
        func deleteLegacyDefaults() {
            legacy = nil
            legacyDeletes += 1
        }
    }

    // MARK: Resolution order

    @Test func environmentWinsOverKeychainAndLegacy() {
        let storage = MemoryStorage(keychain: "kc-key", legacy: "legacy-key")
        let resolved = AnthropicKeyStore.resolve(
            environment: ["ANTHROPIC_API_KEY": "env-key"], storage: storage)
        #expect(resolved == "env-key")
        // The env short-circuit touches no storage.
        #expect(storage.keychain == "kc-key")
        #expect(storage.legacy == "legacy-key")
    }

    @Test func emptyEnvironmentValueIsIgnored() {
        let storage = MemoryStorage(keychain: "kc-key")
        let resolved = AnthropicKeyStore.resolve(
            environment: ["ANTHROPIC_API_KEY": ""], storage: storage)
        #expect(resolved == "kc-key")
    }

    @Test func keychainValueResolvesWithoutEnvironment() {
        let storage = MemoryStorage(keychain: "kc-key")
        #expect(AnthropicKeyStore.resolve(environment: [:], storage: storage) == "kc-key")
    }

    @Test func nothingAnywhereResolvesNil() {
        let storage = MemoryStorage()
        #expect(AnthropicKeyStore.resolve(environment: [:], storage: storage) == nil)
        #expect(AnthropicKeyStore.hasStoredKey(storage: storage) == false)
    }

    @Test func environmentOverridesFlag() {
        #expect(AnthropicKeyStore.environmentOverrides(
            environment: ["ANTHROPIC_API_KEY": "x"]))
        #expect(!AnthropicKeyStore.environmentOverrides(
            environment: ["ANTHROPIC_API_KEY": ""]))
        #expect(!AnthropicKeyStore.environmentOverrides(environment: [:]))
    }

    // MARK: Migration

    @Test func legacyValueMigratesIntoKeychainAndIsDeleted() {
        let storage = MemoryStorage(legacy: "legacy-key")
        let resolved = AnthropicKeyStore.storedKey(storage: storage)
        #expect(resolved == "legacy-key")
        #expect(storage.keychain == "legacy-key")
        #expect(storage.legacy == nil)
        #expect(storage.legacyDeletes == 1)
    }

    @Test func migrationTrimsWhitespace() {
        let storage = MemoryStorage(legacy: "  legacy-key\n")
        #expect(AnthropicKeyStore.storedKey(storage: storage) == "legacy-key")
        #expect(storage.keychain == "legacy-key")
        #expect(storage.legacy == nil)
    }

    @Test func failedMigrationWriteKeepsLegacyForRetryButStillReturnsKey() {
        let storage = MemoryStorage(legacy: "legacy-key")
        storage.writeSucceeds = false
        // Usable this session…
        #expect(AnthropicKeyStore.storedKey(storage: storage) == "legacy-key")
        // …but the plaintext slot survives so the next launch can retry.
        #expect(storage.keychain == nil)
        #expect(storage.legacy == "legacy-key")
        #expect(storage.legacyDeletes == 0)
    }

    @Test func emptyLegacyValueIsCleanedUpNotMigrated() {
        let storage = MemoryStorage(legacy: "   ")
        #expect(AnthropicKeyStore.storedKey(storage: storage) == nil)
        #expect(storage.keychain == nil)
        #expect(storage.legacy == nil)
    }

    @Test func keychainHitSkipsMigration() {
        let storage = MemoryStorage(keychain: "kc-key", legacy: "legacy-key")
        #expect(AnthropicKeyStore.storedKey(storage: storage) == "kc-key")
        // Legacy consulted only on a Keychain miss.
        #expect(storage.legacy == "legacy-key")
    }

    @Test func migrationRunsOnce() {
        let storage = MemoryStorage(legacy: "legacy-key")
        _ = AnthropicKeyStore.storedKey(storage: storage)
        _ = AnthropicKeyStore.storedKey(storage: storage)
        #expect(storage.keychain == "legacy-key")
        #expect(storage.legacyDeletes == 1)
    }

    // MARK: Save semantics

    @Test func saveRoundTrip() {
        let storage = MemoryStorage()
        AnthropicKeyStore.save("  sk-ant-test \n", storage: storage)
        #expect(storage.keychain == "sk-ant-test")
        #expect(AnthropicKeyStore.storedKey(storage: storage) == "sk-ant-test")
    }

    @Test func saveEmptyDeletes() {
        let storage = MemoryStorage(keychain: "kc-key")
        AnthropicKeyStore.save("", storage: storage)
        #expect(storage.keychain == nil)
        #expect(storage.keychainDeletes == 1)
        #expect(AnthropicKeyStore.storedKey(storage: storage) == nil)
    }

    @Test func saveWhitespaceOnlyDeletes() {
        let storage = MemoryStorage(keychain: "kc-key")
        AnthropicKeyStore.save("   \n", storage: storage)
        #expect(storage.keychain == nil)
    }

    @Test func saveClearsLegacyPlaintextSlot() {
        let storage = MemoryStorage(legacy: "legacy-key")
        AnthropicKeyStore.save("new-key", storage: storage)
        #expect(storage.keychain == "new-key")
        #expect(storage.legacy == nil)
    }

    @Test func clearAlsoClearsLegacyPlaintextSlot() {
        let storage = MemoryStorage(keychain: "kc-key", legacy: "legacy-key")
        AnthropicKeyStore.save("", storage: storage)
        #expect(storage.keychain == nil)
        #expect(storage.legacy == nil)
    }
}
