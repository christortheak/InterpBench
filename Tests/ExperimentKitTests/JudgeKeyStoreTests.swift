import Foundation
import Testing

@testable import ExperimentKit

/// The external judge key's store rule (key custody, 2026-07-19), against
/// an in-memory backend — tests never touch the real Keychain. The remote
/// file contents are a cross-engine contract with the server's
/// `judge_credentials._read_key_file`.
struct JudgeKeyStoreTests {

    private final class MemoryStorage: JudgeKeyStorage, @unchecked Sendable {
        var data: Data?
        var writeSucceeds = true
        func read() -> Data? { data }
        func write(_ new: Data) -> Bool {
            guard writeSucceeds else { return false }
            data = new
            return true
        }
        func delete() { data = nil }
    }

    @Test func saveReadRoundTrip() {
        let storage = MemoryStorage()
        #expect(JudgeKeyStore.save(
            kind: "openrouter", key: " sk-or-x ", storage: storage))
        let stored = JudgeKeyStore.stored(storage: storage)
        #expect(stored == .init(kind: "openrouter", key: "sk-or-x"))
    }

    @Test func emptyKeyMeansDelete() {
        let storage = MemoryStorage()
        JudgeKeyStore.save(kind: "openrouter", key: "sk-or-x", storage: storage)
        #expect(JudgeKeyStore.save(kind: "openrouter", key: "  ", storage: storage))
        #expect(JudgeKeyStore.stored(storage: storage) == nil)
    }

    @Test func unknownKindIsRefused() {
        let storage = MemoryStorage()
        #expect(!JudgeKeyStore.save(kind: "mystery", key: "k", storage: storage))
        #expect(JudgeKeyStore.stored(storage: storage) == nil)
    }

    @Test func remoteFileContentsMatchTheServerContract() {
        // Deterministic (sorted keys) + newline-terminated: repeated pushes
        // of the same key are byte-identical, and the server's
        // judge_credentials._read_key_file parses exactly this shape.
        let contents = JudgeKeyStore.remoteFileContents(
            .init(kind: "anthropic", key: "sk-ant-y"))
        #expect(contents == "{\"key\":\"sk-ant-y\",\"kind\":\"anthropic\"}\n")
    }

    @Test func resolveKeyPrefersEnvironmentThenMatchingStoredKind() {
        let storage = MemoryStorage()
        JudgeKeyStore.save(kind: "openrouter", key: "sk-or-stored", storage: storage)
        // Env wins.
        #expect(JudgeKeyStore.resolveKey(
            kind: "openrouter",
            environment: ["OPENROUTER_API_KEY": "sk-or-env"],
            storage: storage) == "sk-or-env")
        // Stored key serves its own kind…
        #expect(JudgeKeyStore.resolveKey(
            kind: "openrouter", environment: [:], storage: storage)
            == "sk-or-stored")
        // …and never a different kind.
        #expect(JudgeKeyStore.resolveKey(
            kind: "anthropic", environment: [:], storage: storage) == nil)
    }
}
