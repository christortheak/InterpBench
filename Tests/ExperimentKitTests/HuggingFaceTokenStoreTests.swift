import Foundation
import Testing

@testable import ExperimentKit

/// The store's contract: Keychain + materialized hub file move together on
/// save, and the file is removed on clear ONLY when it is ours — a token
/// written independently by `hf auth login` is not this store's to delete.
/// All through the in-memory seam; the real Keychain is never touched.
@Suite struct HuggingFaceTokenStoreTests {

    final class MemoryStorage: HuggingFaceTokenStorage {
        var value: String?
        func readKeychain() -> String? { value }
        func writeKeychain(_ token: String) -> Bool { value = token; return true }
        func deleteKeychain() { value = nil }
    }

    private func temporaryHubFile() -> URL {
        FileManager.default.temporaryDirectory
            .appending(components: "hf-store-tests", UUID().uuidString, "token")
    }

    @Test func saveWritesKeychainAndMaterializesModeSixHundred() throws {
        let storage = MemoryStorage()
        let file = temporaryHubFile()
        let error = HuggingFaceTokenStore.save(
            "  hf_abc123  ", storage: storage, hubFile: file)
        #expect(error == nil)
        #expect(storage.value == "hf_abc123")  // trimmed
        #expect(try String(contentsOf: file, encoding: .utf8) == "hf_abc123")
        let mode = try FileManager.default.attributesOfItem(
            atPath: file.path)[.posixPermissions] as? Int
        #expect(mode == 0o600)
    }

    @Test func clearRemovesTheFileOnlyWhenItMatches() throws {
        let storage = MemoryStorage()
        let ours = temporaryHubFile()
        HuggingFaceTokenStore.save("hf_ours", storage: storage, hubFile: ours)
        HuggingFaceTokenStore.save("", storage: storage, hubFile: ours)
        #expect(storage.value == nil)
        #expect(!FileManager.default.fileExists(atPath: ours.path))

        // A foreign file (hf auth login with a different token) survives.
        let foreign = temporaryHubFile()
        HuggingFaceTokenStore.save("hf_ours", storage: storage, hubFile: foreign)
        try Data("hf_theirs".utf8).write(to: foreign)
        HuggingFaceTokenStore.save("", storage: storage, hubFile: foreign)
        #expect(try String(contentsOf: foreign, encoding: .utf8) == "hf_theirs")
    }

    @Test func hubTokenURLFollowsHFHomeThenDefaultsToCache() {
        let overridden = HuggingFaceTokenStore.hubTokenURL(
            environment: ["HF_HOME": "/work/lab/hf-cache"])
        #expect(overridden.path == "/work/lab/hf-cache/token")
        let fallback = HuggingFaceTokenStore.hubTokenURL(
            environment: [:], home: URL(fileURLWithPath: "/Users/x"))
        #expect(fallback.path == "/Users/x/.cache/huggingface/token")
    }

    @Test func environmentWinsAreReportedNotFought() {
        #expect(HuggingFaceTokenStore.environmentOverrides(
            environment: ["HF_TOKEN": "hf_x"]))
        #expect(HuggingFaceTokenStore.environmentOverrides(
            environment: ["HUGGING_FACE_HUB_TOKEN": "hf_x"]))
        #expect(!HuggingFaceTokenStore.environmentOverrides(
            environment: ["HF_TOKEN": ""]))
        #expect(!HuggingFaceTokenStore.environmentOverrides(environment: [:]))
    }

    @Test func storedTokenIgnoresEmptyKeychainValues() {
        let storage = MemoryStorage()
        storage.value = ""
        #expect(HuggingFaceTokenStore.storedToken(storage: storage) == nil)
        #expect(!HuggingFaceTokenStore.hasStoredToken(storage: storage))
        storage.value = "hf_y"
        #expect(HuggingFaceTokenStore.storedToken(storage: storage) == "hf_y")
    }

    @Test func fileWriteFailureIsReportedNotSilent() {
        let storage = MemoryStorage()
        // An unwritable destination: a path under a FILE, not a directory.
        let blocker = FileManager.default.temporaryDirectory
            .appending(components: "hf-store-tests", UUID().uuidString)
        try? FileManager.default.createDirectory(
            at: blocker.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? Data("x".utf8).write(to: blocker)
        let error = HuggingFaceTokenStore.save(
            "hf_z", storage: storage,
            hubFile: blocker.appending(components: "sub", "token"))
        #expect(error != nil)
        #expect(storage.value == "hf_z")  // the Keychain save still stands
    }
}
