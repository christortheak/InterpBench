import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

/// Pure path-resolution seam behind `FileReferenceRow` (UX item: every file
/// path a panel displays is viewable/revealable). Containment mirrors
/// `VectorCatalog.projectFile`: a relative path may never escape the root.
struct FileReferenceTests {

    private let root = URL(filePath: "/tmp/steerlab-fileref-root")

    @Test func relativePathResolvesUnderRoot() {
        let reference = FileReference.resolve("prompts/rubrics/default-paired-v1.md", root: root)
        #expect(reference.url?.path == "/tmp/steerlab-fileref-root/prompts/rubrics/default-paired-v1.md")
        #expect(reference.displayName == "default-paired-v1.md")
        #expect(reference.originalPath == "prompts/rubrics/default-paired-v1.md")
    }

    @Test func traversalOutsideRootResolvesToNilURL() {
        let reference = FileReference.resolve("../outside/secrets.txt", root: root)
        #expect(reference.url == nil)
        // The name still renders — only the affordances disappear.
        #expect(reference.displayName == "secrets.txt")
    }

    @Test func sneakyMidPathTraversalIsContained() {
        let reference = FileReference.resolve("prompts/../../etc/passwd", root: root)
        #expect(reference.url == nil)
    }

    @Test func dotDotThatStaysInsideRootIsFine() {
        let reference = FileReference.resolve("prompts/../experiments/x.json", root: root)
        #expect(reference.url?.path == "/tmp/steerlab-fileref-root/experiments/x.json")
    }

    @Test func absolutePathPassesThroughStandardized() {
        let reference = FileReference.resolve("/private/tmp/../tmp/file.jsonl", root: root)
        #expect(reference.url?.path == "/private/tmp/file.jsonl")
        #expect(reference.displayName == "file.jsonl")
    }

    @Test func emptyAndWhitespacePathsResolveToNothing() {
        #expect(FileReference.resolve("", root: root).url == nil)
        #expect(FileReference.resolve("   ", root: root).url == nil)
        #expect(FileReference.resolve("", root: root).displayName.isEmpty)
    }

    @Test func existsReflectsDiskAndHashMatchesPinDigest() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "fileref-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = Data("{\"text\": \"hello\"}\n".utf8)
        let file = directory.appending(component: "prompts.jsonl")
        try data.write(to: file)

        let reference = FileReference.resolve("prompts.jsonl", root: directory)
        #expect(reference.exists)
        let expected = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(FileReference.currentSHA256(of: try #require(reference.url)) == expected)

        let missing = FileReference.resolve("nope.jsonl", root: directory)
        #expect(!missing.exists)
        #expect(missing.url != nil)  // resolvable, just absent
    }
}
