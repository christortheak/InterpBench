import Foundation
import Testing

@testable import ExperimentKit

/// The native Results Explorer bridge (2026-08-03): pure logic behind the
/// WKURLSchemeHandler serving the embedded explorer SPA — containment for
/// every page-supplied path, tree listings shaped for the page's fetch
/// adapter, and file reads. Escape attempts refuse; the served root is the
/// workspace's runs/ directory, so the discipline mirrors the promotion
/// gates' plain-run-name rule.
struct ResultsExplorerBridgeTests {

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "rex-bridge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func containmentRefusesEveryEscapeShape() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(ResultsExplorerBridge.containedURL(path: "", under: root) == root)
        #expect(
            ResultsExplorerBridge.containedURL(path: "a/b.json", under: root)
                != nil)
        for escape in [
            "../outside", "a/../../outside", "/etc/passwd", "a/./b",
            "a//b", "a\\b", "..",
        ] {
            #expect(
                ResultsExplorerBridge.containedURL(path: escape, under: root)
                    == nil,
                "escape shape '\(escape)' must refuse")
        }
    }

    @Test func treeListsEntriesShapedForTheFetchAdapter() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(component: "20260803T000000000-exp-x-sweep")
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        try Data("{}".utf8).write(
            to: run.appending(component: "report.json"))
        try Data(".hidden".utf8).write(
            to: run.appending(component: ".DS_Store"))

        let top = try ResultsExplorerBridge.tree(path: "", under: root)
        #expect(top.map(\.name) == ["20260803T000000000-exp-x-sweep"])
        #expect(top.first?.kind == "directory")

        let inside = try ResultsExplorerBridge.tree(
            path: "20260803T000000000-exp-x-sweep", under: root)
        #expect(inside.map(\.name) == ["report.json"])  // hidden skipped
        #expect(inside.first?.kind == "file")
        #expect(inside.first?.size == 2)
        #expect((inside.first?.modified ?? 0) > 0)
    }

    @Test func treeAndFileRefuseEscapesAndMissingPaths() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: ExperimentError.self) {
            _ = try ResultsExplorerBridge.tree(path: "../..", under: root)
        }
        #expect(throws: ExperimentError.self) {
            _ = try ResultsExplorerBridge.tree(path: "absent", under: root)
        }
        #expect(throws: ExperimentError.self) {
            _ = try ResultsExplorerBridge.fileData(
                path: "../secret", under: root)
        }
    }

    @Test func symlinksNeverEscapeOrList() throws {
        // Review 2026-08-03, P1: textual containment is not enough — a
        // symlink beneath runs/ can point anywhere. Symlinked files and
        // directories refuse to resolve, and tree listings omit them.
        let root = try temporaryRoot()
        let outside = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        try Data("secret".utf8).write(
            to: outside.appending(component: "secret.txt"))
        try FileManager.default.createSymbolicLink(
            at: root.appending(component: "leak.txt"),
            withDestinationURL: outside.appending(component: "secret.txt"))
        try FileManager.default.createSymbolicLink(
            at: root.appending(component: "leakdir"),
            withDestinationURL: outside)
        #expect(
            ResultsExplorerBridge.containedURL(path: "leak.txt", under: root)
                == nil)
        #expect(
            ResultsExplorerBridge.containedURL(
                path: "leakdir/secret.txt", under: root) == nil)
        #expect(throws: ExperimentError.self) {
            _ = try ResultsExplorerBridge.fileData(
                path: "leak.txt", under: root)
        }
        #expect(throws: ExperimentError.self) {
            _ = try ResultsExplorerBridge.tree(path: "leakdir", under: root)
        }
        // Listings omit the symlinks entirely — never addressable.
        try Data("{}".utf8).write(
            to: root.appending(component: "honest.json"))
        let listed = try ResultsExplorerBridge.tree(path: "", under: root)
        #expect(listed.map(\.name) == ["honest.json"])
    }

    @Test func boundedReadsReturnExactRanges() throws {
        // Review 2026-08-03, P2: the page's bounded preview must be a
        // bounded HOST read — offset/length slice without materializing
        // the whole file.
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("0123456789".utf8).write(
            to: root.appending(component: "g.jsonl"))
        let head = try ResultsExplorerBridge.fileData(
            path: "g.jsonl", under: root, offset: 0, length: 4)
        #expect(String(decoding: head, as: UTF8.self) == "0123")
        let middle = try ResultsExplorerBridge.fileData(
            path: "g.jsonl", under: root, offset: 3, length: 4)
        #expect(String(decoding: middle, as: UTF8.self) == "3456")
        let tail = try ResultsExplorerBridge.fileData(
            path: "g.jsonl", under: root, offset: 8, length: 100)
        #expect(String(decoding: tail, as: UTF8.self) == "89")
        let unbounded = try ResultsExplorerBridge.fileData(
            path: "g.jsonl", under: root)
        #expect(unbounded.count == 10)
    }

    @Test func fileDataRoundTripsBytes() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("hello".utf8).write(to: root.appending(component: "a.jsonl"))
        let data = try ResultsExplorerBridge.fileData(
            path: "a.jsonl", under: root)
        #expect(String(decoding: data, as: UTF8.self) == "hello")
        #expect(
            ResultsExplorerBridge.contentType(for: "a.jsonl")
                .hasPrefix("application/x-ndjson"))
        #expect(
            ResultsExplorerBridge.contentType(for: "index.html")
                .hasPrefix("text/html"))
    }
}
