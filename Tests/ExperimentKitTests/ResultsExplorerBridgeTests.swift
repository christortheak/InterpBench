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

    // MARK: - The one write path: saving a file the reader placed

    @Test func saveWritesTextToThePlaceTheReaderChose() throws {
        let root = try temporaryRoot()
        let chosen = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: chosen)
        }
        let destination = chosen.appending(component: "effect-sizes.csv")
        let text = "condition,endpoint\nsteered,choiceRate\n"
        let written = try ResultsExplorerBridge.save(
            .text(text), to: destination, runsRoot: root)
        #expect(written == text.utf8.count)
        #expect(try String(contentsOf: destination, encoding: .utf8) == text)
        // Saving again under the same name replaces the file, as the save
        // panel's own "Replace" confirmation promised.
        try ResultsExplorerBridge.save(
            .text("a\n"), to: destination, runsRoot: root)
        #expect(try String(contentsOf: destination, encoding: .utf8) == "a\n")
        // Nothing was written under the served root.
        #expect(try ResultsExplorerBridge.tree(path: "", under: root).isEmpty)
    }

    @Test func saveCopiesARunFileByteForByteAndLeavesTheRunAlone() throws {
        let root = try temporaryRoot()
        let chosen = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: chosen)
        }
        let run = root.appending(component: "20260803T000000000-exp-x-run")
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        let bytes = Data((0..<4096).map { UInt8($0 % 251) })
        let source = run.appending(component: "generations.jsonl")
        try bytes.write(to: source)
        let before = try FileManager.default.attributesOfItem(
            atPath: source.path)[.modificationDate] as? Date

        let destination = chosen.appending(component: "generations.jsonl")
        let written = try ResultsExplorerBridge.save(
            .runFile(path: "20260803T000000000-exp-x-run/generations.jsonl"),
            to: destination, runsRoot: root)
        #expect(written == bytes.count)
        #expect(try Data(contentsOf: destination) == bytes)
        // The run folder holds exactly what it held: same file, same
        // bytes, same modification date, and no partial file left behind.
        #expect(try Data(contentsOf: source) == bytes)
        let after = try FileManager.default.attributesOfItem(
            atPath: source.path)[.modificationDate] as? Date
        #expect(before == after)
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: run.path)
                == ["generations.jsonl"])
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: chosen.path)
                == ["generations.jsonl"])

        // Replacing an existing file of that name works too.
        try Data("old".utf8).write(to: destination)
        try ResultsExplorerBridge.save(
            .runFile(path: "20260803T000000000-exp-x-run/generations.jsonl"),
            to: destination, runsRoot: root)
        #expect(try Data(contentsOf: destination) == bytes)
    }

    @Test func saveRefusesAnyPlaceInsideTheRunsFolder() throws {
        let root = try temporaryRoot()
        let elsewhere = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: elsewhere)
        }
        let run = root.appending(component: "20260803T000000000-exp-x-run")
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
        // A folder OUTSIDE runs/ that is a link INTO a run folder: the
        // textual path looks harmless, and the resolved one is refused.
        let door = elsewhere.appending(component: "door")
        try FileManager.default.createSymbolicLink(
            at: door, withDestinationURL: run)

        for destination in [
            root.appending(component: "export.csv"),
            run.appending(component: "export.csv"),
            run.appending(component: "report.json"),
            door.appending(component: "export.csv"),
        ] {
            let error = #expect(throws: ExperimentError.self) {
                try ResultsExplorerBridge.save(
                    .text("x"), to: destination, runsRoot: root)
            }
            #expect(error?.reason.contains("runs folder") == true)
        }
        // The run folder is byte-for-byte what it was.
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: run.path)
                == ["report.json"])
        #expect(
            try Data(contentsOf: run.appending(component: "report.json"))
                == Data("{}".utf8))
    }

    @Test func saveNeverWritesThroughALinkOrOverAFolder() throws {
        let root = try temporaryRoot()
        let chosen = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: chosen)
        }
        let target = chosen.appending(component: "precious.txt")
        try Data("keep".utf8).write(to: target)
        let link = chosen.appending(component: "export.csv")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: target)
        #expect(throws: ExperimentError.self) {
            try ResultsExplorerBridge.save(
                .text("overwrite"), to: link, runsRoot: root)
        }
        #expect(try String(contentsOf: target, encoding: .utf8) == "keep")

        let folder = chosen.appending(component: "a-folder")
        try FileManager.default.createDirectory(
            at: folder, withIntermediateDirectories: true)
        #expect(throws: ExperimentError.self) {
            try ResultsExplorerBridge.save(
                .text("x"), to: folder, runsRoot: root)
        }
    }

    @Test func saveKeepsTheReadRefusalsForRunFiles() throws {
        let root = try temporaryRoot()
        let outside = try temporaryRoot()
        let chosen = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
            try? FileManager.default.removeItem(at: chosen)
        }
        try Data("secret".utf8).write(
            to: outside.appending(component: "secret.txt"))
        try FileManager.default.createSymbolicLink(
            at: root.appending(component: "leak.txt"),
            withDestinationURL: outside.appending(component: "secret.txt"))
        try FileManager.default.createSymbolicLink(
            at: root.appending(component: "leakdir"),
            withDestinationURL: outside)
        try FileManager.default.createDirectory(
            at: root.appending(component: "a-run"),
            withIntermediateDirectories: true)
        let destination = chosen.appending(component: "out.txt")
        for path in [
            "../\(outside.lastPathComponent)/secret.txt", "/etc/passwd",
            "leak.txt", "leakdir/secret.txt", "a//b", "a\\b", "..",
            "absent.jsonl", "a-run",
        ] {
            #expect(
                throws: ExperimentError.self,
                "run-file path '\(path)' must refuse"
            ) {
                try ResultsExplorerBridge.save(
                    .runFile(path: path), to: destination, runsRoot: root)
            }
        }
        // No refusal left anything behind in the chosen folder.
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: chosen.path)
                .isEmpty)
    }

    @Test func saveRequestUnderstandsOnlyItsTwoShapes() {
        let text = ResultsExplorerBridge.saveRequest(fromMessage: [
            "kind": "text", "filename": "a.csv", "text": "x,y\n",
        ])
        #expect(text?.request == .text("x,y\n"))
        #expect(text?.suggestedName == "a.csv")
        let file = ResultsExplorerBridge.saveRequest(fromMessage: [
            "kind": "runFile", "filename": "generations.jsonl",
            "path": "run/generations.jsonl",
        ])
        #expect(file?.request == .runFile(path: "run/generations.jsonl"))

        // A destination is never accepted from the page, and a message
        // that is not one of the two shapes is not a request at all.
        let withDestination = ResultsExplorerBridge.saveRequest(fromMessage: [
            "kind": "text", "filename": "/tmp/elsewhere/a.csv", "text": "x",
            "destination": "/tmp/elsewhere/a.csv",
        ])
        #expect(withDestination?.suggestedName == "a.csv")
        for body: Any? in [
            nil, "text", ["kind": "text", "filename": "a.csv"],
            ["kind": "runFile", "filename": "a", "path": ""],
            ["kind": "delete", "filename": "a", "path": "run"],
            ["kind": "text", "text": "x"], ["kind": 3, "filename": "a"],
        ] {
            #expect(
                ResultsExplorerBridge.saveRequest(fromMessage: body) == nil)
        }
    }

    @Test func suggestedFilenameIsAPlainNameNeverAPath() {
        #expect(
            ResultsExplorerBridge.suggestedFilename("run-effect-sizes.csv")
                == "run-effect-sizes.csv")
        #expect(
            ResultsExplorerBridge.suggestedFilename("../../etc/passwd")
                == "passwd")
        #expect(
            ResultsExplorerBridge.suggestedFilename("a\\b\\c.csv") == "c.csv")
        #expect(
            ResultsExplorerBridge.suggestedFilename(".hidden.csv")
                == "hidden.csv")
        #expect(
            ResultsExplorerBridge.suggestedFilename("line\nbreak.csv")
                == "linebreak.csv")
        #expect(ResultsExplorerBridge.suggestedFilename("") == "export")
        #expect(ResultsExplorerBridge.suggestedFilename("///") == "export")
        #expect(ResultsExplorerBridge.suggestedFilename("..") == "export")
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
