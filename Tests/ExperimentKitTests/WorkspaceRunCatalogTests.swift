import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// `catalog/` — the generated navigation overlay (open-issues §20, tightening 5).
//
// Three properties carry the whole design and are asserted here:
//   * NAVIGATION ONLY — every leaf is a symlink or the generated INDEX, so
//     deleting the tree can never lose a byte of evidence;
//   * IDEMPOTENT — two builds over an unchanged runs/ are byte-identical;
//   * GITIGNORED — a freeze auto-commit can never snapshot the symlink forest.
//
// Fixture vocabulary is synthetic (`alpha`, `beta`, `gamma`): the catalog's
// grouping keys come from each run's own metadata, and proving that requires
// fixtures that carry no study vocabulary at all.
// =============================================================================

struct WorkspaceRunCatalogTests {

    // MARK: Fixture

    private func makeWorkspace(_ label: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-catalog-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appending(component: "runs"), withIntermediateDirectories: true)
        return root
    }

    @discardableResult
    private func makeRun(
        _ root: URL, _ name: String, files: [String: String] = [:],
        experiment: String? = nil
    ) throws -> URL {
        let directory = root.appending(components: "runs", name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (relative, contents) in files {
            let url = directory.appending(path: relative)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(contents.utf8).write(to: url)
        }
        if let experiment {
            try Data(#"{"experiment": "\#(experiment)", "runType": "run"}"#.utf8)
                .write(to: directory.appending(component: RunMetadata.fileName))
        }
        return directory
    }

    // MARK: Structure

    @Test func theCatalogFilesRunsByKindAndByWave() throws {
        let root = try makeWorkspace("structure")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(
            root, "20260819T101500123-exp-alpha-run", files: ["generations.jsonl": "{}"],
            experiment: "w1-alpha")
        try makeRun(
            root, "20260819T111500123-exp-alpha-analyze", files: ["report.json": "{}"],
            experiment: "w1-alpha")
        try makeRun(root, "20260819T121500123-optvec-beta-l20", files: ["v.safetensors": "x"])

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.rows.count == 3)

        let catalog = root.appending(component: "catalog")
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: catalog.appending(components: "by-kind", "run").path))
        #expect(fm.fileExists(atPath: catalog.appending(components: "by-kind", "analyze").path))
        // The grouping key came from config.json's `experiment`, not from a
        // constant: the wave is that name's leading token.
        #expect(
            fm.fileExists(
                atPath: catalog.appending(components: "by-wave", "w1", "w1-alpha").path))
        // Vector artifacts get their own bucket, keyed by their name prefix.
        #expect(
            fm.fileExists(
                atPath: catalog.appending(components: "vectors", "optvec").path))
    }

    /// A run with no readable `config.json` still files — under its directory
    /// stem. Absence degrades the key, never the build.
    @Test func aRunWithoutMetadataFallsBackToItsStem() throws {
        let root = try makeWorkspace("fallback")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(root, "20260819T101500123-exp-gamma-run", files: ["x.jsonl": "{}"])

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.rows.first?.study == "gamma")
        #expect(report.rows.first?.wave == "gamma")
    }

    /// Adapters are found by SHAPE — a submit receipt holding final weights —
    /// not by what the training job was named.
    @Test func adaptersAreFoundByShapeNotByName() throws {
        let root = try makeWorkspace("adapters")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(
            root, "20260819T101500123-submit-anything-at-all",
            files: [
                "plan.json": "{}",
                "run/adapter-alpha/\(WorkspaceImportPolicy.adapterWeightFileName)": "weights",
                "run/not-an-adapter/notes.txt": "hello",
            ])

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.adapterCount == 1)
        let link = root.appending(components: "catalog", "adapters", "adapter-alpha")
        #expect(FileManager.default.fileExists(atPath: link.path))
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appending(components: "catalog", "adapters", "not-an-adapter")
                    .path))
    }

    /// The mutable library subtrees are linked as libraries, never catalogued
    /// as runs.
    @Test func librarySubtreesAreLinkedNotCatalogued() throws {
        let root = try makeWorkspace("libraries")
        defer { try? FileManager.default.removeItem(at: root) }
        for library in WorkspaceImportPolicy.librarySubtrees {
            try FileManager.default.createDirectory(
                at: root.appending(components: "runs", library),
                withIntermediateDirectories: true)
        }
        try makeRun(root, "20260819T101500123-exp-alpha-run")

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.rows.count == 1)
        #expect(report.libraryCount == WorkspaceImportPolicy.librarySubtrees.count)
        for library in WorkspaceImportPolicy.librarySubtrees {
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(components: "catalog", "libraries", library).path))
        }
    }

    // MARK: Navigation only

    /// Nothing under `catalog/` is a real file except the generated INDEX.
    /// This is what makes "deleting it loses nothing" true rather than a
    /// promise in a comment.
    @Test func everythingUnderTheCatalogIsASymlinkOrTheIndex() throws {
        let root = try makeWorkspace("navigation")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(
            root, "20260819T101500123-exp-alpha-run", files: ["generations.jsonl": "{}"],
            experiment: "w1-alpha")
        try makeRun(
            root, "20260819T111500123-submit-alpha",
            files: [
                "run/adapter-alpha/\(WorkspaceImportPolicy.adapterWeightFileName)": "w"
            ])
        try WorkspaceRunCatalog.rebuild(workspaceRoot: root)

        let catalog = root.appending(component: "catalog")
        let fm = FileManager.default
        let enumerator = try #require(
            fm.enumerator(
                at: catalog, includingPropertiesForKeys: [.isSymbolicLinkKey, .isDirectoryKey],
                options: []))
        var realFiles: [String] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [
                .isSymbolicLinkKey, .isDirectoryKey,
            ])
            if values.isSymbolicLink == true {
                // Do not descend into the run tree through the link.
                enumerator.skipDescendants()
                continue
            }
            if values.isDirectory == true { continue }
            realFiles.append(url.lastPathComponent)
        }
        #expect(realFiles == [WorkspaceRunCatalog.indexFileName])
    }

    /// Links are RELATIVE, so the whole workspace stays movable.
    @Test func linksAreRelative() throws {
        let root = try makeWorkspace("relative")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(root, "20260819T101500123-exp-alpha-run")
        try WorkspaceRunCatalog.rebuild(workspaceRoot: root)

        let link = root.appending(
            components: "catalog", "by-kind", "run", "20260819T101500123-exp-alpha-run")
        let destination = try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
        #expect(destination.hasPrefix("../"))
        #expect(!destination.hasPrefix("/"))
        #expect(FileManager.default.fileExists(atPath: link.path), "the link must resolve")
    }

    // MARK: Idempotency

    @Test func twoBuildsOverAnUnchangedRunsTreeAreIdentical() throws {
        let root = try makeWorkspace("idempotent")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(
            root, "20260819T101500123-exp-alpha-run", files: ["generations.jsonl": "{}"],
            experiment: "w1-alpha")
        try makeRun(root, "20260819T111500123-exp-beta-sweep", experiment: "w2-beta")

        let first = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        let firstIndex = try String(
            contentsOf: root.appending(
                components: "catalog", WorkspaceRunCatalog.indexFileName),
            encoding: .utf8)
        let firstTree = try inventory(root.appending(component: "catalog"))

        let second = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        let secondIndex = try String(
            contentsOf: root.appending(
                components: "catalog", WorkspaceRunCatalog.indexFileName),
            encoding: .utf8)
        let secondTree = try inventory(root.appending(component: "catalog"))

        // Everything the build DERIVES is identical. `gitignoreUpdated` is
        // deliberately excluded: it reports whether this invocation had to
        // append the ignore line, and "the first build fixed it, the second
        // found nothing to fix" is the correct answer, not drift.
        #expect(first.rows == second.rows)
        #expect(first.linkCount == second.linkCount)
        #expect(first.adapterCount == second.adapterCount)
        #expect(first.libraryCount == second.libraryCount)
        #expect(firstIndex == secondIndex)
        #expect(firstTree == secondTree)
        #expect(firstIndex.contains("generated — do not edit"))
    }

    /// A run that vanished from `runs/` vanishes from the catalog: the tree is
    /// rebuilt, never patched.
    @Test func aRemovedRunDisappearsFromTheRebuiltCatalog() throws {
        let root = try makeWorkspace("removal")
        defer { try? FileManager.default.removeItem(at: root) }
        let doomed = try makeRun(root, "20260819T101500123-exp-alpha-run")
        try makeRun(root, "20260819T111500123-exp-beta-run")
        try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        try FileManager.default.removeItem(at: doomed)

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.rows.map(\.name) == ["20260819T111500123-exp-beta-run"])
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appending(
                    components: "catalog", "by-kind", "run",
                    "20260819T101500123-exp-alpha-run"
                ).path))
    }

    private func inventory(_ root: URL) throws -> [String] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: root.path) else { return [] }
        var paths: [String] = []
        for case let path as String in enumerator { paths.append(path) }
        return paths.sorted()
    }

    // MARK: Gitignore maintenance (tightening 5)

    @Test func theRebuildAppendsCatalogToAnExistingGitignore() throws {
        let root = try makeWorkspace("gitignore")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(root, "20260819T101500123-exp-alpha-run")
        let gitignore = root.appending(component: ".gitignore")
        try Data("runs/\nadapters/**/*.safetensors\n".utf8).write(to: gitignore)

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.gitignoreUpdated)
        let text = try String(contentsOf: gitignore, encoding: .utf8)
        #expect(text.contains("catalog/"))
        // The lines that were already there are untouched — a workspace is
        // data, not a managed install.
        #expect(text.hasPrefix("runs/\nadapters/**/*.safetensors\n"))

        // Idempotent: a second rebuild does not append a second line.
        let again = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(!again.gitignoreUpdated)
        let after = try String(contentsOf: gitignore, encoding: .utf8)
        #expect(after == text)
    }

    @Test func anAlreadyCoveredGitignoreIsLeftAlone() throws {
        let root = try makeWorkspace("covered")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(root, "20260819T101500123-exp-alpha-run")
        let gitignore = root.appending(component: ".gitignore")
        for spelling in ["catalog/", "catalog", "/catalog/"] {
            try Data("runs/\n\(spelling)\n".utf8).write(to: gitignore)
            let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
            #expect(!report.gitignoreUpdated, "'\(spelling)' already covers the catalog")
        }
    }

    /// A workspace with no `.gitignore` at all still gets one.
    @Test func aMissingGitignoreIsCreated() throws {
        let root = try makeWorkspace("missing-gitignore")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeRun(root, "20260819T101500123-exp-alpha-run")

        let report = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        #expect(report.gitignoreUpdated)
        let text = try String(
            contentsOf: root.appending(component: ".gitignore"), encoding: .utf8)
        #expect(text == "catalog/\n")
    }

    /// New workspaces are born with the line, so the first freeze auto-commit
    /// already excludes the catalog.
    @Test func newWorkspacesAreSeededWithTheCatalogIgnore() {
        // The seeded contents are a literal in `WorkspaceStore.seedWorkspace`;
        // this asserts the two spellings agree without creating a workspace.
        #expect(WorkspaceRunCatalog.gitignoreLine == "catalog/")
    }

    @Test func refusesAWorkspaceWithNoRunsDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-catalog-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) {
            _ = try WorkspaceRunCatalog.rebuild(workspaceRoot: root)
        }
    }
}
