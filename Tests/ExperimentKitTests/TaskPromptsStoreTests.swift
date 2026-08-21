import Foundation
import Testing

@testable import ExperimentKit

/// `TaskPromptsStore` (Data workbench phase 4): the prompt-set listing that
/// closed phase 3's disclosed residual.
///
/// Phase 3 enumerated `prompts/tasks/` and `prompts/dev/` and said on every
/// row that a manifest pins its task prompts by PATH, so the listing was not
/// exhaustive. The rule under test here is the replacement, and its limits
/// matter as much as its reach: manifest-named files are listed WHEREVER they
/// live, and nothing else is invented — no recursive walk, and a pinned path
/// with no file behind it is not conjured into a row.
@Suite(.serialized) @MainActor
struct TaskPromptsStoreTests {

    // MARK: Harness

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "task-prompts-store") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            return try body(root)
        }
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    private func taskRows(_ count: Int) -> String {
        (0 ..< count)
            .map { #"{"id": "item-\#($0)", "prompt": "Decide \#($0)."}"# }
            .joined(separator: "\n") + "\n"
    }

    private func textRows(_ texts: [String]) -> String {
        texts.map { #"{"text": "\#($0)"}"# }.joined(separator: "\n") + "\n"
    }

    /// A manifest written with only the keys the store's peek decode reads —
    /// deliberately NOT a full `ExperimentManifest`, because the peek exists
    /// so a legacy or hand-edited manifest still contributes its pin.
    private func writeMinimalManifest(
        name: String, taskPromptsFile: String?, in root: URL
    ) throws {
        var object: [String: Any] = ["name": name, "someUnknownFutureKey": true]
        if let taskPromptsFile { object["taskPromptsFile"] = taskPromptsFile }
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys])
        let url = root.appending(
            components: "experiments", name, "experiment.json")
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    // MARK: The two conventional roots

    @Test func bothConventionalRootsAreListedWithTheirFamily() throws {
        try withTempWorkspace { root in
            try write(
                taskRows(2),
                to: root.appending(components: "prompts", "tasks", "measured.jsonl"))
            try write(
                textRows(["dev one", "dev two"]),
                to: root.appending(components: "prompts", "dev", "dev-prompts.jsonl"))
            // Not a JSONL file, and not directly in the root — neither is a
            // prompt set, and neither is invented into one.
            try write(
                "notes\n",
                to: root.appending(components: "prompts", "tasks", "README.md"))
            try write(
                taskRows(1),
                to: root.appending(
                    components: "prompts", "tasks", "archive", "old.jsonl"))

            let records = TaskPromptsStore.list(root: root)
            #expect(
                records.map(\.relativePath) == [
                    "prompts/dev/dev-prompts.jsonl",
                    "prompts/tasks/measured.jsonl",
                ])
            #expect(records.map(\.family) == [.dev, .task])
            #expect(records.allSatisfy { $0.pinnedBy.isEmpty })
            #expect(records.map(\.familyLabel) == ["dev", "task"])
        }
    }

    @Test func anEmptyWorkspaceListsNothing() throws {
        try withTempWorkspace { root in
            #expect(TaskPromptsStore.list(root: root).isEmpty)
        }
    }

    // MARK: Manifest-pinned files

    /// The residual phase 3 could only disclose: a manifest pins its task
    /// prompts by path, so a set filed outside both conventional roots was
    /// invisible. It is listed now — labelled `pinned`, naming the study.
    @Test func aManifestPinnedOutlierIsListedAndLabelled() throws {
        try withTempWorkspace { root in
            try write(
                taskRows(3),
                to: root.appending(components: "prompts", "case-sets", "panel.jsonl"))
            try writeMinimalManifest(
                name: "outlier-study",
                taskPromptsFile: "prompts/case-sets/panel.jsonl", in: root)

            let records = TaskPromptsStore.list(root: root)
            #expect(records.count == 1)
            let record = try #require(records.first)
            #expect(record.relativePath == "prompts/case-sets/panel.jsonl")
            #expect(record.family == nil)
            #expect(record.familyLabel == "pinned")
            #expect(record.pinnedBy == ["outlier-study"])
            #expect(record.readingFamily == .task)
            #expect(record.note.contains("outlier-study"))
            #expect(record.note.contains("Filed outside"))
        }
    }

    /// A pinned file that IS in a conventional root is one record, not two:
    /// it keeps its family and gains the pin.
    @Test func aPinnedConventionalFileIsOneRecordCarryingBoth() throws {
        try withTempWorkspace { root in
            try write(
                textRows(["dev one"]),
                to: root.appending(components: "prompts", "dev", "dev-prompts.jsonl"))
            try writeMinimalManifest(
                name: "study-a", taskPromptsFile: "prompts/dev/dev-prompts.jsonl",
                in: root)
            try writeMinimalManifest(
                name: "study-b", taskPromptsFile: "prompts/dev/dev-prompts.jsonl",
                in: root)

            let records = TaskPromptsStore.list(root: root)
            #expect(records.count == 1)
            let record = try #require(records.first)
            #expect(record.family == .dev)
            #expect(record.familyLabel == "dev")
            // Sorted, so the caption is stable across scans.
            #expect(record.pinnedBy == ["study-a", "study-b"])
            #expect(record.pinsSentence?.contains("study-a, study-b") == true)
            // The dev loader still reads it — being pinned as task prompts
            // does not change which root's convention it was filed under.
            #expect(record.readingFamily == .dev)
        }
    }

    /// The real key, through the real pin verb — so the peek decode cannot
    /// drift from what `ExperimentStore.pinTaskPrompts` actually writes.
    @Test func theKeyIsTheOneTheRealPinWrites() throws {
        try withTempWorkspace { root in
            try write(
                taskRows(2),
                to: root.appending(components: "prompts", "tasks", "measured.jsonl"))
            var manifest = try ExperimentStore.create(
                name: "real-study", description: "", modelID: "test/model")
            _ = try ExperimentStore.pinTaskPrompts(
                "prompts/tasks/measured.jsonl", into: &manifest)
            try ExperimentStore.save(manifest)

            let record = try #require(TaskPromptsStore.list(root: root).first)
            #expect(record.relativePath == "prompts/tasks/measured.jsonl")
            #expect(record.pinnedBy == ["real-study"])
        }
    }

    /// Honesty limits: a pin whose bytes are not there is NOT listed (the
    /// inventory reports what exists — a moved pin is a `verify()` violation,
    /// which is where it belongs), and a manifest with no pin adds nothing.
    @Test func aMissingPinnedFileIsNotConjuredIntoARow() throws {
        try withTempWorkspace { root in
            try writeMinimalManifest(
                name: "moved-study",
                taskPromptsFile: "prompts/case-sets/gone.jsonl", in: root)
            try writeMinimalManifest(
                name: "unpinned-study", taskPromptsFile: nil, in: root)
            // A directory at the pinned path is not a file either.
            try FileManager.default.createDirectory(
                at: root.appending(components: "prompts", "case-sets", "gone.jsonl"),
                withIntermediateDirectories: true)

            #expect(TaskPromptsStore.list(root: root).isEmpty)
        }
    }

    /// An ABSOLUTE pinned path resolves by the manifest's own rule
    /// (`ExperimentStore.resolveProjectPath`) rather than being appended to
    /// the root — the same rule `verify()` reads it with.
    @Test func anAbsolutePinnedPathResolvesByTheManifestsOwnRule() throws {
        try withTempWorkspace { root in
            let outside = FileManager.default.temporaryDirectory
                .appending(component: "task-prompts-outside-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: outside) }
            let file = outside.appending(component: "elsewhere.jsonl")
            try write(taskRows(2), to: file)
            try writeMinimalManifest(
                name: "absolute-study", taskPromptsFile: file.path, in: root)

            let record = try #require(TaskPromptsStore.list(root: root).first)
            #expect(record.family == nil)
            #expect(record.pinnedBy == ["absolute-study"])
            // Outside the workspace, so the display path stays absolute
            // rather than pretending to be workspace-relative.
            #expect(record.relativePath.hasPrefix("/"))
        }
    }

    // MARK: The inventory reads through the store

    @Test func theInventoryListsPinnedOutliersAsPromptSetRows() throws {
        try withTempWorkspace { root in
            try write(
                taskRows(4),
                to: root.appending(components: "prompts", "case-sets", "panel.jsonl"))
            try write(
                textRows(["dev one"]),
                to: root.appending(components: "prompts", "dev", "dev-prompts.jsonl"))
            try writeMinimalManifest(
                name: "outlier-study",
                taskPromptsFile: "prompts/case-sets/panel.jsonl", in: root)

            let sets = DatasetInventory.scan(root: root).filter { $0.kind == .promptSet }
            #expect(sets.count == 2)

            let pinned = try #require(sets.first { $0.familyLabel == "pinned" })
            #expect(pinned.name == "panel")
            // Counted by the TASK parser — an outlier is a study's measured
            // task prompts by definition; that is the only way it got here.
            #expect(pinned.itemCount == 4)
            #expect(pinned.issue == nil)
            #expect(pinned.contentHash?.count == 64)
            #expect(pinned.note?.contains("outlier-study") == true)
            #expect(pinned.displayPath(root: root) == "prompts/case-sets/panel.jsonl")

            #expect(sets.contains { $0.familyLabel == "dev" })
        }
    }
}
