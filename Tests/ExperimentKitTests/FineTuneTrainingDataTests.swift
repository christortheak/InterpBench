import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The adapter path/folder workflow: stored paths resolve against the
/// WORKSPACE (never the process CWD — the real-cluster bug where a
/// correctly saved workspace-relative training folder read as "choose a
/// non-empty training data file"), server training accepts the folder
/// model with hash-agreeing enumeration order, workspaces carry a
/// top-level `adapters/` home, per-adapter homes are minted idempotently,
/// and drag-and-drop copies obey the house never-overwrite rule.
///
/// Serialized, and every workspace-override window holds the shared
/// `ExperimentRootOverrideLock` — the workspace root is process-global, so
/// parallel suites reading `VectorCatalog.projectRoot` must never observe
/// our temporary override (the same precedent as FactorialImportTests).
@Suite(.serialized) struct FineTuneTrainingDataTests {

    private func makeTempWorkspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "ft-ws-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        // Resolve /var → /private/var so paths minted from the root and
        // paths returned by directory enumerators agree byte-for-byte.
        return root.resolvingSymlinksInPath().standardizedFileURL
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - Path round-trip (inside relative, outside absolute)

    @Test func storedPathsAreWorkspaceRelativeInsideAndAbsoluteOutside() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }

        // Inside the workspace: stored relative, resolved back to the
        // identical absolute location.
        let inside = workspace.appending(components: "adapters", "x", "training")
        let insideStored = FineTuneStore.relativePath(for: inside)
        #expect(insideStored == "adapters/x/training")
        #expect(FineTuneStore.absoluteURL(insideStored).path == inside.path)

        // Outside the workspace: stored absolute, resolved unchanged.
        let outside = URL(filePath: "/somewhere/else/corpus")
        let outsideStored = FineTuneStore.relativePath(for: outside)
        #expect(outsideStored == "/somewhere/else/corpus")
        #expect(FineTuneStore.absoluteURL(outsideStored).path == outside.path)
    }

    // MARK: - The CWD bug: stored relative paths resolve via the workspace

    @Test func inlineTextResolvesStoredRelativePathAgainstWorkspaceNotCWD() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        try write(
            "The corpus text.",
            to: workspace.appending(components: "adapters", "j", "training", "a.txt"))

        // The process CWD is NOT the workspace (tests run from the repo /
        // build dir), so this only succeeds if resolution goes through the
        // workspace root — the exact bug beginServerTraining had.
        #expect(
            FileManager.default.currentDirectoryPath != workspace.path,
            "test premise: CWD must differ from the workspace")
        let payload = try FineTuneTrainingData.inlineText(
            storedPath: "adapters/j/training")
        #expect(payload.text.contains("The corpus text."))
        #expect(payload.fileCount == 1)
        #expect(payload.resolvedPath.hasPrefix(workspace.path))
    }

    // MARK: - Folder concatenation: order, separators, hash agreement

    @Test func folderPayloadOrdersFilesLikeHashFileOrDirectory() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let folder = workspace.appending(components: "adapters", "j", "training")
        // Deliberate discriminator: shallow name-sort would put "a.txt"
        // (inside z/) first; hashFileOrDirectory's RELATIVE-path sort puts
        // "b.txt" before "z/a.txt". The payload must match the hash's order.
        try write("Bravo body.", to: folder.appending(component: "b.txt"))
        try write("Alpha body.", to: folder.appending(components: "z", "a.txt"))
        try write(
            #"{"text":"hello row"}"#,
            to: folder.appending(component: "c.jsonl"))
        try write("hidden", to: folder.appending(component: ".hidden.txt"))
        try write("readme text", to: folder.appending(component: "README.md"))

        let payload = try FineTuneTrainingData.inlineText(
            storedPath: "adapters/j/training")
        #expect(payload.sources == ["b.txt", "c.jsonl", "z/a.txt"])
        #expect(payload.fileCount == 3)

        // Separators are per-document headers naming the relative file.
        let headerB = FineTuneTrainingData.documentHeader("b.txt")
        let headerC = FineTuneTrainingData.documentHeader("c.jsonl")
        let headerZ = FineTuneTrainingData.documentHeader("z/a.txt")
        let bRange = try #require(payload.text.range(of: headerB))
        let cRange = try #require(payload.text.range(of: headerC))
        let zRange = try #require(payload.text.range(of: headerZ))
        #expect(bRange.lowerBound < cRange.lowerBound)
        #expect(cRange.lowerBound < zRange.lowerBound)
        #expect(payload.text.contains("Bravo body."))
        #expect(payload.text.contains("Alpha body."))
        // JSONL rows are parsed to their text fields, not sent as raw JSON.
        #expect(payload.text.contains("hello row"))
        #expect(!payload.text.contains(#"{"text""#))
        // Hidden files and READMEs never become training text.
        #expect(!payload.text.contains("hidden"))
        #expect(!payload.text.contains("readme text"))

        // Hash agreement: the same folder hashes deterministically under
        // hashFileOrDirectory (same enumeration + relative-path ordering),
        // so the pinned hash and the sent bytes describe the same folder.
        let hash = try #require(FineTuneStore.hashFileOrDirectory(folder))
        #expect(FineTuneStore.hashFileOrDirectory(folder) == hash)
    }

    // MARK: - Refusal wording names the resolved path

    @Test func emptyFolderAndMissingPathRefusalsNameTheResolvedPath() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let empty = workspace.appending(components: "adapters", "j", "training")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)

        do {
            _ = try FineTuneTrainingData.inlineText(storedPath: "adapters/j/training")
            Issue.record("empty folder must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.message.contains(empty.path))
            #expect(problem.message.contains("is empty"))
        }

        do {
            _ = try FineTuneTrainingData.inlineText(storedPath: "adapters/nope/training")
            Issue.record("missing path must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.message.contains(
                workspace.appending(components: "adapters", "nope", "training").path))
            #expect(problem.message.contains("does not exist"))
        }

        do {
            _ = try FineTuneTrainingData.inlineText(storedPath: "   ")
            Issue.record("blank stored path must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.message.contains("no training data is set"))
        }
    }

    @Test func folderWithOnlyBlankTemplateRowsRefusesHonestly() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let folder = workspace.appending(components: "adapters", "j", "training")
        try write(#"{"text":""}"#, to: folder.appending(component: "train.jsonl"))

        do {
            _ = try FineTuneTrainingData.inlineText(storedPath: "adapters/j/training")
            Issue.record("blank-template folder must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.message.contains(folder.path))
            #expect(problem.message.contains("none contained readable training text"))
        }
    }

    // MARK: - Legacy "<folder>/train.jsonl" stored convention

    @Test func legacyTrainJSONLStoredPathGathersTheContainingFolder() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let folder = workspace.appending(components: "data")
        // The old picker stored "<folder>/train.jsonl" even when only loose
        // .txt files exist. The folder is the real selection.
        try write("Case one text.", to: folder.appending(component: "case1.txt"))
        try write(
            #"{"text":"jsonl doc"}"#, to: folder.appending(component: "train.jsonl"))

        let payload = try FineTuneTrainingData.inlineText(
            storedPath: "data/train.jsonl")
        #expect(payload.sources == ["case1.txt", "train.jsonl"])
        #expect(payload.text.contains("Case one text."))
        #expect(payload.text.contains("jsonl doc"))
        #expect(payload.resolvedPath == folder.path)
    }

    @Test func singleFileSelectionKeepsWorking() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let file = workspace.appending(component: "corpus.txt")
        try write("Just this one document.", to: file)

        let payload = try FineTuneTrainingData.inlineText(storedPath: "corpus.txt")
        #expect(payload.text == "Just this one document.")
        #expect(payload.fileCount == 1)
        #expect(payload.sources == ["corpus.txt"])

        // An empty single file refuses with the resolved path.
        let blank = workspace.appending(component: "blank.txt")
        try write("   \n", to: blank)
        do {
            _ = try FineTuneTrainingData.inlineText(storedPath: "blank.txt")
            Issue.record("empty file must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.message.contains(blank.path))
            #expect(problem.message.contains("is empty"))
        }
    }

    // MARK: - Structured upload: both folders, bytes intact

    /// Independent digest, computed the way the SERVER will: SHA-256 over
    /// the file's raw bytes. Deliberately not `FineTuneStore.hashFile` —
    /// the point is that the payload's hash is checkable, not that two calls
    /// of the same helper agree.
    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @Test func structuredPayloadKeepsBytesOrderRolesAndRelativePaths() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let training = workspace.appending(components: "adapters", "j", "training")
        let validation = workspace.appending(components: "adapters", "j", "validation")
        // Same discriminator as the inline test: a relative-path sort puts
        // "b.jsonl" before "z/a.jsonl", a shallow name sort would not.
        let bRows = #"{"text":"bravo row one"}"# + "\n" + #"{"text":"bravo row two"}"# + "\n"
        let zRows = #"{"text":"alpha row"}"# + "\n"
        let vRows = #"{"text":"held out row"}"# + "\n"
        try write(bRows, to: training.appending(component: "b.jsonl"))
        try write(zRows, to: training.appending(components: "z", "a.jsonl"))
        try write("dataset notes", to: training.appending(component: "README.md"))
        try write(vRows, to: validation.appending(component: "v.jsonl"))

        let payload = try FineTuneTrainingData.structuredPayload(
            trainingPath: "adapters/j/training",
            validationPath: "adapters/j/validation")

        // Train first, then validation; within a split, the enumeration
        // order `orderedFiles` (and therefore hashFileOrDirectory) uses.
        #expect(
            payload.files.map(\.path) == [
                "adapters/j/training/b.jsonl",
                "adapters/j/training/z/a.jsonl",
                "adapters/j/validation/v.jsonl",
            ])
        #expect(payload.files.map(\.role) == [.train, .train, .validation])
        #expect(payload.trainFiles.count == 2)
        #expect(payload.validationFiles.count == 1)
        #expect(payload.trainingResolvedPath == training.path)
        #expect(payload.validationResolvedPath == validation.path)

        // Ordering agrees with the enumeration the folder hash pins.
        #expect(
            FineTuneTrainingData.orderedFiles(in: training).map(\.relative)
                == ["b.jsonl", "z/a.jsonl"])

        // Bytes are the FILE's bytes: raw JSON syntax, trailing newline, no
        // document header, no row flattening — the whole point of the
        // structured route.
        let first = try #require(payload.files.first)
        #expect(first.content == bRows)
        #expect(first.content.contains(#"{"text""#))
        #expect(!first.content.contains(FineTuneTrainingData.documentHeader("b.jsonl")))
        #expect(payload.files[2].content == vRows)

        // Hashes are checkable against an independent digest of the bytes.
        for file in payload.files {
            let data = try Data(
                contentsOf: FineTuneStore.absoluteURL(file.path))
            #expect(file.sha256 == sha256(data))
            #expect(file.sha256 == sha256(Data(file.content.utf8)))
        }

        // READMEs are skipped, not refused (dataset folders carry them).
        #expect(!payload.files.contains { $0.path.hasSuffix("README.md") })
    }

    @Test func structuredPayloadRefusesNonJSONLAsNotStructured() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let training = workspace.appending(components: "adapters", "j", "training")
        try write(#"{"text":"row"}"# + "\n", to: training.appending(component: "a.jsonl"))
        try write("a whole opinion", to: training.appending(component: "case.txt"))
        try write(
            #"{"text":"held out"}"# + "\n",
            to: workspace.appending(
                components: "adapters", "j", "validation", "v.jsonl"))

        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training",
                validationPath: "adapters/j/validation")
            Issue.record("a mixed-document folder must refuse structured upload")
        } catch let problem as FineTuneTrainingData.Problem {
            // The KIND is what lets the panel fall back to the legacy inline
            // route instead of refusing a document corpus outright.
            #expect(problem.kind == .notStructured)
            #expect(problem.message.contains("case.txt"))
            #expect(problem.message.contains(training.path))
        }
    }

    @Test func structuredPayloadRefusesMissingOrEmptyValidation() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let training = workspace.appending(components: "adapters", "j", "training")
        let validation = workspace.appending(components: "adapters", "j", "validation")
        try write(#"{"text":"row"}"# + "\n", to: training.appending(component: "a.jsonl"))

        // Validation folder absent: refused, with the resolved path.
        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training",
                validationPath: "adapters/j/validation")
            Issue.record("a missing validation folder must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.kind == .missingValidation)
            #expect(problem.message.contains(validation.path))
            #expect(problem.message.contains("does not exist"))
        }

        // Present but empty: still refused — evidence needs a held-out set.
        try FileManager.default.createDirectory(
            at: validation, withIntermediateDirectories: true)
        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training",
                validationPath: "adapters/j/validation")
            Issue.record("an empty validation folder must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.kind == .missingValidation)
            #expect(problem.message.contains(validation.path))
            #expect(problem.message.contains("is empty"))
        }

        // Unset entirely: named as such, not as a mysterious empty folder.
        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training", validationPath: "  ")
            Issue.record("an unset validation path must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.kind == .missingValidation)
            #expect(problem.message.contains("no validation data is set"))
        }
    }

    @Test func structuredPayloadRefusesEmptyTrainingAndCollapsedSplit() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        let training = workspace.appending(components: "adapters", "j", "training")
        try FileManager.default.createDirectory(
            at: training, withIntermediateDirectories: true)
        try write(
            #"{"text":"held out"}"# + "\n",
            to: workspace.appending(
                components: "adapters", "j", "validation", "v.jsonl"))

        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training",
                validationPath: "adapters/j/validation")
            Issue.record("an empty training folder must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.kind == .unusable)
            #expect(problem.message.contains(training.path))
            #expect(problem.message.contains("is empty"))
        }

        // Both sides pointing at the same folder is not a split at all.
        try write(#"{"text":"row"}"# + "\n", to: training.appending(component: "a.jsonl"))
        do {
            _ = try FineTuneTrainingData.structuredPayload(
                trainingPath: "adapters/j/training",
                validationPath: "adapters/j/training")
            Issue.record("a collapsed split must refuse")
        } catch let problem as FineTuneTrainingData.Problem {
            #expect(problem.kind == .missingValidation)
            #expect(problem.message.contains(training.path))
        }
    }

    /// The legacy stored convention ("<folder>/train.jsonl") resolves to the
    /// folder on the structured route too, so an adapter saved by the old
    /// picker uploads the same file set the folder hash pins.
    @Test func structuredPayloadHonorsTheLegacyDefaultFilenameConvention() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }
        try write(
            #"{"text":"row"}"# + "\n",
            to: workspace.appending(components: "data", "train", "train.jsonl"))
        try write(
            #"{"text":"held out"}"# + "\n",
            to: workspace.appending(components: "data", "val", "validation.jsonl"))

        let payload = try FineTuneTrainingData.structuredPayload(
            trainingPath: "data/train/train.jsonl",
            validationPath: "data/val/validation.jsonl")
        #expect(
            payload.files.map(\.path) == [
                "data/train/train.jsonl", "data/val/validation.jsonl",
            ])
        #expect(
            payload.trainingResolvedPath
                == workspace.appending(components: "data", "train").path)
    }

    // MARK: - Workspace adapters/ seeding + per-adapter homes

    @Test func workspaceCreateSeedsTopLevelAdaptersDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "ft-create-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try WorkspaceStore.create(at: root)

        var isDirectory = ObjCBool(false)
        #expect(
            FileManager.default.fileExists(
                atPath: root.appending(component: "adapters").path,
                isDirectory: &isDirectory) && isDirectory.boolValue)
        // Trained weight binaries are excluded from the workspace repo, the
        // training data itself is versioned.
        let gitignore = try String(
            contentsOf: root.appending(component: ".gitignore"), encoding: .utf8)
        #expect(gitignore.contains("adapters/**/*.safetensors"))
    }

    @Test func createAdapterHomeIsIdempotentAndNeverOverwrites() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }

        let home = try FineTuneStore.createAdapterHome(slug: "judicial-virtues")
        #expect(home.root.path == workspace.appending(
            components: "adapters", "judicial-virtues").path)
        var isDirectory = ObjCBool(false)
        #expect(
            FileManager.default.fileExists(
                atPath: home.training.path, isDirectory: &isDirectory)
                && isDirectory.boolValue)
        #expect(FileManager.default.fileExists(atPath: home.validation.path))

        // Adopt, never overwrite: a file already inside survives a re-create.
        let existing = home.training.appending(component: "keep.txt")
        try write("keep me", to: existing)
        let again = try FineTuneStore.createAdapterHome(slug: "judicial-virtues")
        #expect(again == home)
        #expect(try String(contentsOf: existing, encoding: .utf8) == "keep me")
    }

    // MARK: - Drop-copy semantics (the house never-overwrite rule)

    @Test func copyDroppedFilesCopiesIdenticalIsQuietDifferingRefuses() throws {
        let workspace = try makeTempWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let sourceDir = workspace.appending(component: "sources")
        let folder = workspace.appending(component: "training")
        let fresh = sourceDir.appending(component: "fresh.txt")
        let same = sourceDir.appending(component: "same.txt")
        let conflict = sourceDir.appending(component: "conflict.txt")
        try write("fresh bytes", to: fresh)
        try write("same bytes", to: same)
        try write("new bytes", to: conflict)
        try write("same bytes", to: folder.appending(component: "same.txt"))
        try write("old bytes", to: folder.appending(component: "conflict.txt"))

        let result = FineTuneStore.copyDroppedFiles(
            [fresh, same, conflict, sourceDir], into: folder)

        #expect(result.copied == ["fresh.txt"])
        #expect(
            try String(
                contentsOf: folder.appending(component: "fresh.txt"),
                encoding: .utf8) == "fresh bytes")
        // Identical bytes: quietly fine, untouched.
        #expect(result.identical == ["same.txt"])
        // Differing bytes: refused with a plain-language message; the
        // existing file is never overwritten.
        #expect(result.refusals.count == 2)
        #expect(
            result.refusals.contains {
                $0.contains("conflict.txt") && $0.contains("different contents")
            })
        #expect(
            try String(
                contentsOf: folder.appending(component: "conflict.txt"),
                encoding: .utf8) == "old bytes")
        // A dropped folder is refused, not silently ignored.
        #expect(result.refusals.contains { $0.contains("is a folder") })
    }

    // MARK: - Panel: creation defaults + drop entry point

    @Test @MainActor func createAdapterProjectDefaultsToWorkspaceAdapterHome() throws {
        let workspace = try makeTempWorkspace()
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = workspace
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: workspace)
        }

        let panel = FineTuningPanel()
        panel.notices = PanelNotices(
            fileURL: workspace.appending(component: "notices.json"))
        panel.newAdapterName = "Judicial Virtues"
        panel.newAdapterBaseModelID = "Qwen/Qwen3-4B-MLX-4bit"
        panel.createAdapterProject()

        let record = try #require(panel.selectedAdapter)
        #expect(record.artifact.adapterDirectory == "adapters/judicial-virtues")
        #expect(record.artifact.trainingDataPath == "adapters/judicial-virtues/training")
        #expect(record.artifact.validationDataPath == "adapters/judicial-virtues/validation")
        #expect(
            FileManager.default.fileExists(
                atPath: workspace.appending(
                    components: "adapters", "judicial-virtues", "training").path))
        // The trainer controls picked up the same defaults.
        #expect(panel.trainingDataPath == "adapters/judicial-virtues/training")

        // Idempotent: creating the same name again adopts the folder.
        panel.createAdapterProject()
        #expect(panel.status?.contains("could not") != true)

        // Drop entry point: files are COPIED into training/ and the note
        // reports the count.
        let dropSource = workspace.appending(component: "drop-src.txt")
        try write("dropped doc", to: dropSource)
        panel.importDroppedFiles([dropSource], to: .training)
        #expect(panel.status?.contains("added 1 file to training/") == true)
        #expect(
            try String(
                contentsOf: workspace.appending(
                    components: "adapters", "judicial-virtues", "training",
                    "drop-src.txt"),
                encoding: .utf8) == "dropped doc")
        // Same drop again: identical bytes are quietly fine.
        panel.importDroppedFiles([dropSource], to: .training)
        #expect(panel.status?.contains("1 identical file already present") == true)
    }
}
