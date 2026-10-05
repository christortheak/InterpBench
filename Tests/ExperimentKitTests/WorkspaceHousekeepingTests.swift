import Foundation
import Testing

@testable import ExperimentKit

/// `experiment rename|delete`, `design rename|delete`, and `agent delete` on
/// the Mac command line: each previews by default, applies only with the
/// reviewed digest and `--yes`, and goes through the same owners the app's
/// buttons use.
@Suite(.serialized) struct WorkspaceHousekeepingTests {

    private func withWorkspace(_ body: (URL) async throws -> Void) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "housekeeping-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        try await body(root)
    }

    private func run(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding, workspaceIsResolved: { true })
            .run(namespace: namespace, args + ["--json"])
    }

    private func string(_ outcome: ExperimentCLIOutcome, _ key: String) -> String? {
        if case .string(let value) = outcome.envelope.result?[key] { return value }
        return nil
    }

    private func bool(_ outcome: ExperimentCLIOutcome, _ key: String) -> Bool? {
        if case .bool(let value) = outcome.envelope.result?[key] { return value }
        return nil
    }

    private func setStatus(_ status: ExperimentManifest.Status, _ name: String) throws {
        var manifest = try ExperimentStore.load(name: name)
        manifest.status = status
        try ExperimentStore.save(manifest)
    }

    private func trashFolders(_ parent: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix(".trash-") }
    }

    private func makeAgent(_ root: URL, slug: String = "helper") throws -> String {
        let path = "runs/model-variants/\(slug)/model-variant.json"
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let artifact = ModelVariantArtifact(
            name: slug, baseModelID: "test/model", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Be brief.")
        try JSONEncoder().encode(artifact).write(to: url)
        return path
    }

    // MARK: - Studies

    @Test func experimentRenamePreviewsThenAppliesWithTheReviewedDigest() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(name: "first", description: "d", modelID: "test/model")
            let before = try DraftAuthoringSnapshot(workspaceRoot: root, name: "first")

            let preview = await run("experiment", ["rename", "first", "Second Name"])
            #expect(preview.envelope.exitCode == 0)
            #expect(preview.envelope.changed == false)
            #expect(bool(preview, "applied") == false)
            #expect(string(preview, "newName") == "second-name")
            #expect(string(preview, "manifestFileSHA256") == before.file.sha256)
            #expect(string(preview, "confirmCommand")
                == "steerlab-cli experiment rename first 'Second Name' --manifest-sha256 \(before.file.sha256) --yes")
            #expect(preview.envelope.nextAction?.missingPermissionFlags == ["--manifest-sha256", "--yes"])
            // Nothing moved.
            #expect(try ExperimentStore.load(name: "first").name == "first")

            let applied = await run("experiment", [
                "rename", "first", "Second Name", "--manifest-sha256", before.file.sha256, "--yes",
            ])
            #expect(applied.envelope.exitCode == 0)
            #expect(applied.envelope.changed)
            #expect(bool(applied, "applied") == true)
            #expect(string(applied, "destination") == "experiments/second-name")
            #expect(try ExperimentStore.load(name: "second-name").name == "second-name")
            #expect(!FileManager.default.fileExists(
                atPath: root.appending(path: "experiments/first").path))
        }
    }

    @Test func experimentRenameRefusesAStaleDigestAndAFrozenStudy() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(name: "draft", description: "d", modelID: "test/model")
            let stale = try DraftAuthoringSnapshot(workspaceRoot: root, name: "draft").file.sha256
            var edited = try ExperimentStore.load(name: "draft")
            edited.experimentDescription = "changed elsewhere"
            try ExperimentStore.save(edited)
            let refused = await run("experiment", [
                "rename", "draft", "other", "--manifest-sha256", stale, "--yes",
            ])
            #expect(refused.envelope.exitCode == 65)
            #expect(refused.envelope.error?.code == "staleManifest")
            #expect(try ExperimentStore.load(name: "draft").name == "draft")

            _ = try ExperimentStore.create(name: "kept", description: "d", modelID: "test/model")
            try setStatus(.frozen, "kept")
            let frozen = await run("experiment", ["rename", "kept", "renamed"])
            #expect(frozen.envelope.exitCode == 65)
            #expect(frozen.envelope.error?.gate == "statusImmutable")
            #expect(frozen.envelope.error?.repairAction.contains("experiment duplicate kept kept-v2") == true)
            #expect(try ExperimentStore.load(name: "kept").status == .frozen)

            let unconfirmed = await run("experiment", ["rename", "draft", "other", "--yes"])
            #expect(unconfirmed.envelope.exitCode == 64)
        }
    }

    @Test func experimentDeleteMovesTheDraftToTrashAndLeavesRunsAlone() async throws {
        try await withWorkspace { root in
            let manifest = try ExperimentStore.create(name: "doomed", description: "d", modelID: "test/model")
            let runFolder = root.appending(path: "runs/20261005-doomed-run")
            try FileManager.default.createDirectory(at: runFolder, withIntermediateDirectories: true)
            let stamped = try JSONEncoder().encode(manifest)
            try stamped.write(to: runFolder.appending(component: "experiment.json"))

            let preview = await run("experiment", ["delete", "doomed"])
            #expect(preview.envelope.exitCode == 0)
            #expect(preview.envelope.result?["runsRecordingName"] == .number(1))
            #expect(FileManager.default.fileExists(atPath: root.appending(path: "experiments/doomed").path))
            let digest = try #require(string(preview, "manifestFileSHA256"))

            let applied = await run("experiment", ["delete", "doomed", "--manifest-sha256", digest, "--yes"])
            #expect(applied.envelope.exitCode == 0)
            let destination = try #require(string(applied, "destination"))
            #expect(destination.hasPrefix("experiments/.trash-"))
            #expect(destination.hasSuffix("/doomed"))
            #expect(FileManager.default.fileExists(
                atPath: root.appending(path: destination).appending(component: "experiment.json").path))
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/doomed").path))
            #expect(try Data(contentsOf: runFolder.appending(component: "experiment.json")) == stamped)
            #expect(ExperimentStore.list().isEmpty)
        }
    }

    @Test func experimentDeleteRefusesAFrozenStudy() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(name: "kept", description: "d", modelID: "test/model")
            try setStatus(.complete, "kept")
            let refused = await run("experiment", ["delete", "kept"])
            #expect(refused.envelope.exitCode == 65)
            #expect(refused.envelope.error?.gate == "statusImmutable")
            #expect(refused.envelope.error?.reason.contains("cannot be deleted") == true)
            #expect(FileManager.default.fileExists(atPath: root.appending(path: "experiments/kept").path))
        }
    }

    // MARK: - Templates

    @Test func designRenameAndDeleteTemplatesWithPreviewAndConfirmation() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(name: "source", description: "d", modelID: "test/model")
            let mint = try StudyTemplateStore.templateFromStudy(experimentName: "source", named: "wave")
            #expect(mint.template.name == "wave")
            let before = try StudyDesignSnapshot(workspaceRoot: root, name: "wave")

            let preview = await run("design", ["rename", "wave", "wave-two"])
            #expect(preview.envelope.exitCode == 0)
            #expect(string(preview, "designFileSHA256") == before.file.sha256)
            #expect(string(preview, "kind") == "template")
            #expect(try StudyTemplateStore.load(name: "wave").name == "wave")

            let renamed = await run("design", [
                "rename", "wave", "wave-two", "--file-sha256", before.file.sha256, "--yes",
            ])
            #expect(renamed.envelope.exitCode == 0)
            #expect(try StudyTemplateStore.load(name: "wave-two").name == "wave-two")

            let stale = await run("design", ["delete", "wave-two", "--file-sha256", before.file.sha256, "--yes"])
            #expect(stale.envelope.exitCode == 65)
            #expect(stale.envelope.error?.code == "designChanged")

            let current = try StudyDesignSnapshot(workspaceRoot: root, name: "wave-two").file.sha256
            let deleted = await run("design", ["delete", "wave-two", "--file-sha256", current, "--yes"])
            #expect(deleted.envelope.exitCode == 0)
            let destination = try #require(string(deleted, "destination"))
            #expect(destination.hasPrefix("templates/.trash-"))
            #expect(FileManager.default.fileExists(
                atPath: root.appending(path: destination).appending(component: "template.json").path))
            // The trash folder is not a template and is not reported as a broken one.
            let catalog = try StudyDesignAuthoring.list(workspaceRoot: root)
            #expect(catalog.entries.isEmpty)
            #expect(catalog.issues.isEmpty)

            let missing = await run("design", ["delete", "nowhere"])
            #expect(missing.envelope.exitCode == 66)
        }
    }

    // MARK: - Agents

    @Test func agentDeleteMovesAnUnusedAgentAndRefusesAUsedOne() async throws {
        try await withWorkspace { root in
            let path = try makeAgent(root)
            let digest = try AgentArtifactSnapshot(workspaceRoot: root, path: path).file.sha256

            _ = try ExperimentStore.create(name: "user", description: "d", modelID: "test/model")
            let study = try DraftAuthoringSnapshot(workspaceRoot: root, name: "user")
            let agent = try StudyAgentAuthoring.reviewArtifact(
                path: path, workspaceRoot: root, expectedFileSHA256: digest)
            try StudyAgentAuthoring.attach(agent, reviewed: study)

            let refused = await run("agent", ["delete", path, "--artifact-sha256", digest, "--yes"])
            #expect(refused.envelope.exitCode == 65)
            #expect(refused.envelope.error?.code == WorkspaceHousekeeping.agentInUseCode)
            #expect(refused.envelope.error?.reason.contains("user") == true)
            #expect(refused.envelope.result?["usedBy"] == .array([.string("user")]))
            #expect(FileManager.default.fileExists(atPath: root.appending(path: path).path))

            // Once no study uses it, the same agent can go.
            try ExperimentStore.moveDraftToTrash(name: "user")
            let preview = await run("agent", ["delete", path])
            #expect(preview.envelope.exitCode == 0)
            #expect(string(preview, "artifactFileSHA256") == digest)
            #expect(FileManager.default.fileExists(atPath: root.appending(path: path).path))

            let applied = await run("agent", ["delete", path, "--artifact-sha256", digest, "--yes"])
            #expect(applied.envelope.exitCode == 0)
            let destination = try #require(string(applied, "destination"))
            #expect(destination.hasPrefix("runs/model-variants/.trash-"))
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: path).path))
            #expect(try StudyAgentAuthoring.list(workspaceRoot: root).agents.isEmpty)
        }
    }

    /// A run folder is evidence and `runs/` is append-only, so an agent a run
    /// saved is refused, plainly, and its folder is left exactly as it was.
    @Test func agentSavedByARunIsRefusedAndItsRunFolderStays() async throws {
        try await withWorkspace { root in
            let runFolder = root.appending(path: "runs/20261005-variant-imported")
            try FileManager.default.createDirectory(at: runFolder, withIntermediateDirectories: true)
            try Data(#"{"runType": "variant-save", "schemaVersion": 2}"#.utf8)
                .write(to: runFolder.appending(component: "config.json"))
            let artifact = ModelVariantArtifact(
                name: "imported", baseModelID: "test/model", promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Be brief.")
            try JSONEncoder().encode(artifact).write(to: runFolder.appending(component: "imported.json"))
            let path = "runs/20261005-variant-imported/imported.json"
            #expect(try StudyAgentAuthoring.list(workspaceRoot: root).agents.map(\.path) == [path])

            let digest = try AgentArtifactSnapshot(workspaceRoot: root, path: path).file.sha256
            for args in [["delete", path], ["delete", path, "--artifact-sha256", digest, "--yes"]] {
                let refused = await run("agent", args)
                #expect(refused.envelope.exitCode == 65)
                #expect(refused.envelope.error?.code == WorkspaceHousekeeping.agentIsRunEvidenceCode)
                #expect(refused.envelope.error?.reason.contains("saved by a run") == true)
            }
            #expect(FileManager.default.fileExists(atPath: root.appending(path: path).path))
            #expect(trashFolders(root.appending(component: "runs")).isEmpty)
            #expect(try StudyAgentAuthoring.list(workspaceRoot: root).agents.map(\.path) == [path])
        }
    }

    /// The printed confirmation runs as is: a name with a space or an
    /// apostrophe is quoted the way Python's `shlex.quote` quotes it.
    @Test func theConfirmationCommandQuotesArgumentsLikeShlex() {
        #expect(WorkspaceHousekeepingCLI.shellQuoted("plain-name_1.v2") == "plain-name_1.v2")
        #expect(WorkspaceHousekeepingCLI.shellQuoted("Second Name") == "'Second Name'")
        #expect(WorkspaceHousekeepingCLI.shellQuoted("it's") == "'it'\"'\"'s'")
        #expect(WorkspaceHousekeepingCLI.shellQuoted("") == "''")
    }

    // MARK: - The app's owners

    @Test func theAppsDeleteOwnersMoveToTrashRatherThanErase() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(name: "source", description: "d", modelID: "test/model")
            _ = try StudyTemplateStore.templateFromStudy(experimentName: "source", named: "kept")
            let moved = try StudyTemplateStore.delete(name: "kept")
            #expect(FileManager.default.fileExists(atPath: moved.appending(component: "template.json").path))
            #expect(trashFolders(root.appending(component: "templates")).count == 1)
            #expect(StudyTemplateStore.list().isEmpty)

            let path = try makeAgent(root, slug: "spare")
            let record = try AgentArtifactSnapshot(workspaceRoot: root, path: path)
            let landed = try WorkspaceHousekeeping.deleteAgent(reviewed: record)
            #expect(FileManager.default.fileExists(atPath: landed.appending(component: "model-variant.json").path))
            #expect(ModelVariantStore.scan(
                directory: root.appending(path: "runs/model-variants"),
                importedRoot: root.appending(component: "runs")).isEmpty)
        }
    }
}
