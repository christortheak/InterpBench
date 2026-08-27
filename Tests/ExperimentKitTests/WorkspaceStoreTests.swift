import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

/// The workspace split: runtime root resolution (pure, injected seams —
/// the process env cannot be set in-process), workspace creation/seeding,
/// and open() validation. Nothing here mutates process-global state: the
/// resolution is tested through `WorkspaceRoot.resolve` and open() through
/// its injected defaults + override seams, so parallel suites reading
/// `VectorCatalog.projectRoot` are never disturbed.
@Suite(.serialized) struct WorkspaceStoreTests {

    private func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "ws-\(UUID().uuidString)")
    }

    private func gitAvailable() -> Bool {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = ["--version"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return false }
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    // MARK: - Root resolution precedence

    @Test func resolutionPrecedenceIsEnvOverrideDefaultsFallback() throws {
        let fallback = URL(filePath: "/dev/legacy-repo-root")
        let existing = tempDirectory()
        try FileManager.default.createDirectory(
            at: existing, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: existing) }

        // 1. STEERLAB_WORKSPACE beats override, defaults, and fallback.
        #expect(
            WorkspaceRoot.resolve(
                environment: [WorkspaceRoot.environmentKey: "/env/ws"],
                programmaticOverride: URL(filePath: "/override/ws"),
                persistedPath: existing.path,
                fallback: fallback
            ).path == "/env/ws")

        // A blank env value is ignored, not honored as an empty root.
        #expect(
            WorkspaceRoot.resolve(
                environment: [WorkspaceRoot.environmentKey: "   "],
                programmaticOverride: nil,
                persistedPath: nil,
                fallback: fallback
            ) == fallback)

        // 2. Programmatic override beats defaults and fallback.
        #expect(
            WorkspaceRoot.resolve(
                environment: [:],
                programmaticOverride: URL(filePath: "/override/ws"),
                persistedPath: existing.path,
                fallback: fallback
            ).path == "/override/ws")

        // 3. Persisted UserDefaults choice — honored only while the
        //    directory still exists.
        #expect(
            WorkspaceRoot.resolve(
                environment: [:],
                programmaticOverride: nil,
                persistedPath: existing.path,
                fallback: fallback
            ).path == existing.standardizedFileURL.path)
        #expect(
            WorkspaceRoot.resolve(
                environment: [:],
                programmaticOverride: nil,
                persistedPath: "/no/such/dir-\(UUID().uuidString)",
                fallback: fallback
            ) == fallback)

        // 4. Nothing configured → the legacy dev/test fallback.
        #expect(
            WorkspaceRoot.resolve(
                environment: [:],
                programmaticOverride: nil,
                persistedPath: nil,
                fallback: fallback
            ) == fallback)
    }

    @Test func legacyFallbackIsTheCodeCheckout() {
        // bundledSeedRoot is the compile-time repo root — the seed source and
        // resolution fallback #4. Sanity: it contains the package manifest.
        #expect(
            FileManager.default.fileExists(
                atPath: VectorCatalog.bundledSeedRoot
                    .appending(component: "Package.swift").path))
    }

    // MARK: - create()

    @Test func createSeedsSkeletonManifestFilesAndGit() throws {
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let created = try WorkspaceStore.create(at: root)
        let fm = FileManager.default

        for sub in WorkspaceStore.promptSubdirectories {
            var isDirectory: ObjCBool = false
            #expect(
                fm.fileExists(
                    atPath: created.appending(components: "prompts", sub).path,
                    isDirectory: &isDirectory) && isDirectory.boolValue,
                "missing prompts/\(sub)")
        }
        #expect(fm.fileExists(atPath: created.appending(component: "experiments").path))
        #expect(fm.fileExists(atPath: created.appending(component: "runs").path))
        #expect(
            fm.fileExists(
                atPath: created.appending(component: WorkspaceStore.markerFileName).path))

        // Seeded byte-identical from the curated seed tree — every manifest
        // entry, and nothing else (WP1: the seed root is
        // `<checkout>/WorkspaceSeed/`, not the research checkout itself).
        let seed = try CodeResources.workspaceSeed()
        #expect(seed.lastPathComponent == "WorkspaceSeed")
        for relative in WorkspaceStore.seedManifest {
            #expect(
                (try? Data(contentsOf: created.appending(path: relative)))
                    == (try? Data(contentsOf: seed.appending(path: relative))),
                "seed mismatch for \(relative)")
        }

        // runs/ are bulk immutable outputs: gitignored in the workspace repo
        // (the frozen experiments' pinned/ snapshots are the input floor).
        let gitignore = try String(
            contentsOf: created.appending(component: ".gitignore"), encoding: .utf8)
        #expect(gitignore.contains("runs/"))

        #expect(WorkspaceStore.isWorkspace(url: created))
        // Creating on top of an existing workspace refuses.
        #expect(throws: ExperimentError.self) {
            try WorkspaceStore.create(at: created)
        }

        if gitAvailable() {
            #expect(
                fm.fileExists(atPath: created.appending(component: ".git").path),
                "workspace should be git-initialized when git is available")
        }
    }

    /// Creation is TRANSACTIONAL: it stages a whole workspace in a temp
    /// sibling and installs it with one rename, so a failure part-way
    /// through leaves NO destination — not a directory with `prompts/` in it
    /// that `isWorkspace` would happily accept and that the next verb would
    /// read as real. The failure is injected the way it actually happens: a
    /// seed file the copier cannot read.
    @Test func aFailedCreateLeavesNoWorkspaceBehind() throws {
        let fm = FileManager.default
        let seed = tempDirectory()
        let destination = tempDirectory()
        defer {
            // Restore permissions first or the seed tree is undeletable.
            let victim = seed.appending(path: WorkspaceStore.seedManifest[0])
            try? fm.setAttributes([.posixPermissions: 0o644], ofItemAtPath: victim.path)
            try? fm.removeItem(at: seed)
            try? fm.removeItem(at: destination)
        }

        // A seed root with every manifest file present …
        for relative in WorkspaceStore.seedManifest {
            let url = seed.appending(path: relative)
            try fm.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("seed\n".utf8).write(to: url)
        }
        // … one of which cannot be read. (Skipped for root, who can read it
        // anyway and would see a spurious success.)
        let victim = seed.appending(path: WorkspaceStore.seedManifest[0])
        try fm.setAttributes([.posixPermissions: 0o000], ofItemAtPath: victim.path)
        try #require(
            (try? Data(contentsOf: victim)) == nil,
            """
            this process can read a 0000 file (running as root?) — the \
            failure cannot be injected
            """)

        #expect(throws: (any Error).self) {
            try WorkspaceStore.create(at: destination, seedingFrom: seed)
        }
        // The whole point: nothing at the destination, and no staging tree
        // left in its parent.
        #expect(
            !fm.fileExists(atPath: destination.path),
            "a failed create left a partial workspace at \(destination.path)")
        #expect(!WorkspaceStore.isWorkspace(url: destination))
        // The staging tree THIS create made, and only that one. `create`
        // names it after its own destination
        // (`.<destination>.steerlab-staging-<uuid>`), so the destination-
        // specific prefix is exactly what the transaction guarantees to
        // clean up. Matching the bare word "steerlab-staging" across the
        // shared system temp directory asserted something else entirely —
        // that no OTHER process on this machine has ever been interrupted
        // mid-create — and a single stale directory from any earlier run
        // failed a test about this one (review round 7, finding 7).
        let prefix = ".\(destination.lastPathComponent).steerlab-staging-"
        let siblings = (try? fm.contentsOfDirectory(
            atPath: destination.deletingLastPathComponent().path)) ?? []
        let ours = siblings.filter { $0.hasPrefix(prefix) }
        #expect(ours.isEmpty, "staging tree was not cleaned up: \(ours)")
    }

    /// The preflight half: an existing NON-EMPTY destination is refused
    /// outright rather than merged into. (An empty directory is fine — the
    /// app's folder picker creates one.)
    @Test func createRefusesANonEmptyDestinationAndAcceptsAnEmptyOne() throws {
        let fm = FileManager.default
        let occupied = tempDirectory()
        let empty = tempDirectory()
        defer {
            try? fm.removeItem(at: occupied)
            try? fm.removeItem(at: empty)
        }

        try fm.createDirectory(at: occupied, withIntermediateDirectories: true)
        try Data("mine\n".utf8).write(to: occupied.appending(component: "notes.txt"))
        #expect(throws: ExperimentError.self) {
            try WorkspaceStore.create(at: occupied)
        }
        // The caller's file is untouched.
        #expect(fm.fileExists(atPath: occupied.appending(component: "notes.txt").path))
        #expect(!WorkspaceStore.isWorkspace(url: occupied))

        try fm.createDirectory(at: empty, withIntermediateDirectories: true)
        #expect(WorkspaceStore.isWorkspace(url: try WorkspaceStore.create(at: empty)))
    }

    /// The seed tree carries its own copy of the shared templates (a bundle
    /// must be self-contained), while the research checkout keeps
    /// `prompts/templates/` because the Python engine reads its shipped
    /// defaults from the CODE root. Two copies drift unless something holds
    /// them together — this does.
    @Test func seedTemplatesMatchTheResearchTemplateTree() throws {
        let seed = try CodeResources.workspaceSeed()
        let checkout = VectorCatalog.bundledSeedRoot
        var compared = 0
        for relative in WorkspaceStore.seedManifest
        where relative.hasPrefix("prompts/templates/") {
            let research = checkout.appending(path: relative)
            guard FileManager.default.fileExists(atPath: research.path) else {
                continue  // seed-only file (nothing to drift against)
            }
            compared += 1
            #expect(
                (try? Data(contentsOf: seed.appending(path: relative)))
                    == (try? Data(contentsOf: research)),
                "WorkspaceSeed/\(relative) has drifted from \(relative)")
        }
        #expect(compared >= 15, "the drift gate compared almost nothing")
    }

    /// The seeded task-prompts example is documentation that RUNS: it must
    /// parse through the run loop's own loader, and it must exercise the
    /// `responseFormat` declaration it exists to demonstrate — including
    /// the case the compatibility rule exists for (options present, format
    /// not answer-token scorable).
    @Test func seededExampleTaskPromptsParseAndDeclareResponseFormats() throws {
        let seed = try CodeResources.workspaceSeed()
            .appending(path: "prompts/tasks/example-task-prompts.jsonl")
        let data = try Data(contentsOf: seed)
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts.count >= 4)

        let document = try TaskPromptsDocument.load(data)
        let items = document.responseFormatItems
        #expect(items.allSatisfy { $0.format != nil }, "every row declares a format")
        #expect(items.contains { $0.format == .label && $0.hasOptions })
        #expect(items.contains { $0.format == .json })
        #expect(items.contains { $0.format == .freeText })
        // The json row carries options on purpose: it is the shape the rule
        // refuses for answer-token scoring, and refusing is the lesson.
        #expect(ResponseFormat.unscorableItems(items).count == 1)
    }

    // MARK: - Nested-workspace auto-commit safety-skip (WP1)

    /// Freeze must never commit into a repository the researcher did not
    /// hand it. The server has skipped auto-commit for a workspace nested
    /// inside a larger repo since 2026-07-13; Swift only scoped its
    /// `git add -A .` to the workspace directory, which still landed a
    /// commit in the PARENT repo's history. This is the Swift twin.
    @Test func nestedWorkspaceIsDetectedAndStandaloneIsNot() throws {
        guard gitAvailable() else { return }
        let fm = FileManager.default
        let parent = tempDirectory()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: parent) }

        func git(_ arguments: [String], in directory: URL) {
            let process = Process()
            process.executableURL = URL(filePath: "/usr/bin/git")
            process.arguments = ["-C", directory.path] + arguments
            process.standardOutput = Pipe()
            process.standardError = Pipe()
            try? process.run()
            process.waitUntilExit()
        }

        // An outer repository with a HAND-ASSEMBLED workspace inside it —
        // the live hazard. (`WorkspaceStore.create` git-inits, which would
        // make the inner folder its own work-tree root and no longer
        // nested; that is exactly why app-made workspaces are safe and
        // hand-made ones are not.)
        git(["init"], in: parent)
        let nested = parent.appending(component: "workspace")
        try fm.createDirectory(
            at: nested.appending(component: "prompts"),
            withIntermediateDirectories: true)
        #expect(WorkspaceStore.isWorkspace(url: nested))

        let enclosing = try #require(
            ExperimentStore.nestedWorkspaceRepositoryRoot(of: nested),
            "a workspace inside a larger repo must report the enclosing repo")
        #expect(
            enclosing.path
                == parent.resolvingSymlinksInPath().standardizedFileURL.path)

        // The advisory names both roots and says what to do instead.
        let advisory = ExperimentStore.nestedWorkspaceAutoCommitAdvisory(
            freezing: "demo", workspace: nested, repository: enclosing)
        #expect(advisory.contains("auto-commit skipped"))
        #expect(advisory.contains("subdirectory of the git repository"))
        #expect(advisory.contains(nested.path))
        #expect(advisory.contains(enclosing.path))
        #expect(advisory.contains("Commit the pinned inputs yourself"))

        // `create` git-inits the workspace itself, so the standalone case is
        // the one the workspace manager produces: outside any parent repo,
        // its own work-tree root, and therefore NOT nested.
        let standalone = tempDirectory()
        defer { try? fm.removeItem(at: standalone) }
        _ = try WorkspaceStore.create(at: standalone)
        #expect(ExperimentStore.nestedWorkspaceRepositoryRoot(of: standalone) == nil)

        // A plain folder that is no git work tree at all is not "nested".
        let plain = tempDirectory()
        try fm.createDirectory(at: plain, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: plain) }
        #expect(ExperimentStore.nestedWorkspaceRepositoryRoot(of: plain) == nil)
    }

    // MARK: - The agent contract (WP0 step 10)

    @Test func newWorkspaceCarriesAgentContract() throws {
        let parent = tempDirectory()
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }
        let fm = FileManager.default

        let created = try WorkspaceStore.create(at: parent.appending(component: "ws"))
        let contract = created.appending(component: AgentContract.fileName)
        let text = try String(contentsOf: contract, encoding: .utf8)

        // Header first, then the contract body verbatim.
        #expect(text.hasPrefix(AgentContract.generatedHeader))
        #expect(text.contains(AgentContract.body))
        #expect(text == AgentContract.contents())
        #expect(text.contains("# AGENTS.md"))

        // Written before the initial commit, so a workspace's own history
        // carries it (git may be unavailable in CI — then there is no repo
        // to check and creation is still correct).
        if gitAvailable() {
            let process = Process()
            process.executableURL = URL(filePath: "/usr/bin/git")
            process.arguments = ["-C", created.path, "ls-files", AgentContract.fileName]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = Pipe()
            try process.run()
            let listed = String(
                decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            process.waitUntilExit()
            #expect(listed.contains(AgentContract.fileName))
        }

        // Lazy write for workspaces that predate the contract: opening one
        // without the file writes it…
        let legacy = parent.appending(component: "legacy")
        try fm.createDirectory(
            at: legacy.appending(component: "prompts"), withIntermediateDirectories: true)
        let legacyContract = legacy.appending(component: AgentContract.fileName)
        #expect(!fm.fileExists(atPath: legacyContract.path))
        let suiteName = "workspace-contract-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        try WorkspaceStore.open(at: legacy, defaults: defaults, setOverride: { _ in })
        #expect(
            (try? String(contentsOf: legacyContract, encoding: .utf8))
                == AgentContract.contents())

        // …and an EXISTING one is never overwritten: a researcher may have
        // edited it, and a workspace is data, not a managed install.
        let edited = "# AGENTS.md\n\nmy own notes\n"
        try edited.write(to: legacyContract, atomically: true, encoding: .utf8)
        try WorkspaceStore.open(at: legacy, defaults: defaults, setOverride: { _ in })
        #expect(WorkspaceStore.ensureAgentContract(at: legacy) == false)
        #expect(
            (try? String(contentsOf: legacyContract, encoding: .utf8)) == edited)
    }

    // MARK: - open() / isWorkspace()

    @Test func openValidatesPersistsAndSetsOverrideThroughSeams() throws {
        let parent = tempDirectory()
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }

        let suiteName = "workspace-tests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        var overridden: [URL] = []

        // A plain folder is not a workspace — refused before any side effect.
        #expect(throws: ExperimentError.self) {
            try WorkspaceStore.open(
                at: parent, defaults: defaults,
                setOverride: { overridden.append($0) })
        }
        #expect(overridden.isEmpty)
        #expect(defaults.string(forKey: WorkspaceRoot.defaultsKey) == nil)

        // A created workspace opens: persists the choice and applies the
        // runtime override (captured via the seam — no global mutation).
        let created = try WorkspaceStore.create(at: parent.appending(component: "ws"))
        try WorkspaceStore.open(
            at: created, defaults: defaults, setOverride: { overridden.append($0) })
        #expect(overridden == [created.standardizedFileURL])
        #expect(
            defaults.string(forKey: WorkspaceRoot.defaultsKey)
                == created.standardizedFileURL.path)

        // A hand-assembled folder with just prompts/ (no marker) is accepted.
        let bare = parent.appending(component: "bare")
        try FileManager.default.createDirectory(
            at: bare.appending(component: "prompts"), withIntermediateDirectories: true)
        #expect(WorkspaceStore.isWorkspace(url: bare))
        // …but a file, or a folder with neither marker nor prompts/, is not.
        #expect(!WorkspaceStore.isWorkspace(url: parent))
        #expect(
            !WorkspaceStore.isWorkspace(
                url: created.appending(component: WorkspaceStore.markerFileName)))
    }
}

/// Freeze lifecycle under the workspace split: the full pinned snapshot and
/// the appVersion stamp. Extends the serialized `ExperimentStoreTests` suite
/// for its `rootOverride` seam — the snapshot is pure file copies and runs
/// under the override; the git auto-commit is deliberately skipped there
/// (like the cleanliness gate), which the first test asserts.
extension ExperimentStoreTests {

    private func wsSHA256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @Test func freezeSnapshotsEveryPinnedInputWithoutRewritingPromptPaths() throws {
        try withTempRoot {
            let root = try #require(ExperimentStore.rootOverride)
            let fm = FileManager.default

            // Workspace-resident pinned inputs planted under the temp root.
            let promptsRelative = "prompts/tasks/items.jsonl"
            let promptsURL = root.appending(path: promptsRelative)
            try fm.createDirectory(
                at: promptsURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let promptBytes = Data(#"{"id":"a","prompt":"p"}"#.utf8)
            try promptBytes.write(to: promptsURL)

            let baselineRelative = "prompts/human/baseline.csv"
            let baselineURL = root.appending(path: baselineRelative)
            try fm.createDirectory(
                at: baselineURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            // Loader-shaped columns: verify shape-checks a hash-clean
            // pinned baseline (2026-07-19), and freeze runs verify.
            let baselineBytes = Data(
                "endpoint,deltaHuman,ciLower,ciUpper\naffirmRate,0.12,0.05,0.19\n"
                    .utf8)
            try baselineBytes.write(to: baselineURL)

            // A grand-mean corpus member under the override-aware emotions
            // directory.
            let storiesURL = ExperimentStore.storiesURL(for: "calm")
            try fm.createDirectory(
                at: storiesURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let storyBytes = Data(
                #"{"concept": "calm", "text": "A calm story for the corpus."}"#.utf8)
            try storyBytes.write(to: storiesURL)

            var manifest = try ExperimentStore.create(
                name: "snap", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            try ExperimentStore.attachGrandMeanConcepts(["calm"], into: &manifest)
            manifest.taskPromptsFile = promptsRelative
            manifest.taskPromptsHash = wsSHA256(promptBytes)
            manifest.humanBaseline = .init(
                path: baselineRelative, hash: wsSHA256(baselineBytes))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)

            let frozen = try ExperimentStore.freeze(name: "snap")
            #expect(frozen.status == .frozen)

            // Prompts-resident paths are NOT rewritten — hashes prove
            // identity; the snapshot is only the reproducibility floor.
            #expect(frozen.taskPromptsFile == promptsRelative)
            #expect(frozen.humanBaseline?.path == baselineRelative)

            // Verification-bytes copies of every pinned input.
            let pinned = root.appending(components: "experiments", "snap", "pinned")
            #expect(
                (try? Data(
                    contentsOf: pinned.appending(
                        components: "concepts", "french", "positive.jsonl")))
                    == (try? Data(
                        contentsOf: VectorCatalog.conceptsDirectory.appending(
                            components: "french", "positive.jsonl"))))
            #expect(
                (try? Data(
                    contentsOf: pinned.appending(path: "emotions/calm/stories.jsonl")))
                    == storyBytes)
            #expect(
                (try? Data(
                    contentsOf: pinned.appending(component: "task-prompts-items.jsonl")))
                    == promptBytes)
            #expect(
                (try? Data(
                    contentsOf: pinned.appending(
                        component: "human-baseline-baseline.csv")))
                    == baselineBytes)

            // Auto-commit is skipped under the test root override (same
            // pattern as the cleanliness gate): no git repo appears.
            #expect(!fm.fileExists(atPath: root.appending(component: ".git").path))

            // appVersion is stamped at freeze, from the single version
            // constant, and the frozen manifest still verifies clean.
            #expect(frozen.appVersion == SteerLabVersion.current)
            #expect(frozen.appVersion?.hasPrefix("swift-app ") == true)
            #expect(ExperimentStore.verify(frozen).isEmpty)
        }
    }

    @Test func appVersionIsProvenanceNotContent() throws {
        var manifest = ExperimentManifest(
            name: "ver", description: "", modelID: "test/model")
        let plainHash = ExperimentStore.manifestHash(manifest)

        // nil appVersion is omitted from the encoding, so every existing
        // manifest keeps its bytes and content hash.
        let encoder = JSONEncoder()
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(manifest))
                as? [String: Any])
        #expect(json["appVersion"] == nil, "nil appVersion must be omitted")

        // Stamping it round-trips but never moves the content hash — it is
        // lifecycle provenance like gitCommit, not a pinned input.
        manifest.appVersion = "swift-app 9.9.9-test"
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(decoded.appVersion == "swift-app 9.9.9-test")
        #expect(ExperimentStore.manifestHash(manifest) == plainHash)
    }

    @Test func duplicateClearsAppVersionStamp() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "verdup", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "verdup")
            #expect(frozen.appVersion != nil)

            // Duplicate is a fresh draft: no freeze residue, no version stamp.
            let copy = try ExperimentStore.duplicate(name: "verdup", as: "verdup-2")
            #expect(copy.appVersion == nil)
        }
    }
}
