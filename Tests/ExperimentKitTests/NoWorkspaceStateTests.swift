import Foundation
import Testing

@testable import ExperimentKit

/// The explicit "no workspace yet" state (release review A1).
///
/// With nothing chosen, the workspace used to resolve to the compiled source
/// path of the machine that built the binary, with no existence check — so a
/// newcomer's first launch landed on a Home that named a folder they did not
/// have, and the command line read and wrote wherever that path happened to
/// point. These tests pin the replacement: a state with its own name, a root
/// nothing can be written beneath, and a refusal with a repair.
///
/// Everything here goes through injected seams. No test sets the process
/// environment, the saved choice, or the programmatic override, so suites
/// that read the live root are never disturbed.
@Suite(.serialized) struct NoWorkspaceStateTests {

    private func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "no-ws-\(UUID().uuidString)")
    }

    // MARK: - Resolution

    @Test func nothingChosenAndNoDeveloperCheckoutIsTheNoWorkspaceState() {
        let resolution = WorkspaceRoot.resolution(
            environment: [:], programmaticOverride: nil, persistedPath: nil,
            developerCheckout: nil)
        #expect(resolution.source == .none)
        #expect(!resolution.hasWorkspace)
        #expect(resolution.url == WorkspaceRoot.noWorkspacePlaceholder)
        // A blank environment value and an empty saved path are "nothing".
        let blank = WorkspaceRoot.resolution(
            environment: [WorkspaceRoot.environmentKey: "  "],
            programmaticOverride: nil, persistedPath: "",
            developerCheckout: nil)
        #expect(blank.source == .none)
    }

    @Test func aDeveloperCheckoutKeepsTheDevelopmentFallback() {
        let checkout = URL(filePath: "/dev/some-checkout")
        let resolution = WorkspaceRoot.resolution(
            environment: [:], programmaticOverride: nil, persistedPath: nil,
            developerCheckout: checkout)
        #expect(resolution.source == .developerCheckout)
        #expect(resolution.hasWorkspace)
        #expect(resolution.url == checkout)
    }

    @Test func aSavedChoiceWinsWhileItsFolderExistsAndDegradesWhenItDoesNot() throws {
        let existing = tempDirectory()
        try FileManager.default.createDirectory(
            at: existing, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: existing) }

        let saved = WorkspaceRoot.resolution(
            environment: [:], programmaticOverride: nil,
            persistedPath: existing.path,
            developerCheckout: URL(filePath: "/dev/some-checkout"))
        #expect(saved.source == .persistedChoice)
        #expect(saved.url.path == existing.standardizedFileURL.path)

        // The folder was deleted: a developer build falls back to its
        // checkout, and a distributed build has no workspace — never a
        // dangling root, and never a source path.
        let missing = "/no/such/dir-\(UUID().uuidString)"
        #expect(
            WorkspaceRoot.resolution(
                environment: [:], programmaticOverride: nil, persistedPath: missing,
                developerCheckout: URL(filePath: "/dev/some-checkout")
            ).source == .developerCheckout)
        #expect(
            WorkspaceRoot.resolution(
                environment: [:], programmaticOverride: nil, persistedPath: missing,
                developerCheckout: nil
            ).source == .none)
    }

    @Test func explicitChoicesAreNamedAndTheCheckoutIsNotEvenLookedFor() {
        var looked = false
        func checkout() -> URL? {
            looked = true
            return URL(filePath: "/dev/some-checkout")
        }
        let environment = WorkspaceRoot.resolution(
            environment: [WorkspaceRoot.environmentKey: "/env/ws"],
            programmaticOverride: URL(filePath: "/override/ws"),
            persistedPath: nil, developerCheckout: checkout())
        #expect(environment.source == .environment)
        #expect(environment.url.path == "/env/ws")

        let override = WorkspaceRoot.resolution(
            environment: [:], programmaticOverride: URL(filePath: "/override/ws"),
            persistedPath: nil, developerCheckout: checkout())
        #expect(override.source == .programmaticOverride)
        #expect(override.url.path == "/override/ws")
        #expect(!looked, "the developer checkout is the LAST rule, evaluated lazily")
    }

    /// The older, URL-returning resolver keeps its exact behavior: it is what
    /// existing callers and `resolutionPrecedenceIsEnvOverrideDefaultsFallback`
    /// use, and the two must agree on rules 1–3.
    @Test func theTwoResolversAgreeOnEveryExplicitChoice() throws {
        let existing = tempDirectory()
        try FileManager.default.createDirectory(
            at: existing, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: existing) }
        let fallback = URL(filePath: "/dev/legacy-repo-root")
        let cases: [([String: String], URL?, String?)] = [
            ([WorkspaceRoot.environmentKey: "/env/ws"], nil, nil),
            ([:], URL(filePath: "/override/ws"), existing.path),
            ([:], nil, existing.path),
            ([:], nil, nil),
        ]
        for (environment, override, persisted) in cases {
            let url = WorkspaceRoot.resolve(
                environment: environment, programmaticOverride: override,
                persistedPath: persisted, fallback: fallback)
            let resolution = WorkspaceRoot.resolution(
                environment: environment, programmaticOverride: override,
                persistedPath: persisted, developerCheckout: fallback)
            #expect(url == resolution.url)
        }
    }

    // MARK: - The compiled-in path: real, gone, and remapped

    /// A release build may remap compiled source paths. A relative result
    /// must never be resolved against the working directory — that would
    /// make whatever folder the process started in "the checkout".
    @Test func aRemappedCompiledPathIsNoDeveloperCheckout() {
        let everythingExists: (String) -> Bool = { _ in true }
        for remapped in [
            "./Sources/ExperimentKit/CodeResources.swift",
            "Sources/ExperimentKit/CodeResources.swift",
            "CodeResources.swift",
            "",
        ] {
            #expect(CodeResources.checkoutRoot(compiledFilePath: remapped) == nil)
            #expect(
                CodeResources.developerCheckoutRoot(
                    compiledFilePath: remapped, fileExists: everythingExists) == nil,
                "'\(remapped)' resolved to a checkout because a marker exists somewhere")
        }
        // Absolute but meaningless (a prefix remapped to a fixed root): the
        // marker is not there, so it is no checkout either.
        #expect(
            CodeResources.developerCheckoutRoot(
                compiledFilePath: "/steerlab-src/Sources/ExperimentKit/CodeResources.swift",
                fileExists: { _ in false }) == nil)
    }

    @Test func aRealCompiledPathResolvesOnlyWithThePackageMarker() throws {
        let fm = FileManager.default
        let root = tempDirectory()
        let source = root.appending(path: "Sources/ExperimentKit/CodeResources.swift")
        try fm.createDirectory(
            at: source.deletingLastPathComponent(), withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        // The folder exists and has no marker: not a checkout.
        #expect(CodeResources.developerCheckoutRoot(compiledFilePath: source.path) == nil)
        try "// swift-tools-version: 6.2\n".write(
            to: root.appending(component: "Package.swift"),
            atomically: true, encoding: .utf8)
        #expect(
            CodeResources.developerCheckoutRoot(compiledFilePath: source.path)?.path
                == root.standardizedFileURL.path)
    }

    /// The regression guard for developer builds: this test process was
    /// compiled inside a checkout, so the live resolution still has a
    /// workspace and never reports the placeholder.
    @Test func thisDeveloperBuildStillResolvesAWorkspace() throws {
        let checkout = try #require(CodeResources.developerCheckoutRoot)
        #expect(
            CodeResources.compiledCheckoutPath.standardizedFileURL.path == checkout.path)
        #expect(WorkspaceRoot.hasWorkspace)
        #expect(WorkspaceRoot.current != WorkspaceRoot.noWorkspacePlaceholder)
    }

    /// A packaged app or its bundled command line asserts release mode, and
    /// then never falls back to a checkout that happens to be on the machine.
    @Test func aDistributedBuildNeverFallsBackToACheckout() throws {
        ExperimentRootOverrideLock.acquire()
        defer {
            CodeResources.releaseModeAsserted = false
            ExperimentRootOverrideLock.release()
        }
        #expect(CodeResources.workspaceFallbackCheckout != nil)
        CodeResources.releaseModeAsserted = true
        #expect(CodeResources.workspaceFallbackCheckout == nil)
    }

    // MARK: - The placeholder root

    @Test func nothingCanBeWrittenBeneathThePlaceholder() {
        let fm = FileManager.default
        let placeholder = WorkspaceRoot.noWorkspacePlaceholder
        #expect(!fm.fileExists(atPath: placeholder.path))
        #expect(!WorkspaceStore.isWorkspace(url: placeholder))
        #expect(throws: (any Error).self) {
            try fm.createDirectory(
                at: placeholder.appending(components: "experiments", "demo"),
                withIntermediateDirectories: true)
        }
        #expect(throws: (any Error).self) {
            try Data("x".utf8).write(to: placeholder.appending(component: "AGENTS.md"))
        }
        #expect(!fm.fileExists(atPath: placeholder.path))
        // The compiled-checkout placeholder has the same property, and the
        // two can never be mistaken for each other or for a real root.
        #expect(CodeResources.unresolvedCheckoutPlaceholder != placeholder)
        #expect(!fm.fileExists(atPath: CodeResources.unresolvedCheckoutPlaceholder.path))
    }

    // MARK: - The app's store

    @MainActor @Test func theStoreReportsNoWorkspaceWithoutShowingAPath() throws {
        let store = WorkspaceStore(
            resolution: .init(source: .none, url: WorkspaceRoot.noWorkspacePlaceholder))
        #expect(!store.hasWorkspace)
        #expect(store.chosenRootURL == nil)
        #expect(!store.isLegacyRepoRoot)
        #expect(store.displayName == WorkspaceStore.noWorkspaceDisplayName)
        #expect(!store.displayName.contains("/"))
        // Nothing here attempts a write: no contract upkeep, no declaration.
        #expect(store.noteAgentContractUpkeep() == nil)
        #expect(throws: ExperimentError.self) { try store.declareCompute(.localMLX) }
        #expect(
            !FileManager.default.fileExists(
                atPath: WorkspaceRoot.noWorkspacePlaceholder.path))
    }

    @MainActor @Test func aDeveloperCheckoutIsAWorkspaceButNotAChosenOne() {
        let store = WorkspaceStore(
            resolution: .init(
                source: .developerCheckout, url: VectorCatalog.bundledSeedRoot))
        #expect(store.hasWorkspace)
        #expect(store.isLegacyRepoRoot)
        // Research Setup and the handoff must not treat the source tree as
        // the researcher's data.
        #expect(store.chosenRootURL == nil)
    }

    @MainActor @Test func aChosenFolderIsOfferedToResearchSetup() throws {
        let root = tempDirectory()
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(
            resolution: .init(source: .persistedChoice, url: root))
        #expect(store.hasWorkspace)
        #expect(store.chosenRootURL == root)
        #expect(store.displayName == root.lastPathComponent)
    }

    // MARK: - The command line

    private func runner() -> ExperimentCLIRunner {
        ExperimentCLIRunner(
            sink: .discarding, now: { Date(timeIntervalSince1970: 1_000) },
            workspaceIsResolved: { false })
    }

    /// The refusal, read the way a coding assistant reads one: a state, an
    /// exit code, a stable code, and a repair it can run.
    @Test func aVerbThatNeedsAWorkspaceRefusesWithARunnableRepair() async throws {
        for (namespace, args) in [
            ("experiment", ["list"]),
            ("experiment", ["create", "demo", "--model", "test/model"]),
            ("data", ["check", "demo"]),
            ("design", ["list"]),
            ("workspace", ["inspect"]),
            ("science", ["probe-list"]),
            ("remote", ["import"]),
        ] {
            let outcome = await runner().run(namespace: namespace, args)
            let label = "\(namespace) \(args.joined(separator: " "))"
            #expect(outcome.envelope.state == .refused, "\(label): state")
            #expect(outcome.envelope.exitCode == 65, "\(label): document exit code")
            #expect(outcome.exitCode == 65, "\(label): process exit code")
            #expect(
                outcome.envelope.error?.code == ExperimentCLIRunner.noWorkspaceCode,
                "\(label): code")
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("steerlab-cli workspace init <dir>"), "\(label)")
            #expect(repair.contains("--workspace <dir>"), "\(label)")
            #expect(repair.contains(WorkspaceRoot.environmentKey), "\(label)")
            // The document names no place: not the placeholder, not a source
            // path, and no `workspace` field at all.
            #expect(outcome.envelope.workspace == nil, "\(label): workspace field")
            let document = try outcome.envelope.jsonText()
            #expect(!document.contains("/dev/null"), "\(label)")
            #expect(!document.contains(CodeResources.compiledCheckoutPath.path), "\(label)")
            // Human mode prints the same two lines.
            #expect(outcome.failure?.reason == ExperimentCLIRunner.noWorkspaceReason)
            #expect(outcome.failure?.repairAction == ExperimentCLIRunner.noWorkspaceRepair)
        }
    }

    /// The verbs that create a workspace, set this machine up, or describe
    /// the installed program must still answer — otherwise the repair above
    /// would itself be refused.
    @Test func theRepairAndTheSetupVerbsRunWithNoWorkspace() async throws {
        ExperimentRootOverrideLock.acquire()
        defer { ExperimentRootOverrideLock.release() }
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let created = await runner().run(namespace: "workspace", ["init", root.path])
        #expect(created.envelope.state.isSuccess)
        #expect(created.exitCode == 0)
        #expect(WorkspaceStore.isWorkspace(url: root))
        #expect(created.envelope.workspace == root.standardizedFileURL.path)

        let help = await runner().run(namespace: "experiment", ["list", "--help"])
        #expect(help.exitCode == 0)
        #expect(help.envelope.workspace == nil)

        let catalog = await runner().run(namespace: "science", ["list"])
        #expect(catalog.envelope.error?.code != ExperimentCLIRunner.noWorkspaceCode)

        // A sub-verb that does not exist is still a usage error, answered by
        // the dispatch with its verb list — not a workspace refusal.
        let unknown = await runner().run(namespace: "experiment", ["no-such-verb"])
        #expect(unknown.envelope.error?.code != ExperimentCLIRunner.noWorkspaceCode)
        #expect(unknown.envelope.state == .blocked)
    }

    @Test func everyDeclaredVerbIsClassifiedAndTheExemptionsAreTheIntendedOnes() {
        // The exemptions, spelled out: a new verb needs a workspace unless
        // someone decides otherwise here.
        let exempt = Set(
            ExperimentCLIParser.specs
                .filter {
                    !ExperimentCLIRunner.needsWorkspace(
                        namespace: $0.namespace, verb: $0.verb)
                }
                .map { $0.verb.isEmpty ? $0.namespace : "\($0.namespace) \($0.verb)" })
        #expect(
            exempt == [
                "init",
                "workspace init",
                "setup inspect", "setup plan", "setup apply", "setup repair", "setup start",
                "install version", "install verify", "install stamp",
                "docs cli-reference",
                "authoring prompt", "authoring study",
                "science list", "science guide",
                "model plan", "model install",
                "remote capabilities", "remote jobs", "remote logs", "remote cancel",
                "remote chat", "remote variants", "remote model-plan",
                "remote model-install", "remote model-status", "remote model-cancel",
            ])
        // And the families that hold study data are never exempt.
        for namespace in ["experiment", "data", "vectors", "design", "pack", "agent", "panel"] {
            for spec in ExperimentCLIParser.specs where spec.namespace == namespace {
                #expect(
                    ExperimentCLIRunner.needsWorkspace(
                        namespace: namespace, verb: spec.verb),
                    "\(namespace) \(spec.verb)")
            }
        }
    }

    @Test func aResolvedWorkspaceIsNeverRefusedByThisGate() async throws {
        try await withTempRootAsync { _ in
            let outcome = await ExperimentCLIRunner(sink: .discarding)
                .run(namespace: "experiment", ["list"])
            #expect(outcome.envelope.error?.code != ExperimentCLIRunner.noWorkspaceCode)
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.workspace != nil)
        }
    }

    private func withTempRootAsync(_ body: (URL) async throws -> Void) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = tempDirectory()
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        try await body(root)
    }
}

/// The words a newcomer reads before a workspace exists, and in Research
/// Setup (release review A1, A2). They are data in ExperimentKit so that this
/// suite can hold them to the rule: plain words, no command line, no path,
/// and "coding assistant" for the researcher's AI tool.
struct FirstLaunchWordingTests {

    /// Text that belongs to the command line or to a developer.
    private static let developerText = [
        "steerlab-cli", "steerlab ", "STEERLAB_", "--", "rebuild", "payload",
        "interpreter", "venv", "checkout", "docs/", ".md", "Python", "CPU",
        "override", "source mismatch", "/dev/null",
    ]

    private func offences(in text: String) -> [String] {
        var found = Self.developerText.filter { text.contains($0) }
        // "agent" means a configured model under study. The researcher's AI
        // tool is a "coding assistant".
        if text.range(of: "agent", options: .caseInsensitive) != nil {
            found.append("agent")
        }
        return found
    }

    private static let researchSetupCopy: [String] = [
        ResearchSetupCopy.title, ResearchSetupCopy.introduction,
        ResearchSetupCopy.workspaceStepTitle, ResearchSetupCopy.workspacePrompt,
        ResearchSetupCopy.workspaceCaption, ResearchSetupCopy.demoCaption,
        ResearchSetupCopy.computeStepTitle,
        ResearchSetupCopy.computeNeedsWorkspace, ResearchSetupCopy.computeCaption,
        ResearchSetupCopy.helperStepTitle,
        ResearchSetupCopy.helperExplanation, ResearchSetupCopy.helperReady,
        ResearchSetupCopy.helperUpdateTitle, ResearchSetupCopy.helperSetupTitle,
        ResearchSetupCopy.helperNeeded, ResearchSetupCopy.helperUpdateNeeded,
        ResearchSetupCopy.planCaption, ResearchSetupCopy.installing,
        ResearchSetupCopy.cancelInstallButton, ResearchSetupCopy.cancellingInstall,
        ResearchSetupCopy.installCancelled,
        ResearchSetupCopy.installed, ResearchSetupCopy.installedButNotReady,
        ResearchSetupCopy.beginStepTitle, ResearchSetupCopy.beginExplanation,
        ResearchSetupCopy.copyInstructions, ResearchSetupCopy.instructionsCopied,
        ResearchSetupCopy.workInTheApp, ResearchSetupCopy.readyFooter,
        ResearchSetupCopy.returnFooter, ResearchSetupCopy.installerMissing,
        ResearchSetupCopy.reinstallRepair, ResearchSetupCopy.helperRepair,
    ]

    private static let firstLaunchCopy: [String] = [
        FirstLaunchCopy.welcomeTitle, FirstLaunchCopy.welcomeBody,
        FirstLaunchCopy.sectionPrompt(section: "Studies"),
        FirstLaunchCopy.createButton, FirstLaunchCopy.openButton,
        FirstLaunchCopy.researchSetupButton, FirstLaunchCopy.afterwards,
        FirstLaunchCopy.viewerPlaceholder, FirstLaunchCopy.menuHelp,
        WorkspaceStore.noWorkspaceDisplayName, WorkspaceRoot.noWorkspaceReason,
    ]

    @Test func researchSetupSaysNothingInCommandLineOrDeveloperTerms() {
        for text in Self.researchSetupCopy {
            #expect(offences(in: text).isEmpty, "\(offences(in: text)) in: \(text)")
        }
        // The coding assistant is named where the old text said "agent".
        #expect(ResearchSetupCopy.introduction.contains("coding assistant"))
        #expect(ResearchSetupCopy.beginExplanation.contains("coding assistant"))
        #expect(ResearchSetupCopy.workspaceCaption.contains("coding assistant"))
    }

    @Test func theWelcomeNeverShowsAPathOrACommand() {
        for text in Self.firstLaunchCopy {
            #expect(offences(in: text).isEmpty, "\(offences(in: text)) in: \(text)")
            #expect(!text.contains("/"), "a path-shaped fragment in: \(text)")
        }
    }

    /// The sheet itself renders `ResearchSetupCopy` and adds only button
    /// titles. Its source is checked too, so a sentence typed straight into
    /// the view cannot bring the developer text back.
    @Test func theSheetSourceCarriesNoDeveloperTextOfItsOwn() throws {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources/SteerLabApp/ResearchSetupSheet.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        // String literals only: comments and identifiers are for developers.
        var literals: [String] = []
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.hasPrefix("//") else { continue }
            let parts = line.split(separator: "\"", omittingEmptySubsequences: false)
            for index in stride(from: 1, to: parts.count, by: 2) {
                literals.append(String(parts[index]))
            }
        }
        #expect(literals.count >= 6, "the literal scan found almost nothing")
        for literal in literals {
            #expect(
                offences(in: literal).isEmpty,
                "\(offences(in: literal)) in a ResearchSetupSheet literal: \(literal)")
        }
        // The readiness report's command-line repair is no longer rendered.
        #expect(!source.contains("readiness[\"repairAction\"]"))
        #expect(!source.contains("readiness[\"reason\"]"))
    }

    private func appSource(_ name: String) throws -> String {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources/SteerLabApp/\(name)")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Stale text on the basic path (release review A5). The app target has
    /// no unit tests of its own, so the three corrections are held here by
    /// reading the sources.
    @Test func staleTextOnTheBasicPathStaysGone() throws {
        // Playground, no vectors yet: it named a command against a file that
        // no workspace contains. It now points at the place in the app.
        let chat = try appSource("ChatView.swift")
        #expect(!chat.contains("toy-french.json"))
        #expect(!chat.contains("steerlab-cli --config"))
        #expect(chat.contains("StudyControlCopy.playgroundNoVectors"))

        let copy = try appSource("StudyControlCopy.swift")
        #expect(copy.contains("Concepts & Vectors"))
        // Run help said sampling needs temperature 0. Local runs sample with
        // a seeded stream per record.
        #expect(!copy.contains("requires Temperature = 0"))
        #expect(!copy.contains("mlx-swift-lm does not"))
        // Freeze help cited a command-line flag.
        #expect(!copy.contains("freeze --force"))

        let temperature = try appSource("TemperatureRow.swift")
        #expect(!temperature.contains("currently require 0"))
    }

    /// The command line keeps its own hint, unchanged: it is the right text
    /// for a terminal and the wrong text for the sheet.
    @Test func theCommandLineHintIsKeptForTheCommandLine() {
        #expect(ScientificPythonRuntime.setupHint.contains("steerlab-cli setup plan"))
        #expect(ClientSetup.installerMissingRepair.contains("STEERLAB_CLIENT_RELEASE"))
    }

    @Test func aPlanOrInstallFailureKeepsItsRepair() {
        // A repair written by the installer travels as it is.
        let installer = ResearchSetupCopy.failure(
            ExperimentError.malformed(
                "The download did not finish.",
                repair: "Check the internet connection, then review a fresh setup plan."))
        #expect(installer.reason == "The download did not finish.")
        #expect(
            installer.repair
                == "Check the internet connection, then review a fresh setup plan.")

        // The command line's hint becomes one step in this sheet.
        let hint = ResearchSetupCopy.failure(
            ExperimentError.malformed(
                "The local helper returned no result.",
                repair: ScientificPythonRuntime.setupHint))
        #expect(hint.repair == ResearchSetupCopy.helperRepair)

        // A missing installer names the app, not a build script.
        let missing = ResearchSetupCopy.failure(
            ExperimentError.malformed(
                ClientSetup.installerMissingReason,
                repair: ClientSetup.installerMissingRepair))
        #expect(missing.reason == ResearchSetupCopy.installerMissing)
        #expect(missing.repair == ResearchSetupCopy.reinstallRepair)

        // An error with no repair still shows its reason.
        let bare = ResearchSetupCopy.failure(ExperimentError(reason: "It stopped."))
        #expect(bare.reason == "It stopped.")
        #expect(bare.repair == nil)
    }

    @Test func researchSetupOpensAtEveryLaunchUntilAWorkspaceExists() {
        // No workspace: always, whatever the "already shown" flag says.
        for presented in [false, true] {
            for ready in [false, true] {
                #expect(
                    ResearchSetupModel.opensAtLaunch(
                        hasWorkspace: false, alreadyPresented: presented,
                        authoringReady: ready))
            }
        }
        // A workspace exists: once, and only while study design is not ready.
        #expect(
            ResearchSetupModel.opensAtLaunch(
                hasWorkspace: true, alreadyPresented: false, authoringReady: false))
        #expect(
            !ResearchSetupModel.opensAtLaunch(
                hasWorkspace: true, alreadyPresented: true, authoringReady: false))
        #expect(
            !ResearchSetupModel.opensAtLaunch(
                hasWorkspace: true, alreadyPresented: false, authoringReady: true))
    }

    @MainActor @Test func helperGuidanceIsPlainAndGoesAwayWhenReady() {
        let model = ResearchSetupModel()
        // Nothing inspected yet reads as "not installed", in plain words.
        #expect(model.helperGuidance == ResearchSetupCopy.helperNeeded)
        #expect(model.errorRepair == nil)
    }
}
