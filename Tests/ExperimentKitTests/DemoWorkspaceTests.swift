import Foundation
import Testing

@testable import ExperimentKit

/// Demo Workspaces on the Mac: shipping, copying, and verifying.
///
/// The product side is tested against a small synthetic placeholder under
/// `Tests/Fixtures/DemoWorkspaces/`. Every test that opens a demo also runs
/// over whatever `DemoWorkspaces/` at the top of the repository carries, so
/// real content is held to the same checks the day it lands.
///
/// Serialized, and holding `ExperimentRootOverrideLock` wherever a copy's
/// studies are checked: that check reads the process-wide workspace root.
/// Python twin: `Server/tests/test_demo_workspaces.py`.
@Suite(.serialized) struct DemoWorkspaceTests {

    // MARK: Fixtures

    static let repository = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let fixtures = repository.appending(path: "Tests/Fixtures/DemoWorkspaces")
    static let shipped = repository.appending(path: "DemoWorkspaces")

    private let fm = FileManager.default

    private func temporary(_ name: String = "demo") -> URL {
        fm.temporaryDirectory.appending(component: "steerlab-\(name)-\(UUID().uuidString)")
    }

    /// A demo root carrying the placeholder under `backend`'s name.
    private func staged(_ backend: DemoWorkspace.Backend) throws -> URL {
        let root = temporary("carried")
        let directory = root.appending(component: backend.rawValue)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.copyItem(at: Self.fixtures.appending(component: "mlx"), to: directory)
        let url = directory.appending(component: "demo.json")
        guard
            var document = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any]
        else { throw ExperimentError(reason: "the placeholder's demo.json is not an object") }
        document["backend"] = backend.rawValue
        try JSONSerialization.data(withJSONObject: document).write(to: url)
        return root
    }

    private func bytes(_ root: URL, _ names: [String]) -> [String: Data] {
        Dictionary(
            uniqueKeysWithValues: names.map {
                ($0, (try? Data(contentsOf: root.appending(path: $0))) ?? Data())
            })
    }

    /// Holds the shared override window for a body that opens a demo with
    /// its studies checked, and proves the scoped read was put back.
    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        defer {
            #expect(WorkspaceRoot.scopedReadRoot == nil, "the scoped read was left set")
            #expect(ExperimentStore.rootOverride == nil, "the root override was left set")
            ExperimentRootOverrideLock.release()
        }
        return try body()
    }

    private func gitAvailable() -> Bool {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = ["--version"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return false }
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    private func git(_ arguments: [String], in root: URL) -> String {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = ["-C", root.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: The family and its count

    /// `steerlab-cli --version` counts the families, and four documents and a
    /// test stub quote the count. A family added without them would leave a
    /// reader told to expect a number the program no longer prints.
    @Test func theFamilyCountIsSevenAndEveryQuotationAgrees() throws {
        let count = CodeResources.Family.allCases.count
        #expect(count == 7)
        #expect(CodeResources.Family.demoWorkspaces.rawValue == "DemoWorkspaces")
        #expect(try CodeResources.demoWorkspaces().lastPathComponent == "DemoWorkspaces")

        let expected = "\(count)/\(count)"
        let pattern = try NSRegularExpression(
            pattern: #"(?:Fewer than |— |reports? |\*\*)(\d+/\d+)(?: resource families resolved| resolved| means)"#)
        var quotations = 0
        for relative in [
            "README.md", "AGENTS.md", "docs/ONBOARDING.md", "docs/CLI-REFERENCE.md",
            "scripts/tests/launch-check-test.sh",
        ] {
            let text = try String(
                contentsOf: Self.repository.appending(path: relative), encoding: .utf8)
            let matches = pattern.matches(
                in: text, range: NSRange(text.startIndex..., in: text))
            #expect(!matches.isEmpty, "\(relative) no longer quotes the family count")
            for match in matches {
                let quoted = String(text[try #require(Range(match.range(at: 1), in: text))])
                #expect(quoted == expected, "\(relative) quotes \(quoted), not \(expected)")
                quotations += 1
            }
        }
        #expect(quotations >= 9, "the scan found fewer quotations than the documents hold")
    }

    /// A packaged build with no backend still resolves the family (its README
    /// ships), reports no problem, and offers no demo.
    @Test func aBuildThatCarriesNoBackendResolvesTheFamilyAndOffersNothing() throws {
        let bundle = temporary("bundle")
        defer { try? fm.removeItem(at: bundle) }
        let family = bundle.appending(component: "DemoWorkspaces")
        try fm.createDirectory(at: family, withIntermediateDirectories: true)
        try Data("# Demo Workspaces\n".utf8).write(to: family.appending(component: "README.md"))

        ExperimentRootOverrideLock.acquire()
        CodeResources.bundleOverrideForTesting = bundle
        defer {
            CodeResources.bundleOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }
        #expect(DemoWorkspace.carriedRoot()?.path == family.standardizedFileURL.path)
        #expect(DemoWorkspace.available().isEmpty)
        let row = try #require(
            CodeResources.selfCheck().rows.first { $0.family == .demoWorkspaces })
        #expect(row.problem == nil)
        #expect(row.resolvedPath == family.standardizedFileURL.path)
    }

    // MARK: Opening a copy

    @Test(arguments: DemoWorkspace.Backend.allCases)
    func everyBackendOpensAVerifiedCopyWithItsComputeBinding(
        backend: DemoWorkspace.Backend
    ) throws {
        let root = try staged(backend)
        let target = temporary("copy")
        defer {
            try? fm.removeItem(at: root)
            try? fm.removeItem(at: target)
        }
        let source = root.appending(component: backend.rawValue)
        let names = try DemoWorkspace.files(in: source)
        let opened = try locked {
            try DemoWorkspace.open(backend, at: target, in: root, verifyingStudies: true)
        }

        // Every carried byte arrived unchanged: prompts, the frozen study's
        // pins, the draft, and the completed run.
        #expect(bytes(opened.root, names) == bytes(source, names))
        #expect(opened.fileCount == names.count)
        #expect(opened.byteCount == bytes(source, names).values.reduce(0) { $0 + $1.count })
        #expect(names.contains { $0.hasPrefix("experiments/placeholder-study/pinned/") })
        #expect(names.contains { $0.hasPrefix("runs/") && $0.hasSuffix("/generations.jsonl") })
        // Each study passed the check `experiment verify` makes, in the copy.
        #expect(
            opened.studies == [
                .init(name: "placeholder-study", status: "frozen", verified: true, violations: []),
                .init(
                    name: "placeholder-study-draft", status: "draft", verified: true,
                    violations: []),
            ])
        // The copy is a complete workspace: the seed filled in what the demo
        // does not carry, and the generated files are this build's own.
        #expect(WorkspaceStore.isWorkspace(url: opened.root))
        #expect(try WorkspaceBootstrap.inspect(opened.root)["missingSeedFiles"] == .array([]))
        #expect(
            try String(
                contentsOf: opened.root.appending(component: "AGENTS.md"), encoding: .utf8)
                == AgentContract.contents())
        #expect(
            try String(
                contentsOf: opened.root.appending(component: ".gitignore"), encoding: .utf8)
                == WorkspaceBootstrapText.gitignore)
        #expect(
            try Data(contentsOf: opened.readme)
                == Data(contentsOf: source.appending(component: "README.md")))
        // The compute binding: the three named choices, in the exact bytes
        // the Python client writes for the same backend.
        let binding = try String(
            contentsOf: opened.root.appending(path: ".steerlab/workspace.json"), encoding: .utf8)
        let expected: String =
            switch backend {
            case .mlx: "{\n  \"computeSubstrate\" : \"local-mlx\"\n}"
            case .mps:
                "{\n  \"computeLocation\" : \"this-mac\",\n  \"computeSubstrate\" : \"cluster\"\n}"
            case .cuda:
                "{\n  \"computeLocation\" : \"another-machine\",\n  \"computeSubstrate\" : \"cluster\"\n}"
            }
        #expect(binding == expected)
        #expect(WorkspaceCompute.declaredChoice(root: opened.root) == backend.computeChoice)
        // The carried original was read and never written to.
        #expect(!fm.fileExists(atPath: source.appending(component: ".steerlab").path))
        #expect(!fm.fileExists(atPath: source.appending(component: "AGENTS.md").path))
        #expect(!fm.fileExists(atPath: source.appending(component: ".git").path))

        if gitAvailable() {
            #expect(
                git(["rev-list", "--count", "HEAD"], in: opened.root)
                    .trimmingCharacters(in: .whitespacesAndNewlines) == "1")
            #expect(git(["status", "--porcelain"], in: opened.root).isEmpty)
            let tracked = git(["ls-files"], in: opened.root).split(separator: "\n").map(String.init)
            #expect(tracked.contains(".steerlab/workspace.json"))
            #expect(tracked.contains("demo.json") && tracked.contains("README.md"))
            // Runs are bulk outputs and stay out of the repository, as in any workspace.
            #expect(!tracked.contains { $0.hasPrefix("runs/") })
        }
    }

    /// A frozen study is its manifest and its pinned snapshot. Both arrive
    /// byte for byte, so the copy's study has the hash it was frozen with and
    /// still records that its freeze was forced.
    @Test func aFrozenStudysPinsSurviveTheCopyByteForByte() throws {
        let target = temporary("pins")
        defer { try? fm.removeItem(at: target) }
        let source = Self.fixtures.appending(component: "mlx")
        let opened = try locked {
            try DemoWorkspace.open(.mlx, at: target, in: Self.fixtures, verifyingStudies: true)
        }
        let pinned = try DemoWorkspace.files(in: source).filter {
            $0.hasPrefix("experiments/placeholder-study/")
        }
        #expect(pinned.count >= 6, "the placeholder lost its pinned snapshot")
        for name in pinned {
            #expect(
                try Data(contentsOf: opened.root.appending(path: name))
                    == Data(contentsOf: source.appending(path: name)),
                "\(name) changed in the copy")
        }
        let manifest = try ExperimentRepository(workspaceRoot: opened.root)
            .load(name: "placeholder-study")
        #expect(manifest.status == .frozen)
        #expect(manifest.freezeForced == true)
        #expect(ExperimentStore.manifestHash(manifest) == manifest.freezeHash)
    }

    /// The results views read a run through these two repositories.
    @Test func aCompletedRunInTheCopyLoadsInTheResultsRepositories() throws {
        let target = temporary("results")
        defer { try? fm.removeItem(at: target) }
        let opened = try DemoWorkspace.open(
            .mlx, at: target, in: Self.fixtures, verifyingStudies: false)
        #expect(opened.studies == nil)

        let repository = StudyResultRepository(workspaceRoot: opened.root)
        let runs = repository.list(experimentName: "placeholder-study")
        let run = try #require(runs.first)
        #expect(runs.count == 1)
        #expect(run.kind == .run)
        #expect(run.generationCount == 5)
        #expect(run.hasReport)
        #expect(
            URL(filePath: run.path).resolvingSymlinksInPath().path.hasPrefix(
                opened.root.resolvingSymlinksInPath().path + "/runs/"),
            "the run was read from somewhere other than the copy")
        let detail = repository.detail(for: run)
        let report = try #require(detail.report)
        #expect(report.experiment == "placeholder-study")
        #expect(report.promptCount == 5)
        #expect(report.conditions.map(\.name) == ["baseline"])
        #expect(report.conditions.first?.generations == 5)
        #expect(detail.generations.count == 5)
        #expect(detail.generations.map(\.promptID).contains("free-01"))
        // The draft has no run, and says so by listing none.
        #expect(repository.list(experimentName: "placeholder-study-draft").isEmpty)
        // The analysis side finds the same run as the newest completed one.
        let analysis = StudyAnalysisRepository(workspaceRoot: opened.root, promptRoot: opened.root)
        #expect(
            analysis.newestCompletedRun(experimentName: "placeholder-study")?.lastPathComponent
                == URL(filePath: run.path).lastPathComponent)
    }

    /// The gate real content meets: every demo the repository carries opens
    /// as a byte-identical copy whose studies verify. With none carried, this
    /// passes by checking nothing.
    @Test(arguments: [DemoWorkspaceTests.fixtures, DemoWorkspaceTests.shipped])
    func everyCarriedDemoVerifiesAfterCopying(root: URL) throws {
        for entry in DemoWorkspace.available(in: root) {
            let target = temporary("carried-\(entry.backend.rawValue)")
            defer { try? fm.removeItem(at: target) }
            let names = try DemoWorkspace.files(in: entry.directory)
            let opened = try locked {
                try DemoWorkspace.open(
                    entry.backend, at: target, in: root, verifyingStudies: true)
            }
            #expect(bytes(opened.root, names) == bytes(entry.directory, names))
            #expect(opened.studies?.map(\.name) == entry.description.studies.map(\.name))
            #expect(opened.studies?.allSatisfy(\.verified) == true)
        }
        // The folder itself is part of every build.
        #expect(fm.fileExists(atPath: root.appending(component: "README.md").path))
    }

    // MARK: Refusals

    @Test func aBackendThisBuildDoesNotCarryIsRefusedPlainly() throws {
        let target = temporary("absent")
        #expect(throws: DemoWorkspace.Refusal.self) {
            try DemoWorkspace.open(.cuda, at: target, in: Self.fixtures, verifyingStudies: false)
        }
        do {
            try DemoWorkspace.open(.cuda, at: target, in: Self.fixtures, verifyingStudies: false)
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .demoNotCarried)
            #expect(refusal.carried == [.mlx])
            #expect(
                refusal.reason
                    == "This copy of SteerLab carries no Demo Workspace for cuda "
                    + "(Another machine). It carries one for mlx.")
        }
        // A build with no demo at all.
        let empty = temporary("no-demos")
        try fm.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: empty) }
        do {
            try DemoWorkspace.open(.mlx, at: target, in: empty, verifyingStudies: false)
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .demoNotCarried)
            #expect(refusal.carried.isEmpty)
            #expect(refusal.reason.hasSuffix("It carries none."))
        }
        #expect(!fm.fileExists(atPath: target.path))
    }

    @Test func aFolderThatIsNotEmptyIsRefusedAndLeftAlone() throws {
        let target = temporary("occupied")
        try fm.createDirectory(at: target, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: target) }
        try Data("keep me\n".utf8).write(to: target.appending(component: "notes.txt"))
        do {
            try DemoWorkspace.open(.mlx, at: target, in: Self.fixtures, verifyingStudies: false)
            Issue.record("a non-empty folder was copied into")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .destinationNotEmpty)
            #expect(refusal.reason.contains("already exists and is not empty"))
        }
        #expect(try fm.contentsOfDirectory(atPath: target.path) == ["notes.txt"])

        // An empty folder is fine: the app's folder panel makes one.
        let empty = temporary("empty")
        try fm.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: empty) }
        let opened = try DemoWorkspace.open(
            .mlx, at: empty, in: Self.fixtures, verifyingStudies: false)
        #expect(fm.fileExists(atPath: opened.root.appending(component: "demo.json").path))
    }

    /// A demo whose pinned input drifted fails `experiment verify` in the
    /// copy, so the command line creates nothing — no folder, and no staging
    /// tree beside where it would have been.
    @Test func aDemoWhosePinsDriftedIsRefusedAndLeavesNothing() throws {
        let root = try staged(.mlx)
        let target = temporary("drifted")
        defer {
            try? fm.removeItem(at: root)
            try? fm.removeItem(at: target)
        }
        let drifted = root.appending(path: "mlx/prompts/concepts/courtesy/positive.jsonl")
        try (String(contentsOf: drifted, encoding: .utf8)
            + "{\"text\": \"One more line the study never pinned.\"}\n")
            .write(to: drifted, atomically: true, encoding: .utf8)
        do {
            _ = try locked {
                try DemoWorkspace.open(.mlx, at: target, in: root, verifyingStudies: true)
            }
            Issue.record("a demo with a drifted pin was opened")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .demoCopyUnverified)
            #expect(refusal.reason.contains("placeholder-study"))
            #expect(refusal.reason.hasSuffix("so nothing was created."))
            let failed = try #require(refusal.studies.first { $0.name == "placeholder-study" })
            #expect(!failed.verified && !failed.violations.isEmpty)
        }
        #expect(!fm.fileExists(atPath: target.path))
        let siblings = try fm.contentsOfDirectory(
            atPath: target.deletingLastPathComponent().path)
        #expect(!siblings.contains { $0.hasPrefix(".\(target.lastPathComponent).steerlab-staging-") })
    }

    @Test func anIncompleteDemoIsNamedAsDamagedAndNotOffered() throws {
        let root = try staged(.mlx)
        defer { try? fm.removeItem(at: root) }
        try fm.removeItem(at: root.appending(path: "mlx/experiments/placeholder-study-draft"))
        #expect(DemoWorkspace.available(in: root).isEmpty)
        do {
            try DemoWorkspace.open(.mlx, at: temporary(), in: root, verifyingStudies: false)
            Issue.record("an incomplete demo was opened")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .demoDamaged)
            #expect(refusal.reason.contains("placeholder-study-draft"))
            #expect(refusal.repair.contains("Reinstall"))
        }
        // A demo.json written for another backend's folder is not offered.
        let other = try staged(.mps)
        defer { try? fm.removeItem(at: other) }
        try fm.removeItem(at: other.appending(path: "mps/demo.json"))
        try fm.copyItem(
            at: Self.fixtures.appending(path: "mlx/demo.json"),
            to: other.appending(path: "mps/demo.json"))
        do {
            _ = try DemoWorkspace.describe(other.appending(component: "mps"))
            Issue.record("a demo.json for the wrong folder was accepted")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.reason.contains("names the backend 'mlx', but its folder is 'mps'"))
        }
    }

    /// What a copy leaves out: workspace-local state, bytecode, and the two
    /// files every copy gets fresh. A symbolic link is refused outright.
    @Test func whatACopyLeavesOut() throws {
        let root = try staged(.mlx)
        let target = temporary("left-out")
        defer {
            try? fm.removeItem(at: root)
            try? fm.removeItem(at: target)
        }
        let source = root.appending(component: "mlx")
        let names = try DemoWorkspace.files(in: source)
        try fm.createDirectory(
            at: source.appending(component: ".steerlab"), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: source.appending(path: ".steerlab/workspace.json"))
        try Data("runs/\n".utf8).write(to: source.appending(component: ".gitignore"))
        try Data("stale guide".utf8).write(to: source.appending(component: "AGENTS.md"))
        try Data("stale marker".utf8).write(to: source.appending(component: "WORKSPACE.md"))
        try fm.createDirectory(
            at: source.appending(path: "prompts/__pycache__"), withIntermediateDirectories: true)
        try Data([0]).write(to: source.appending(path: "prompts/__pycache__/policy.cpython-312.pyc"))
        #expect(try DemoWorkspace.files(in: source) == names)

        let opened = try DemoWorkspace.open(.mlx, at: target, in: root, verifyingStudies: false)
        #expect(
            try String(
                contentsOf: opened.root.appending(component: "AGENTS.md"), encoding: .utf8)
                == AgentContract.contents())
        #expect(WorkspaceCompute.declaredChoice(root: opened.root) == .macQuickStart)
        #expect(!fm.fileExists(atPath: opened.root.appending(path: "prompts/__pycache__").path))

        try fm.createSymbolicLink(
            at: source.appending(path: "prompts/linked.jsonl"),
            withDestinationURL: source.appending(component: "README.md"))
        do {
            _ = try DemoWorkspace.files(in: source)
            Issue.record("a symbolic link was accepted")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.reason.contains("symbolic link"))
        }
    }

    // MARK: The command line

    /// Runs `workspace init` with the placeholder standing in for what the
    /// build carries (other families still resolve from the checkout).
    private func initialize(
        _ arguments: [String], carrying demos: URL? = DemoWorkspaceTests.fixtures
    ) async throws -> ExperimentCLIOutcome {
        let bundle = temporary("bundle")
        try fm.createDirectory(at: bundle, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: bundle) }
        if let demos {
            try fm.copyItem(at: demos, to: bundle.appending(component: "DemoWorkspaces"))
        } else {
            let family = bundle.appending(component: "DemoWorkspaces")
            try fm.createDirectory(at: family, withIntermediateDirectories: true)
            try Data("# Demo Workspaces\n".utf8).write(to: family.appending(component: "README.md"))
        }
        // The workspace this process is "in" while the verb runs: an empty
        // scratch folder, so nothing here ever reads or writes the checkout,
        // and so the test can see that the verb put the root back.
        let elsewhere = bundle.appending(component: "elsewhere")
        try fm.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        CodeResources.bundleOverrideForTesting = bundle
        ExperimentStore.rootOverride = elsewhere
        defer {
            CodeResources.bundleOverrideForTesting = nil
            #expect(WorkspaceRoot.scopedReadRoot == nil, "the scoped read was left set")
            #expect(
                ExperimentStore.rootOverride == elsewhere,
                "the verb did not put the workspace root back")
            ExperimentStore.rootOverride = nil
            ExperimentRootOverrideLock.release()
        }
        return await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "workspace", ["init"] + arguments)
    }

    @Test func theCommandLineOpensACopyAndPointsAtItsReadme() async throws {
        let target = temporary("cli")
        defer { try? fm.removeItem(at: target) }
        let outcome = try await initialize([target.path, "--demo", "mlx"])
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.state == .ready)
        #expect(outcome.envelope.changed)
        let root = target.standardizedFileURL.path
        #expect(outcome.envelope.workspace == root)
        let result = try #require(outcome.envelope.result)
        #expect(result["workspace"] == .string(root))
        #expect(result["demoReadme"] == .string(root + "/README.md"))
        #expect(result["compute"] == .object(["computeSubstrate": .string("local-mlx")]))
        guard case .object(let demo)? = result["demo"],
            case .object(let verification)? = result["verification"],
            case .array(let studies)? = verification["studies"]
        else {
            Issue.record("the envelope carries no demo or verification")
            return
        }
        #expect(demo["title"] == .string("Placeholder demo"))
        #expect(verification["identical"] == .bool(true))
        #expect(
            studies == [
                .object([
                    "name": .string("placeholder-study"), "status": .string("frozen"),
                    "verified": .bool(true), "violations": .array([]),
                ]),
                .object([
                    "name": .string("placeholder-study-draft"), "status": .string("draft"),
                    "verified": .bool(true), "violations": .array([]),
                ]),
            ])
        let next = try #require(outcome.envelope.nextAction)
        #expect(next.verb == "experiment list")
        #expect(next.detail?.contains("\(root)/README.md") == true)
        #expect(next.detail?.contains("--workspace \(root)") == true)
        #expect(fm.fileExists(atPath: root + "/README.md"))

        // The flag may come before the path.
        let second = temporary("cli-order")
        defer { try? fm.removeItem(at: second) }
        let reordered = try await initialize(["--demo", "mlx", second.path])
        #expect(reordered.exitCode == 0)
        #expect(fm.fileExists(atPath: second.appending(component: "demo.json").path))
    }

    @Test func theCommandLineRefusesPlainlyAndLeavesNothing() async throws {
        let target = temporary("cli-refused")
        defer { try? fm.removeItem(at: target) }

        // A backend this build does not carry.
        let absent = try await initialize([target.path, "--demo", "cuda"])
        #expect(absent.exitCode == 65)
        #expect(absent.envelope.state == .refused)
        #expect(absent.envelope.error?.code == "demoNotCarried")
        #expect(
            absent.envelope.error?.reason
                == "This copy of SteerLab carries no Demo Workspace for cuda "
                + "(Another machine). It carries one for mlx.")
        #expect(
            absent.envelope.error?.repairAction.hasPrefix(
                "steerlab-cli workspace init <path> --demo mlx") == true)
        #expect(
            absent.envelope.result == [
                "backend": .string("cuda"), "carried": .array([.string("mlx")]),
            ])

        // A build that carries none.
        let none = try await initialize([target.path, "--demo", "mlx"], carrying: nil)
        #expect(none.exitCode == 65)
        #expect(none.envelope.error?.code == "demoNotCarried")
        #expect(none.envelope.error?.reason.hasSuffix("It carries none.") == true)
        #expect(
            none.envelope.error?.repairAction
                == "steerlab-cli workspace init <path>  (an ordinary new workspace)")

        // A name that is not a backend: a malformed request, with the three choices.
        let unknown = try await initialize([target.path, "--demo", "gpu"])
        #expect(unknown.envelope.state == .blocked)
        #expect(unknown.envelope.exitCode == 64)
        #expect(unknown.envelope.error?.code == "usage")
        for backend in DemoWorkspace.Backend.allCases {
            #expect(unknown.envelope.error?.repairAction.contains(backend.rawValue) == true)
        }
        #expect(!fm.fileExists(atPath: target.path))

        // A folder that is not empty.
        try fm.createDirectory(at: target, withIntermediateDirectories: true)
        try Data("keep me\n".utf8).write(to: target.appending(component: "notes.txt"))
        let occupied = try await initialize([target.path, "--demo", "mlx"])
        #expect(occupied.exitCode == 65)
        #expect(occupied.envelope.error?.code == "destinationNotEmpty")
        #expect(
            occupied.envelope.error?.repairAction
                == "steerlab-cli workspace init <a-new-or-empty-path> --demo mlx")
        #expect(try fm.contentsOfDirectory(atPath: target.path) == ["notes.txt"])
    }

    /// Without `--demo`, the verb is exactly what it was — whether or not the
    /// build carries a demo.
    @Test func withoutTheFlagWorkspaceInitIsUnchanged() async throws {
        for carried in [DemoWorkspaceTests.fixtures, nil] {
            let target = temporary("plain")
            defer { try? fm.removeItem(at: target) }
            let outcome = try await initialize([target.path], carrying: carried)
            #expect(outcome.exitCode == 0)
            #expect(Set(try #require(outcome.envelope.result).keys) == ["workspace", "seededFrom"])
            #expect(outcome.envelope.nextAction?.verb == "authoring study <intent>")
            #expect(!fm.fileExists(atPath: target.appending(component: "demo.json").path))
            #expect(!fm.fileExists(atPath: target.appending(component: ".steerlab").path))
            #expect(
                try fm.contentsOfDirectory(
                    atPath: target.appending(path: "experiments").path
                ).isEmpty)
        }
        let spec = try #require(ExperimentCLIParser.spec(namespace: "workspace", verb: "init"))
        #expect(spec.valueFlags == ["--demo"])
        #expect(CLIFlagVocabulary.metavar("--demo") == "<mlx|mps|cuda>")
        #expect(!CLIFlagVocabulary.purpose("--demo").isEmpty)
    }

    // MARK: The app

    /// The app's sequence: copy with every byte checked, switch to the copy,
    /// then check its studies in place. The switch is injected so this never
    /// writes the saved workspace choice.
    @MainActor @Test func theAppOpensACopySwitchesToItAndChecksItsStudies() throws {
        let target = temporary("app")
        defer { try? fm.removeItem(at: target) }
        let store = WorkspaceStore(
            resolution: .init(source: .none, url: WorkspaceRoot.noWorkspacePlaceholder))
        ExperimentRootOverrideLock.acquire()
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
        }
        var switched: URL?
        let opening = try store.openDemoWorkspace(.mlx, at: target, in: Self.fixtures) { root in
            // Stand in for `switchTo`: the copy becomes the workspace this
            // process reads.
            #expect(WorkspaceStore.isWorkspace(url: root), "switched before the copy was whole")
            switched = root
            WorkspaceRoot.programmaticOverride = root
            ExperimentStore.rootOverride = root
        }
        #expect(switched?.path == opening.opened.root.path)
        #expect(opening.studies?.map(\.name) == ["placeholder-study", "placeholder-study-draft"])
        #expect(opening.unverified.isEmpty)
        #expect(WorkspaceCompute.declaredChoice(root: opening.opened.root) == .macQuickStart)
        // The scoped read is the command line's tool; the app never uses it.
        #expect(WorkspaceRoot.scopedReadRoot == nil)

        // Not reading the copy: nothing is checked, and nothing is claimed.
        ExperimentStore.rootOverride = fm.temporaryDirectory
        #expect(DemoWorkspace.checkStudiesInCurrentWorkspace(opening.opened) == nil)
    }

    /// A refusal reaches the app as a reason and a repair, and nothing is
    /// switched.
    @MainActor @Test func theAppDoesNotSwitchWhenTheCopyIsRefused() throws {
        let store = WorkspaceStore(
            resolution: .init(source: .none, url: WorkspaceRoot.noWorkspacePlaceholder))
        var switched = false
        do {
            _ = try store.openDemoWorkspace(.mps, at: temporary(), in: Self.fixtures) { _ in
                switched = true
            }
            Issue.record("a demo this build does not carry was opened")
        } catch let refusal as DemoWorkspace.Refusal {
            #expect(refusal.code == .demoNotCarried)
            #expect(refusal.localizedDescription == "\(refusal.reason) \(refusal.repair)")
        }
        #expect(!switched)
    }

    /// One plain line for what a demo needs and one for what it shows, and
    /// no command, flag, or path anywhere in the app's words.
    @Test func theAppsWordsArePlain() throws {
        let description = try DemoWorkspace.describe(Self.fixtures.appending(component: "mlx"))
        #expect(
            DemoWorkspace.needsLine(description)
                == "Needs the model example/placeholder-model, a download of about 0.5 GB. "
                + "Runs on this Mac with nothing else to install.")
        #expect(DemoWorkspace.showsLine(description) == description.summary)
        #expect(DemoWorkspaceCopy.rowTitle(description) == "Placeholder demo (This Mac, quick start)")

        var lines = [
            DemoWorkspaceCopy.button, DemoWorkspaceCopy.title, DemoWorkspaceCopy.introduction,
            DemoWorkspaceCopy.noneCarried, DemoWorkspaceCopy.openButton,
            DemoWorkspaceCopy.panelTitle, DemoWorkspaceCopy.panelPrompt,
            DemoWorkspaceCopy.defaultFolderName, DemoWorkspaceCopy.opened,
            DemoWorkspaceCopy.showGuide, DemoWorkspaceCopy.unverified(["one", "two"]),
            DemoWorkspaceCopy.failedTitle, DemoWorkspaceCopy.unavailableWhilePinned,
            ResearchSetupCopy.demoCaption,
        ]
        for backend in DemoWorkspace.Backend.allCases {
            let root = try staged(backend)
            defer { try? fm.removeItem(at: root) }
            let staged = try DemoWorkspace.describe(root.appending(component: backend.rawValue))
            lines.append(DemoWorkspace.needsLine(staged))
            lines.append(DemoWorkspaceCopy.rowTitle(staged))
            #expect(DemoWorkspace.needsLine(staged).contains("0.5 GB"))
        }
        let developerText = [
            "steerlab-cli", "steerlab ", "STEERLAB_", "--", "rebuild", "payload", "venv",
            "checkout", "docs/", ".md", ".json", "Python", "override", "/dev/null", "verify ",
        ]
        for line in lines {
            let found = developerText.filter { line.contains($0) }
            #expect(found.isEmpty, "\(found) in: \(line)")
            #expect(line.range(of: "agent", options: .caseInsensitive) == nil, "agent in: \(line)")
        }
        // Refusals are written for a person too; the command line adds commands.
        #expect(!DemoWorkspace.Refusal.reinstall.contains("steerlab"))
    }

    // MARK: The two clients agree

    /// Both clients copy the same placeholder to the same bytes: every carried
    /// file, every seed file, the agent guide, the ignore rules, and the
    /// compute binding. Only the marker differs (it carries a date).
    @Test(arguments: DemoWorkspace.Backend.allCases)
    func pythonOpensTheSameCopyAsTheMac(backend: DemoWorkspace.Backend) throws {
        let python = try #require(
            ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
            "Set STEERLAB_TEST_PYTHON (TEST_RUNNER_STEERLAB_TEST_PYTHON for xcodebuild) to a Python with the client's dependencies.")
        let root = try staged(backend)
        let temp = temporary("parity")
        try fm.createDirectory(at: temp, withIntermediateDirectories: true)
        defer {
            try? fm.removeItem(at: root)
            try? fm.removeItem(at: temp)
        }
        let mac = temp.appending(component: "mac")
        let other = temp.appending(component: "python")
        _ = try locked {
            try DemoWorkspace.open(backend, at: mac, in: root, verifyingStudies: true)
        }

        let script = """
            import sys
            from pathlib import Path
            from steerlab_server import client_cli
            from steerlab_server.client import demo_workspaces as demos
            demos.PACKAGED = Path(sys.argv[1]) / 'not-packaged'
            demos.CHECKOUT = Path(sys.argv[1])
            sys.exit(client_cli.main(['workspace', 'init', sys.argv[2], '--demo', sys.argv[3], '--no-git', '--json']))
            """
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/env")
        process.arguments = [python, "-c", script, root.path, other.path, backend.rawValue]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = Self.repository.appending(component: "Server").path
        environment.removeValue(forKey: "STEERLAB_WORKSPACE")
        process.environment = environment
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        try process.run()
        let document = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)

        var names = try DemoWorkspace.files(in: root.appending(component: backend.rawValue))
        names += WorkspaceStore.seedManifest + ["AGENTS.md", ".gitignore", ".steerlab/workspace.json"]
        for name in Set(names).sorted() {
            #expect(
                try Data(contentsOf: mac.appending(path: name))
                    == Data(contentsOf: other.appending(path: name)),
                "the two clients' copies differ in \(name)")
        }
        // The same next step, in the same words, apart from the flag's name.
        guard case .object(let envelope) = try JSONDecoder().decode(JSONValue.self, from: document),
            case .object(let next)? = envelope["nextAction"],
            case .object(let result)? = envelope["result"],
            case .string(let pythonRoot)? = result["workspaceRoot"]
        else {
            Issue.record("the Python client returned no document")
            return
        }
        let expected = DemoWorkspace.nextAction(rootPath: pythonRoot, workspaceFlag: "--root")
        #expect(next["verb"] == .string(expected.verb))
        #expect(next["detail"] == .string(try #require(expected.detail)))
        #expect(result["compute"] != nil && result["verification"] != nil && result["demo"] != nil)
    }
}
