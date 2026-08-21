import Foundation
import Testing

@testable import ExperimentKit

/// `steerlab-cli init` and the `HomeLayout` seam behind it
/// (GENERAL-DISTRIBUTION-WORK-PLAN decision 8, work item (a) — WP1's last open
/// item).
///
/// **Every test here drives a TEMP home.** `~/SteerLab` exists on the machines
/// this suite runs on and is in use; a test that materialized into it would be
/// writing into the researcher's live layout, and a test that asserted on its
/// contents would be reading one. The default home is exercised through
/// `HomeLayout.defaultHome`'s SHAPE alone, never by creating it.
@Suite struct HomeLayoutTests {

    // MARK: Harness

    private func tempHome() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "steerlab-home-\(UUID().uuidString)")
            .standardizedFileURL
    }

    @discardableResult
    private func runInit(
        _ args: [String], recorder: ExperimentCLIRecorder = ExperimentCLIRecorder()
    ) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: recorder.sink).run(namespace: "init", args)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    /// A directory that passes the content test: `Package.swift` at the root
    /// plus the Python engine's package directory.
    private func makeCheckout(at url: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: url.appending(path: "Server/steerlab_server"),
            withIntermediateDirectories: true)
        try Data("// swift-tools-version: 6.2\n".utf8)
            .write(to: url.appending(path: "Package.swift"))
    }

    // MARK: The plan step is pure

    @Test func planOnAnAbsentHomeCreatesNothingAndReportsBothDirectories() throws {
        let home = tempHome()
        let plan = try HomeLayout.plan(home: home)

        #expect(!plan.homeExisted)
        #expect(!plan.isComplete)
        #expect(plan.entries.map(\.name) == ["Workspaces", "Sites"])
        #expect(plan.missing.count == 2)
        #expect(plan.checkouts.isEmpty)
        // Pure: planning is what the app's first-run sheet shows BEFORE the
        // user has agreed to anything, so it must not have made the folder.
        #expect(!FileManager.default.fileExists(atPath: home.path))
    }

    // MARK: Materialization, and running it twice

    @Test func initCreatesBothDirectoriesAndIsIdempotent() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }

        let recorder = ExperimentCLIRecorder()
        let first = await runInit(["--home", home.path], recorder: recorder)
        #expect(first.exitCode == 0)
        #expect(first.verb == "init")
        #expect(first.envelope.changed == true)
        #expect(isDirectory(home.appending(component: "Workspaces")))
        #expect(isDirectory(home.appending(component: "Sites")))
        #expect(recorder.standardOutput.contains("created"))

        // Second run: everything already exists, nothing changed, still 0.
        let secondRecorder = ExperimentCLIRecorder()
        let second = await runInit(["--home", home.path], recorder: secondRecorder)
        #expect(second.exitCode == 0)
        #expect(second.envelope.changed == false)
        #expect(second.envelope.message.contains("already complete"))
        #expect(!secondRecorder.standardOutput.contains("created"))
        #expect(secondRecorder.standardOutput.contains("existing"))
    }

    @Test func aPartialLayoutCreatesOnlyWhatIsMissing() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        let workspaces = home.appending(component: "Workspaces")
        try FileManager.default.createDirectory(
            at: workspaces.appending(component: "existing-study"),
            withIntermediateDirectories: true)

        let outcome = await runInit(["--home", home.path])
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.changed == true)
        // The pre-existing workspace folder is untouched — never destructive.
        #expect(isDirectory(workspaces.appending(component: "existing-study")))
        #expect(isDirectory(home.appending(component: "Sites")))

        guard case .array(let entries)? = outcome.envelope.result?["entries"] else {
            Issue.record("no entries in the payload")
            return
        }
        #expect(entries.count == 2)
        #expect(entries[0] == .object([
            "name": .string("Workspaces"), "path": .string(workspaces.path),
            "created": .bool(false),
        ]))
        guard case .object(let sites) = entries[1] else {
            Issue.record("Sites entry is not an object")
            return
        }
        #expect(sites["created"] == .bool(true))
    }

    /// `Sites/` is typically a clone target, and `git clone <repo> Sites`
    /// refuses a non-empty directory — so nothing may ever be written into it.
    @Test func sitesIsCreatedEmpty() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }

        await runInit(["--home", home.path])
        let contents = try FileManager.default.contentsOfDirectory(
            atPath: home.appending(component: "Sites").path)
        #expect(contents.isEmpty, "init wrote \(contents) into Sites/")
        // …and it does not seed a workspace either: the home holds exactly the
        // two layout directories.
        let homeContents = try FileManager.default
            .contentsOfDirectory(atPath: home.path).sorted()
        #expect(homeContents == ["Sites", "Workspaces"])
    }

    // MARK: Refusals

    @Test func aFileWhereTheHomeGoesIsRefused() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        try FileManager.default.createDirectory(
            at: home.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not a directory".utf8).write(to: home)

        let outcome = await runInit(["--home", home.path])
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.state == .blocked)
        #expect(outcome.envelope.error?.code == "usage")
        #expect(outcome.envelope.error?.reason.contains("is a file") == true)
    }

    @Test func aFileWhereALayoutDirectoryGoesIsRefused() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        try FileManager.default.createDirectory(
            at: home, withIntermediateDirectories: true)
        try Data("occupied".utf8).write(to: home.appending(component: "Sites"))

        let outcome = await runInit(["--home", home.path])
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.state == .blocked)
        #expect(outcome.envelope.error?.repairAction.contains("Sites") == true)
        // The refusal came BEFORE any creation: a half-materialized layout is
        // never observable.
        #expect(!FileManager.default.fileExists(
            atPath: home.appending(component: "Workspaces").path))
    }

    @Test func anUndeclaredFlagIsAUsageErrorAndCreatesNothing() async throws {
        let home = tempHome()
        let outcome = await runInit(["--home", home.path, "--seed"])
        #expect(outcome.exitCode == 64)
        #expect(outcome.envelope.error?.code == ExperimentCLIUsageError.code)
        #expect(!FileManager.default.fileExists(atPath: home.path))
    }

    @Test func aStrayPositionalIsAUsageErrorAndCreatesNothing() async throws {
        let home = tempHome()
        let outcome = await runInit([home.path])
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.error?.reason.contains("usage: init") == true)
        #expect(!FileManager.default.fileExists(atPath: home.path))
    }

    @Test func homeWithNoValueIsAUsageError() async throws {
        let outcome = await runInit(["--home"])
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.error?.reason.contains("usage: init") == true)
    }

    // MARK: `--help` runs nothing

    @Test func helpPrintsTheSurfaceAndMaterializesNothing() async throws {
        let home = tempHome()
        let recorder = ExperimentCLIRecorder()
        let outcome = await runInit(
            ["--home", home.path, "--help"], recorder: recorder)
        #expect(outcome.exitCode == 0)
        #expect(recorder.standardOutput.hasPrefix("usage: steerlab-cli init"))
        #expect(recorder.standardOutput.contains("--home <dir>"))
        #expect(!FileManager.default.fileExists(atPath: home.path))
    }

    // MARK: Checkout detection — by content, never by name

    @Test func aCheckoutIsDetectedWhateverTheFolderIsCalled() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        // A name nothing in the codebase knows about.
        let checkout = home.appending(component: "zzz-some-clone-name")
        try makeCheckout(at: checkout)
        // …and a decoy with only half the markers.
        let decoy = home.appending(component: "not-a-checkout")
        try FileManager.default.createDirectory(
            at: decoy, withIntermediateDirectories: true)
        try Data("// swift-tools-version: 6.2\n".utf8)
            .write(to: decoy.appending(path: "Package.swift"))

        #expect(HomeLayout.isCheckout(checkout))
        #expect(!HomeLayout.isCheckout(decoy))

        let recorder = ExperimentCLIRecorder()
        let outcome = await runInit(["--home", home.path], recorder: recorder)
        #expect(outcome.exitCode == 0)
        // Compared by suffix: the temp root is reached through a symlink on
        // macOS (`/var` → `/private/var`), so the reported absolute path is
        // the resolved one and an equality check would be testing Foundation's
        // symlink handling rather than the detection rule.
        guard case .array(let found)? = outcome.envelope.result?["checkouts"],
            case .string(let path)? = found.first
        else {
            Issue.record("no checkout reported: \(String(describing: outcome.envelope.result))")
            return
        }
        #expect(found.count == 1, "the decoy was reported as a checkout")
        #expect(path.hasSuffix("/zzz-some-clone-name"))
        // The name is printed in full — a column width must never truncate a
        // directory name into one that is not on disk.
        #expect(recorder.standardOutput.contains("zzz-some-clone-name/"))
    }

    @Test func aHomeWithNoCheckoutNamesTheRunningBinarysInstead() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        let recorder = ExperimentCLIRecorder()
        let outcome = await runInit(["--home", home.path], recorder: recorder)

        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.result?["checkouts"] == .array([]))
        // Informational, never an error: an app-only user legitimately has no
        // checkout at all, and this suite runs from one that is not in the
        // temp home.
        if let external = HomeLayout.externalRunningCheckout(relativeTo: home) {
            #expect(
                outcome.envelope.result?["runningBinaryCheckout"]
                    == .string(external.path))
            #expect(recorder.standardOutput.contains("no code checkout in this home"))
        }
    }

    // MARK: Resolution-chain neutrality

    /// `init` materializes directories. It must not touch how a workspace root
    /// is resolved — no UserDefaults write, no override, no workspace.
    @Test func initLeavesTheWorkspaceResolutionChainAlone() async throws {
        let home = tempHome()
        defer { try? FileManager.default.removeItem(at: home) }

        let persistedBefore = UserDefaults.standard.string(
            forKey: WorkspaceRoot.defaultsKey)
        let overrideBefore = WorkspaceRoot.programmaticOverride

        await runInit(["--home", home.path])

        #expect(
            UserDefaults.standard.string(forKey: WorkspaceRoot.defaultsKey)
                == persistedBefore)
        #expect(WorkspaceRoot.programmaticOverride == overrideBefore)
        // The created Workspaces/ folder is not itself a workspace, and init
        // did not make one inside it.
        let workspaces = home.appending(component: "Workspaces")
        #expect(!WorkspaceStore.isWorkspace(url: workspaces))
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: workspaces.path)
                .isEmpty)
    }

    // MARK: The default home

    @Test func theDefaultHomeIsSteerLabInTheUsersHomeDirectory() {
        // Shape only — never created here.
        #expect(HomeLayout.defaultHome.lastPathComponent == "SteerLab")
        #expect(
            HomeLayout.defaultHome.deletingLastPathComponent().path
                == URL(filePath: NSHomeDirectory()).standardizedFileURL.path)
    }

    // MARK: The declared surface

    @Test func initIsDeclaredAsTheOneBareVerb() {
        let spec = ExperimentCLIParser.bareSpec(namespace: "init")
        #expect(spec != nil)
        #expect(spec?.label == "init")
        #expect(spec?.valueFlags == ["--home"])
        // Exactly one bare-verb family exists; a second would need the
        // top-level page and the reference regions revisited.
        #expect(ExperimentCLIParser.specs.filter { $0.verb.isEmpty }.count == 1)
        #expect(ExperimentCLIHelp.topLevelText.contains("init [--home <dir>]"))
    }
}
