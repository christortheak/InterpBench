import Foundation
import Testing

@testable import ExperimentKit

// One-link agent setup (2026-10-21): a coding assistant installs the app from
// docs/AGENT-SETUP.md, links its command line onto PATH, finds it again in a
// later session through the workspace, and opens the app on the workspace.

private func temporaryDirectory(_ label: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appending(component: "agent-setup-\(label)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Suite struct CommandLineLinkTests {

    let app = URL(filePath: "/Volumes/Lab/SteerLab/SteerLab.app")

    private var helper: String { app.appending(path: CommandLineLink.helperPath).path }

    @Test func nothingThereIsCreatedAndTheSameLinkIsLeftAlone() {
        #expect(CommandLineLink.decide(app: app, existing: .absent) == .create)
        #expect(
            CommandLineLink.decide(app: app, existing: .symlink(target: helper))
                == .alreadyCurrent)
    }

    /// Another copy's link is reported, and replaced only on request: a
    /// researcher may run two copies, and the link is theirs to point.
    @Test func aLinkToAnotherCopyIsAReportedRepoint() {
        let other = "/Applications/SteerLab.app/Contents/Helpers/steerlab-cli"
        #expect(
            CommandLineLink.decide(app: app, existing: .symlink(target: other))
                == .repoint(from: other))
    }

    @Test func someoneElsesFileOrLinkIsNeverReplaced() {
        guard case .foreign(let reason) = CommandLineLink.decide(app: app, existing: .otherFile)
        else { Issue.record("a regular file must be foreign"); return }
        #expect(reason.contains(helper))
        for target in ["/opt/homebrew/bin/steerlab-cli", "/usr/local/bin/something",
                       "/Volumes/Lab/SteerLab/Server/.venv/bin/steerlab-cli"] {
            guard case .foreign = CommandLineLink.decide(
                app: app, existing: .symlink(target: target))
            else { Issue.record("\(target) must be foreign"); continue }
        }
    }

    /// A link to a copy whose path will not last would break: the reasons are
    /// the updater's own.
    @Test func aCopyWhosePathWillNotLastIsRefused() {
        let translocated = URL(
            filePath: "/private/var/folders/x/AppTranslocation/ABC/d/SteerLab.app")
        let kept = URL(filePath: "/Volumes/Lab/SteerLab/SteerLab-0.9.12-previous.app")
        for (copy, words) in [(translocated, "temporary"), (kept, "kept copy")] {
            guard case .badLocation(let reason) = CommandLineLink.decide(
                app: copy, existing: .absent)
            else { Issue.record("\(copy.path) must be refused"); continue }
            #expect(reason.contains(words), "\(reason)")
        }
        guard case .badLocation(let reason) = CommandLineLink.decide(
            app: nil, existing: .absent)
        else { Issue.record("a build folder must be refused"); return }
        #expect(reason.contains("build folder"))
    }

    @Test func onlyAHelperInsideAnAppCountsAsSteerLabs() {
        #expect(CommandLineLink.isSteerLabHelper(helper))
        #expect(!CommandLineLink.isSteerLabHelper("/usr/local/bin/steerlab-cli"))
        #expect(!CommandLineLink.isSteerLabHelper(
            "/Volumes/Lab/SteerLab.app/Contents/MacOS/steerlab-cli"))
        #expect(!CommandLineLink.isSteerLabHelper(
            "/Volumes/Lab/build/Contents/Helpers/steerlab-cli"))
    }

    @Test func pathAdviceAppearsOnlyWhenTheFolderIsMissingFromPath() {
        let directory = URL(filePath: "/Volumes/Lab/.local/bin")
        #expect(
            CommandLineLink.pathAdvice(
                directory: directory, path: "/usr/bin:/Volumes/Lab/.local/bin/") == nil)
        let advice = CommandLineLink.pathAdvice(directory: directory, path: "/usr/bin:/bin")
        #expect(advice?.hasPrefix("export PATH=\"") == true)
        #expect(advice?.hasSuffix(":$PATH\"") == true)
    }

    @Test func installMakesReplacesOnRequestAndNeverTouchesSteerlab() throws {
        let directory = try temporaryDirectory("link")
        defer { try? FileManager.default.removeItem(at: directory) }
        // A file named for the OTHER product sits beside it and is untouched.
        let pythonClient = directory.appending(component: "steerlab")
        try Data("#!/bin/sh\n".utf8).write(to: pythonClient)

        var outcome = try CommandLineLink.install(
            app: app, directory: directory, replace: false, path: directory.path)
        #expect(outcome.decision == .create && outcome.changed)
        #expect(outcome.pathAdvice == nil)
        let link = directory.appending(component: "steerlab-cli")
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: link.path) == helper)

        outcome = try CommandLineLink.install(
            app: app, directory: directory, replace: false, path: "/usr/bin")
        #expect(outcome.decision == .alreadyCurrent && !outcome.changed)
        #expect(CommandLineLink.sentence(outcome).contains("not on your PATH"))

        let other = URL(filePath: "/Applications/SteerLab.app")
        outcome = try CommandLineLink.install(
            app: other, directory: directory, replace: false, path: directory.path)
        #expect(outcome.decision == .repoint(from: helper) && !outcome.changed)
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: link.path) == helper)
        outcome = try CommandLineLink.install(
            app: other, directory: directory, replace: true, path: directory.path)
        #expect(outcome.changed)
        #expect(
            try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
                == other.appending(path: CommandLineLink.helperPath).path)

        #expect(try String(contentsOf: pythonClient, encoding: .utf8) == "#!/bin/sh\n")
        #expect(
            (try? FileManager.default.destinationOfSymbolicLink(atPath: pythonClient.path))
                == nil)
    }

    @Test func aForeignFileStaysExactlyAsItWas() throws {
        let directory = try temporaryDirectory("foreign")
        defer { try? FileManager.default.removeItem(at: directory) }
        let link = directory.appending(component: "steerlab-cli")
        try Data("mine".utf8).write(to: link)
        let outcome = try CommandLineLink.install(
            app: app, directory: directory, replace: true, path: directory.path)
        guard case .foreign = outcome.decision else {
            Issue.record("expected foreign, got \(outcome.decision)"); return
        }
        #expect(!outcome.changed)
        #expect(try String(contentsOf: link, encoding: .utf8) == "mine")
    }

    /// The test process is not inside SteerLab.app, so the verb refuses as a
    /// build folder would, typed, with a repair.
    @Test func theVerbRefusesOutsideAnApp() async throws {
        let directory = try temporaryDirectory("verb")
        defer { try? FileManager.default.removeItem(at: directory) }
        let result = await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "install", ["link", "--directory", directory.path])
        #expect(result.envelope.exitCode == 65)
        #expect(result.envelope.error?.code == "missingPrerequisite")
        #expect(result.envelope.error?.reason.contains("build folder") == true)
        #expect(
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) == [])
    }
}

@Suite struct ExternalWorkspaceOpenTests {

    let workspace = URL(filePath: "/Volumes/Lab/SteerLab/Workspaces/first-study")
    let other = URL(filePath: "/Volumes/Lab/SteerLab/Workspaces/other")

    private func folders(_ set: Set<String>) -> (URL) -> Bool {
        { set.contains($0.standardizedFileURL.path) }
    }

    @Test func theFirstFolderOpens() {
        let file = URL(filePath: "/Volumes/Lab/notes.txt")
        #expect(
            ExternalWorkspaceOpen.decide(
                urls: [file, workspace, other], current: other, pinned: false,
                isDirectory: folders([workspace.path, other.path]))
                == .open(workspace))
    }

    @Test func theOpenWorkspaceIsLeftAsItIs() {
        #expect(
            ExternalWorkspaceOpen.decide(
                urls: [URL(filePath: workspace.path + "/")], current: workspace,
                pinned: false, isDirectory: folders([workspace.path])) == .alreadyOpen)
    }

    @Test func aPinnedProcessSaysWhyAndDoesNotSwitch() {
        guard case .pinned(let reason) = ExternalWorkspaceOpen.decide(
            urls: [workspace], current: other, pinned: true,
            isDirectory: folders([workspace.path]))
        else { Issue.record("a pinned process must not switch"); return }
        #expect(reason.contains("STEERLAB_WORKSPACE"))
    }

    @Test func filesAloneAreNotAWorkspace() {
        guard case .noFolder(let reason) = ExternalWorkspaceOpen.decide(
            urls: [URL(filePath: "/tmp/a.json")], current: nil, pinned: false,
            isDirectory: folders([]))
        else { Issue.record("a file is not a workspace"); return }
        #expect(reason.contains("open -a SteerLab"))
    }
}

@Suite struct MacCommandLineRecordTests {

    let app = URL(filePath: "/Volumes/Lab/SteerLab/SteerLab.app")

    @Test func theRecordNamesTheCommandLineAndStaysOutOfGit() throws {
        let root = try temporaryDirectory("record")
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(MacCommandLineRecord.note(workspaceRoot: root, app: app, appVersion: "9.9.9"))
        let record = try #require(MacCommandLineRecord.read(workspaceRoot: root))
        #expect(record.commandLine == app.appending(path: CommandLineLink.helperPath).path)
        #expect(record.app == app.path && record.appVersion == "9.9.9")
        let ignore = root.appending(path: ".steerlab/mac-app/.gitignore")
        #expect(try String(contentsOf: ignore, encoding: .utf8) == "*\n")
        #expect(MacCommandLineRecord.relativePath == ".steerlab/mac-app/command-line.json")
        // Unchanged → no write; a moved app → rewritten.
        #expect(!MacCommandLineRecord.note(workspaceRoot: root, app: app, appVersion: "9.9.9"))
        let moved = URL(filePath: "/Applications/SteerLab.app")
        #expect(MacCommandLineRecord.note(workspaceRoot: root, app: moved, appVersion: "9.9.9"))
        #expect(MacCommandLineRecord.read(workspaceRoot: root)?.app == moved.path)
    }

    @Test func nothingIsRecordedFromAPlaceThatWillNotLast() throws {
        let root = try temporaryDirectory("record-bad")
        defer { try? FileManager.default.removeItem(at: root) }
        for copy in [
            nil,
            URL(filePath: "/private/var/folders/x/AppTranslocation/ABC/d/SteerLab.app"),
            URL(filePath: "/Volumes/Lab/SteerLab/SteerLab-0.9.12-previous.app"),
        ] as [URL?] {
            #expect(!MacCommandLineRecord.note(workspaceRoot: root, app: copy))
        }
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: ".steerlab").path))
    }

    /// The handoff carries the command that opens the app on this workspace,
    /// and records the command line for the next session, only from an app.
    @Test func theHandoffOffersTheAppAndRecordsTheCommandLine() throws {
        let root = try temporaryDirectory("handoff")
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try WorkspaceStore.create(at: root)
        let bare = try WorkspaceBootstrap.handoff(root, executable: "steerlab-cli", app: nil)
        #expect(bare["openInApp"] == nil)
        #expect(MacCommandLineRecord.read(workspaceRoot: root) == nil)

        let handed = try WorkspaceBootstrap.handoff(root, executable: "steerlab-cli", app: app)
        #expect(
            handed["openInApp"]
                == .array(["open", "-a", app.path, root.standardizedFileURL.path]
                    .map(JSONValue.string)))
        #expect(MacCommandLineRecord.read(workspaceRoot: root)?.app == app.path)
    }
}
