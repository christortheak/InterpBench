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

    /// A relative link target is resolved where the system resolves it: in
    /// the link's own folder, never in this process's working directory.
    @Test func aRelativeLinkToThisAppIsCurrent() throws {
        // The app sits beside the link's folder: `../Lab/SteerLab.app/…`
        // names it only from that folder, never from the process's own.
        let root = try temporaryDirectory("relative")
        defer { try? FileManager.default.removeItem(at: root) }
        let local = root.appending(path: "Lab/SteerLab.app")
        let directory = root.appending(component: "bin")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            atPath: directory.appending(component: "steerlab-cli").path,
            withDestinationPath: "../Lab/SteerLab.app/" + CommandLineLink.helperPath)
        let outcome = try CommandLineLink.install(
            app: local, directory: directory, replace: false, path: directory.path)
        #expect(outcome.decision == .alreadyCurrent)
    }

    /// A `~/.local/bin` that is itself a symlink (a dotfiles setup): `..` in a
    /// relative target is where the kernel resolves it, not where the path's
    /// letters suggest.
    @Test func aLinkInASymlinkedFolderIsJudgedWhereTheSystemResolvesIt() throws {
        // dotfiles/bin is the real folder; home/.local/bin is a link to it.
        // From the link's real folder, `../Lab` is dotfiles/Lab; read by its
        // letters from home/.local/bin, it would be home/.local/Lab.
        let root = try temporaryDirectory("dotfiles")
        defer { try? FileManager.default.removeItem(at: root) }
        let dotfiles = root.appending(component: "dotfiles")
        try FileManager.default.createDirectory(
            at: dotfiles.appending(component: "bin"), withIntermediateDirectories: true)
        let local = root.appending(path: "home/.local")
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: local.appending(component: "bin"),
            withDestinationURL: dotfiles.appending(component: "bin"))
        try FileManager.default.createSymbolicLink(
            atPath: dotfiles.appending(path: "bin/steerlab-cli").path,
            withDestinationPath: "../Lab/SteerLab.app/" + CommandLineLink.helperPath)
        let outcome = try CommandLineLink.install(
            app: dotfiles.appending(path: "Lab/SteerLab.app"),
            directory: local.appending(component: "bin"), replace: false,
            path: "/usr/bin")
        #expect(outcome.decision == .alreadyCurrent)
    }

    /// An absolute target that reaches this app through a symlinked folder is
    /// this app's link: both sides are compared with their links resolved.
    @Test func anAbsoluteTargetThroughALinkedFolderIsThisApp() throws {
        // real/SteerLab.app is the app; alias is a link to real.
        let root = try temporaryDirectory("alias").resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        let real = root.appending(component: "real")
        let helperFile = real.appending(path: "SteerLab.app/" + CommandLineLink.helperPath)
        try FileManager.default.createDirectory(
            at: helperFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("#!/bin/sh\n".utf8).write(to: helperFile)
        let alias = root.appending(component: "alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)
        let viaAlias = alias.appending(path: "SteerLab.app/" + CommandLineLink.helperPath).path
        #expect(
            CommandLineLink.decide(
                app: real.appending(component: "SteerLab.app"),
                existing: .symlink(target: viaAlias)) == .alreadyCurrent)
    }

    /// Each cause has its own repair, and none is this copy's own path: a
    /// kept copy is not told to move itself, and a build folder has no app
    /// to move.
    @Test func eachBadLocationHasItsOwnRepair() {
        let translocated = URL(
            filePath: "/private/var/folders/x/AppTranslocation/ABC/d/SteerLab.app")
        let kept = URL(filePath: "/Volumes/Lab/SteerLab/SteerLab-0.9.12-previous.app")
        let moving = CommandLineLink.badLocationRepair(app: translocated)
        let keptRepair = CommandLineLink.badLocationRepair(app: kept)
        let build = CommandLineLink.badLocationRepair(app: nil)
        #expect(moving.hasPrefix("move SteerLab.app"))
        #expect(!keptRepair.contains("move") && keptRepair.contains("current SteerLab.app"))
        #expect(!build.contains("move") && build.contains("inside SteerLab.app"))
        for (copy, repair) in [(translocated, moving), (kept, keptRepair)] {
            #expect(!repair.contains(copy.path), "\(repair)")
        }
        #expect(Set([moving, keptRepair, build]).count == 3)
    }

    /// The repoint is a rename over the old link, and leaves nothing behind.
    @Test func aRepointLeavesNoStagedLinkBehind() throws {
        let directory = try temporaryDirectory("staged")
        defer { try? FileManager.default.removeItem(at: directory) }
        let other = "/Applications/SteerLab.app/Contents/Helpers/steerlab-cli"
        try FileManager.default.createSymbolicLink(
            atPath: directory.appending(component: "steerlab-cli").path,
            withDestinationPath: other)
        let outcome = try CommandLineLink.install(
            app: app, directory: directory, replace: true, path: directory.path)
        #expect(outcome.changed)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["steerlab-cli"])
    }

    @Test func shellQuotingKeepsPlainPathsBareAndQuotesTheRest() {
        #expect(CommandLineLink.shellQuoted("/Volumes/Lab/bin") == "/Volumes/Lab/bin")
        #expect(CommandLineLink.shellQuoted("/Volumes/My Lab/bin") == "'/Volumes/My Lab/bin'")
        #expect(CommandLineLink.shellQuoted("it's") == "'it'\\''s'")
        #expect(CommandLineLink.shellQuoted("") == "''")
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
        // The copy is the problem, so the repair is never its own path.
        #expect(result.envelope.error?.repairAction == CommandLineLink.badLocationRepair(app: nil))
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
        #expect(reason == ExternalWorkspaceOpen.notAFolder)
    }

    /// What a researcher reads in the alert: the folder's name and the way
    /// forward, in the app's words — no path, no file names, no command.
    @Test func theAlertSpeaksPlainly() {
        let text = ExternalWorkspaceOpen.notAWorkspace(folderName: "Downloads", hadWorkspace: true)
        let first = ExternalWorkspaceOpen.notAWorkspace(folderName: "Downloads", hadWorkspace: false)
        #expect(text.hasPrefix("“Downloads” is not a SteerLab workspace"))
        #expect(text.contains("Open Workspace…") && text.contains("New Workspace…"))
        // Nothing "stays open" on a first launch, when nothing is open.
        #expect(text.contains("stays open") && !first.contains("stays open"))
        #expect(first.hasPrefix("“Downloads” is not a SteerLab workspace. A workspace"))
        for sentence in [text, first, ExternalWorkspaceOpen.notAFolder] {
            for jargon in ["/", "prompts", "marker", "open -a", "steerlab-cli", "--"] {
                #expect(!sentence.contains(jargon), "\(jargon) in: \(sentence)")
            }
        }
    }
}

extension ExternalWorkspaceOpenTests {

    /// Every way into a workspace asks `refusal` first: a workspace goes
    /// ahead, anything else gets the plain sentence named by the folder.
    @Test func aFolderThatIsNotAWorkspaceIsTurnedAwayPlainly() {
        let downloads = URL(filePath: "/Volumes/Lab/Downloads/")
        #expect(
            ExternalWorkspaceOpen.refusal(
                for: workspace, hasWorkspace: true, isWorkspace: { _ in true }) == nil)
        let refusal = ExternalWorkspaceOpen.refusal(
            for: downloads, hasWorkspace: false, isWorkspace: { _ in false })
        #expect(
            refusal == ExternalWorkspaceOpen.notAWorkspace(
                folderName: "Downloads", hadWorkspace: false))
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

        // A copy whose path will not last is offered to no one.
        let translocated = URL(
            filePath: "/private/var/folders/x/AppTranslocation/ABC/d/SteerLab.app")
        let fleeting = try WorkspaceBootstrap.handoff(
            root, executable: "steerlab-cli", app: translocated)
        #expect(fleeting["openInApp"] == nil)
        #expect(MacCommandLineRecord.read(workspaceRoot: root)?.app == app.path)
    }

    /// The ignore rule is what keeps the path out of git, so it comes back
    /// even when the record itself is unchanged.
    @Test func aDeletedIgnoreRuleComesBack() throws {
        let root = try temporaryDirectory("ignore")
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(MacCommandLineRecord.note(workspaceRoot: root, app: app, appVersion: "1"))
        let ignore = root.appending(path: ".steerlab/mac-app/.gitignore")
        try FileManager.default.removeItem(at: ignore)
        #expect(!MacCommandLineRecord.note(workspaceRoot: root, app: app, appVersion: "1"))
        #expect(try String(contentsOf: ignore, encoding: .utf8) == "*\n")
    }
}
