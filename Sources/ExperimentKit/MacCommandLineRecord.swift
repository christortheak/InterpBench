import Foundation

/// Where this Mac's SteerLab command line is, recorded in the workspace
/// (2026-10-21), so a coding assistant's NEXT session can find it.
///
/// Before this, the path lived only in the handoff Research Setup copies:
/// a new session without the pasted text had to guess, and the guess could
/// land on an old copy in a build folder. The workspace's AGENTS.md now
/// points here, and the app keeps the record current:
///
///   * the app writes it when it opens, creates, or starts up on this
///     workspace (`WorkspaceStore.switchTo`, and the app's launch), and
///     whenever it or its command line prepares a handoff
///     (`WorkspaceBootstrap.handoff`);
///   * it names the app the researcher actually uses, so a moved app is
///     corrected the next time it opens the workspace;
///   * it is never written from somewhere that will not last (a translocated
///     copy, a kept `-previous` copy, a build folder), for the reasons
///     `CommandLineLink.locationProblem` gives.
///
/// It names a path on this machine, so it lives in its own folder with its
/// own `.gitignore` (`*`), the rule `WorkspaceJobOrigins` uses: it stays out
/// of the workspace's history even in a workspace whose top-level
/// `.gitignore` predates `.steerlab/*`.
public enum MacCommandLineRecord {
    public static let schemaVersion = 1
    public static let relativeDirectory = [".steerlab", "mac-app"]
    public static let fileName = "command-line.json"

    /// The workspace-relative path the agent guide names.
    public static var relativePath: String {
        (relativeDirectory + [fileName]).joined(separator: "/")
    }

    public struct Record: Codable, Sendable, Equatable {
        public var schemaVersion: Int
        /// The command line to run: `<app>/Contents/Helpers/steerlab-cli`.
        public var commandLine: String
        /// The app that wrote this record, which `open -a` opens.
        public var app: String
        public var appVersion: String
    }

    public static func fileURL(workspaceRoot: URL) -> URL {
        relativeDirectory.reduce(workspaceRoot) { $0.appending(component: $1) }
            .appending(component: fileName)
    }

    public static func read(workspaceRoot: URL) -> Record? {
        guard let data = try? Data(contentsOf: fileURL(workspaceRoot: workspaceRoot)) else {
            return nil
        }
        return try? JSONDecoder().decode(Record.self, from: data)
    }

    /// Records `app` for this workspace when it is a lasting place and the
    /// record would change. Best effort: a workspace that cannot be written
    /// to keeps working, and a coding assistant falls back to PATH or asks.
    /// Returns whether it wrote.
    @discardableResult
    public static func note(
        workspaceRoot: URL, app: URL? = CodeResources.enclosingAppBundle,
        appVersion: String = SteerLabVersion.version
    ) -> Bool {
        guard let app, CommandLineLink.locationProblem(app: app) == nil else { return false }
        let standardized = app.standardizedFileURL
        let record = Record(
            schemaVersion: schemaVersion,
            commandLine: standardized.appending(path: CommandLineLink.helperPath).path,
            app: standardized.path, appVersion: appVersion)
        let url = fileURL(workspaceRoot: workspaceRoot)
        let folder = url.deletingLastPathComponent()
        let ignore = folder.appending(component: ".gitignore")
        if read(workspaceRoot: workspaceRoot) == record {
            // Unchanged, but the ignore rule is what keeps the path out of
            // git, so a deleted one comes back.
            if !FileManager.default.fileExists(atPath: ignore.path) {
                try? Data("*\n".utf8).write(to: ignore, options: .atomic)
            }
            return false
        }
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: ignore.path) {
                try Data("*\n".utf8).write(to: ignore, options: .atomic)
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try (encoder.encode(record) + Data("\n".utf8)).write(to: url, options: .atomic)
            return true
        } catch {
            return false
        }
    }
}
