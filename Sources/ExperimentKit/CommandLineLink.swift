import Foundation

/// Putting the app's command line on the researcher's PATH (2026-10-21).
///
/// A coding assistant that works in a workspace runs the app's bundled
/// command line, `SteerLab.app/Contents/Helpers/steerlab-cli`. Before this,
/// the only way to find it in a new session was the handoff Research Setup
/// copies, and putting it on PATH was a symlink the README asked the
/// researcher to type. This is that symlink, made by the app or by
/// `steerlab-cli install link`: `~/.local/bin/steerlab-cli`, pointing into
/// the app that made it.
///
/// The link outlives updates by construction: Install Update moves the new
/// app into the old one's path (`UpdateInstaller`), so the link keeps
/// pointing at the current version.
///
/// Three rules decide everything, and each guards against a way this could
/// do harm:
///   * **Only `steerlab-cli`.** The name `steerlab` belongs to the
///     cross-platform Python client, a different product with different
///     verbs (repository AGENTS.md, "Two products, two names"). Nothing here
///     creates or touches it.
///   * **Never over someone else's file.** An existing file, or a link that
///     does not point at a SteerLab app's helper, is left alone and named.
///   * **Only from a place that lasts.** A copy macOS runs from a temporary
///     read-only mount (translocated), a kept `-previous` copy, or a build
///     folder would leave a link that breaks; those refuse with the same
///     reasons the updater gives.
public enum CommandLineLink {

    /// The link's name. Never `steerlab`.
    public static let name = "steerlab-cli"

    /// Where the helper sits inside the app.
    public static let helperPath = "Contents/Helpers/steerlab-cli"

    /// `~/.local/bin` — the directory the README and the repository's
    /// AGENTS.md have always named.
    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".local/bin")
    }

    /// What is at the link's path now.
    public enum Existing: Sendable, Equatable {
        case absent
        /// A symbolic link, with its target made absolute against the
        /// link's own folder (`observe`).
        case symlink(target: String)
        /// A regular file, a directory, or anything else that is not a link.
        case otherFile
    }

    /// What making the link would do.
    public enum Decision: Sendable, Equatable {
        /// Nothing is there: create it.
        case create
        /// It already points at this app's helper.
        case alreadyCurrent
        /// A SteerLab link to ANOTHER app's helper. Replaced only when the
        /// caller says so, because the researcher may run two copies.
        case repoint(from: String)
        /// Not a SteerLab link: left alone, whatever the caller says.
        case foreign(String)
        /// This app is somewhere a link would break.
        case badLocation(String)
    }

    /// True for a target that is some SteerLab app's helper.
    public static func isSteerLabHelper(_ target: String) -> Bool {
        let url = URL(filePath: target)
        guard url.lastPathComponent == name else { return false }
        let contents = url.deletingLastPathComponent().deletingLastPathComponent()
        return url.deletingLastPathComponent().lastPathComponent == "Helpers"
            && contents.lastPathComponent == "Contents"
            && contents.deletingLastPathComponent().pathExtension == "app"
    }

    /// Why this app is not a place to link to, or nil when it is.
    public static func locationProblem(app: URL?) -> String? {
        guard let app else {
            return "this steerlab-cli runs from a build folder, not from SteerLab.app, "
                + "so a link to it would point at a build product. Run the copy "
                + "inside SteerLab.app instead: <app>/" + helperPath
        }
        if UpdateInstallContextReader.isTranslocated(app) {
            return UpdateInstallRefusal.translocated.line
        }
        if app.lastPathComponent.hasSuffix("-previous.app") {
            return "this is a kept copy of SteerLab from an earlier update, which a "
                + "later update moves to the Trash. Run the current SteerLab.app's "
                + "command line instead"
        }
        return nil
    }

    /// What making the link would do. A repoint is reported as one; whether
    /// it happens is the caller's `replace`, applied by `install`.
    public static func decide(app: URL?, existing: Existing) -> Decision {
        if let problem = locationProblem(app: app) { return .badLocation(problem) }
        let helper = app!.appending(path: helperPath).standardizedFileURL.path
        switch existing {
        case .absent:
            return .create
        case .otherFile:
            return .foreign(
                "a file that is not a link already has this name, and SteerLab "
                    + "will not replace it. Move it aside, or run steerlab-cli by "
                    + "its full path: \(helper)")
        case .symlink(let target):
            let resolved = URL(filePath: target).standardizedFileURL.path
            if canonical(resolved) == canonical(helper) { return .alreadyCurrent }
            guard isSteerLabHelper(resolved) else {
                return .foreign(
                    "a link to something else (\(target)) already has this name, and "
                        + "SteerLab will not replace it. Remove it yourself if it is "
                        + "stale, or run steerlab-cli by its full path: \(helper)")
            }
            return .repoint(from: target)
        }
    }

    /// The line to add to a shell profile when `directory` is not on PATH,
    /// or nil when it is. Advice only: nothing here edits a shell file.
    public static func pathAdvice(directory: URL, path: String?) -> String? {
        let wanted = directory.standardizedFileURL.path
        let entries = (path ?? "").split(separator: ":").map {
            URL(filePath: String($0)).standardizedFileURL.path
        }
        if entries.contains(wanted) { return nil }
        let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
        let spelled = wanted.hasPrefix(home + "/")
            ? "$HOME" + wanted.dropFirst(home.count) : wanted
        return "export PATH=\"\(spelled):$PATH\""
    }

    /// One spelling for two paths that name the same file: symlinked folders
    /// (`/tmp`, a dotfiles `~/.local/bin`) resolved, then standardized.
    static func canonical(_ path: String) -> String {
        URL(filePath: path).resolvingSymlinksInPath().standardizedFileURL.path
    }

    /// The repair for a copy whose path will not last, one per cause, and
    /// never this copy's own path: a build folder and a kept copy point at
    /// the current SteerLab.app; a translocated copy is moved to where
    /// SteerLab lives first.
    public static func badLocationRepair(app: URL?) -> String {
        let command = "<app>/" + helperPath + " install link"
        guard let app else {
            return "run the command line inside SteerLab.app: " + command
        }
        if UpdateInstallContextReader.isTranslocated(app) {
            return "move SteerLab.app to your SteerLab folder or to Applications, open "
                + "it from there, then run that copy's command line: " + command
        }
        return "run the current SteerLab.app's command line: " + command
    }

    /// A path or word as a shell reads it back: bare when it holds only
    /// characters no shell treats specially, otherwise in single quotes.
    public static func shellQuoted(_ word: String) -> String {
        let plain = CharacterSet(
            charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./+:=@%")
        if !word.isEmpty, word.unicodeScalars.allSatisfy(plain.contains) { return word }
        return "'" + word.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    // MARK: - Live

    /// What is at `link` now, without following it. A relative link target
    /// is resolved against the link's own folder, where the system resolves
    /// it, never against this process's working directory.
    public static func observe(_ link: URL) -> Existing {
        let fm = FileManager.default
        if let target = try? fm.destinationOfSymbolicLink(atPath: link.path) {
            let absolute = target.hasPrefix("/")
                ? target
                : link.deletingLastPathComponent().resolvingSymlinksInPath()
                    .appending(path: target).standardizedFileURL.path
            return .symlink(target: absolute)
        }
        // `fileExists` follows links; a dangling link was handled above.
        var isDirectory: ObjCBool = false
        return fm.fileExists(atPath: link.path, isDirectory: &isDirectory) ? .otherFile : .absent
    }

    /// The outcome of `install`.
    public struct Outcome: Sendable, Equatable {
        public var decision: Decision
        public var link: URL
        public var target: String?
        /// True when this call created or replaced the link.
        public var changed: Bool
        /// The PATH line to add, when the directory is not on PATH.
        public var pathAdvice: String?
    }

    /// Makes the link when the decision allows it. Refusals are returned,
    /// never thrown, so every surface can show the sentence beside its
    /// button; a filesystem failure throws.
    public static func install(
        app: URL?, directory: URL = defaultDirectory, replace: Bool,
        path: String? = ProcessInfo.processInfo.environment["PATH"]
    ) throws -> Outcome {
        let link = directory.appending(component: name)
        let decision = decide(app: app, existing: observe(link))
        let helper = app.map { $0.appending(path: helperPath).standardizedFileURL.path }
        var outcome = Outcome(
            decision: decision, link: link, target: helper, changed: false,
            pathAdvice: pathAdvice(directory: directory, path: path))
        switch decision {
        case .create:
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(
                atPath: link.path, withDestinationPath: helper!)
            outcome.changed = true
        case .repoint where replace:
            // Atomically: a new link beside the old one, renamed over it. A
            // remove-then-create could delete whatever replaced the link in
            // between, and rename(2) refuses to put a link over a directory.
            let staged = directory.appending(component: ".\(name).\(UUID().uuidString)")
            try FileManager.default.createSymbolicLink(
                atPath: staged.path, withDestinationPath: helper!)
            guard Darwin.rename(staged.path, link.path) == 0 else {
                let failure = POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                try? FileManager.default.removeItem(at: staged)
                throw failure
            }
            outcome.changed = true
        case .repoint, .alreadyCurrent, .foreign, .badLocation:
            break
        }
        return outcome
    }

    /// One sentence for the outcome, the same on every surface.
    public static func sentence(_ outcome: Outcome) -> String {
        let where_ = outcome.link.path
        var text: String
        switch outcome.decision {
        case .create:
            text = "Linked \(where_) to this app's command line."
        case .alreadyCurrent:
            text = "\(where_) already points at this app's command line."
        case .repoint(let from):
            text = outcome.changed
                ? "Pointed \(where_) at this app's command line (it pointed at \(from))."
                : "\(where_) points at another copy of SteerLab (\(from)). Replace it "
                    + "to use this one."
        case .foreign(let reason):
            return "Did not link \(where_): \(reason)."
        case .badLocation(let reason):
            return "Did not link \(where_): \(reason)."
        }
        if let advice = outcome.pathAdvice {
            text += " \(outcome.link.deletingLastPathComponent().path) is not on your "
                + "PATH yet; add this line to your shell profile: \(advice)"
        }
        return text
    }
}
