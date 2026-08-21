import Foundation

// =============================================================================
// The SteerLab home layout, and its first-run materialization
// (GENERAL-DISTRIBUTION-WORK-PLAN decision 8, work item (a); WP1's last open
// item).
//
// A git repository cannot ship its own parent directory, so the layout the
// distribution decision settled on — one home folder holding the study data,
// the private site library, and the code checkout as siblings — has until now
// been a README convention that every user had to reproduce by hand:
//
//     ~/SteerLab/
//     ├── Workspaces/     study workspaces, one folder each, each its own repo
//     ├── Sites/          the user's PRIVATE site registry
//     └── <checkout>/     the code checkout, under whatever name it was cloned
//
// This type is the mechanism behind it, in two halves so the app can share the
// seam with the CLI:
//
//   * `plan(home:)` is PURE — it observes what is there and reports what would
//     be created, and writes nothing. First launch can show it before asking.
//   * `apply(_:)` creates the missing directories and nothing else.
//
// Three properties are deliberate, and each closes a way this could do harm:
//
//   1. NEVER DESTRUCTIVE. Existing directories are reported and left exactly
//      as they are; a path occupied by a FILE is refused rather than replaced.
//      Running it twice is a no-op that says so.
//   2. `Sites/` STAYS EMPTY. It is typically a clone of the user's own private
//      repository, and `git clone <repo> Sites` refuses a non-empty target —
//      so this must not drop a README (or anything else) inside it. The
//      explanation lives in the command's output and in the shipped docs.
//   3. RESOLUTION-CHAIN NEUTRAL. Materializing directories is all it does. It
//      does not write UserDefaults, does not set `STEERLAB_WORKSPACE`, does not
//      create or open a workspace, does not `git init`, and does not clone or
//      create the checkout. `WorkspaceRoot`'s precedence (env → `--workspace` /
//      app override → UserDefaults → the legacy repo root) is untouched, and a
//      researcher whose workspaces live somewhere else entirely keeps working
//      exactly as before.
//
// `Sites/` is FIRST-CLASS (maintainer ruling, 2026-08-21). It is no longer a
// named-but-unused folder: `Sites/cluster-sites/` is THE canonical site
// registry for both the app and the CLI — one human-editable JSON file per
// site, resolved through `clusterSitesDirectory` below. Researchers work on
// several machines and keep `Sites/` as a private git repository, so git is
// the sync mechanism and UserDefaults has no future.
//
// Two properties of that follow from the list above rather than contradicting
// it. `Sites/` still STAYS EMPTY at materialization — `apply` creates the
// folder and nothing inside it, so `git clone <repo> Sites` still works; the
// `cluster-sites/` subdirectory is created lazily by the first WRITE, which is
// a researcher's own act. And nothing here runs git: the registry works as a
// plain directory, app and CLI writes simply leave the tree dirty, and
// committing is always the researcher's step.
// =============================================================================

/// The home folder's structure: what it should contain, what it does contain,
/// and how to make the second match the first.
public enum HomeLayout {

    // MARK: The layout

    /// Study workspaces, one folder each, each its own git repository.
    public static let workspacesDirectoryName = "Workspaces"

    /// The user's private site library — the canonical cluster-site registry,
    /// presets, private operations notes. Typically a clone of a private
    /// repository, which is why `apply` never writes anything INTO it: the
    /// registry subdirectory is born at the first site write.
    public static let sitesDirectoryName = "Sites"

    /// The canonical cluster-site registry, inside `Sites/`: one JSON file per
    /// site, named by the site's stable id.
    public static let clusterSitesDirectoryName = "cluster-sites"

    /// The directories `apply` materializes, in reading order.
    public static let directoryNames = [workspacesDirectoryName, sitesDirectoryName]

    /// What each directory is for, in one line — the CLI's output and the app's
    /// first-run sheet render this rather than retyping it.
    public static func purpose(of directoryName: String) -> String {
        switch directoryName {
        case workspacesDirectoryName:
            return "study workspaces, one folder each, each its own git repository"
        case sitesDirectoryName:
            return "your private site registry — cluster-site profiles, presets, ops notes"
        default:
            return ""
        }
    }

    /// The one paragraph about `Sites/` that has to travel with the layout: it
    /// is yours, it is private, it stays empty here so a clone can land in it,
    /// the app and the CLI both READ AND WRITE the registry inside it, git is
    /// how it reaches your other machines, and secrets never enter it.
    public static let sitesExplanation =
        "Sites/ is your private site registry — clone your site repository into "
        + "it, and both the app and steerlab-cli read and write "
        + "Sites/cluster-sites/<site-id>.json there. SteerLab never runs git: "
        + "its writes leave the tree dirty and committing/pushing is yours. "
        + "Tokens, keys, and passwords are never written into it — they stay in "
        + "this Mac's Keychain, so a new machine re-prompts once. Keep site "
        + "configuration out of workspaces you share."

    /// Test seam: when set, `defaultHome` — and therefore the canonical site
    /// registry — lives here instead of `~/SteerLab`. `nonisolated(unsafe)` is
    /// justified exactly as `ClusterSupportPaths.rootOverride` is: it is
    /// written by tests only, before the stores it scopes are touched. No test
    /// may ever leave it pointing at the researcher's real home.
    nonisolated(unsafe) public static var homeOverride: URL?

    /// `~/SteerLab`. The default home, and the only path this type invents.
    public static var defaultHome: URL {
        if let homeOverride { return homeOverride.standardizedFileURL }
        return URL(filePath: NSHomeDirectory())
            .appending(component: "SteerLab").standardizedFileURL
    }

    /// `~/SteerLab/Sites` — the private site library.
    public static var sitesDirectory: URL {
        defaultHome.appending(component: sitesDirectoryName)
    }

    /// `~/SteerLab/Sites/cluster-sites` — THE canonical site registry both
    /// clients read and write. Created lazily by the first write; a plain
    /// directory works, git is optional and never invoked.
    public static var clusterSitesDirectory: URL {
        sitesDirectory.appending(component: clusterSitesDirectoryName)
    }

    // MARK: Checkout detection

    /// Is this directory a SteerLab code checkout?
    ///
    /// **By content, never by name.** A clone of the release repository lands
    /// in a folder named after that repository, this research tree's folder is
    /// named something else again, and a user may rename either — so a name
    /// test would report "no checkout" for a checkout that is right there. The
    /// two markers are the ones that are true of every checkout and of nothing
    /// else in a home folder: the SwiftPM manifest at the root, and the Python
    /// engine's package directory.
    public static func isCheckout(_ url: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return false }
        guard fm.fileExists(atPath: url.appending(path: "Package.swift").path)
        else { return false }
        var packageIsDirectory: ObjCBool = false
        guard
            fm.fileExists(
                atPath: url.appending(path: "Server/steerlab_server").path,
                isDirectory: &packageIsDirectory), packageIsDirectory.boolValue
        else { return false }
        return true
    }

    /// The checkouts sitting directly inside `home`, by content, sorted by
    /// name. Never recursive: a checkout is a sibling of `Workspaces/`, and
    /// walking a home folder that may contain gigabytes of runs to find one is
    /// not worth an informational line.
    static func checkouts(in home: URL) -> [URL] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: home, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
        else { return [] }
        return
            entries
            .filter { !directoryNames.contains($0.lastPathComponent) }
            .filter { isCheckout($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// The checkout the RUNNING binary was compiled in, when there is one and
    /// it is not inside `home`.
    ///
    /// Informational, and deliberately so: an app-only user has no checkout at
    /// all and is not doing anything wrong, and a developer running from a
    /// checkout outside their home is the normal state of this repository
    /// today. Naming it is how someone reading the output understands why
    /// `init` reported no checkout in the home while a CLI plainly exists.
    static func externalRunningCheckout(relativeTo home: URL) -> URL? {
        let compiled = CodeResources.compiledCheckoutPath.standardizedFileURL
        guard isCheckout(compiled) else { return nil }
        guard !compiled.path.hasPrefix(home.standardizedFileURL.path + "/"),
            compiled.path != home.standardizedFileURL.path
        else { return nil }
        return compiled
    }

    // MARK: Plan

    /// One directory the layout wants, and what is there now.
    public struct Entry: Sendable, Equatable {

        /// `Workspaces` / `Sites`.
        public let name: String
        /// Where it goes.
        public let url: URL
        /// True when the directory was already there — so `apply` will leave
        /// it alone, and the report says `existing` rather than `created`.
        public let existed: Bool

        public init(name: String, url: URL, existed: Bool) {
            self.name = name
            self.url = url
            self.existed = existed
        }

        /// This directory's one-line purpose.
        public var purpose: String { HomeLayout.purpose(of: name) }
    }

    /// What `apply` would do, computed without touching anything.
    public struct Plan: Sendable, Equatable {

        /// The home this plan is about.
        public let home: URL
        /// True when the home folder itself was already there.
        public let homeExisted: Bool
        /// Every directory the layout wants, in reading order.
        public let entries: [Entry]
        /// Code checkouts found INSIDE the home, by content.
        public let checkouts: [URL]
        /// The running binary's own checkout, when it is somewhere else.
        public let externalCheckout: URL?

        public init(
            home: URL, homeExisted: Bool, entries: [Entry], checkouts: [URL],
            externalCheckout: URL?
        ) {
            self.home = home
            self.homeExisted = homeExisted
            self.entries = entries
            self.checkouts = checkouts
            self.externalCheckout = externalCheckout
        }

        /// The directories that are not there yet.
        public var missing: [Entry] { entries.filter { !$0.existed } }

        /// True when the layout is already materialized — the second run of an
        /// idempotent command.
        public var isComplete: Bool { homeExisted && missing.isEmpty }
    }

    /// Observe `home` and report what materializing the layout would do.
    ///
    /// Pure: creates nothing, reads nothing outside `home` except the running
    /// binary's own compiled-in checkout path.
    ///
    /// Refuses (malformed invocation, so the caller retypes rather than
    /// retries) when the home path or one of the layout's directory paths is
    /// occupied by a FILE. Replacing it is never right, and merging into it is
    /// not possible.
    public static func plan(home: URL) throws -> Plan {
        let fm = FileManager.default
        let root = home.standardizedFileURL

        var homeIsDirectory: ObjCBool = false
        let homeExists = fm.fileExists(atPath: root.path, isDirectory: &homeIsDirectory)
        if homeExists, !homeIsDirectory.boolValue {
            throw ExperimentError.malformed(
                "\(root.path) is a file, not a directory — a SteerLab home has "
                    + "to be a folder",
                repair: "steerlab-cli init --home <a directory path>")
        }

        var entries: [Entry] = []
        for name in directoryNames {
            let url = root.appending(component: name)
            var isDirectory: ObjCBool = false
            let exists = fm.fileExists(atPath: url.path, isDirectory: &isDirectory)
            if exists, !isDirectory.boolValue {
                throw ExperimentError.malformed(
                    "\(url.path) is a file — the SteerLab home needs \(name)/ to "
                        + "be a directory",
                    repair: "move or remove \(url.path), then steerlab-cli init "
                        + "--home \(root.path)")
            }
            entries.append(Entry(name: name, url: url, existed: exists))
        }

        return Plan(
            home: root, homeExisted: homeExists, entries: entries,
            checkouts: homeExists ? checkouts(in: root) : [],
            externalCheckout: externalRunningCheckout(relativeTo: root))
    }

    /// Create the directories the plan found missing, and nothing else.
    ///
    /// Returns the plan re-observed, so every entry reads `existed == true`
    /// afterwards and a caller can report created-vs-existing from the plan it
    /// passed in.
    @discardableResult
    public static func apply(_ plan: Plan) throws -> Plan {
        let fm = FileManager.default
        for entry in plan.missing {
            try fm.createDirectory(at: entry.url, withIntermediateDirectories: true)
        }
        return try Self.plan(home: plan.home)
    }
}
