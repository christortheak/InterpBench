import AppKit
import CryptoKit
import Darwin
import Foundation

// MARK: - Updating SteerLab.app from its own command line (public issue #5)
//
// Some researchers reach SteerLab only through a coding assistant that runs
// the app's bundled command line (`SteerLab.app/Contents/Helpers/steerlab-cli`,
// linked at `~/.local/bin/steerlab-cli`). The app's updater lives behind a
// menu item, a banner, and a confirmation sheet, so for them an update was
// invisible and out of reach: an assistant authored a batch of studies with a
// 0.9.8 command line while 0.9.18 was current, and nothing told it.
//
// This file gives the command line the same updater, and nothing of its own
// where the app already has an answer:
//
//   * THE CHECK is `UpdateCheckService`, with the app's privacy contract
//     unchanged: one unauthenticated release lookup, nothing about this Mac
//     sent. Its SETTINGS are the app's: the opt-out, the once-a-day stamp,
//     the dismissed release, and the newest release the last check saw all
//     live in the app's own preferences domain (its bundle identifier), which
//     the command line opens by name (`UpdateCheckPreferences.forApp`). So a
//     researcher who turned checks off in the app has turned them off here,
//     and a check made by either counts for both.
//   * Only `install version` (`--version`) may refresh a stale answer, with a
//     short deadline, and only while automatic checks are on. `--check` asks
//     once whatever the setting, as Check for Updates… does. Every other verb
//     reads the recorded answer and never waits on the network.
//   * THE INSTALL is `UpdateInstaller`, unchanged in every check it makes:
//     GitHub over HTTPS only, the checksum, the code signature and its team,
//     the stapled notarization, the version, the kept `-previous` copy, and
//     the move into place. It runs in two steps, on the pattern of `setup
//     plan` / `setup apply`: `install update-plan` looks the release up and
//     downloads nothing else; `install update --expect <digest> --yes`
//     re-derives that plan, refuses if anything bound into its digest moved,
//     and only then downloads, verifies, and swaps.
//
// THE RUNNING PROCESS. The command line installing the update runs from
// inside the bundle it replaces. The swap is renames on one volume: this
// process's executable is renamed with the old app to
// `SteerLab-<version>-previous.app`, and its pages stay valid (a rename does
// not change the file under a running process). Nothing it does after the
// swap reads its own bundle: the envelope is built from values computed
// before. `~/.local/bin/steerlab-cli` names the helper by PATH, and the new
// app is moved into the same path, so the link runs the new version from the
// next command on; `install update` reads the link back to say so.

public enum CommandLineUpdate {

    /// Where to tell a person to look when no copy can update itself.
    static var releasesPage: URL { UpdateRepository.default.releasesPage }

    // MARK: - The app this command line belongs to

    /// The `.app` around this command line, read from its Info.plist.
    public struct App: Sendable, Equatable {
        public var bundle: URL
        public var bundleIdentifier: String?
        public var version: SemanticVersion
        public var sourceRevision: String?

        public init(
            bundle: URL, bundleIdentifier: String?, version: SemanticVersion,
            sourceRevision: String? = nil
        ) {
            self.bundle = bundle
            self.bundleIdentifier = bundleIdentifier
            self.version = version
            self.sourceRevision = sourceRevision
        }

        /// The command line inside it, as a path a person can type.
        public var helper: URL { bundle.appending(path: CommandLineLink.helperPath) }
    }

    // MARK: - The seams

    /// Everything the command line's updater touches. `live` is the real
    /// machine; a test builds one from the installer's own fakes.
    public struct Environment: Sendable {
        /// Nil for a command line that is not inside SteerLab.app (a build
        /// folder, or a `scripts/install-cli.sh` install).
        public var app: App?
        /// This build's version: the app's when bundled.
        public var version: SemanticVersion
        /// The app's preferences, shared with it. Nil when not bundled.
        public var preferences: UpdateCheckPreferences?
        /// The domain `preferences` reads, reported so a reader can check it.
        public var preferencesDomain: String?
        /// Set when this process may not use the network at all (a build
        /// script's launch check), with the reason to report.
        public var offlineReason: String?
        /// How long an automatic refresh may take, start to finish.
        public var automaticDeadline: Duration
        /// How long an asked-for lookup may take.
        public var explicitDeadline: Duration
        /// The check, built for an automatic (`false`) or asked-for (`true`)
        /// lookup.
        public var makeService: @Sendable (_ explicit: Bool) -> UpdateCheckService
        /// The installer's eligibility input for an app on disk.
        public var installContext: @Sendable (URL) -> UpdateInstallContext
        /// The installer's seams (network, signatures, tools, file system).
        public var installEnvironment: @Sendable () -> UpdateInstallEnvironment
        /// Executable paths of other processes running out of an app.
        public var processesInsideApp: @Sendable (URL) -> [String]
        /// Copies of the app that are open now, by bundle identifier.
        public var openApps: @Sendable (String) -> [String]
        /// Folders searched for SteerLab.app copies.
        public var copyFolders: [URL]
        /// The folder holding the `steerlab-cli` link.
        public var linkDirectory: URL
        public var fileSystem: any UpdateFileSystem
        /// Ignore Ctrl-C and hang-ups while the app is being swapped, so an
        /// interruption cannot stop between two renames. Off in tests.
        public var deferSignalsDuringSwap: Bool

        public init(
            app: App?, version: SemanticVersion,
            preferences: UpdateCheckPreferences?, preferencesDomain: String?,
            offlineReason: String? = nil,
            automaticDeadline: Duration = .seconds(5),
            explicitDeadline: Duration = .seconds(20),
            makeService: @escaping @Sendable (_ explicit: Bool) -> UpdateCheckService,
            installContext: @escaping @Sendable (URL) -> UpdateInstallContext,
            installEnvironment: @escaping @Sendable () -> UpdateInstallEnvironment,
            processesInsideApp: @escaping @Sendable (URL) -> [String],
            openApps: @escaping @Sendable (String) -> [String],
            copyFolders: [URL], linkDirectory: URL,
            fileSystem: any UpdateFileSystem = LiveUpdateFileSystem(),
            deferSignalsDuringSwap: Bool = false
        ) {
            self.app = app
            self.version = version
            self.preferences = preferences
            self.preferencesDomain = preferencesDomain
            self.offlineReason = offlineReason
            self.automaticDeadline = automaticDeadline
            self.explicitDeadline = explicitDeadline
            self.makeService = makeService
            self.installContext = installContext
            self.installEnvironment = installEnvironment
            self.processesInsideApp = processesInsideApp
            self.openApps = openApps
            self.copyFolders = copyFolders
            self.linkDirectory = linkDirectory
            self.fileSystem = fileSystem
            self.deferSignalsDuringSwap = deferSignalsDuringSwap
        }

        /// The real machine, for the command line described by `provenance`.
        public static func live(
            provenance: InstallProvenance = .resolve(),
            environment: [String: String] = ProcessInfo.processInfo.environment
        ) -> Environment {
            let fileSystem = LiveUpdateFileSystem()
            let fallback =
                SemanticVersion(SteerLabVersion.version)
                ?? SemanticVersion(major: 0, minor: 0, patch: 0)
            let app = provenance.app.map { bundled in
                App(
                    bundle: bundled.bundle,
                    bundleIdentifier: fileSystem.bundleInfo(at: bundled.bundle).identifier,
                    version: bundled.version.flatMap(SemanticVersion.init) ?? fallback,
                    sourceRevision: bundled.sourceRevision)
            }
            let version = app?.version ?? fallback
            // A build script's launch check performs no network activity and
            // never touches the researcher's real preferences: the app uses
            // an isolated suite then, and so does its command line.
            let offline = LaunchCheckMode.isActive
                ? "this is a launch check (\(LaunchCheckMode.environmentVariable)), "
                    + "which makes no network requests"
                : nil
            let domain = app?.bundleIdentifier.map {
                offline == nil ? $0 : LaunchCheckMode.isolatedDefaultsSuiteName
            }
            let preferences = domain.flatMap { UpdateCheckPreferences.forApp(bundleIdentifier: $0) }
            let revision = app?.sourceRevision
            var folders = [
                URL(filePath: "/Applications"),
                FileManager.default.homeDirectoryForCurrentUser.appending(component: "Applications"),
            ]
            if let app { folders.append(app.bundle.deletingLastPathComponent()) }
            folders.append(HomeLayout.defaultHome)
            return Environment(
                app: app, version: version, preferences: preferences,
                preferencesDomain: preferences == nil ? nil : domain,
                offlineReason: offline,
                makeService: { explicit in
                    UpdateCheckService(
                        currentVersion: version, builtRevision: revision,
                        checkout: CodeResources.executableCheckoutRoot,
                        feed: GitHubReleaseFeed(timeout: explicit ? 8 : 4),
                        git: SystemUpdateGitProbe(timeout: explicit ? .seconds(10) : .seconds(4)))
                },
                installContext: { UpdateInstallContextReader.context(forApp: $0) },
                installEnvironment: {
                    UpdateInstallEnvironment(
                        transport: GitHubUpdateTransport(),
                        codeSigning: SecurityFrameworkCodeSigning())
                },
                processesInsideApp: { UpdateBundleProcesses.executablePaths(insideApp: $0) },
                openApps: { identifier in
                    NSRunningApplication.runningApplications(withBundleIdentifier: identifier)
                        .filter { $0.processIdentifier != getpid() && !$0.isTerminated }
                        .map { $0.bundleURL?.path ?? "process \($0.processIdentifier)" }
                },
                copyFolders: uniqueFolders(folders),
                linkDirectory: CommandLineLink.defaultDirectory,
                fileSystem: fileSystem,
                deferSignalsDuringSwap: true)
        }

        static func uniqueFolders(_ folders: [URL]) -> [URL] {
            var seen: Set<String> = []
            return folders.filter { seen.insert(CommandLineLink.canonical($0.path)).inserted }
        }
    }

    // MARK: - Copies of the app, and the link on PATH

    /// One SteerLab.app found on this Mac.
    public struct AppCopy: Sendable, Equatable {
        public var path: URL
        public var version: String?
        public var bundleIdentifier: String?
        /// The copy this command line runs from: the one an update replaces.
        public var isThisCopy: Bool
        /// `~/.local/bin/steerlab-cli` runs this copy's command line.
        public var isLinked: Bool
        /// A copy an earlier update kept (`SteerLab-<version>-previous.app`).
        public var isKeptCopy: Bool
    }

    /// What `~/.local/bin/steerlab-cli` is.
    public enum LinkState: Sendable, Equatable {
        case absent(link: URL)
        /// A link to the command line inside `app`.
        case steerLab(link: URL, app: URL)
        /// A file, or a link to something that is not a SteerLab command line.
        case other(link: URL)

        public var link: URL {
            switch self {
            case .absent(let link), .steerLab(let link, _), .other(let link): link
            }
        }

        public var app: URL? {
            if case .steerLab(_, let app) = self { return app }
            return nil
        }
    }

    public static func linkState(directory: URL) -> LinkState {
        let link = directory.appending(component: CommandLineLink.name)
        switch CommandLineLink.observe(link) {
        case .absent: return .absent(link: link)
        case .otherFile: return .other(link: link)
        case .symlink(let target):
            guard CommandLineLink.isSteerLabHelper(target) else { return .other(link: link) }
            let app = URL(filePath: target).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
            return .steerLab(link: link, app: app.standardizedFileURL)
        }
    }

    static func samePath(_ left: URL?, _ right: URL?) -> Bool {
        guard let left, let right else { return false }
        return CommandLineLink.canonical(left.path) == CommandLineLink.canonical(right.path)
    }

    /// Every SteerLab*.app in the usual places, plus this copy and the one
    /// the link runs, each once.
    public static func copies(_ env: Environment, link: LinkState) -> [AppCopy] {
        let fs = env.fileSystem
        var candidates: [URL] = []
        if let app = env.app { candidates.append(app.bundle) }
        for folder in env.copyFolders {
            let entries = (try? fs.contents(of: folder)) ?? []
            candidates += entries.filter {
                let name = $0.lastPathComponent
                return name.hasPrefix("SteerLab") && name.hasSuffix(".app") && fs.isDirectory($0)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        if let linked = link.app { candidates.append(linked) }
        var seen: Set<String> = []
        var found: [AppCopy] = []
        for candidate in candidates {
            guard seen.insert(CommandLineLink.canonical(candidate.path)).inserted else { continue }
            let isThis = samePath(candidate, env.app?.bundle)
            let isLinked = samePath(candidate, link.app)
            let info = fs.bundleInfo(at: candidate)
            // A folder named like the app with no Info.plist is not a copy.
            guard info.identifier != nil || isThis || isLinked else { continue }
            found.append(
                AppCopy(
                    path: candidate.standardizedFileURL, version: info.version,
                    bundleIdentifier: info.identifier, isThisCopy: isThis, isLinked: isLinked,
                    isKeptCopy: UpdateStagingLayout.previousCopyVersion(
                        fromName: candidate.lastPathComponent) != nil))
        }
        return found
    }

    // MARK: - The report (`install version`)

    /// The four states the app's check distinguishes, on the wire.
    public enum State: String, Sendable {
        case releaseAvailable, newerSourceAvailable, upToDate, unknown
    }

    /// What `install version` says about updates.
    public struct Report: Sendable {
        public var thisVersion: SemanticVersion
        public var state: State
        /// The newest release known, from this invocation's check or the
        /// recorded answer of the last one.
        public var latest: UpdateLatestRelease?
        /// Nil when nothing is known.
        public var newerThanThisBuild: Bool?
        /// Whether this invocation asked GitHub.
        public var checkedNow: Bool
        /// `checkedNow`, `lastCheck`, or `nothing`: where `state` came from.
        public var answeredBy: String
        /// Why nothing is known, or what a check that reached nothing said.
        public var reason: String?
        /// The remote's `main`, when it is all that differs.
        public var remoteMainRevision: String?
        public var automaticChecksEnabled: Bool?
        public var lastCheck: Date?
        public var nextAutomaticCheck: Date?
        public var dismissedVersion: String?
        public var preferencesDomain: String?
        /// Whether the `clientUpdateAvailable` advisory goes with it.
        public var announces: Bool
        public var copies: [AppCopy]
        public var link: LinkState
        /// The copy an update would replace.
        public var updateTarget: URL?
    }

    /// One check, bounded by `deadline`. A lookup that runs out of time is
    /// the same fact as one that reached nothing: unknown.
    static func check(
        _ service: UpdateCheckService, within deadline: Duration
    ) async -> UpdateCheckResult {
        await withTaskGroup(of: UpdateCheckResult?.self) { group in
            group.addTask { await service.checkDetailed() }
            group.addTask {
                try? await Task.sleep(for: deadline)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            let seconds = Int(deadline.components.seconds)
            return first
                ?? UpdateCheckResult(
                    availability: .unknown(
                        reason: "GitHub did not answer within \(seconds) seconds"),
                    release: nil)
        }
    }

    /// The report, refreshing the recorded answer first when that is
    /// allowed: `explicitCheck` (`--check`) always asks, as the app's Check
    /// for Updates… does; otherwise only a due automatic check asks, and
    /// only while automatic checks are on. Both record the check in the
    /// app's own settings, the way the app does.
    public static func report(_ env: Environment, explicitCheck: Bool) async -> Report {
        let link = linkState(directory: env.linkDirectory)
        var report = Report(
            thisVersion: env.version, state: .unknown, latest: nil, newerThanThisBuild: nil,
            checkedNow: false, answeredBy: "nothing", reason: nil, remoteMainRevision: nil,
            automaticChecksEnabled: env.preferences?.automaticChecksEnabled,
            lastCheck: nil, nextAutomaticCheck: nil,
            dismissedVersion: env.preferences?.dismissedVersion,
            preferencesDomain: env.preferencesDomain, announces: false,
            copies: copies(env, link: link), link: link, updateTarget: env.app?.bundle)

        var live: UpdateCheckResult?
        if let offline = env.offlineReason {
            report.reason = "not checked: \(offline)"
        } else if explicitCheck {
            env.preferences?.recordCheck()
            live = await check(env.makeService(true), within: env.explicitDeadline)
        } else if let preferences = env.preferences, preferences.shouldRunAutomaticCheck {
            preferences.recordCheck()
            live = await check(env.makeService(false), within: env.automaticDeadline)
        }
        if let live {
            env.preferences?.recordAnswer(live)
            report.checkedNow = true
        }

        switch live?.availability {
        case .releaseAvailable(let version, let page)?:
            // The release the decision announced, so the version named and
            // the state never disagree.
            report.state = .releaseAvailable
            report.latest = live?.latest.flatMap { $0.version == version ? $0 : nil }
                ?? UpdateLatestRelease(version: version, tag: "v\(version.releaseDescription)", page: page)
            report.answeredBy = "checkedNow"
        case .newerSourceAvailable(let remote, _)?:
            report.state = .newerSourceAvailable
            report.latest = live?.latest
            report.remoteMainRevision = remote
            report.answeredBy = "checkedNow"
        case .upToDate?:
            report.state = .upToDate
            report.latest = live?.latest
            report.answeredBy = "checkedNow"
        case .unknown(let reason)?:
            report.reason = "the check could not finish: \(reason)"
        case nil:
            break
        }
        if report.answeredBy == "nothing", let recorded = env.preferences?.latestRelease {
            report.latest = recorded
            report.state = recorded.version > env.version ? .releaseAvailable : .upToDate
            report.answeredBy = "lastCheck"
        }
        if report.answeredBy == "nothing", report.reason == nil {
            report.reason = noAnswerReason(env)
        }
        if report.state != .unknown {
            report.newerThanThisBuild = report.latest.map { $0.version > env.version } ?? false
        }

        if let preferences = env.preferences {
            report.lastCheck = preferences.lastCheck
            if preferences.automaticChecksEnabled {
                report.nextAutomaticCheck =
                    preferences.lastCheck?.addingTimeInterval(UpdateCheckPolicy.automaticInterval)
            }
        }
        report.announces =
            report.state == .releaseAvailable
            && (explicitCheck || announcesAmbiently(env, latest: report.latest))
        return report
    }

    static func noAnswerReason(_ env: Environment) -> String {
        guard let preferences = env.preferences else {
            return "this command line is not part of SteerLab.app, so it shares no update "
                + "settings and does not check on its own; install version --check asks "
                + "GitHub once"
        }
        if !preferences.automaticChecksEnabled {
            return "automatic update checks are off in SteerLab's settings; install "
                + "version --check asks GitHub once"
        }
        return "no check has recorded an answer yet; install version --check asks GitHub now"
    }

    /// The rule for saying so without being asked: the app's banner rule. A
    /// newer release recorded by a check, automatic checks on, and not a
    /// release the researcher dismissed in the app.
    static func announcesAmbiently(_ env: Environment, latest: UpdateLatestRelease?) -> Bool {
        guard let preferences = env.preferences, env.app != nil,
            preferences.automaticChecksEnabled,
            let latest, latest.version > env.version
        else { return false }
        return preferences.bannerMessage(
            for: .releaseAvailable(version: latest.version, page: latest.page ?? releasesPage))
            != nil
    }

    // MARK: - The advisory

    /// `clientUpdateAvailable`, naming both versions.
    public static func advisory(
        latest: UpdateLatestRelease, env: Environment
    ) -> SteerLabCLIEnvelope.Advisory {
        var detail = "SteerLab \(latest.version.releaseDescription) is available"
        if let observedAt = latest.observedAt {
            detail += " (as of \(observedAt.ISO8601Format()))"
        }
        detail += "; this command line is SteerLab \(env.version.releaseDescription)"
        if let app = env.app { detail += ", part of \(app.bundle.path)" }
        detail += ". Tell the researcher, show them the plan from steerlab-cli install "
            + "update-plan --json, and install it only after they approve that plan."
        return .init(.clientUpdateAvailable, detail)
    }

    /// The advisory for a verb that was not asked about updates (`workspace
    /// guide`, `setup inspect`): from the RECORDED answer only, never the
    /// network, and only by the banner rule.
    public static func ambientAdvisories(_ env: Environment) -> [SteerLabCLIEnvelope.Advisory] {
        guard let latest = env.preferences?.latestRelease,
            announcesAmbiently(env, latest: latest)
        else { return [] }
        return [advisory(latest: latest, env: env)]
    }

    // MARK: - Refusals

    /// Why the command line will not plan or install an update, with the
    /// state, the stable code, and the repair.
    public struct Refusal: Error, Sendable, Equatable {
        public var code: String
        public var state: SteerLabCLIState
        public var reason: String
        public var repairAction: String
        public var payload: [String: JSONValue]

        public init(
            code: String, state: SteerLabCLIState = .refused, reason: String,
            repairAction: String, payload: [String: JSONValue] = [:]
        ) {
            self.code = code
            self.state = state
            self.reason = reason
            self.repairAction = repairAction
            self.payload = payload
        }
    }

    /// The command that plans from a given copy, for a repair.
    static func planCommand(app: URL?) -> String {
        let executable = app.map { CommandLineLink.shellQuoted($0.appending(path: CommandLineLink.helperPath).path) }
            ?? "steerlab-cli"
        return executable + " install update-plan --json"
    }

    /// The installer's own refusal line, and the move that changes it.
    static func eligibilityRefusal(_ refusal: UpdateInstallRefusal, app: URL?) -> Refusal {
        let repair: String
        switch refusal {
        case .translocated, .readOnlyVolume, .folderNotWritable:
            repair = "Move SteerLab.app to your SteerLab folder or to Applications, open it "
                + "from there once, then run that copy's command line: "
                + "<app>/\(CommandLineLink.helperPath) install update-plan --json"
        case .runningKeptCopy:
            let current = app.map {
                $0.deletingLastPathComponent().appending(component: "SteerLab.app")
            }
            repair = "Run the current SteerLab.app's command line instead: "
                + planCommand(app: current)
        case .notAnAppBundle:
            repair = "Update the checkout and rebuild (scripts/install-cli.sh), or use the "
                + "packaged app's command line: <app>/\(CommandLineLink.helperPath) "
                + "install update-plan --json. Releases: \(releasesPage.absoluteString)"
        case .adHocSignature, .otherCertificate, .unsigned, .notNotarized,
            .signatureUnreadable, .missingBundleIdentifier:
            repair = "Download the new release from the Releases page and replace "
                + "SteerLab.app with it: \(releasesPage.absoluteString)"
        }
        return Refusal(
            code: "updateNotEligible", reason: refusal.line, repairAction: repair,
            payload: ["eligibility": .string(eligibilityName(refusal))])
    }

    static func eligibilityName(_ refusal: UpdateInstallRefusal) -> String {
        switch refusal {
        case .notAnAppBundle: "notAnAppBundle"
        case .adHocSignature: "adHocSignature"
        case .otherCertificate: "otherCertificate"
        case .unsigned: "unsigned"
        case .notNotarized: "notNotarized"
        case .signatureUnreadable: "signatureUnreadable"
        case .missingBundleIdentifier: "missingBundleIdentifier"
        case .translocated: "translocated"
        case .readOnlyVolume: "readOnlyVolume"
        case .folderNotWritable: "folderNotWritable"
        case .runningKeptCopy: "runningKeptCopy"
        }
    }

    /// An installer error, in the command line's words where the app's
    /// repair names one of its buttons.
    static func installRefusal(_ error: UpdateInstallError) -> Refusal {
        let page = releasesPage.absoluteString
        switch error {
        case .busy:
            return Refusal(
                code: "updateBusy", reason: error.summary,
                repairAction: "Wait for it to finish, or stop it, then run the same "
                    + "command again. Nothing was changed.")
        case .cancelled:
            return Refusal(
                code: "updateCancelled", state: .failed, reason: error.summary,
                repairAction: "Nothing was changed. Run the same command again when you "
                    + "are ready.")
        case .noAppArchive, .ambiguousAppArchive, .missingChecksum:
            return Refusal(
                code: "releaseFilesUnavailable", reason: error.summary,
                repairAction: error.repair + " " + page)
        case .untrustedDownloadLocation, .notNewer, .checksumMismatch, .archiveRejected,
            .signatureRejected, .versionMismatch:
            return Refusal(
                code: "updateRejected", reason: error.summary,
                repairAction: error.repair + " " + page)
        case .archiveTooLarge, .downloadFailed, .downloadTimedOut, .checksumUnreadable,
            .unpackFailed, .stagingUnavailable, .installFailedRestored,
            .installFailedNotRestored:
            return Refusal(
                code: "updateFailed", state: .failed, reason: error.summary,
                repairAction: error.repair
                    + (error.repair.contains("Releases page") ? " " + page : ""))
        }
    }

    // MARK: - The plan (`install update-plan`)

    /// A reviewed update: everything `install update` will do, and the
    /// digest that binds it.
    public struct Plan: Sendable, Equatable {
        public var currentVersion: SemanticVersion
        public var release: PublishedRelease
        public var assets: UpdateAssetPair
        public var target: UpdateInstallTarget
        public var previousCopy: URL
        /// A kept copy already has that name (the same version installed
        /// twice); it goes to the Trash once the install is in place.
        public var previousCopyExists: Bool
        /// Older kept copies of this app that go to the Trash.
        public var olderCopiesToTrash: [URL]
        public var link: LinkState
        /// What stands in the way right now. Not bound into the digest: it
        /// changes from minute to minute, and `install update` checks it
        /// again, twice.
        public var blockers: [String]
        public var digest: String

        /// The sentences a researcher approves, in order.
        public var steps: [String] {
            let size = assets.archive.size.map {
                ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) + " "
            } ?? ""
            let version = release.version.releaseDescription
            let folder = target.appBundle.deletingLastPathComponent().path
            var steps = [
                "Download \(assets.archive.name) (\(size)from the project's GitHub "
                    + "Releases page, release \(release.tag)) and its checksum file "
                    + "\(assets.checksum.name) from the same release, over HTTPS from "
                    + "GitHub's hosts only. Nothing about this Mac is sent.",
                "Check it on this Mac, offline: the file against its checksum (which "
                    + "shows it arrived intact, not who made it), then the app inside "
                    + "against Apple's notarization ticket stapled in it and this copy's "
                    + "own developer signature (Developer ID team \(target.teamID), "
                    + "bundle identifier \(target.bundleIdentifier)), and that it is "
                    + "SteerLab \(version).",
                "Rename the copy you have now (\(currentVersion.releaseDescription)) to "
                    + "\(previousCopy.lastPathComponent), in \(folder), and move "
                    + "SteerLab \(version) into its place at \(target.appBundle.path). "
                    + "If anything fails, the current copy is put back. To go back later, "
                    + "move the new SteerLab.app to the Trash and rename the kept copy to "
                    + "\(target.appBundle.lastPathComponent).",
            ]
            if previousCopyExists {
                steps.append(
                    "A kept copy named \(previousCopy.lastPathComponent) is already there; "
                        + "it goes to the Trash once the new version is in place.")
            }
            if !olderCopiesToTrash.isEmpty {
                steps.append(
                    UpdateInstallReview.olderCopiesLine + " Today that is: "
                        + olderCopiesToTrash.map(\.lastPathComponent).joined(separator: ", ")
                        + ".")
            }
            switch link {
            case .steerLab(let path, let app) where samePath(app, target.appBundle):
                steps.append(
                    "\(path.path) keeps pointing at \(target.appBundle.path), so it runs "
                        + "SteerLab \(version) from the next command on.")
            case .steerLab(let path, let app):
                steps.append(
                    "\(path.path) runs the copy at \(app.path), which this update does "
                        + "not touch.")
            case .absent(let path), .other(let path):
                steps.append(
                    "\(path.path) does not run this copy; nothing on PATH changes.")
            }
            steps.append(
                "Your workspaces, studies, and models are not touched, and SteerLab is not "
                    + "opened afterwards.")
            return steps
        }

        /// What to bring up to date afterwards, in the app's own words.
        public var afterInstall: [String] {
            [
                UpdateInstallReview.engineLine(
                    newVersion: release.version, sourceCheckout: target.sourceCheckout),
                CommandLineUpdate.clusterStagingLine,
                CommandLineUpdate.identityCheckLine,
            ]
        }
    }

    /// The cluster consequence, named after the checks that report it: the
    /// deployed-payload currency check (`cluster status`, `cluster ensure`)
    /// and the engine-lag warning `remote submit-bundle` prints. Neither
    /// refuses (`ClusterProvisioningOperations.engineLagConsequence`).
    public static let clusterStagingLine =
        "Each cluster site keeps the engine this Mac last pushed. Before staging or "
        + "submitting studies there again, run steerlab-cli cluster push --site <site> "
        + "for that site, then restart its controller. Until then cluster status still "
        + "reads the deployed engine as current (it is the one last pushed) and adds that "
        + "this version's engine-side changes are not running there, and remote "
        + "submit-bundle prints the same warning; neither refuses."

    /// The client identity check, named accurately: it compares this Mac's
    /// compiled command line with the Python client files on this Mac, and
    /// no server or cluster takes part (`ClientIdentityFailure`).
    public static let identityCheckLine =
        "The client identity check (error code clientIdentity) compares a running "
        + "steerlab-cli with the Python client files on this Mac, not with any cluster. A "
        + "command already running when the new copy is put in place keeps the earlier "
        + "build in memory and is refused when it next stages a study; that is why the "
        + "update waits while another steerlab-cli runs from this copy. Run such a "
        + "command again after the update."

    /// The fields the digest binds, in a fixed encoding.
    struct Binding: Codable {
        var schema = "steerlab-cli-update-plan/1"
        var currentVersion: String
        var installLocation: String
        var bundleIdentifier: String
        var teamID: String
        var previousCopy: String
        var previousCopyExists: Bool
        var olderCopiesToTrash: [String]
        var releaseTag: String
        var releaseVersion: String
        var archiveName: String
        var archiveSize: Int64?
        var archiveURL: String
        var checksumName: String
        var checksumURL: String
    }

    static func digest(_ binding: Binding) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = (try? encoder.encode(binding)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// What a plan request concluded.
    public enum PlanAnswer: Sendable, Equatable {
        /// Nothing newer to install.
        case current(thisVersion: SemanticVersion, latest: UpdateLatestRelease?, state: State)
        case plan(Plan)
    }

    /// Look the release up (the one request this makes), and say exactly
    /// what installing it would do. Downloads nothing else and changes
    /// nothing but the check's own record in the app's settings, as Check for
    /// Updates… does.
    public static func plan(_ env: Environment) async throws(Refusal) -> PlanAnswer {
        guard let app = env.app else {
            throw eligibilityRefusal(.notAnAppBundle, app: nil)
        }
        if let offline = env.offlineReason {
            throw Refusal(
                code: "updateCheckUnavailable", state: .degraded,
                reason: "SteerLab did not look for a release: \(offline).",
                repairAction: "Run this outside the launch check.")
        }
        env.preferences?.recordCheck()
        let result = await check(env.makeService(true), within: env.explicitDeadline)
        env.preferences?.recordAnswer(result)

        let available: SemanticVersion
        switch result.availability {
        case .unknown(let reason):
            throw Refusal(
                code: "updateCheckUnavailable", state: .degraded,
                reason: UpdateCheckPolicy.manualReport(for: result.availability),
                repairAction: "Nothing was changed. Try again later, or download the "
                    + "release from the Releases page: \(releasesPage.absoluteString)",
                payload: ["checkError": .string(reason)])
        case .upToDate:
            return .current(thisVersion: env.version, latest: result.latest, state: .upToDate)
        case .newerSourceAvailable:
            return .current(
                thisVersion: env.version, latest: result.latest, state: .newerSourceAvailable)
        case .releaseAvailable(let version, _):
            available = version
        }
        guard let release = result.release else {
            // Seen only as a git tag: GitHub hands out no files for it.
            throw Refusal(
                code: "releaseFilesUnavailable",
                reason: "SteerLab \(available.releaseDescription) is tagged, but its files "
                    + "are not on GitHub's public release page, so SteerLab cannot "
                    + "download them itself.",
                repairAction: "Open the Releases page instead: \(releasesPage.absoluteString)")
        }

        let target: UpdateInstallTarget
        switch UpdateInstallEligibility.evaluate(env.installContext(app.bundle)) {
        case .failure(let refusal): throw eligibilityRefusal(refusal, app: app.bundle)
        case .success(let eligible): target = eligible
        }
        let assets: UpdateAssetPair
        switch UpdateReleaseAssets.select(tag: release.tag, assets: release.assets) {
        case .failure(let error): throw installRefusal(error)
        case .success(let pair): assets = pair
        }

        let installer = UpdateInstaller(
            target: target, currentVersion: env.version,
            environment: env.installEnvironment())
        let previous = installer.previousCopy
        let previousExists = env.fileSystem.exists(previous)
        let older = installer.olderPreviousCopies()
        let binding = Binding(
            currentVersion: env.version.description,
            installLocation: target.appBundle.path, bundleIdentifier: target.bundleIdentifier,
            teamID: target.teamID, previousCopy: previous.path,
            previousCopyExists: previousExists, olderCopiesToTrash: older.map(\.path),
            releaseTag: release.tag, releaseVersion: release.version.description,
            archiveName: assets.archive.name, archiveSize: assets.archive.size,
            archiveURL: assets.archive.downloadURL.absoluteString,
            checksumName: assets.checksum.name,
            checksumURL: assets.checksum.downloadURL.absoluteString)
        return .plan(
            Plan(
                currentVersion: env.version, release: release, assets: assets, target: target,
                previousCopy: previous, previousCopyExists: previousExists,
                olderCopiesToTrash: older, link: linkState(directory: env.linkDirectory),
                blockers: blockers(env, app: target.appBundle, identifier: target.bundleIdentifier)
                    .map(\.reason),
                digest: digest(binding)))
    }

    // MARK: - What stands in the way

    /// The app open (any copy with this bundle identifier, or this copy's
    /// own executable running), then the app's busy check over what is
    /// visible without the app: other processes running out of this copy,
    /// which are its command line most often (`UpdateBusyState`). Everything
    /// else that check reads lives in the app, which is refused outright.
    static func blockers(_ env: Environment, app: URL, identifier: String) -> [Refusal] {
        let inside = env.processesInsideApp(app)
        let ownExecutable = inside.filter { $0.contains("/Contents/MacOS/") }
        var open = env.openApps(identifier)
        if open.isEmpty, !ownExecutable.isEmpty { open = [app.path] }
        var found: [Refusal] = []
        if !open.isEmpty {
            found.append(
                Refusal(
                    code: "appRunning",
                    reason: "SteerLab is open (\(open.joined(separator: ", "))), and an update "
                        + "replaces the app it runs from.",
                    repairAction: "Quit SteerLab, then run the same command again. "
                        + "(Install Update… in the app updates the copy it runs from.)"))
        }
        let busy = UpdateBusyState.from(commandLineProcesses: inside.count - ownExecutable.count)
        if !busy.isIdle {
            found.append(installRefusal(.busy(busy.reasons)))
        }
        return found
    }

    // MARK: - The install (`install update`)

    /// What an install did.
    public struct Receipt: Sendable, Equatable {
        public var plan: Plan
        public var installed: UpdateInstallReceipt
        /// The link, read back after the swap.
        public var link: LinkState
        /// The version the link's command line now reports, when it runs
        /// this copy.
        public var linkedVersion: String?
    }

    /// Re-derive the plan, refuse if its digest is not `expected`, and run
    /// the app's own installer: download, verify, then swap.
    public static func apply(
        _ env: Environment, expected: String,
        progress: @escaping @Sendable (UpdateInstallProgress) -> Void = { _ in }
    ) async throws(Refusal) -> Receipt {
        let plan: Plan
        switch try await self.plan(env) {
        case .current(let version, let latest, _):
            throw Refusal(
                code: "planChanged",
                reason: "No release newer than SteerLab \(version.releaseDescription) is "
                    + "available now"
                    + (latest.map { " (the newest is \($0.version.releaseDescription))" } ?? "")
                    + ", so there is nothing to install.",
                repairAction: "Nothing was changed. Run "
                    + planCommand(app: env.app?.bundle) + " to see what is current.")
        case .plan(let derived):
            plan = derived
        }
        guard plan.digest == expected.lowercased() else {
            throw Refusal(
                code: "planChanged",
                reason: "The update plan changed since it was reviewed (reviewed "
                    + "\(expected.prefix(12)), now \(plan.digest.prefix(12))): the release, "
                    + "its files, or what happens to the copies here is not what was "
                    + "approved.",
                repairAction: "Nothing was changed. Run " + planCommand(app: env.app?.bundle)
                    + ", review the new plan with the researcher, and install it with its "
                    + "own digest.",
                payload: ["planSHA256": .string(plan.digest)])
        }
        if let first = blockers(env, app: plan.target.appBundle, identifier: plan.target.bundleIdentifier).first {
            throw first
        }

        let installer = UpdateInstaller(
            target: plan.target, currentVersion: plan.currentVersion,
            environment: env.installEnvironment())
        let staged: UpdateStagedApp
        do {
            staged = try await installer.prepare(plan.release, progress: progress)
        } catch {
            throw installRefusal(
                error as? UpdateInstallError ?? .downloadFailed(error.localizedDescription))
        }
        // The app's own rule: the last step happens only when nothing is
        // running, so ask again now that the download is done.
        if let first = blockers(env, app: plan.target.appBundle, identifier: plan.target.bundleIdentifier).first {
            installer.discardStaging()
            throw first
        }
        let outcome = await withTerminationDeferred(env.deferSignalsDuringSwap) {
            await installer.installDetached(staged)
        }
        let installed: UpdateInstallReceipt
        switch outcome {
        case .failure(let error): throw installRefusal(error)
        case .success(let receipt): installed = receipt
        }
        let link = linkState(directory: env.linkDirectory)
        var linkedVersion: String?
        if let app = link.app, samePath(app, plan.target.appBundle),
            env.fileSystem.exists(app.appending(path: CommandLineLink.helperPath))
        {
            linkedVersion = env.fileSystem.bundleInfo(at: app).version
        }
        return Receipt(plan: plan, installed: installed, link: link, linkedVersion: linkedVersion)
    }

    /// Run `body` with Ctrl-C, termination, and hang-up ignored, so the
    /// installer's last signature check and the renames that swap the app
    /// cannot be interrupted between two renames. Restored afterwards.
    static func withTerminationDeferred<T: Sendable>(
        _ enabled: Bool, _ body: () async -> T
    ) async -> T {
        guard enabled else { return await body() }
        let signals = [SIGINT, SIGTERM, SIGHUP]
        let previous = signals.map { signal($0, SIG_IGN) }
        defer { for (number, handler) in zip(signals, previous) { signal(number, handler) } }
        return await body()
    }

    // MARK: - Wire shapes

    static func iso(_ date: Date?) -> JSONValue {
        date.map { .string($0.ISO8601Format()) } ?? .null
    }

    static func json(_ latest: UpdateLatestRelease?) -> JSONValue {
        guard let latest else { return .null }
        return .object([
            "version": .string(latest.version.releaseDescription),
            "tag": .string(latest.tag),
            "page": latest.page.map { .string($0.absoluteString) } ?? .null,
            "observedAt": iso(latest.observedAt),
        ])
    }

    static func json(_ link: LinkState, thisCopy: URL?) -> JSONValue {
        var record: [String: JSONValue] = ["path": .string(link.link.path)]
        switch link {
        case .absent: record["state"] = .string("absent")
        case .other: record["state"] = .string("notSteerLab")
        case .steerLab(_, let app):
            record["state"] = .string(samePath(app, thisCopy) ? "linksThisCopy" : "linksAnotherCopy")
            record["app"] = .string(app.path)
        }
        return .object(record)
    }

    /// The keys `install version` adds to its payload.
    public static func payload(_ report: Report) -> [String: JSONValue] {
        var update: [String: JSONValue] = [
            "state": .string(report.state.rawValue),
            "thisVersion": .string(report.thisVersion.releaseDescription),
            "latestRelease": json(report.latest),
            "newerThanThisBuild": report.newerThanThisBuild.map(JSONValue.bool) ?? .null,
            "checkedNow": .bool(report.checkedNow),
            "answeredBy": .string(report.answeredBy),
            "updates": report.updateTarget.map { .string($0.path) } ?? .null,
        ]
        if let reason = report.reason { update["reason"] = .string(reason) }
        if let remote = report.remoteMainRevision { update["remoteMainRevision"] = .string(remote) }
        if let domain = report.preferencesDomain {
            update["settings"] = .object([
                "domain": .string(domain),
                "automaticChecks": report.automaticChecksEnabled.map(JSONValue.bool) ?? .null,
                "lastCheck": iso(report.lastCheck),
                "nextAutomaticCheck": iso(report.nextAutomaticCheck),
                "dismissedVersion": report.dismissedVersion.map(JSONValue.string) ?? .null,
            ])
        } else {
            update["settings"] = .null
        }
        return [
            "update": .object(update),
            "appCopies": .array(
                report.copies.map { copy in
                    .object([
                        "path": .string(copy.path.path),
                        "version": copy.version.map(JSONValue.string) ?? .null,
                        "bundleIdentifier": copy.bundleIdentifier.map(JSONValue.string) ?? .null,
                        "thisCopy": .bool(copy.isThisCopy),
                        "linked": .bool(copy.isLinked),
                        "keptCopy": .bool(copy.isKeptCopy),
                    ])
                }),
            "commandLineLink": json(report.link, thisCopy: report.updateTarget),
        ]
    }

    /// The rows `install version` adds to its human report.
    public static func lines(_ report: Report) -> [String] {
        func row(_ label: String, _ value: String) -> String {
            "  \(label)" + String(repeating: " ", count: max(1, 14 - label.count)) + value
        }
        let this = report.thisVersion.releaseDescription
        var lines: [String] = []
        switch report.state {
        case .releaseAvailable:
            let latest = report.latest?.version.releaseDescription ?? "a newer release"
            lines.append(
                row("update", "SteerLab \(latest) is available; this copy is \(this)"
                    + (report.updateTarget == nil
                        ? "" : ". Review it: steerlab-cli install update-plan")))
        case .newerSourceAvailable:
            lines.append(
                row("update", "no newer release; the repository's main is at "
                    + String((report.remoteMainRevision ?? "").prefix(8))))
        case .upToDate:
            lines.append(row("update", "SteerLab \(this) is the newest release"))
        case .unknown:
            lines.append(row("update", "unknown: \(report.reason ?? "nothing answered")"))
        }
        if report.state != .unknown, let reason = report.reason {
            lines.append(row("", reason))
        }
        if report.answeredBy == "lastCheck", let at = report.latest?.observedAt {
            lines.append(row("", "as of the check at \(at.ISO8601Format())"))
        }
        if let domain = report.preferencesDomain {
            let mode = report.automaticChecksEnabled == false
                ? "automatic checks off" : "automatic, at most once a day"
            lines.append(
                row("checks", "\(mode) (SteerLab's own setting, \(domain)); last asked "
                    + (report.lastCheck?.ISO8601Format() ?? "never")))
        }
        for (index, copy) in report.copies.enumerated() {
            var notes: [String] = []
            if copy.isThisCopy { notes.append("this copy") }
            if copy.isLinked { notes.append("\(report.link.link.path) runs it") }
            if copy.isKeptCopy { notes.append("kept by an earlier update") }
            lines.append(
                row(index == 0 ? "app copies" : "", "\(copy.path.path) "
                    + (copy.version ?? "version not recorded")
                    + (notes.isEmpty ? "" : " (" + notes.joined(separator: "; ") + ")")))
        }
        return lines
    }

    /// The keys `install update-plan` reports.
    public static func payload(_ plan: Plan) -> [String: JSONValue] {
        [
            "updateAvailable": .bool(true),
            "currentVersion": .string(plan.currentVersion.releaseDescription),
            "release": .object([
                "version": .string(plan.release.version.releaseDescription),
                "tag": .string(plan.release.tag),
                "page": plan.release.page.map { .string($0.absoluteString) } ?? .null,
            ]),
            "archive": .object([
                "name": .string(plan.assets.archive.name),
                "bytes": plan.assets.archive.size.map { .number(Double($0)) } ?? .null,
                "url": .string(plan.assets.archive.downloadURL.absoluteString),
            ]),
            "checksum": .object([
                "name": .string(plan.assets.checksum.name),
                "url": .string(plan.assets.checksum.downloadURL.absoluteString),
                "source": .string(
                    "the .sha256 file attached to the same GitHub release; it checks that "
                        + "the download arrived intact, and the code signature checks who "
                        + "made it"),
            ]),
            "signature": .object([
                "teamID": .string(plan.target.teamID),
                "bundleIdentifier": .string(plan.target.bundleIdentifier),
                "notarized": .bool(true),
            ]),
            "installLocation": .string(plan.target.appBundle.path),
            "previousCopy": .object([
                "path": .string(plan.previousCopy.path),
                "alreadyExists": .bool(plan.previousCopyExists),
            ]),
            "olderCopiesToTrash": .array(plan.olderCopiesToTrash.map { .string($0.path) }),
            "commandLineLink": json(plan.link, thisCopy: plan.target.appBundle),
            "steps": .array(plan.steps.map(JSONValue.string)),
            "afterInstall": .array(plan.afterInstall.map(JSONValue.string)),
            "blockers": .array(plan.blockers.map(JSONValue.string)),
            "planSHA256": .string(plan.digest),
            "apply": .array(
                ["steerlab-cli", "install", "update", "--expect", plan.digest, "--yes", "--json"]
                    .map(JSONValue.string)),
        ]
    }

    /// The keys `install update` reports.
    public static func payload(_ receipt: Receipt) -> [String: JSONValue] {
        [
            "installedVersion": .string(receipt.installed.installedVersion.releaseDescription),
            "previousVersion": .string(receipt.plan.currentVersion.releaseDescription),
            "installLocation": .string(receipt.plan.target.appBundle.path),
            "previousCopy": .string(receipt.installed.previousCopy.path),
            "trashedCopies": .array(receipt.installed.trashedCopies.map { .string($0.path) }),
            "commandLineLink": json(receipt.link, thisCopy: receipt.plan.target.appBundle),
            "linkedVersion": receipt.linkedVersion.map(JSONValue.string) ?? .null,
            "afterInstall": .array(receipt.plan.afterInstall.map(JSONValue.string)),
            "planSHA256": .string(receipt.plan.digest),
        ]
    }

    /// One line per step of the install, for a person watching stderr.
    static func progressLine(_ progress: UpdateInstallProgress) -> String {
        switch progress {
        case .downloading(let received, let total):
            guard let total, total > 0 else {
                return "Downloading (\(ByteCountFormatter.string(fromByteCount: received, countStyle: .file)))…"
            }
            return "Downloading: \(Int(Double(received) / Double(total) * 100))% of "
                + ByteCountFormatter.string(fromByteCount: total, countStyle: .file) + "…"
        case .verifyingChecksum: return "Checking the download against its checksum…"
        case .unpacking: return "Unpacking it…"
        case .checkingSignature:
            return "Checking its signature and notarization, then putting it in place…"
        }
    }

    /// One sentence for a finished install.
    public static func sentence(_ receipt: Receipt) -> String {
        let new = receipt.installed.installedVersion.releaseDescription
        let old = receipt.plan.currentVersion.releaseDescription
        var text = "Installed SteerLab \(new) at \(receipt.plan.target.appBundle.path); "
            + "SteerLab \(old) is kept beside it as "
            + "\(receipt.installed.previousCopy.lastPathComponent)."
        if let linked = receipt.linkedVersion {
            text += " \(receipt.link.link.path) now runs \(linked)."
        }
        text += " This command finished on \(old); the next one runs \(new)."
        return text
    }
}

/// Says each step of an install once on stderr, and the download in tenths,
/// rather than once per chunk. `@unchecked Sendable`: its one mutable field
/// is read and written under `lock`.
final class CommandLineUpdateProgress: @unchecked Sendable {
    private let sink: ExperimentCLISink
    private let lock = NSLock()
    private var last = ""

    init(sink: ExperimentCLISink) { self.sink = sink }

    func report(_ progress: UpdateInstallProgress) {
        var line = CommandLineUpdate.progressLine(progress)
        if case .downloading(let received, let total?) = progress, total > 0 {
            let tenth = Int(Double(received) / Double(total) * 10) * 10
            line = "Downloading: \(tenth)% of "
                + ByteCountFormatter.string(fromByteCount: total, countStyle: .file) + "…"
        }
        lock.lock()
        let isNew = line != last
        if isNew { last = line }
        lock.unlock()
        if isNew { sink.err(line + "\n") }
    }
}
