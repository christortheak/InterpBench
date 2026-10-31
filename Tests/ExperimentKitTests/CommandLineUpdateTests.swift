import Foundation
import Synchronization
import Testing

@testable import ExperimentKit

/// Updating SteerLab.app from its own command line (public issue #5,
/// `Sources/ExperimentKit/CommandLineUpdate.swift`).
///
/// Nothing here opens a socket, reads the real preferences, or touches a real
/// app: the release feed is a stub, the settings are a scratch domain, and
/// the install runs over the installer's OWN fakes (`UpdateInstallerTests`),
/// with `zipinfo` and `ditto` real on archives the tests build. The doctrines
/// under test:
///
///   * the app and its command line share one setting and one daily limit;
///   * no verb but `install version` waits on the network, and it asks only
///     while checks are on and due, or when `--check` asks;
///   * "unknown" is said as unknown;
///   * the plan downloads nothing, its digest binds what it says, and the
///     install refuses a plan that moved, an open app, and a busy copy;
///   * the install is the app's own installer, and the link on PATH runs the
///     new version afterwards.
@Suite(.serialized)
struct CommandLineUpdateTests {

    typealias Installer = UpdateInstallerTests

    // MARK: - Seams

    /// A release feed that answers one way and counts how often it is asked.
    final class CountingFeed: UpdateReleaseFeed, @unchecked Sendable {
        let outcome: ReleaseFeedOutcome
        let delay: Duration?
        let asked = Mutex(0)
        init(_ outcome: ReleaseFeedOutcome, delay: Duration? = nil) {
            self.outcome = outcome
            self.delay = delay
        }
        func latestRelease(for repository: UpdateRepository) async -> ReleaseFeedOutcome {
            asked.withLock { $0 += 1 }
            if let delay { try? await Task.sleep(for: delay) }
            return outcome
        }
        var count: Int { asked.withLock { $0 } }
    }

    struct ScriptedGit: UpdateGitProbe {
        let responses: [String: UpdateGitOutcome]
        func git(_ arguments: [String], in checkout: URL) async -> UpdateGitOutcome {
            responses[arguments.joined(separator: " ")] ?? .failed("no scripted answer")
        }
    }

    final class Counter: @unchecked Sendable {
        private let value = Mutex(0)
        func next() -> Int { value.withLock { $0 += 1; return $0 } }
    }

    /// A scratch preferences domain, removed by the caller.
    struct Domain {
        let name = "steerlab.cli-update.tests.\(UUID().uuidString)"
        var defaults: UserDefaults { UserDefaults(suiteName: name)! }
        func remove() { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
    }

    static let repository = UpdateRepository(owner: "acme", name: "widget")
    static let noon = Date(timeIntervalSince1970: 1_800_000_000)

    static func release(_ version: String, assets: [ReleaseAsset] = []) -> ReleaseFeedOutcome {
        .release(tag: "v\(version)", page: repository.releasePage(tag: "v\(version)"), assets: assets)
    }

    /// An environment over the fakes. `app` nil is a command line outside
    /// SteerLab.app.
    static func environment(
        app: URL?, version: String = "0.9.8", domain: Domain?,
        feed: CountingFeed, transport: any UpdateTransport = Installer.FakeTransport([:]),
        signing: any UpdateCodeSigning = Installer.FakeSigning(),
        context: @escaping @Sendable (URL) -> UpdateInstallContext = { Installer.context(app: $0) },
        inside: @escaping @Sendable (URL) -> [String] = { _ in [] },
        open: @escaping @Sendable (String) -> [String] = { _ in [] },
        folders: [URL] = [], linkDirectory: URL,
        checkout: URL? = nil, git: any UpdateGitProbe = ScriptedGit(responses: [:]),
        offline: String? = nil, automaticDeadline: Duration = .seconds(5),
        pinned: String? = nil,
        now: @escaping @Sendable () -> Date = { noon }
    ) -> CommandLineUpdate.Environment {
        let current = SemanticVersion(version)!
        let preferences = domain.map { UpdateCheckPreferences(defaults: $0.defaults, now: now) }
        return CommandLineUpdate.Environment(
            app: app.map {
                .init(bundle: $0, bundleIdentifier: Installer.identifier, version: current)
            },
            version: current, preferences: preferences, preferencesDomain: domain?.name,
            offlineReason: offline, automaticDeadline: automaticDeadline,
            makeService: { _ in
                UpdateCheckService(
                    currentVersion: current, builtRevision: nil, checkout: checkout,
                    feed: feed, git: git, repository: repository)
            },
            installContext: context,
            installEnvironment: {
                UpdateInstallEnvironment(
                    transport: transport, codeSigning: signing,
                    fileSystem: Installer.FaultyFileSystem(),
                    clock: Installer.FakeClock(expired: false), isProcessAlive: { _ in false })
            },
            processesInsideApp: inside, openApps: open,
            copyFolders: folders, linkDirectory: linkDirectory,
            fileSystem: Installer.FaultyFileSystem(), pinnedReleaseTag: pinned, now: now)
    }

    static func link(_ app: URL, in directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: directory.appending(component: CommandLineLink.name),
            withDestinationURL: app.appending(path: CommandLineLink.helperPath))
    }

    // MARK: - One setting, shared with the app

    /// The command line opens the APP's domain by name: what the app turns
    /// off is off here, and a check made here counts against the app's
    /// once-a-day limit.
    @Test func theCommandLineReadsAndWritesTheAppsOwnSettings() throws {
        let domain = Domain()
        defer { domain.remove() }
        // What the app sees: its standard domain IS this suite.
        let app = UpdateCheckPreferences(
            defaults: domain.defaults, now: { Self.noon.addingTimeInterval(3600) })
        let opened = UpdateCheckPreferences.forApp(
            bundleIdentifier: domain.name, processBundleIdentifier: nil, now: { Self.noon })
        let commandLine = try #require(opened)

        #expect(commandLine.automaticChecksEnabled)
        app.automaticChecksEnabled = false
        #expect(!commandLine.automaticChecksEnabled)
        #expect(!commandLine.shouldRunAutomaticCheck)

        // A check the command line makes counts against the app's day.
        app.automaticChecksEnabled = true
        #expect(app.shouldRunAutomaticCheck)
        commandLine.recordCheck()
        #expect(app.lastCheck == Self.noon)
        #expect(!app.shouldRunAutomaticCheck)

        app.dismiss(SemanticVersion("0.9.18")!)
        #expect(commandLine.dismissedVersion == "0.9.18")

        // Nothing that could not name a domain opens one.
        #expect(UpdateCheckPreferences.forApp(bundleIdentifier: "not an identifier") == nil)
        #expect(UpdateCheckPreferences.forApp(bundleIdentifier: "") == nil)
    }

    /// The live environment for a bundled command line names the app's
    /// bundle identifier as its domain, and a launch check an isolated one
    /// with no network.
    @Test func theLiveEnvironmentUsesTheAppsBundleIdentifier() async throws {
        let scratch = try Installer.Scratch()
        defer { scratch.remove() }
        try Installer.makeBundle(at: scratch.installed, version: "0.9.8", marker: "app")
        let helper = scratch.installed.appending(path: CommandLineLink.helperPath)
        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])

        let env = CommandLineUpdate.Environment.live(provenance: provenance)
        #expect(env.app?.bundle.standardizedFileURL == scratch.installed.standardizedFileURL)
        #expect(env.app?.bundleIdentifier == Installer.identifier)
        #expect(env.version == SemanticVersion("0.9.8")!)
        #expect(env.preferencesDomain == Installer.identifier)
        #expect(env.offlineReason == nil)
        #expect(env.deferSignalsDuringSwap)

        let isolated = await LaunchCheckMode.withOverride(true) {
            CommandLineUpdate.Environment.live(provenance: provenance)
        }
        #expect(isolated.preferencesDomain == LaunchCheckMode.isolatedDefaultsSuiteName)
        #expect(isolated.offlineReason != nil)

        // Outside an app there are no shared settings at all.
        let outside = CommandLineUpdate.Environment.live(
            provenance: InstallProvenance.resolve(
                executable: scratch.root.appending(component: "steerlab-cli"), environment: [:]))
        #expect(outside.app == nil)
        #expect(outside.preferences == nil)
    }

    /// What a check learned is recorded for the other process; a check that
    /// reached nothing, or one pinned to a tag, records nothing.
    @Test func onlyAnAnswerIsRecorded() throws {
        let domain = Domain()
        defer { domain.remove() }
        let preferences = UpdateCheckPreferences(defaults: domain.defaults, now: { Self.noon })
        let latest = UpdateLatestRelease(
            version: SemanticVersion("0.9.18")!, tag: "v0.9.18",
            page: Self.repository.releasePage(tag: "v0.9.18"))
        preferences.recordAnswer(
            UpdateCheckResult(
                availability: .upToDate(version: SemanticVersion("0.9.18")!), release: nil,
                latest: latest),
            releaseTagOverride: nil)
        #expect(preferences.latestRelease?.version == SemanticVersion("0.9.18")!)
        #expect(preferences.latestRelease?.observedAt == Self.noon)

        preferences.recordAnswer(
            UpdateCheckResult(availability: .unknown(reason: "offline"), release: nil),
            releaseTagOverride: nil)
        #expect(preferences.latestRelease?.version == SemanticVersion("0.9.18")!)

        preferences.recordAnswer(
            UpdateCheckResult(
                availability: .upToDate(version: SemanticVersion("1.0.0-rc.1")!), release: nil,
                latest: .init(version: SemanticVersion("1.0.0-rc.1")!, tag: "v1.0.0-rc.1", page: nil)),
            releaseTagOverride: "v1.0.0-rc.1")
        #expect(preferences.latestRelease?.tag == "v0.9.18")

        // Garbage in the slot reads as nothing.
        domain.defaults.set(["version": "not a version", "tag": "x"], forKey: UpdateCheckPreferences.latestReleaseKey)
        #expect(preferences.latestRelease == nil)
    }

    /// The newest release is recorded whether or not it is newer than the
    /// copy that asked, so an OLDER copy can compare against it.
    @Test func aCheckReportsTheNewestReleaseEvenWhenUpToDate() async {
        let service = UpdateCheckService(
            currentVersion: SemanticVersion("0.9.18")!, builtRevision: nil, checkout: nil,
            feed: CountingFeed(Self.release("0.9.18")), git: ScriptedGit(responses: [:]),
            repository: Self.repository)
        let result = await service.checkDetailed()
        #expect(result.availability == .upToDate(version: SemanticVersion("0.9.18")!))
        #expect(result.latest?.version == SemanticVersion("0.9.18")!)
        #expect(result.latest?.tag == "v0.9.18")

        // The decision table's precedence: the published release when the
        // feed names one, else the remote's newest tag.
        let tags = ScriptedGit(responses: [
            "ls-remote --tags origin": .output("abc\trefs/tags/v0.9.18\n"),
        ])
        let published = await UpdateCheckService(
            currentVersion: SemanticVersion("0.9.8")!, builtRevision: nil,
            checkout: URL(filePath: "/nowhere"), feed: CountingFeed(Self.release("0.9.17")),
            git: tags, repository: Self.repository
        ).checkDetailed()
        #expect(published.latest?.version == SemanticVersion("0.9.17")!)
        let tagged = await UpdateCheckService(
            currentVersion: SemanticVersion("0.9.8")!, builtRevision: nil,
            checkout: URL(filePath: "/nowhere"), feed: CountingFeed(.unavailable("404")),
            git: tags, repository: Self.repository
        ).checkDetailed()
        #expect(tagged.latest?.version == SemanticVersion("0.9.18")!)
        #expect(tagged.availability == .releaseAvailable(
            version: SemanticVersion("0.9.18")!, page: Self.repository.releasePage(tag: "v0.9.18")))
    }

    /// Run `body` and hand back the refusal it threw, if it threw one.
    static func refusal<T>(
        _ body: () async throws -> T
    ) async -> CommandLineUpdate.Refusal? {
        do {
            _ = try await body()
            return nil
        } catch let refusal as CommandLineUpdate.Refusal {
            return refusal
        } catch {
            Issue.record("not a refusal: \(error)")
            return nil
        }
    }

    // MARK: - The report: when it asks, and what it says

    @Test func aDueCheckAsksOnceRecordsItAndAnnounces() async throws {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))
        let report = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 1)
        #expect(report.checkedNow)
        #expect(report.state == .releaseAvailable)
        #expect(report.newerThanThisBuild == true)
        #expect(report.latest?.version == SemanticVersion("0.9.18")!)
        #expect(report.announces)
        #expect(env.preferences?.lastCheck == Self.noon)
        #expect(env.preferences?.latestRelease?.version == SemanticVersion("0.9.18")!)

        // Within the day: the recorded answer, and no request.
        let again = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 1)
        #expect(!again.checkedNow)
        #expect(again.answeredBy == "lastCheck")
        #expect(again.state == .releaseAvailable)
        #expect(again.announces)

        let advisory = CommandLineUpdate.advisory(latest: try #require(report.latest), env: env)
        #expect(advisory.code == "clientUpdateAvailable")
        #expect(advisory.detail.contains("SteerLab 0.9.18 is available"))
        #expect(advisory.detail.contains("this command line is SteerLab 0.9.8"))
        // This copy's own command line, never a bare `steerlab-cli`.
        #expect(advisory.detail.contains(
            "/opt/lab/SteerLab.app/Contents/Helpers/steerlab-cli install update-plan --json"))
        #expect(!advisory.detail.contains(" steerlab-cli install"))
    }

    // MARK: - A check made by either counts for both, and silences neither

    /// The command line's check is recorded only with its answer, and the
    /// app's launch, with no check of its own due, announces that answer.
    @Test func aCheckFromTheCommandLineAnnouncesInTheApp() async {
        let domain = Domain()
        defer { domain.remove() }
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain,
            feed: CountingFeed(Self.release("0.9.18")), linkDirectory: URL(filePath: "/nowhere/bin"))
        _ = await CommandLineUpdate.report(env, explicitCheck: false)

        // The app, an hour later: not due, and the recorded answer speaks.
        let app = UpdateCheckPreferences(
            defaults: domain.defaults, now: { Self.noon.addingTimeInterval(3600) })
        #expect(!app.shouldRunAutomaticCheck)
        let announced = app.recordedAnnouncement(currentVersion: SemanticVersion("0.9.8")!)
        guard case .releaseAvailable(let version, _)? = announced else {
            Issue.record("the app does not announce what its command line found")
            return
        }
        #expect(version == SemanticVersion("0.9.18")!)
        #expect(app.bannerMessage(for: announced!) == "SteerLab 0.9.18 is available")
        // …by the banner rule: not for a copy already current, not once
        // dismissed, and not with checks off.
        #expect(app.recordedAnnouncement(currentVersion: SemanticVersion("0.9.18")!) == nil)
        app.dismiss(SemanticVersion("0.9.18")!)
        #expect(app.recordedAnnouncement(currentVersion: SemanticVersion("0.9.8")!) == nil)
        app.dismissedVersion = nil
        app.automaticChecksEnabled = false
        #expect(app.recordedAnnouncement(currentVersion: SemanticVersion("0.9.8")!) == nil)
    }

    /// The other direction: what the app recorded is what the command line
    /// reports, without asking again.
    @Test func aCheckFromTheAppAnswersTheCommandLine() async {
        let domain = Domain()
        defer { domain.remove() }
        let app = UpdateCheckPreferences(defaults: domain.defaults, now: { Self.noon })
        app.recordCheck()
        app.recordAnswer(
            UpdateCheckResult(
                availability: .upToDate(version: SemanticVersion("0.9.18")!), release: nil,
                latest: .init(version: SemanticVersion("0.9.18")!, tag: "v0.9.18", page: nil)),
            releaseTagOverride: nil)
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"),
            now: { Self.noon.addingTimeInterval(3600) })
        let report = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 0)
        #expect(report.answeredBy == "lastCheck")
        #expect(report.state == .releaseAvailable)
        #expect(report.announces)
        #expect(report.answerIsStale == false)
        #expect(abs((report.answerAge ?? 0) - 3600) < 1)
    }

    /// A check that reached nothing records nothing: the app's own check
    /// stays due, and nothing stands in for it.
    @Test func aCheckThatReachedNothingDoesNotSilenceTheApp() async {
        let domain = Domain()
        defer { domain.remove() }
        for explicit in [false, true] {
            let env = Self.environment(
                app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain,
                feed: CountingFeed(.unavailable("could not reach the release feed")),
                linkDirectory: URL(filePath: "/nowhere/bin"))
            _ = await CommandLineUpdate.report(env, explicitCheck: explicit)
            #expect(env.preferences?.lastCheck == nil)
            #expect(env.preferences?.latestRelease == nil)
        }
        #expect(UpdateCheckPreferences(defaults: domain.defaults).shouldRunAutomaticCheck)
    }

    /// A check pinned to one tag says so, and records nothing: it is not an
    /// answer about the latest release.
    @Test func aPinnedReleaseIsSaidAndNotRecorded() async throws {
        let domain = Domain()
        defer { domain.remove() }
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain,
            feed: CountingFeed(Self.release("1.0.0-rc.1")), linkDirectory: URL(filePath: "/nowhere/bin"),
            pinned: "v1.0.0-rc.1")
        let report = await CommandLineUpdate.report(env, explicitCheck: true)
        #expect(report.pinnedReleaseTag == "v1.0.0-rc.1")
        #expect(report.state == .releaseAvailable)
        #expect(env.preferences?.lastCheck == nil)
        #expect(env.preferences?.latestRelease == nil)
        #expect(CommandLineUpdate.lines(report).contains {
            $0.contains("asked for the release tagged v1.0.0-rc.1")
                && $0.contains("STEERLAB_UPDATE_RELEASE_TAG")
        })
        guard case .object(let update)? = CommandLineUpdate.payload(report)["update"] else {
            Issue.record("no update record")
            return
        }
        #expect(update["pinnedReleaseTag"] == .string("v1.0.0-rc.1"))

        let stage = try Stage()
        defer { stage.remove() }
        var pinnedEnv = stage.environment()
        pinnedEnv.pinnedReleaseTag = stage.release.tag
        let plan = try await Self.plan(pinnedEnv)
        #expect(plan.steps.first?.contains("tagged \(stage.release.tag)") == true)
        #expect(CommandLineUpdate.payload(plan)["pinnedReleaseTag"] == .string(stage.release.tag))
    }

    /// An answer of any age is reported with its age, and an answer older
    /// than the daily window says it may be out of date.
    @Test func aRecordedAnswerCarriesItsAge() async {
        let domain = Domain()
        defer { domain.remove() }
        let preferences = UpdateCheckPreferences(defaults: domain.defaults)
        preferences.automaticChecksEnabled = false
        preferences.latestRelease = .init(
            version: SemanticVersion("0.9.8")!, tag: "v0.9.8", page: nil,
            observedAt: Self.noon.addingTimeInterval(-3 * 86_400))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain,
            feed: CountingFeed(Self.release("0.9.18")), linkDirectory: URL(filePath: "/nowhere/bin"))
        let report = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(report.state == .upToDate)
        #expect(report.answeredBy == "lastCheck")
        #expect(report.answerIsStale)
        // A date read back from the preferences store is exact to within a
        // second, not to the last bit.
        #expect(abs((report.answerAge ?? 0) - 3 * 86_400) < 1)
        #expect(CommandLineUpdate.lines(report).contains {
            $0.contains("3 days ago") && $0.contains("older than a day")
        })
        guard case .object(let update)? = CommandLineUpdate.payload(report)["update"] else {
            Issue.record("no update record")
            return
        }
        #expect(update["answerIsStale"] == .bool(true))
        #expect(update["answerAgeSeconds"] == .number(3 * 86_400))
    }

    /// The checkout probe never stops to ask on the terminal: git, ssh, and
    /// Git Credential Manager are each told not to prompt.
    @Test func theCheckoutProbeRunsWithPromptsOff() async {
        final class Recording: ProvisionCommandRunner, @unchecked Sendable {
            let argv = Mutex<[String]>([])
            func run(
                _ argv: [String], onLine: @escaping @Sendable (String) -> Void
            ) async throws -> Int32 {
                self.argv.withLock { $0 = argv }
                onLine("abc\trefs/heads/main")
                return 0
            }
        }
        let runner = Recording()
        let probe = SystemUpdateGitProbe(
            runner: runner, environment: SystemUpdateGitProbe.nonInteractiveEnvironment)
        let checkout = URL(filePath: "/opt/lab/InterpBench")
        _ = await probe.git(["ls-remote", "origin", "main"], in: checkout)
        #expect(
            runner.argv.withLock { $0 } == [
                "/usr/bin/env", "GCM_INTERACTIVE=never",
                "GIT_SSH_COMMAND=ssh -oBatchMode=yes", "GIT_TERMINAL_PROMPT=0",
                "/usr/bin/git", "-C", "/opt/lab/InterpBench", "ls-remote", "origin", "main",
            ])
        // The app's own probe is unchanged: git, with no environment added.
        #expect(
            SystemUpdateGitProbe.argv(["remote", "get-url", "origin"], checkout: checkout, environment: [:])
                == ["/usr/bin/git", "-C", "/opt/lab/InterpBench", "remote", "get-url", "origin"])
    }

    /// Checks turned off in the app: no request, however stale; the recorded
    /// answer is still reported, but not announced. `--check` asks once, as
    /// Check for Updates… does, and records it.
    @Test func checksTurnedOffInTheAppAreOffHere() async {
        let domain = Domain()
        defer { domain.remove() }
        let preferences = UpdateCheckPreferences(defaults: domain.defaults)
        preferences.automaticChecksEnabled = false
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))

        let quiet = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 0)
        #expect(quiet.state == .unknown)
        #expect(quiet.reason?.contains("automatic update checks are off") == true)
        #expect(!quiet.announces)

        let asked = await CommandLineUpdate.report(env, explicitCheck: true)
        #expect(feed.count == 1)
        #expect(asked.state == .releaseAvailable)
        #expect(asked.announces)
        #expect(preferences.lastCheck == Self.noon)
        #expect(!preferences.automaticChecksEnabled)

        // Recorded now, so reported, and still not announced unasked.
        let after = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 1)
        #expect(after.state == .releaseAvailable)
        #expect(!after.announces)
        #expect(CommandLineUpdate.ambientAdvisories(env).isEmpty)
    }

    /// The banner rule: a release the researcher dismissed in the app is not
    /// announced unasked, and is when they ask.
    @Test func aDismissedReleaseIsAnnouncedOnlyWhenAsked() async {
        let domain = Domain()
        defer { domain.remove() }
        let preferences = UpdateCheckPreferences(defaults: domain.defaults)
        preferences.dismiss(SemanticVersion("0.9.18")!)
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain,
            feed: CountingFeed(Self.release("0.9.18")), linkDirectory: URL(filePath: "/nowhere/bin"))
        let unasked = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(unasked.state == .releaseAvailable)
        #expect(!unasked.announces)
        #expect(CommandLineUpdate.ambientAdvisories(env).isEmpty)
        let asked = await CommandLineUpdate.report(env, explicitCheck: true)
        #expect(asked.announces)
    }

    /// Unreachable is unknown, never "up to date", and the last answer stays.
    @Test func anUnreachableFeedIsUnknownAndKeepsTheLastAnswer() async {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(.unavailable("could not reach the release feed"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))
        let first = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(feed.count == 1)
        #expect(first.state == .unknown)
        #expect(first.newerThanThisBuild == nil)
        #expect(first.reason?.contains("could not reach the release feed") == true)
        #expect(!first.announces)
        #expect(env.preferences?.lastCheck == nil)

        env.preferences?.latestRelease = .init(
            version: SemanticVersion("0.9.18")!, tag: "v0.9.18", page: nil,
            observedAt: Self.noon.addingTimeInterval(-86_400 * 3))
        let second = await CommandLineUpdate.report(env, explicitCheck: true)
        #expect(second.state == .releaseAvailable)
        #expect(second.answeredBy == "lastCheck")
        #expect(second.reason?.contains("could not finish") == true)
        #expect(second.answerIsStale)
    }

    /// The automatic refresh has a short deadline; running out of it is the
    /// same fact as reaching nothing.
    @Test func aSlowFeedRunsOutOfTimeAsUnknown() async {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(Self.release("0.9.18"), delay: .seconds(30))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"), automaticDeadline: .milliseconds(100))
        let clock = ContinuousClock()
        let started = clock.now
        let report = await CommandLineUpdate.report(env, explicitCheck: false)
        #expect(clock.now - started < .seconds(10))
        #expect(report.state == .unknown)
        #expect(report.reason?.contains("did not answer within") == true)
        #expect(env.preferences?.lastCheck == nil)
    }

    /// A build script's launch check, and a command line outside the app,
    /// ask nothing on their own.
    @Test func offlineAndUnbundledCommandLinesDoNotAskOnTheirOwn() async {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(Self.release("0.9.18"))
        let offline = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"), offline: "this is a launch check")
        for explicit in [false, true] {
            let report = await CommandLineUpdate.report(offline, explicitCheck: explicit)
            #expect(report.state == .unknown)
        }
        #expect(feed.count == 0)
        #expect(offline.preferences?.lastCheck == nil)

        let outside = Self.environment(
            app: nil, domain: nil, feed: feed, linkDirectory: URL(filePath: "/nowhere/bin"))
        let quiet = await CommandLineUpdate.report(outside, explicitCheck: false)
        #expect(feed.count == 0)
        #expect(quiet.state == .unknown)
        #expect(quiet.reason?.contains("not part of SteerLab.app") == true)
        let asked = await CommandLineUpdate.report(outside, explicitCheck: true)
        #expect(feed.count == 1)
        #expect(asked.state == .releaseAvailable)
        #expect(asked.announces)
        #expect(asked.updateTarget == nil)
    }

    // MARK: - Copies, and the link on PATH

    @Test func everyCopyIsListedWithTheOneTheLinkRuns() throws {
        let scratch = try Installer.Scratch()
        defer { scratch.remove() }
        try Installer.makeBundle(at: scratch.installed, version: "0.9.8", marker: "this")
        let home = scratch.root.appending(component: "Home")
        let other = home.appending(component: "SteerLab.app")
        try Installer.makeBundle(at: other, version: "0.9.18", marker: "other")
        try Installer.makeBundle(
            at: scratch.apps.appending(component: "SteerLab-0.9.7-previous.app"),
            version: "0.9.7", marker: "kept")
        // A different app with a similar name: listed, and marked foreign.
        try Installer.makeBundle(
            at: home.appending(component: "SteerLab Tools.app"), version: "2.0",
            identifier: "org.example.SomethingElse", marker: "foreign")
        // Not copies: a folder with no Info.plist, and something else.
        try FileManager.default.createDirectory(
            at: home.appending(component: "SteerLab Notes.app"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: home.appending(component: "Workspaces"), withIntermediateDirectories: true)
        let bin = scratch.root.appending(component: "bin")
        try Self.link(other, in: bin)

        let env = Self.environment(
            app: scratch.installed, domain: nil, feed: CountingFeed(.noReleases),
            folders: [scratch.apps, home, scratch.apps], linkDirectory: bin)
        let link = CommandLineUpdate.linkState(directory: bin)
        #expect(CommandLineUpdate.samePath(link.app, other))
        let copies = CommandLineUpdate.copies(env, link: link)
        #expect(
            copies.map(\.path.lastPathComponent)
                == ["SteerLab.app", "SteerLab-0.9.7-previous.app", "SteerLab Tools.app",
                    "SteerLab.app"])
        #expect(copies[0].isThisCopy && !copies[0].isLinked && copies[0].version == "0.9.8")
        #expect(copies[1].isKeptCopy && !copies[1].isThisCopy)
        #expect(copies[2].isForeign && copies[2].bundleIdentifier == "org.example.SomethingElse")
        #expect(copies[3].isLinked && !copies[3].isThisCopy && copies[3].version == "0.9.18")
        #expect(copies.filter(\.isForeign).count == 1)

        // The report says which copy an update replaces, and the link.
        let payload = CommandLineUpdate.payload(
            CommandLineUpdate.Report(
                thisVersion: env.version, state: .unknown, latest: nil, newerThanThisBuild: nil,
                checkedNow: false, answeredBy: "nothing", reason: "x", remoteMainRevision: nil,
                automaticChecksEnabled: nil, lastCheck: nil, nextAutomaticCheck: nil,
                dismissedVersion: nil, preferencesDomain: nil, announces: false,
                copies: copies, link: link, updateTarget: scratch.installed))
        guard case .object(let linkRecord)? = payload["commandLineLink"] else {
            Issue.record("no link record")
            return
        }
        #expect(linkRecord["state"] == .string("linksAnotherCopy"))
        guard case .array(let listed)? = payload["appCopies"] else {
            Issue.record("no copies")
            return
        }
        #expect(listed.count == 4)
        guard case .object(let foreign) = listed[2] else {
            Issue.record("no foreign copy record")
            return
        }
        #expect(foreign["foreign"] == .bool(true))
    }

    @Test func theStagingOwnerCheckRecognizesTheCommandLineToo() {
        let app = URL(filePath: "/opt/lab/SteerLab.app/Contents/MacOS/SteerLabApp")
        let helper = URL(filePath: "/opt/lab/SteerLab.app/Contents/Helpers/steerlab-cli")
        for own in [app, helper, nil] {
            #expect(UpdateStaging.isSteerLabExecutable(path: app.path, ownExecutable: own))
            #expect(UpdateStaging.isSteerLabExecutable(path: helper.path, ownExecutable: own))
            #expect(!UpdateStaging.isSteerLabExecutable(path: "/usr/bin/git", ownExecutable: own))
        }
    }

    // MARK: - The plan

    /// A scratch install of 0.9.8, a link to it, and a 0.9.9 release.
    struct Stage {
        let scratch: Installer.Scratch
        let release: PublishedRelease
        let transport: Installer.FakeTransport
        let bin: URL
        func remove() { scratch.remove() }

        init() throws {
            scratch = try Installer.Scratch()
            try Installer.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
            let made = try Installer.release(in: scratch)
            release = made.0
            transport = made.1
            bin = scratch.root.appending(component: "bin")
            try CommandLineUpdateTests.link(scratch.installed, in: bin)
        }

        func environment(
            domain: CommandLineUpdateTests.Domain? = nil,
            feed: ReleaseFeedOutcome? = nil,
            signing: any UpdateCodeSigning = Installer.FakeSigning(),
            context: (@Sendable (URL) -> UpdateInstallContext)? = nil,
            inside: @escaping @Sendable (URL) -> [String] = { _ in [] },
            open: @escaping @Sendable (String) -> [String] = { _ in [] }
        ) -> CommandLineUpdate.Environment {
            CommandLineUpdateTests.environment(
                app: scratch.installed, domain: domain,
                feed: CountingFeed(
                    feed ?? .release(tag: release.tag, page: nil, assets: release.assets)),
                transport: transport, signing: signing,
                context: context ?? { Installer.context(app: $0) },
                inside: inside, open: open, folders: [scratch.apps], linkDirectory: bin)
        }
    }

    static func plan(_ env: CommandLineUpdate.Environment) async throws -> CommandLineUpdate.Plan {
        guard case .plan(let plan) = try await CommandLineUpdate.plan(env) else {
            throw CommandLineUpdate.Refusal(code: "test", reason: "no plan", repairAction: "")
        }
        return plan
    }

    @Test func thePlanSaysWhatWillHappenAndDownloadsNothing() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        try Installer.makeBundle(
            at: stage.scratch.apps.appending(component: "SteerLab-0.9.7-previous.app"),
            version: "0.9.7", marker: "older")
        let domain = Domain()
        defer { domain.remove() }
        let env = stage.environment(domain: domain)
        let plan = try await Self.plan(env)

        // Only the release lookup: no file was fetched, nothing moved.
        #expect(stage.transport.opened.withLock { $0 }.isEmpty)
        #expect(LiveUpdateFileSystem().bundleInfo(at: stage.scratch.installed).version == "0.9.8")
        #expect(env.preferences?.lastCheck == Self.noon)

        #expect(plan.release.tag == "v0.9.9")
        #expect(plan.assets.archive.name == "SteerLab-0.9.9+e6bd5258.zip")
        #expect(plan.assets.checksum.name == "SteerLab-0.9.9+e6bd5258.zip.sha256")
        #expect(plan.target.appBundle == stage.scratch.installed)
        #expect(plan.previousCopy.lastPathComponent == "SteerLab-0.9.8-previous.app")
        #expect(!plan.previousCopyExists)
        #expect(plan.olderCopiesToTrash.map(\.lastPathComponent) == ["SteerLab-0.9.7-previous.app"])
        #expect(plan.digest.count == 64)
        #expect(plan.blockers.isEmpty)

        let steps = plan.steps.joined(separator: "\n")
        #expect(steps.contains("SteerLab-0.9.9+e6bd5258.zip"))
        #expect(steps.contains(".sha256"))
        #expect(steps.contains("notarization"))
        #expect(steps.contains(Installer.team))
        #expect(steps.contains("SteerLab-0.9.8-previous.app"))
        #expect(steps.contains("keeps pointing at"))
        let after = plan.afterInstall.joined(separator: "\n")
        #expect(after.contains("cluster push --site <site>"))
        #expect(after.contains("clientIdentity"))

        let payload = CommandLineUpdate.payload(plan)
        #expect(payload["planSHA256"] == .string(plan.digest))
        // This copy's own command line by its full path, never a bare
        // `steerlab-cli`.
        let helper = stage.scratch.installed.appending(path: CommandLineLink.helperPath).path
        #expect(
            payload["apply"]
                == .array(
                    [helper, "install", "update", "--expect", plan.digest, "--yes", "--json"]
                        .map(JSONValue.string)))
        #expect(payload["installLocation"] == .string(stage.scratch.installed.path))
    }

    /// The digest binds the release, its files, the install location, and
    /// what happens to the copies here; what merely stands in the way now
    /// does not move it.
    @Test func theDigestBindsWhatThePlanSays() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let first = try await Self.plan(stage.environment())
        let again = try await Self.plan(stage.environment())
        #expect(first.digest == again.digest)

        let blocked = try await Self.plan(stage.environment(open: { _ in ["/Applications/SteerLab.app"] }))
        #expect(blocked.digest == first.digest)
        #expect(blocked.blockers.first?.contains("SteerLab is open") == true)

        try Installer.makeBundle(
            at: stage.scratch.apps.appending(component: "SteerLab-0.9.8-previous.app"),
            version: "0.9.8", marker: "kept")
        let withKept = try await Self.plan(stage.environment())
        #expect(withKept.previousCopyExists)
        #expect(withKept.digest != first.digest)

        var assets = stage.release.assets
        assets[0] = ReleaseAsset(
            name: assets[0].name, downloadURL: assets[0].downloadURL,
            size: (assets[0].size ?? 0) + 1)
        let resized = try await Self.plan(
            stage.environment(feed: .release(tag: stage.release.tag, page: nil, assets: assets)))
        #expect(resized.digest != withKept.digest)
    }

    @Test func thePlanRefusesWithTheInstallersOwnExplanation() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let cases: [(RunningAppSignature, UpdateInstallRefusal)] = [
            (.developerID(teamID: Installer.team, notarized: false), .notNotarized),
            (.adHoc, .adHocSignature),
        ]
        for (signature, refusal) in cases {
            let env = stage.environment(
                context: { Installer.context(app: $0, signature: signature) })
            let error = await Self.refusal { try await CommandLineUpdate.plan(env) }
            #expect(error?.code == "updateNotEligible")
            #expect(error?.reason == refusal.line)
            #expect(error?.repairAction.contains("Releases page") == true)
        }

        // A kept copy's command line points at the current app's.
        let kept = stage.scratch.apps.appending(component: "SteerLab-0.9.8-previous.app")
        let keptEnv = stage.environment(context: { _ in Installer.context(app: kept) })
        let keptError = await Self.refusal { try await CommandLineUpdate.plan(keptEnv) }
        #expect(keptError?.code == "updateNotEligible")
        #expect(keptError?.reason == UpdateInstallRefusal.runningKeptCopy("0.9.8").line)
        #expect(
            keptError?.repairAction.contains("SteerLab.app/Contents/Helpers/steerlab-cli") == true)

        // Outside an app: the build-folder line.
        let outside = Self.environment(
            app: nil, domain: nil, feed: CountingFeed(Self.release("0.9.9")),
            linkDirectory: stage.bin)
        let outsideError = await Self.refusal { try await CommandLineUpdate.plan(outside) }
        #expect(outsideError?.reason == UpdateInstallRefusal.notAnAppBundle.line)
    }

    @Test func thePlanSaysUnknownCurrentAndTagOnlyHonestly() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let offline = stage.environment(feed: .unavailable("could not reach the release feed"))
        let unreachable = await Self.refusal { try await CommandLineUpdate.plan(offline) }
        #expect(unreachable?.code == "updateCheckUnavailable")
        #expect(unreachable?.state == .degraded)
        #expect(
            unreachable?.reason == "Could not check for updates: could not reach the release feed.")

        let current = try await CommandLineUpdate.plan(stage.environment(feed: Self.release("0.9.8")))
        guard case .current(let version, let latest, let state) = current else {
            Issue.record("a plan for the version already installed")
            return
        }
        #expect(version == SemanticVersion("0.9.8")!)
        #expect(latest?.version == SemanticVersion("0.9.8")!)
        #expect(state == .upToDate)

        // Newer only as a git tag: there are no public files to install.
        let tagOnly = Self.environment(
            app: stage.scratch.installed, domain: nil, feed: CountingFeed(.noReleases),
            linkDirectory: stage.bin, checkout: URL(filePath: "/nowhere/checkout"),
            git: ScriptedGit(responses: [
                "ls-remote --tags origin": .output("abc\trefs/tags/v0.9.9\n"),
            ]))
        let tagError = await Self.refusal { try await CommandLineUpdate.plan(tagOnly) }
        #expect(tagError?.code == "releaseFilesUnavailable")

        // A release with no checksum file is the installer's refusal.
        let noChecksum = stage.environment(
            feed: .release(
                tag: "v0.9.9", page: nil,
                assets: stage.release.assets.filter { !$0.name.hasSuffix(".sha256") }))
        let checksumError = await Self.refusal { try await CommandLineUpdate.plan(noChecksum) }
        #expect(checksumError?.code == "releaseFilesUnavailable")
        #expect(checksumError?.reason.contains(".sha256") == true)
    }

    // MARK: - The install

    @Test func theInstallReplacesThisCopyAndTheLinkRunsTheNewVersion() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let env = stage.environment()
        let plan = try await Self.plan(env)
        let steps = Mutex<[UpdateInstallProgress]>([])
        let receipt = try await CommandLineUpdate.apply(
            env, expected: plan.digest, progress: { step in steps.withLock { $0.append(step) } })

        let fs = LiveUpdateFileSystem()
        #expect(receipt.installed.installedVersion == SemanticVersion("0.9.9")!)
        #expect(fs.bundleInfo(at: stage.scratch.installed).version == "0.9.9")
        #expect(fs.bundleInfo(at: receipt.installed.previousCopy).version == "0.9.8")
        #expect(receipt.installed.previousCopy.lastPathComponent == "SteerLab-0.9.8-previous.app")
        // The link names the helper by path, so it runs the new copy now.
        #expect(receipt.linkedVersion == "0.9.9")
        let linked = stage.bin.appending(component: CommandLineLink.name)
        #expect(try String(contentsOf: linked, encoding: .utf8) == "new 0.9.9")
        #expect(!FileManager.default.fileExists(
            atPath: UpdateStagingLayout.stagingRoot(forApp: stage.scratch.installed).path))
        #expect(steps.withLock { $0 }.contains(.checkingSignature))

        let sentence = CommandLineUpdate.sentence(receipt)
        #expect(sentence.contains("Installed SteerLab 0.9.9"))
        #expect(sentence.contains("SteerLab-0.9.8-previous.app"))
        #expect(sentence.contains("the next one runs 0.9.9"))
    }

    @Test func theInstallRefusesAPlanThatMoved() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let env = stage.environment()
        let plan = try await Self.plan(env)
        let stale = String(plan.digest.reversed())
        let error = await Self.refusal { try await CommandLineUpdate.apply(env, expected: stale) }
        #expect(error?.code == "planChanged")
        // The new digest is never handed back: a changed plan goes to the
        // researcher again, through a fresh plan.
        #expect(error?.payload["planSHA256"] == nil)
        #expect(error?.payload.isEmpty == true)
        #expect(error?.reason.contains(plan.digest) == false)
        #expect(error?.reason.contains(plan.digest.prefix(12)) == false)
        #expect(error?.repairAction.contains(plan.digest.prefix(12)) == false)
        #expect(
            error?.repairAction.contains(
                stage.scratch.installed.appending(path: CommandLineLink.helperPath).path
                    + " install update-plan --json") == true)
        #expect(error?.repairAction.contains("show the new plan to the researcher") == true)
        #expect(stage.transport.opened.withLock { $0 }.isEmpty)
        #expect(LiveUpdateFileSystem().bundleInfo(at: stage.scratch.installed).version == "0.9.8")
    }

    /// The link runs ANOTHER copy: every command the plan and the advisory
    /// print names this copy's own command line, which is the one updated.
    @Test func everyCommandNamesThisCopyWhenTheLinkRunsAnother() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let other = stage.scratch.root.appending(components: "Elsewhere", "SteerLab.app")
        try Installer.makeBundle(at: other, version: "0.9.18", marker: "other")
        try FileManager.default.removeItem(at: stage.bin.appending(component: CommandLineLink.name))
        try Self.link(other, in: stage.bin)

        let env = stage.environment()
        let helper = stage.scratch.installed.appending(path: CommandLineLink.helperPath).path
        let plan = try await Self.plan(env)
        guard case .array(let apply)? = CommandLineUpdate.payload(plan)["apply"] else {
            Issue.record("no apply command")
            return
        }
        #expect(apply.first == .string(helper))
        #expect(plan.applyCommand.hasPrefix(CommandLineLink.shellQuoted(helper) + " install update"))
        guard case .object(let link)? = CommandLineUpdate.payload(plan)["commandLineLink"] else {
            Issue.record("no link record")
            return
        }
        #expect(link["state"] == .string("linksAnotherCopy"))
        #expect(plan.steps.contains { $0.contains("by its full path: \(helper)") })
        #expect(plan.afterInstall.contains { $0.contains("\(helper) cluster push --site <site>") })

        let advisory = CommandLineUpdate.advisory(
            latest: .init(version: SemanticVersion("0.9.9")!, tag: "v0.9.9", page: nil), env: env)
        #expect(advisory.detail.contains("\(helper) install update-plan --json"))

        let report = await CommandLineUpdate.report(env, explicitCheck: true)
        guard case .object(let update)? = CommandLineUpdate.payload(report)["update"] else {
            Issue.record("no update record")
            return
        }
        #expect(
            update["planCommand"]
                == .array([helper, "install", "update-plan", "--json"].map(JSONValue.string)))
        #expect(CommandLineUpdate.lines(report).first?.contains("Review it: \(helper)") == true)
    }

    /// Outside an app, the advice is the route that works: rebuild, or the
    /// Releases page, never a plan this command line would refuse.
    @Test func outsideAnAppTheAdviceIsTheRebuildOrTheReleasesPage() async {
        let env = Self.environment(
            app: nil, domain: nil, feed: CountingFeed(Self.release("0.9.18")),
            linkDirectory: URL(filePath: "/nowhere/bin"))
        let report = await CommandLineUpdate.report(env, explicitCheck: true)
        #expect(report.announces)
        let advisory = CommandLineUpdate.advisory(latest: report.latest!, env: env)
        #expect(!advisory.detail.contains("update-plan"))
        #expect(advisory.detail.contains("rebuild"))
        #expect(advisory.detail.contains("Releases page"))
        #expect(CommandLineUpdate.lines(report).first?.contains("update-plan") == false)
    }

    @Test func theInstallRefusesWhileTheAppIsOpenOrThisCopyIsBusy() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let digest = try await Self.plan(stage.environment()).digest
        let app = stage.scratch.installed.path

        let refusals: [(CommandLineUpdate.Environment, String, String)] = [
            (stage.environment(open: { _ in ["/Applications/SteerLab.app"] }),
             "appRunning", "SteerLab is open"),
            (stage.environment(inside: { _ in [app + "/Contents/MacOS/SteerLabApp"] }),
             "appRunning", "SteerLab is open"),
            (stage.environment(inside: { _ in [app + "/Contents/Helpers/steerlab-cli"] }),
             "updateBusy", "this app's command line (steerlab-cli) is running"),
        ]
        for (env, code, words) in refusals {
            let error = await Self.refusal {
                try await CommandLineUpdate.apply(env, expected: digest)
            }
            #expect(error?.code == code)
            #expect(error?.reason.contains(words) == true, "\(error?.reason ?? "")")
            #expect(error?.repairAction.isEmpty == false)
        }
        #expect(stage.transport.opened.withLock { $0 }.isEmpty)
        #expect(LiveUpdateFileSystem().bundleInfo(at: stage.scratch.installed).version == "0.9.8")
    }

    /// The app's own rule: asked again once the download is done, and a
    /// command started meanwhile stops the swap with nothing replaced.
    @Test func somethingStartedDuringTheDownloadStopsTheSwap() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let digest = try await Self.plan(stage.environment()).digest
        let calls = Counter()
        let helper = stage.scratch.installed.path + "/Contents/Helpers/steerlab-cli"
        // The plan inside apply, then the check before the download: idle.
        let env = stage.environment(inside: { _ in calls.next() <= 2 ? [] : [helper] })
        let error = await Self.refusal { try await CommandLineUpdate.apply(env, expected: digest) }
        #expect(error?.code == "updateBusy")
        #expect(!stage.transport.opened.withLock { $0 }.isEmpty)
        #expect(LiveUpdateFileSystem().bundleInfo(at: stage.scratch.installed).version == "0.9.8")
        #expect(!FileManager.default.fileExists(
            atPath: UpdateStagingLayout.stagingRoot(forApp: stage.scratch.installed).path))
    }

    @Test func aDownloadThatFailsItsSignatureCheckChangesNothing() async throws {
        let stage = try Stage()
        defer { stage.remove() }
        let env = stage.environment(signing: Installer.FakeSigning(accept: false))
        let digest = try await Self.plan(env).digest
        let error = await Self.refusal { try await CommandLineUpdate.apply(env, expected: digest) }
        #expect(error?.code == "updateRejected")
        #expect(error?.reason.contains("signature check") == true)
        #expect(LiveUpdateFileSystem().bundleInfo(at: stage.scratch.installed).version == "0.9.8")
        #expect(!FileManager.default.fileExists(
            atPath: stage.scratch.apps.appending(component: "SteerLab-0.9.8-previous.app").path))
    }

    // MARK: - Through the command line

    private func withTemporaryRoot<T>(_ body: () async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "cli-update-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        return try await body()
    }

    @Test func versionReportsTheUpdateAndPointsAtThePlan() async throws {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))
        let recorder = ExperimentCLIRecorder()
        let outcome = await withTemporaryRoot {
            await ExperimentCLIRunner.$commandLineUpdateOverride.withValue(env) {
                await ExperimentCLIRunner(sink: recorder.sink)
                    .run(namespace: "install", ["version", "--json"])
            }
        }
        #expect(outcome.envelope.exitCode == 0)
        #expect(outcome.envelope.state == .okWithAdvisories)
        #expect(outcome.envelope.advisories?.map(\.code) == ["clientUpdateAvailable"])
        #expect(outcome.envelope.nextAction?.verb == "install update-plan")
        let result = try #require(outcome.envelope.result)
        guard case .object(let update)? = result["update"] else {
            Issue.record("no update record")
            return
        }
        #expect(update["state"] == .string("releaseAvailable"))
        #expect(update["thisVersion"] == .string("0.9.8"))
        #expect(update["newerThanThisBuild"] == .bool(true))
        #expect(update["checkedNow"] == .bool(true))
        #expect(update["updates"] == .string("/opt/lab/SteerLab.app"))
        guard case .object(let settings)? = update["settings"] else {
            Issue.record("no settings record")
            return
        }
        #expect(settings["domain"] == .string(domain.name))
        #expect(settings["automaticChecks"] == .bool(true))
        // The keys the version report always carried are still there.
        #expect(result["layout"] != nil && result["version"] != nil)
        #expect(recorder.standardOutput.contains("SteerLab 0.9.18 is available"))
        #expect(feed.count == 1)
    }

    /// The advisory rides on the session-start verbs from the recorded
    /// answer, never from the network, and not on every verb.
    @Test func theAdvisoryRidesOnSessionStartVerbsFromTheRecordedAnswerOnly() async throws {
        let domain = Domain()
        defer { domain.remove() }
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: domain, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))
        // Due, but these verbs never ask: nothing recorded, nothing said.
        let before = await withTemporaryRoot {
            await ExperimentCLIRunner.$commandLineUpdateOverride.withValue(env) {
                await ExperimentCLIRunner(sink: .discarding)
                    .run(namespace: "workspace", ["guide", "--json"])
            }
        }
        #expect(before.envelope.advisories == nil)
        #expect(feed.count == 0)

        env.preferences?.recordCheck()
        env.preferences?.latestRelease = .init(
            version: SemanticVersion("0.9.18")!, tag: "v0.9.18", page: nil, observedAt: Self.noon)
        let (guide, other) = await withTemporaryRoot {
            await ExperimentCLIRunner.$commandLineUpdateOverride.withValue(env) {
                let runner = ExperimentCLIRunner(sink: .discarding)
                return (
                    await runner.run(namespace: "workspace", ["guide", "verbs", "--json"]),
                    await runner.run(namespace: "science", ["list", "--brief", "--json"]))
            }
        }
        #expect(guide.envelope.exitCode == 0)
        #expect(guide.envelope.state == .okWithAdvisories)
        #expect(guide.envelope.advisories?.map(\.code) == ["clientUpdateAvailable"])
        #expect(guide.envelope.advisories?.first?.detail.contains("0.9.8") == true)
        #expect(other.envelope.advisories?.contains { $0.code == "clientUpdateAvailable" } != true)
        #expect(feed.count == 0)
        #expect(ExperimentCLIRunner.carriesClientUpdateAdvisory(namespace: "setup", verb: "inspect"))
        #expect(!ExperimentCLIRunner.carriesClientUpdateAdvisory(namespace: "experiment", verb: "list"))
    }

    @Test func theInstallVerbNeedsTheDigestAndTheApproval() async throws {
        let feed = CountingFeed(Self.release("0.9.18"))
        let env = Self.environment(
            app: URL(filePath: "/opt/lab/SteerLab.app"), domain: nil, feed: feed,
            linkDirectory: URL(filePath: "/nowhere/bin"))
        let digest = String(repeating: "a", count: 64)
        let (noYes, noDigest, plan) = await withTemporaryRoot {
            await ExperimentCLIRunner.$commandLineUpdateOverride.withValue(env) {
                let runner = ExperimentCLIRunner(sink: .discarding)
                return (
                    await runner.run(namespace: "install", ["update", "--expect", digest, "--json"]),
                    await runner.run(namespace: "install", ["update", "--yes", "--json"]),
                    await runner.run(namespace: "install", ["update-plan", "--json"]))
            }
        }
        #expect(noYes.envelope.state == .needsApproval)
        #expect(noYes.envelope.exitCode == 11)
        #expect(noYes.envelope.error?.code == "approvalRequired")
        #expect(noDigest.envelope.exitCode == 64)
        #expect(noDigest.envelope.error?.code == "usage")
        // A release that lists no app file: the installer's own refusal,
        // typed, with a repair, and only the lookup was made.
        #expect(plan.envelope.exitCode == 65)
        #expect(plan.envelope.error?.code == "releaseFilesUnavailable")
        #expect(plan.envelope.error?.repairAction.isEmpty == false)
        #expect(feed.count == 1)
    }
}
