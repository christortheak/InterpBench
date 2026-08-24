import Foundation
import Testing

@testable import ExperimentKit

/// The update SIGNPOST (`Sources/ExperimentKit/UpdateCheck.swift`).
///
/// Everything here is headless: the release feed and the git remote are
/// injected seams, so no test opens a socket, spawns `git`, or reads the real
/// `UserDefaults`. The doctrines under test are the ones a signpost can get
/// wrong in a way a person would feel:
///
///   * "unknown" is never dressed up as "up to date";
///   * a build-metadata suffix (`0.9.1+f1480d51`) is not a newer version;
///   * a prerelease never outranks the release it precedes;
///   * a dismissed release never banners again;
///   * the automatic check runs at most once a day, on an INJECTED clock.
struct UpdateCheckTests {

    // MARK: - Seams

    private struct StubFeed: UpdateReleaseFeed {
        let outcome: ReleaseFeedOutcome
        func latestRelease(for repository: UpdateRepository) async -> ReleaseFeedOutcome {
            outcome
        }
    }

    /// Replays canned `git` output keyed by the first argument, so a whole
    /// check runs against a scripted remote.
    private struct StubGit: UpdateGitProbe {
        /// Keyed by the joined argument list.
        let responses: [String: UpdateGitOutcome]
        func git(_ arguments: [String], in checkout: URL) async -> UpdateGitOutcome {
            responses[arguments.joined(separator: " ")]
                ?? .failed("no scripted answer for \(arguments.joined(separator: " "))")
        }
    }

    /// Injected clock. `@unchecked Sendable`: mutated only by the single
    /// synchronous test that owns it, before each read.
    private final class TestClock: @unchecked Sendable {
        var now: Date
        init(_ now: Date) { self.now = now }
    }

    private let repository = UpdateRepository(owner: "acme", name: "widget")

    // MARK: - Semantic version parsing and ordering

    @Test func parsesTheFormsTheProjectSpells() {
        #expect(SemanticVersion("0.9.1") == SemanticVersion(major: 0, minor: 9, patch: 1))
        #expect(SemanticVersion("v0.9.1") == SemanticVersion(major: 0, minor: 9, patch: 1))
        #expect(SemanticVersion("0.9") == SemanticVersion(major: 0, minor: 9, patch: 0))
        #expect(SemanticVersion("0.9.1+f1480d51")?.build == "f1480d51")
        #expect(SemanticVersion("0.9.0-dev")?.prerelease == ["dev"])
        #expect(SemanticVersion("1.2.3-rc.2+abc")?.prerelease == ["rc", "2"])
    }

    @Test func rejectsThingsThatAreNotVersions() {
        #expect(SemanticVersion("") == nil)
        #expect(SemanticVersion("main") == nil)
        #expect(SemanticVersion("release-candidate") == nil)
        #expect(SemanticVersion("0") == nil)
        #expect(SemanticVersion("0.9.1.4") == nil)
        #expect(SemanticVersion("0.x.1") == nil)
    }

    @Test func buildMetadataIsIgnoredForOrdering() {
        let plain = SemanticVersion("0.9.1")!
        let stamped = SemanticVersion("0.9.1+f1480d51")!
        #expect(plain == stamped)
        #expect(!(stamped > plain))
        #expect(!(plain > stamped))
    }

    @Test func ordersMajorMinorPatchAndPrereleases() {
        #expect(SemanticVersion("0.9.1")! < SemanticVersion("0.9.2")!)
        #expect(SemanticVersion("0.9.9")! < SemanticVersion("0.10.0")!)
        #expect(SemanticVersion("0.10.0")! < SemanticVersion("1.0.0")!)
        // A prerelease sorts BELOW its own release …
        #expect(SemanticVersion("0.9.1-dev")! < SemanticVersion("0.9.1")!)
        // … and above the previous release.
        #expect(SemanticVersion("0.9.0")! < SemanticVersion("0.9.1-dev")!)
        #expect(SemanticVersion("1.0.0-rc.1")! < SemanticVersion("1.0.0-rc.2")!)
        #expect(SemanticVersion("1.0.0-alpha")! < SemanticVersion("1.0.0-beta")!)
    }

    @Test func releaseDescriptionDropsBuildMetadata() {
        #expect(SemanticVersion("v0.9.1+f1480d51")!.releaseDescription == "0.9.1")
        #expect(SemanticVersion("1.0.0-rc.1")!.releaseDescription == "1.0.0-rc.1")
    }

    // MARK: - Remote parsing

    @Test func parsesVersionTagsFromLsRemote() {
        let output = """
            9c1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/tags/v0.9.0
            aa1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/tags/v0.9.0^{}
            bb1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/tags/v0.9.1
            cc1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/tags/nightly
            dd1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/heads/main
            """
        let tags = GitRemoteParsing.tags(lsRemoteOutput: output)
        #expect(tags.map(\.releaseDescription) == ["0.9.0", "0.9.1"])
    }

    @Test func emptyRemoteYieldsNoTags() {
        #expect(GitRemoteParsing.tags(lsRemoteOutput: "").isEmpty)
        #expect(GitRemoteParsing.tags(lsRemoteOutput: "fatal: repository not found").isEmpty)
    }

    @Test func readsTheRevisionFromAnLsRemoteLine() {
        let output = "1f2e3d4c5b6a798877665544332211aabbccddee\trefs/heads/main"
        #expect(
            GitRemoteParsing.revision(lsRemoteOutput: output)
                == "1f2e3d4c5b6a798877665544332211aabbccddee")
        #expect(GitRemoteParsing.revision(lsRemoteOutput: "") == nil)
        #expect(GitRemoteParsing.revision(lsRemoteOutput: "fatal: could not read") == nil)
    }

    @Test func parsesTheRemoteURLFormsGitPrints() {
        let expected = UpdateRepository(owner: "acme", name: "widget")
        #expect(UpdateRepository.parse(remoteURL: "https://github.com/acme/widget.git") == expected)
        #expect(UpdateRepository.parse(remoteURL: "https://github.com/acme/widget") == expected)
        #expect(UpdateRepository.parse(remoteURL: "git@github.com:acme/widget.git") == expected)
        #expect(
            UpdateRepository.parse(remoteURL: "ssh://git@github.com/acme/widget.git") == expected)
        // A non-GitHub remote has no release feed this code knows how to ask.
        #expect(UpdateRepository.parse(remoteURL: "https://git.example.com/acme/widget") == nil)
        #expect(UpdateRepository.parse(remoteURL: "") == nil)
    }

    // MARK: - The decision table

    private func decide(
        current: String,
        builtRevision: String? = nil,
        release: ReleaseFeedOutcome = .unavailable("offline"),
        tags: [String]? = nil,
        main: String? = nil
    ) -> UpdateAvailability {
        UpdateDecision.decide(
            currentVersion: SemanticVersion(current)!,
            builtRevision: builtRevision,
            release: release,
            remoteTags: tags.map { $0.compactMap { SemanticVersion($0) } },
            remoteMainRevision: main,
            repository: repository)
    }

    @Test func aNewerPublishedReleaseWins() {
        let page = URL(string: "https://github.com/acme/widget/releases/tag/v0.9.2")!
        let answer = decide(
            current: "0.9.1", release: .release(tag: "v0.9.2", page: page))
        #expect(answer == .releaseAvailable(version: SemanticVersion("0.9.2")!, page: page))
    }

    @Test func aNewerTagCountsWhenTheFeedIsUnavailable() {
        // The normal shape while the repository is private: the API 404s and
        // the person's own git credentials still see the tags.
        let answer = decide(
            current: "0.9.1", release: .unavailable("no public release feed"),
            tags: ["0.9.0", "0.9.1", "0.9.2"])
        guard case .releaseAvailable(let version, let page) = answer else {
            Issue.record("expected a release, got \(answer)")
            return
        }
        #expect(version == SemanticVersion("0.9.2")!)
        #expect(page.absoluteString.hasSuffix("/releases/tag/v0.9.2"))
    }

    @Test func aBuildStampedVersionIsNotOutdatedByItsOwnRelease() {
        let answer = decide(
            current: "0.9.1+f1480d51", release: .release(tag: "v0.9.1", page: nil))
        #expect(answer == .upToDate(version: SemanticVersion("0.9.1+f1480d51")!))
    }

    @Test func anOlderReleaseTagIsNotAnUpdate() {
        let answer = decide(current: "0.9.1", release: .release(tag: "v0.9.0", page: nil))
        #expect(answer == .upToDate(version: SemanticVersion("0.9.1")!))
    }

    @Test func mainAheadIsTheWeakerSignalAndOnlyWithoutANewerRelease() {
        let answer = decide(
            current: "0.9.1", builtRevision: "f1480d51", release: .noReleases,
            tags: ["0.9.1"], main: "cd34299aa11223344556677889900aabbccddeeff")
        #expect(
            answer
                == .newerSourceAvailable(
                    remoteRevision: "cd34299aa11223344556677889900aabbccddeeff",
                    builtRevision: "f1480d51"))
    }

    @Test func aNewerReleaseOutranksMainHavingMoved() {
        let answer = decide(
            current: "0.9.1", builtRevision: "f1480d51",
            release: .release(tag: "v0.9.2", page: nil),
            main: "cd34299aa11223344556677889900aabbccddeeff")
        guard case .releaseAvailable(let version, _) = answer else {
            Issue.record("expected the release to win, got \(answer)")
            return
        }
        #expect(version == SemanticVersion("0.9.2")!)
    }

    @Test func anAbbreviatedBuildRevisionMatchesTheFullRemoteSHA() {
        let answer = decide(
            current: "0.9.1", builtRevision: "cd34299a", release: .noReleases,
            main: "cd34299aa11223344556677889900aabbccddeeff")
        #expect(answer == .upToDate(version: SemanticVersion("0.9.1")!))
    }

    @Test func nothingReachableIsUnknownNotUpToDate() {
        let answer = decide(current: "0.9.1", release: .unavailable("could not reach"))
        #expect(answer == .unknown(reason: "could not reach"))
    }

    @Test func aDevBuildWithNoCheckoutStillReportsUnknown() {
        let answer = decide(
            current: "0.9.1", builtRevision: nil,
            release: .unavailable("no public release feed for acme/widget"))
        guard case .unknown(let reason) = answer else {
            Issue.record("expected unknown, got \(answer)")
            return
        }
        #expect(reason.contains("acme/widget"))
    }

    @Test func aReachableFeedWithNoReleasesIsUpToDate() {
        #expect(decide(current: "0.9.1", release: .noReleases)
            == .upToDate(version: SemanticVersion("0.9.1")!))
    }

    // MARK: - The service over both seams

    @Test func serviceUsesTheCheckoutsOwnOriginAndItsTags() async {
        let service = UpdateCheckService(
            currentVersion: SemanticVersion("0.9.1")!,
            builtRevision: "f1480d51",
            checkout: URL(filePath: "/nowhere/checkout"),
            feed: StubFeed(outcome: .unavailable("private")),
            git: StubGit(responses: [
                "remote get-url origin": .output("git@github.com:acme/widget.git"),
                "ls-remote --tags origin": .output(
                    "aa1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3\trefs/tags/v0.9.3"),
                "ls-remote origin main": .output(
                    "f1480d51a11223344556677889900aabbccddeeff\trefs/heads/main"),
            ]))
        let resolved = await service.resolveRepository()
        #expect(resolved == UpdateRepository(owner: "acme", name: "widget"))
        let answer = await service.check()
        guard case .releaseAvailable(let version, let page) = answer else {
            Issue.record("expected v0.9.3 to be offered")
            return
        }
        #expect(version == SemanticVersion("0.9.3")!)
        #expect(page.absoluteString == "https://github.com/acme/widget/releases/tag/v0.9.3")
    }

    @Test func serviceWithNoCheckoutAndNoFeedIsSilentlyUnknown() async {
        let service = UpdateCheckService(
            currentVersion: SemanticVersion("0.9.1")!,
            builtRevision: nil,
            checkout: nil,
            feed: StubFeed(outcome: .unavailable("could not reach the release feed")),
            git: StubGit(responses: [:]),
            repository: repository)
        let answer = await service.check()
        #expect(answer == .unknown(reason: "could not reach the release feed"))
    }

    @Test func serviceFallsBackToTheCompiledInRepositoryWhenOriginIsUnparseable() async {
        let service = UpdateCheckService(
            currentVersion: SemanticVersion("0.9.1")!,
            checkout: URL(filePath: "/nowhere/checkout"),
            feed: StubFeed(outcome: .noReleases),
            git: StubGit(responses: [
                "remote get-url origin": .output("https://git.example.com/acme/widget")
            ]))
        let resolved = await service.resolveRepository()
        #expect(resolved == UpdateRepository.default)
    }

    // MARK: - Banner suppression

    @Test func aDismissedReleaseNeverBannersAgain() {
        let available = UpdateAvailability.releaseAvailable(
            version: SemanticVersion("0.9.2")!, page: repository.releasesPage)
        #expect(
            UpdateCheckPolicy.bannerMessage(for: available, dismissedVersion: nil)
                == "SteerLab 0.9.2 is available")
        #expect(
            UpdateCheckPolicy.bannerMessage(for: available, dismissedVersion: "0.9.2") == nil)
        // A newer release than the dismissed one banners again.
        #expect(
            UpdateCheckPolicy.bannerMessage(for: available, dismissedVersion: "0.9.1")
                == "SteerLab 0.9.2 is available")
        // Dismissing a LATER version also suppresses this one (clock/registry
        // skew must never nag).
        #expect(
            UpdateCheckPolicy.bannerMessage(for: available, dismissedVersion: "0.9.3") == nil)
        // Garbage in the defaults slot does not suppress the banner.
        #expect(
            UpdateCheckPolicy.bannerMessage(for: available, dismissedVersion: "not-a-version")
                == "SteerLab 0.9.2 is available")
    }

    @Test func onlyAReleaseEverBanners() {
        let weaker = UpdateAvailability.newerSourceAvailable(
            remoteRevision: "cd34299a", builtRevision: "f1480d51")
        #expect(UpdateCheckPolicy.bannerMessage(for: weaker, dismissedVersion: nil) == nil)
        #expect(
            UpdateCheckPolicy.bannerMessage(
                for: .upToDate(version: SemanticVersion("0.9.1")!), dismissedVersion: nil) == nil)
        #expect(
            UpdateCheckPolicy.bannerMessage(
                for: .unknown(reason: "offline"), dismissedVersion: nil) == nil)
    }

    @Test func theManualCheckReportsEveryStateHonestly() {
        #expect(
            UpdateCheckPolicy.manualReport(for: .unknown(reason: "could not reach the feed"))
                == "Could not check for updates: could not reach the feed.")
        #expect(
            UpdateCheckPolicy.manualReport(
                for: .upToDate(version: SemanticVersion("0.9.1+f1480d51")!))
                == "SteerLab 0.9.1 is the newest version available.")
        let weaker = UpdateCheckPolicy.manualReport(
            for: .newerSourceAvailable(
                remoteRevision: "cd34299aa1122334", builtRevision: "f1480d51"))
        #expect(weaker.contains("no packaged release yet"))
        #expect(weaker.contains("cd34299a"))
    }

    // MARK: - The 24 h throttle, on an injected clock

    @Test func theAutomaticCheckRunsAtMostOncePerDay() {
        let noon = Date(timeIntervalSince1970: 1_800_000_000)
        #expect(
            UpdateCheckPolicy.shouldRunAutomaticCheck(
                enabled: true, lastCheck: nil, now: noon))
        #expect(
            !UpdateCheckPolicy.shouldRunAutomaticCheck(
                enabled: true, lastCheck: noon.addingTimeInterval(-3600), now: noon))
        #expect(
            UpdateCheckPolicy.shouldRunAutomaticCheck(
                enabled: true, lastCheck: noon.addingTimeInterval(-86_400), now: noon))
        // Off means off, however long it has been.
        #expect(
            !UpdateCheckPolicy.shouldRunAutomaticCheck(
                enabled: false, lastCheck: nil, now: noon))
        // A stamp in the future (clock moved backwards) must not silence the
        // check forever.
        #expect(
            UpdateCheckPolicy.shouldRunAutomaticCheck(
                enabled: true, lastCheck: noon.addingTimeInterval(86_400), now: noon))
    }

    @Test func preferencesPersistTheThrottleAndTheDismissalOnAnInjectedClock() throws {
        let domain = "SteerLabUpdateCheckTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: domain))
        defer { defaults.removePersistentDomain(forName: domain) }

        let start = Date(timeIntervalSince1970: 1_800_000_000)
        let clock = TestClock(start)
        let preferences = UpdateCheckPreferences(defaults: defaults, now: { clock.now })

        // Default ON, never checked → due.
        #expect(preferences.automaticChecksEnabled)
        #expect(preferences.shouldRunAutomaticCheck)

        preferences.recordCheck()
        #expect(preferences.lastCheck == start)
        clock.now = start.addingTimeInterval(23 * 3600)
        #expect(!preferences.shouldRunAutomaticCheck)
        clock.now = start.addingTimeInterval(24 * 3600 + 1)
        #expect(preferences.shouldRunAutomaticCheck)

        preferences.automaticChecksEnabled = false
        #expect(!preferences.shouldRunAutomaticCheck)
        preferences.automaticChecksEnabled = true

        let available = UpdateAvailability.releaseAvailable(
            version: SemanticVersion("0.9.2")!, page: repository.releasesPage)
        #expect(preferences.bannerMessage(for: available) == "SteerLab 0.9.2 is available")
        preferences.dismiss(SemanticVersion("0.9.2+abc")!)
        #expect(preferences.dismissedVersion == "0.9.2")
        #expect(preferences.bannerMessage(for: available) == nil)
    }
}
