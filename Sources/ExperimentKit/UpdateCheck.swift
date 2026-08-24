import Foundation

// MARK: - Update availability: a SIGNPOST, never an updater
//
// Most people running SteerLab have no Xcode and no signing identity, so
// their update story is "download the new zip from the repository's Releases
// page" — never "rebuild from source". This file is the small amount of
// machinery that lets the app SAY SO when it is warranted, and nothing more:
//
//   * nothing is ever downloaded, unpacked, installed, or replaced;
//   * no auth is ever attempted and no token is ever read or stored — while
//     the repository is private the public API answers 404 and the whole
//     check degrades to "unknown", silently;
//   * the only bytes that leave this machine are an unauthenticated
//     `GET /repos/<owner>/<name>/releases/latest` and, when a code checkout
//     sits beside the app, `git ls-remote` run IN that checkout through the
//     person's OWN git credential helpers.
//
// PRIVACY: nothing about this installation is transmitted. No identifier, no
// version string, no workspace name, no telemetry of any kind is sent
// anywhere — the requests above carry no body and no custom headers, and the
// comparison happens entirely on this machine. An automatic check that
// cannot reach anything says nothing at all; only a manual check reports the
// failure, and it reports it honestly rather than as "you are up to date".

// MARK: - Semantic version

/// The subset of semantic versioning the project actually spells: `MAJOR.
/// MINOR.PATCH`, optionally `-prerelease` (`0.9.0-dev`), optionally
/// `+build` (`0.9.1+f1480d51`, the engine stamp's short SHA).
///
/// Ordering follows semver: build metadata is IGNORED entirely (two builds of
/// 0.9.1 are the same release), and a prerelease sorts BELOW its own release
/// (`0.9.1-dev` < `0.9.1`) so a dev build never reads as newer than the
/// packaged release it precedes.
public struct SemanticVersion: Sendable, Hashable, Comparable, CustomStringConvertible {
    public let major: Int
    public let minor: Int
    public let patch: Int
    /// Dot-separated prerelease identifiers, or empty for a final release.
    public let prerelease: [String]
    /// Kept for display only — never compared.
    public let build: String?

    public init(
        major: Int, minor: Int, patch: Int,
        prerelease: [String] = [], build: String? = nil
    ) {
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = prerelease
        self.build = build
    }

    /// Parses "0.9.1", "v0.9.1", "0.9.1-dev", "0.9.1+f1480d51",
    /// "v0.9.2-rc.1+abc" and the two-component "0.9" form. Returns nil for
    /// anything else — a tag that is not a version is not a signal.
    public init?(_ raw: String) {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if text.first == "v" || text.first == "V" { text.removeFirst() }

        var build: String?
        if let plus = text.firstIndex(of: "+") {
            build = String(text[text.index(after: plus)...])
            text = String(text[..<plus])
        }
        var prerelease: [String] = []
        if let dash = text.firstIndex(of: "-") {
            let tail = String(text[text.index(after: dash)...])
            guard !tail.isEmpty else { return nil }
            prerelease = tail.split(separator: ".", omittingEmptySubsequences: false)
                .map(String.init)
            guard !prerelease.contains(where: \.isEmpty) else { return nil }
            text = String(text[..<dash])
        }

        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard (2...3).contains(parts.count) else { return nil }
        var numbers: [Int] = []
        for part in parts {
            guard !part.isEmpty, part.allSatisfy(\.isNumber), let value = Int(part)
            else { return nil }
            numbers.append(value)
        }
        self.init(
            major: numbers[0], minor: numbers[1],
            patch: numbers.count > 2 ? numbers[2] : 0,
            prerelease: prerelease, build: build?.isEmpty == true ? nil : build)
    }

    public var description: String {
        var text = "\(major).\(minor).\(patch)"
        if !prerelease.isEmpty { text += "-" + prerelease.joined(separator: ".") }
        if let build, !build.isEmpty { text += "+" + build }
        return text
    }

    /// The release name a human reads — no build metadata, since two builds
    /// of one release are the same thing to download.
    public var releaseDescription: String {
        var text = "\(major).\(minor).\(patch)"
        if !prerelease.isEmpty { text += "-" + prerelease.joined(separator: ".") }
        return text
    }

    public static func < (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        if lhs.major != rhs.major { return lhs.major < rhs.major }
        if lhs.minor != rhs.minor { return lhs.minor < rhs.minor }
        if lhs.patch != rhs.patch { return lhs.patch < rhs.patch }
        // A prerelease sorts below the release it precedes.
        if lhs.prerelease.isEmpty != rhs.prerelease.isEmpty {
            return !lhs.prerelease.isEmpty
        }
        for (left, right) in zip(lhs.prerelease, rhs.prerelease) where left != right {
            switch (Int(left), Int(right)) {
            case let (leftNumber?, rightNumber?): return leftNumber < rightNumber
            case (.some, .none): return true  // numeric identifiers sort first
            case (.none, .some): return false
            case (.none, .none): return left < right
            }
        }
        return lhs.prerelease.count < rhs.prerelease.count
    }

    /// Equality ignores build metadata, matching the ordering.
    public static func == (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.patch == rhs.patch
            && lhs.prerelease == rhs.prerelease
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(major)
        hasher.combine(minor)
        hasher.combine(patch)
        hasher.combine(prerelease)
    }
}

// MARK: - Which repository to ask about

/// The GitHub repository whose Releases page is the download story. Derived
/// from the sibling checkout's `origin` when there is one (so a fork checks
/// its own fork), else the compiled-in default.
public struct UpdateRepository: Sendable, Equatable {
    public let owner: String
    public let name: String

    public init(owner: String, name: String) {
        self.owner = owner
        self.name = name
    }

    /// The repository the packaged app was cut from — used when there is no
    /// checkout beside the app to ask.
    public static let `default` = UpdateRepository(
        owner: "christortheak", name: "InterpBench")

    /// Unauthenticated latest-release endpoint. 404 while the repository is
    /// private, which this file treats as "unknown", never as an error worth
    /// authenticating past.
    public var latestReleaseEndpoint: URL {
        URL(string: "https://api.github.com/repos/\(owner)/\(name)/releases/latest")!
    }

    /// Where a person is sent to download a build.
    public var releasesPage: URL {
        URL(string: "https://github.com/\(owner)/\(name)/releases")!
    }

    public func releasePage(tag: String) -> URL {
        let escaped =
            tag.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? tag
        return URL(string: "https://github.com/\(owner)/\(name)/releases/tag/\(escaped)")
            ?? releasesPage
    }

    /// Parses the forms `git remote get-url origin` prints:
    /// `https://github.com/owner/name(.git)`, `git@github.com:owner/name.git`,
    /// `ssh://git@github.com/owner/name.git`. Non-GitHub remotes return nil —
    /// the release feed only knows how to ask GitHub.
    public static func parse(remoteURL raw: String) -> UpdateRepository? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if text.hasSuffix(".git") { text.removeLast(4) }

        let hostAndPath: String
        if let range = text.range(of: "://") {
            var rest = String(text[range.upperBound...])
            if let at = rest.firstIndex(of: "@") { rest = String(rest[rest.index(after: at)...]) }
            hostAndPath = rest
        } else if let at = text.firstIndex(of: "@") {
            // scp-style: user@host:owner/name
            hostAndPath = String(text[text.index(after: at)...]).replacingOccurrences(
                of: ":", with: "/")
        } else {
            hostAndPath = text
        }

        let parts = hostAndPath.split(separator: "/").map(String.init)
        guard parts.count >= 3 else { return nil }
        let host = (parts[0].split(separator: ":").first.map(String.init) ?? parts[0])
            .lowercased()
        guard host == "github.com" || host.hasSuffix(".github.com") else { return nil }
        let owner = parts[parts.count - 2]
        let name = parts[parts.count - 1]
        guard !owner.isEmpty, !name.isEmpty else { return nil }
        return UpdateRepository(owner: owner, name: name)
    }
}

// MARK: - Seam 1: the public release feed

/// What the latest-release endpoint said. `unavailable` is deliberately not
/// an error type: offline, private-repo 404, and rate limiting are all the
/// same fact to a signpost — *we do not know*.
public enum ReleaseFeedOutcome: Sendable, Equatable {
    /// A published (non-draft) release, with its tag and page.
    case release(tag: String, page: URL?)
    /// Reachable, and the repository publishes no releases yet.
    case noReleases
    /// Could not find out, with a short human reason for the manual check.
    case unavailable(String)
}

/// Injectable network seam — no test ever touches the real network.
public protocol UpdateReleaseFeed: Sendable {
    func latestRelease(for repository: UpdateRepository) async -> ReleaseFeedOutcome
}

/// Live feed: one unauthenticated GET with a short timeout. Sends no
/// credentials, no identifiers, and no body; reads only `tag_name`,
/// `html_url`, and `draft` from the response.
public struct GitHubReleaseFeed: UpdateReleaseFeed {
    private let session: URLSession
    private let timeout: TimeInterval

    public init(session: URLSession = .shared, timeout: TimeInterval = 8) {
        self.session = session
        self.timeout = timeout
    }

    private struct Payload: Decodable {
        let tagName: String?
        let htmlURL: String?
        let draft: Bool?

        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
            case htmlURL = "html_url"
            case draft
        }
    }

    public func latestRelease(for repository: UpdateRepository) async -> ReleaseFeedOutcome {
        var request = URLRequest(url: repository.latestReleaseEndpoint)
        request.timeoutInterval = timeout
        request.httpMethod = "GET"
        // The one header, and it selects a response format — it says nothing
        // about this machine.
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .unavailable("the release feed returned an unexpected response")
            }
            switch http.statusCode {
            case 200:
                guard let payload = try? JSONDecoder().decode(Payload.self, from: data),
                    let tag = payload.tagName, !tag.isEmpty, payload.draft != true
                else { return .noReleases }
                return .release(tag: tag, page: payload.htmlURL.flatMap(URL.init(string:)))
            case 404:
                // The normal answer for a private repository. No auth is
                // attempted, now or ever.
                return .unavailable(
                    "no public release feed for \(repository.owner)/\(repository.name)")
            default:
                return .unavailable("the release feed answered HTTP \(http.statusCode)")
            }
        } catch {
            return .unavailable("could not reach the release feed")
        }
    }
}

// MARK: - Seam 2: the sibling checkout's git remote

/// One `git` invocation's result: its output, or a short reason there is
/// none. (Not `Result`, whose failure must be an `Error` — a git command that
/// declines to answer is data here, not a thrown condition.)
public enum UpdateGitOutcome: Sendable, Equatable {
    case output(String)
    case failed(String)
}

/// Injectable git seam. Conformers run `git <arguments>` inside a checkout
/// and hand back its stdout, or a short reason it produced nothing usable.
public protocol UpdateGitProbe: Sendable {
    func git(_ arguments: [String], in checkout: URL) async -> UpdateGitOutcome
}

/// Live probe over the house process runner (`ProvisionCommandRunner`), with
/// a hard timeout so a remote that wants to prompt for credentials can never
/// hang a launch check — cancelling the runner terminates the child.
public struct SystemUpdateGitProbe: UpdateGitProbe {
    private let runner: any ProvisionCommandRunner
    private let timeout: Duration

    public init(
        runner: any ProvisionCommandRunner = SystemProvisionRunner(),
        timeout: Duration = .seconds(10)
    ) {
        self.runner = runner
        self.timeout = timeout
    }

    public func git(_ arguments: [String], in checkout: URL) async -> UpdateGitOutcome {
        let argv = ["/usr/bin/git", "-C", checkout.path] + arguments
        let lines = ProvisionLineBuffer()
        let runner = self.runner
        let verb = arguments.first ?? "git"
        return await withTaskGroup(of: UpdateGitOutcome?.self) { group in
            group.addTask {
                do {
                    let code = try await runner.run(argv) { lines.append($0) }
                    guard code == 0 else { return .failed("git \(verb) exited \(code)") }
                    return .output(lines.snapshot().joined(separator: "\n"))
                } catch {
                    return .failed("git could not be run")
                }
            }
            group.addTask { [timeout] in
                try? await Task.sleep(for: timeout)
                return nil
            }
            for await first in group {
                group.cancelAll()
                return first ?? .failed("git \(verb) timed out")
            }
            return .failed("git \(verb) produced no result")
        }
    }
}

// MARK: - Parsing what the remote says

public enum GitRemoteParsing {
    /// Version tags from `git ls-remote --tags <remote>` output. Peeled
    /// entries (`…^{}`) collapse onto their tag, non-version tags are
    /// dropped, and the result is sorted ascending.
    public static func tags(lsRemoteOutput: String) -> [SemanticVersion] {
        var found: Set<SemanticVersion> = []
        for line in lsRemoteOutput.split(separator: "\n", omittingEmptySubsequences: true) {
            let fields = line.split(separator: "\t", omittingEmptySubsequences: true)
            guard fields.count >= 2 else { continue }
            var ref = String(fields[1]).trimmingCharacters(in: .whitespaces)
            guard ref.hasPrefix("refs/tags/") else { continue }
            ref.removeFirst("refs/tags/".count)
            if ref.hasSuffix("^{}") { ref.removeLast(3) }
            if let version = SemanticVersion(ref) { found.insert(version) }
        }
        return found.sorted()
    }

    /// The full SHA a `git ls-remote <remote> <ref>` line reports, or nil.
    public static func revision(lsRemoteOutput: String) -> String? {
        for line in lsRemoteOutput.split(separator: "\n", omittingEmptySubsequences: true) {
            let fields = line.split(
                whereSeparator: { $0 == "\t" || $0 == " " })
            guard let sha = fields.first else { continue }
            let text = String(sha).trimmingCharacters(in: .whitespaces)
            guard text.count >= 7, text.allSatisfy(\.isHexDigit) else { continue }
            return text.lowercased()
        }
        return nil
    }
}

// MARK: - The answer

/// What a completed check concluded. Deliberately four states, because
/// "unknown" is a real and common answer that must never be dressed up as
/// "up to date".
public enum UpdateAvailability: Sendable, Equatable {
    /// A newer packaged RELEASE exists — the only state that may banner.
    case releaseAvailable(version: SemanticVersion, page: URL)
    /// No newer release tag, but the remote's `main` is not the revision this
    /// build was assembled from. A weaker signal, and subordinate to a
    /// release: shown only in the manual check, never as a banner.
    case newerSourceAvailable(remoteRevision: String, builtRevision: String)
    /// Everything reachable agrees this build is current.
    case upToDate(version: SemanticVersion)
    /// Offline, no checkout, a private-repo 404, a git failure — with the
    /// reason, which the manual check reports verbatim.
    case unknown(reason: String)
}

/// The pure decision table. Every input is already-observed data, so the
/// whole policy is testable without a network, a remote, or a clock.
public enum UpdateDecision {
    public static func decide(
        currentVersion: SemanticVersion,
        builtRevision: String?,
        release: ReleaseFeedOutcome,
        remoteTags: [SemanticVersion]?,
        remoteMainRevision: String?,
        repository: UpdateRepository
    ) -> UpdateAvailability {
        // 1. The published release feed wins when it names something newer.
        if case .release(let tag, let page) = release,
            let version = SemanticVersion(tag), version > currentVersion
        {
            return .releaseAvailable(
                version: version, page: page ?? repository.releasePage(tag: tag))
        }
        // 2. Else a version TAG on the remote is still a release signal (the
        //    normal case while the repository is private and the API 404s).
        if let newest = remoteTags?.last, newest > currentVersion {
            return .releaseAvailable(
                version: newest,
                page: repository.releasePage(tag: "v\(newest.releaseDescription)"))
        }
        // 3. Subordinate signal: the remote's main is not what we were built
        //    from. A DIFFERENCE, not a proven ancestry — worded accordingly.
        if let remoteMainRevision, let builtRevision, !builtRevision.isEmpty,
            !revisionsMatch(remoteMainRevision, builtRevision)
        {
            return .newerSourceAvailable(
                remoteRevision: remoteMainRevision, builtRevision: builtRevision)
        }
        // 4. Up to date only when SOMETHING answered.
        let releaseAnswered: Bool
        switch release {
        case .release, .noReleases: releaseAnswered = true
        case .unavailable: releaseAnswered = false
        }
        if releaseAnswered || remoteTags != nil || remoteMainRevision != nil {
            return .upToDate(version: currentVersion)
        }
        // 5. Nothing answered. Say so; never claim currency.
        if case .unavailable(let reason) = release { return .unknown(reason: reason) }
        return .unknown(reason: "no release feed and no code checkout to ask")
    }

    /// The Info.plist stamp is an abbreviated SHA and `ls-remote` reports the
    /// full one, so a prefix match in either direction is the same commit.
    static func revisionsMatch(_ left: String, _ right: String) -> Bool {
        let a = left.lowercased()
        let b = right.lowercased()
        guard !a.isEmpty, !b.isEmpty else { return false }
        return a.hasPrefix(b) || b.hasPrefix(a)
    }
}

// MARK: - The service

/// Runs the check: resolve the repository, ask the public feed, ask the
/// sibling checkout's remote, then apply `UpdateDecision`. Pure over its
/// seams — the feed, the git probe, and the checkout location are all
/// injected, so the whole thing runs headless in tests.
public struct UpdateCheckService: Sendable {
    private let currentVersion: SemanticVersion
    private let builtRevision: String?
    private let checkout: URL?
    private let feed: any UpdateReleaseFeed
    private let git: any UpdateGitProbe
    private let repositoryOverride: UpdateRepository?

    /// - Parameters:
    ///   - currentVersion: this build's version; defaults to the engine
    ///     constant, or the packaged bundle's `SLFullVersionString`.
    ///   - builtRevision: the revision this BUILD was assembled from
    ///     (`SLSourceRevision`) — never a live read of any checkout here.
    ///   - checkout: the code checkout to ask about the remote, resolved
    ///     through `CodeResources.executableCheckout()` by default.
    public init(
        currentVersion: SemanticVersion? = nil,
        builtRevision: String? = CodeResources.bundleSourceRevision,
        checkout: URL? = CodeResources.executableCheckoutRoot,
        feed: any UpdateReleaseFeed = GitHubReleaseFeed(),
        git: any UpdateGitProbe = SystemUpdateGitProbe(),
        repository: UpdateRepository? = nil
    ) {
        self.currentVersion =
            currentVersion
            ?? SemanticVersion(CodeResources.bundleFullVersion ?? SteerLabVersion.version)
            ?? SemanticVersion(major: 0, minor: 0, patch: 0)
        self.builtRevision = builtRevision
        self.checkout = checkout
        self.feed = feed
        self.git = git
        self.repositoryOverride = repository
    }

    /// The version this check compares against — surfaced so the UI can name
    /// it ("SteerLab 0.9.1 — up to date").
    public var version: SemanticVersion { currentVersion }

    public func check() async -> UpdateAvailability {
        let repository = await resolveRepository()
        let release = await feed.latestRelease(for: repository)

        var remoteTags: [SemanticVersion]?
        var remoteMain: String?
        if let checkout {
            if case .output(let text) = await git(["ls-remote", "--tags", "origin"], checkout) {
                remoteTags = GitRemoteParsing.tags(lsRemoteOutput: text)
            }
            if case .output(let text) = await git(["ls-remote", "origin", "main"], checkout) {
                remoteMain = GitRemoteParsing.revision(lsRemoteOutput: text)
            }
        }
        return UpdateDecision.decide(
            currentVersion: currentVersion, builtRevision: builtRevision,
            release: release, remoteTags: remoteTags, remoteMainRevision: remoteMain,
            repository: repository)
    }

    /// The repository the check asks about: the sibling checkout's `origin`
    /// when it parses as a GitHub remote, else the compiled-in default.
    public func resolveRepository() async -> UpdateRepository {
        if let repositoryOverride { return repositoryOverride }
        guard let checkout else { return .default }
        guard case .output(let text) = await git(["remote", "get-url", "origin"], checkout),
            let parsed = UpdateRepository.parse(remoteURL: text)
        else { return .default }
        return parsed
    }

    private func git(_ arguments: [String], _ checkout: URL) async -> UpdateGitOutcome {
        await git.git(arguments, in: checkout)
    }
}

// MARK: - Presentation-independent policy (throttle, banner, dismissal)

/// The rules the UI applies around a check, kept out of the views so they can
/// be exercised with an injected clock and a scratch defaults domain.
public enum UpdateCheckPolicy {
    /// At most one automatic check per day.
    public static let automaticInterval: TimeInterval = 24 * 60 * 60

    /// Whether the launch check should run at all: the preference is on and
    /// the last check (if any) is older than the interval.
    public static func shouldRunAutomaticCheck(
        enabled: Bool, lastCheck: Date?, now: Date
    ) -> Bool {
        guard enabled else { return false }
        guard let lastCheck else { return true }
        // A last-check stamp in the future (clock moved backwards) is not a
        // reason to go quiet forever.
        if lastCheck > now { return true }
        return now.timeIntervalSince(lastCheck) >= automaticInterval
    }

    /// The banner sentence for an outcome, or nil when nothing should show.
    /// Only a newer RELEASE ever banners, and never one already dismissed.
    public static func bannerMessage(
        for availability: UpdateAvailability, dismissedVersion: String?
    ) -> String? {
        guard case .releaseAvailable(let version, _) = availability else { return nil }
        if let dismissedVersion,
            let dismissed = SemanticVersion(dismissedVersion), dismissed >= version
        { return nil }
        return "SteerLab \(version.releaseDescription) is available"
    }

    /// What a MANUAL check reports — every state gets an honest sentence,
    /// including the ones an automatic check passes over in silence.
    public static func manualReport(for availability: UpdateAvailability) -> String {
        switch availability {
        case .releaseAvailable(let version, _):
            return "SteerLab \(version.releaseDescription) is available."
        case .newerSourceAvailable(let remote, _):
            return "Newer source is available (no packaged release yet) — "
                + "the repository's main is at \(remote.prefix(8))."
        case .upToDate(let version):
            return "SteerLab \(version.releaseDescription) is the newest version available."
        case .unknown(let reason):
            return "Could not check for updates: \(reason)."
        }
    }
}

/// The small amount of per-machine state the check remembers: the preference,
/// the last-run stamp, and the release whose banner was dismissed. Injectable
/// defaults domain and clock so tests never touch the real ones.
///
/// `@unchecked Sendable`: the only stored reference is a `UserDefaults`,
/// which Apple documents as thread-safe; nothing else here is mutable.
public struct UpdateCheckPreferences: @unchecked Sendable {
    /// Automatic launch checks (default ON, per the product ruling).
    public static let automaticEnabledKey = "UpdateCheckAutomaticEnabled"
    public static let lastCheckKey = "UpdateCheckLastRun"
    public static let dismissedVersionKey = "UpdateCheckDismissedVersion"

    private let defaults: UserDefaults
    private let now: @Sendable () -> Date

    public init(
        defaults: UserDefaults = .standard,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.defaults = defaults
        self.now = now
    }

    public var automaticChecksEnabled: Bool {
        get {
            // Absent means "never chosen", which is ON.
            defaults.object(forKey: Self.automaticEnabledKey) as? Bool ?? true
        }
        nonmutating set { defaults.set(newValue, forKey: Self.automaticEnabledKey) }
    }

    public var lastCheck: Date? {
        get { defaults.object(forKey: Self.lastCheckKey) as? Date }
        nonmutating set { defaults.set(newValue, forKey: Self.lastCheckKey) }
    }

    public var dismissedVersion: String? {
        get { defaults.string(forKey: Self.dismissedVersionKey) }
        nonmutating set { defaults.set(newValue, forKey: Self.dismissedVersionKey) }
    }

    public var shouldRunAutomaticCheck: Bool {
        UpdateCheckPolicy.shouldRunAutomaticCheck(
            enabled: automaticChecksEnabled, lastCheck: lastCheck, now: now())
    }

    public func recordCheck() { lastCheck = now() }

    /// Remember a release as dismissed, so the same version never banners
    /// again on this machine.
    public func dismiss(_ version: SemanticVersion) {
        dismissedVersion = version.releaseDescription
    }

    public func bannerMessage(for availability: UpdateAvailability) -> String? {
        UpdateCheckPolicy.bannerMessage(
            for: availability, dismissedVersion: dismissedVersion)
    }
}
