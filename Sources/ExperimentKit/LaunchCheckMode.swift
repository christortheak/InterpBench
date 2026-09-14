import Foundation

// =============================================================================
// Launch-check mode (2026-09-13 incident: a live controller wedged)
//
// `scripts/build-app.sh` ends by LAUNCHING the freshly assembled bundle for a
// few seconds to prove it initialises (resource families resolve, the code
// signature loads, the process stays up). Until this switch existed that
// launch was a full, real launch: the user's saved sites, defaults, Keychain
// token, and — on a Mac with a live SSH tunnel — the controller behind it. On
// 2026-09-13 the eight-second check connected to a live controller, its
// evidence auto-import began fetching every succeeded job the local ledger
// had never seen (hundreds of bundle downloads), the script then killed the
// app mid-transfer, and the controller stopped answering for half an hour.
//
// With `STEERLAB_LAUNCH_CHECK=1` in the environment the app performs NO
// network activity of any kind:
//
//   - `ClusterConnectionStore.client` is nil and `connect()` refuses, so no
//     site handshake and no remote polling can start;
//   - every `ClusterClient` is built on a URLSession whose only protocol
//     refuses the request (defence in depth for clients constructed outside
//     the store: panels, setup sheets, the local-engine probe);
//   - `ClusterTunnel` runs no `ssh` — not to check the ControlMaster, not to
//     recover a persisted forward, not to open one;
//   - `EvidenceAutoImportService` neither polls nor imports;
//   - the update signpost's automatic release check is skipped.
//
// Every refused attempt is RECORDED here, and the app prints one verdict line
// to stderr a few seconds after launch — `offlineLine` when nothing was
// attempted, `violationPrefix` + a count otherwise. The build script greps
// for the first and fails on the second, so a future code path that reaches
// the network at launch becomes a failed build, not a wedged controller.
//
// The mode is read from the process environment once; tests force it with
// `withOverride`, a task-local (every reader — client init, tunnel init, the
// store, the service — runs inside the calling task, and a task-local cannot
// leak into an unrelated test running concurrently in the same process),
// which is why every reader goes through `isActive` rather than the
// environment directly.
// =============================================================================

public enum LaunchCheckMode {

    /// The environment variable the build script sets for its launch check.
    /// Any non-empty value other than `0`, `false`, or `no` arms the mode.
    public static let environmentVariable = "STEERLAB_LAUNCH_CHECK"

    /// Printed (to stderr) when the verdict window closes with no attempt
    /// recorded. `scripts/build-app.sh` requires this exact line.
    public static let offlineLine = "launch-check: offline mode, no network activity"

    /// Prefix of the verdict when at least one attempt was refused. The
    /// build script fails the check when it sees this prefix.
    public static let violationPrefix = "launch-check: offline mode VIOLATED"

    /// Printed at arm time so a log that ends before the verdict still says
    /// the switch was honoured.
    public static let armedLine =
        "launch-check: offline mode armed (\(environmentVariable)) — site connection, "
        + "SSH tunnel, evidence auto-import, remote polling, and update checks are disabled"

    /// How long after launch the app waits before printing its verdict. The
    /// build script's observation window is longer than this by design.
    public static let verdictDelay: Duration = .seconds(4)

    /// Defaults suite the app uses under launch check, so the check never
    /// reads or writes the researcher's real preferences (active workspace,
    /// per-site toggles, recent roots).
    public static let isolatedDefaultsSuiteName = "org.steerlab.SteerLab.launch-check"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var attempts: [String] = []

    /// nil → read the environment; a value forces the mode for the tasks
    /// under `withOverride` (tests only).
    @TaskLocal private static var testingOverride: Bool?

    private static let environmentValue: Bool = {
        let raw = ProcessInfo.processInfo.environment[environmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        return !raw.isEmpty && raw != "0" && raw != "false" && raw != "no"
    }()

    /// Whether this process is a build-script launch check. Environment
    /// unless a test override is in effect for the current task.
    public static var isActive: Bool {
        testingOverride ?? environmentValue
    }

    /// Test seam: run `body` with the mode forced on (or off) for that task
    /// and its children. Recorded attempts are cleared on entry so the test
    /// observes only what it provokes.
    public static func withOverride<T>(
        _ active: Bool,
        isolation: isolated (any Actor)? = #isolation,
        _ body: () async throws -> T
    ) async rethrows -> T {
        resetBlockedAttemptsForTesting()
        return try await $testingOverride.withValue(active, operation: body, isolation: isolation)
    }

    /// Forget recorded attempts (tests only).
    public static func resetBlockedAttemptsForTesting() {
        lock.withLock { attempts = [] }
    }

    /// Record a network attempt the mode refused. Callers pass a short,
    /// stable description ("cluster connect", "ssh -O check", a URL).
    public static func recordBlockedAttempt(_ description: String) {
        lock.withLock { attempts.append(description) }
    }

    /// Every refused attempt so far, in order.
    public static var blockedAttempts: [String] {
        lock.withLock { attempts }
    }

    /// The single verdict line the build script reads.
    public static var verdictLine: String {
        let recorded = blockedAttempts
        guard !recorded.isEmpty else { return offlineLine }
        let sample = recorded.prefix(3).joined(separator: "; ")
        return "\(violationPrefix): \(recorded.count) network attempt(s) refused — \(sample)"
    }

    /// The error every refused request surfaces as.
    public static func refusalError(_ description: String) -> URLError {
        URLError(
            .notConnectedToInternet,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "launch check: offline mode refused \(description)"
            ])
    }

    /// A URLSession that refuses every request (and records it). Used for
    /// every `ClusterClient` built while the mode is active.
    public static func blockingSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [LaunchCheckBlockingURLProtocol.self]
        return URLSession(configuration: configuration)
    }
}

/// Fails every request immediately, recording it. Registered as the ONLY
/// protocol of the launch-check session, so no scheme reaches the network.
final class LaunchCheckBlockingURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canInit(with task: URLSessionTask) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let method = request.httpMethod ?? "GET"
        let target = request.url?.absoluteString ?? "<no url>"
        let description = "\(method) \(target)"
        LaunchCheckMode.recordBlockedAttempt(description)
        client?.urlProtocol(self, didFailWithError: LaunchCheckMode.refusalError(description))
    }

    override func stopLoading() {}
}
