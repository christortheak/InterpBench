import Foundation
#if canImport(AppKit)
import AppKit
#endif

// =============================================================================
// The headless provisioning operations (CLUSTER-CLI-LIFECYCLE-PLAN Phase B).
//
// These are the SAME steps the setup wizard performs — push, bootstrap
// plan/apply, profile validate, controller submit/status, tunnel, token import
// — lifted out of `ClusterProvisioner`'s observable, main-actor, UI-oriented
// state so a CLI (and the coordinator) can run them with no SwiftUI, no
// Observation, and no window. `ClusterProvisioner` keeps its API and delegates
// here: one implementation, two owners.
//
// The argv COMPOSITION deliberately stays on `ClusterProvisioner` as pure
// `nonisolated static` builders. It is already the shared, exhaustively
// asserted command vocabulary; re-typing it here is exactly the "second
// cluster implementation" the plan's review checklist warns about.
//
// Security: no operation takes, returns, logs, or stores a credential. The
// token import reads the remote file straight into the secret store and
// returns only whether it succeeded.
// =============================================================================

// MARK: - Seams

/// One completed command. Every external process the lifecycle runs goes
/// through this seam, so the whole engine is testable with canned transcripts.
public struct ClusterShellResult: Sendable, Equatable {
    public var exitCode: Int32
    public var lines: [String]

    public init(exitCode: Int32, lines: [String] = []) {
        self.exitCode = exitCode
        self.lines = lines
    }

    public var text: String {
        lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public var succeeded: Bool { exitCode == 0 }
}

/// The injectable process seam for headless lifecycle work.
public protocol ClusterShellRunner: Sendable {
    func run(_ argv: [String]) async -> ClusterShellResult
}

/// Adapter over the wizard's existing streaming runner, so the live path uses
/// `SystemProvisionRunner` and tests reuse the scripted runners they already
/// have.
public struct ProvisionShellRunner: ClusterShellRunner {
    private let runner: any ProvisionCommandRunner
    /// Optional live tee. The wizard passes one so its observable transcript
    /// still fills in as a long command runs; headless callers pass none.
    private let sink: (@Sendable (String) -> Void)?

    public init(
        _ runner: any ProvisionCommandRunner = SystemProvisionRunner(),
        sink: (@Sendable (String) -> Void)? = nil
    ) {
        self.runner = runner
        self.sink = sink
    }

    public func run(_ argv: [String]) async -> ClusterShellResult {
        let buffer = ProvisionLineBuffer()
        let sink = self.sink
        do {
            let code = try await runner.run(argv) { line in
                buffer.append(line)
                sink?(line)
            }
            return ClusterShellResult(exitCode: code, lines: buffer.snapshot())
        } catch {
            // A process that cannot START is not a remote failure — say so
            // rather than inventing an exit status the far side never gave.
            return ClusterShellResult(
                exitCode: 127,
                lines: buffer.snapshot() + ["could not run: \(error.localizedDescription)"])
        }
    }
}

/// What answered at an endpoint. Identity, not "the port is open".
public struct ClusterEndpointProbeResult: Sendable, Equatable {
    public var reachable: Bool
    public var authFailed: Bool
    public var serverBuild: String?
    public var serverRole: String?
    public var root: String?
    public var detail: String?

    public init(
        reachable: Bool, authFailed: Bool = false, serverBuild: String? = nil,
        serverRole: String? = nil, root: String? = nil, detail: String? = nil
    ) {
        self.reachable = reachable
        self.authFailed = authFailed
        self.serverBuild = serverBuild
        self.serverRole = serverRole
        self.root = root
        self.detail = detail
    }
}

public protocol ClusterEndpointProbe: Sendable {
    /// `token` is passed straight to the transport and never retained,
    /// echoed, or logged by any conformer.
    func probe(baseURL: URL, token: String?) async -> ClusterEndpointProbeResult
}

/// Live probe: `GET /api/capabilities` through the existing typed client.
public struct HTTPClusterEndpointProbe: ClusterEndpointProbe {
    public init() {}

    public func probe(baseURL: URL, token: String?) async -> ClusterEndpointProbeResult {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: baseURL), token: token)
        do {
            let capabilities = try await client.capabilities()
            return ClusterEndpointProbeResult(
                reachable: true,
                serverBuild: [capabilities.engine, capabilities.serverVersion]
                    .compactMap { $0 }
                    .joined(separator: " ")
                    .trimmingCharacters(in: .whitespaces),
                serverRole: capabilities.serverRole,
                root: capabilities.root)
        } catch ClusterClient.ClientError.badResponse(let code, let body) {
            if code == 401 || code == 403 {
                return ClusterEndpointProbeResult(reachable: false, authFailed: true)
            }
            return ClusterEndpointProbeResult(
                reachable: false, detail: "HTTP \(code): \(body.prefix(200))")
        } catch {
            return ClusterEndpointProbeResult(
                reachable: false, detail: error.localizedDescription)
        }
    }
}

/// The Keychain seam. Values enter and leave ONLY through this protocol, so
/// the lifecycle's "no secret ever reaches a record" property is testable with
/// a fake that holds a known token.
public protocol ClusterSecretStore: Sendable {
    /// Reads the SECRET. Only for callers that will actually authenticate with
    /// it — on the real Keychain this is an ACL-gated access that can put up a
    /// system password prompt (see `ClusterTokenStore.presence`).
    func token(forKey key: String) -> String?
    /// Answers "is a token stored?" WITHOUT reading it, so presence claims —
    /// every envelope's `tokenAvailable` on a read-only listing verb — cannot
    /// block an unattended agent on a keychain prompt. Deliberately has no
    /// default implementation: a conformer that forwards to `token(forKey:)`
    /// rebuilds exactly the bug this seam exists to prevent, so every
    /// conformer must say so explicitly.
    func hasToken(forKey key: String) -> Bool
    func store(_ token: String, forKey key: String) throws
    func removeToken(forKey key: String)
}

/// The dev-checkout default (plan §0.2): the app and CLI run unsigned from one
/// user account, so same-user Keychain access suffices. There is deliberately
/// no plaintext fallback.
public struct KeychainClusterSecretStore: ClusterSecretStore {
    public init() {}

    public func token(forKey key: String) -> String? {
        ClusterTokenStore.load(key: key)
    }

    public func hasToken(forKey key: String) -> Bool {
        ClusterTokenStore.presence(key: key)
    }

    public func store(_ token: String, forKey key: String) throws {
        try ClusterTokenStore.save(token, key: key)
    }

    public func removeToken(forKey key: String) {
        ClusterTokenStore.delete(key: key)
    }
}

/// Opening the visible authentication Terminal. The CLI may open it; it may
/// never read it, type into it, or select a Duo option (plan §4.1).
public protocol ClusterAuthenticationLauncher: Sendable {
    func authenticationCommand(for site: ClusterSiteProfile) -> String?
    /// True when a Terminal window was opened.
    func openAuthenticationTerminal(for site: ClusterSiteProfile) async -> Bool
}

public struct TerminalAuthenticationLauncher: ClusterAuthenticationLauncher {
    public init() {}

    public func authenticationCommand(for site: ClusterSiteProfile) -> String? {
        ClusterTunnel.authenticationCommand(for: site)
    }

    public func openAuthenticationTerminal(for site: ClusterSiteProfile) async -> Bool {
        guard let command = ClusterTunnel.authenticationCommand(for: site) else {
            return false
        }
        do {
            let url = try ClusterTunnel.writeAuthCommandFile(
                command: command, siteName: site.name)
            #if canImport(AppKit)
            return await MainActor.run { NSWorkspace.shared.open(url) }
            #else
            _ = url
            return false
            #endif
        } catch {
            return false
        }
    }
}

// MARK: - Configuration

/// Everything the provisioning steps need beyond the site profile — the
/// wizard's editable fields, as plain data an operation record can stamp.
public struct ClusterProvisioningConfiguration: Codable, Sendable, Equatable {
    /// Local payload root the server bundle is pushed FROM.
    public var localPayloadPath: String
    /// Remote bundle root containing `Server/` (bootstrap's `--repo`).
    public var remoteRepoPath: String
    public var envPrefix: String
    public var pythonVersion: String
    public var bootstrapExecutionTarget: BootstrapExecutionTarget
    public var bootstrapJobPartition: String
    public var bootstrapJobCPUs: Int
    public var bootstrapJobMemory: String
    public var bootstrapJobWalltime: String
    /// Some sites ship a wrapper instead of raw `squeue`.
    public var squeueCommand: String
    public var bootstrapForce: Bool
    public var bootstrapHello: Bool
    /// The environment file the validate and controller steps source. The
    /// bootstrap report names it once bootstrap has run; before that it is
    /// the script's own default. Deliberately NOT part of the plan hash — it
    /// is an output of bootstrap, not an input to it.
    public var bootstrapEnvFile: String
    /// **WP5 Step 7 — materialization. DEFAULT ON.**
    ///
    /// On (the default since Step 7): `bootstrapApply` pushes
    /// `ClusterEnvironmentRenderer.renderEnvFile`'s output to
    /// `ClusterProvisioner.renderedEnvironmentPath(for:)` and invokes bootstrap
    /// with `--env-file-from <that path> --env-file-sha256 <digest>`. The digest
    /// is part of the argv, hence part of the reviewed plan hash, so approving a
    /// plan approves the environment (audit §3.3, §6.4). The env file the
    /// cluster sources is then the SITE PROFILE's — including the facts
    /// `bootstrap.sh` never knew (archive root, purge window, GPU vocabulary,
    /// compute egress: audit a1–a7).
    ///
    /// The bytes do not change for an existing site: a schema-1 profile renders
    /// the renderer's `legacyV1` default set, which reproduces `bootstrap.sh`'s
    /// own constants (pinned by
    /// `ClusterEnvironmentRendererTests.v1MaximalRendersTheGoldenEnvFile`).
    /// What changes is WHERE they come from.
    ///
    /// Off: the historical argv, word for word, and `bootstrap.sh` writes its
    /// built-in fallback heredoc instead — values that are the script's
    /// defaults, not this site's declared facts. That is the manual path, and
    /// the plan transcript says so out loud.
    public var materializeEnvironmentFile: Bool

    public init(
        localPayloadPath: String = ClusterProvisioner.defaultLocalRepoPath(),
        remoteRepoPath: String = "~/steerlab",
        envPrefix: String = "",
        pythonVersion: String = "",
        bootstrapExecutionTarget: BootstrapExecutionTarget = .sshHost,
        bootstrapJobPartition: String = "",
        bootstrapJobCPUs: Int = 4,
        bootstrapJobMemory: String = "16G",
        bootstrapJobWalltime: String = "02:00:00",
        squeueCommand: String = "squeue",
        bootstrapForce: Bool = false,
        bootstrapHello: Bool = false,
        bootstrapEnvFile: String = ClusterProvisioningConfiguration.defaultEnvFile,
        materializeEnvironmentFile: Bool = true
    ) {
        self.localPayloadPath = localPayloadPath
        self.remoteRepoPath = remoteRepoPath
        self.envPrefix = envPrefix
        self.pythonVersion = pythonVersion
        self.bootstrapExecutionTarget = bootstrapExecutionTarget
        self.bootstrapJobPartition = bootstrapJobPartition
        self.bootstrapJobCPUs = bootstrapJobCPUs
        self.bootstrapJobMemory = bootstrapJobMemory
        self.bootstrapJobWalltime = bootstrapJobWalltime
        self.squeueCommand = squeueCommand
        self.bootstrapForce = bootstrapForce
        self.bootstrapHello = bootstrapHello
        self.bootstrapEnvFile = bootstrapEnvFile
        self.materializeEnvironmentFile = materializeEnvironmentFile
    }

    /// The wizard's own defaults for a site (Slurm sites bootstrap in a CPU
    /// job; everything else on the SSH host). Since WP5 Step 9 the setup job's
    /// RESOURCES come from the site too (`scheduler.setupJob`, audit c19) —
    /// they were wizard constants, so a site could describe its bootstrap job
    /// and be ignored.
    public static func defaults(for site: ClusterSiteProfile?) -> Self {
        let placement = ClusterProvisioner.bootstrapExecutionDefaults(for: site)
        var configuration = ClusterProvisioningConfiguration()
        configuration.bootstrapExecutionTarget = placement.target
        configuration.bootstrapJobPartition = placement.partition
        if let site, case .slurm(let slurm) = site.scheduler {
            let setup = ClusterProvisioner.setupJobDefaults(for: slurm)
            configuration.bootstrapJobCPUs = setup.cpus
            configuration.bootstrapJobMemory = setup.memory
            configuration.bootstrapJobWalltime = setup.walltime
        }
        return configuration
    }

    /// Env file / prefix the validate and controller steps source. The
    /// bootstrap report supplies them when one is at hand; otherwise the
    /// script's own default.
    public static let defaultEnvFile = "~/steerlab-cluster.env"
}

// MARK: - Outcomes

/// The common shape of a completed operation: did it work, what to say about
/// it, and the transcript (redacted by construction — no operation here ever
/// puts a secret on a command line).
public struct ClusterOperationOutcome: Sendable, Equatable {
    public var succeeded: Bool
    public var message: String
    public var transcript: [String]
    public var exitCode: Int32
    /// The step did not apply here (direct transport, wrong topology). Kept
    /// distinct from success so the wizard can stamp it LOUDLY — a skip is
    /// never silent.
    public var wasSkipped: Bool

    public init(
        succeeded: Bool, message: String, transcript: [String] = [],
        exitCode: Int32 = 0, wasSkipped: Bool = false
    ) {
        self.succeeded = succeeded
        self.message = message
        self.transcript = transcript
        self.exitCode = exitCode
        self.wasSkipped = wasSkipped
    }

    public static func skipped(_ reason: String) -> Self {
        ClusterOperationOutcome(succeeded: true, message: reason, wasSkipped: true)
    }
}

public struct ClusterBootstrapPlanOutcome: Sendable, Equatable {
    /// Canonical hash of the reviewed plan. Nil when the dry run did not
    /// produce a machine-readable plan — there is then nothing to approve.
    public var planHash: String?
    public var report: BootstrapReport?
    public var outcome: ClusterOperationOutcome
}

/// What `submit-bootstrap-job.sh` said when it returned from submitting.
///
/// The helper prints these markers immediately after `sbatch` and then EXITS
/// (review finding 5, 2026-08-13) — the job id reaches us before anything can
/// be interrupted, which is exactly what the old streaming wait could lose.
public struct ClusterBootstrapSubmission: Sendable, Equatable {
    public var jobID: String
    public var statusFile: String?
    /// The helper found a bootstrap job already in flight for this workspace
    /// and attached to it instead of queueing a second one.
    public var adopted: Bool

    public init(jobID: String, statusFile: String? = nil, adopted: Bool = false) {
        self.jobID = jobID
        self.statusFile = statusFile
        self.adopted = adopted
    }

    static func value(ofMarker marker: String, in lines: [String]) -> String? {
        let prefix = marker + "="
        return lines.last { $0.hasPrefix(prefix) }
            .map { String($0.dropFirst(prefix.count)) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .flatMap { $0.isEmpty ? nil : $0 }
    }

    public static func parse(lines: [String]) -> ClusterBootstrapSubmission? {
        guard let jobID = value(ofMarker: "STEERLAB_BOOTSTRAP_JOB_ID", in: lines)
        else { return nil }
        return ClusterBootstrapSubmission(
            jobID: jobID,
            statusFile: value(ofMarker: "STEERLAB_BOOTSTRAP_STATUS_FILE", in: lines),
            adopted: value(ofMarker: "STEERLAB_BOOTSTRAP_ADOPT", in: lines) != nil)
    }
}

/// One reading of a submitted bootstrap job, from the helper's `--status`
/// mode. Each reading is a separate short-lived command, so an interrupted
/// poll loop costs nothing but the poll.
public enum ClusterBootstrapJobVerdict: Sendable, Equatable {
    case pending(state: String?)
    case running(state: String?)
    /// The job wrote a status record. `exitCode == 0` is a successful
    /// bootstrap.
    case completed(exitCode: Int)
    /// The job left the queue without writing a status record.
    case vanished(state: String?)
    /// The status could not be read. NEVER treated as death — an unproven
    /// death never licenses a resubmit.
    case unknown(reason: String)

    /// Whether this verdict ends the poll. `unknown` deliberately does not.
    public var isTerminal: Bool {
        switch self {
        case .completed, .vanished: true
        case .pending, .running, .unknown: false
        }
    }

    public var summary: String {
        switch self {
        case .pending(let state): "queued\(state.map { " (\($0))" } ?? "")"
        case .running(let state): "running\(state.map { " (\($0))" } ?? "")"
        case .completed(let code):
            code == 0 ? "completed successfully" : "completed with exit \(code)"
        case .vanished(let state):
            "left the queue without a status record"
                + (state.map { " (last state \($0))" } ?? "")
        case .unknown(let reason): "state unknown: \(reason)"
        }
    }

    /// Parse the helper's single verdict line:
    /// `STEERLAB_BOOTSTRAP_STATUS=<verdict> job=<id> [exit=N] [statusFile=…]
    /// [state=…]`. Later lines (the forwarded job log) cannot displace it.
    public static func parse(lines: [String]) -> ClusterBootstrapJobVerdict? {
        let marker = "STEERLAB_BOOTSTRAP_STATUS="
        guard let line = lines.last(where: {
            $0.trimmingCharacters(in: .whitespaces).hasPrefix(marker)
        }) else { return nil }
        let fields = line.trimmingCharacters(in: .whitespaces)
            .split(separator: " ")
            .map(String.init)
        guard let head = fields.first else { return nil }
        let verdict = String(head.dropFirst(marker.count))
        var values: [String: String] = [:]
        for field in fields.dropFirst() {
            let parts = field.split(separator: "=", maxSplits: 1).map(String.init)
            if parts.count == 2 { values[parts[0]] = parts[1] }
        }
        let state = values["state"]
        switch verdict {
        case "pending": return .pending(state: state)
        case "running": return .running(state: state)
        case "completed-ok": return .completed(exitCode: 0)
        case "vanished": return .vanished(state: state)
        case "unknown":
            return .unknown(reason: "the cluster could not read the job's state")
        default:
            let prefix = "completed-code-"
            guard verdict.hasPrefix(prefix),
                let code = Int(verdict.dropFirst(prefix.count))
            else { return nil }
            return .completed(exitCode: code)
        }
    }
}

public struct ClusterBootstrapApplyOutcome: Sendable, Equatable {
    public var jobID: String?
    public var report: BootstrapReport?
    public var outcome: ClusterOperationOutcome
    /// Where the job records its exit code, when a scheduler job carries it.
    /// Persisted with the job id so a later process can poll without
    /// re-deriving the path.
    public var statusFile: String?
    /// The apply attached to a job that was ALREADY in flight — either from
    /// this app's durable record or from the helper's own cluster-side
    /// breadcrumb — instead of submitting a second one.
    public var adopted: Bool = false
    /// The job is submitted and durably recorded but has not reached a
    /// verdict inside this invocation's poll budget. NOT a failure: running
    /// the step again resumes the same job.
    public var stillPending: Bool = false
    /// The last verdict the status probe returned, when there was a job.
    public var verdict: ClusterBootstrapJobVerdict?
}

public struct ClusterValidateOutcome: Sendable, Equatable {
    public var observation: ClusterProfileValidationObservation
    public var lines: [ProfileValidateLine]
    public var outcome: ClusterOperationOutcome
}

public struct ClusterControllerStartOutcome: Sendable, Equatable {
    public var jobID: String?
    public var outcome: ClusterOperationOutcome
}

/// What `cluster controller adopt` learned about a hand-started job.
///
/// `verified` is the ONLY thing that licenses recording the job id. Everything
/// else is reported so the refusal names the layer that failed.
public struct ClusterControllerAdoptionOutcome: Sendable, Equatable {
    public var jobID: String
    public var verified: Bool
    /// Nil when verification passed.
    public var refusalReason: String?
    public var controller: ClusterControllerObservation
    public var daemonHost: ClusterDaemonHostObservation
    /// What the endpoint said, when there was one to ask. Nil means no forward
    /// is up yet — which is not a failure, just nothing to prove there.
    public var serverHTTP: ClusterServerHTTPObservation?
    public var outcome: ClusterOperationOutcome
}

/// What the token import did. There is no case that carries the token.
public enum ClusterTokenImportOutcome: Sendable, Equatable {
    case imported
    case alreadyPresent
    case unavailable(reason: String)

    public var succeeded: Bool {
        switch self {
        case .imported, .alreadyPresent: true
        case .unavailable: false
        }
    }
}

// MARK: - Operations

/// The headless provisioning operations. A value type: hold one, call it from
/// anywhere, inject a scripted shell in tests.
public struct ClusterProvisioningOperations: Sendable {

    public let shell: any ClusterShellRunner
    public let secrets: any ClusterSecretStore
    /// Gap between bootstrap status probes. Each probe is its own short-lived
    /// command, so this is a cadence, not a held connection. Tests use `.zero`.
    public let bootstrapPollDelay: Duration
    /// How many probes one invocation makes before returning "still pending"
    /// with the job durably recorded. The default covers the helper's default
    /// two-hour setup walltime; exhausting it is not a failure.
    public let bootstrapPollLimit: Int

    public init(
        shell: any ClusterShellRunner,
        secrets: any ClusterSecretStore = KeychainClusterSecretStore(),
        bootstrapPollDelay: Duration = .seconds(15),
        bootstrapPollLimit: Int = 480
    ) {
        self.shell = shell
        self.secrets = secrets
        self.bootstrapPollDelay = bootstrapPollDelay
        self.bootstrapPollLimit = bootstrapPollLimit
    }

    /// Where bootstrap writes the server's API token. Read over the
    /// authenticated ControlMaster straight into the secret store.
    public static let remoteTokenPath = "~/.steerlab-token"

    // MARK: Authenticate (observation only — the human owns the credential)

    /// `ssh -O check`, then one cheap REAL command through the master. This
    /// is the ENTIRE extent of the lifecycle's involvement in authentication.
    ///
    /// The second probe is load-bearing: a master that survives a network
    /// change (observed live 2026-08-12) keeps answering `-O check` from its
    /// local socket while every actual command fails — and an "alive" answer
    /// here licenses the whole remote probe cascade. Only an executed command
    /// proves the mux still carries sessions.
    public func checkControlMaster(
        site: ClusterSiteProfile
    ) async -> ClusterControlMasterObservation {
        guard case .ssh(let host, _, _, _) = site.transport else { return .notApplicable }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .absent }
        let result = await shell.run(ClusterProvisioner.masterCheckArgv(host: trimmed))
        guard result.succeeded else {
            // 255 is ssh's own "could not talk to the master" — the socket may
            // exist but be wedged; anything else means no master at all.
            return result.exitCode == 255
                && result.text.localizedCaseInsensitiveContains("control")
                ? .unresponsive : .absent
        }
        let command = await shell.run(
            ClusterProvisioner.masterCommandProbeArgv(host: trimmed))
        return command.succeeded ? .alive : .unresponsive
    }

    // MARK: Deployment identity (§7.8)

    /// The local payload's identity: its deployment manifest's content hash,
    /// or nil for a plain dev checkout (which has no manifest and therefore
    /// no reproducible identity — reported honestly, never faked).
    public static func localDeploymentIdentity(payloadRoot: String) -> String? {
        let url = URL(filePath: payloadRoot)
            .appending(component: ClusterProvisioner.deploymentManifestFileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return ClusterSupportPaths.sha256Hex(data)
    }

    /// The python package's path relative to the payload root — where the
    /// engine's `BUILD_COMMIT` stamp lives (`build_identity.py` reads it as
    /// the deployed build identity). Nil when the payload holds no package.
    static func packageSubpath(payloadRoot: String) -> String? {
        for candidate in ["Server/steerlab_server", "steerlab_server"]
        where FileManager.default.fileExists(
            atPath: URL(filePath: payloadRoot)
                .appending(path: candidate).path)
        {
            return candidate
        }
        return nil
    }

    /// A DEV checkout's payload identity: `<sha8>` of the checkout's HEAD,
    /// `-dirty` when the pushed subtree has uncommitted changes — the same
    /// vocabulary `build_identity.py` speaks, because the remote
    /// `BUILD_COMMIT` stamp is both the engine's build identity and the
    /// deployed-payload marker this identity is compared against. Nil when
    /// the payload root is not a git checkout (then nothing can be proven,
    /// and the observation stays honestly unknown).
    ///
    /// Dirty payloads compare loosely (two different dirts share a stamp);
    /// the `-dirty` suffix is the warning label, per plan §7.8: "a dirty
    /// development payload must be identified honestly."
    func devPayloadStamp(payloadRoot: String) async -> String? {
        let head = await shell.run(
            ["/usr/bin/git", "-C", payloadRoot, "rev-parse", "--short=8", "HEAD"])
        guard head.succeeded else { return nil }
        let sha = head.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sha.isEmpty else { return nil }
        // Dirty is scoped to what the push actually ships (the server
        // subtree), mirroring build_identity.py's own scoping — an edited
        // SwiftUI view must not mark the PYTHON payload dirty.
        let scope = Self.packageSubpath(payloadRoot: payloadRoot)
            .map { $0.hasPrefix("Server/") ? "Server" : $0 } ?? "."
        let status = await shell.run(
            ["/usr/bin/git", "-C", payloadRoot, "status", "--porcelain",
             "--", scope])
        let dirty = status.succeeded
            && !status.text.trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
        return dirty ? sha + "-dirty" : sha
    }

    /// The remote path of the engine's `BUILD_COMMIT` stamp for this
    /// configuration, or nil when the local payload carries no package (there
    /// is then nothing to stamp or compare).
    func remoteBuildCommitPath(
        configuration: ClusterProvisioningConfiguration
    ) -> String? {
        let localPath = configuration.localPayloadPath
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let subpath = Self.packageSubpath(payloadRoot: localPath) else {
            return nil
        }
        return configuration.remoteRepoPath + "/" + subpath + "/BUILD_COMMIT"
    }

    /// Compare the local payload against the deployed marker.
    public func observePayload(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) async -> ClusterPayloadObservation {
        guard site.isSSHTransport else { return .notApplicable }
        let localPath = configuration.localPayloadPath
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !localPath.isEmpty else {
            return .unknown(reason: "no local server payload is configured")
        }
        let remoteManifest = configuration.remoteRepoPath
            + "/" + ClusterProvisioner.deploymentManifestFileName
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site, remoteWords: ["cat", remoteManifest])
        guard !argv.isEmpty else {
            return .unknown(reason: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        guard let local = Self.localDeploymentIdentity(payloadRoot: localPath) else {
            // A development checkout has no manifest — compare git identity
            // against the deployed BUILD_COMMIT stamp instead (the stamp a
            // successful push writes). Without this, a dev checkout could
            // never prove currency and `ensure --target connected` was
            // unreachable without pushing on EVERY invocation (found live,
            // 2026-08-11 shakedown).
            return await observeDevPayload(
                site: site, configuration: configuration,
                payloadRoot: localPath)
        }
        guard result.succeeded, !result.text.isEmpty else {
            return .absent
        }
        let remote = ClusterSupportPaths.sha256Hex(Data(result.text.utf8))
        return remote == local
            ? .current(deploymentHash: local)
            : .stale(reason: "the deployed payload does not match the local one")
    }

    /// Dev-checkout payload comparison: local git stamp vs the deployed
    /// `BUILD_COMMIT`. Every mismatch names both sides — "push to be sure"
    /// is only ever suggested about a payload we could actually read.
    private func observeDevPayload(
        site: ClusterSiteProfile,
        configuration: ClusterProvisioningConfiguration,
        payloadRoot: String
    ) async -> ClusterPayloadObservation {
        guard let local = await devPayloadStamp(payloadRoot: payloadRoot) else {
            return .unknown(
                reason: "the local payload is a development checkout with no "
                    + "deployment manifest and no readable git identity, so "
                    + "its identity cannot be proven")
        }
        guard let remotePath = remoteBuildCommitPath(
            configuration: configuration)
        else {
            return .unknown(
                reason: "the local payload carries no python package to stamp")
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site, remoteWords: ["cat", remotePath])
        guard !argv.isEmpty else {
            return .unknown(reason: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        guard result.succeeded else {
            return .unknown(
                reason: "the deployed tree carries no BUILD_COMMIT stamp to "
                    + "compare — one push stamps it")
        }
        let remote = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !remote.isEmpty else {
            return .unknown(
                reason: "the deployed BUILD_COMMIT stamp is empty — one push "
                    + "restamps it")
        }
        return remote == local
            ? .current(deploymentHash: local)
            : .stale(
                reason: "deployed \(remote) does not match local \(local)")
    }

    // MARK: Push

    public func push(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) async -> ClusterOperationOutcome {
        let localPath = configuration.localPayloadPath
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !localPath.isEmpty else {
            let reason = (try? CodeResources.clusterPayload().path).map {
                "\($0) resolved but the local payload path is blank — set it"
            } ?? "no deployable server payload is available in this build"
            return ClusterOperationOutcome(succeeded: false, message: reason)
        }
        guard let argv = ClusterProvisioner.pushArgv(
            site: site, localRepoPath: localPath,
            remoteRepoPath: configuration.remoteRepoPath,
            includeDeploymentManifest: ClusterProvisioner.deploymentManifestExists(
                atPayloadRoot: localPath))
        else {
            return .skipped(
                "no SSH path to push over (direct transport or blank host) — "
                    + "place the server bundle on the box yourself")
        }
        // Verify the payload against its manifest BEFORE any bytes move, so a
        // drifted payload never reaches the cluster.
        if let failure = ClusterProvisioner.deploymentManifestFailure(
            atPayloadRoot: localPath)
        {
            return ClusterOperationOutcome(
                succeeded: false, message: failure, transcript: [failure])
        }
        let result = await shell.run(argv)
        var transcript = ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        guard result.succeeded else {
            return ClusterOperationOutcome(
                succeeded: false,
                message: "rsync exited \(result.exitCode) — is the ControlMaster still alive?",
                transcript: transcript, exitCode: result.exitCode)
        }
        let hostLabel: String
        if case .ssh(let host, _, _, _) = site.transport {
            hostLabel = host.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            hostLabel = "?"
        }

        // The push template's `--delete` removes remote files the payload
        // does not carry — including the BUILD_COMMIT stamp every deploy
        // must (re)write, or the engine records runs with no build identity
        // (the first live CLI push silently un-stamped the engine,
        // 2026-08-11 shakedown). Stamp in the same operation. The stamp is
        // also the deployed-payload marker `observePayload` compares dev
        // checkouts against, so a stamped push reads back `current`.
        var stampNote = ""
        if let remotePath = remoteBuildCommitPath(configuration: configuration) {
            if let stamp = await devPayloadStamp(payloadRoot: localPath) {
                // Both words are shell-safe by construction (a short git hex
                // stamp; a config path) and the path must stay UNQUOTED in
                // the far shell so a `~/` prefix still expands there.
                let stampArgv = ClusterProvisioner.sshRemoteArgv(
                    site: site,
                    remoteWords: ["sh", "-c",
                                  "printf %s \(stamp) > \(remotePath)"])
                let stamped = await shell.run(stampArgv)
                transcript.append(
                    "$ " + ClusterProvisioner.displayCommand(stampArgv))
                if stamped.succeeded {
                    stampNote = ", BUILD_COMMIT stamped \(stamp)"
                } else {
                    transcript.append(contentsOf: stamped.lines)
                    stampNote = " — WARNING: the BUILD_COMMIT stamp could not "
                        + "be written (exit \(stamped.exitCode)); the deployed "
                        + "engine has no build identity until it is"
                }
            } else {
                stampNote = " — note: no git identity to stamp (payload is "
                    + "neither packaged nor a git checkout)"
            }
        }
        // A deploy ships a new controller-job TEMPLATE and leaves last week's
        // RENDERED child of it sitting in the metadata root, re-submittable by
        // hand — the gap that cost serverd 47564632 its successor
        // (open-issues §1 field report, 2026-08-20). This flow may write, so
        // it repairs rather than merely reporting: the rendered artifact is
        // brought back into step with the template that just landed. Nothing
        // is submitted, and a running controller is untouched (its successor
        // reads the file at USR1 time, which is the point).
        let renderNote = await refreshRenderedControllerScript(
            site: site, configuration: configuration, transcript: &transcript)
        return ClusterOperationOutcome(
            succeeded: true,
            message: "server bundle pushed to "
                + "\(hostLabel):\(configuration.remoteRepoPath)\(stampNote)"
                + renderNote,
            transcript: transcript)
    }

    /// Post-deploy repair of the rendered controller script. Returns the
    /// sentence to append to the push message (empty when there is nothing to
    /// say), and appends its own commands to the push transcript.
    ///
    /// It NEVER fails the push: a deploy that landed is a deploy that landed.
    /// A render that could not run becomes a WARNING naming the one re-render
    /// command, because the alternative — silence — is the failure mode this
    /// whole change exists to end.
    private func refreshRenderedControllerScript(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        transcript: inout [String]
    ) async -> String {
        guard site.topology == .daemonInJob, site.isSSHTransport else { return "" }
        let observed = await observeControllerScript(
            site: site, configuration: configuration)
        switch observed {
        case .current, .notApplicable:
            return ""
        case .absent:
            // Nothing rendered yet: `controller start` will render one. Not a
            // defect, and not this verb's business to pre-create.
            return ""
        case .unknown(let reason):
            transcript.append("controller script: \(reason)")
            return " — NOTE: the rendered controller script could not be "
                + "checked against the template just deployed (\(reason)); "
                + "re-render before the next controller cycle with "
                + "`\(ClusterProvisioner.renderControllerCommand(siteID: ""))`"
        case .stale(let reason):
            let envFile = ClusterLifecycleInspector.envFile(for: configuration)
            var prefix: String?
            if case .valid(_, let value) = await observeBootstrap(
                site: site, configuration: configuration, envFile: envFile)
            {
                prefix = value
            }
            let outcome = await controllerRender(
                site: site, configuration: configuration,
                envFile: envFile, prefix: prefix)
            transcript.append(contentsOf: outcome.transcript)
            guard outcome.succeeded else {
                return " — WARNING: the rendered controller script is STALE "
                    + "(\(reason)) and the automatic re-render failed "
                    + "(\(outcome.message)). A hand `sbatch` of it launches a "
                    + "controller that may not self-chain at walltime — "
                    + "re-render with "
                    + "`\(ClusterProvisioner.renderControllerCommand(siteID: ""))`"
            }
            return " — the stale rendered controller script was re-rendered "
                + "against the template just deployed (was: \(reason))"
        }
    }

    // MARK: Bootstrap (§7.9 — the dry-run-before-real gate, made durable)

    /// The canonical plan hash for a site + configuration. Any site, payload,
    /// root, resource, or command change invalidates it — it is a hash of the
    /// exact composed dry-run command PLUS the site profile's own hash.
    ///
    /// **WP5 §3.3 / §6.4:** under materialization (the default since Step 7)
    /// the composed command carries `--env-file-sha256 <digest of the rendered
    /// env file>`, so the rendered environment is INSIDE this hash and approving
    /// a plan approves the environment, not just the argv. A profile edit that
    /// only the ENVIRONMENT sees — a module list, an archive root — therefore
    /// invalidates the plan, which is the firewall working. With materialization
    /// off the flag is absent and this value is the pre-WP5 one.
    public static func bootstrapPlanHash(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) -> String? {
        guard let argv = bootstrapArgv(
            site: site, configuration: configuration, dryRun: true)
        else { return nil }
        let profileHash = (try? site.encoded()).map(ClusterSupportPaths.sha256Hex) ?? ""
        let canonical = ([profileHash] + argv).joined(separator: "\u{1F}")
        return ClusterSupportPaths.sha256Hex(Data(canonical.utf8))
    }

    /// The rendered environment this configuration implies, or nil when
    /// materialization is off — the manual, no-profile path, on which
    /// `bootstrap.sh` falls back to its own built-in heredoc.
    static func renderedEnvironment(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) -> ClusterProvisioner.RenderedEnvironmentPlan? {
        guard configuration.materializeEnvironmentFile else { return nil }
        return ClusterProvisioner.renderedEnvironmentPlan(for: site)
    }

    static func bootstrapArgv(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        dryRun: Bool, forceNewJob: Bool = false
    ) -> [String]? {
        ClusterProvisioner.bootstrapArgv(
            site: site,
            remoteRepoPath: configuration.remoteRepoPath,
            envPrefix: configuration.envPrefix,
            pythonVersion: configuration.pythonVersion,
            executionTarget: configuration.bootstrapExecutionTarget,
            jobPartition: configuration.bootstrapJobPartition,
            jobCPUs: configuration.bootstrapJobCPUs,
            jobMemory: configuration.bootstrapJobMemory,
            jobWalltime: configuration.bootstrapJobWalltime,
            squeueCommand: configuration.squeueCommand,
            force: configuration.bootstrapForce,
            hello: configuration.bootstrapHello,
            dryRun: dryRun,
            renderedEnvironment: renderedEnvironment(
                site: site, configuration: configuration),
            forceNewJob: forceNewJob)
    }

    /// Render + run the dry run. Returns the plan's hash so the caller can
    /// persist it: the review must survive process exit.
    public func bootstrapPlan(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) async -> ClusterBootstrapPlanOutcome {
        guard let argv = Self.bootstrapArgv(
            site: site, configuration: configuration, dryRun: true)
        else {
            return ClusterBootstrapPlanOutcome(
                planHash: nil, report: nil,
                outcome: .skipped(
                    "direct transport — run Server/scripts/bootstrap.sh on the box "
                        + "yourself"))
        }
        let result = await shell.run(argv)
        // Under materialization the plan a human approves must name the
        // environment it approves. The dry run itself prints the install step
        // (bootstrap.sh's `[envFile] install …`); this line names the digest so
        // it can be compared against `steerlab-cli cluster preview`, which
        // prints the same render's bytes and the same hash.
        //
        // WP5 Step 7: materialization is the default, so the interesting case
        // is now the OPT-OUT. A plan that will let `bootstrap.sh` synthesize
        // its own environment must say so — silently shipping the script's
        // fallback constants to a site whose profile declares otherwise is the
        // split-brain this step retired.
        var preamble: [String] = []
        if let plan = Self.renderedEnvironment(site: site, configuration: configuration) {
            preamble = [
                "bootstrap: materializing the site environment — \(plan.remotePath) "
                    + "(sha256 \(plan.sha256)) will be pushed at apply, and this digest is "
                    + "part of the plan hash below"
            ]
        } else {
            preamble = [
                "bootstrap: WARNING — environment materialization is OFF for this plan, so "
                    + "bootstrap.sh will write its own built-in fallback env file. This site's "
                    + "declared GPU vocabulary, purge window, archive root, egress, and node "
                    + "staging will NOT reach the cluster. Re-plan without "
                    + "--no-materialize-env to install the rendered profile instead."
            ]
        }
        let transcript = preamble
            + ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        let report = BootstrapReport.parse(fromTranscript: result.lines)
        guard result.succeeded, report != nil else {
            return ClusterBootstrapPlanOutcome(
                planHash: nil, report: report,
                outcome: ClusterOperationOutcome(
                    succeeded: false,
                    message: report == nil
                        ? "dry run produced no machine-readable plan (exit \(result.exitCode))"
                        : "dry run exited \(result.exitCode)",
                    transcript: transcript, exitCode: result.exitCode))
        }
        return ClusterBootstrapPlanOutcome(
            planHash: Self.bootstrapPlanHash(site: site, configuration: configuration),
            report: report,
            outcome: ClusterOperationOutcome(
                succeeded: true,
                message: "bootstrap plan ready for review",
                transcript: transcript))
    }

    /// Run the real bootstrap — ONLY against the exact plan that was reviewed.
    /// A mismatched or missing hash throws before any command runs.
    ///
    /// On a SLURM site this is three separable acts, never one long-held SSH
    /// session (review finding 5, 2026-08-13):
    ///
    ///   1. SUBMIT — one short command that sbatches and returns the job id;
    ///   2. PERSIST — the pending job goes to the durable journal BEFORE the
    ///      first poll, so a Mac that sleeps here leaves an adoptable record;
    ///   3. POLL — repeated short `--status` commands, each its own
    ///      connection. Interrupting the poll loses nothing.
    ///
    /// A pending record therefore RESUMES rather than resubmits, and even
    /// without a journal the helper's own cluster-side breadcrumb refuses to
    /// queue a second bootstrap for the same workspace.
    ///
    /// The SSH-host execution target has no job to record and keeps its old
    /// single-command behaviour.
    public func bootstrapApply(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        reviewedPlanHash: String?,
        journal: (any ClusterBootstrapJobJournal)? = nil,
        siteID: String? = nil,
        forceNewJob: Bool = false
    ) async throws -> ClusterBootstrapApplyOutcome {
        guard let expected = Self.bootstrapPlanHash(
            site: site, configuration: configuration)
        else {
            return ClusterBootstrapApplyOutcome(
                jobID: nil, report: nil,
                outcome: .skipped(
                    "direct transport — run Server/scripts/bootstrap.sh on the box "
                        + "yourself"))
        }
        guard let reviewedPlanHash, !reviewedPlanHash.isEmpty else {
            throw ClusterLifecycleError.bootstrapPlanMissing
        }
        guard reviewedPlanHash == expected else {
            throw ClusterLifecycleError.bootstrapPlanMismatch(
                expected: expected, supplied: reviewedPlanHash)
        }
        guard let argv = Self.bootstrapArgv(
            site: site, configuration: configuration, dryRun: false,
            forceNewJob: forceNewJob)
        else {
            return ClusterBootstrapApplyOutcome(
                jobID: nil, report: nil, outcome: .skipped("nothing to bootstrap"))
        }
        // WP5 Step 6 — materialization, opt-in. The rendered file is pushed
        // AFTER the plan-hash gate and BEFORE anything is submitted: the bytes
        // whose digest the approved plan carries are the bytes that land, and
        // an unreviewed plan never gets this far.
        var prefixTranscript: [String] = []
        if let plan = Self.renderedEnvironment(site: site, configuration: configuration) {
            switch await pushRenderedEnvironment(site: site, plan: plan) {
            case .pushed(let lines):
                prefixTranscript = lines
            case .refused(let outcome):
                return ClusterBootstrapApplyOutcome(
                    jobID: nil, report: nil, outcome: outcome)
            }
        }
        if configuration.bootstrapExecutionTarget == .slurmBatch {
            return await bootstrapApplyViaScheduler(
                site: site, configuration: configuration, submitArgv: argv,
                planHash: reviewedPlanHash, journal: journal, siteID: siteID,
                forceNewJob: forceNewJob, transcriptPrefix: prefixTranscript)
        }
        let result = await shell.run(argv)
        let transcript = prefixTranscript
            + ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        let report = BootstrapReport.parse(fromTranscript: result.lines)
        if result.succeeded, let report, report.ok {
            var summary = "bootstrap ok"
            if let prefix = report.prefix { summary += " — env \(prefix)" }
            return ClusterBootstrapApplyOutcome(
                jobID: nil, report: report,
                outcome: ClusterOperationOutcome(
                    succeeded: true, message: summary, transcript: transcript))
        }
        let failed = report?.failedStepNames.joined(separator: ", ") ?? ""
        return ClusterBootstrapApplyOutcome(
            jobID: nil, report: report,
            outcome: ClusterOperationOutcome(
                succeeded: false,
                message: failed.isEmpty
                    ? "bootstrap exited \(result.exitCode)"
                    : "bootstrap step(s) failed: \(failed)",
                transcript: transcript, exitCode: result.exitCode))
    }

    /// Push the rendered env file to its staging path. Returns the transcript
    /// lines on success, or the failure outcome to return as-is.
    ///
    /// The transcript deliberately carries a DESCRIPTION rather than
    /// `displayCommand(argv)`: the argv embeds the whole env file in a heredoc,
    /// and a wizard transcript is not where a reader should meet it. The exact
    /// bytes are what `steerlab-cli cluster preview` prints, and the digest
    /// below identifies which render this was.
    private enum RenderedEnvironmentPushOutcome {
        case pushed([String])
        case refused(ClusterOperationOutcome)
    }

    private func pushRenderedEnvironment(
        site: ClusterSiteProfile, plan: ClusterProvisioner.RenderedEnvironmentPlan
    ) async -> RenderedEnvironmentPushOutcome {
        guard let argv = ClusterProvisioner.renderedEnvironmentPushArgv(
            site: site, plan: plan)
        else {
            return .refused(
                ClusterOperationOutcome(
                    succeeded: false,
                    message: "the rendered environment could not be composed for transfer "
                        + "— the site has no SSH transport, or a site fact contains the "
                        + "transfer delimiter; nothing was submitted",
                    exitCode: 64))
        }
        let description = "$ ssh … install rendered environment → \(plan.remotePath) "
            + "(sha256 \(plan.sha256))"
        let result = await shell.run(argv)
        guard result.succeeded else {
            return .refused(
                ClusterOperationOutcome(
                    succeeded: false,
                    message: "could not push the rendered environment to "
                        + "\(plan.remotePath) (exit \(result.exitCode)) — nothing was "
                        + "submitted",
                    transcript: [description] + result.lines, exitCode: result.exitCode))
        }
        return .pushed(
            [description] + result.lines
                + [
                    "bootstrap: bootstrap.sh will verify that digest before installing the "
                        + "file; a mismatch refuses rather than sourcing unreviewed bytes"
                ])
    }

    /// Submit → persist → poll, for the Slurm execution target.
    private func bootstrapApplyViaScheduler(
        site: ClusterSiteProfile,
        configuration: ClusterProvisioningConfiguration,
        submitArgv: [String],
        planHash: String,
        journal: (any ClusterBootstrapJobJournal)?,
        siteID: String?,
        forceNewJob: Bool,
        transcriptPrefix: [String] = []
    ) async -> ClusterBootstrapApplyOutcome {
        var transcript: [String] = transcriptPrefix
        var submission: ClusterBootstrapSubmission?

        // 1. RESUME. A durable pending record outranks submitting: the job it
        //    names is either still queued or already has a verdict waiting.
        if !forceNewJob, let journal, let siteID,
            let pending = journal.pendingBootstrapJob(forSite: siteID)
        {
            submission = ClusterBootstrapSubmission(
                jobID: pending.jobID, statusFile: pending.statusFile, adopted: true)
            transcript.append(
                "bootstrap: resuming job \(pending.jobID) from the durable record — "
                    + "a submitted job is polled, never resubmitted")
        }

        // 2. SUBMIT. The helper sbatches, prints the job id and status file,
        //    and exits; nothing is held open across the bootstrap itself.
        if submission == nil {
            let result = await shell.run(submitArgv)
            transcript += ["$ " + ClusterProvisioner.displayCommand(submitArgv)]
                + result.lines
            guard result.succeeded,
                let parsed = ClusterBootstrapSubmission.parse(lines: result.lines)
            else {
                return ClusterBootstrapApplyOutcome(
                    jobID: nil,
                    report: BootstrapReport.parse(fromTranscript: result.lines),
                    outcome: ClusterOperationOutcome(
                        succeeded: false,
                        message: result.succeeded
                            ? "the bootstrap helper reported no job id — nothing was "
                                + "queued to follow"
                            : "bootstrap submission exited \(result.exitCode)",
                        transcript: transcript, exitCode: result.exitCode))
            }
            submission = parsed
            if parsed.adopted {
                transcript.append(
                    "bootstrap: the cluster already had bootstrap job \(parsed.jobID) "
                        + "in flight for this workspace — adopted, not resubmitted")
            }
            // 3. PERSIST, before the first poll. Everything after this point
            //    can be interrupted without losing the job.
            if let journal, let siteID {
                do {
                    try journal.recordPendingBootstrapJob(
                        ClusterPendingBootstrapJob(
                            siteID: siteID, jobID: parsed.jobID,
                            statusFile: parsed.statusFile, planHash: planHash))
                } catch {
                    transcript.append(
                        "bootstrap: WARNING job \(parsed.jobID) is submitted but its "
                            + "pending record could not be saved "
                            + "(\(error.localizedDescription)) — the cluster-side "
                            + "breadcrumb still identifies it")
                }
            }
        }
        guard let submission else {
            return ClusterBootstrapApplyOutcome(
                jobID: nil, report: nil,
                outcome: ClusterOperationOutcome(
                    succeeded: false, message: "no bootstrap job to follow",
                    transcript: transcript))
        }

        // 4. POLL, as separate short-lived status commands.
        guard let statusArgv = ClusterProvisioner.bootstrapJobStatusArgv(
            site: site, remoteRepoPath: configuration.remoteRepoPath,
            squeueCommand: configuration.squeueCommand,
            jobID: submission.jobID, statusFile: submission.statusFile)
        else {
            return pendingApplyOutcome(
                submission: submission, verdict: .unknown(
                    reason: "this site has no workspace to read the job's status from"),
                transcript: transcript)
        }
        var verdict = ClusterBootstrapJobVerdict.pending(state: nil)
        var terminalLines: [String] = []
        var lastSummary = ""
        for attempt in 0..<max(1, bootstrapPollLimit) {
            if attempt > 0, bootstrapPollDelay > .zero {
                try? await Task.sleep(for: bootstrapPollDelay)
            }
            let probe = await shell.run(statusArgv)
            guard let parsed = ClusterBootstrapJobVerdict.parse(lines: probe.lines)
            else {
                // A failed probe is a failed PROBE. The job keeps running and
                // the record keeps pointing at it.
                verdict = .unknown(
                    reason: probe.succeeded
                        ? "the status probe printed no verdict"
                        : "the status probe exited \(probe.exitCode)")
                if verdict.summary != lastSummary {
                    transcript.append("bootstrap: job \(submission.jobID) \(verdict.summary)")
                    lastSummary = verdict.summary
                }
                continue
            }
            verdict = parsed
            if parsed.summary != lastSummary {
                transcript.append("bootstrap: job \(submission.jobID) \(parsed.summary)")
                lastSummary = parsed.summary
            }
            if parsed.isTerminal {
                // The terminal probe forwards the job log, which carries
                // bootstrap's own JSON report.
                terminalLines = probe.lines
                break
            }
        }

        guard verdict.isTerminal else {
            return pendingApplyOutcome(
                submission: submission, verdict: verdict, transcript: transcript)
        }
        transcript += terminalLines
        let report = BootstrapReport.parse(fromTranscript: terminalLines)
        var succeeded = false
        var message: String
        switch verdict {
        case .completed(let code) where code == 0:
            succeeded = report?.ok ?? true
            if succeeded {
                message = "bootstrap ok — job \(submission.jobID)"
                if let prefix = report?.prefix { message += " — env \(prefix)" }
                if report == nil {
                    message += " (the job's own report could not be read back)"
                }
            } else {
                let failed = report?.failedStepNames.joined(separator: ", ") ?? ""
                message = failed.isEmpty
                    ? "bootstrap job \(submission.jobID) reported failure"
                    : "bootstrap step(s) failed: \(failed)"
            }
        case .completed(let code):
            let failed = report?.failedStepNames.joined(separator: ", ") ?? ""
            message = failed.isEmpty
                ? "bootstrap job \(submission.jobID) exited \(code)"
                : "bootstrap step(s) failed: \(failed)"
        default:
            message = "bootstrap job \(submission.jobID) \(verdict.summary) — inspect "
                + "the job log on the cluster"
        }
        if let journal, let siteID {
            journal.finishPendingBootstrapJob(
                forSite: siteID, jobID: submission.jobID, succeeded: succeeded,
                message: message)
        }
        return ClusterBootstrapApplyOutcome(
            jobID: submission.jobID, report: report,
            outcome: ClusterOperationOutcome(
                succeeded: succeeded, message: message, transcript: transcript),
            statusFile: submission.statusFile, adopted: submission.adopted,
            stillPending: false, verdict: verdict)
    }

    /// The job is submitted and durably recorded, but this invocation did not
    /// see it finish. Deliberately `succeeded: false` — nothing downstream may
    /// treat the environment as bootstrapped — with a message that says this
    /// is a WAIT, not a failure, and names the resumption.
    private func pendingApplyOutcome(
        submission: ClusterBootstrapSubmission,
        verdict: ClusterBootstrapJobVerdict,
        transcript: [String]
    ) -> ClusterBootstrapApplyOutcome {
        ClusterBootstrapApplyOutcome(
            jobID: submission.jobID, report: nil,
            outcome: ClusterOperationOutcome(
                succeeded: false,
                message: "bootstrap job \(submission.jobID) \(verdict.summary) — it is "
                    + "recorded, so running bootstrap apply again resumes following "
                    + "this job instead of submitting another",
                transcript: transcript),
            statusFile: submission.statusFile, adopted: submission.adopted,
            stillPending: true, verdict: verdict)
    }

    /// Whether the bootstrapped environment file exists and sources cleanly.
    public func observeBootstrap(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        envFile: String
    ) async -> ClusterBootstrapObservation {
        guard site.isSSHTransport else { return .notApplicable }
        let payload =
            ". \(ClusterProvisioner.shellQuoted(envFile)) >/dev/null 2>&1 && "
            + "printf '%s' \"${STEERLAB_PREFIX:-}\""
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site, remoteWords: ["bash", "-lc", payload])
        guard !argv.isEmpty else {
            return .unknown(reason: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        guard result.succeeded else {
            return result.exitCode == 255
                ? .unknown(reason: "the SSH connection failed, so \(envFile) was not read")
                : .absent
        }
        let prefix = result.text.isEmpty ? nil : result.text
        return .valid(envFile: envFile, prefix: prefix)
    }

    // MARK: Validate

    public func validate(
        site: ClusterSiteProfile, envFile: String, prefix: String?
    ) async -> ClusterValidateOutcome {
        guard site.isSSHTransport else {
            return ClusterValidateOutcome(
                observation: .notApplicable, lines: [],
                outcome: .skipped(
                    "direct transport — run `steerlab-server profile validate` on the box"))
        }
        let argv = ClusterProvisioner.validateArgv(
            site: site, envFile: envFile, prefix: prefix)
        let result = await shell.run(argv)
        let lines = result.lines.map(ProfileValidateLine.classify)
        let transcript = ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        let failures = lines.filter { $0.kind == .fail }.map(\.text)
        let warnings = lines.filter { $0.kind == .warn }.map(\.text)
        if result.succeeded {
            return ClusterValidateOutcome(
                observation: warnings.isEmpty ? .pass : .warn(messages: warnings),
                lines: lines,
                outcome: ClusterOperationOutcome(
                    succeeded: true,
                    message: warnings.isEmpty
                        ? "profile validate passed"
                        : "profile validate passed with \(warnings.count) warning(s)",
                    transcript: transcript))
        }
        return ClusterValidateOutcome(
            observation: .fail(
                messages: failures.isEmpty
                    ? ["profile validate exited \(result.exitCode)"] : failures),
            lines: lines,
            outcome: ClusterOperationOutcome(
                succeeded: false,
                message: failures.isEmpty
                    ? "profile validate exited \(result.exitCode)"
                    : failures.joined(separator: " · "),
                transcript: transcript, exitCode: result.exitCode))
    }

    // MARK: Controller (§7.10 — submit and RETURN; never hold a foreground wait)

    public func controllerStart(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        envFile: String, prefix: String?
    ) async -> ClusterControllerStartOutcome {
        guard site.topology == .daemonInJob else {
            return ClusterControllerStartOutcome(
                jobID: nil,
                outcome: .skipped(
                    "topology is \(site.topology.rawValue) — no controller job needed"))
        }
        guard site.isSSHTransport else {
            return ClusterControllerStartOutcome(
                jobID: nil, outcome: .skipped("direct transport — no scheduler access"))
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site,
            rawCommand: ClusterProvisioner.controllerRemoteCommand(
                site: site, remoteRepoPath: configuration.remoteRepoPath,
                envPrefix: prefix, envFile: envFile))
        let result = await shell.run(argv)
        let transcript = ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        guard result.succeeded,
            let jobID = ClusterProvisioner.parseSbatchJobID(from: result.lines)
        else {
            return ClusterControllerStartOutcome(
                jobID: nil,
                outcome: ClusterOperationOutcome(
                    succeeded: false,
                    message: "sbatch did not report a job id (exit \(result.exitCode)) — "
                        + "check partition/walltime/account for this site",
                    transcript: transcript, exitCode: result.exitCode))
        }
        return ClusterControllerStartOutcome(
            jobID: jobID,
            outcome: ClusterOperationOutcome(
                succeeded: true,
                message: "controller job \(jobID) submitted — its queue state is "
                    + "observed later, not waited on here",
                transcript: transcript))
    }

    // MARK: Rendered controller script (open-issues §1 field report, 2026-08-20)

    /// Read-only: is `<metadataRoot>/controller-job.sbatch` the child of the
    /// template currently deployed here?
    public func observeControllerScript(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) async -> ClusterControllerScriptObservation {
        guard site.topology == .daemonInJob, site.isSSHTransport else {
            return .notApplicable
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site,
            rawCommand: ClusterProvisioner.controllerScriptProbeCommand(
                site: site, remoteRepoPath: configuration.remoteRepoPath))
        guard !argv.isEmpty else {
            return .unknown(reason: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        return ClusterProvisioner.parseControllerScriptProbe(
            exitCode: result.exitCode, lines: result.lines)
    }

    /// Re-render the controller script WITHOUT submitting anything.
    ///
    /// The same command `controllerStart` runs, stopped before `sbatch`. That
    /// identity is the point: two renderers would be exactly the drift this
    /// change exists to close, and the mode a running site needs is "fix the
    /// artifact, do not add a second controller to the queue".
    public func controllerRender(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        envFile: String, prefix: String?
    ) async -> ClusterOperationOutcome {
        guard site.topology == .daemonInJob else {
            return .skipped(
                "topology is \(site.topology.rawValue) — no controller script to render")
        }
        guard site.isSSHTransport else {
            return .skipped("direct transport — no remote metadata root to render into")
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site,
            rawCommand: ClusterProvisioner.controllerRemoteCommand(
                site: site, remoteRepoPath: configuration.remoteRepoPath,
                envPrefix: prefix, envFile: envFile, submit: false))
        let result = await shell.run(argv)
        let transcript = ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        guard result.succeeded else {
            return ClusterOperationOutcome(
                succeeded: false,
                message: "the controller script could not be re-rendered "
                    + "(exit \(result.exitCode)) — is the template deployed and "
                    + "the ControlMaster alive?",
                transcript: transcript, exitCode: result.exitCode)
        }
        // The render-only command echoes the stamp it just wrote; report it
        // verbatim so the operator sees the identity, not a claim about it.
        let stamp = result.lines
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .last { !$0.isEmpty } ?? ""
        return ClusterOperationOutcome(
            succeeded: true,
            message: "controller script re-rendered at "
                + "\(site.metadataRoot)/\(ClusterProvisioner.controllerScriptFileName)"
                + (stamp.isEmpty ? "" : " — \(stamp)")
                + ". Nothing was submitted; the next controller generation "
                + "reads this file.",
            transcript: transcript)
    }

    /// One scheduler state probe, classified by the existing (tested) rule.
    public func controllerProbe(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        jobID: String
    ) async -> ControllerJobProbe {
        let argv = ClusterProvisioner.controllerStateArgv(
            site: site, squeueCommand: configuration.squeueCommand, jobID: jobID)
        let result = await shell.run(argv)
        return ClusterProvisioner.parseControllerJobProbe(
            exitCode: result.exitCode, lines: result.lines)
    }

    /// Reconcile the durable job id with the scheduler. A FAILED QUERY yields
    /// `.unknown`, never `.absent`: the repository's doctrine is that an
    /// unproven death never licenses a resubmit.
    public func observeController(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration,
        recordedJobID: String?, daemonHostPresent: Bool
    ) async -> ClusterControllerObservation {
        guard site.topology == .daemonInJob, site.isSSHTransport else {
            return .notApplicable
        }
        guard let recordedJobID, !recordedJobID.isEmpty else {
            // No job of OURS. A host file we cannot attribute is not proof of
            // life — but it is also not proof of death, so refuse to claim
            // `absent` (which would license starting a second controller).
            return daemonHostPresent
                ? .unknown(
                    reason: "a serverd.host record exists but no controller job is "
                        + "recorded for this site — it cannot be attributed")
                : .absent
        }
        switch await controllerProbe(
            site: site, configuration: configuration, jobID: recordedJobID)
        {
        case .state(let state, let reason) where state == "PENDING" || state == "CONFIGURING":
            return .pending(jobID: recordedJobID, reason: reason)
        case .state(let state, _) where state == "RUNNING" || state == "COMPLETING":
            return .running(jobID: recordedJobID)
        case .state(let state, _):
            return .failed(jobID: recordedJobID, state: state)
        case .absent:
            return .absent
        case .unavailable:
            return .unknown(
                reason: "the scheduler probe failed, so the job's state is unknown — "
                    + "an unproven death never licenses a resubmit")
        }
    }

    // MARK: Controller adoption (Phase C)

    /// Stop a controller job. Narrowly typed, like every other remote step —
    /// there is deliberately no general `cluster exec` (§4.4).
    public func controllerStop(
        site: ClusterSiteProfile, jobID: String
    ) async -> ClusterOperationOutcome {
        guard site.topology == .daemonInJob, site.isSSHTransport else {
            return .skipped(
                "topology is \(site.topology.rawValue) — there is no controller job to stop")
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site, remoteWords: ["scancel", jobID])
        guard !argv.isEmpty else {
            return ClusterOperationOutcome(
                succeeded: false, message: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        let transcript = ["$ " + ClusterProvisioner.displayCommand(argv)] + result.lines
        guard result.succeeded else {
            return ClusterOperationOutcome(
                succeeded: false,
                message: "scancel exited \(result.exitCode) — the job may already be gone",
                transcript: transcript, exitCode: result.exitCode)
        }
        return ClusterOperationOutcome(
            succeeded: true,
            message: "cancelled controller job \(jobID)", transcript: transcript)
    }

    /// Whether a hand-started job may be recorded as this site's controller.
    /// Pure, so every branch is covered by fixtures rather than by ssh.
    ///
    /// The doctrine mirrors the rest of the controller layer: the SCHEDULER is
    /// the authority on whether the job exists, and a host record only counts
    /// once the scheduler says the job that owns it is running. A job the
    /// scheduler will not vouch for is never adopted — recording it would make
    /// every later inspection believe in a controller that may not exist, and
    /// the planner would then never start the real one.
    ///
    /// Returns nil when the job is adoptable, or the refusal reason.
    public static func controllerAdoptionRefusal(
        controller: ClusterControllerObservation,
        daemonHost: ClusterDaemonHostObservation,
        serverHTTP: ClusterServerHTTPObservation?
    ) -> String? {
        switch controller {
        case .notApplicable:
            return "this site's topology has no controller job to adopt"
        case .absent:
            return "the scheduler does not know this job — it has left the queue"
        case .failed(_, let state):
            return "the scheduler reports the job as \(state)"
        case .unknown(let reason):
            return "the job's state could not be read (\(reason)), so it cannot be verified"
        case .pending:
            // A queued job IS this site's controller-to-be. Recording it is
            // exactly what stops a later `ensure` submitting a second one.
            return nil
        case .running:
            break
        }
        // A RUNNING controller must have published where it runs; otherwise
        // the forward has nowhere to point and "adopted" would mean nothing.
        switch daemonHost {
        case .current:
            break
        case .absent:
            return "the job is running but has not published a serverd.host record yet"
        case .stale(_, let reason):
            return "the serverd.host record cannot be trusted (\(reason))"
        case .notApplicable:
            return "this site's topology publishes no serverd.host record"
        }
        // If something IS answering on the endpoint, it must be the right
        // something — an identity mismatch means the port belongs to another
        // server, and adopting would attach this site to it.
        if case .identityMismatch(let reason) = serverHTTP {
            return "the endpoint answered with a different identity (\(reason))"
        }
        return nil
    }

    /// Verify a hand-started controller job against the scheduler, the
    /// `serverd.host` record, and (when a forward is already up) the endpoint's
    /// identity. Does NOT record anything — persistence is the caller's, so
    /// this stays a pure observation with a verdict.
    public func adoptController(
        site: ClusterSiteProfile,
        configuration: ClusterProvisioningConfiguration,
        jobID: String,
        endpointProbe: (any ClusterEndpointProbe)? = nil,
        baseURL: URL? = nil,
        token: String? = nil
    ) async -> ClusterControllerAdoptionOutcome {
        let trimmed = jobID.trimmingCharacters(in: .whitespacesAndNewlines)
        let host = await readDaemonHost(site: site)
        let controller = await observeController(
            site: site, configuration: configuration, recordedJobID: trimmed,
            daemonHostPresent: host != nil)
        let daemonHost = site.topology == .daemonInJob
            ? Self.classifyDaemonHost(host: host, controller: controller)
            : .notApplicable
        var serverHTTP: ClusterServerHTTPObservation?
        if let endpointProbe, let baseURL {
            let probe = await endpointProbe.probe(baseURL: baseURL, token: token)
            if probe.reachable {
                serverHTTP = .reachable(
                    build: probe.serverBuild, role: probe.serverRole, root: probe.root)
            } else if probe.authFailed {
                serverHTTP = .authFailed
            } else {
                serverHTTP = .unreachable(reason: probe.detail ?? "no answer")
            }
        }
        let refusal = Self.controllerAdoptionRefusal(
            controller: controller, daemonHost: daemonHost, serverHTTP: serverHTTP)
        return ClusterControllerAdoptionOutcome(
            jobID: trimmed,
            verified: refusal == nil,
            refusalReason: refusal,
            controller: controller,
            daemonHost: daemonHost,
            serverHTTP: serverHTTP,
            outcome: ClusterOperationOutcome(
                succeeded: refusal == nil,
                message: refusal ?? "controller job \(trimmed) verified — "
                    + "\(controller.summary), host \(daemonHost.summary)"))
    }

    /// The daemon-host read command — the TUNNEL's own remote-read
    /// convention (`ssh … <host> cat <quoted path>`), so the setup path and
    /// the tunnel ask the far side the same question the same way.
    public static func daemonHostReadArgv(site: ClusterSiteProfile) -> [String]? {
        guard case .ssh(let host, let proxyJump, _, _) = site.transport else { return nil }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return [ClusterProvisioner.sshExecutablePath]
            + ClusterTunnel.remoteReadArguments(
                host: trimmed, proxyJump: proxyJump, path: site.daemonHostFilePath)
    }

    /// Read `<metadataRoot>/serverd.host`. Nil when there is nothing to read
    /// (or the read failed) — the caller decides what that means.
    public func readDaemonHost(site: ClusterSiteProfile) async -> String? {
        guard site.topology == .daemonInJob, site.isSSHTransport else { return nil }
        guard let argv = Self.daemonHostReadArgv(site: site) else { return nil }
        let result = await shell.run(argv)
        guard result.succeeded, !result.text.isEmpty else { return nil }
        return result.text
    }

    /// Read `<metadataRoot>/serverd.host` and decide whether it can be
    /// trusted, given what the scheduler says about the controller job.
    public func observeDaemonHost(
        site: ClusterSiteProfile, controller: ClusterControllerObservation
    ) async -> ClusterDaemonHostObservation {
        guard site.topology == .daemonInJob, site.isSSHTransport else {
            return .notApplicable
        }
        return Self.classifyDaemonHost(
            host: await readDaemonHost(site: site), controller: controller)
    }

    /// Pure half: a host record is only CURRENT when the scheduler says the
    /// job that owns it is running. Every other answer is stale-with-a-reason
    /// — a stale host file must never pass for proof of life (§7.10).
    public static func classifyDaemonHost(
        host: String?, controller: ClusterControllerObservation
    ) -> ClusterDaemonHostObservation {
        guard let host, !host.isEmpty else { return .absent }
        switch controller {
        case .running:
            return .current(host: host)
        case .pending:
            return .stale(
                host: host,
                reason: "written by an earlier controller job — the current job is "
                    + "still queued")
        case .absent, .failed:
            return .stale(
                host: host,
                reason: "the controller job that wrote it is no longer running")
        case .unknown, .notApplicable:
            return .stale(
                host: host,
                reason: "the controller job's state could not be read, so this "
                    + "record cannot be confirmed current")
        }
    }

    // MARK: Token import (§7.11 — straight into the secret store)

    /// Read the remote token over the authenticated ControlMaster and put it
    /// in the secret store. The value never enters a return value, a record,
    /// a transcript, or a log line.
    public func importRemoteToken(
        site: ClusterSiteProfile, tokenKey: String, overwriteExisting: Bool = false
    ) async -> ClusterTokenImportOutcome {
        guard site.isSSHTransport else {
            return .unavailable(reason: "direct transport — no remote token to import")
        }
        // Presence, not the value: "do we already have one?" must not read
        // (and so must not prompt for) a secret this branch immediately drops.
        if !overwriteExisting, secrets.hasToken(forKey: tokenKey) {
            return .alreadyPresent
        }
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site, remoteWords: ["cat", Self.remoteTokenPath])
        guard !argv.isEmpty else {
            return .unavailable(reason: "site has no SSH host configured")
        }
        let result = await shell.run(argv)
        guard result.succeeded, !result.text.isEmpty else {
            return .unavailable(
                reason: "could not read \(Self.remoteTokenPath) — complete "
                    + "authentication, then retry")
        }
        do {
            try secrets.store(result.text, forKey: tokenKey)
        } catch {
            // Deliberately does not interpolate the value into the reason.
            return .unavailable(
                reason: "the token could not be saved to the Keychain "
                    + "(\(error.localizedDescription))")
        }
        return .imported
    }
}
