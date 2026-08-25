import Foundation

// =============================================================================
// Inspection: site → `ClusterObservedState` (CLUSTER-CLI-LIFECYCLE-PLAN §7.2).
//
// Inspection is ALWAYS allowed and never mutates anything remote. Every
// external probe here is behind a seam — the shell, the HTTP endpoint, the
// forward, the secret store, and the clock — so the entire observation model
// is reachable from a test with no ssh, no sockets, and no Keychain.
// =============================================================================

// MARK: - Tunnel control seam

public struct ClusterTunnelOpenOutcome: Sendable, Equatable {
    public var observation: ClusterTunnelObservation
    public var message: String
    /// The forward identity that was installed or adopted (§7.7).
    public var forwardIdentity: String?
    /// False when an already-correct forward was ADOPTED rather than opened —
    /// opening an already-correct forward must be a no-op.
    public var changed: Bool

    public init(
        observation: ClusterTunnelObservation, message: String,
        forwardIdentity: String? = nil, changed: Bool = false
    ) {
        self.observation = observation
        self.message = message
        self.forwardIdentity = forwardIdentity
        self.changed = changed
    }
}

/// Owning the local forward as an app/CLI PEER contract: either process may
/// create one, and the other adopts it rather than adding a second.
public protocol ClusterTunnelControlling: Sendable {
    /// What the persisted forward for this site currently is. `targetHost` is
    /// the CURRENT daemon host, when one is known: a persisted forward whose
    /// recorded target disagrees with it is stale, not up — a listening local
    /// port says nothing about which node it forwards to (§7.7).
    func observe(
        site: ClusterSiteProfile, persistedPort: Int?, targetHost: String?
    ) async -> ClusterTunnelObservation
    /// Install or adopt the forward. Adoption must verify both the forward's
    /// recorded target and the HTTP server identity through it (§7.7) — a
    /// listener that fails the identity probe is never "the existing forward".
    func open(
        site: ClusterSiteProfile, targetHost: String, persistedPort: Int?
    ) async -> ClusterTunnelOpenOutcome
    /// Cancel SteerLab's exact forward — never anything else's. Returns a
    /// human-readable error when the forward was NOT verifiably removed
    /// (a ControlMaster cancel whose spec mismatches fails with "port not
    /// forwarded" and must never read as success), or nil once the listener
    /// is actually gone.
    @discardableResult
    func close(site: ClusterSiteProfile, localPort: Int, targetHost: String) async -> String?
}

// MARK: - Persisted forward records (§7.7)

/// Durable record of the forward SteerLab itself installed for a site — the
/// half of §7.7 a listening port cannot carry: WHICH node the forward targets,
/// and the exact `-L` spec (bind-address form included) a later cancel must
/// reuse. The ControlMaster matches cancels by spec string, so a cancel
/// reconstructed with a different target or bind form fails with "port not
/// forwarded" while the real forward lives on.
protocol ClusterForwardRecordStoring: Sendable {
    func record(forSite identity: String) -> ClusterTunnel.ForwardRecord?
    func save(_ record: ClusterTunnel.ForwardRecord, forSite identity: String)
    func removeRecord(forSite identity: String)
}

/// File-backed store in the shared SteerLab support directory, so a forward
/// installed by one CLI invocation is inspectable — and cancellable by exact
/// spec — from the next one, or from the app.
struct FileClusterForwardRecordStore: ClusterForwardRecordStoring {

    init() {}

    /// Computed per call so the test `rootOverride` applies.
    private var fileURL: URL {
        ClusterSupportPaths.root.appending(component: "cluster-forwards.json")
    }

    private func load() -> [String: ClusterTunnel.ForwardRecord] {
        guard let data = try? Data(contentsOf: fileURL) else { return [:] }
        return (try? ClusterSupportPaths.decoder()
            .decode([String: ClusterTunnel.ForwardRecord].self, from: data)) ?? [:]
    }

    private func store(_ records: [String: ClusterTunnel.ForwardRecord]) {
        guard let data = try? ClusterSupportPaths.encoder().encode(records) else { return }
        try? ClusterSupportPaths.writeAtomically(data, to: fileURL)
    }

    func record(forSite identity: String) -> ClusterTunnel.ForwardRecord? {
        load()[identity]
    }

    func save(_ record: ClusterTunnel.ForwardRecord, forSite identity: String) {
        var records = load()
        records[identity] = record
        store(records)
    }

    func removeRecord(forSite identity: String) {
        var records = load()
        records.removeValue(forKey: identity)
        store(records)
    }
}

/// Live implementation over the SAME `ssh -O forward/cancel` commands and the
/// SAME forward-identity rule `ClusterTunnel` uses, so a forward opened by the
/// app is adoptable by the CLI and vice versa.
public struct SSHClusterTunnelController: ClusterTunnelControlling {

    private let runner: any TunnelProcessRunner
    private let endpoint: any ClusterEndpointProbe
    private let forwards: any ClusterForwardRecordStoring

    public init(runner: any TunnelProcessRunner = SystemTunnelProcessRunner()) {
        self.init(
            runner: runner, endpoint: HTTPClusterEndpointProbe(),
            forwards: FileClusterForwardRecordStore())
    }

    init(
        runner: any TunnelProcessRunner,
        endpoint: any ClusterEndpointProbe,
        forwards: any ClusterForwardRecordStoring
    ) {
        self.runner = runner
        self.endpoint = endpoint
        self.forwards = forwards
    }

    /// Forward identity = site remote identity + target host + remote port +
    /// local port. Two forwards agreeing on this string ARE the same forward.
    public static func forwardIdentity(
        site: ClusterSiteProfile, targetHost: String, localPort: Int
    ) -> String? {
        guard case .ssh(let host, _, let remotePort, _) = site.transport else { return nil }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return "ssh://\(trimmed)/\(targetHost):\(remotePort)->127.0.0.1:\(localPort)"
    }

    private func record(
        site: ClusterSiteProfile, targetHost: String, localPort: Int
    ) -> ClusterTunnel.ForwardRecord? {
        guard case .ssh(let host, let proxyJump, let remotePort, _) = site.transport
        else { return nil }
        return ClusterTunnel.ForwardRecord(
            host: host.trimmingCharacters(in: .whitespacesAndNewlines),
            proxyJump: proxyJump, localPort: localPort, targetHost: targetHost,
            remotePort: remotePort)
    }

    public func observe(
        site: ClusterSiteProfile, persistedPort: Int?, targetHost: String?
    ) async -> ClusterTunnelObservation {
        guard site.isSSHTransport else { return .notApplicable }
        guard let persistedPort else { return .absent }
        // A port that is FREE to bind has nothing listening: the forward we
        // remembered is gone.
        let free = await runner.isLocalPortFree(persistedPort)
        if free {
            return .stale(localPort: persistedPort, reason: "nothing is listening on the port")
        }
        // A busy port is not yet a correct forward: after a controller
        // restart the daemon moves nodes, and the old forward keeps LISTENING
        // locally while targeting a machine the server no longer runs on.
        // The persisted record knows the target; the caller passed where the
        // controller runs NOW.
        if let identity = site.registryIdentity,
            let record = forwards.record(forSite: identity),
            record.localPort == persistedPort,
            let targetHost, !targetHost.isEmpty,
            record.targetHost != targetHost
        {
            return .stale(
                localPort: persistedPort,
                reason: "forward targets \(record.targetHost) but the controller "
                    + "now runs on \(targetHost)")
        }
        // Our forward as far as this layer can tell — the HTTP identity check
        // settles whether the listener is really ours.
        return .up(localPort: persistedPort)
    }

    /// Cancel a persisted forward with the EXACT spec it was created with —
    /// the ControlMaster matches cancels by string, so any other bind-address
    /// form or target fails with "port not forwarded" while the forward lives
    /// on. Returns a human-readable error, or nil on success.
    private func cancelExactSpec(_ record: ClusterTunnel.ForwardRecord) async -> String? {
        let result = await runner.run(
            ClusterTunnel.sshExecutablePath,
            arguments: ClusterTunnel.cancelArguments(record))
        guard result.exitCode == 0 else {
            let detail = result.standardError
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return detail.isEmpty ? "ssh -O cancel exited \(result.exitCode)" : detail
        }
        return nil
    }

    /// Whether a steerlab server answers through `127.0.0.1:port`. An HTTP
    /// 401 is the server ANSWERING — identity is proven; the missing token is
    /// the registration step's problem, not the tunnel's.
    private func identityAnswers(port: Int) async -> (ok: Bool, detail: String?) {
        let probe = await endpoint.probe(
            baseURL: ClusterSiteProfile.localhostBaseURL(port: port), token: nil)
        return (probe.reachable || probe.authFailed, probe.detail)
    }

    public func open(
        site: ClusterSiteProfile, targetHost: String, persistedPort: Int?
    ) async -> ClusterTunnelOpenOutcome {
        guard site.isSSHTransport else {
            return ClusterTunnelOpenOutcome(
                observation: .notApplicable, message: "direct transport — no tunnel needed")
        }
        let identity = site.registryIdentity
        if let persistedPort {
            switch await observe(
                site: site, persistedPort: persistedPort, targetHost: targetHost)
            {
            case .up:
                // §7.7: "Before adoption, verify both the SSH forward and the
                // HTTP server identity." A listening port whose listener is
                // not our server is a conflict, not an existing forward — the
                // live failure mode is a half-cancelled mux forward that
                // keeps the listener while the claim "tunnel open" is false.
                let answer = await identityAnswers(port: persistedPort)
                if answer.ok {
                    return ClusterTunnelOpenOutcome(
                        observation: .up(localPort: persistedPort),
                        message: "adopted the existing forward on "
                            + "127.0.0.1:\(persistedPort) (server identity verified)",
                        forwardIdentity: Self.forwardIdentity(
                            site: site, targetHost: targetHost, localPort: persistedPort),
                        changed: false)
                }
                guard let identity,
                    let record = forwards.record(forSite: identity),
                    record.localPort == persistedPort
                else {
                    let reason = "the listener on 127.0.0.1:\(persistedPort) failed "
                        + "the server identity probe "
                        + "(\(answer.detail ?? "no answer")) and no SteerLab forward "
                        + "record names it — close whatever holds the port"
                    return ClusterTunnelOpenOutcome(
                        observation: .conflicted(
                            localPort: persistedPort, reason: reason),
                        message: reason)
                }
                // Repair: cancel OUR forward by its exact recorded spec, then
                // reinstall. A failed cancel with the listener still there is
                // an error to surface, never a success to build on.
                if let cancelError = await cancelExactSpec(record) {
                    let reason = "could not cancel the stale forward on "
                        + "127.0.0.1:\(persistedPort): \(cancelError)"
                    return ClusterTunnelOpenOutcome(
                        observation: .conflicted(
                            localPort: persistedPort, reason: reason),
                        message: reason)
                }
                forwards.removeRecord(forSite: identity)
            case .stale(let port, _):
                // Either nothing listens (the forward died with its ssh
                // master) or the live forward targets a node the controller
                // left. Cancel any recorded forward by exact spec before
                // reinstalling; a failure only counts when the listener is
                // actually still there to fight over.
                if let identity,
                    let record = forwards.record(forSite: identity),
                    record.localPort == port
                {
                    let listening = !(await runner.isLocalPortFree(port))
                    if let cancelError = await cancelExactSpec(record), listening {
                        let reason = "could not cancel the stale forward on "
                            + "127.0.0.1:\(port): \(cancelError)"
                        return ClusterTunnelOpenOutcome(
                            observation: .conflicted(localPort: port, reason: reason),
                            message: reason)
                    }
                    forwards.removeRecord(forSite: identity)
                }
            case .absent, .conflicted, .notApplicable:
                break
            }
        }
        let preferred = persistedPort ?? site.preferredLocalPort ?? 8700
        guard let localPort = await ClusterTunnel.firstFreeLocalPort(
            from: preferred, runner: runner)
        else {
            return ClusterTunnelOpenOutcome(
                observation: .conflicted(
                    localPort: preferred, reason: "no free local port near \(preferred)"),
                message: "no free local port near \(preferred)")
        }
        guard let forward = record(
            site: site, targetHost: targetHost, localPort: localPort)
        else {
            return ClusterTunnelOpenOutcome(
                observation: .absent, message: "site has no SSH host configured")
        }
        let result = await runner.run(
            ClusterTunnel.sshExecutablePath,
            arguments: ClusterTunnel.forwardArguments(forward))
        guard result.exitCode == 0 else {
            let detail = result.standardError
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ClusterTunnelOpenOutcome(
                observation: .conflicted(
                    localPort: localPort,
                    reason: detail.isEmpty
                        ? "ssh -O forward exited \(result.exitCode)" : detail),
                message: detail.isEmpty
                    ? "could not install the SSH forward (exit \(result.exitCode))"
                    : "could not install the SSH forward: \(detail)")
        }
        // The record is what makes the forward repairable later: the exact
        // spec for a future cancel, and the target the next observation
        // compares against wherever the controller then runs.
        if let identity = site.registryIdentity {
            forwards.save(forward, forSite: identity)
        }
        return ClusterTunnelOpenOutcome(
            observation: .up(localPort: localPort),
            message: "forward installed on 127.0.0.1:\(localPort)",
            forwardIdentity: Self.forwardIdentity(
                site: site, targetHost: targetHost, localPort: localPort),
            changed: true)
    }

    @discardableResult
    public func close(
        site: ClusterSiteProfile, localPort: Int, targetHost: String
    ) async -> String? {
        // Prefer the persisted record: it holds the exact spec (bind-address
        // form and target) the forward was created with, which is the ONLY
        // spec the ControlMaster will cancel. Reconstructing one from the
        // current daemon host after the controller moved nodes produces a
        // cancel that fails "port not forwarded" while the forward survives.
        let forward: ClusterTunnel.ForwardRecord
        if let identity = site.registryIdentity,
            let persisted = forwards.record(forSite: identity),
            persisted.localPort == localPort
        {
            forward = persisted
        } else if let constructed = record(
            site: site, targetHost: targetHost, localPort: localPort)
        {
            forward = constructed
        } else {
            return "site has no SSH host configured"
        }
        let result = await runner.run(
            ClusterTunnel.sshExecutablePath,
            arguments: ClusterTunnel.cancelArguments(forward))
        // Success is the LISTENER being gone, not the cancel's exit code: a
        // cancel can fail because the forward is already gone (fine), and a
        // cancel can "succeed" while a second same-port forward request keeps
        // the mux listener alive (the live 2026-08-12 failure).
        if !(await runner.isLocalPortFree(localPort)) {
            let detail = result.standardError
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return "the forward on 127.0.0.1:\(localPort) was not removed"
                + (result.exitCode == 0
                    ? " — the cancel reported success but something still listens there"
                    : ": \(detail.isEmpty ? "ssh -O cancel exited \(result.exitCode)" : detail)")
        }
        if let identity = site.registryIdentity {
            forwards.removeRecord(forSite: identity)
        }
        return nil
    }
}

// MARK: - Inspector

/// Produces one `ClusterObservedState` for a site. Read-only by construction:
/// every call it makes is a probe.
public struct ClusterLifecycleInspector: Sendable {

    private let operations: ClusterProvisioningOperations
    private let tunnel: any ClusterTunnelControlling
    private let endpoint: any ClusterEndpointProbe
    private let secrets: any ClusterSecretStore
    private let now: @Sendable () -> Date

    public init(
        operations: ClusterProvisioningOperations,
        tunnel: any ClusterTunnelControlling,
        endpoint: any ClusterEndpointProbe,
        secrets: any ClusterSecretStore = KeychainClusterSecretStore(),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.operations = operations
        self.tunnel = tunnel
        self.endpoint = endpoint
        self.secrets = secrets
        self.now = now
    }

    /// Blocking configuration defects — the site cannot be acted on at all.
    /// Deliberately narrow: the profile schema is permissive on purpose, and
    /// only a genuinely unusable site belongs here.
    public static func configurationProblems(
        site: ClusterSiteProfile, configuration: ClusterProvisioningConfiguration
    ) -> [String] {
        var problems: [String] = []
        switch site.transport {
        case .ssh(let host, _, let remotePort, _):
            if host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                problems.append(
                    "the site has no SSH host — set one before provisioning it")
            }
            if !(1...65_535).contains(remotePort) {
                problems.append("the site's remote port \(remotePort) is not a valid port")
            }
        case .direct:
            break
        }
        if site.isSSHTransport,
            configuration.bootstrapExecutionTarget == .slurmBatch
        {
            if case .slurm = site.scheduler {} else {
                problems.append(
                    "bootstrap is configured to run as a Slurm job, but the site "
                        + "declares no Slurm scheduler")
            }
        }
        return problems
    }

    /// One inspection pass.
    ///
    /// `recordedControllerJobID` and `recordedValidation` come from the
    /// DURABLE operation store, not from memory: they are what makes a later
    /// CLI invocation able to reconcile work an earlier one submitted.
    /// Validation is remembered rather than re-run — it is a remote command
    /// whose answer only changes when the environment or the profile does,
    /// and the caller scopes it by the profile hash it was recorded against.
    ///
    /// `recordedDeployIntent` is the same shape of fact from the PER-MACHINE
    /// runtime cache: what this Mac last pushed to the site. The payload
    /// comparison prefers it over the app bundle's own identity, so a site
    /// deployed from a generated payload reads `current` instead of offering a
    /// push that would roll it back (2026-08-24 field report §2.1).
    public func observe(
        site: ClusterSiteRecord,
        configuration: ClusterProvisioningConfiguration,
        recordedControllerJobID: String? = nil,
        recordedValidation: ClusterProfileValidationObservation = .notRun,
        recordedBootstrapPlanHash: String? = nil,
        recordedDeployIntent: ClusterProvisioningOperations.ClusterDeployIntent =
            .init(),
        registrationDuplicates: Int = 1
    ) async -> ClusterObservedState {
        let profile = site.profile
        var state = ClusterObservedState(
            siteID: site.id, siteName: site.displayName,
            siteProfileHash: site.profileHash, observedAt: now())
        // The durable review counts only against the plan this exact site +
        // configuration would render right now.
        if let recordedBootstrapPlanHash,
            let expected = ClusterProvisioningOperations.bootstrapPlanHash(
                site: profile, configuration: configuration)
        {
            state.bootstrapPlanReviewed = recordedBootstrapPlanHash == expected
        }

        let problems = Self.configurationProblems(
            site: profile, configuration: configuration)
        guard problems.isEmpty else {
            state.siteConfiguration = .invalid(problems)
            return state
        }

        // 1. The SSH master. Everything remote depends on it, so a dead master
        //    short-circuits the remote probes rather than emitting a cascade
        //    of misleading "absent" answers.
        state.controlMaster = await operations.checkControlMaster(site: profile)
        let remoteReachable =
            state.controlMaster == .alive || state.controlMaster == .notApplicable

        if profile.isSSHTransport, !remoteReachable {
            let reason = "no SSH ControlMaster — authenticate first"
            state.payload = .unknown(reason: reason)
            state.bootstrap = .unknown(reason: reason)
            state.profileValidation = recordedValidation
            state.controller = profile.topology == .daemonInJob
                ? .unknown(reason: reason) : .notApplicable
            state.controllerScript = profile.topology == .daemonInJob
                ? .unknown(reason: reason) : .notApplicable
            state.daemonHost = profile.topology == .daemonInJob
                ? .absent : .notApplicable
        } else {
            // 2. Deployment identity, 3. bootstrap environment.
            state.payload = await operations.observePayload(
                site: profile, configuration: configuration,
                intent: recordedDeployIntent)
            state.bootstrap = await operations.observeBootstrap(
                site: profile, configuration: configuration,
                envFile: Self.envFile(for: configuration))
            // 4. Profile validation — remembered, scoped by the caller.
            state.profileValidation = profile.isSSHTransport
                ? recordedValidation : .notApplicable
            // 5/6. Controller job, then the host record it is supposed to own.
            //      Order matters: the host file is read once, and is only
            //      trustworthy in light of what the scheduler then says.
            let daemonHost = await operations.readDaemonHost(site: profile)
            state.controller = await operations.observeController(
                site: profile, configuration: configuration,
                recordedJobID: recordedControllerJobID,
                daemonHostPresent: daemonHost != nil)
            state.daemonHost = profile.topology == .daemonInJob
                ? ClusterProvisioningOperations.classifyDaemonHost(
                    host: daemonHost, controller: state.controller)
                : .notApplicable
            // 6b. The RENDERED controller script vs the deployed template. A
            //     read, not a repair: `status`/`plan`/`ensure` are the flows
            //     that must not silently write, so they report the finding and
            //     name the re-render command (§1 field report, 2026-08-20).
            state.controllerScript = await operations.observeControllerScript(
                site: profile, configuration: configuration)
        }

        // 7. The local forward — checked against where the controller runs
        //    NOW. Only a CURRENT daemon-host record can indict the forward's
        //    target; a stale record is not evidence of the new node.
        let persistedPort = Self.persistedPort(for: site)
        let currentDaemonHost: String?
        if case .current(let host) = state.daemonHost {
            currentDaemonHost = host
        } else {
            currentDaemonHost = nil
        }
        state.tunnel = await tunnel.observe(
            site: profile, persistedPort: persistedPort,
            targetHost: currentDaemonHost)

        // 8/9/10. The endpoint, the token, and the saved registration.
        let token = secrets.token(forKey: site.tokenKey)
        state.bearerToken = token == nil ? .absent : .present
        if let baseURL = Self.baseURL(for: site, tunnel: state.tunnel) {
            let probe = await endpoint.probe(baseURL: baseURL, token: token)
            if probe.reachable {
                state.serverHTTP = .reachable(
                    build: probe.serverBuild.flatMap { $0.isEmpty ? nil : $0 },
                    role: probe.serverRole, root: probe.root)
                if token != nil { state.bearerToken = .accepted }
            } else if probe.authFailed {
                state.serverHTTP = .authFailed
                state.bearerToken = token == nil ? .absent : .rejected
            } else {
                state.serverHTTP = .unreachable(
                    reason: probe.detail ?? "the endpoint did not answer")
            }
        } else {
            state.serverHTTP = .unreachable(reason: "no endpoint to probe yet")
        }

        if registrationDuplicates > 1 {
            state.registration = .duplicate(count: registrationDuplicates)
        } else if let endpointString = site.lastEndpoint,
            Self.registrationMatches(site: site, tunnel: state.tunnel)
        {
            state.registration = .present(endpoint: endpointString)
        } else {
            state.registration = .absent
        }
        return state
    }

    // MARK: Derivations

    /// The env file the validate/controller steps source.
    public static func envFile(
        for configuration: ClusterProvisioningConfiguration
    ) -> String {
        let trimmed = configuration.bootstrapEnvFile
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty
            ? ClusterProvisioningConfiguration.defaultEnvFile : trimmed
    }

    /// The local port the registry remembers for this site, if any.
    static func persistedPort(for site: ClusterSiteRecord) -> Int? {
        guard let endpoint = site.lastEndpoint,
            let url = ClusterConnectionStore.endpointURL(from: endpoint)
        else { return site.profile.isSSHTransport ? site.profile.preferredLocalPort : nil }
        return url.port
    }

    /// Where HTTP should go right now: the direct URL, or the live forward's
    /// local label. Nil when there is nothing to talk to.
    static func baseURL(
        for site: ClusterSiteRecord, tunnel: ClusterTunnelObservation
    ) -> URL? {
        if let direct = site.profile.directURLString {
            return ClusterConnectionStore.endpointURL(from: direct)
        }
        guard case .up(let localPort) = tunnel else { return nil }
        return ClusterSiteProfile.localhostBaseURL(port: localPort)
    }

    /// A registration is only PRESENT when the recorded endpoint is the one
    /// the live forward actually uses — a remembered port that moved is a
    /// registration to update, not one to trust.
    static func registrationMatches(
        site: ClusterSiteRecord, tunnel: ClusterTunnelObservation
    ) -> Bool {
        guard site.profile.isSSHTransport else { return site.lastEndpoint != nil }
        guard case .up(let localPort) = tunnel, let endpoint = site.lastEndpoint,
            let url = ClusterConnectionStore.endpointURL(from: endpoint)
        else { return false }
        return url.port == localPort
    }
}
