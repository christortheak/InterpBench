import Foundation

// =============================================================================
// `cluster` verb dispatch (CLUSTER-CLI-LIFECYCLE-PLAN §6) — Phase C.
//
// The orchestration the plan forbids putting in `steerlab-cli/main.swift`
// (§7.1). The binary parses, calls `run`, and serializes what comes back.
//
// Every seam the lifecycle touches — the shell, the endpoint, the forward, the
// Keychain, the Terminal, the clock — is injected, so the entire verb surface
// is exercised in tests with no ssh, no sockets, and no Keychain.
//
// `plan` and `ensure` delegate straight to `ClusterLifecycleCoordinator`: the
// high-level verbs must be the SAME code path the wizard drives, not a second
// implementation. The individual verbs (§6.3) call the same extracted
// operations the coordinator does, for the same reason.
// =============================================================================

/// Streaming for `controller logs --follow`, where the value of the command is
/// lines arriving over time rather than a final exit code.
public protocol ClusterLogStreamer: Sendable {
    func stream(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async -> ClusterShellResult
}

/// Live streamer over the same process runner the rest of the lifecycle uses.
public struct ProvisionLogStreamer: ClusterLogStreamer {
    public init() {}

    public func stream(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async -> ClusterShellResult {
        await ProvisionShellRunner(SystemProvisionRunner(), sink: onLine).run(argv)
    }
}

/// What one `cluster` invocation produced.
public struct ClusterCLIOutcome: Sendable {
    public var envelope: ClusterCLIEnvelope
    public var exitCode: Int32

    public init(envelope: ClusterCLIEnvelope) {
        self.envelope = envelope
        self.exitCode = envelope.exitCode
    }
}

public struct ClusterCLIRunner: Sendable {

    private let repository: ClusterSiteRepository
    private let operationStore: ClusterOperationStore
    private let shell: any ClusterShellRunner
    private let secrets: any ClusterSecretStore
    private let tunnel: any ClusterTunnelControlling
    private let endpointProbe: any ClusterEndpointProbe
    private let authenticationLauncher: any ClusterAuthenticationLauncher
    private let logStreamer: any ClusterLogStreamer
    private let now: @Sendable () -> Date
    /// Cadence and budget for following a submitted bootstrap job. Each probe
    /// is its own short command; tests set the delay to `.zero`.
    private let bootstrapPollDelay: Duration
    private let bootstrapPollLimit: Int
    /// `cluster import`'s engine seam. Nil builds the live ssh/rsync engine
    /// from the resolved site; tests inject a fake so the verb surface is
    /// exercised with no cluster (the same seam style the lifecycle uses).
    private let workspaceImportEngine:
        (@Sendable (ClusterSiteRecord, @escaping @Sendable (String) -> Void) throws
            -> WorkspaceRunImport.Engine)?

    public init(
        repository: ClusterSiteRepository = ClusterSiteRepository(),
        operationStore: ClusterOperationStore = ClusterOperationStore(),
        shell: any ClusterShellRunner = ProvisionShellRunner(),
        secrets: any ClusterSecretStore = KeychainClusterSecretStore(),
        tunnel: any ClusterTunnelControlling = SSHClusterTunnelController(),
        endpointProbe: any ClusterEndpointProbe = HTTPClusterEndpointProbe(),
        authenticationLauncher: any ClusterAuthenticationLauncher =
            TerminalAuthenticationLauncher(),
        logStreamer: any ClusterLogStreamer = ProvisionLogStreamer(),
        now: @escaping @Sendable () -> Date = Date.init,
        bootstrapPollDelay: Duration = .seconds(15),
        bootstrapPollLimit: Int = 480,
        workspaceImportEngine:
            (@Sendable (ClusterSiteRecord, @escaping @Sendable (String) -> Void) throws
                -> WorkspaceRunImport.Engine)? = nil
    ) {
        self.workspaceImportEngine = workspaceImportEngine
        self.repository = repository
        self.operationStore = operationStore
        self.shell = shell
        self.secrets = secrets
        self.tunnel = tunnel
        self.endpointProbe = endpointProbe
        self.authenticationLauncher = authenticationLauncher
        self.logStreamer = logStreamer
        self.now = now
        self.bootstrapPollDelay = bootstrapPollDelay
        self.bootstrapPollLimit = bootstrapPollLimit
    }

    /// How long an opened authentication Terminal counts as "already open", so
    /// repeated `auth open` calls report it instead of spawning windows (§6.2).
    public static let authenticationAttemptWindow: TimeInterval = 120

    // MARK: Entry point

    /// Run one parsed invocation. Never throws: every refusal is an envelope
    /// with a stable code, because a caller must be able to determine the next
    /// action from the machine output alone (§6.5).
    ///
    /// `emit` receives streamed lines (`controller logs --follow`). The caller
    /// decides where they go — stdout in human mode, stderr in JSON mode, so
    /// the one-JSON-document invariant survives.
    public func run(
        _ invocation: ClusterCLIInvocation,
        emit: @escaping @Sendable (String) -> Void = { _ in }
    ) async -> ClusterCLIOutcome {
        // `--help` runs nothing and touches no site: the page is the whole
        // answer, and it is the same generated text in both output modes
        // (WP0 step 11).
        if invocation.help {
            return ClusterCLIOutcome(
                envelope: ClusterCLIEnvelope(
                    verb: invocation.verb.displayName, state: .ready,
                    message: invocation.verb.helpText, observedAt: now()))
        }
        // Absorbing the legacy stores happens at most once per machine and is
        // never silent: what moved, and what was left alone because the
        // canonical registry already held it, goes out on the diagnostic
        // channel before any verb runs (`emit` is stdout in human mode and
        // stderr in JSON mode, so the one-document invariant survives).
        if let summary = (try? repository.migrateLegacyStoresIfNeeded(now: now()))?
            .summary {
            emit("site registry: \(summary)")
        }
        do {
            return ClusterCLIOutcome(envelope: try await dispatch(invocation, emit: emit))
        } catch let error as ClusterLifecycleError {
            return ClusterCLIOutcome(
                envelope: .failure(
                    verb: invocation.verb.displayName, code: error.code,
                    reason: error.errorDescription ?? error.code,
                    repairAction: Self.repairAction(for: error),
                    siteID: invocation.siteReference))
        } catch let error as ClusterCLIError {
            var envelope = ClusterCLIEnvelope.failure(
                verb: invocation.verb.displayName, code: error.code,
                reason: error.reason, repairAction: error.repairAction,
                state: error.exitCode == 13 ? .degraded : .blocked,
                siteID: invocation.siteReference)
            envelope.message = error.reason
            return ClusterCLIOutcome(envelope: envelope)
        } catch {
            return ClusterCLIOutcome(
                envelope: .failure(
                    verb: invocation.verb.displayName, code: "unexpectedFailure",
                    reason: error.localizedDescription,
                    repairAction: "re-run with `cluster status --site <id>` to see "
                        + "which layer failed",
                    siteID: invocation.siteReference))
        }
    }

    static func repairAction(for error: ClusterLifecycleError) -> String {
        switch error {
        case .unknownSite:
            "list the saved sites with `steerlab-cli cluster sites list`"
        case .siteConfigurationInvalid:
            "fix the site profile in the app's cluster settings, or re-import it "
                + "with `cluster sites import`"
        case .operationInProgress(let siteID, _):
            "observe it with `steerlab-cli cluster status --site \(siteID)` rather "
                + "than starting a second one"
        case .bootstrapPlanMismatch, .bootstrapPlanMissing:
            "re-run `cluster bootstrap plan --site <id>` and pass the printed "
                + "--plan-hash"
        case .storeUnwritable:
            "check permissions on the SteerLab home's Sites/cluster-sites "
                + "directory and on ~/Library/Application Support/SteerLab"
        case .siteFileExists:
            "re-run with --force to replace the saved site, or edit the "
                + "profile's name so it imports as a new one"
        case .sshLoginMissing(let host, _):
            "set the profile's transport.ssh.host to 'user@\(host)', or import "
                + "it as-is if this site really authenticates through a `User` "
                + "entry in ~/.ssh/config"
        case .siteFileWouldLeak:
            "remove those keys from the profile JSON — credentials belong in "
                + "the Keychain (`cluster connect` imports the bearer token) "
                + "and connection state is per machine"
        case .controllerAdoptionUnverified(_, _):
            "confirm the job id with `squeue` on the cluster, or start a fresh "
                + "controller with `cluster controller start --site <id>`"
        case .sshLoginDropped(let siteID, let host, let expectedUser, _):
            "set the profile's transport.ssh.host to "
                + "'\(expectedUser)@\(host)' and import it again "
                + "(`cluster sites export --site \(siteID) --out site.json`, "
                + "edit, `cluster sites import site.json`). If this site really "
                + "authenticates through a `User` entry in ~/.ssh/config, "
                + "delete the stored site first (app → cluster settings, or "
                + "remove its file from the Sites registry) and import the "
                + "login-less profile as a new one"
        }
    }

    // MARK: Dispatch

    private func dispatch(
        _ invocation: ClusterCLIInvocation,
        emit: @escaping @Sendable (String) -> Void
    ) async throws -> ClusterCLIEnvelope {
        switch invocation.verb {
        case .sitesList: return try sitesList(invocation)
        case .sitesShow: return try sitesShow(invocation)
        case .sitesExport: return try sitesExport(invocation)
        case .sitesImport: return try sitesImport(invocation)
        case .preview: return try previewVerb(invocation)
        case .status, .diagnose: return try await statusOrDiagnose(invocation)
        case .authCommand: return try authCommand(invocation)
        case .authOpen: return try await authOpen(invocation)
        case .authStatus: return try await authStatus(invocation)
        case .authClose: return try await authClose(invocation)
        case .push: return try await push(invocation)
        case .bootstrapPlan: return try await bootstrapPlan(invocation)
        case .bootstrapApply: return try await bootstrapApply(invocation)
        case .bootstrapStatus: return try await bootstrapStatus(invocation)
        case .validate: return try await validate(invocation)
        case .controllerStart: return try await controllerStart(invocation)
        case .controllerStatus: return try await controllerStatus(invocation)
        case .controllerLogs: return try await controllerLogs(invocation, emit: emit)
        case .controllerStop: return try await controllerStop(invocation)
        case .controllerAdopt: return try await controllerAdopt(invocation)
        case .tunnelOpen: return try await tunnelOpen(invocation)
        case .tunnelStatus: return try await tunnelStatus(invocation)
        case .tunnelClose: return try await tunnelClose(invocation)
        case .connect: return try await connect(invocation)
        case .disconnect: return try await disconnect(invocation)
        case .importRuns: return try await importRuns(invocation, emit: emit)
        case .plan: return try await planVerb(invocation)
        case .ensure: return try await ensureVerb(invocation)
        }
    }

    // MARK: Site resolution and wiring

    private func resolveSite(_ invocation: ClusterCLIInvocation) throws -> ClusterSiteRecord {
        let reference = invocation.siteReference ?? ""
        guard let site = try repository.resolve(reference: reference) else {
            throw ClusterLifecycleError.unknownSite(reference)
        }
        return site
    }

    private func configuration(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord
    ) -> ClusterProvisioningConfiguration {
        invocation.overrides.resolved(for: site.profile)
    }

    private var operations: ClusterProvisioningOperations {
        ClusterProvisioningOperations(
            shell: shell, secrets: secrets,
            bootstrapPollDelay: bootstrapPollDelay,
            bootstrapPollLimit: bootstrapPollLimit)
    }

    private func inspector() -> ClusterLifecycleInspector {
        ClusterLifecycleInspector(
            operations: operations, tunnel: tunnel, endpoint: endpointProbe,
            secrets: secrets, now: now)
    }

    private func coordinator(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord
    ) -> ClusterLifecycleCoordinator {
        ClusterLifecycleCoordinator(
            repository: repository, operationStore: operationStore,
            operations: operations, inspector: inspector(), tunnel: tunnel,
            endpointProbe: endpointProbe,
            authenticationLauncher: authenticationLauncher, secrets: secrets,
            configuration: configuration(invocation, site: site), now: now)
    }

    /// One read-only observation pass. `status`/`diagnose` build it here rather
    /// than through the coordinator so they can fold in the profile-validate
    /// probe (which the coordinator deliberately does NOT re-run per
    /// inspection) when `--refresh` asks for it.
    private func observe(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord,
        recordedValidation: ClusterProfileValidationObservation = .notRun
    ) async -> ClusterObservedState {
        let duplicates = (try? repository.duplicateCount(
            forIdentity: site.canonicalIdentity)) ?? 1
        return await inspector().observe(
            site: site,
            configuration: configuration(invocation, site: site),
            recordedControllerJobID: operationStore.lastControllerJobID(forSite: site.id),
            recordedValidation: recordedValidation,
            recordedBootstrapPlanHash: operationStore.lastBootstrapPlanHash(
                forSite: site.id),
            registrationDuplicates: duplicates)
    }

    /// Persist one non-lifecycle operation's durable facts. Individual verbs
    /// checkpoint through here so a submitted job id, a reviewed plan hash, or
    /// an authentication attempt survives this process exiting (§7.4).
    @discardableResult
    private func record(
        site: ClusterSiteRecord, target: ClusterLifecycleTarget,
        step: ClusterLifecycleStep, status: ClusterOperationStepRecord.Status,
        message: String, state: ClusterLifecycleState,
        controllerJobID: String? = nil, bootstrapJobID: String? = nil,
        bootstrapStatusFile: String? = nil,
        bootstrapPlanHash: String? = nil, localPort: Int? = nil,
        transcript: [String] = []
    ) -> ClusterOperationRecord {
        var record = ClusterOperationRecord(
            operationID: ClusterOperationStore.newOperationID(now: now()),
            siteID: site.id, siteProfileHash: site.profileHash, target: target,
            state: state, startedAt: now(), updatedAt: now())
        record.note(step: step, status: status, message: message, now: now())
        record.controllerJobID = controllerJobID
        record.bootstrapJobID = bootstrapJobID
        record.bootstrapStatusFile = bootstrapStatusFile
        record.bootstrapPlanHash = bootstrapPlanHash
        record.localPort = localPort
        record.transcript = transcript
        try? operationStore.save(record)
        return record
    }

    private func envelope(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord,
        state: ClusterLifecycleState, message: String, changed: Bool = false
    ) -> ClusterCLIEnvelope {
        var envelope = ClusterCLIEnvelope(
            verb: invocation.verb.displayName, state: state, message: message,
            changed: changed, observedAt: now())
        envelope.siteID = site.id
        envelope.siteName = site.displayName
        return envelope
    }

    // MARK: sites

    private func sitesList(_ invocation: ClusterCLIInvocation) throws -> ClusterCLIEnvelope {
        let sites = try repository.sites()
        // A file the registry could not read is a site that would otherwise
        // have vanished in silence. Leniency, then the report.
        let unreadable = repository.unreadableFiles()
        var envelope = ClusterCLIEnvelope(
            verb: invocation.verb.displayName, state: unreadable.isEmpty ? .ready : .degraded,
            message: "\(sites.count) saved cluster site(s) in "
                + repository.directoryURL.path
                + (unreadable.isEmpty
                    ? ""
                    : " — \(unreadable.count) unreadable file(s): "
                        + unreadable.joined(separator: "; ")),
            observedAt: now())
        envelope.sites = sites.map(summary(of:))
        return envelope
    }

    private func sitesShow(_ invocation: ClusterCLIInvocation) throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        var envelope = self.envelope(
            invocation, site: site, state: .ready,
            message: "\(site.displayName) — \(transportSummary(site.profile))")
        envelope.sites = [summary(of: site)]
        envelope.endpoint = site.lastEndpoint
        envelope.serverBuild = site.lastServerBuild
        // Presence only — `sites show` is read-only and must never raise a
        // keychain prompt an unattended agent would wait at forever.
        envelope.tokenAvailable = secrets.hasToken(forKey: site.tokenKey)
        return envelope
    }

    private func sitesExport(_ invocation: ClusterCLIInvocation) throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard let outPath = invocation.outPath else {
            throw ClusterCLIError.missingArgument(
                verb: invocation.verb, what: "--out <file>")
        }
        // The PROFILE only (§6.1): no credentials, and no ephemeral tunnel
        // detail — `lastEndpoint`, the local forward port, and the token key
        // are runtime facts about THIS Mac, never part of the shareable
        // document. The same sanitizer guards the canonical registry files, so
        // an export and a Sites file can never disagree about what is shareable.
        let data = try ClusterSiteSanitizer.encodedExport(for: site)
        let url = URL(filePath: outPath)
        try data.write(to: url, options: .atomic)
        var envelope = self.envelope(
            invocation, site: site, state: .ready,
            message: "exported \(site.displayName)'s profile "
                + "(no credentials, no tunnel detail)",
            changed: true)
        envelope.outputPath = url.path
        return envelope
    }

    private func sitesImport(_ invocation: ClusterCLIInvocation) throws -> ClusterCLIEnvelope {
        guard let path = invocation.positional else {
            throw ClusterCLIError.missingArgument(
                verb: invocation.verb, what: "a profile JSON path")
        }
        let data = try Data(contentsOf: URL(filePath: path))
        // ONE import entry point for every client (the app's menu, the wizard,
        // and here), so the validations below cannot be true of one and not
        // another. `importProfile` refuses to clobber a site the canonical
        // registry already holds unless `--force`: the registry is a git
        // repository the researcher syncs, and a replaced profile is a
        // conflict discovered at connect time.
        //
        // Its `upsert` dedupes by canonical remote identity, so a forced
        // re-import refreshes the site instead of forking the registry — and
        // that same canonical identity ignores the ssh `user@` half, which is
        // how a login-less profile once replaced a login-carrying one in
        // silence (open-issues §17). The write refuses when a KNOWN login
        // would be dropped; a site with no login anywhere is legal and warns
        // here, in the message, so it is visible in both output modes.
        var loginWarnings: [String] = []
        let record = try repository.importProfile(
            from: data, force: invocation.force, now: now(),
            warn: { loginWarnings.append($0) })
        var envelope = ClusterCLIEnvelope(
            verb: invocation.verb.displayName, state: .ready,
            message: "imported '\(record.displayName)' as site \(record.id) in "
                + repository.directoryURL.path
                + loginWarnings.map { " — WARNING: \($0)" }.joined(),
            changed: true, observedAt: now())
        envelope.outputPath = repository.fileURL(forSite: record.id).path
        envelope.siteID = record.id
        envelope.siteName = record.displayName
        envelope.sites = [summary(of: record)]
        return envelope
    }

    // MARK: preview (WP5 §3.3)

    /// Render the saved profile and return it. Deliberately offline: no shell,
    /// no scheduler, no probe — the whole point is that an admin can read the
    /// generated environment and scheduler commands BEFORE anything runs, and a
    /// preview that needed a live cluster could not be read first.
    private func previewVerb(
        _ invocation: ClusterCLIInvocation
    ) throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let preview = ClusterSitePreview(
            site.profile,
            jobClasses: invocation.jobClass.map { [$0] }
                ?? ClusterEnvironmentRenderer.JobClass.allCases)
        var envelope = self.envelope(
            invocation, site: site, state: .ready,
            message: "\(site.displayName) — \(preview.summaryLine)")
        envelope.preview = preview
        return envelope
    }

    /// The listing payload. `tokenAvailable` is a PRESENCE claim and is
    /// answered without reading the secret (`ClusterTokenStore.presence`), so
    /// `sites list` / `sites show` / `sites import` stay promptless by
    /// construction for a freshly installed binary whose identity no keychain
    /// ACL names yet.
    private func summary(of site: ClusterSiteRecord) -> ClusterCLIEnvelope.Site {
        ClusterCLIEnvelope.Site(
            id: site.id,
            name: site.displayName,
            transport: transportSummary(site.profile),
            topology: site.profile.topology.rawValue,
            scheduler: schedulerSummary(site.profile),
            lastEndpoint: site.lastEndpoint,
            lastServerBuild: site.lastServerBuild,
            tokenAvailable: secrets.hasToken(forKey: site.tokenKey))
    }

    private func transportSummary(_ profile: ClusterSiteProfile) -> String {
        profile.registryIdentity ?? profile.directURLString ?? "unconfigured"
    }

    private func schedulerSummary(_ profile: ClusterSiteProfile) -> String {
        if case .slurm = profile.scheduler { return "slurm" }
        return "none"
    }

    // MARK: status / diagnose

    private func statusOrDiagnose(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        // `--refresh` (and `diagnose` always) folds in the read-only remote
        // profile-validate probe, which inspection otherwise remembers rather
        // than re-running on every status call.
        var validation: ClusterProfileValidationObservation = .notRun
        if invocation.refresh || invocation.verb == .diagnose {
            let configuration = configuration(invocation, site: site)
            let bootstrap = await operations.observeBootstrap(
                site: site.profile, configuration: configuration,
                envFile: ClusterLifecycleInspector.envFile(for: configuration))
            var prefix: String?
            if case .valid(_, let value) = bootstrap { prefix = value }
            validation = await operations.validate(
                site: site.profile,
                envFile: ClusterLifecycleInspector.envFile(for: configuration),
                prefix: prefix
            ).observation
        }
        let observed = await observe(
            invocation, site: site, recordedValidation: validation)
        // Read-only report: the plan is computed with every mutation permitted,
        // so `status` says what is actually true rather than "you have not
        // passed --allow-push". Approval is `ensure`'s business, not a fact
        // about the site.
        let permissions: ClusterLifecyclePermissions = .allMutations
        let plan = ClusterLifecyclePlanner.plan(
            site: site.profile, observed: observed, target: invocation.target,
            permissions: permissions)
        let state = plan.state(permissions: permissions)
        var envelope = self.envelope(
            invocation, site: site, state: state,
            message: state == .ready
                ? "\(site.displayName) is \(invocation.target.cliName)"
                : (plan.next?.reason ?? "work remains"))
        envelope.attach(observed: observed)
        envelope.attach(plan: plan)
        envelope.step = plan.next?.step.rawValue
        envelope.endpoint = ClusterLifecycleInspector.baseURL(
            for: site, tunnel: observed.tunnel)?.absoluteString
        envelope.tokenAvailable = observed.bearerToken.isAvailable
        envelope.tokenSource = observed.bearerToken.isAvailable ? "keychain" : nil
        envelope.serverBuild = observed.serverHTTP.serverBuild
        envelope.schedulerJobID = observed.controller.jobID
        envelope.schedulerState = observed.controller.schedulerState

        if invocation.verb == .diagnose {
            envelope.command = authenticationArgv(site.profile)
            if let jobID = observed.controller.jobID {
                envelope.logPath = ClusterProvisioner.controllerLogHint(
                    jobID: jobID, site: site.profile)
            }
            envelope.operations = operationStore.records(forSite: site.id)
                .prefix(5)
                .map { record in
                    ClusterCLIEnvelope.Operation(
                        operationID: record.operationID, state: record.state.rawValue,
                        target: record.target.cliName, startedAt: record.startedAt,
                        updatedAt: record.updatedAt,
                        authorizedMutations: record.authorizedMutations,
                        controllerJobID: record.controllerJobID,
                        bootstrapJobID: record.bootstrapJobID,
                        failureCode: record.failureCode,
                        repairAction: record.repairAction)
                }
            if invocation.redact { envelope = envelope.redacted() }
        }
        return envelope
    }

    // MARK: auth (§4.1 — the CLI opens a window and nothing else)

    /// The authentication one-liner as an argv array (§6.5). The composition in
    /// `ClusterTunnel.authenticationCommand` joins space-free words, so the
    /// split is exact rather than a shell-parsing guess.
    private func authenticationArgv(_ profile: ClusterSiteProfile) -> [String]? {
        authenticationLauncher.authenticationCommand(for: profile)?
            .split(separator: " ").map(String.init)
    }

    private func authCommand(_ invocation: ClusterCLIInvocation) throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard let argv = authenticationArgv(site.profile) else {
            return envelope(
                invocation, site: site, state: .ready,
                message: "this site needs no SSH authentication (direct transport)")
        }
        var envelope = self.envelope(
            invocation, site: site, state: .ready,
            message: "run this in YOUR OWN Terminal to authenticate to "
                + "\(site.displayName); SteerLab never sees the password or the "
                + "multi-factor response")
        envelope.command = argv
        return envelope
    }

    private func authStatus(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let master = await operations.checkControlMaster(site: site.profile)
        switch master {
        case .alive:
            return envelope(
                invocation, site: site, state: .ready,
                message: "the SSH ControlMaster is alive")
        case .notApplicable:
            return envelope(
                invocation, site: site, state: .ready,
                message: "direct transport — nothing to authenticate")
        case .absent, .unresponsive:
            var envelope = self.envelope(
                invocation, site: site, state: .needsHumanAuthentication,
                message: master == .unresponsive
                    ? "the ControlMaster socket exists but did not answer — "
                        + "authenticate again"
                    : "no SSH ControlMaster — a human must authenticate")
            envelope.command = authenticationArgv(site.profile)
            envelope.retryAfterSeconds = 5
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.authOpen.displayName, requiresHuman: true,
                missingPermissionFlags: [],
                detail: authenticationLauncher.authenticationCommand(for: site.profile))
            return envelope
        }
    }

    private func authOpen(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let master = await operations.checkControlMaster(site: site.profile)
        // 1. A live master: return ready WITHOUT opening a window (§6.2).
        if master == .alive || master == .notApplicable {
            return envelope(
                invocation, site: site, state: .ready,
                message: master == .alive
                    ? "already authenticated — no Terminal was opened"
                    : "direct transport — nothing to authenticate")
        }
        // 2. An attempt opened recently: report IT rather than spawning a
        //    second window.
        if let opened = lastAuthenticationAttempt(siteID: site.id),
            now().timeIntervalSince(opened) < Self.authenticationAttemptWindow
        {
            let age = Int(now().timeIntervalSince(opened))
            var envelope = self.envelope(
                invocation, site: site, state: .needsHumanAuthentication,
                message: "an authentication Terminal was already opened \(age)s ago — "
                    + "complete it there; no second window was opened")
            envelope.command = authenticationArgv(site.profile)
            envelope.retryAfterSeconds = 5
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.authStatus.displayName, requiresHuman: true)
            return envelope
        }
        // 3. Open exactly one visible Terminal. The CLI never reads it.
        let opened = await authenticationLauncher.openAuthenticationTerminal(
            for: site.profile)
        record(
            site: site, target: .authenticated, step: .authenticate,
            status: opened ? .running : .failed,
            message: opened
                ? "opened a visible Terminal for authentication"
                : "could not open Terminal — run the printed command yourself",
            state: .needsHumanAuthentication)
        var envelope = self.envelope(
            invocation, site: site, state: .needsHumanAuthentication,
            message: opened
                ? "complete password and multi-factor authentication in the opened "
                    + "Terminal window, then repeat this command"
                : "could not open a Terminal — run the printed command yourself",
            changed: opened)
        envelope.command = authenticationArgv(site.profile)
        envelope.retryAfterSeconds = 5
        envelope.nextAction = .init(
            verb: ClusterCLIVerb.authStatus.displayName, requiresHuman: true,
            detail: authenticationLauncher.authenticationCommand(for: site.profile))
        return envelope
    }

    /// When this site last had an authentication Terminal opened for it — the
    /// durable half of `auth open`'s idempotence, so window spam is prevented
    /// across processes rather than only within one.
    private func lastAuthenticationAttempt(siteID: String) -> Date? {
        operationStore.records(forSite: siteID)
            .compactMap { record in
                record.steps.first { $0.step == .authenticate }?.startedAt
            }
            .max()
    }

    private func authClose(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard case .ssh(let host, _, _, _) = site.profile.transport else {
            return envelope(
                invocation, site: site, state: .ready,
                message: "direct transport — there is no ControlMaster to close")
        }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let result = await shell.run(ClusterProvisioner.masterExitArgv(host: trimmed))
        // `-O exit` against SteerLab's own ControlPath: unrelated SSH sessions
        // and sockets are untouched (§6.2).
        var envelope = self.envelope(
            invocation, site: site,
            state: .ready,
            message: result.succeeded
                ? "closed the SSH ControlMaster for \(trimmed)"
                : "no ControlMaster was open for \(trimmed)",
            changed: result.succeeded)
        envelope.command = ClusterProvisioner.masterExitArgv(host: trimmed)
        return envelope
    }

    // MARK: push

    private func push(_ invocation: ClusterCLIInvocation) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let configuration = configuration(invocation, site: site)
        if invocation.dryRun {
            let argv = ClusterProvisioner.pushArgv(
                site: site.profile,
                localRepoPath: configuration.localPayloadPath,
                remoteRepoPath: configuration.remoteRepoPath,
                includeDeploymentManifest: ClusterProvisioner.deploymentManifestExists(
                    atPayloadRoot: configuration.localPayloadPath))
            var envelope = self.envelope(
                invocation, site: site, state: .planned,
                message: argv == nil
                    ? "no SSH path to push over — place the server bundle on the box "
                        + "yourself"
                    : "dry run: this is the exact command `cluster push` would run")
            envelope.command = argv
            return envelope
        }
        let outcome = await operations.push(site: site.profile, configuration: configuration)
        return operationEnvelope(
            invocation, site: site, step: .pushCode, target: .codeDeployed,
            outcome: outcome)
    }

    // MARK: import (open-issues §20)

    /// Bring the paired cluster workspace's runs home under
    /// `WorkspaceImportPolicy`, then rebuild `catalog/`.
    ///
    /// The verb is a thin shell over `WorkspaceRunImport.run` — the SAME code
    /// path the app's import hook drives — because the policy having two
    /// implementations is exactly the drift §20 exists to end. Every line the
    /// operation emits goes to `emit` (stderr in JSON mode), so the one-JSON-
    /// document invariant survives a long transfer that streams progress.
    private func importRuns(
        _ invocation: ClusterCLIInvocation,
        emit: @escaping @Sendable (String) -> Void
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let engine: WorkspaceRunImport.Engine
        do {
            if let workspaceImportEngine {
                engine = try workspaceImportEngine(site, emit)
            } else {
                engine = try await WorkspaceRunImport.liveEngine(
                    site: site.profile, siteID: site.id,
                    workspaceRoot: ExperimentStore.workspaceRoot,
                    shell: shell, emit: emit)
            }
        } catch let error as WorkspaceRunImport.SetupError {
            var envelope = ClusterCLIEnvelope.failure(
                verb: invocation.verb.displayName, code: error.code,
                reason: error.reason, repairAction: error.repairAction,
                state: .blocked, siteID: site.id)
            envelope.message = error.reason
            return envelope
        }

        let report = await WorkspaceRunImport.run(
            engine: engine,
            options: WorkspaceRunImport.Options(
                since: invocation.since, dryRun: invocation.dryRun),
            emit: emit)

        let loud = report.hasLoudPurgeFindings || report.hasAuthoringDivergences
        let broken = !report.violations.isEmpty || !report.failures.isEmpty
        let state: ClusterLifecycleState = broken ? .failed : (loud ? .degraded : .ready)
        var envelope = self.envelope(
            invocation, site: site, state: state,
            message: WorkspaceRunImport.summaryLines(report).joined(separator: "\n"),
            // A dry run changes nothing anywhere, by construction.
            changed: !invocation.dryRun && !report.imported.isEmpty)
        envelope.importSummary = summary(of: report)
        if broken {
            envelope.error = ClusterCLIEnvelope.Failure(
                code: "importIncomplete",
                reason: "\(report.violations.count) violation(s) and "
                    + "\(report.failures.count) failed director(ies) — nothing "
                    + "was overwritten and nothing was deleted",
                repairAction: "read the VIOLATIONS section: a byte-drift "
                    + "refusal needs a human decision about which copy is the "
                    + "real run, and never a re-run of this verb")
        } else if loud {
            var details: [String] = []
            if report.hasLoudPurgeFindings {
                details.append(
                    "shard partials without an evidenced merge are on the "
                        + "cluster — merge (or resume) those fan-outs before "
                        + "the site's retention window drops them")
            }
            if report.hasAuthoringDivergences {
                details.append(
                    "cluster-side authoring diverges from the live workspace "
                        + "manifests — read the AUTHORING DIVERGENCE section "
                        + "and decide, study by study, whether to adopt the "
                        + "run snapshot by hand (the import never writes "
                        + "experiments/)")
            }
            envelope.nextAction = .init(
                verb: invocation.verb.displayName, requiresHuman: true,
                detail: details.joined(separator: "; and "))
        }
        return envelope
    }

    private func summary(
        of report: WorkspaceRunImport.Report
    ) -> ClusterCLIEnvelope.ImportSummary {
        ClusterCLIEnvelope.ImportSummary(
            dryRun: report.dryRun,
            imported: report.imported.map(\.name),
            alreadyComplete: report.directories.compactMap { directory in
                if case .alreadyComplete = directory.outcome { return directory.name }
                return nil
            },
            skippedByPolicy: report.skippedByPolicy.map(\.name),
            unknownShapes: report.unknowns,
            purgeEligible: report.purgeFamilies
                .filter { $0.verdict.isPurgeEligible }.map(\.message),
            purgeBlocked: report.purgeFamilies
                .filter { $0.verdict.isLoud }.map(\.message),
            authoringDivergences: report.authoringDivergences.map {
                WorkspaceImportPolicy.message(divergence: $0)
            },
            violations: report.violations,
            catalogRuns: report.catalog?.rows.count)
    }

    // MARK: bootstrap

    private func bootstrapPlan(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let outcome = await operations.bootstrapPlan(
            site: site.profile, configuration: configuration(invocation, site: site))
        // The review must survive this process exiting (§7.9): the hash goes to
        // the durable store, so a later `ensure` honours it too.
        if let planHash = outcome.planHash {
            record(
                site: site, target: .bootstrapped, step: .bootstrapPlan,
                status: .succeeded, message: outcome.outcome.message, state: .planned,
                bootstrapPlanHash: planHash, transcript: outcome.outcome.transcript)
        }
        var envelope = operationEnvelope(
            invocation, site: site, step: .bootstrapPlan, target: .bootstrapped,
            outcome: outcome.outcome,
            successState: outcome.planHash == nil ? .ready : .planned)
        envelope.planHash = outcome.planHash
        if let planHash = outcome.planHash {
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.bootstrapApply.displayName,
                detail: "review the plan above, then run `cluster bootstrap apply "
                    + "--site \(site.id) --plan-hash \(planHash)`")
        }
        return envelope
    }

    private func bootstrapApply(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        // Throws `bootstrapPlanMismatch` / `bootstrapPlanMissing` BEFORE any
        // command runs — the gate is in the operation, not in this layer.
        // The journal is how a submitted job survives this process exiting:
        // apply persists the pending job before it polls, and a later apply
        // RESUMES that job rather than queueing a second one.
        let outcome = try await operations.bootstrapApply(
            site: site.profile, configuration: configuration(invocation, site: site),
            reviewedPlanHash: invocation.planHash,
            journal: operationStore, siteID: site.id)
        // A still-pending job is not a terminal outcome: leaving the record
        // non-terminal is what makes the next invocation resume it.
        record(
            site: site, target: .bootstrapped, step: .bootstrapApply,
            status: outcome.outcome.succeeded
                ? .succeeded : (outcome.stillPending ? .awaitingScheduler : .failed),
            message: outcome.outcome.message,
            state: outcome.outcome.succeeded
                ? .ready : (outcome.stillPending ? .pending : .failed),
            bootstrapJobID: outcome.jobID,
            bootstrapStatusFile: outcome.statusFile,
            bootstrapPlanHash: invocation.planHash,
            transcript: outcome.outcome.transcript)
        var envelope = operationEnvelope(
            invocation, site: site, step: .bootstrapApply, target: .bootstrapped,
            outcome: outcome.outcome,
            successState: .ready,
            failureState: outcome.stillPending ? .pending : .failed)
        envelope.schedulerJobID = outcome.jobID
        envelope.planHash = invocation.planHash
        if outcome.stillPending {
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.bootstrapApply.displayName,
                detail: "job \(outcome.jobID ?? "?") is still in the queue — rerun "
                    + "`cluster bootstrap apply --site \(site.id) --plan-hash "
                    + "\(invocation.planHash ?? "")` to resume following it; it will "
                    + "not be resubmitted")
        }
        return envelope
    }

    private func bootstrapStatus(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let configuration = configuration(invocation, site: site)
        let observation = await operations.observeBootstrap(
            site: site.profile, configuration: configuration,
            envFile: ClusterLifecycleInspector.envFile(for: configuration))
        let state: ClusterLifecycleState
        switch observation {
        case .valid, .notApplicable: state = .ready
        // An absent or invalid environment is a FACT, not a failure: it is
        // exactly the work `bootstrap plan` then `apply` exists to do.
        case .absent, .invalid: state = .planned
        case .unknown: state = .degraded
        }
        var envelope = self.envelope(
            invocation, site: site, state: state,
            message: "bootstrap environment: \(observation.summary)")
        if state == .planned {
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.bootstrapPlan.displayName,
                detail: "render and review the bootstrap plan first")
        }
        if state == .degraded { envelope.retryAfterSeconds = 15 }
        return envelope
    }

    // MARK: validate

    private func validate(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let configuration = configuration(invocation, site: site)
        let bootstrap = await operations.observeBootstrap(
            site: site.profile, configuration: configuration,
            envFile: ClusterLifecycleInspector.envFile(for: configuration))
        var prefix: String?
        if case .valid(_, let value) = bootstrap { prefix = value }
        let outcome = await operations.validate(
            site: site.profile,
            envFile: ClusterLifecycleInspector.envFile(for: configuration),
            prefix: prefix)
        var envelope = operationEnvelope(
            invocation, site: site, step: .validate, target: .validated,
            outcome: outcome.outcome)
        envelope.layers = [
            .init(layer: "profileValidation", state: outcome.observation.summary)
        ]
        return envelope
    }

    // MARK: controller

    private func controllerStart(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let configuration = configuration(invocation, site: site)
        // `--render-only` (open-issues §1 field report, 2026-08-20): refresh
        // the RENDERED script and stop. Deliberately BEFORE the reconcile
        // below — the site that needs this most is one whose controller is
        // running, and "a controller is already running" must not turn the
        // repair into a no-op.
        if invocation.renderOnly {
            let envFile = ClusterLifecycleInspector.envFile(for: configuration)
            var prefix: String?
            if case .valid(_, let value) = await operations.observeBootstrap(
                site: site.profile, configuration: configuration, envFile: envFile)
            {
                prefix = value
            }
            let outcome = await operations.controllerRender(
                site: site.profile, configuration: configuration,
                envFile: envFile, prefix: prefix)
            var envelope = self.envelope(
                invocation, site: site,
                state: outcome.succeeded ? .ready : .failed,
                message: outcome.message)
            envelope.changed = outcome.succeeded
            if outcome.succeeded {
                envelope.nextAction = .init(
                    verb: ClusterCLIVerb.controllerStart.displayName,
                    detail: "the RUNNING controller still came from the old "
                        + "script and cannot chain — cycle it once (let it "
                        + "expire or `controller stop`, then start) and every "
                        + "generation after that self-chains")
            }
            return envelope
        }
        // Reconcile FIRST: a site that already has a live or queued controller
        // must never get a second one (§7.10).
        let recorded = operationStore.lastControllerJobID(forSite: site.id)
        let existing = await operations.observeController(
            site: site.profile, configuration: configuration,
            recordedJobID: recorded,
            daemonHostPresent: await operations.readDaemonHost(site: site.profile) != nil)
        switch existing {
        case .running(let jobID), .pending(let jobID, _):
            var envelope = self.envelope(
                invocation, site: site,
                state: existing.schedulerState == "RUNNING" ? .ready : .pending,
                message: "controller job \(jobID) is already "
                    + "\(existing.schedulerState ?? "in flight") — not submitting a second")
            envelope.schedulerJobID = jobID
            envelope.schedulerState = existing.schedulerState
            envelope.retryAfterSeconds = existing.schedulerState == "RUNNING" ? nil : 30
            return envelope
        case .unknown(let reason):
            var envelope = self.envelope(
                invocation, site: site, state: .degraded,
                message: "the existing controller's state could not be read (\(reason)) "
                    + "— an unproven death never licenses a resubmit")
            envelope.retryAfterSeconds = 15
            return envelope
        case .absent, .failed, .notApplicable:
            break
        }
        let bootstrap = await operations.observeBootstrap(
            site: site.profile, configuration: configuration,
            envFile: ClusterLifecycleInspector.envFile(for: configuration))
        var prefix: String?
        if case .valid(_, let value) = bootstrap { prefix = value }
        let outcome = await operations.controllerStart(
            site: site.profile, configuration: configuration,
            envFile: ClusterLifecycleInspector.envFile(for: configuration),
            prefix: prefix)
        if let jobID = outcome.jobID {
            record(
                site: site, target: .controllerRunning, step: .controllerStart,
                status: .awaitingScheduler, message: outcome.outcome.message,
                state: .pending, controllerJobID: jobID,
                transcript: outcome.outcome.transcript)
        }
        // Submit and RETURN (§6.3): the queue wait belongs to `controller
        // status` and `ensure`, never a foreground ssh session.
        var envelope = operationEnvelope(
            invocation, site: site, step: .controllerStart,
            target: .controllerRunning, outcome: outcome.outcome,
            successState: outcome.jobID == nil ? .ready : .pending)
        envelope.schedulerJobID = outcome.jobID
        if outcome.jobID != nil {
            envelope.retryAfterSeconds = 30
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.controllerStatus.displayName,
                detail: "poll the queue; a queued job stays pending, it never times out")
        }
        return envelope
    }

    private func controllerObservation(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord
    ) async -> (ClusterControllerObservation, ClusterDaemonHostObservation) {
        let configuration = configuration(invocation, site: site)
        let jobID = invocation.jobID
            ?? operationStore.lastControllerJobID(forSite: site.id)
        let host = await operations.readDaemonHost(site: site.profile)
        let controller = await operations.observeController(
            site: site.profile, configuration: configuration, recordedJobID: jobID,
            daemonHostPresent: host != nil)
        let daemonHost = site.profile.topology == .daemonInJob
            ? ClusterProvisioningOperations.classifyDaemonHost(
                host: host, controller: controller)
            : ClusterDaemonHostObservation.notApplicable
        return (controller, daemonHost)
    }

    private func controllerStatus(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let (controller, daemonHost) = await controllerObservation(invocation, site: site)
        let state: ClusterLifecycleState
        switch controller {
        case .running, .notApplicable: state = .ready
        case .pending: state = .pending
        case .absent: state = .planned
        case .failed: state = .failed
        case .unknown: state = .degraded
        }
        var envelope = self.envelope(
            invocation, site: site, state: state,
            message: "controller: \(controller.summary) · serverd.host: "
                + daemonHost.summary)
        envelope.schedulerJobID = controller.jobID
        envelope.schedulerState = controller.schedulerState
        envelope.layers = [
            .init(layer: "controller", state: controller.summary),
            .init(layer: "daemonHost", state: daemonHost.summary),
        ]
        if let jobID = controller.jobID {
            envelope.logPath = ClusterProvisioner.controllerLogHint(
                jobID: jobID, site: site.profile)
        }
        if state == .pending { envelope.retryAfterSeconds = 30 }
        if state == .degraded { envelope.retryAfterSeconds = 15 }
        if state == .planned {
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.controllerStart.displayName)
        }
        return envelope
    }

    private func controllerLogs(
        _ invocation: ClusterCLIInvocation,
        emit: @escaping @Sendable (String) -> Void
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard let jobID = invocation.jobID
            ?? operationStore.lastControllerJobID(forSite: site.id)
        else {
            var envelope = self.envelope(
                invocation, site: site, state: .planned,
                message: "no controller job is recorded for this site — nothing to tail")
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.controllerAdopt.displayName,
                detail: "if a controller is already running, record it with "
                    + "`cluster controller adopt --site \(site.id) --job-id <id>`")
            return envelope
        }
        let path = ClusterProvisioner.controllerLogHint(jobID: jobID, site: site.profile)
        let argv = ClusterProvisioner.sshRemoteArgv(
            site: site.profile,
            remoteWords: invocation.follow
                ? ["tail", "-n", "200", "-f", path] : ["tail", "-n", "200", path])
        guard !argv.isEmpty else {
            return envelope(
                invocation, site: site, state: .blocked,
                message: "direct transport — read the controller log on the box itself")
        }
        let result: ClusterShellResult
        if invocation.follow {
            result = await logStreamer.stream(argv, onLine: emit)
        } else {
            result = await shell.run(argv)
            for line in result.lines { emit(line) }
        }
        var envelope = self.envelope(
            invocation, site: site,
            state: result.succeeded ? .ready : .degraded,
            message: result.succeeded
                ? "tailed \(result.lines.count) line(s) of controller job \(jobID)"
                : "could not read the controller log (exit \(result.exitCode))")
        envelope.schedulerJobID = jobID
        envelope.logPath = path
        envelope.command = argv
        if !result.succeeded { envelope.retryAfterSeconds = 15 }
        return envelope
    }

    private func controllerStop(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard let jobID = invocation.jobID
            ?? operationStore.lastControllerJobID(forSite: site.id)
        else {
            var envelope = self.envelope(
                invocation, site: site, state: .ready,
                message: "no controller job is recorded for this site — nothing to stop")
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.controllerAdopt.displayName,
                detail: "record a hand-started controller before stopping it")
            return envelope
        }
        let outcome = await operations.controllerStop(site: site.profile, jobID: jobID)
        record(
            site: site, target: .controllerRunning, step: .controllerStart,
            status: outcome.succeeded ? .succeeded : .failed,
            message: outcome.message,
            state: outcome.succeeded ? .ready : .failed,
            transcript: outcome.transcript)
        var envelope = operationEnvelope(
            invocation, site: site, step: .controllerStart,
            target: .controllerRunning, outcome: outcome)
        envelope.schedulerJobID = jobID
        return envelope
    }

    private func controllerAdopt(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        guard let jobID = invocation.jobID else {
            throw ClusterCLIError.missingArgument(
                verb: invocation.verb, what: "--job-id <scheduler job id>")
        }
        let configuration = configuration(invocation, site: site)
        // Probe the endpoint too when a forward already exists — adopting a job
        // whose endpoint answers as someone ELSE would attach this site to the
        // wrong server.
        let tunnelObservation = await tunnel.observe(
            site: site.profile,
            persistedPort: ClusterLifecycleInspector.persistedPort(for: site),
            targetHost: nil)
        let baseURL = ClusterLifecycleInspector.baseURL(
            for: site, tunnel: tunnelObservation)
        let outcome = await operations.adoptController(
            site: site.profile, configuration: configuration, jobID: jobID,
            endpointProbe: baseURL == nil ? nil : endpointProbe, baseURL: baseURL,
            token: secrets.token(forKey: site.tokenKey))
        guard outcome.verified else {
            // NOT recorded. A poisoned job id would make every later inspection
            // believe in a controller that does not exist.
            throw ClusterLifecycleError.controllerAdoptionUnverified(
                jobID: outcome.jobID,
                reason: outcome.refusalReason ?? "verification failed")
        }
        record(
            site: site, target: .controllerRunning, step: .controllerStart,
            status: outcome.controller.schedulerState == "RUNNING"
                ? .succeeded : .awaitingScheduler,
            message: "adopted hand-started controller job \(outcome.jobID)",
            state: outcome.controller.schedulerState == "RUNNING" ? .ready : .pending,
            controllerJobID: outcome.jobID)
        var envelope = self.envelope(
            invocation, site: site,
            state: outcome.controller.schedulerState == "RUNNING" ? .ready : .pending,
            message: "recorded controller job \(outcome.jobID) for \(site.displayName) — "
                + outcome.outcome.message,
            changed: true)
        envelope.schedulerJobID = outcome.jobID
        envelope.schedulerState = outcome.controller.schedulerState
        envelope.layers = [
            .init(layer: "controller", state: outcome.controller.summary),
            .init(layer: "daemonHost", state: outcome.daemonHost.summary),
            .init(
                layer: "serverHTTP",
                state: outcome.serverHTTP?.summary ?? "not probed (no forward yet)"),
        ]
        if outcome.controller.schedulerState != "RUNNING" {
            envelope.retryAfterSeconds = 30
        }
        return envelope
    }

    // MARK: tunnel

    /// Where the forward should point. Daemon-in-a-job forwards to the node the
    /// controller published, and ONLY when that record is current — a stale
    /// host file is not a target (§7.10).
    private func tunnelTargetHost(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord
    ) async -> (host: String?, daemonHost: ClusterDaemonHostObservation) {
        guard site.profile.topology == .daemonInJob else { return ("localhost", .notApplicable) }
        let (_, daemonHost) = await controllerObservation(invocation, site: site)
        if case .current(let host) = daemonHost { return (host, daemonHost) }
        return (nil, daemonHost)
    }

    private func tunnelOpen(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let (targetHost, daemonHost) = await tunnelTargetHost(invocation, site: site)
        guard let targetHost else {
            var envelope = self.envelope(
                invocation, site: site, state: .degraded,
                message: "the controller has not published a usable node record "
                    + "(\(daemonHost.summary)) — there is nothing to forward to")
            envelope.retryAfterSeconds = 30
            envelope.nextAction = .init(
                verb: ClusterCLIVerb.controllerStatus.displayName)
            return envelope
        }
        let outcome = await tunnel.open(
            site: site.profile, targetHost: targetHost,
            persistedPort: ClusterLifecycleInspector.persistedPort(for: site))
        var isUp = false
        if case .up = outcome.observation { isUp = true }
        if isUp {
            record(
                site: site, target: .connected, step: .tunnelOpen,
                status: .succeeded, message: outcome.message, state: .ready,
                localPort: outcome.observation.localPort)
        }
        var envelope = self.envelope(
            invocation, site: site, state: isUp ? .ready : .degraded,
            message: outcome.message, changed: outcome.changed)
        envelope.layers = [.init(layer: "tunnel", state: outcome.observation.summary)]
        if let port = outcome.observation.localPort, isUp {
            envelope.endpoint = ClusterSiteProfile.localhostBaseURL(port: port)
                .absoluteString
        }
        if !isUp { envelope.retryAfterSeconds = 15 }
        return envelope
    }

    private func tunnelStatus(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let observation = await tunnel.observe(
            site: site.profile,
            persistedPort: ClusterLifecycleInspector.persistedPort(for: site),
            targetHost: nil)
        let state: ClusterLifecycleState
        switch observation {
        case .up, .notApplicable: state = .ready
        case .absent: state = .planned
        case .stale, .conflicted: state = .degraded
        }
        var envelope = self.envelope(
            invocation, site: site, state: state,
            message: "tunnel: \(observation.summary)")
        envelope.layers = [.init(layer: "tunnel", state: observation.summary)]
        if case .up(let port) = observation {
            envelope.endpoint = ClusterSiteProfile.localhostBaseURL(port: port)
                .absoluteString
        }
        if state == .degraded { envelope.retryAfterSeconds = 15 }
        return envelope
    }

    private func tunnelClose(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let persisted = ClusterLifecycleInspector.persistedPort(for: site)
        let observation = await tunnel.observe(
            site: site.profile, persistedPort: persisted, targetHost: nil)
        guard let port = observation.localPort else {
            return envelope(
                invocation, site: site, state: .ready,
                message: "no forward is recorded for this site — nothing to close")
        }
        let (targetHost, _) = await tunnelTargetHost(invocation, site: site)
        // The controller may be gone; the cancel still has to name the exact
        // forward SteerLab installed, so fall back to the published host or the
        // login host's own localhost rather than guessing something else's.
        if let closeError = await tunnel.close(
            site: site.profile, localPort: port, targetHost: targetHost ?? "localhost")
        {
            // A cancel that did not verifiably remove the listener is a
            // failure to report, never a success to log (§7.7): the live bug
            // was a "closed" claim over a forward that kept listening.
            var failure = self.envelope(
                invocation, site: site, state: .degraded, message: closeError)
            failure.layers = [.init(layer: "tunnel", state: "conflicted (\(port))")]
            failure.retryAfterSeconds = 15
            return failure
        }
        return envelope(
            invocation, site: site, state: .ready,
            message: "closed SteerLab's forward on 127.0.0.1:\(port)", changed: true)
    }

    // MARK: connect / disconnect

    private func connect(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        // `connect` IS `ensure --target connected` with no mutation authority:
        // the tunnel and the registration are the connect rung and need none,
        // and anything below that is missing stops at its own boundary rather
        // than being silently performed.
        let result = try await coordinator(invocation, site: site).ensure(
            siteReference: site.id, target: .connected, permissions: [])
        // There is no `--activate-in-app` any more, and nothing replaces it:
        // both clients read the same Sites registry, so a connection the CLI
        // established is already the app's too — activation stopped being a
        // separate step when the two stores became one.
        return ClusterCLIEnvelope.lifecycle(verb: invocation.verb, result: result)
    }

    private func disconnect(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let persisted = ClusterLifecycleInspector.persistedPort(for: site)
        let observation = await tunnel.observe(
            site: site.profile, persistedPort: persisted, targetHost: nil)
        var closeError: String?
        if let port = observation.localPort {
            let (targetHost, _) = await tunnelTargetHost(invocation, site: site)
            closeError = await tunnel.close(
                site: site.profile, localPort: port, targetHost: targetHost ?? "localhost")
        }
        // Clear the endpoint, keep the token: the bearer token is the USER's
        // credential for a server that is still there, and disconnecting a
        // tunnel is not a reason to throw it away. The registration is
        // cleared even when the cancel failed — a forward we could not remove
        // is the LAST thing to keep routing requests at.
        _ = try? repository.noteConnection(
            siteID: site.id, endpoint: nil, serverBuild: nil, now: now())
        if let closeError {
            var failure = self.envelope(
                invocation, site: site, state: .degraded,
                message: "cleared \(site.displayName)'s registered endpoint, but "
                    + closeError,
                changed: true)
            failure.retryAfterSeconds = 15
            return failure
        }
        return envelope(
            invocation, site: site, state: .ready,
            message: observation.localPort == nil
                ? "\(site.displayName) was not connected"
                : "closed the forward and cleared \(site.displayName)'s registered "
                    + "endpoint (the Keychain token is untouched)",
            changed: observation.localPort != nil)
    }

    // MARK: plan / ensure

    private func planVerb(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let result = try await coordinator(invocation, site: site).plan(
            siteReference: site.id, target: invocation.target,
            permissions: invocation.permissions)
        return ClusterCLIEnvelope.lifecycle(verb: invocation.verb, result: result)
    }

    private func ensureVerb(
        _ invocation: ClusterCLIInvocation
    ) async throws -> ClusterCLIEnvelope {
        let site = try resolveSite(invocation)
        let result = try await coordinator(invocation, site: site).ensure(
            siteReference: site.id, target: invocation.target,
            permissions: invocation.permissions)
        return ClusterCLIEnvelope.lifecycle(verb: invocation.verb, result: result)
    }

    // MARK: Shared shaping

    /// The envelope for a single extracted operation. A skip is reported as
    /// such rather than as success (the operations layer keeps them distinct on
    /// purpose — a silent skip is how a step gets believed).
    private func operationEnvelope(
        _ invocation: ClusterCLIInvocation, site: ClusterSiteRecord,
        step: ClusterLifecycleStep, target: ClusterLifecycleTarget,
        outcome: ClusterOperationOutcome,
        successState: ClusterLifecycleState = .ready,
        /// What a non-success means. Almost always `.failed`; bootstrap apply
        /// passes `.pending` for a job that is queued rather than broken, so
        /// the envelope does not report a healthy in-flight job as an error.
        failureState: ClusterLifecycleState = .failed
    ) -> ClusterCLIEnvelope {
        var envelope = self.envelope(
            invocation, site: site,
            state: outcome.succeeded ? successState : failureState,
            message: outcome.message,
            changed: outcome.succeeded && !outcome.wasSkipped)
        envelope.step = step.rawValue
        envelope.target = target.cliName
        if !outcome.succeeded, failureState == .failed {
            envelope.error = .init(
                code: "\(step.rawValue)Failed", reason: outcome.message,
                repairAction: "inspect the failing layer with "
                    + "`cluster status --site \(site.id)`")
        }
        return envelope
    }
}

// MARK: - Redaction

extension ClusterCLIEnvelope {
    /// A shareable copy: usernames and home paths removed from every string
    /// field (plan §6.1). Structure and codes are untouched — a redacted report
    /// must still be diagnosable.
    func redacted() -> ClusterCLIEnvelope {
        var copy = self
        let redact = ClusterCLIRedaction.redact
        copy.message = redact(message)
        copy.siteName = siteName.map(redact)
        copy.endpoint = endpoint.map(redact)
        copy.serverBuild = serverBuild.map(redact)
        copy.logPath = logPath.map(redact)
        copy.outputPath = outputPath.map(redact)
        copy.command = command.map { $0.map(redact) }
        copy.blockers = blockers.map { $0.map(redact) }
        copy.layers = layers.map {
            $0.map { Layer(layer: $0.layer, state: redact($0.state)) }
        }
        copy.plan = plan.map {
            $0.map { step in
                PlanStep(
                    step: step.step, reason: redact(step.reason), gating: step.gating,
                    satisfied: step.satisfied,
                    requiredPermissionFlags: step.requiredPermissionFlags)
            }
        }
        copy.sites = sites.map {
            $0.map { site in
                Site(
                    id: site.id, name: redact(site.name),
                    transport: redact(site.transport), topology: site.topology,
                    scheduler: site.scheduler,
                    lastEndpoint: site.lastEndpoint.map(redact),
                    lastServerBuild: site.lastServerBuild.map(redact),
                    tokenAvailable: site.tokenAvailable)
            }
        }
        copy.nextAction = nextAction.map {
            NextAction(
                verb: $0.verb, requiresHuman: $0.requiresHuman,
                missingPermissionFlags: $0.missingPermissionFlags,
                detail: $0.detail.map(redact))
        }
        copy.error = error.map {
            Failure(
                code: $0.code, reason: redact($0.reason),
                repairAction: redact($0.repairAction))
        }
        copy.operations = operations.map {
            $0.map { operation in
                var updated = operation
                updated.repairAction = operation.repairAction.map(redact)
                return updated
            }
        }
        return copy
    }
}
