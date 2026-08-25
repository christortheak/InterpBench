import Foundation

// =============================================================================
// The headless coordinator (CLUSTER-CLI-LIFECYCLE-PLAN §7.1).
//
// Composes the site repository, the inspector, the pure planner, the extracted
// provisioning operations, the tunnel controller, and the durable operation
// store into ONE idempotent entry point:
//
//     ensure(site:target:permissions:) -> ClusterLifecycleResult
//
// Contracts held here (all tested):
//   * inspect first, then execute ONLY permitted transitions;
//   * stop cleanly at the human-authentication and approval boundaries with a
//     typed state, never a half-done mutation;
//   * a queued scheduler job is `pending` forever if need be — repeating the
//     command polls it, and it never becomes a timeout failure;
//   * repeating at ready changes nothing (`changed: false`);
//   * existing scheduler jobs, forwards, and registrations are ADOPTED, never
//     re-created;
//   * a second process observes the in-flight operation instead of starting a
//     duplicate (the per-site lock);
//   * no credential enters a record, a result, or a log line.
//
// It is an actor: one site's lifecycle is a sequence, and the shared mutable
// state (the current operation record) must not be raced inside a process
// either.
// =============================================================================

public actor ClusterLifecycleCoordinator {

    private let repository: ClusterSiteRepository
    private let operationStore: ClusterOperationStore
    private let operations: ClusterProvisioningOperations
    private let inspector: ClusterLifecycleInspector
    private let tunnel: any ClusterTunnelControlling
    private let endpointProbe: any ClusterEndpointProbe
    private let authenticationLauncher: any ClusterAuthenticationLauncher
    private let secrets: any ClusterSecretStore
    private let configuration: ClusterProvisioningConfiguration
    private let now: @Sendable () -> Date

    public init(
        repository: ClusterSiteRepository,
        operationStore: ClusterOperationStore,
        operations: ClusterProvisioningOperations,
        inspector: ClusterLifecycleInspector,
        tunnel: any ClusterTunnelControlling,
        endpointProbe: any ClusterEndpointProbe = HTTPClusterEndpointProbe(),
        authenticationLauncher: any ClusterAuthenticationLauncher =
            TerminalAuthenticationLauncher(),
        secrets: any ClusterSecretStore = KeychainClusterSecretStore(),
        configuration: ClusterProvisioningConfiguration =
            ClusterProvisioningConfiguration(),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.repository = repository
        self.operationStore = operationStore
        self.operations = operations
        self.inspector = inspector
        self.tunnel = tunnel
        self.endpointProbe = endpointProbe
        self.authenticationLauncher = authenticationLauncher
        self.secrets = secrets
        self.configuration = configuration
        self.now = now
    }

    // MARK: Inspection

    /// Read-only: observe every layer and report it. Never mutates, never
    /// needs a permission, never takes the site lock.
    public func status(siteReference: String) async throws -> ClusterObservedState {
        let site = try resolve(siteReference)
        return await observe(site: site)
    }

    /// Observation + planning, with nothing executed (`cluster plan`).
    public func plan(
        siteReference: String,
        target: ClusterLifecycleTarget = .connected,
        permissions: ClusterLifecyclePermissions = []
    ) async throws -> ClusterLifecycleResult {
        let site = try resolve(siteReference)
        let observed = await observe(site: site)
        let plan = ClusterLifecyclePlanner.plan(
            site: site.profile, observed: observed, target: target,
            permissions: permissions)
        return result(
            operationID: ClusterOperationStore.newOperationID(now: now()),
            site: site, target: target, observed: observed, plan: plan,
            permissions: permissions, changed: false)
    }

    // MARK: Ensure

    /// Execute the permitted missing transitions, in ladder order, stopping at
    /// the first boundary the caller has not authorized.
    public func ensure(
        siteReference: String,
        target: ClusterLifecycleTarget = .connected,
        permissions: ClusterLifecyclePermissions = []
    ) async throws -> ClusterLifecycleResult {
        let site = try resolve(siteReference)

        // A second caller OBSERVES; it never starts a duplicate.
        guard let lock = try operationStore.acquireLock(siteID: site.id) else {
            let active = operationStore.activeRecord(forSite: site.id)
            throw ClusterLifecycleError.operationInProgress(
                siteID: site.id,
                operationID: active?.operationID ?? "(unrecorded)")
        }
        defer { lock.release() }

        var record = ClusterOperationRecord(
            operationID: ClusterOperationStore.newOperationID(now: now()),
            siteID: site.id,
            siteProfileHash: site.profileHash,
            target: target,
            authorizedMutations: permissions.identifiers,
            state: .running,
            startedAt: now(),
            updatedAt: now())
        try? operationStore.save(record)

        var currentSite = site
        var changed = false
        var lastStep: ClusterLifecycleStep?

        // Bounded: each pass must make progress, and every transition is
        // executed at most once per invocation. Without the bound a
        // mis-observed layer could loop forever re-running a mutation.
        var executed: Set<ClusterLifecycleStep> = []
        var observed = await observe(site: currentSite)
        var plan = ClusterLifecyclePlanner.plan(
            site: currentSite.profile, observed: observed, target: target,
            permissions: permissions)

        while let next = plan.next {
            // Boundaries: stop cleanly, with the state that names what the
            // caller must do.
            if next.gating.requiresHumanAuthentication {
                if permissions.contains(.openAuthTerminal) {
                    let opened = await authenticationLauncher
                        .openAuthenticationTerminal(for: currentSite.profile)
                    record.note(
                        step: .authenticate,
                        status: opened ? .running : .failed,
                        message: opened
                            ? "opened a visible Terminal for authentication"
                            : "could not open Terminal — run the printed command yourself",
                        now: now())
                    changed = changed || opened
                }
                lastStep = next.step
                break
            }
            if !next.isPermitted(by: permissions) { lastStep = next.step; break }
            if next.step == .controllerWait { lastStep = next.step; break }
            guard executed.insert(next.step).inserted else {
                // The transition ran and the layer still is not satisfied:
                // report the honest state rather than looping.
                lastStep = next.step
                break
            }

            let outcome = await execute(
                next.step, site: currentSite, observed: observed, record: &record)
            changed = changed || outcome.changed
            lastStep = next.step
            if let updated = outcome.site { currentSite = updated }
            try? operationStore.save(record)
            if !outcome.succeeded { break }

            observed = await observe(site: currentSite)
            plan = ClusterLifecyclePlanner.plan(
                site: currentSite.profile, observed: observed, target: target,
                permissions: permissions)
        }

        let final = result(
            operationID: record.operationID, site: currentSite, target: target,
            observed: observed, plan: plan, permissions: permissions,
            changed: changed, step: lastStep)
        record.state = final.state
        record.controllerJobID = observed.controller.jobID ?? record.controllerJobID
        record.lastDaemonHost = observed.daemonHost.host
        record.lastServerBuild = observed.serverHTTP.serverBuild
        record.localPort = observed.tunnel.localPort
        record.tokenAvailable = observed.bearerToken.isAvailable
        record.tokenSource = observed.bearerToken.isAvailable ? "keychain" : nil
        record.updatedAt = now()
        if final.state == .failed || final.state == .blocked {
            record.failureCode = final.state.rawValue
            record.repairAction = final.nextAction?.detail ?? final.message
        }
        try? operationStore.save(record)
        return final
    }

    // MARK: Transition execution

    private struct ExecutionOutcome {
        var succeeded: Bool
        var changed: Bool
        var site: ClusterSiteRecord?
    }

    private func execute(
        _ step: ClusterLifecycleStep,
        site: ClusterSiteRecord,
        observed: ClusterObservedState,
        record: inout ClusterOperationRecord
    ) async -> ExecutionOutcome {
        switch step {
        case .authenticate, .controllerWait:
            // Never executed: both are boundaries handled by the caller loop.
            return ExecutionOutcome(succeeded: false, changed: false, site: nil)

        case .pushCode:
            record.note(step: step, status: .running, now: now())
            let outcome = await operations.push(
                site: site.profile, configuration: configuration)
            // The deploy INTENT, recorded per machine the moment it becomes
            // true: the next observation compares the deployed engine against
            // this rather than against whatever binary is running.
            if outcome.succeeded, !outcome.wasSkipped,
                let deployed = outcome.deployed, !deployed.isEmpty
            {
                _ = try? repository.runtime.recordPush(
                    siteID: site.id, payloadRevision: deployed.payloadRevision,
                    buildStamp: deployed.buildStamp, at: now())
            }
            record.note(
                step: step, status: outcome.succeeded ? .succeeded : .failed,
                message: outcome.message, now: now())
            record.transcript += outcome.transcript
            return ExecutionOutcome(
                succeeded: outcome.succeeded, changed: outcome.succeeded, site: nil)

        case .bootstrapPlan:
            record.note(step: step, status: .running, now: now())
            let outcome = await operations.bootstrapPlan(
                site: site.profile, configuration: configuration)
            // The reviewed plan's hash is DURABLE: the review must survive
            // this process exiting.
            record.bootstrapPlanHash = outcome.planHash
            reviewedBootstrapPlanHash = outcome.planHash
            record.note(
                step: step, status: outcome.outcome.succeeded ? .succeeded : .failed,
                message: outcome.outcome.message, now: now())
            record.transcript += outcome.outcome.transcript
            // Rendering a plan changes nothing remotely.
            return ExecutionOutcome(
                succeeded: outcome.outcome.succeeded, changed: false, site: nil)

        case .bootstrapApply:
            record.note(step: step, status: .running, now: now())
            // Prefer this invocation's own plan hash; fall back to the newest
            // durable one so a review from an earlier process still counts.
            let reviewed = record.bootstrapPlanHash
                ?? reviewedBootstrapPlanHash
                ?? operationStore.lastBootstrapPlanHash(forSite: site.id)
            do {
                // The journal makes a submitted bootstrap job durable BEFORE
                // it is polled, so an interrupted ensure resumes that job
                // instead of queueing a second one.
                let outcome = try await operations.bootstrapApply(
                    site: site.profile, configuration: configuration,
                    reviewedPlanHash: reviewed,
                    journal: operationStore, siteID: site.id)
                record.bootstrapJobID = outcome.jobID ?? record.bootstrapJobID
                record.bootstrapStatusFile =
                    outcome.statusFile ?? record.bootstrapStatusFile
                record.note(
                    step: step,
                    status: outcome.outcome.succeeded
                        ? .succeeded
                        : (outcome.stillPending ? .awaitingScheduler : .failed),
                    message: outcome.outcome.message, now: now())
                record.transcript += outcome.outcome.transcript
                return ExecutionOutcome(
                    succeeded: outcome.outcome.succeeded, changed: true, site: nil)
            } catch {
                record.note(
                    step: step, status: .failed,
                    message: (error as? ClusterLifecycleError)?.errorDescription
                        ?? error.localizedDescription,
                    now: now())
                record.failureCode = (error as? ClusterLifecycleError)?.code
                return ExecutionOutcome(succeeded: false, changed: false, site: nil)
            }

        case .validate:
            record.note(step: step, status: .running, now: now())
            let bootstrapPrefix: String?
            if case .valid(_, let prefix) = observed.bootstrap {
                bootstrapPrefix = prefix
            } else {
                bootstrapPrefix = nil
            }
            let outcome = await operations.validate(
                site: site.profile,
                envFile: ClusterLifecycleInspector.envFile(for: configuration),
                prefix: bootstrapPrefix)
            recordedValidation = outcome.observation
            recordedValidationProfileHash = site.profileHash
            record.note(
                step: step, status: outcome.outcome.succeeded ? .succeeded : .failed,
                message: outcome.outcome.message, now: now())
            record.transcript += outcome.outcome.transcript
            return ExecutionOutcome(
                succeeded: outcome.outcome.succeeded, changed: false, site: nil)

        case .controllerStart:
            record.note(step: step, status: .running, now: now())
            let bootstrapPrefix: String?
            if case .valid(_, let prefix) = observed.bootstrap {
                bootstrapPrefix = prefix
            } else {
                bootstrapPrefix = nil
            }
            let outcome = await operations.controllerStart(
                site: site.profile, configuration: configuration,
                envFile: ClusterLifecycleInspector.envFile(for: configuration),
                prefix: bootstrapPrefix)
            if let jobID = outcome.jobID { record.controllerJobID = jobID }
            record.note(
                step: step,
                status: outcome.outcome.succeeded ? .awaitingScheduler : .failed,
                message: outcome.outcome.message, now: now())
            record.transcript += outcome.outcome.transcript
            return ExecutionOutcome(
                succeeded: outcome.outcome.succeeded, changed: true, site: nil)

        case .tunnelOpen:
            record.note(step: step, status: .running, now: now())
            guard let targetHost = tunnelTargetHost(site: site, observed: observed) else {
                record.note(
                    step: step, status: .failed,
                    message: "the controller has not published a usable node record "
                        + "yet — there is nothing to forward to",
                    now: now())
                return ExecutionOutcome(succeeded: false, changed: false, site: nil)
            }
            let outcome = await tunnel.open(
                site: site.profile, targetHost: targetHost,
                persistedPort: observed.tunnel.localPort
                    ?? ClusterLifecycleInspector.persistedPort(for: site))
            record.forwardIdentity = outcome.forwardIdentity
            record.localPort = outcome.observation.localPort
            let up: Bool
            if case .up = outcome.observation { up = true } else { up = false }
            record.note(
                step: step, status: up ? .succeeded : .failed,
                message: outcome.message, now: now())
            return ExecutionOutcome(
                succeeded: up, changed: outcome.changed, site: nil)

        case .registerConnection:
            record.note(step: step, status: .running, now: now())
            // 1. Import the token (straight into the secret store — the value
            //    never touches this method).
            var importNote = ""
            if site.profile.isSSHTransport, observed.bearerToken != .accepted {
                let outcome = await operations.importRemoteToken(
                    site: site.profile, tokenKey: site.tokenKey,
                    overwriteExisting: observed.bearerToken == .rejected)
                switch outcome {
                case .imported: importNote = "imported the server token into Keychain; "
                case .alreadyPresent: importNote = ""
                case .unavailable(let reason):
                    record.note(
                        step: step, status: .failed, message: reason, now: now())
                    return ExecutionOutcome(succeeded: false, changed: false, site: nil)
                }
            }
            // 2. Prove endpoint IDENTITY and authentication before registering
            //    — re-probed here rather than reused from the pre-import
            //    observation, since the token that just arrived is exactly
            //    what the endpoint was missing. An open port is not a
            //    connection (§7.11).
            guard let baseURL = ClusterLifecycleInspector.baseURL(
                for: site, tunnel: observed.tunnel)
            else {
                record.note(
                    step: step, status: .failed,
                    message: "there is no endpoint to verify yet — the tunnel is not up",
                    now: now())
                return ExecutionOutcome(succeeded: false, changed: false, site: nil)
            }
            let verification = await endpointProbe.probe(
                baseURL: baseURL, token: secrets.token(forKey: site.tokenKey))
            guard verification.reachable else {
                record.note(
                    step: step, status: .failed,
                    message: verification.authFailed
                        ? "the endpoint rejected the stored bearer token"
                        : "the endpoint did not prove its identity "
                            + "(\(verification.detail ?? "no answer")) — an open port "
                            + "is not a connection",
                    now: now())
                return ExecutionOutcome(succeeded: false, changed: false, site: nil)
            }
            let build = verification.serverBuild.flatMap { $0.isEmpty ? nil : $0 }
            // 3. Update THE ONE saved registration (never add a second).
            let endpoint = baseURL.absoluteString
            let updated = try? repository.noteConnection(
                siteID: site.id, endpoint: endpoint, serverBuild: build, now: now())
            record.lastServerBuild = build
            record.tokenAvailable = true
            record.tokenSource = "keychain"
            record.note(
                step: step, status: .succeeded,
                message: importNote + "registered \(endpoint) for \(site.displayName)",
                now: now())
            return ExecutionOutcome(succeeded: true, changed: true, site: updated)
        }
    }

    /// Where the forward should point: the published node for daemon-in-a-job
    /// (only when the record is CURRENT — a stale one is not a target), else
    /// the login host's own localhost.
    private func tunnelTargetHost(
        site: ClusterSiteRecord, observed: ClusterObservedState
    ) -> String? {
        guard site.profile.topology == .daemonInJob else { return "localhost" }
        if case .current(let host) = observed.daemonHost { return host }
        return nil
    }

    // MARK: Remembered validation

    /// The last profile-validate answer, and the site profile hash it was
    /// produced for. Validation is a remote command whose answer changes only
    /// when the environment or the profile does, so re-running it on every
    /// inspection would cost an ssh round trip per status call for no new
    /// information. Scoping it by the profile hash keeps it honest: edit the
    /// site and the answer stops counting.
    private var recordedValidation: ClusterProfileValidationObservation = .notRun
    private var recordedValidationProfileHash: String?

    /// The plan hash this coordinator reviewed in THIS process, preferred over
    /// the durable one so a fresh review inside one `ensure` is visible to the
    /// re-observation that follows it.
    private var reviewedBootstrapPlanHash: String?

    // MARK: Helpers

    private func resolve(_ reference: String) throws -> ClusterSiteRecord {
        guard let site = try repository.resolve(reference: reference) else {
            throw ClusterLifecycleError.unknownSite(reference)
        }
        return site
    }

    private func observe(site: ClusterSiteRecord) async -> ClusterObservedState {
        let duplicates = (try? repository.duplicateCount(
            forIdentity: site.canonicalIdentity)) ?? 1
        let validation = recordedValidationProfileHash == site.profileHash
            ? recordedValidation : .notRun
        return await inspector.observe(
            site: site,
            configuration: configuration,
            recordedControllerJobID: operationStore.lastControllerJobID(forSite: site.id),
            recordedValidation: validation,
            recordedBootstrapPlanHash: reviewedBootstrapPlanHash
                ?? operationStore.lastBootstrapPlanHash(forSite: site.id),
            // What THIS machine last deployed here (per-machine runtime cache).
            // Without it the payload check compares the deployed engine only
            // against the app bundle, which cannot tell a needed push from a
            // rollback — see `observePayload`.
            recordedDeployIntent: repository.deployIntent(forSite: site.id),
            registrationDuplicates: duplicates)
    }

    private func result(
        operationID: String,
        site: ClusterSiteRecord,
        target: ClusterLifecycleTarget,
        observed: ClusterObservedState,
        plan: ClusterLifecyclePlan,
        permissions: ClusterLifecyclePermissions,
        changed: Bool,
        step: ClusterLifecycleStep? = nil
    ) -> ClusterLifecycleResult {
        let state = plan.state(permissions: permissions)
        let endpoint = ClusterLifecycleInspector.baseURL(
            for: site, tunnel: observed.tunnel)?.absoluteString
        var message: String
        var nextAction: ClusterLifecycleResult.NextAction?
        var retryAfter: Int?

        switch state {
        case .ready:
            message = "\(site.displayName) is \(target.cliName)"
        case .blocked:
            message = plan.blockers.joined(separator: " · ")
        case .needsHumanAuthentication:
            message =
                "complete password and multi-factor authentication in the opened "
                + "Terminal window, then repeat this command"
            nextAction = .init(
                verb: "cluster ensure", requiresHuman: true,
                detail: authenticationLauncher.authenticationCommand(for: site.profile))
            retryAfter = 5
        case .needsApproval:
            let missing = plan.next?.gating.requiredPermission?.flags ?? []
            message = "\(plan.next?.reason ?? "a mutation is required") — re-run with "
                + missing.joined(separator: " ")
            nextAction = .init(
                verb: "cluster ensure", missingPermissionFlags: missing,
                detail: plan.next?.reason)
        case .pending:
            message = plan.next?.reason ?? "asynchronous work is still in flight"
            nextAction = .init(verb: "cluster ensure", detail: plan.next?.reason)
            retryAfter = 30
        case .degraded:
            message = plan.next?.reason ?? "a layer could not be observed"
            nextAction = .init(verb: "cluster status", detail: plan.next?.reason)
            retryAfter = 15
        case .planned, .running, .failed:
            message = plan.next?.reason ?? "work remains"
            nextAction = .init(verb: "cluster ensure", detail: plan.next?.reason)
        }
        // Every failure names the failing LAYER, not just the step.
        if state != .ready, let failing = observed.layerSummaries.first(where: {
            $0.layer == failingLayer(for: plan.next?.step)
        }) {
            message += " [\(failing.layer): \(failing.state)]"
        }

        return ClusterLifecycleResult(
            operationID: operationID,
            siteID: site.id,
            siteName: site.displayName,
            target: target,
            state: state,
            step: step ?? plan.next?.step,
            changed: changed,
            observedAt: observed.observedAt,
            message: message,
            retryAfterSeconds: retryAfter,
            nextAction: nextAction,
            endpoint: endpoint,
            tokenAvailable: observed.bearerToken.isAvailable,
            tokenSource: observed.bearerToken.isAvailable ? "keychain" : nil,
            serverBuild: observed.serverHTTP.serverBuild,
            schedulerJobID: observed.controller.jobID,
            schedulerState: observed.controller.schedulerState,
            observed: observed,
            plan: plan)
    }

    /// Step → the observation layer that explains it.
    private func failingLayer(for step: ClusterLifecycleStep?) -> String {
        switch step {
        case .authenticate: "controlMaster"
        case .pushCode: "payload"
        case .bootstrapPlan, .bootstrapApply: "bootstrap"
        case .validate: "profileValidation"
        case .controllerStart, .controllerWait: "controller"
        case .tunnelOpen: "tunnel"
        case .registerConnection: "serverHTTP"
        case nil: "siteConfiguration"
        }
    }
}
