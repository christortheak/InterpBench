import Foundation
import SteeringKit

/// Server-resident execution and post-submission following, over captured connections.
@MainActor
public final class StudyServerJobCoordinator {
    private let jobs: StudyRemoteJobController
    var presentation = StudyServerJobPresentation()
    init(jobs: StudyRemoteJobController) { self.jobs = jobs }
    private func note(_ text: String, severity: PanelNotice.Severity) {
        jobs.presentation.note(text, severity)
    }

    public func run(
        experimentName name: String, verb: String, in environment: StudyOperationEnvironment
    ) async {
        guard let context = environment.current(), context.isServer else {
            note("no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        guard environment.isCurrent(context) else { return }
        guard let client = environment.connect() else {
            note("invalid server URL", severity: .error)
            return
        }
        guard environment.isCurrent(context), client.profile == context.client?.profile else {
            note("server run stopped because the workspace or server changed; review it and try again", severity: .warning)
            return
        }
        await run(
            experimentName: name, verb: verb, substrate: context.substrate,
            transport: StudyServerJobTransport(client: client, jobs: jobs, workspaceRoot: context.workspaceRoot),
            isCurrent: { environment.isCurrent(context) })
    }

    func run(
        experimentName name: String, verb: String,
        substrate: String, transport: StudyServerJobTransport,
        isCurrent: @escaping @MainActor () -> Bool = { true }
    ) async {
        do {
            // Preflight: direct experiment verbs run SERVER-RESIDENT studies
            // only. On a workspace-paired server this always passes; on an
            // unpaired (remote) server it refuses with the portable path
            // named, instead of a confusing job-side missing-file failure.
            if let names = try? await transport.experimentNames() {
                let resident = names.contains(name)
                // The cached residency answer belongs to the Studies
                // selection; an Optimizations-initiated verb for another study must
                // not overwrite it.
                if isCurrent() { presentation.residency(name, resident) }
                if !resident {
                    note(
                        "study '\(name)' is not in \(substrate)'s workspace — "
                            + "direct runs execute the server-resident copy only. Pair "
                            + "the server to this workspace (serve --root <workspace>) "
                            + "or use Submit Bundle, the portable path for remote engines",
                        severity: .info)
                    return
                }
            }
            guard isCurrent(), !Task.isCancelled else { return }
            note("submitting \(verb) for '\(name)' to \(substrate)…", severity: .info)
            let jobID = try await transport.submit(name, verb)
            jobs.recordOrigin(transport.origin, jobID: jobID)
            jobs.remoteJobID = jobID
            jobs.activeServerJob = StudyRemoteJobController.ActiveServerJob(
                id: jobID, verb: verb, study: name)
            // Sweep jobs additionally occupy their own slot so Optimizations'
            // Cancel targets exactly this job, never another flow's.
            if verb == "sweep" { jobs.activeSweepJob = jobs.activeServerJob }
            jobs.noteRecentServerJob(id: jobID, verb: verb, study: name, state: "pending")
            note(
                "server \(verb) job \(jobID) submitted for '\(name)' on \(substrate)",
                severity: .info)
            let job = await transport.follow(
                jobID, "Server study \(verb) — \(name) [job \(jobID)]", "study \(verb)", false)
            if let job {
                // Terminal: the cancel affordance retires. A timed-out follow
                // (job == nil) deliberately KEEPS jobs.activeServerJob set — the
                // job is still running server-side and must stay cancellable.
                if jobs.activeServerJob?.id == jobID { jobs.activeServerJob = nil }
                if verb == "sweep", jobs.activeSweepJob?.id == jobID { jobs.activeSweepJob = nil }
                jobs.noteRecentServerJob(id: jobID, verb: verb, study: name, state: job.status)
                if let result = job.result,
                    let directory = Self.findString(
                        in: .object(result), keyPath: ["runDirectory"])
                {
                    jobs.lastServerRunDirectory = directory
                }
                if let error = job.error, !error.isEmpty {
                    note("server \(verb) job \(jobID) \(job.status): \(error)", severity: .error)
                } else {
                    note(
                        "server \(verb) job \(jobID) \(job.status)"
                            + (jobs.lastServerRunDirectory.map { " → \($0)" } ?? ""),
                        severity: .info)
                }
            } else {
                note(
                    "server \(verb) job \(jobID) still running on \(substrate) — "
                        + "reconnect from Compute or the recent-jobs list", severity: .info)
            }
            if isCurrent() { await presentation.refreshRuns() }
            await transport.refreshRecentJobs()
        } catch {
            note(
                "Couldn't submit the \(verb) job to the server — nothing is "
                    + "running on the server; check the connection in Compute "
                    + "and submit again. Details: \(error)",
                severity: .error)
        }
    }

    func followBundle(
        jobID: String, verb: String, study: String, dryRun: Bool,
        client: ClusterClient, hasDisplay: Bool,
        isCurrent: @escaping @MainActor () -> Bool
    ) async {
        guard hasDisplay else {
            // No display host (headless/test) — fall back to the plain
            // remote-log stream so the lines still land somewhere.
            await jobs.streamRemoteJobLog(jobID: jobID, client: client)
            return
        }
        let label = dryRun ? "\(verb) (dry run)" : verb
        let job = await jobs.followServerJobInDisplay(
            jobID: jobID,
            client: client,
            title: "Bundled study \(label) — \(study) [job \(jobID)]",
            label: "bundle \(label)",
            mirrorToRemoteLog: true)
        guard let job else { return }
        jobs.noteRecentServerJob(
            id: jobID, verb: "\(verb) (bundle)", study: study, state: job.status)
        if job.status == "succeeded", isCurrent(), !Task.isCancelled {
            // Executing a bundle imports the study into the server's tree —
            // the residency preflight may now say yes; drop the cache and
            // re-check so Run Server Copy re-enables without reselecting.
            await presentation.refreshResidency()
            // Evidence comes home (Mac-authority mode, 2026-07-21): a
            // bundled validate exists to mint freeze evidence for THIS
            // workspace — import its evidence bundle now and recompute
            // freeze readiness, instead of waiting for the auto-import
            // poll. Hash-verified by the same importer either way; a
            // failure surfaces in the status line and the run stays
            // importable manually.
            if verb == "validate", isCurrent() {
                await presentation.importEvidence(jobID)
                if isCurrent() { presentation.refresh() }
            }
        }
        if job.status == "prepared" {
            jobs.remoteStatus =
                "job \(jobID) prepared — dry run staged the bundle; "
                + "nothing executed"
        } else if let error = job.error, !error.isEmpty {
            jobs.remoteStatus = "job \(jobID) \(job.status): \(error)"
        } else {
            jobs.remoteStatus = "job \(jobID) \(job.status)"
        }
    }

    private static func findString(in value: JSONValue, keyPath: [String]) -> String? {
        guard let first = keyPath.first else {
            if case .string(let value) = value { return value }
            return nil
        }
        guard case .object(let object) = value, let child = object[first] else { return nil }
        return findString(in: child, keyPath: Array(keyPath.dropFirst()))
    }
}

@MainActor
struct StudyServerJobPresentation {
    var residency: (String, Bool) -> Void = { _, _ in }
    var refreshResidency: () async -> Void = {}
    var refreshRuns: () async -> Void = {}
    var importEvidence: (String) async -> Void = { _ in }
    var refresh: () -> Void = {}
}

@MainActor
struct StudyServerJobTransport {
    var origin: RemoteJobOrigin? = nil
    var experimentNames: () async throws -> [String]
    var submit: (String, String) async throws -> String
    var follow: (String, String, String, Bool) async -> RemoteJobRecord?
    var refreshRecentJobs: () async -> Void

    init(
        experimentNames: @escaping () async throws -> [String],
        submit: @escaping (String, String) async throws -> String,
        follow: @escaping (String, String, String, Bool) async -> RemoteJobRecord?,
        refreshRecentJobs: @escaping () async -> Void = {}
    ) {
        self.experimentNames = experimentNames
        self.submit = submit
        self.follow = follow
        self.refreshRecentJobs = refreshRecentJobs
    }

    init(client: ClusterClient, jobs: StudyRemoteJobController, workspaceRoot: URL) {
        origin = RemoteJobOrigin(connection: client.profile, workspaceRoot: workspaceRoot)
        experimentNames = { try await client.experimentNames() }
        submit = { try await client.submitExperimentJob(experiment: $0, verb: $1) }
        follow = { id, title, label, mirror in
            await jobs.followServerJobInDisplay(
                jobID: id, client: client, title: title,
                label: label, mirrorToRemoteLog: mirror)
        }
        refreshRecentJobs = { await jobs.refreshRecentServerJobs(client: client) }
    }
}
