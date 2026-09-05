import Foundation
import SteeringKit

extension ExperimentPanel {
    public var pipelineRuns: [ClusterClient.PipelineRunSummary] { pipelines.pipelineRuns }
    public var localPipelineRuns: [ClusterClient.PipelineRunSummary] { pipelines.localPipelineRuns }

    @discardableResult
    func submitStudyRemotely(
        _ manifest: ExperimentManifest, verbOverride: String? = nil, followLog: Bool
    ) async -> Result<String, StudyBatchSubmission.Failure> {
        await submitCapturedStudyBundle(
            manifest, request: submission.snapshot(verb: verbOverride), followLog: followLog)
    }

    public func runExperimentVerbOnActiveServer(experimentName name: String, verb: String) async {
        guard isServerWorkspace else {
            note(
                "no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        cluster?.loadStoredToken()
        guard let client = cluster?.client else {
            note("invalid server URL", severity: .error)
            return
        }
        let context = remoteContextIdentity
        await serverExecution.run(
            experimentName: name, verb: verb,
            substrate: cluster?.substrateLabel ?? "server",
            transport: StudyServerJobTransport(client: client, jobs: remoteJobs),
            isCurrent: { [weak self] in self?.remoteContextIdentity == context })
    }

    public func refreshPipelineRuns() async {
        let context = remoteContextIdentity
        let name = selectedName
        await pipelines.refresh(
            name: name, client: isServerWorkspace ? cluster?.client : nil,
            isCurrent: { [weak self] in
                guard let self else { return false }
                return self.selectedName == name && self.remoteContextIdentity == context
            })
    }

    public func savePipelineDeclaration(_ draft: PipelineDraft?) {
        pipelines.saveDeclaration(draft, manifest: selected)
    }

    /// Capture the intended workspace and options before presenting a GPU warning.
    public func pipelineSubmissionAction(
        manifest: ExperimentManifest, request: StudySubmissionRequest
    )
        -> @MainActor () async -> Void
    {
        let context = remoteContextIdentity
        let manifestData = ExperimentStore.manifestData(name: manifest.name)
        return { [weak self] in
            guard let self else { return }
            guard self.remoteContextIdentity == context,
                ExperimentStore.manifestData(name: manifest.name) == manifestData
            else {
                self.note(
                    "pipeline submission stopped because the workspace, server or study changed; review it and submit again",
                    severity: .warning)
                return
            }
            self.remoteVerb = "pipeline"
            _ = await self.submitCapturedStudyBundle(
                manifest, request: request.replacingVerb("pipeline"), followLog: true)
        }
    }

    private func submitCapturedStudyBundle(
        _ manifest: ExperimentManifest,
        request: StudySubmissionRequest, followLog: Bool
    ) async -> Result<String, StudyBatchSubmission.Failure> {
        cluster?.loadStoredToken()
        guard let client = cluster?.client else {
            remoteStatus = "invalid server URL"
            let refusal =
                "remote submit refused: no server connection — connect a "
                + "server in the substrate selector first"
            note(refusal, severity: .error)
            return .failure(.init(reason: refusal))
        }
        let context = remoteContextIdentity
        let isCurrent: @MainActor () -> Bool = { [weak self] in
            self?.remoteContextIdentity == context
        }
        let hasDisplay = host != nil
        var follower: (@MainActor (String, String, String, Bool) -> Void)?
        if followLog {
            follower = {
                [weak self] id, verb, study, dryRun in
                guard let self else { return }
                self.remoteJobs.follow { [weak self] in
                    await self?.serverExecution.followBundle(
                        jobID: id, verb: verb, study: study,
                        dryRun: dryRun, client: client, hasDisplay: hasDisplay, isCurrent: isCurrent
                    )
                }
            }
        }
        return await bundleSubmission.submit(
            manifest, request: request,
            capabilities: cluster?.capabilities, substrate: cluster?.substrateLabel,
            transport: StudyBundleTransport(client: client), isCurrent: isCurrent, follow: follower)
    }

    var remoteContextIdentity: StudyRemoteContextIdentity {
        StudyRemoteContextIdentity(
            root: VectorCatalog.projectRoot.standardizedFileURL,
            serverURL: cluster?.serverURL, isServer: isServerWorkspace,
            pairing: cluster?.activeServerPairing)
    }
}

struct StudyRemoteContextIdentity: Equatable {
    let root: URL
    let serverURL: String?
    let isServer: Bool
    let pairing: WorkspaceScoping.ServerPairing?
}
