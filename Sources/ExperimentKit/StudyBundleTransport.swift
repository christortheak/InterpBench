import Foundation
import SteeringKit

/// Replaceable at the network/package boundary for deterministic coordination tests.
@MainActor
struct StudyBundleTransport {
    var origin: RemoteJobOrigin? = nil
    var frozenConflict: (ExperimentManifest) async -> String?
    var package: (ExperimentManifest) async throws -> URL
    var upload: (URL) async throws -> String
    var submit: (String, StudySubmissionRequest) async throws -> RemoteStudySubmission

    init(
        frozenConflict: @escaping (ExperimentManifest) async -> String?,
        package: @escaping (ExperimentManifest) async throws -> URL,
        upload: @escaping (URL) async throws -> String,
        submit: @escaping (String, StudySubmissionRequest) async throws -> RemoteStudySubmission
    ) {
        self.frozenConflict = frozenConflict
        self.package = package
        self.upload = upload
        self.submit = submit
    }

    init(client: ClusterClient, workspaceRoot: URL) {
        origin = RemoteJobOrigin(connection: client.profile, workspaceRoot: workspaceRoot)
        frozenConflict = { manifest in
            await client.frozenOnServerConflict(study: manifest.name, localStatus: manifest.status)
        }
        package = { manifest in
            guard ExperimentStore.workspaceRoot.standardizedFileURL == workspaceRoot.standardizedFileURL else {
                throw ChatServiceError(reason: "The workspace changed before packaging; submit from the intended workspace.")
            }
            let source = RunBundlePackager.captureSource(manifest)
            return try await Task.detached {
                try RunBundlePackager.packageExperiment(manifest, source: source)
            }.value
        }
        upload = { try await client.uploadBundle($0).path }
        submit = { path, request in
            try await client.submitBundle(
                path: path, verb: request.verb, executor: request.executor,
                dryRun: request.dryRun, resources: request.resources,
                resumePolicy: request.effectiveResumePolicy, parallelJobs: request.parallelJobs)
        }
    }
}
