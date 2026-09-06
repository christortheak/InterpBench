import Foundation
import SteeringKit

/// Review both files of a vector before attaching it. Digests describe the
/// reviewed bytes, outside the study document; the store owns scientific admission.
public enum StudyArtifactAuthoring {
    public struct Review: Codable, Sendable {
        public let workspaceRoot: String
        public let reference: String
        public let artifactSHA256: String
        public let sidecarSHA256: String
        public let sidecar: SteeringVectorSidecar
    }

    public static func inspect(_ reference: String, workspaceRoot: URL) throws -> Review {
        var path = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        for suffix in [".safetensors", ".json"] where path.hasSuffix(suffix) {
            path = String(path.dropLast(suffix.count))
        }
        guard !path.isEmpty, !(path as NSString).isAbsolutePath,
            !path.split(separator: "/").contains("..") else {
            throw ExperimentError.malformed("Supply a workspace-relative vector artifact path.",
                repair: "Inspect the extension-less path to an existing .safetensors/.json pair in this workspace.")
        }
        let root = try ManifestFileTransaction.canonicalPath(workspaceRoot)
        let urls = ["safetensors", "json"].map { workspaceRoot.appending(path: path).appendingPathExtension($0) }
        for url in urls {
            guard try ManifestFileTransaction.canonicalPath(url).hasPrefix(root + "/") else {
                throw ExperimentError.malformed("The vector artifact resolves outside this workspace.",
                    repair: "Use an artifact contained in the study's workspace.")
            }
        }
        let tensor = try ManifestFileTransaction.snapshot(at: urls[0])
        let sidecar = try ManifestFileTransaction.snapshot(at: urls[1])
        return Review(workspaceRoot: root, reference: path, artifactSHA256: tensor.sha256,
            sidecarSHA256: sidecar.sha256, sidecar: try JSONDecoder().decode(SteeringVectorSidecar.self, from: sidecar.data))
    }

    public static func attach(_ concept: String, artifact: Review, reviewed: DraftAuthoringSnapshot,
                              sourceConcept: String? = nil, evalRun: String? = nil) throws -> DraftAuthoringSnapshot {
        guard try ManifestFileTransaction.canonicalPath(reviewed.workspaceRoot) == artifact.workspaceRoot else { throw stale() }
        return try DraftAuthoringTransaction.perform(reviewed: reviewed) { name in
            let base = reviewed.workspaceRoot.appending(path: artifact.reference)
            // All study authoring takes the manifest lock first. Artifact files
            // are read-only here; no run or vector bytes are edited.
            return try ManifestFileTransaction.withLock(manifestURL: base.appendingPathExtension("json"), workspaceRoot: reviewed.workspaceRoot) {
                try ManifestFileTransaction.withLock(manifestURL: base.appendingPathExtension("safetensors"), workspaceRoot: reviewed.workspaceRoot) {
                    let current = try inspect(artifact.reference, workspaceRoot: reviewed.workspaceRoot)
                    guard current.artifactSHA256 == artifact.artifactSHA256,
                        current.sidecarSHA256 == artifact.sidecarSHA256 else { throw stale() }
                    _ = try ExperimentStore.attachArtifact(concept, artifact: current.reference,
                        sourceConcept: sourceConcept, evalRun: evalRun, experimentName: name)
                    return try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: name)
                }
            }
        }
    }

    static func requireDigests(_ artifact: Review, tensor: String, sidecar: String) throws {
        guard artifact.artifactSHA256 == tensor, artifact.sidecarSHA256 == sidecar else { throw stale() }
    }
    private static func stale() -> ExperimentError {
        .refusing(.staleManifest, "The reviewed vector bytes or workspace changed; nothing was attached.",
            repair: "Inspect the artifact and draft again, review the changes, then attach with both artifact digests and manifestFileSHA256.")
    }
}
