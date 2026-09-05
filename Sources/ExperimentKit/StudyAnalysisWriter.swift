import Foundation
import SteeringKit

/// Publishes only NEW run directories under the explicitly bound workspace.
struct StudyAnalysisWriter {
    enum Task: String {
        case analyze
        case rescoreStyle = "rescore-style"
    }

    let workspaceRoot: URL

    func write(_ artifacts: StudyAnalysisArtifacts, manifest: ExperimentManifest, task: Task)
        throws -> URL
    {
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "exp-\(manifest.name)-\(task.rawValue)",
            under: workspaceRoot.appending(component: "runs"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(manifest).write(
            to: directory.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        // Offline tasks neither render nor generate: no capability or sampling stamp.
        try RunMetadata.write(
            runType: task.rawValue, to: directory, modelID: manifest.modelID,
            revision: manifest.modelRevision, experiment: manifest.name,
            experimentHash: ExperimentStore.manifestHash(manifest))
        for (name, data) in artifacts.files.sorted(by: { $0.key < $1.key }) {
            try data.write(to: directory.appending(component: name), options: .atomic)
        }
        return directory
    }
}
