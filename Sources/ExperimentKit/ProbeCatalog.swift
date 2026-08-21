import Foundation
import SteeringKit

public struct ProbeArtifactRecord: Identifiable, Sendable {
    public let url: URL
    public let artifact: ReadingProbeArtifact

    public var id: String { url.path }

    public var label: String {
        [
            artifact.concept,
            "L\(artifact.layer)",
            artifact.recipeName,
            String(artifact.createdAt.prefix(10)),
        ].joined(separator: " · ")
    }
}

public enum ProbeCatalog {
    public static func scan(runsDirectory: URL = VectorCatalog.runsDirectory) -> [ProbeArtifactRecord] {
        let fm = FileManager.default
        guard
            let runDirs = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return [] }

        var records: [ProbeArtifactRecord] = []
        for runDir in runDirs.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                let files = try? fm.contentsOfDirectory(
                    at: runDir, includingPropertiesForKeys: nil)
            else { continue }
            for file in files where file.lastPathComponent.hasSuffix(".probe.json") {
                guard
                    let data = try? Data(contentsOf: file),
                    let artifact = try? JSONDecoder().decode(
                        ReadingProbeArtifact.self, from: data)
                else { continue }
                records.append(ProbeArtifactRecord(url: file, artifact: artifact))
            }
        }
        return records
    }

    public static func save(
        _ artifact: ReadingProbeArtifact, to directory: URL, name: String
    ) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = directory.appending(component: "\(name).probe.json")
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return url
    }
}
