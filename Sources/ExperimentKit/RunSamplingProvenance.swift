import Foundation
import Metal
import MLXLMCommon
import SteeringKit

enum RunSamplingProvenance {
    static func write(container: ModelContainer, modelID: String, revision: String?,
                      to directory: URL) async throws {
        let dtypes = await container.perform { context in
            Array(Set(context.model.parameters().flattened().map { String(describing: $0.1.dtype) })).sorted()
        }
        let snapshot = SteeredContainerLoader.cachedLoadableSnapshot(modelID: modelID, revision: revision)
        let config = snapshot.flatMap { try? Data(contentsOf: $0.appending(component: "config.json")) }
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let value: [String: Any] = [
            "schemaVersion": 1, "rng": "mlx-random-key-split-categorical",
            "streamScope": "one record or turn; uninterrupted across reasoning and answer",
            "modelID": modelID, "modelRevision": snapshot?.lastPathComponent ?? NSNull(),
            "tokenizerRevision": snapshot?.lastPathComponent ?? NSNull(),
            "parameterDtypes": dtypes, "quantization": config?["quantization"] ?? NSNull(),
            "dependencies": SamplingDependencies.pins,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "gpu": MTLCreateSystemDefaultDevice()?.name ?? "unavailable",
            "prefillStepSize": 512, "topP": 1.0, "topK": 0, "minP": 0.0,
            "qualification": "Configuration provenance; not a repeatability certificate."
        ]
        let url = directory.appending(component: "sampling-provenance.json")
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }
}
