import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import SteeringKit
import Testing
import Tokenizers
@testable import ExperimentKit

/// Opt-in live measurement. No downloads and no researcher workspace mutations.
struct MLXSamplingQualificationTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["STEERLAB_MLX_QUALIFICATION_DIR"] != nil))
    func cachedModelRecordReplay() async throws {
        let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["STEERLAB_MLX_QUALIFICATION_DIR"]))
        try #require(!FileManager.default.fileExists(atPath: output.path))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let modelID = "Qwen/Qwen3-4B-MLX-4bit"
        let revision = "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
        let snapshot = try #require(SteeredContainerLoader.cachedLoadableSnapshot(modelID: modelID, revision: revision))
        let container = try await SteeredModels.factory.loadContainer(
            from: snapshot, using: #huggingFaceTokenizerLoader())
        try await RunSamplingProvenance.write(container: container, modelID: modelID,
            revision: revision, to: output)
        var rows: [[String: Any]] = []
        for index in 0..<10 {
            let prompt = "Continue this list with one item: \(index), \(index + 1),"
            let seed = StudySampling.deriveSeed(experimentHash: "qualification-v1",
                condition: "baseline", promptID: "item-\(index)", sampleIndex: 0)
            func generate(_ seed: UInt64, reasoning: Bool = false) async throws -> ExperimentTasks.MeasuredGeneration {
                try await ExperimentTasks.generateMeasured(container, prompt: prompt, modelID: modelID,
                    maxTokens: 16, temperature: 0.7, seed: seed,
                    reasoningEffort: reasoning ? .on : .off,
                    reasoningMaxTokens: reasoning ? 8 : nil)
            }
            let first = try await generate(seed)
            _ = try await generate(17) // A different record cannot advance this one's stream.
            let replay = try await generate(seed)
            let budgeted = try await generate(seed, reasoning: true)
            let budgetReplay = try await generate(seed, reasoning: true)
            rows.append(["prompt": prompt, "seed": seed, "maxTokens": 16, "temperature": 0.7,
                "promptTokenCount": first.promptTokenCount ?? 0, "output": first.text, "replay": replay.text,
                "repeatable": first.text == replay.text && first.finishReason == replay.finishReason,
                "budgetOutput": budgeted.text, "budgetReplay": budgetReplay.text,
                "budgetRepeatable": budgeted.text == budgetReplay.text && budgeted.finishReason == budgetReplay.finishReason,
                "finishReason": first.finishReason, "budgetFinishReason": budgeted.finishReason])
        }
        // Measurement writes observed deviations; numerical nondeterminism is
        // evidence to inspect, not a hidden attempt to make the test pass.
        try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "rows": rows], options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(component: "repeatability.json"), options: .withoutOverwriting)
    }
}
