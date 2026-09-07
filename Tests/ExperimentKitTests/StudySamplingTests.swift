import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

struct StudySamplingTests {
    @Test func savedAgentThinkingRemainsIndependentOfTheBaselineEffort() {
        var manifest = ExperimentManifest(name: "sampling", description: "", modelID: "test/model")
        manifest.reasoningEffort = "off"
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: nil) == .off)
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: true) == .xhigh)
        manifest.reasoningEffort = "on"
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: false) == .off)
        #expect(StudySampling.reasoningEffort(manifest, variantThinking: nil) == .on)
    }

    @Test func identitiesMatchThePythonContractAndPreserveLiteralSeeds() {
        #expect(StudySampling.deriveSeed(experimentHash: "abc", condition: "baseline",
            promptID: "item", sampleIndex: 0) == 4727079337508148899)
        #expect(StudySampling.deriveSeed(experimentHash: "abc", condition: "",
            promptID: "turn", sampleIndex: 3) == 5849915954937287103)
        #expect(StudySampling.deriveSeed(experimentHash: "é", condition: "条件",
            promptID: "項目🧪", sampleIndex: 3) == 4540562418375410840)
        var manifest = ExperimentManifest(name: "sampling", description: "", modelID: "test/model")
        manifest.temperature = 0.7
        manifest.seeds = [UInt64.max, 9007199254740993]
        #expect(StudySampling.policy(manifest) == "manifestSeeds")
        #expect(StudySampling.seed(manifest, experimentHash: "abc", condition: "baseline",
            promptID: "item", sampleIndex: 0) == UInt64.max)
        manifest.samplesPerItem = 3
        #expect(StudySampling.count(manifest) == 3)
        #expect(StudySampling.policy(manifest) == "derivedSHA256")
        let full = (0..<3).map { StudySampling.seed(manifest, experimentHash: "abc",
            condition: "baseline", promptID: "item", sampleIndex: $0) }
        #expect(full[0] == 4727079337508148899)
        // A resumed record is addressed by index, not a previously advanced RNG.
        #expect(full[2] == StudySampling.seed(manifest, experimentHash: "abc",
            condition: "baseline", promptID: "item", sampleIndex: 2))
        manifest.temperature = 0
        #expect(StudySampling.count(manifest) == 2)
        #expect(StudySampling.policy(manifest) == "manifestSeeds")
    }
}
