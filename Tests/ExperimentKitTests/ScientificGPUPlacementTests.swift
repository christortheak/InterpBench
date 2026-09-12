import Foundation
import Testing
@testable import ExperimentKit

@Suite struct ScientificGPUPlacementTests {
    @Test func controllerVocabularyAndMemoryAreNotWorkloadFit() throws {
        let options = try JSONDecoder().decode(ScientificGPUPlacement.self, from: Data(#"{"available":true,"gpuTypes":["A100","H100"],"defaultGPUType":"A100","gpuVRAMGB":{"A100":40,"H100":80}}"#.utf8))
        #expect(options.defaultLabel == "Site default — A100")
        #expect(options.capacityLabel("").contains("40 GB"))
        #expect(options.capacityLabel("H100").contains("80 GB"))
        #expect(options.capacityLabel("H100").contains("not a peak-memory estimate"))
        #expect(options.capacityLabel("other").contains("not declared"))
    }

    @Test func placementStaysInSubmissionBodyAndCPUOperationsHideIt() throws {
        let body = ScientificGPUPlacement.roundBody(hash: "review", gpuType: "A100", overrides: [0:"", 1:"H100"])
        #expect(body == ["planSHA256":.string("review"), "confirmAction":.bool(true), "gpuType":.string("A100"), "shardGPUTypes":.object(["1":.string("H100")])])
        #expect(ScientificGPUPlacement.roundBody().isEmpty)
        for operation in ["jlens-fit-round", "jlens-fit-merge", "rescore-style"] {
            #expect(!ScientificGPUPlacement.usesGPU(.object(["operation":.string(operation)])))
        }
        #expect(ScientificGPUPlacement.usesGPU(.object(["operation":.string("jlens-fit")])))
    }

    @Test func reviewShowsActualShardPlacementAndPilotLimitations() throws {
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(#"{"submitIndices":[1],"shards":[{"index":0,"status":"uncertain","gpuType":"A100"},{"index":1,"status":"pending","gpuType":"H100"}],"gpuReview":{"throughputScope":"Cross-hardware extrapolation","pilotHardware":{"deviceName":"measured GPU"}}}"#.utf8))
        #expect(ScientificGPUPlacement.submitIndices(value) == [1])
        let lines = ScientificGPUPlacement.reviewLines(value)
        #expect(lines.contains("Shard 0: uncertain — A100"))
        #expect(lines.contains("Shard 1: pending — H100"))
        #expect(lines.contains("Throughput pilot GPU: measured GPU"))
        #expect(lines.contains("Cross-hardware extrapolation"))
    }
}
