import Foundation
import MLX
import Testing
@testable import SteeringKit

@Suite struct ResidualRuntimeTests {
    final class Events: @unchecked Sendable {
        var phases: [String] = []
        var values: [[Float]] = []
        var counts: [Int] = []
    }

    @Test func phasesAndLegacyArithmeticArePreserved() throws {
        let events = Events()
        let runtime = try ResidualRuntime(readings: [
            .init(id: "after", site: .post(0), stage: .postAction) { h, c, _ in
                events.phases.append("after"); events.values.append(h.asArray(Float.self))
                #expect(c.positions == 2..<4)
            },
            .init(id: "before", site: .post(0), stage: .preAction) { h, _, _ in
                events.phases.append("before"); events.values.append(h.asArray(Float.self))
            },
        ], promptTokenCount: 4)
        let chain = try InterventionPlan.interventions([
            .init(layer: 0, vector: [1,0,0], strength: 0.5, mode: .ablate, concept: "axis"),
            .init(layer: 0, vector: [1,1,0], strength: 2, mode: .add, concept: "offset"),
            .init(layer: 0, vector: [0,0,1], strength: 0.125, mode: .add, concept: "other"),
        ], promptTokenCount: 4)
        let h = MLXArray([Float(4),3,2,8,5,6], [1,2,3])
        let out = runtime.apply(h, site: .post(0), offset: 2, interventions: chain)
        #expect(events.phases == ["before", "after"])
        #expect(out.asArray(Float.self) == [2,3,2,6,7,6.125])
        #expect(h.asArray(Float.self) == [4,3,2,8,5,6])
        #expect(ResidualRuntime.applyLegacy(h, interventions: chain, layer: 0, offset: 2).asArray(Float.self) == out.asArray(Float.self))
        let mid = ResidualRuntime.applyLegacy(h, interventions: chain, layer: 0, offset: 0)
        #expect(mid.asArray(Float.self) == [2,3,2,4,5,6])
    }

    @Test func providerStateIsSeparatePerBatchRowAndResponse() throws {
        let events = Events()
        let reading = ResidualRuntime.Reading(id: "counter", site: .pre(1), stage: .preAction) { _, c, state in
            let count = (state["count"]?.item(Int.self) ?? 0) + 1
            state["count"] = MLXArray(count)
            events.counts.append(count)
            #expect(c.identity["agent"] == "seat")
        }
        let h = MLXArray.ones([2,1,3])
        for _ in 0..<2 {
            let runtime = try ResidualRuntime(readings: [reading], identity: ["agent":"seat"])
            _ = runtime.apply(h, site: .pre(1), offset: 0)
            _ = runtime.apply(h, site: .pre(1), offset: 1)
            runtime.close()
            _ = runtime.apply(h, site: .pre(1), offset: 2)
        }
        #expect(events.counts == [1,1,2,2,1,1,2,2])
    }

    @Test func realModelBoundariesAndScopeCleanup() throws {
        let json = """
        {"hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,
        "num_attention_heads":2,"rms_norm_eps":1e-5,"vocab_size":32,
        "num_key_value_heads":1,"head_dim":8,"tie_word_embeddings":true}
        """
        let model = try SteeredQwen3Model(JSONDecoder().decode(SteeredQwen3Configuration.self, from: Data(json.utf8)))
        let input = MLXArray([Int32(1),2,3], [1,3])
        let baseline = model(input, cache: nil).asArray(Float.self)
        let events = Events()
        let runtime = try ResidualRuntime(readings: [
            .init(id: "input", site: .pre(0), stage: .preAction) { _, _, _ in events.phases.append("input") },
            .init(id: "output", site: .post(0), stage: .postAction) { _, _, _ in events.phases.append("output") },
        ])
        try model.withResidualRuntime(runtime) {
            #expect(throws: ResidualRuntime.ConfigurationError.self) {
                try model.withResidualRuntime(runtime) {}
            }
            #expect(model(input, cache: nil).asArray(Float.self) == baseline)
        }
        #expect(throws: ResidualRuntime.ConfigurationError.self) {
            try model.withResidualRuntime(runtime) {}
        }
        #expect(events.phases == ["input", "output"])
        #expect(model.residualRuntime == nil)
        enum Failure: Error { case requested }
        let other = try ResidualRuntime(readings: [])
        #expect(throws: Failure.self) {
            try model.withResidualRuntime(other) { throw Failure.requested }
        }
        #expect(model.residualRuntime == nil)
        let bad = try ResidualRuntime(readings: [.init(id: "bad", site: .post(9), stage: .postAction) { _,_,_ in }])
        #expect(throws: ResidualRuntime.ConfigurationError.self) { try model.withResidualRuntime(bad) {} }
        #expect(model.residualRuntime == nil)
    }
    @Test func tensorGradientAndCrossSiteStateArePreserved() throws {
        let events = Events()
        let callback: @Sendable (MLXArray, ResidualRuntime.Context, inout [String: MLXArray]) -> Void = { _, _, state in
            let count = (state["count"]?.item(Int.self) ?? 0) + 1
            state["count"] = MLXArray(count); events.counts.append(count)
        }
        let runtime = try ResidualRuntime(readings: [
            .init(id: "input", site: .pre(0), stage: .preAction, providerID: "shared", observe: callback),
            .init(id: "output", site: .post(0), stage: .postAction, providerID: "shared", observe: callback),
        ])
        let h = MLXArray([Float(4),3], [1,1,2])
        _ = runtime.apply(h, site: .pre(0), offset: 0)
        _ = runtime.apply(h, site: .post(0), offset: 0)
        #expect(events.counts == [1,2])
        let chain: [any LayerIntervention] = [SubspaceAblator(layers: [0], vector: [1,0], strength: 0.5),
                                             VectorInjector(layer: 0, vector: [0,1], alpha: 2)]
        let gradient = grad { value in
            ResidualRuntime.applyLegacy(value, interventions: chain, layer: 0, offset: 0).sum()
        }(h)
        #expect(gradient.asArray(Float.self) == [0.5,1])
    }

    @Test func gemmaBoundariesPreserveReadOnlyOutput() throws {
        let json = """
        {"model_type":"gemma3_text","hidden_size":16,"num_hidden_layers":4,
        "intermediate_size":32,"num_attention_heads":2,"head_dim":8,
        "vocab_size":32,"num_key_value_heads":1,"sliding_window":8,"sliding_window_pattern":2}
        """
        let model = SteeredGemma3TextModel(try JSONDecoder().decode(SteeredGemma3TextConfiguration.self, from: Data(json.utf8)))
        let h = MLXArray([Int32(1),2,3], [1,3]); let baseline = model(h, cache: nil).asArray(Float.self)
        let events = Events()
        let runtime = try ResidualRuntime(readings: [
            .init(id: "input", site: .pre(3), stage: .preAction) { _, _, _ in events.phases.append("input") },
            .init(id: "output", site: .post(3), stage: .postAction) { _, _, _ in events.phases.append("output") },
        ])
        try model.withResidualRuntime(runtime) { #expect(model(h, cache: nil).asArray(Float.self) == baseline) }
        #expect(events.phases == ["input", "output"])
    }

}
