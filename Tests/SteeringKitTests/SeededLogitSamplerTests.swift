import MLX
import MLXLMCommon
import MLXNN
import SteeringKit
import Testing

struct SeededLogitSamplerTests {
    @Test func repeatedResumedAndInterleavedStreamsAgree() {
        let logits = MLXArray([Float(0), 0.3, -0.2, 0.7]).reshaped([1, 4])
        let parameters = GenerateParameters(temperature: 0.8)
        func draw(_ sampler: SeededLogitSampler) -> Int {
            sampler.sample(logits: logits).item(Int.self)
        }
        let uninterrupted = SeededLogitSampler(parameters: parameters, seed: UInt64.max)
        let expected = (0..<64).map { _ in draw(uninterrupted) }
        let resumed = SeededLogitSampler(parameters: parameters, seed: UInt64.max)
        let unrelated = SeededLogitSampler(parameters: parameters, seed: 5)
        var actual: [Int] = []
        for _ in 0..<64 {
            actual.append(draw(resumed))
            _ = draw(unrelated)
            _ = MLXRandom.uniform(0..<1, [17]).asArray(Float.self)
        }
        #expect(actual == expected)
        // Independent reference: the library's explicit key/state API.
        let reference = MLXRandom.RandomState(seed: UInt64.max)
        let direct = (0..<64).map { _ in
            MLXRandom.categorical(logits * (1 / MLXArray(Float(0.8))), key: reference).item(Int.self)
        }
        #expect(actual == direct)
    }

    @Test func filtersAndGreedyHaveIndependentExpectations() {
        let logits = MLXArray([Float(-10), 0, 10, -20]).reshaped([1, 4])
        for parameters in [GenerateParameters(temperature: 0),
            GenerateParameters(temperature: 0.7, topK: 1),
            GenerateParameters(temperature: 0.7, minP: 0.9),
            GenerateParameters(temperature: 0.7, topP: 0.1)] {
            let sampler = SeededLogitSampler(parameters: parameters, seed: 7)
            for _ in 0..<8 {
                let token = sampler.sample(logits: logits).item(Int.self)
                #expect(token == 2, "token=\(token) temp=\(parameters.temperature) p=\(parameters.topP) k=\(parameters.topK) min=\(parameters.minP)")
            }
        }
    }
}
