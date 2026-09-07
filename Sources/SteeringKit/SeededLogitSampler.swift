import MLX
import MLXLMCommon
import MLXNN

/// A record-local stream. Filter arithmetic/order matches the pinned
/// mlx-swift-lm TopPSampler; only the categorical draw receives an explicit key.
/// Equal seeds across MLX and PyTorch do not imply equal token draws.
public final class SeededLogitSampler: LogitSampler {
    private let parameters: GenerateParameters
    private let state: MLXRandom.RandomState

    public init(parameters: GenerateParameters, seed: UInt64) {
        self.parameters = parameters
        self.state = MLXRandom.RandomState(seed: seed)
    }

    public func sample(logits: MLXArray) -> MLXArray {
        guard parameters.temperature != 0 else { return argMax(logits, axis: -1) }
        let filtered = (parameters.topP > 0 && parameters.topP < 1)
            || parameters.topK > 0 || parameters.minP > 0
        var values = logits
        if filtered {
            if values.dtype == .bfloat16 { values = values.asType(.float32) }
            values = logSoftmax(values)
            let negativeInfinity = MLXArray(-Float.infinity)
            if parameters.topP > 0 && parameters.topP < 1 {
                let indices = argSort(values, axis: -1)
                let sorted = takeAlong(values, indices, axis: -1)
                let keep = cumsum(exp(sorted), axis: -1) .> (1 - parameters.topP)
                values = putAlong(values, indices,
                    values: MLX.where(keep, sorted, negativeInfinity), axis: -1)
            }
            if parameters.minP > 0 {
                let threshold = values.max(axis: -1, keepDims: true)
                    + log(MLXArray(parameters.minP))
                values = MLX.where(values .>= threshold, values, negativeInfinity)
            }
            if parameters.topK > 0 && parameters.topK < values.dim(-1) {
                let partition = argPartition(-values, kth: parameters.topK - 1, axis: -1)
                let indices = partition[0..., parameters.topK...]
                values = putAlong(values, indices, values: negativeInfinity, axis: -1)
            }
        }
        return MLXRandom.categorical(
            values * (1 / MLXArray(parameters.temperature)), key: state)
    }
}
