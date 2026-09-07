import MLXLMCommon
import SteeringKit

/// Keep iterator creation in one owner for chunk and reasoning-budget streams.
/// One sampler lives for the entire iterator, including the reasoning/answer
/// transition; starting another record constructs another independent stream.
enum SeededGeneration {
    static func iterator(
        input: LMInput, context: ModelContext, parameters: GenerateParameters,
        seed: UInt64?
    ) throws -> TokenIterator {
        guard let seed else {
            return try TokenIterator(input: input, model: context.model, parameters: parameters)
        }
        return try TokenIterator(input: input, model: context.model,
            cache: context.model.newCache(parameters: parameters),
            processor: parameters.processor(),
            sampler: SeededLogitSampler(parameters: parameters, seed: seed),
            prefillStepSize: parameters.prefillStepSize, maxTokens: parameters.maxTokens)
    }

    static func chunks(
        container: ModelContainer, input: consuming sending LMInput,
        parameters: GenerateParameters, seed: UInt64?
    ) async throws -> AsyncStream<Generation> {
        try await container.perform(nonSendable: input) { context, input in
            let iterator = try Self.iterator(
                input: input, context: context, parameters: parameters, seed: seed)
            return MLXLMCommon.generateTask(
                promptTokenCount: input.text.tokens.size,
                modelConfiguration: context.configuration, tokenizer: context.tokenizer,
                iterator: iterator).0
        }
    }
}
