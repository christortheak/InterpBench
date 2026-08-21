import Foundation
import MLX
import MLXLMCommon
import SteeringKit

public struct ToyConceptConfig: Codable, Sendable {
    public var task: String?
    public var models: [SmokeTestConfig.ModelSpec]
    /// Directory containing {positive,negative}.jsonl, relative to cwd.
    public var conceptDirectory: String
    public var prompt: String
    public var maxTokens: Int
    /// Depth fractions to try (0.5 = middle block).
    public var layerFractions: [Float]
    public var alphas: [Float]
    public var seed: UInt64
}

public struct ToyConceptFailure: Error, CustomStringConvertible {
    public let model: String
    public let reason: String
    public var description: String { "[\(model)] \(reason)" }
}

/// Phase 0 exit: prove extraction → persistence → injection end to end with
/// a toy concept. A CAA "French" vector extracted from paired translations
/// must move generation toward French at some (layer, alpha), and must
/// behave differently from a matched-norm random vector (the smoke-test
/// assertion (b) that needed a real concept vector to exist).
public enum ToyConceptRun {

    public static func run(config: ToyConceptConfig) async throws {
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024

        let stimuli = try StimulusSet(
            directory: URL(filePath: config.conceptDirectory))
        print("stimulus set '\(stimuli.name)': \(stimuli.positive.count) pairs, "
            + "hash \(stimuli.hash.prefix(12))…")

        // One immutable run directory per invocation (CLAUDE.md › Data &
        // reproducibility). Never overwritten, never reused.
        let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "toy-\(stimuli.name)")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(config).write(
            to: runDirectory.appending(component: "config.json"))

        var generationsLog: [String] = []

        for spec in config.models {
            try await run(
                spec: spec, config: config, stimuli: stimuli,
                runDirectory: runDirectory, log: &generationsLog)
        }

        let logURL = runDirectory.appending(component: "generations.jsonl")
        try generationsLog.joined(separator: "\n").appending("\n")
            .write(to: logURL, atomically: true, encoding: .utf8)
        print("run artifacts: \(runDirectory.path)")
    }

    private static func run(
        spec: SmokeTestConfig.ModelSpec, config: ToyConceptConfig,
        stimuli: StimulusSet, runDirectory: URL, log: inout [String]
    ) async throws {
        print("\n=== \(spec.id) ===")
        let container = try await SteeredContainerLoader.load(modelID: spec.id)

        // Extract and persist the concept vector with full provenance.
        let extraction = try await ConceptExtractor.extract(
            container: container, stimuli: stimuli)
        let vectors = extraction.vectors
        let sidecar = SteeringVectorSidecar(
            modelID: spec.id,
            revision: SteeredContainerLoader.cachedRevision(for: spec.id),
            concept: stimuli.name,
            stimulusSetHash: stimuli.hash, vectors: vectors,
            options: extraction.options,
            residualNormPerLayer: extraction.residualNormPerLayer,
            residualNormSource: extraction.residualNormSource,
            residualNormConvention: extraction.residualNormConvention)
        let family = spec.family
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar,
            to: runDirectory, name: "\(stimuli.name)-\(family)")
        print(
            "extracted \(vectors.layerCount) layer vectors, hidden \(vectors.hiddenSize), "
                + "norm @ mid: \(vectors.norm(at: vectors.layerCount / 2))")

        let prompt = spec.family == "qwen3" ? config.prompt + " /no_think" : config.prompt
        let greedy = GenerateParameters(maxTokens: config.maxTokens, temperature: 0)

        try await setInterventions(container, [])
        let baseline = try await generate(container, prompt: prompt, parameters: greedy)
        let baselineMarkers = FrenchMarkers.count(in: baseline)
        print("baseline (markers: \(baselineMarkers)): \(flatten(baseline).prefix(110))…")
        log.append(record(spec.id, "baseline", nil, nil, baseline, baselineMarkers))

        // Layer/alpha grid with the concept vector.
        var best: (layer: Int, alpha: Float, markers: Int, text: String)?
        for fraction in config.layerFractions {
            let layer = min(
                vectors.layerCount - 1, Int(Float(vectors.layerCount) * fraction))
            for alpha in config.alphas {
                try await setInterventions(
                    container,
                    [VectorInjector(layer: layer, vector: vectors.perLayer[layer], alpha: alpha)])
                let steered = try await generate(
                    container, prompt: prompt, parameters: greedy)
                let markers = FrenchMarkers.count(in: steered)
                print(
                    "L\(layer) α\(alpha) (markers: \(markers)): "
                        + "\(flatten(steered).prefix(90))…")
                log.append(record(spec.id, "concept", layer, alpha, steered, markers))
                if best == nil || markers > best!.markers {
                    best = (layer, alpha, markers, steered)
                }
            }
        }
        guard let best else {
            throw ToyConceptFailure(model: spec.id, reason: "no steered generations ran")
        }

        // Matched-norm random control at the best (layer, alpha).
        var rng = SplitMix64(seed: config.seed)
        let random = try SteeringVectorMath.randomVector(
            dimension: vectors.hiddenSize,
            norm: vectors.norm(at: best.layer), using: &rng)
        try await setInterventions(
            container,
            [VectorInjector(layer: best.layer, vector: random, alpha: best.alpha)])
        let randomSteered = try await generate(container, prompt: prompt, parameters: greedy)
        let randomMarkers = FrenchMarkers.count(in: randomSteered)
        print("random control (markers: \(randomMarkers)): \(flatten(randomSteered).prefix(90))…")
        log.append(record(spec.id, "random", best.layer, best.alpha, randomSteered, randomMarkers))
        try await setInterventions(container, [])

        // Assertions.
        guard best.markers > baselineMarkers else {
            throw ToyConceptFailure(
                model: spec.id,
                reason: "concept vector never increased French markers "
                    + "(best \(best.markers) vs baseline \(baselineMarkers))")
        }
        guard best.text != randomSteered else {
            throw ToyConceptFailure(
                model: spec.id, reason: "concept-steered output equals random-steered output")
        }
        guard best.markers > randomMarkers else {
            throw ToyConceptFailure(
                model: spec.id,
                reason: "random vector matched concept vector on French markers "
                    + "(\(randomMarkers) vs \(best.markers)) — vector is not concept-specific")
        }
        print(
            "PASS \(spec.id): best L\(best.layer) α\(best.alpha) markers "
                + "\(best.markers) (baseline \(baselineMarkers), random \(randomMarkers))")
    }

    // MARK: - Helpers

    private static func record(
        _ model: String, _ condition: String, _ layer: Int?, _ alpha: Float?,
        _ text: String, _ markers: Int
    ) -> String {
        struct Record: Encodable {
            let model: String
            let condition: String
            let layer: Int?
            let alpha: Float?
            let markers: Int
            let text: String
        }
        let data = (try? JSONEncoder().encode(
            Record(
                model: model, condition: condition, layer: layer, alpha: alpha,
                markers: markers, text: text))) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    private static func flatten(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ")
    }

    private static func setInterventions(
        _ container: ModelContainer, _ interventions: [any LayerIntervention]
    ) async throws {
        try await container.perform { context in
            guard let model = context.model as? InterventionHookable else {
                throw ToyConceptFailure(
                    model: "\(type(of: context.model))",
                    reason: "model is not InterventionHookable — wrong factory?")
            }
            model.interventions = interventions
        }
    }

    private static func generate(
        _ container: ModelContainer, prompt: String, parameters: GenerateParameters
    ) async throws -> String {
        let input = try await container.prepare(input: UserInput(prompt: prompt))
        let stream = try await container.generate(input: input, parameters: parameters)
        var text = ""
        for await event in stream {
            if case .chunk(let chunk) = event { text += chunk }
        }
        return text
    }
}
