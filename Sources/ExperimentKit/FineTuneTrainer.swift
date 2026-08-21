import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import PDFKit
import SteeringKit

public enum FineTuneTrainingMode: String, Codable, Sendable, CaseIterable, Identifiable {
    case document
    case instructionChat = "instruction_chat"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .document: "Document adaptation"
        case .instructionChat: "Instruction/chat tuning"
        }
    }
}

public struct FineTuneTrainingRequest: Sendable {
    public var name: String
    public var baseModelID: String
    public var trainingMode: FineTuneTrainingMode
    public var fineTuneType: String
    public var rank: Int
    public var scale: Float
    public var adaptedLayers: Int
    public var batchSize: Int
    public var iterations: Int
    public var learningRate: Double
    public var adapterDirectory: URL
    public var trainingDataPath: String
    public var validationDataPath: String

    public init(
        name: String,
        baseModelID: String,
        trainingMode: FineTuneTrainingMode = .document,
        fineTuneType: String,
        rank: Int,
        scale: Float,
        adaptedLayers: Int,
        batchSize: Int,
        iterations: Int,
        learningRate: Double,
        adapterDirectory: URL,
        trainingDataPath: String,
        validationDataPath: String
    ) {
        self.name = name
        self.baseModelID = baseModelID
        self.trainingMode = trainingMode
        self.fineTuneType = fineTuneType
        self.rank = rank
        self.scale = scale
        self.adaptedLayers = adaptedLayers
        self.batchSize = batchSize
        self.iterations = iterations
        self.learningRate = learningRate
        self.adapterDirectory = adapterDirectory
        self.trainingDataPath = trainingDataPath
        self.validationDataPath = validationDataPath
    }
}

public struct FineTuneTrainingResult: Sendable {
    public var adapterURL: URL
    public var configURL: URL
    public var trainExamples: Int
    public var validationExamples: Int
}

public enum FineTuneTrainingEvent: Sendable, CustomStringConvertible {
    case loadingModel(Double?)
    case prepared(trainExamples: Int, validationExamples: Int)
    case chunked(trainChunks: Int, validationChunks: Int, maxTokens: Int, batchSize: Int)
    case trainingLoopStarting(iterations: Int)
    case train(iteration: Int, iterations: Int, loss: Float, iterationsPerSecond: Double, tokensPerSecond: Double)
    case validation(iteration: Int, iterations: Int, loss: Float, seconds: Double)
    case save(iteration: Int, url: URL)

    public var description: String {
        switch self {
        case .loadingModel(let fraction):
            if let fraction {
                return "loading base model \(Int(fraction * 100))%"
            }
            return "loading base model"
        case .prepared(let trainExamples, let validationExamples):
            return "prepared \(trainExamples) training examples and \(validationExamples) validation examples"
        case .chunked(let trainChunks, let validationChunks, let maxTokens, let batchSize):
            return "chunked data to \(trainChunks) train chunks and \(validationChunks) validation chunks, max \(maxTokens) tokens/chunk, effective batch \(batchSize)"
        case .trainingLoopStarting(let iterations):
            return "starting LoRA training loop for \(iterations) iterations"
        case .train(let iteration, let iterations, let loss, let iterSec, let tokSec):
            return "iteration \(iteration + 1)/\(iterations): loss \(loss.formatted(.number.precision(.fractionLength(4)))), \(iterSec.formatted(.number.precision(.fractionLength(2)))) it/s, \(tokSec.formatted(.number.precision(.fractionLength(0)))) tok/s"
        case .validation(let iteration, let iterations, let loss, let seconds):
            return "validation \(iteration + 1)/\(iterations): loss \(loss.formatted(.number.precision(.fractionLength(4)))) in \(seconds.formatted(.number.precision(.fractionLength(1))))s"
        case .save(let iteration, let url):
            return "saved checkpoint at iteration \(iteration + 1) to \(url.lastPathComponent)"
        }
    }
}

public enum FineTuneTrainingError: Error, CustomStringConvertible {
    case emptyTrainingData
    case emptyValidationData
    case noInstructionRows
    case incompatibleModel
    case cancelled

    public var description: String {
        switch self {
        case .emptyTrainingData:
            "training data folder has no usable examples"
        case .emptyValidationData:
            "validation data folder has no usable held-out examples"
        case .noInstructionRows:
            "instruction/chat mode requires structured rows with user/prompt and assistant/completion fields"
        case .incompatibleModel:
            "loaded model does not expose LoRA-trainable modules"
        case .cancelled:
            "training cancelled"
        }
    }
}

public enum FineTuneTrainer {
    private static let maxTrainingChunkTokens = 768
    private static let tokenChunkOverlap = 48

    public static func train(
        _ request: FineTuneTrainingRequest,
        progress: @Sendable @escaping (FineTuneTrainingEvent) -> Void
    ) async throws -> FineTuneTrainingResult {
        let rawTrain = loadDataset(path: request.trainingDataPath, defaultFilename: "train.jsonl")
        let rawValidation = loadDataset(
            path: request.validationDataPath, defaultFilename: "validation.jsonl")
        let instructionTrain = loadInstructionExamples(
            path: request.trainingDataPath, defaultFilename: "train.jsonl")
        let instructionValidation = loadInstructionExamples(
            path: request.validationDataPath, defaultFilename: "validation.jsonl")
        let trainExampleCount: Int
        let validationExampleCount: Int
        switch request.trainingMode {
        case .document:
            guard !rawTrain.isEmpty else { throw FineTuneTrainingError.emptyTrainingData }
            guard !rawValidation.isEmpty else { throw FineTuneTrainingError.emptyValidationData }
            trainExampleCount = rawTrain.count
            validationExampleCount = rawValidation.count
        case .instructionChat:
            guard !instructionTrain.isEmpty, !instructionValidation.isEmpty else {
                throw FineTuneTrainingError.noInstructionRows
            }
            trainExampleCount = instructionTrain.count
            validationExampleCount = instructionValidation.count
        }
        progress(.prepared(
            trainExamples: trainExampleCount,
            validationExamples: validationExampleCount))

        try FileManager.default.createDirectory(
            at: request.adapterDirectory,
            withIntermediateDirectories: true)
        let adapterURL = request.adapterDirectory.appending(component: "adapters.safetensors")
        let configURL = request.adapterDirectory.appending(component: "adapter_config.json")
        let configuration = LoRAConfiguration(
            numLayers: request.adaptedLayers,
            fineTuneType: LoRAConfiguration.FineTuneType(rawValue: request.fineTuneType) ?? .lora,
            loraParameters: .init(rank: request.rank, scale: request.scale))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(configuration).write(to: configURL, options: .atomic)

        let container = try await SteeredContainerLoader.load(modelID: request.baseModelID) { loadProgress in
            progress(.loadingModel(loadProgress.fractionCompleted))
        }
        if Task.isCancelled { throw FineTuneTrainingError.cancelled }

        try await container.perform { context in
            let module = context.model as Module
            _ = try LoRAContainer.from(model: context.model, configuration: configuration)
            let effectiveBatchSize = effectiveBatchSize(
                requested: request.batchSize,
                modelID: request.baseModelID)
            let optimizer = Adam(learningRate: Float(request.learningRate))
            switch request.trainingMode {
            case .document:
                let train = chunkExamples(rawTrain, tokenizer: context.tokenizer)
                let validation = chunkExamples(rawValidation, tokenizer: context.tokenizer)
                guard !train.isEmpty else { throw FineTuneTrainingError.emptyTrainingData }
                guard !validation.isEmpty else { throw FineTuneTrainingError.emptyValidationData }
                progress(.chunked(
                    trainChunks: train.count,
                    validationChunks: validation.count,
                    maxTokens: maxTrainingChunkTokens,
                    batchSize: effectiveBatchSize))
                let parameters = LoRATrain.Parameters(
                    batchSize: effectiveBatchSize,
                    iterations: request.iterations,
                    stepsPerReport: max(1, min(10, request.iterations)),
                    stepsPerEval: max(1, min(100, request.iterations)),
                    validationBatches: min(10, max(1, validation.count)),
                    saveEvery: max(1, min(100, request.iterations)),
                    adapterURL: adapterURL)

                var stoppedByCancellation = false
                progress(.trainingLoopStarting(iterations: request.iterations))
                try LoRATrain.train(
                    model: module,
                    train: train,
                    validate: validation,
                    optimizer: optimizer,
                    tokenizer: context.tokenizer,
                    parameters: parameters
                ) { event in
                    if Task.isCancelled {
                        stoppedByCancellation = true
                        return .stop
                    }
                    switch event {
                    case .train(let iteration, let loss, let iterSec, let tokSec):
                        progress(.train(
                            iteration: iteration,
                            iterations: request.iterations,
                            loss: loss,
                            iterationsPerSecond: iterSec,
                            tokensPerSecond: tokSec))
                    case .validation(let iteration, let loss, let seconds):
                        progress(.validation(
                            iteration: iteration,
                            iterations: request.iterations,
                            loss: loss,
                            seconds: seconds))
                    case .save(let iteration, let url):
                        progress(.save(iteration: iteration, url: url))
                    }
                    return .more
                }
                if stoppedByCancellation { throw FineTuneTrainingError.cancelled }

            case .instructionChat:
                let train = try tokenizeInstructionExamples(
                    instructionTrain,
                    tokenizer: context.tokenizer,
                    modelID: request.baseModelID)
                let validation = try tokenizeInstructionExamples(
                    instructionValidation,
                    tokenizer: context.tokenizer,
                    modelID: request.baseModelID)
                guard !train.isEmpty, !validation.isEmpty else {
                    throw FineTuneTrainingError.noInstructionRows
                }
                progress(.chunked(
                    trainChunks: train.count,
                    validationChunks: validation.count,
                    maxTokens: maxTrainingChunkTokens,
                    batchSize: effectiveBatchSize))
                progress(.trainingLoopStarting(iterations: request.iterations))
                try trainMaskedInstructionAdapter(
                    model: module,
                    train: train,
                    validation: validation,
                    optimizer: optimizer,
                    iterations: request.iterations,
                    batchSize: effectiveBatchSize,
                    adapterURL: adapterURL,
                    progress: progress)
            }
            try LoRATrain.saveLoRAWeights(model: module, url: adapterURL)
        }

        if Task.isCancelled { throw FineTuneTrainingError.cancelled }
        return .init(
            adapterURL: adapterURL,
            configURL: configURL,
            trainExamples: trainExampleCount,
            validationExamples: validationExampleCount)
    }

    public static func loadDataset(path: String, defaultFilename: String) -> [String] {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let primaryURL = datasetURL(path: trimmed, defaultFilename: defaultFilename)
        let folderURL = datasetFolderURL(path: trimmed, defaultFilename: defaultFilename)
        var examples: [String] = []

        if FileManager.default.fileExists(atPath: primaryURL.path) {
            examples.append(contentsOf: loadExamples(from: primaryURL))
        }
        if let urls = try? FileManager.default.contentsOfDirectory(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            let primaryPath = primaryURL.standardizedFileURL.path
            for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard url.standardizedFileURL.path != primaryPath,
                    isSupportedTrainingSource(url),
                    !url.lastPathComponent.lowercased().hasPrefix("readme")
                else { continue }
                examples.append(contentsOf: loadExamples(from: url))
            }
        }
        return examples.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    public struct InstructionExample: Sendable, Equatable {
        public var system: String
        public var user: String
        public var assistant: String

        public init(system: String = "", user: String, assistant: String) {
            self.system = system
            self.user = user
            self.assistant = assistant
        }
    }

    struct TokenizedInstructionExample: Sendable {
        var tokens: [Int]
        var weights: [Float]
    }

    public static func loadInstructionExamples(
        path: String,
        defaultFilename: String
    ) -> [InstructionExample] {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let primaryURL = datasetURL(path: trimmed, defaultFilename: defaultFilename)
        let folderURL = datasetFolderURL(path: trimmed, defaultFilename: defaultFilename)
        var examples: [InstructionExample] = []

        if FileManager.default.fileExists(atPath: primaryURL.path) {
            examples.append(contentsOf: loadInstructionExamples(from: primaryURL))
        }
        if let urls = try? FileManager.default.contentsOfDirectory(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            let primaryPath = primaryURL.standardizedFileURL.path
            for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard url.standardizedFileURL.path != primaryPath,
                    ["json", "jsonl"].contains(url.pathExtension.lowercased()),
                    !url.lastPathComponent.lowercased().hasPrefix("readme")
                else { continue }
                examples.append(contentsOf: loadInstructionExamples(from: url))
            }
        }
        return examples
    }

    private static func loadInstructionExamples(from url: URL) -> [InstructionExample] {
        guard ["json", "jsonl", ""].contains(url.pathExtension.lowercased()),
            let content = try? String(contentsOf: url, encoding: .utf8)
        else { return [] }
        return parseInstructionExamples(content)
    }

    static func parseInstructionExamples(_ content: String) -> [InstructionExample] {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if let data = trimmed.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        {
            return array.compactMap(instructionExample(from:))
        }
        return trimmed.components(separatedBy: .newlines).compactMap { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty,
                let data = line.data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            return instructionExample(from: object)
        }
    }

    private static func instructionExample(from object: [String: Any]) -> InstructionExample? {
        let system = (object["system"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let user = (object["user"] as? String ?? object["prompt"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let assistant = (object["assistant"] as? String ?? object["completion"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !user.isEmpty, !assistant.isEmpty else { return nil }
        return InstructionExample(system: system, user: user, assistant: assistant)
    }

    private static func datasetURL(path: String, defaultFilename: String) -> URL {
        var url = FineTuneStore.absoluteURL(path)
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            url = url.appending(component: defaultFilename)
        }
        return url
    }

    private static func datasetFolderURL(path: String, defaultFilename: String) -> URL {
        let url = FineTuneStore.absoluteURL(path)
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return url
        }
        if url.lastPathComponent == defaultFilename {
            return url.deletingLastPathComponent()
        }
        return url.deletingLastPathComponent()
    }

    private static func isSupportedTrainingSource(_ url: URL) -> Bool {
        ["json", "jsonl", "txt", "md", "pdf"].contains(url.pathExtension.lowercased())
    }

    private static func loadExamples(from url: URL) -> [String] {
        switch url.pathExtension.lowercased() {
        case "json", "jsonl", "":
            guard let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
            return parseStructuredExamples(content)
        case "txt", "md":
            guard let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
            return [content]
        case "pdf":
            guard let text = extractPDFText(url) else { return [] }
            return [text]
        default:
            return []
        }
    }

    // Internal: shared with `FineTuneTrainingData` so server-bound folder
    // payloads parse structured rows exactly as local training does.
    static func parseStructuredExamples(_ content: String) -> [String] {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if let data = trimmed.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        {
            return array.compactMap(trainingText(from:))
        }
        return trimmed.components(separatedBy: .newlines).compactMap { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { return nil }
            guard let data = line.data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                return line
            }
            return trainingText(from: object)
        }
    }

    private static func trainingText(from object: [String: Any]) -> String? {
        if let text = object["text"] as? String,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return text
        }
        let system = object["system"] as? String ?? ""
        let user = object["user"] as? String ?? object["prompt"] as? String ?? ""
        let assistant = object["assistant"] as? String ?? object["completion"] as? String ?? ""
        let joined = [system, user, assistant]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        return joined.isEmpty ? nil : joined
    }

    // Internal: shared with `FineTuneTrainingData` (same rationale as
    // `parseStructuredExamples`).
    static func extractPDFText(_ url: URL) -> String? {
        guard let document = PDFDocument(url: url) else { return nil }
        let text = (0 ..< document.pageCount).compactMap { index in
            document.page(at: index)?.string?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { !$0.isEmpty }
        .joined(separator: "\n\n")
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func chunkExamples(
        _ examples: [String],
        tokenizer: Tokenizer,
        maxTokens: Int = maxTrainingChunkTokens
    ) -> [String] {
        examples.flatMap { chunkText($0, tokenizer: tokenizer, maxTokens: maxTokens) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func chunkText(_ text: String, tokenizer: Tokenizer, maxTokens: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let tokens = tokenizer.encode(text: trimmed, addSpecialTokens: false)
        guard tokens.count > maxTokens else { return [trimmed] }

        var chunks: [String] = []
        var start = 0
        let stride = max(1, maxTokens - min(tokenChunkOverlap, maxTokens / 4))
        while start < tokens.count {
            let end = min(tokens.count, start + maxTokens)
            let piece = tokenizer.decode(
                tokenIds: Array(tokens[start ..< end]),
                skipSpecialTokens: true)
            let clean = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            if !clean.isEmpty {
                chunks.append(clean)
            }
            if end == tokens.count { break }
            start += stride
        }

        if !chunks.isEmpty { return chunks }
        return chunkByCharacters(trimmed, maxCharacters: maxTokens * 4)
    }

    private static func chunkByCharacters(_ text: String, maxCharacters: Int) -> [String] {
        var chunks: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            let end = text.index(start, offsetBy: maxCharacters, limitedBy: text.endIndex)
                ?? text.endIndex
            let clean = String(text[start ..< end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !clean.isEmpty {
                chunks.append(clean)
            }
            start = end
        }
        return chunks
    }

    static func tokenizeInstructionExamples(
        _ examples: [InstructionExample],
        tokenizer: Tokenizer,
        modelID: String,
        maxTokens: Int = maxTrainingChunkTokens
    ) throws -> [TokenizedInstructionExample] {
        try examples.compactMap { example in
            let promptTokens = try instructionTokens(
                example: example,
                includeAssistant: false,
                tokenizer: tokenizer,
                modelID: modelID)
            let fullTokens = try instructionTokens(
                example: example,
                includeAssistant: true,
                tokenizer: tokenizer,
                modelID: modelID)
            guard fullTokens.count >= 2,
                promptTokens.count < fullTokens.count,
                fullTokens.count <= maxTokens + 1
            else { return nil }
            var weights = Array(repeating: Float(0), count: fullTokens.count - 1)
            let firstAssistantTarget = max(0, promptTokens.count - 1)
            if firstAssistantTarget < weights.count {
                for index in firstAssistantTarget ..< weights.count {
                    weights[index] = 1
                }
            }
            guard weights.contains(1) else { return nil }
            return TokenizedInstructionExample(tokens: fullTokens, weights: weights)
        }
    }

    private static func instructionTokens(
        example: InstructionExample,
        includeAssistant: Bool,
        tokenizer: Tokenizer,
        modelID: String
    ) throws -> [Int] {
        let lowerModel = modelID.lowercased()
        var user = example.user
        var messages: [[String: any Sendable]] = []
        if !example.system.isEmpty {
            if lowerModel.contains("gemma") {
                user = example.system + "\n\n" + user
            } else {
                messages.append(["role": "system", "content": example.system])
            }
        }
        messages.append(["role": "user", "content": user])
        if includeAssistant {
            messages.append(["role": "assistant", "content": example.assistant])
        }
        do {
            return try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ExperimentTasks.qwenContext(
                    modelID: modelID,
                    qwenThinkingEnabled: false))
        } catch TokenizerError.missingChatTemplate {
            return fallbackInstructionTokens(
                example: example,
                includeAssistant: includeAssistant,
                tokenizer: tokenizer,
                modelID: modelID)
        }
    }

    private static func fallbackInstructionTokens(
        example: InstructionExample,
        includeAssistant: Bool,
        tokenizer: Tokenizer,
        modelID: String
    ) -> [Int] {
        let lowerModel = modelID.lowercased()
        var parts: [String] = []
        if !example.system.isEmpty, !lowerModel.contains("gemma") {
            parts.append("System: \(example.system)")
        }
        let user = !example.system.isEmpty && lowerModel.contains("gemma")
            ? example.system + "\n\n" + example.user
            : example.user
        parts.append("User: \(user)")
        parts.append(includeAssistant ? "Assistant: \(example.assistant)" : "Assistant:")
        return tokenizer.encode(text: parts.joined(separator: "\n"), addSpecialTokens: true)
    }

    private static func trainMaskedInstructionAdapter(
        model: Module,
        train: [TokenizedInstructionExample],
        validation: [TokenizedInstructionExample],
        optimizer: Optimizer,
        iterations: Int,
        batchSize: Int,
        adapterURL: URL,
        progress: @Sendable @escaping (FineTuneTrainingEvent) -> Void
    ) throws {
        let lossValueGrad = valueAndGrad(model: model) { model, arrays in
            let (ce, ntoks) = maskedInstructionLoss(
                model: model,
                inputs: arrays[0],
                targets: arrays[1],
                weights: arrays[2])
            return [ce, ntoks]
        }

        var iterator = TokenizedInstructionBatchIterator(
            dataset: train, batchSize: batchSize, train: true)
        var losses: [Float] = []
        var tokenCount = 0
        var start = Date.timeIntervalSinceReferenceDate
        let stepsPerReport = max(1, min(10, iterations))
        let stepsPerEval = max(1, min(100, iterations))
        let validationBatches = min(10, max(1, validation.count))
        let saveEvery = max(1, min(100, iterations))

        for iteration in 0 ..< iterations {
            if Task.isCancelled { throw FineTuneTrainingError.cancelled }
            guard let (inputs, targets, weights) = iterator.next() else { break }
            let (resultArray, grad) = lossValueGrad(model, [inputs, targets, weights])
            let lvalue = resultArray[0]
            let tokens = resultArray[1]

            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, lvalue)

            losses.append(lvalue.item(Float.self))
            tokenCount += tokens.item(Int.self)

            if (iteration + 1) % stepsPerReport == 0 {
                let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
                let now = Date.timeIntervalSinceReferenceDate
                progress(.train(
                    iteration: iteration,
                    iterations: iterations,
                    loss: trainingLoss,
                    iterationsPerSecond: Double(stepsPerReport) / (now - start),
                    tokensPerSecond: Double(tokenCount) / (now - start)))
                losses.removeAll()
                tokenCount = 0
                start = Date.timeIntervalSinceReferenceDate
            }

            if iteration == 0 || (iteration + 1) % stepsPerEval == 0 {
                let validationStart = Date.timeIntervalSinceReferenceDate
                let validationLoss = evaluateMaskedInstruction(
                    model: model,
                    dataset: validation,
                    batchSize: batchSize,
                    batchCount: validationBatches)
                progress(.validation(
                    iteration: iteration,
                    iterations: iterations,
                    loss: validationLoss,
                    seconds: Date.timeIntervalSinceReferenceDate - validationStart))
                start = Date.timeIntervalSinceReferenceDate
            }

            if (iteration + 1) % saveEvery == 0 {
                try LoRATrain.saveLoRAWeights(model: model, url: adapterURL)
                progress(.save(iteration: iteration, url: adapterURL))
                start = Date.timeIntervalSinceReferenceDate
            }
        }
    }

    private struct TokenizedInstructionBatchIterator: Sequence, IteratorProtocol {
        let dataset: [TokenizedInstructionExample]
        let batchSize: Int
        let train: Bool
        var indices: [Int]
        var index = 0

        init(dataset: [TokenizedInstructionExample], batchSize: Int, train: Bool) {
            self.dataset = dataset
            self.batchSize = batchSize
            self.train = train
            self.indices = Array(0 ..< dataset.count)
            if train { self.indices.shuffle() }
        }

        mutating func next() -> (MLXArray, MLXArray, MLXArray)? {
            if index >= indices.count {
                guard train else { return nil }
                indices.shuffle()
                index = 0
            }
            let endIndex = Swift.min(index + batchSize, indices.count)
            let examples = (index ..< endIndex).map { dataset[indices[$0]] }
            index = endIndex
            return batch(examples)
        }

        private func batch(_ examples: [TokenizedInstructionExample]) -> (MLXArray, MLXArray, MLXArray)? {
            guard let maxLength = examples.map({ $0.tokens.count - 1 }).max(), maxLength > 0 else {
                return nil
            }
            let inputRows = examples.map { padded($0.tokens.dropLast().map(Int32.init), to: maxLength) }
            let targetRows = examples.map { padded($0.tokens.dropFirst().map(Int32.init), to: maxLength) }
            let weightRows = examples.map { padded($0.weights, to: maxLength) }
            let shape = [examples.count, maxLength]
            return (
                MLXArray(inputRows.flatMap { $0 }, shape),
                MLXArray(targetRows.flatMap { $0 }, shape),
                MLXArray(weightRows.flatMap { $0 }, shape)
            )
        }

        private func padded<T>(_ values: some Sequence<T>, to count: Int) -> [T]
        where T: ExpressibleByIntegerLiteral {
            var row = Array(values)
            if row.count < count {
                row.append(contentsOf: Array(repeating: 0, count: count - row.count))
            }
            return row
        }
    }

    private static func maskedInstructionLoss(
        model: Module,
        inputs: MLXArray,
        targets: MLXArray,
        weights: MLXArray
    ) -> (MLXArray, MLXArray) {
        let model = model as! any LLMModel
        let logits = model(inputs, cache: nil).asType(.float32)
        let ntoks = weights.sum()
        let ce = (crossEntropy(logits: logits, targets: targets) * weights).sum() / ntoks
        return (ce, ntoks)
    }

    private static func evaluateMaskedInstruction(
        model: Module,
        dataset: [TokenizedInstructionExample],
        batchSize: Int,
        batchCount: Int
    ) -> Float {
        var allLosses = [Float]()
        var tokenCount = 0
        var iterator = TokenizedInstructionBatchIterator(
            dataset: dataset, batchSize: batchSize, train: false)
        var batchIndex = 0
        while let (inputs, targets, weights) = iterator.next() {
            let (losses, tokens) = maskedInstructionLoss(
                model: model, inputs: inputs, targets: targets, weights: weights)
            allLosses.append((losses * tokens).item(Float.self))
            tokenCount += tokens.item(Int.self)
            batchIndex += 1
            if batchCount != 0 && batchIndex >= batchCount { break }
        }
        guard tokenCount > 0 else { return .nan }
        return (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
    }

    private static func effectiveBatchSize(requested: Int, modelID: String) -> Int {
        let size = modelSizeBillions(modelID) ?? 4
        if size >= 10 { return 1 }
        return max(1, min(requested, 2))
    }

    private static func modelSizeBillions(_ modelID: String) -> Double? {
        let pattern = #"(?i)(\d+(?:\.\d+)?)\s*b"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(
                in: modelID,
                range: NSRange(modelID.startIndex..., in: modelID)),
            let range = Range(match.range(at: 1), in: modelID)
        else { return nil }
        return Double(modelID[range])
    }
}
