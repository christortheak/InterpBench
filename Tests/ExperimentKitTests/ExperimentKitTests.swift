import Foundation
import MLXLMCommon
import Testing
@testable import ExperimentKit
import SteeringKit

@Suite struct GameProtocolTests {
    /// Keeps the Game interface honest while Experiment B ships first.
    @Test func prisonersDilemmaIsTwoPlayer() {
        let game = PrisonersDilemma()
        #expect(game.playerCount == 2)
        #expect(game.name == "prisoners-dilemma")
    }
}

@Suite struct VectorArtifactLabelTests {
    @Test func labelIncludesRecipeProvenanceWhenPresent() {
        let vectors = ConceptVectors(perLayer: [[1, 0, 0]])
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model",
            concept: "joy",
            stimulusSetHash: "hash",
            vectors: vectors,
            recipeMethod: .emotionGrandMean,
            recipeHash: "recipehash",
            recipeName: "joy-emotion-grand-mean",
            extractionDate: Date(timeIntervalSince1970: 0))
        let artifact = VectorArtifact(
            directory: URL(filePath: "/tmp/run"),
            name: "joy-qwen",
            sidecar: sidecar)
        #expect(artifact.label.contains("emotion-grand-mean"))
    }

    @Test func grandMeanSidecarCanAvoidContrastiveMethodStamp() {
        let vectors = ConceptVectors(perLayer: [[1, 0, 0]])
        let screening = ExtractionScreening(
            readingPosition: .meanFromToken(50),
            sourceCount: 12,
            includedCount: 11,
            excludedShortCount: 1)
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model",
            concept: "joy",
            stimulusSetHash: "hash",
            vectors: vectors,
            options: nil,
            recipeMethod: .emotionGrandMean,
            recipeHash: "recipehash",
            recipeName: "joy-emotion-grand-mean",
            appliedReadingPosition: .meanFromToken(50),
            neutralProjection: "none",
            screening: screening,
            comparisonConcepts: ["anger", "joy"],
            selectedTopics: ["commute"],
            selectedSplits: ["build"],
            extractionDate: Date(timeIntervalSince1970: 0))
        #expect(sidecar.schemaVersion == 2)
        #expect(sidecar.extractionMethod == nil)
        #expect(sidecar.recipeMethod == VectorExtractionRecipe.Method.emotionGrandMean.rawValue)
        #expect(sidecar.readingPosition == "mean from token 50")
        #expect(sidecar.excludedShortStimulusCount == 1)
    }

    @Test func artifactAuditFlagsAmbiguousGrandMeanSidecar() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "artifact-audit-\(UUID().uuidString)")
        let run = root.appending(component: "20260622T000000Z-test")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data().write(to: run.appending(component: "joy.safetensors"))
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model",
            concept: "joy",
            stimulusSetHash: "hash",
            vectors: ConceptVectors(perLayer: [[1, 0, 0]]),
            options: ExtractionOptions(method: .meanDifference, readingPosition: .meanFromToken(50)),
            recipeMethod: .emotionGrandMean,
            recipeHash: "recipehash",
            recipeName: "joy-emotion-grand-mean")
        let encoder = JSONEncoder()
        try encoder.encode(sidecar).write(to: run.appending(component: "joy.json"))

        let findings = VectorCatalog.auditArtifacts(runsDirectory: root)
        #expect(findings.contains { $0.issue.contains("contrastive extractionMethod") })
    }
}

@Suite struct EmotionCorpusInputTests {
    @Test func plainStoryTextBecomesDraftCorpusRows() throws {
        let rows = try ConceptBuilder.parseMultiConceptRows(
            Data("A short story about a careful morning.\n\nAnother scene with the same mood.".utf8),
            filename: "manual",
            defaultConcept: "calm",
            defaultTopic: "morning",
            defaultSplit: "build")
        #expect(rows.count == 2)
        #expect(rows.allSatisfy { $0.concept == "calm" })
        #expect(rows.allSatisfy { $0.topic == "morning" })
        #expect(rows.allSatisfy { $0.split == "build" })
    }

    @Test func legacySplitsNormalizeToBuildAndValidation() throws {
        let rows = try ConceptBuilder.parseMultiConceptRows(
            Data(
                """
                {"concept":"joy","topic":"commute","text":"A.","split":"train"}
                {"concept":"fear","topic":"commute","text":"B.","split":"dev"}
                {"concept":"anger","topic":"work","text":"C.","split":"test"}
                """.utf8),
            filename: "legacy.jsonl")
        #expect(rows.map(\.split) == ["build", "validation", "validation"])
    }

    @Test func coworkJSONLCorpusCanContainMultipleConcepts() throws {
        let rows = try ConceptBuilder.parseMultiConceptRows(
            Data(
                """
                {"id":"fear-commute-1","concept":"fear","topic":"commute","split":"build","text":"The platform emptied as the footsteps kept pace behind her.","source":"cowork-agent-draft"}
                {"id":"joy-commute-1","concept":"joy","topic":"commute","split":"build","text":"The delayed train gave him ten extra minutes to finish the song he loved.","source":"cowork-agent-draft"}
                {"id":"anger-commute-1","concept":"anger","topic":"commute","split":"build","text":"The train stalled again while the announcement blamed everyone but the operator.","source":"cowork-agent-draft"}
                """.utf8),
            filename: "cowork.jsonl")
        #expect(rows.count == 3)
        #expect(Set(rows.map(\.concept)) == ["fear", "joy", "anger"])
        #expect(Set(rows.map { $0.topic ?? "" }) == ["commute"])
        #expect(rows.allSatisfy { $0.split == "build" })
    }

    @Test func neutralCorpusParserDeduplicatesJSONLRows() throws {
        let rows = try ConceptBuilder.parseNeutralTexts(
            Data(
                """
                {"text":"The storage cabinet was labeled according to shelf height and box size."}
                {"text":"The storage cabinet was labeled according to shelf height and box size."}
                {"text":"A maintenance schedule listed the inspection dates for each hallway fixture."}
                """.utf8))
        #expect(rows.count == 2)
        #expect(rows[0].contains("storage cabinet"))
        #expect(rows[1].contains("maintenance schedule"))
    }
}

@Suite struct FineTuneDatasetTests {
    @Test func loaderIgnoresTemplateAndReadsDroppedTextFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "fine-tune-data-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try """
            {"text":"","metadata":{"split":"train","notes":"template only"}}
            """.write(to: root.appending(component: "train.jsonl"), atomically: true, encoding: .utf8)
        try "A real document for adapter training.\nWith another line."
            .write(to: root.appending(component: "source.txt"), atomically: true, encoding: .utf8)

        let examples = FineTuneTrainer.loadDataset(path: root.path, defaultFilename: "train.jsonl")
        #expect(examples == ["A real document for adapter training.\nWith another line."])
    }

    @Test func instructionLoaderRequiresUserAndAssistantRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "fine-tune-instructions-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try """
            {"text":"raw document rows are ignored in instruction mode"}
            {"system":"sys","user":"question","assistant":"answer"}
            {"prompt":"second","completion":"reply"}
            """.write(to: root.appending(component: "train.jsonl"), atomically: true, encoding: .utf8)

        let examples = FineTuneTrainer.loadInstructionExamples(
            path: root.path,
            defaultFilename: "train.jsonl")
        #expect(examples.count == 2)
        #expect(examples[0].system == "sys")
        #expect(examples[0].user == "question")
        #expect(examples[0].assistant == "answer")
    }

    @Test func chunkerBoundsLongExamplesByTokenCount() {
        let text = (0 ..< 25).map { "w\($0)" }.joined(separator: " ")
        let chunks = FineTuneTrainer.chunkExamples([text], tokenizer: TestFineTuneTokenizer(), maxTokens: 10)
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { TestFineTuneTokenizer().encode(text: $0, addSpecialTokens: false).count <= 10 })
    }

    @Test func instructionTokenizationMasksPromptTokens() throws {
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [.init(system: "", user: "w1 w2", assistant: "w3 w4")],
            renderer: TestFineTuneRenderer(),
            modelID: "test/qwen")
        #expect(examples.count == 1)
        #expect(examples[0].weights.first == 0)
        #expect(examples[0].weights.suffix(2).allSatisfy { $0 == 1 })
        // The completed render must not end in a generation prompt — the
        // defect this fake now models (external review, 2026-09-05 — SCI-03).
        #expect(examples[0].tokens.last != TestFineTuneRenderer.assistantHeaderToken)
    }
}

/// The instruction-render fake. Unlike `TestFineTuneTokenizer` (which is now
/// only the document chunker's), it MODELS the assistant header: emitted
/// before every assistant turn AND as the generation prompt, exactly as a real
/// chat template does. That is what makes the masking test able to see a
/// completed example rendered with a generation prompt — the old fake ignored
/// the flag entirely and so could never have caught SCI-03 (external review,
/// 2026-09-05). Real-template coverage lives in `FineTuneTokenizationTests`.
private struct TestFineTuneRenderer: FineTuneInstructionRenderer {
    static let assistantHeaderToken = 99

    func chatTemplateTokens(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var ids: [Int] = []
        for message in messages {
            if message["role"] as? String == "assistant" {
                ids.append(Self.assistantHeaderToken)
            }
            ids.append(
                contentsOf: encode(
                    text: message["content"] as? String ?? "", addSpecialTokens: false))
        }
        if addGenerationPrompt { ids.append(Self.assistantHeaderToken) }
        return ids
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(whereSeparator: \.isWhitespace).compactMap { piece in
            let digits = piece.drop { !$0.isNumber }
            return Int(digits)
        }
    }
}

private struct TestFineTuneTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.split(whereSeparator: \.isWhitespace).compactMap { piece in
            let digits = piece.drop { !$0.isNumber }
            return Int(digits)
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "w\($0)" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            let content = message["content"] as? String ?? ""
            return encode(text: content, addSpecialTokens: false)
        }
    }
}
