import Foundation
import Testing
@testable import SteeringKit

@Suite struct StimulusSetSchemaTests {

    @Test func loadPairedStimuliJSONL() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "pairs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(component: "pairs.jsonl")
        try """
            {"id":"p1","topic":"weather","positive":"The room grew tense before the storm.","negative":"The room grew quiet before the storm.","split":"train"}
            {"id":"p2","positive":"He checked the locked door twice.","negative":"He checked the painted door twice.","split":"test"}

            """.write(to: url, atomically: true, encoding: .utf8)

        let loaded = try StimulusSet.loadPairs(url: url)
        #expect(loaded.pairs.count == 2)
        #expect(loaded.pairs[0].positive.contains("tense"))
        #expect(loaded.hash.count == 64)
    }

    @Test func loadMultiConceptStimuliJSONL() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "stories-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(component: "stories.jsonl")
        try """
            {"id":"joy-1","concept":"joy","topic":"commute","text":"She hummed through the delayed train ride.","split":"train"}
            {"id":"anger-1","concept":"anger","topic":"commute","text":"He gripped the pole and counted each delay.","split":"train"}

            """.write(to: url, atomically: true, encoding: .utf8)

        let loaded = try StimulusSet.loadMultiConceptTexts(url: url)
        #expect(loaded.rows.map(\.concept) == ["joy", "anger"])
        #expect(loaded.hash.count == 64)
    }

    @Test func vectorExtractionRecipeHasStableHash() throws {
        let recipe = VectorExtractionRecipe(
            name: "joy-emotion-v1",
            method: .emotionGrandMean,
            targetConcept: "joy",
            datasets: [
                .init(
                    role: .multiConceptStories,
                    path: "prompts/emotions/pilot/stories.jsonl",
                    sha256: String(repeating: "a", count: 64)),
                .init(
                    role: .neutralCorpus,
                    path: "prompts/neutral/corpus.jsonl",
                    sha256: String(repeating: "b", count: 64)),
            ],
            readingPosition: .meanFromToken(50),
            neutralPCSelection: .explainedVariance(0.5))

        let decoded = try JSONDecoder().decode(
            VectorExtractionRecipe.self,
            from: JSONEncoder().encode(recipe))
        #expect(decoded == recipe)
        #expect(recipe.canonicalHash() == decoded.canonicalHash())
        #expect(recipe.canonicalHash().count == 64)
    }

    @Test func meanFromTokenAdvertisesStrictMinimumLength() {
        #expect(ReadingPosition.lastToken.minimumTokenCount == 1)
        #expect(ReadingPosition.meanFromToken(50).minimumTokenCount == 51)
        #expect(ReadingPosition.meanFromToken(50).requestedStartIndex == 50)
    }
}
