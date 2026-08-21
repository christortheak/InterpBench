import Foundation
import Testing
@testable import ExperimentKit

@Suite struct ImportParserTests {

    @Test func parsesPairJSONL() throws {
        let data = Data("""
            {"positive": "Il a peur.", "negative": "He is calm."}
            {"positive": "Elle tremble.", "negative": "She waves."}
            """.utf8)
        let pairs = try ConceptBuilder.parsePairs(data, filename: "pairs.jsonl")
        #expect(pairs.count == 2)
        #expect(pairs[0].positive == "Il a peur.")
        #expect(pairs[1].negative == "She waves.")
    }

    @Test func parsesFencedPairJSONLWithLegacyMetadata() throws {
        let data = Data("""
            Here is the requested dataset:
            ```jsonl
            {"id":"anger-001","topic":"commute","positive":"The delay made her furious.","negative":"The delay left her calm.","split":"train","notes":"matched"}
            {"id":"anger-002","topic":"work","positive":"He seethed at the mistake.","negative":"He noted the mistake evenly.","split":"test"}
            ```
            """.utf8)
        let pairs = try ConceptBuilder.parsePairs(data, filename: "pairs.jsonl")
        #expect(pairs.count == 2)
        #expect(pairs[0].positive == "The delay made her furious.")
        #expect(pairs[0].negative == "The delay left her calm.")
        #expect(pairs[1].negative == "He noted the mistake evenly.")
    }

    @Test func pastedPairDetectorDoesNotTreatPlainManualRowsAsJSONL() throws {
        #expect(try ConceptBuilder.parsePastedPairs("first row\nsecond row") == nil)
        let parsed = try ConceptBuilder.parsePastedPairs("""
                ```jsonl
                {"positive":"concept present","negative":"matched control"}
                ```
                """)
        let pairs = try #require(parsed)
        #expect(pairs.count == 1)
        #expect(pairs[0].negative == "matched control")
    }

    @Test func parsesJSONArray() throws {
        let data = Data("""
            [{"positive": "a", "negative": "b"}, {"positive": "c", "negative": "d"}]
            """.utf8)
        let pairs = try ConceptBuilder.parsePairs(data, filename: "pairs.json")
        #expect(pairs.count == 2)
        #expect(pairs[1].positive == "c")
    }

    @Test func parsesCSVWithHeaderAndQuotedCommas() throws {
        let data = Data("""
            positive,negative
            "The alarm rang, loud and shrill","The clock ticked quietly"
            Storm clouds gathered fast,The sky stayed clear
            """.utf8)
        let pairs = try ConceptBuilder.parsePairs(data, filename: "pairs.csv")
        #expect(pairs.count == 2)
        #expect(pairs[0].positive == "The alarm rang, loud and shrill")
        #expect(pairs[1].negative == "The sky stayed clear")
    }

    @Test func malformedCSVThrows() {
        let data = Data("just-one-column\n".utf8)
        #expect(throws: (any Error).self) {
            try ConceptBuilder.parsePairs(data, filename: "bad.csv")
        }
    }

    @Test func parsesProbeJSONL() throws {
        let data = Data("""
            {"id":"fair-probe-001","text":"The mediator gave each side the same chance to respond.","expresses":true,"topic":"work","split":"test"}
            {"id":"fair-probe-002","text":"The manager picked the louder bidder without reading either proposal.","expresses":false,"topic":"work","split":"dev"}
            """.utf8)
        let examples = try ConceptBuilder.parseProbeExamples(
            data, filename: "items.jsonl", concept: "fair")
        #expect(examples.count == 2)
        #expect(examples[0].expresses)
        #expect(!examples[1].expresses)
        #expect(examples[0].split == "validation")
    }

    @Test func parsesPlainProbeLines() throws {
        let data = Data("""
            positive: Everyone received the same rules before the contest began.
            negative: The official changed the rules after seeing who was ahead.
            """.utf8)
        let examples = try ConceptBuilder.parseProbeExamples(
            data, filename: "items.txt", concept: "fair")
        #expect(examples.count == 2)
        #expect(examples.map(\.expresses) == [true, false])
    }
}
