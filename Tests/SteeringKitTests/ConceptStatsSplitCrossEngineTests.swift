import Foundation
import Testing

@testable import SteeringKit

/// The screening diagnostics' SPLIT MEMBERSHIP, checked against bytes the
/// PYTHON engine produced (2026-08-28 audit, F3).
///
/// Until this fixture the two engines split differently and neither split was
/// pinned: Python held out the last ~20% of each class in FILE order and cut
/// even/odd halves, while this engine held out `index % 5 == 4`. So the same
/// stimulus file scored differently on the two engines, and on either of them
/// the number moved with how the file happened to be ordered — a file authored
/// in topic runs handed its LAST topic block over whole as the "held-out" set.
///
/// The rule is now content-derived: sort each class's rows ascending by the
/// lowercase SHA-256 hex of the row's UTF-8 text, hold out sorted position
/// % 5 == 4, halve on sorted position parity. Regenerate the fixture with
/// `scripts/regenerate-cross-engine-fixtures.py`; a diff means one engine's
/// split rule changed and the review has to decide whether the other follows.
@Suite struct ConceptStatsSplitCrossEngineTests {

    struct Case {
        let label: String
        let texts: [String]
        let contentHashOrder: [Int]
        let heldOutTexts: [String]
        let splitHalfSecondTexts: [String]
    }

    static func load() throws -> (
        minimumRowsPerClass: Int, minimumRowsPerClassSplitHalf: Int, cases: [Case]
    ) {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent()  // …/Tests/SteeringKitTests
            .deletingLastPathComponent()  // …/Tests
            .appending(components: "Fixtures", "cross-engine",
                       "concept-stats-splits.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [String: Any])
        let cases = try (try #require(root["cases"] as? [[String: Any]])).map { entry in
            Case(
                label: try #require(entry["label"] as? String),
                texts: try #require(entry["texts"] as? [String]),
                contentHashOrder: try #require(entry["contentHashOrder"] as? [Int]),
                heldOutTexts: try #require(entry["heldOutTexts"] as? [String]),
                splitHalfSecondTexts:
                    try #require(entry["splitHalfSecondTexts"] as? [String]))
        }
        return (
            try #require(root["minimumRowsPerClass"] as? Int),
            try #require(root["minimumRowsPerClassSplitHalf"] as? Int),
            cases)
    }

    @Test func splitMembershipMatchesTheServerByteForByte() throws {
        let (minimum, minimumSplitHalf, cases) = try Self.load()
        #expect(minimum == ConceptStats.minimumRowsPerClass)
        #expect(minimumSplitHalf == ConceptStats.minimumRowsPerClassSplitHalf)
        for fixture in cases {
            #expect(
                ConceptStats.contentHashOrder(fixture.texts) == fixture.contentHashOrder,
                "content-hash order diverged for \(fixture.label)")
            let held = ConceptStats.heldOutIndices(fixture.texts)
                .map { fixture.texts[$0] }.sorted()
            #expect(held == fixture.heldOutTexts,
                    "held-out membership diverged for \(fixture.label)")
            let second = ConceptStats.splitHalfSecondIndices(fixture.texts)
                .map { fixture.texts[$0] }.sorted()
            #expect(second == fixture.splitHalfSecondTexts,
                    "split-half membership diverged for \(fixture.label)")
        }
    }

    /// The claim the change exists to buy, asserted against the fixture's own
    /// pair of cases: the same texts in a different file order select the same
    /// rows. The old positional rules failed this by construction.
    @Test func scramblingTheFileDoesNotMoveTheSplit() throws {
        let cases = try Self.load().cases
        let blocked = try #require(cases.first { $0.label == "topic-blocked" })
        let scrambled = try #require(
            cases.first { $0.label == "topic-blocked-scrambled" })
        #expect(Set(blocked.texts) == Set(scrambled.texts))
        #expect(blocked.texts != scrambled.texts)
        #expect(blocked.heldOutTexts == scrambled.heldOutTexts)
        #expect(blocked.splitHalfSecondTexts == scrambled.splitHalfSecondTexts)
    }

    /// A file authored in topic runs used to hand the LAST topic block over
    /// whole as the test set. Under the content-hash rule the held-out rows
    /// come from more than one block.
    @Test func topicBlockedAuthoringNoLongerDecidesTheHeldOutSet() throws {
        let cases = try Self.load().cases
        let blocked = try #require(cases.first { $0.label == "topic-blocked" })
        let topics = Set(blocked.heldOutTexts.compactMap {
            $0.split(separator: " ").first.map(String.init)
        })
        #expect(topics.count > 1)
    }

    /// The statistic itself, not just the membership: the same activations and
    /// texts in a scrambled row order produce the identical numbers.
    @Test func statisticsAreInvariantUnderRowOrder() throws {
        let texts = (0 ..< 12).map { "stimulus \($0)" }
        let positive = (0 ..< 12).map { i in
            [Float(0.9) + Float(i) * 0.01, Float(i % 3) * 0.02, 0, 0]
        }
        let negative = positive.map { [-$0[0], $0[1], 0, 0] }
        let order = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4, 6]

        let straight = try #require(
            ConceptStats.heldOutAccuracy(
                positive: positive, negative: negative,
                positiveTexts: texts, negativeTexts: texts))
        let shuffled = try #require(
            ConceptStats.heldOutAccuracy(
                positive: order.map { positive[$0] }, negative: order.map { negative[$0] },
                positiveTexts: order.map { texts[$0] },
                negativeTexts: order.map { texts[$0] }))
        #expect(straight == shuffled)

        let straightHalf = try #require(
            ConceptStats.splitHalfCosine(
                positive: positive, negative: negative,
                positiveTexts: texts, negativeTexts: texts))
        let shuffledHalf = try #require(
            ConceptStats.splitHalfCosine(
                positive: order.map { positive[$0] }, negative: order.map { negative[$0] },
                positiveTexts: order.map { texts[$0] },
                negativeTexts: order.map { texts[$0] }))
        #expect(abs(straightHalf - shuffledHalf) < 1e-6)
    }

    /// Texts that do not line up with the activation rows cannot be split by
    /// content, so the diagnostic is absent rather than silently positional.
    @Test func misalignedTextsYieldNoStatistic() {
        let rows = (0 ..< 8).map { i in [Float(i), 1, 0] }
        #expect(
            ConceptStats.heldOutAccuracy(
                positive: rows, negative: rows,
                positiveTexts: ["a", "b"], negativeTexts: (0 ..< 8).map { "\($0)" })
                == nil)
        #expect(
            ConceptStats.splitHalfCosine(
                positive: rows, negative: rows,
                positiveTexts: (0 ..< 8).map { "\($0)" }, negativeTexts: ["a"])
                == nil)
    }
}
