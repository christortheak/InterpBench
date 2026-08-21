import Foundation
import Testing
@testable import ExperimentKit

/// The probe train/validation split rule is a CROSS-ENGINE CONTRACT
/// (2026-07-13): untagged examples are sorted ascending by the SHA-256 hex
/// of their text; every 5th of sorted order (0-based index % 5 == 4) is
/// validation. The fixture below is shared with the Python engine's test —
/// the orchestrator cross-checks that both produce the same membership.
struct ProbeSplitContractTests {

    func example(_ text: String, split: String? = nil, expresses: Bool = true)
        -> ConceptBuilder.ProbeExample
    {
        ConceptBuilder.ProbeExample(
            id: text, text: text, expresses: expresses, topic: nil,
            split: split, notes: nil)
    }

    @Test func sharedFixtureValidationMembership() {
        // Cross-engine fixture: sorted by sha256(text) hex the order is
        // night market, burning bridge, the court affirms, storm at sea,
        // frozen river, green meadow, quiet library — index 4 = frozen river.
        let texts = [
            "the court affirms", "storm at sea", "quiet library",
            "burning bridge", "green meadow", "night market", "frozen river",
        ]
        let split = ConceptBuilder.probeTrainValidationSplit(texts.map { example($0) })
        #expect(split.validation.map(\.text) == ["frozen river"])
        #expect(
            Set(split.train.map(\.text))
                == Set(texts).subtracting(["frozen river"]))
        #expect(split.train.count == 6)
    }

    @Test func splitIsContentDerivedNotPositional() {
        // Shuffled input order must produce the SAME membership (the
        // historical positional rule diverged with row order).
        let texts = [
            "green meadow", "frozen river", "night market", "storm at sea",
            "the court affirms", "burning bridge", "quiet library",
        ]
        let split = ConceptBuilder.probeTrainValidationSplit(texts.map { example($0) })
        #expect(split.validation.map(\.text) == ["frozen river"])
    }

    @Test func smallPoolsGetNoHoldout() {
        // Fewer than 5 untagged items: the rule applies literally — no
        // index reaches 4, so there is NO holdout (never force one).
        let split = ConceptBuilder.probeTrainValidationSplit(
            ["alpha", "bravo", "charlie", "delta"].map { example($0) })
        #expect(split.validation.isEmpty)
        #expect(split.train.count == 4)
    }

    @Test func explicitTagsStillWinWhenUsable() {
        // A usable explicit split (build has both classes, validation
        // non-empty) is honored verbatim — the contract rule is only the
        // untagged fallback.
        let examples = [
            example("pos build", split: "build", expresses: true),
            example("neg build", split: "build", expresses: false),
            example("held out", split: "validation"),
            example("untagged drifter"),
        ]
        let split = ConceptBuilder.probeTrainValidationSplit(examples)
        #expect(split.validation.map(\.text) == ["held out"])
        #expect(Set(split.train.map(\.text)) == ["pos build", "neg build"])
    }

    @Test func fallbackPreservesExplicitTagsAndSplitsUntagged() {
        // Unusable explicit split (no explicit validation): tagged items
        // keep their tags; the 7 untagged follow the contract rule.
        let untagged = [
            "the court affirms", "storm at sea", "quiet library",
            "burning bridge", "green meadow", "night market", "frozen river",
        ]
        let examples =
            [
                example("pos build", split: "build", expresses: true),
                example("neg build", split: "build", expresses: false),
            ] + untagged.map { example($0) }
        let split = ConceptBuilder.probeTrainValidationSplit(examples)
        #expect(split.validation.map(\.text) == ["frozen river"])
        #expect(split.train.map(\.text).contains("pos build"))
        #expect(split.train.map(\.text).contains("neg build"))
        #expect(split.train.count == 8)
    }
}
