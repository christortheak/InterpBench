import CryptoKit
import Foundation

/// Statistics for a concept under construction, computed over captured
/// last-token activations at a single layer. Pure CPU functions —
/// concept-agnostic and unit-testable against fixtures.
///
/// These are stimulus-design aids. The headline number (held-out accuracy)
/// is computed on stimuli excluded from the direction, so optimizing it is
/// not pure in-sample polishing — but none of these certify that the
/// direction means what its label claims. Convergent/discriminant
/// validation before experimental use remains the gate (CLAUDE.md ›
/// Steering method).
public enum ConceptStats {

    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Minimum stimuli per class before either split-based diagnostic is
    /// computed. One rule on both engines (server twin
    /// `concept_stats.MINIMUM_ROWS_PER_CLASS`): below this a ~20% holdout is
    /// one or two rows and the "accuracy" it reports is noise wearing a
    /// percentage sign.
    public static let minimumRowsPerClass = 6
    public static let minimumRowsPerClassSplitHalf = 4

    static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// Row indices sorted ascending by the lowercase SHA-256 hex of the row's
    /// UTF-8 text (ties broken by the text itself).
    ///
    /// CROSS-ENGINE SPLIT CONTRACT (2026-08-28 audit, F3). Server twin:
    /// `concept_stats.content_hash_order`. The same rule already governs the
    /// reading-probe validation split (`ConceptBuilder.splitExamples` /
    /// `probes._content_hash_split`, 2026-07-13); this brings the concept
    /// screening diagnostics under it.
    ///
    /// **Why content-derived, not index-derived.** The old rules were
    /// positional: this engine held out `index % 5 == 4` and split even/odd
    /// halves, while the server held out the LAST ~20% in FILE order. Both make
    /// the statistic a function of how the stimulus file happens to be ORDERED,
    /// and the two engines reported different held-out numbers on identical
    /// data. Stimulus files are routinely authored in topic blocks, so the
    /// file-order tail is a single topic cluster (pessimistic held-out
    /// accuracy) while a parity split puts adjacent near-duplicates on both
    /// sides of the halves (optimistic split-half cosine) — and neither shows
    /// any variance, because the split is deterministic. Sorting by a hash of
    /// the row's own CONTENT makes membership independent of row order: shuffle
    /// the file and every number is unchanged.
    ///
    /// **Why a hash ORDER rather than a hash MODULUS.** `hash % 5 == 4` would
    /// also be content-derived, but the test-set SIZE would then be binomial
    /// rather than exactly ~20%, so the reported `testCount` would wander with
    /// the concept. Sorting and taking every fifth of sorted order keeps the
    /// exact proportions the positional rules had while throwing away the order.
    ///
    /// **Why no RNG.** A seeded shuffle would need the two engines to share a
    /// generator; sibling paths in this repo either share SplitMix64 or
    /// explicitly document the divergence. A content hash needs neither —
    /// SHA-256 of UTF-8 is the same on every platform, so there is no RNG left
    /// to diverge.
    public static func contentHashOrder(_ texts: [String]) -> [Int] {
        let keyed = texts.enumerated().map { (index: $0.offset, key: sha256Hex($0.element),
                                              text: $0.element) }
        return keyed.sorted {
            $0.key == $1.key ? $0.text < $1.text : $0.key < $1.key
        }.map(\.index)
    }

    /// The ~20% test membership: every 5th row of `contentHashOrder`
    /// (0-based sorted position % 5 == 4). Server twin `held_out_indices`.
    public static func heldOutIndices(_ texts: [String]) -> Set<Int> {
        Set(contentHashOrder(texts).enumerated().filter { $0.offset % 5 == 4 }
            .map(\.element))
    }

    /// The SECOND half's membership: odd positions of `contentHashOrder`.
    /// Server twin `split_half_indices`.
    public static func splitHalfSecondIndices(_ texts: [String]) -> Set<Int> {
        Set(contentHashOrder(texts).enumerated().filter { !$0.offset.isMultiple(of: 2) }
            .map(\.element))
    }

    /// `(outside, inside)` — rows whose index is not in `members` first.
    private static func partition(
        _ rows: [[Float]], _ members: Set<Int>
    ) -> ([[Float]], [[Float]]) {
        (
            rows.indices.filter { !members.contains($0) }.map { rows[$0] },
            rows.indices.filter { members.contains($0) }.map { rows[$0] }
        )
    }

    public struct HeldOut: Sendable, Equatable {
        public let accuracy: Float
        public let testCount: Int
    }

    /// Builds the direction from training stimuli only, then classifies the
    /// held-out stimuli by their projection relative to the midpoint of the
    /// training class means. Returns nil when there is too little data.
    ///
    /// The split is `heldOutIndices` per class (stratified), so it depends on
    /// the stimulus TEXTS and never on their order in the file — which is why
    /// the texts are required arguments rather than an option.
    ///
    /// Texts that do not line up with the activation rows yield nil, since this
    /// API is non-throwing by design and every other insufficiency returns nil
    /// here too. The server, whose `held_out_accuracy` can raise, raises a
    /// `ValueError` on the same input instead — the SPLIT is the cross-engine
    /// contract; how each engine reports a caller's misuse is not.
    public static func heldOutAccuracy(
        positive: [[Float]], negative: [[Float]],
        positiveTexts: [String], negativeTexts: [String],
        method: ExtractionMethod = .meanDifference
    ) -> HeldOut? {
        guard positive.count >= minimumRowsPerClass,
            negative.count >= minimumRowsPerClass,
            positiveTexts.count == positive.count,
            negativeTexts.count == negative.count
        else { return nil }
        let (posTrain, posTest) = partition(positive, heldOutIndices(positiveTexts))
        let (negTrain, negTest) = partition(negative, heldOutIndices(negativeTexts))

        guard posTrain.count >= 2, negTrain.count >= 2, posTest.count + negTest.count >= 2,
            let direction = try? SteeringVectorMath.direction(
                positive: posTrain, negative: negTrain, method: method),
            let posMean = try? SteeringVectorMath.mean(posTrain),
            let negMean = try? SteeringVectorMath.mean(negTrain)
        else { return nil }

        let midpoint = (dot(direction, posMean) + dot(direction, negMean)) / 2
        var correct = 0
        for row in posTest where dot(direction, row) > midpoint { correct += 1 }
        for row in negTest where dot(direction, row) < midpoint { correct += 1 }
        let total = posTest.count + negTest.count
        return HeldOut(accuracy: Float(correct) / Float(total), testCount: total)
    }

    /// Cosine between directions extracted from the two content-hash halves
    /// (even vs odd positions of `contentHashOrder`, per class). High = the
    /// stimuli agree on what the concept is. Content-derived, so authoring the
    /// file in topic blocks no longer splits adjacent near-duplicates across
    /// the halves and flatters the number.
    public static func splitHalfCosine(
        positive: [[Float]], negative: [[Float]],
        positiveTexts: [String], negativeTexts: [String],
        method: ExtractionMethod = .meanDifference
    ) -> Float? {
        guard positive.count >= minimumRowsPerClassSplitHalf,
            negative.count >= minimumRowsPerClassSplitHalf,
            positiveTexts.count == positive.count,
            negativeTexts.count == negative.count
        else { return nil }
        let (pos0, pos1) = partition(positive, splitHalfSecondIndices(positiveTexts))
        let (neg0, neg1) = partition(negative, splitHalfSecondIndices(negativeTexts))
        guard pos0.count >= 2, pos1.count >= 2, neg0.count >= 2, neg1.count >= 2,
            let v0 = try? SteeringVectorMath.direction(
                positive: pos0, negative: neg0, method: method),
            let v1 = try? SteeringVectorMath.direction(
                positive: pos1, negative: neg1, method: method),
            let cosine = try? SteeringVectorMath.cosineSimilarity(v0, v1)
        else { return nil }
        return cosine
    }

    public struct Outlier: Sendable, Identifiable, Equatable {
        public let index: Int
        public let isPositive: Bool
        /// Signed margin from the decision midpoint along the unit
        /// direction — negative means the stimulus sits on the wrong side.
        public let margin: Float
        public var id: String { "\(isPositive ? "pos" : "neg")-\(index)" }
    }

    /// The stimuli least aligned with the current direction, worst first —
    /// candidates for rewriting or pruning.
    public static func outliers(
        direction: [Float], positive: [[Float]], negative: [[Float]], count: Int = 3
    ) -> [Outlier] {
        let norm = SteeringVectorMath.l2Norm(direction)
        guard norm > 0,
            let posMean = try? SteeringVectorMath.mean(positive),
            let negMean = try? SteeringVectorMath.mean(negative)
        else { return [] }

        let unit = direction.map { $0 / norm }
        let midpoint = (dot(unit, posMean) + dot(unit, negMean)) / 2

        var all: [Outlier] = []
        for (index, row) in positive.enumerated() {
            all.append(
                Outlier(index: index, isPositive: true, margin: dot(unit, row) - midpoint))
        }
        for (index, row) in negative.enumerated() {
            all.append(
                Outlier(index: index, isPositive: false, margin: midpoint - dot(unit, row)))
        }
        return Array(all.sorted { $0.margin < $1.margin }.prefix(count))
    }

    /// Decision threshold for a grand-mean concept, which has no negative
    /// class: the population (every row of the multi-concept corpus) plays
    /// the reference role, so the midpoint sits between the concept-mean and
    /// population-mean projections — the exact analogue of the paired
    /// midpoint rule.
    public static func grandMeanMidpoint(
        direction: [Float], concept: [[Float]], population: [[Float]]
    ) -> Float? {
        guard let conceptMean = try? SteeringVectorMath.mean(concept),
            let populationMean = try? SteeringVectorMath.mean(population)
        else { return nil }
        return (dot(direction, conceptMean) + dot(direction, populationMean)) / 2
    }

    /// Held-out accuracy for a grand-mean concept on labeled scenarios
    /// (ports the server's `scenario_accuracy_grand_mean` exactly): a
    /// scenario is correct when `(projection > midpoint) == expresses`.
    public static func scenarioAccuracyGrandMean(
        direction: [Float], concept: [[Float]], population: [[Float]],
        scenarios: [[Float]], labels: [Bool]
    ) -> Float? {
        guard !scenarios.isEmpty,
            let midpoint = grandMeanMidpoint(
                direction: direction, concept: concept, population: population)
        else { return nil }
        let correct = zip(scenarios, labels).count { activation, label in
            (dot(direction, activation) > midpoint) == label
        }
        return Float(correct) / Float(scenarios.count)
    }

    /// Unlabeled fallback: the fraction of scenarios projecting above the
    /// same concept-vs-population midpoint (no accuracy claim possible).
    public static func fractionAboveMidpoint(
        direction: [Float], concept: [[Float]], population: [[Float]],
        scenarios: [[Float]]
    ) -> Float? {
        guard !scenarios.isEmpty,
            let midpoint = grandMeanMidpoint(
                direction: direction, concept: concept, population: population)
        else { return nil }
        let above = scenarios.count { dot(direction, $0) > midpoint }
        return Float(above) / Float(scenarios.count)
    }
}
