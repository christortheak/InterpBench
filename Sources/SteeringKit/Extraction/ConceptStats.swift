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

    /// Deterministic ~20% holdout: every fifth stimulus, so repeated
    /// rebuilds hold out the same items and the headline stat is stable.
    static func isHeldOut(_ index: Int) -> Bool { index % 5 == 4 }

    public struct HeldOut: Sendable, Equatable {
        public let accuracy: Float
        public let testCount: Int
    }

    /// Builds the direction from training stimuli only, then classifies the
    /// held-out stimuli by their projection relative to the midpoint of the
    /// training class means. Returns nil when there is too little data.
    public static func heldOutAccuracy(
        positive: [[Float]], negative: [[Float]],
        method: ExtractionMethod = .meanDifference
    ) -> HeldOut? {
        let posTrain = positive.indices.filter { !isHeldOut($0) }.map { positive[$0] }
        let posTest = positive.indices.filter { isHeldOut($0) }.map { positive[$0] }
        let negTrain = negative.indices.filter { !isHeldOut($0) }.map { negative[$0] }
        let negTest = negative.indices.filter { isHeldOut($0) }.map { negative[$0] }

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

    /// Cosine between directions extracted from disjoint halves (even vs
    /// odd indices). High = the stimuli agree on what the concept is.
    public static func splitHalfCosine(
        positive: [[Float]], negative: [[Float]],
        method: ExtractionMethod = .meanDifference
    ) -> Float? {
        func halves(_ rows: [[Float]]) -> ([[Float]], [[Float]]) {
            (
                rows.indices.filter { $0.isMultiple(of: 2) }.map { rows[$0] },
                rows.indices.filter { !$0.isMultiple(of: 2) }.map { rows[$0] }
            )
        }
        let (pos0, pos1) = halves(positive)
        let (neg0, neg1) = halves(negative)
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
