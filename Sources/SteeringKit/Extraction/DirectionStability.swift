import Foundation

/// Resampling diagnoses sensitivity to rows/order, not concept validity or
/// behavioral efficacy. Arithmetic follows vector_math.direction_stability.
public struct DirectionStability: Codable, Sendable {
    public struct DegenerateDraw: Codable, Sendable {
        public let kind: String
        public let reason: String
        public let seed: UInt64
    }
    public let method: String
    public let pairCount: Int
    public let subsampleSize: Int
    public let paired: Bool
    public let resamples: Int
    public let fraction: Double
    public let seed: UInt64
    public let orderShuffles: Int
    public let resampleSeeds: [UInt64]
    public let resampleCosines: [Double]
    public let orderShuffleSeeds: [UInt64]
    public let orderShuffleCosines: [Double]
    public let degenerateDraws: [DegenerateDraw]
    public let signFlips: Int
    public let minCosine: Double
    public let meanCosine: Double
    public let medianCosine: Double
    public let percentile5Cosine: Double
}

extension SteeringVectorMath {
    public struct StabilityError: Error, CustomStringConvertible {
        public let description: String
        public init(_ description: String) { self.description = description }
    }

    public static func resolvedSubsampleSize(rowCount: Int, fraction: Double) -> Int {
        Int(floor(fraction * Double(rowCount) + 0.5))
    }

    /// Cheap admission is shared by the public math function and the CLI before
    /// loading a model. Counts are not clamped into a different diagnostic.
    public static func checkStabilityRequest(method: ExtractionMethod, positiveCount: Int,
        negativeCount: Int, resamples: Int, fraction: Double) throws {
        guard method.isPaired || method == .designatedReference else {
            throw StabilityError("\(method.rawValue) does not compare two row populations. Use meanDifference, lat, or designatedReference for this diagnostic.")
        }
        guard resamples >= 2 else { throw StabilityError("A stability summary needs at least 2 draws; raise --resamples.") }
        guard fraction > 0, fraction <= 1 else { throw StabilityError("Pass --fraction in (0, 1].") }
        guard positiveCount > 0, negativeCount > 0 else { throw SteeringVectorError.emptyInput }
        if method == .pairedDifferencePCA, positiveCount != negativeCount {
            throw SteeringVectorError.unpairedStimuli(positive: positiveCount, negative: negativeCount)
        }
        guard resolvedSubsampleSize(rowCount: min(positiveCount, negativeCount), fraction: fraction) >= 2 else {
            throw StabilityError("A draw needs at least 2 rows; raise --fraction or supply more stimuli.")
        }
    }

    public static func stabilityShuffledOrder(count: Int, seed: UInt64) -> [Int] {
        guard count > 1 else { return Array(0 ..< max(0, count)) }
        var random = SplitMix64(seed: seed)
        var order = Array(0 ..< count)
        for position in 0 ..< count - 1 {
            order.swapAt(position, position + Int(random.next() % UInt64(count - position)))
        }
        return order
    }

    public static func directionStability(positive: [[Float]], negative: [[Float]],
        method: ExtractionMethod, resamples: Int, fraction: Double, seed: UInt64,
        orderShuffles: Int = 0) throws -> DirectionStability {
        try checkStabilityRequest(method: method, positiveCount: positive.count,
            negativeCount: negative.count, resamples: resamples, fraction: fraction)
        let full = try direction(positive: positive, negative: negative, method: method)
        guard l2Norm(full) > 0 else { throw StabilityError("The full-data direction has zero norm; there is no direction to compare draws against.") }
        let paired = positive.count == negative.count
        let count = min(positive.count, negative.count)
        let size = resolvedSubsampleSize(rowCount: count, fraction: fraction)
        var random = SplitMix64(seed: seed)
        let seeds = (0 ..< resamples).map { _ in random.next() }
        let shuffleSeeds = (0 ..< max(0, orderShuffles)).map { _ in random.next() }
        var cosines: [Double] = [], shuffleCosines: [Double] = []
        var degenerate: [DirectionStability.DegenerateDraw] = []
        func next(_ seed: UInt64) -> UInt64 {
            var random = SplitMix64(seed: seed)
            return random.next()
        }
        func record(_ kind: String, _ drawSeed: UInt64, _ p: [[Float]], _ n: [[Float]], _ sink: inout [Double]) {
            do {
                let candidate = try direction(positive: p, negative: n, method: method)
                sink.append(Double(try cosineSimilarity(candidate, full)))
            } catch {
                degenerate.append(.init(kind: kind, reason: String(describing: error), seed: drawSeed))
            }
        }
        for drawSeed in seeds {
            let p = TokenBankDownsampler.selectedIndices(count: positive.count,
                cap: resolvedSubsampleSize(rowCount: positive.count, fraction: fraction), seed: drawSeed)
            let n = paired ? p : TokenBankDownsampler.selectedIndices(count: negative.count,
                cap: resolvedSubsampleSize(rowCount: negative.count, fraction: fraction), seed: next(drawSeed))
            record("resample", drawSeed, p.map { positive[$0] }, n.map { negative[$0] }, &cosines)
        }
        for drawSeed in shuffleSeeds {
            let p = stabilityShuffledOrder(count: positive.count, seed: drawSeed)
            let n = paired ? p : stabilityShuffledOrder(count: negative.count, seed: next(drawSeed))
            record("orderShuffle", drawSeed, p.map { positive[$0] }, n.map { negative[$0] }, &shuffleCosines)
        }
        guard cosines.count >= 2 else { throw StabilityError("Only \(cosines.count) of \(resamples) draws produced a direction; there is no distribution to summarize. Check for degenerate stimuli.") }
        let sorted = cosines.sorted()
        func percentile(_ fraction: Double) -> Double {
            let position = Double(sorted.count - 1) * fraction
            let lower = Int(floor(position)), upper = Int(ceil(position))
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
        }
        return .init(method: method.rawValue, pairCount: count, subsampleSize: size, paired: paired,
            resamples: resamples, fraction: fraction, seed: seed, orderShuffles: max(0, orderShuffles),
            resampleSeeds: seeds, resampleCosines: cosines, orderShuffleSeeds: shuffleSeeds,
            orderShuffleCosines: shuffleCosines, degenerateDraws: degenerate,
            signFlips: (cosines + shuffleCosines).filter { $0 < 0 }.count,
            minCosine: sorted[0], meanCosine: cosines.reduce(0, +) / Double(cosines.count),
            medianCosine: percentile(0.5), percentile5Cosine: percentile(0.05))
    }

    public static func stabilityByLayer(_ rows: [Int: (positive: [[Float]], negative: [[Float]])],
        method: ExtractionMethod, resamples: Int, fraction: Double, seed: UInt64,
        orderShuffles: Int = 0) throws -> [Int: DirectionStability] {
        var result: [Int: DirectionStability] = [:]
        for layer in rows.keys.sorted() {
            let pair = rows[layer]!
            result[layer] = try directionStability(positive: pair.positive, negative: pair.negative,
                method: method, resamples: resamples, fraction: fraction, seed: seed, orderShuffles: orderShuffles)
        }
        return result
    }
}
