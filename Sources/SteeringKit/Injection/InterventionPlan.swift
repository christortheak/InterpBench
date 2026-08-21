import Foundation

/// Turns a condition's declared edits into the intervention chain, and is the
/// ONLY place allowed to build one.
///
/// Steering used to be assembled inline at each call site — `injections.map {
/// VectorInjector(...) }` — which was safe because additions commute, so no
/// site could order them wrongly. Ablation removes that safety: it must read
/// the block's unmodified output `h₀`, and in a sequential chain that means
/// running first, and all of a layer's ablated directions must be removed as
/// ONE subspace or their shared component is subtracted twice.
///
/// Both requirements are satisfied by the same construction — a single
/// ablator at the head of the chain — so they are built here once rather than
/// documented at five call sites and eventually gotten wrong at one of them.
public enum InterventionPlan {

    /// How an edit acts on the residual stream.
    public enum Mode: String, Sendable, Codable, CaseIterable {
        /// `h + α·v` — a fixed offset, whatever is present.
        case add
        /// `h − λ·(h·v̂)v̂` — removes what is present.
        case ablate
    }

    /// One declared edit, with its vector already resolved.
    public struct Edit: Sendable {
        public let layer: Int
        public let vector: [Float]
        /// α for `.add`, λ for `.ablate`.
        public let strength: Float
        public let mode: Mode
        /// Names the concept. Used ONLY to order the ablation basis: modified
        /// Gram-Schmidt depends on input order, so a stable, data-derived
        /// order is what makes the removed subspace reproducible across runs
        /// and engines rather than a function of dictionary iteration.
        public let concept: String

        public init(
            layer: Int, vector: [Float], strength: Float, mode: Mode,
            concept: String
        ) {
            self.layer = layer
            self.vector = vector
            self.strength = strength
            self.mode = mode
            self.concept = concept
        }
    }

    public struct AmbiguousStrength: Error, CustomStringConvertible {
        public let layer: Int
        public let strengths: [Float]
        public var description: String {
            "layer \(layer) declares ablations with different strengths "
                + "(\(strengths.map { "\($0)" }.joined(separator: ", "))) — λ "
                + "scales the removal of a SUBSPACE, and once the directions "
                + "are orthonormalized its rows no longer correspond to the "
                + "concepts that produced them, so there is no defensible way "
                + "to apply two. Use one λ per layer"
        }
    }

    /// The chain, ablator first.
    ///
    /// `promptTokenCount` gates the injectors' chunked-prefill behaviour and
    /// is deliberately not passed to the ablator, which applies at every
    /// position.
    ///
    /// Additions keep exactly their historical shape: one `VectorInjector`
    /// per edit, which composes additively, so a condition with no ablations
    /// produces byte-identical behaviour to before this type existed.
    public static func interventions(
        _ edits: [Edit], promptTokenCount: Int? = nil
    ) throws -> [any LayerIntervention] {
        var chain: [any LayerIntervention] = []
        if let ablator = try ablator(edits) { chain.append(ablator) }
        for edit in edits where edit.mode == .add {
            chain.append(
                VectorInjector(
                    layer: edit.layer, vector: edit.vector,
                    alpha: edit.strength, promptTokenCount: promptTokenCount))
        }
        return chain
    }

    /// The single ablator covering every ablated layer, or nil when the
    /// condition ablates nothing.
    public static func ablator(_ edits: [Edit]) throws -> SubspaceAblator? {
        let ablations = edits.filter { $0.mode == .ablate }
        guard !ablations.isEmpty else { return nil }

        var perLayer: [Int: SubspaceAblator.Ablation] = [:]
        for (layer, group) in Dictionary(grouping: ablations, by: \.layer) {
            let strengths = Set(group.map(\.strength))
            guard strengths.count == 1, let strength = strengths.first else {
                throw AmbiguousStrength(
                    layer: layer, strengths: strengths.sorted())
            }
            // Sorted by concept: Gram-Schmidt is order-dependent, so the
            // basis must not depend on how the caller happened to build its
            // list. Ties (the same concept twice at a layer) are dropped by
            // the orthonormalizer as linearly dependent.
            let ordered = group.sorted { $0.concept < $1.concept }
            let basis = SubspaceAblator.orthonormalized(ordered.map(\.vector))
            guard !basis.isEmpty else { continue }
            perLayer[layer] = .init(basis: basis, strength: strength)
        }
        return perLayer.isEmpty ? nil : SubspaceAblator(ablations: perLayer)
    }

    /// Whether a chain satisfies the invariant: at most one ablator, and if
    /// present it is first. Exposed so tests can assert it of chains built
    /// anywhere, not only of this builder's output.
    public static func satisfiesOrderingInvariant(
        _ chain: [any LayerIntervention]
    ) -> Bool {
        let ablatorPositions = chain.indices.filter {
            chain[$0] is SubspaceAblator
        }
        if ablatorPositions.isEmpty { return true }
        return ablatorPositions == [0]
    }
}
