import MLX

/// Removes concept directions from the residual stream by projection —
/// "the model cannot represent this here" — rather than pushing against them.
///
/// `h' = h − λ·P h`, where `P` projects onto the span of the ablated
/// directions. Contrast `VectorInjector`, which adds a FIXED offset `α·v`
/// regardless of what is present: negative steering overshoots wherever the
/// concept is weakly represented (driving the projection negative, i.e.
/// asserting the anti-concept) and undershoots wherever it is strong.
/// Ablation removes exactly what is there, so it needs no strength parameter
/// and no residual-norm denominator — `α`'s whole purpose is comparability,
/// and there is nothing here to make comparable.
///
/// Three properties follow, and the tests pin all three:
/// - `h'·v̂ = 0` for every ablated direction (λ = 1)
/// - `‖h'‖² = ‖h‖² − Σ cₖ²` — the norm falls by exactly the removed part
/// - idempotent at λ = 1; norm-PRESERVING at λ = 2, which reflects the
///   component through the hyperplane (a better-behaved "reverse the
///   concept" than a large negative α, since the norm is untouched)
///
/// ## Two invariants that are really one
///
/// **All directions at a layer live in ONE ablator.** Subtracting each
/// direction's projection separately double-counts whatever they share:
/// for non-orthogonal `v̂, ŵ` it leaves `h'·v̂ = −c_w(ŵ·v̂)`, so neither
/// direction is removed and correlated concepts get driven negative by an
/// uncontrolled amount. Directions are orthonormalized first (below) and
/// removed as a subspace.
///
/// **The ablator runs BEFORE any injector.** Every intervention's edit must
/// be computed from the block's unmodified output `h₀`, so the per-layer
/// result is the order-independent
///
///     h' = h₀ − λ·P h₀ + Σᵢ αᵢvᵢ
///
/// Interventions are applied by the vendored models as a sequential chain, so
/// "reads `h₀`" means "runs first". Pure additions commute among themselves,
/// so one ablator in front is sufficient — and it is also exactly what the
/// subspace invariant requires. A second ablator in the list breaks both at
/// once, which is why the builder emits one and a test asserts it.
///
/// The semantics this fixes on: ablation removes what the MODEL produced, not
/// what you injected. Steering a non-orthogonal vector therefore reintroduces
/// some of the ablated direction (`h'·v̂ = α(w·v̂)`), which is the intended
/// reading — the researcher injecting it knows what they are doing.
///
/// ## Positions
///
/// Applies at EVERY position, unlike injection. The claim being made is that
/// the direction is unavailable; letting the model read the whole prompt with
/// the concept intact and stripping it only while writing would be a
/// different, muddier intervention. This deliberately inverts the injector's
/// chunked-prefill gate — `VectorInjector.shouldInject` exists to keep
/// steering off mid-prompt chunk tails, and there is no analogous hazard here.
public struct SubspaceAblator: LayerIntervention {

    /// One layer's ablated subspace.
    public struct Ablation: Sendable {
        /// Orthonormal rows spanning the ablated subspace. Built by
        /// `orthonormalized(_:)`, never supplied raw.
        public let basis: [[Float]]
        /// λ. 1 = full ablation, (0,1) = partial, 2 = reflection.
        public let strength: Float

        public init(basis: [[Float]], strength: Float = 1) {
            self.basis = basis
            self.strength = strength
        }

        /// Rank of the removed subspace — below the number of directions
        /// supplied when some were linearly dependent on the others.
        public var rank: Int { basis.count }
    }

    private let ablations: [Int: Ablation]

    public init(ablations: [Int: Ablation]) {
        self.ablations = ablations
    }

    /// Convenience for a single direction at a set of layers — the common
    /// case (one concept, ablated across the network).
    public init(layers: [Int], vector: [Float], strength: Float = 1) {
        let basis = Self.orthonormalized([vector])
        var out: [Int: Ablation] = [:]
        for layer in layers {
            out[layer] = Ablation(basis: basis, strength: strength)
        }
        self.init(ablations: out)
    }

    /// The subspace removed at `layer`, if any.
    public func ablation(at layer: Int) -> Ablation? { ablations[layer] }

    // MARK: - Orthonormalization

    /// Directions below this relative norm after orthogonalization are
    /// already inside the span of the earlier ones and are dropped: keeping
    /// them would divide by a near-zero norm and produce a basis vector of
    /// numerical noise.
    public static let dependenceTolerance: Double = 1e-6

    /// Modified Gram-Schmidt over the supplied directions, in Double.
    ///
    /// Deterministic and engine-shared: the CALLER fixes the order (by
    /// concept name), the algorithm is modified — not classical — Gram-Schmidt
    /// for numerical stability, and accumulation is Double regardless of the
    /// residual stream's dtype. The Python twin does the identical thing, so
    /// the two engines agree to float precision instead of to whatever their
    /// reduction strategies happen to share.
    ///
    /// Zero and dependent directions are dropped, so the result may be
    /// shorter than the input; an all-zero input yields an empty basis, which
    /// makes `apply` a no-op rather than a division by zero.
    public static func orthonormalized(_ directions: [[Float]]) -> [[Float]] {
        var basis: [[Double]] = []
        for direction in directions {
            var residual = direction.map(Double.init)
            let originalNorm = norm(residual)
            guard originalNorm > 0 else { continue }
            // Modified: subtract each accepted basis vector's component from
            // the RUNNING residual, not from the original.
            for accepted in basis {
                let coefficient = dot(residual, accepted)
                for index in residual.indices {
                    residual[index] -= coefficient * accepted[index]
                }
            }
            let remaining = norm(residual)
            guard remaining > dependenceTolerance * originalNorm else {
                continue  // already spanned by the earlier directions
            }
            basis.append(residual.map { $0 / remaining })
        }
        return basis.map { $0.map(Float.init) }
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        var total = 0.0
        for index in a.indices { total += a[index] * b[index] }
        return total
    }

    private static func norm(_ a: [Double]) -> Double { dot(a, a).squareRoot() }

    // MARK: - LayerIntervention

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard let ablation = ablations[layer], !ablation.basis.isEmpty else {
            return h
        }
        // Float32 throughout: the residual stream is bf16/fp16, and a
        // reduction over `hidden` terms in bf16 drifts enough that the two
        // engines' differing reduction strategies would show up as a
        // different ablation for the same configuration. The cast back is at
        // the end, so only the stored result is narrowed.
        let h32 = h.asType(.float32)

        // Elementwise multiply + explicit sum, NOT `matmul`. MLX's matmul is
        // not float32-accurate for this shape: on float32 inputs whose exact
        // product is 2.2 × 0.6 = 1.32 it returned 1.3186722, and the implied
        // basis differed per row (0.599396 vs 0.599519) — reduced-precision
        // arithmetic inside the kernel, ~1e-3 relative. Ablation is defined by
        // `h'·v̂ = 0`, an exact cancellation, so a 1e-3 error in the removal is
        // a 1e-3 residue of the very direction being ablated, and it would
        // have differed from the server's (accurate) torch matmul for the
        // same configuration.
        //
        // Rank is 1–3 in practice, so looping the basis costs nothing.
        // Coefficients are read from `h32` — the block's unmodified output —
        // which is the order-independence rule, and is also why an orthonormal
        // basis is required: only then does removing the components one at a
        // time equal projecting out their span.
        var result = h32
        for row in ablation.basis {
            let unit = MLXArray(row).asType(.float32)
            let coefficient = (h32 * unit).sum(axis: -1, keepDims: true)
            result = result - coefficient * unit * ablation.strength
        }
        return result.asType(h.dtype)
    }
}
