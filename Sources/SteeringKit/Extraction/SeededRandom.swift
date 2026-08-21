import Foundation

/// Deterministic RNG (SplitMix64) for reproducible random control vectors.
/// The matched-norm random-vector coherence control must be regenerable
/// from the seed recorded in the run config (CLAUDE.md › Experiment B).
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

extension SteeringVectorMath {
    /// Canonical identifier for the matched-norm random-control recipe,
    /// stamped wherever a random control is recorded (both engines emit the
    /// same string): i.i.d. standard-normal components (an isotropic Gaussian
    /// direction), rescaled to the target L2 norm. The server twin is
    /// `_matched_norm_random` in `Server/steerlab_server/experiment/tasks.py`.
    /// Byte-identical vectors across engines are NOT the contract (the RNGs
    /// differ per substrate); the DISTRIBUTION and this stamp are.
    ///
    /// Provenance rule for readers: a run whose random control carries no
    /// `randomVectorAlgorithm` stamp is legacy — on Swift that means
    /// cube-uniform-then-rescale (NOT isotropic; density concentrates toward
    /// cube corners), on the server it was already Gaussian.
    public static let randomVectorAlgorithm = "gaussian-isotropic-v1"

    /// A random direction of the given dimension scaled to `norm`, suitable
    /// as the coherence-control vector: `gaussian-isotropic-v1` — i.i.d.
    /// N(0, 1) components via Box–Muller over the seeded RNG (deterministic:
    /// same seed → same vector, forever — pinned by a golden test), then a
    /// single rescale to the target norm, mirroring the server's
    /// `_matched_norm_random` recipe (distribution and normalization order).
    public static func randomVector(
        dimension: Int, norm: Float, using rng: inout some RandomNumberGenerator
    ) throws -> [Float] {
        var raw: [Float] = []
        raw.reserveCapacity(dimension + 1)
        while raw.count < dimension {
            // 53-bit mantissa uniforms; u1 shifted into (0, 1] so log(u1) is
            // finite. No Foundation randomness — every draw comes from `rng`.
            let u1 = (Double(rng.next() >> 11) + 1) * 0x1p-53
            let u2 = Double(rng.next() >> 11) * 0x1p-53
            let radius = (-2 * Foundation.log(u1)).squareRoot()
            let angle = 2 * Double.pi * u2
            raw.append(Float(radius * Foundation.cos(angle)))
            raw.append(Float(radius * Foundation.sin(angle)))
        }
        if raw.count > dimension { raw.removeLast() }
        return try rescaled(raw, toNorm: norm)
    }
}
