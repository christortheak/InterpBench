import Foundation
import MLX
import Testing

@testable import SteeringKit

/// Ablation is a projection, and the properties that make it one are the
/// point — not the line count.
///
/// These run engine-pure (no model, no Metal): the ablator is vector math over
/// small arrays, so every claim below is checkable in milliseconds.
struct SubspaceAblatorTests {

    // MARK: helpers

    private func vector(_ values: [Float]) -> [Float] { values }

    private func hidden(_ rows: [[Float]]) -> MLXArray {
        // [1, seq, hidden]
        MLXArray(rows.flatMap { $0 }, [1, rows.count, rows[0].count])
    }

    private func rows(_ h: MLXArray) -> [[Float]] {
        let seq = h.dim(1), width = h.dim(2)
        let flat = h.asType(.float32).asArray(Float.self)
        return (0..<seq).map { Array(flat[($0 * width)..<(($0 + 1) * width)]) }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private func norm(_ a: [Float]) -> Float { dot(a, a).squareRoot() }

    // MARK: orthonormalization

    @Test func theBasisIsOrthonormal() {
        let basis = SubspaceAblator.orthonormalized([
            [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1],
        ])
        #expect(basis.count == 3)
        for row in basis {
            #expect(abs(norm(row) - 1) < 1e-5)
        }
        for i in basis.indices {
            for j in basis.indices where j > i {
                #expect(abs(dot(basis[i], basis[j])) < 1e-5)
            }
        }
    }

    /// A direction already spanned by the earlier ones is DROPPED rather than
    /// normalized from numerical noise — the rank falls, which is the honest
    /// report of what was asked for.
    @Test func dependentDirectionsAreDroppedNotNormalizedFromNoise() {
        let basis = SubspaceAblator.orthonormalized([
            [1, 0, 0], [2, 0, 0], [0, 1, 0],
        ])
        #expect(basis.count == 2)
        #expect(abs(norm(basis[0]) - 1) < 1e-5)
        #expect(abs(norm(basis[1]) - 1) < 1e-5)
    }

    @Test func zeroDirectionsProduceAnEmptyBasisRatherThanDividingByZero() {
        #expect(SubspaceAblator.orthonormalized([[0, 0, 0]]).isEmpty)
        #expect(SubspaceAblator.orthonormalized([]).isEmpty)
    }

    // MARK: the three defining properties

    /// λ=1 leaves the residual ORTHOGONAL to the ablated direction.
    @Test func fullAblationZeroesTheProjection() {
        let v: [Float] = [3, 4, 0]
        let ablator = SubspaceAblator(layers: [0], vector: v)
        let h = hidden([[1, 2, 3], [-5, 0.5, 2]])
        let out = rows(ablator.apply(h, layer: 0, offset: 0))
        let unit = v.map { $0 / norm(v) }
        for row in out {
            #expect(abs(dot(row, unit)) < 1e-5)
        }
    }

    /// ‖h'‖² = ‖h‖² − c²: the norm falls by exactly the removed part, not by
    /// the vector's own norm (which is what a fixed −α·v would cost).
    @Test func theNormFallsByExactlyTheRemovedComponent() {
        let v: [Float] = [1, 1, 0]
        let ablator = SubspaceAblator(layers: [0], vector: v)
        let input: [Float] = [2, -1, 4]
        let out = rows(ablator.apply(hidden([input]), layer: 0, offset: 0))[0]
        let unit = v.map { $0 / norm(v) }
        let c = dot(input, unit)
        let expected = (dot(input, input) - c * c).squareRoot()
        #expect(abs(norm(out) - expected) < 1e-4)
    }

    @Test func fullAblationIsIdempotent() {
        let ablator = SubspaceAblator(layers: [0], vector: [1, 2, 3])
        let h = hidden([[4, -1, 0.5]])
        let once = ablator.apply(h, layer: 0, offset: 0)
        let twice = ablator.apply(once, layer: 0, offset: 0)
        for (a, b) in zip(rows(once)[0], rows(twice)[0]) {
            #expect(abs(a - b) < 1e-5)
        }
    }

    /// λ=2 reflects the component through the hyperplane: the projection
    /// flips sign and the NORM IS PRESERVED — the well-behaved "reverse the
    /// concept" that a large negative α cannot give, since its norm change is
    /// uncontrolled.
    @Test func lambdaTwoReflectsAndPreservesTheNorm() {
        let v: [Float] = [0, 1, 0]
        let ablator = SubspaceAblator(layers: [0], vector: v, strength: 2)
        let input: [Float] = [3, 5, -1]
        let out = rows(ablator.apply(hidden([input]), layer: 0, offset: 0))[0]
        #expect(abs(norm(out) - norm(input)) < 1e-4)
        let unit = v.map { $0 / norm(v) }
        #expect(abs(dot(out, unit) + dot(input, unit)) < 1e-4)
    }

    @Test func partialAblationRemovesPartOfTheComponent() {
        let v: [Float] = [1, 0, 0]
        let ablator = SubspaceAblator(layers: [0], vector: v, strength: 0.25)
        let out = rows(
            ablator.apply(hidden([[4, 1, 1]]), layer: 0, offset: 0))[0]
        #expect(abs(out[0] - 3) < 1e-4)  // 4 − 0.25·4
        #expect(abs(out[1] - 1) < 1e-5)
    }

    // MARK: the subspace invariant

    /// THE reason all directions at a layer share one ablator.
    ///
    /// Subtracting two correlated directions' projections separately
    /// double-counts what they share: neither ends up removed, and the shared
    /// component is driven NEGATIVE by an uncontrolled amount — silent
    /// negative steering behind a control labelled "ablation". Ablating them
    /// as a subspace zeroes both.
    @Test func correlatedDirectionsAreRemovedJointlyNotTwice() {
        // cos ≈ 0.98 — the regime real concept directions sit in (the
        // project's own depth-drift cosines reach 0.84).
        let v: [Float] = [1, 0, 0]
        let w: [Float] = [1, 0.2, 0]
        let input: [Float] = [1, 1, 1]

        // Wrong: two independent projections off the SAME input.
        let unitV = v.map { $0 / norm(v) }
        let unitW = w.map { $0 / norm(w) }
        let cV = dot(input, unitV), cW = dot(input, unitW)
        var doubled = input
        for index in doubled.indices {
            doubled[index] -= cV * unitV[index] + cW * unitW[index]
        }
        #expect(
            dot(doubled, unitV) < -0.1,
            "the double-subtraction should overshoot into negative steering — if it does not, this test no longer demonstrates the hazard")

        // Right: one ablator over the orthonormalized span.
        let basis = SubspaceAblator.orthonormalized([v, w])
        #expect(basis.count == 2)
        let ablator = SubspaceAblator(
            ablations: [0: .init(basis: basis, strength: 1)])
        let out = rows(
            ablator.apply(hidden([input]), layer: 0, offset: 0))[0]
        #expect(abs(dot(out, unitV)) < 1e-5)
        #expect(abs(dot(out, unitW)) < 1e-5)
    }

    /// Order-independence, the property that lets the ablator live in a
    /// sequential chain: with the ablator FIRST, the per-layer result is
    /// `h₀ − λ·P h₀ + Σαᵢvᵢ` however the injectors are ordered.
    @Test func anAblatorFirstMakesTheChainOrderIndependent() {
        let ablator = SubspaceAblator(layers: [0], vector: [1, 0, 0])
        let first = VectorInjector(layer: 0, vector: [0, 1, 0], alpha: 2)
        let second = VectorInjector(layer: 0, vector: [0, 0, 1], alpha: 3)
        let h = hidden([[5, 1, 1]])

        func chain(_ interventions: [any LayerIntervention]) -> [Float] {
            var current = h
            for intervention in interventions {
                current = intervention.apply(current, layer: 0, offset: 0)
            }
            return rows(current)[0]
        }
        let a = chain([ablator, first, second])
        let b = chain([ablator, second, first])
        for (x, y) in zip(a, b) { #expect(abs(x - y) < 1e-5) }
        // And it is the stated formula: x ablated away, both alphas added.
        #expect(abs(a[0]) < 1e-5)
        #expect(abs(a[1] - 3) < 1e-5)  // 1 + 2
        #expect(abs(a[2] - 4) < 1e-5)  // 1 + 3
    }

    /// The documented semantics, made a test so it cannot drift into folklore:
    /// ablation removes what the MODEL produced, so an injected
    /// non-orthogonal vector reintroduces some of the ablated direction.
    @Test func aNonOrthogonalInjectionReintroducesTheAblatedDirection() {
        let ablator = SubspaceAblator(layers: [0], vector: [1, 0, 0])
        let injector = VectorInjector(layer: 0, vector: [1, 1, 0], alpha: 1)
        var current = hidden([[4, 0, 0]])
        current = ablator.apply(current, layer: 0, offset: 0)
        current = injector.apply(current, layer: 0, offset: 0)
        // 4 removed, then the injected vector's x-component put back.
        #expect(abs(rows(current)[0][0] - 1) < 1e-5)
    }

    // MARK: positions

    /// Ablation fires at EVERY position — the inverse of the injector's
    /// chunked-prefill gate, which deliberately leaves mid-prompt chunk tails
    /// untouched. Here the whole prompt must be stripped.
    @Test func everyPositionIsAblatedIncludingMidPromptChunks() {
        let ablator = SubspaceAblator(layers: [0], vector: [1, 0, 0])
        let h = hidden([[9, 1, 0], [8, 2, 0], [7, 3, 0]])
        // offset/seqLen that VectorInjector's gate would suppress entirely.
        let out = rows(ablator.apply(h, layer: 0, offset: 0))
        #expect(out.count == 3)
        for row in out { #expect(abs(row[0]) < 1e-5) }
        // The orthogonal content of every position survives untouched.
        #expect(abs(out[0][1] - 1) < 1e-5)
        #expect(abs(out[1][1] - 2) < 1e-5)
        #expect(abs(out[2][1] - 3) < 1e-5)
    }

    @Test func unconfiguredLayersAreUntouched() {
        let ablator = SubspaceAblator(layers: [3], vector: [1, 0, 0])
        let h = hidden([[9, 1, 0]])
        let out = rows(ablator.apply(h, layer: 0, offset: 0))[0]
        #expect(out == [9, 1, 0])
    }
}
