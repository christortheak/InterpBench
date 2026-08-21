import Foundation
import MLX
import Testing

@testable import SteeringKit

/// The builder is the only place a chain is assembled, because the ablator's
/// two requirements — read `h₀`, remove a layer's directions as one subspace —
/// are satisfied by one construction and violated by any call site that
/// improvises.
struct InterventionPlanTests {

    private func edit(
        _ concept: String, layer: Int = 0, _ vector: [Float],
        strength: Float = 1, mode: InterventionPlan.Mode
    ) -> InterventionPlan.Edit {
        .init(
            layer: layer, vector: vector, strength: strength, mode: mode,
            concept: concept)
    }

    private func hidden(_ row: [Float]) -> MLXArray {
        MLXArray(row, [1, 1, row.count])
    }

    private func out(_ chain: [any LayerIntervention], _ row: [Float]) -> [Float] {
        var current = hidden(row)
        for intervention in chain {
            current = intervention.apply(current, layer: 0, offset: 0)
        }
        return current.asType(.float32).asArray(Float.self)
    }

    @Test func aPureSteeringConditionIsUnchangedFromBefore() throws {
        let chain = try InterventionPlan.interventions([
            edit("a", [0, 1, 0], strength: 2, mode: .add),
            edit("b", [0, 0, 1], strength: 3, mode: .add),
        ])
        #expect(chain.count == 2)
        #expect(chain.allSatisfy { $0 is VectorInjector })
        #expect(InterventionPlan.satisfiesOrderingInvariant(chain))
    }

    /// The invariant: however many concepts are ablated, at how many layers,
    /// there is exactly ONE ablator and it is first.
    @Test func everyChainHasAtMostOneAblatorAndItIsFirst() throws {
        let chain = try InterventionPlan.interventions([
            edit("z", layer: 0, [1, 0, 0], mode: .ablate),
            edit("a", layer: 1, [0, 1, 0], mode: .ablate),
            edit("m", layer: 0, [0, 1, 0], mode: .ablate),
            edit("q", layer: 0, [0, 0, 1], strength: 2, mode: .add),
        ])
        #expect(chain.filter { $0 is SubspaceAblator }.count == 1)
        #expect(chain.first is SubspaceAblator)
        #expect(InterventionPlan.satisfiesOrderingInvariant(chain))
    }

    /// Two concepts at one layer become one rank-2 subspace, not two
    /// sequential removals — the double-subtraction hazard, closed by
    /// construction rather than by a comment.
    @Test func concentricAblationsAtALayerBecomeOneSubspace() throws {
        let ablator = try #require(
            try InterventionPlan.ablator([
                edit("fear", [1, 0, 0], mode: .ablate),
                edit("anger", [1, 0.2, 0], mode: .ablate),
            ]))
        #expect(ablator.ablation(at: 0)?.rank == 2)
        let result = out([ablator], [1, 1, 1])
        // Both directions removed, neither overshot.
        #expect(abs(result[0]) < 1e-5)
        #expect(abs(result[1]) < 1e-5)
        #expect(abs(result[2] - 1) < 1e-5)
    }

    /// The basis must not depend on the caller's list order, or two runs of
    /// the same condition would ablate subtly different subspaces.
    @Test func theBasisIsOrderedByConceptNotByCallerOrder() throws {
        let forward = try #require(
            try InterventionPlan.ablator([
                edit("anger", [1, 0.2, 0], mode: .ablate),
                edit("fear", [1, 0, 0], mode: .ablate),
            ]))
        let reversed = try #require(
            try InterventionPlan.ablator([
                edit("fear", [1, 0, 0], mode: .ablate),
                edit("anger", [1, 0.2, 0], mode: .ablate),
            ]))
        let a = try #require(forward.ablation(at: 0))
        let b = try #require(reversed.ablation(at: 0))
        #expect(a.rank == b.rank)
        for (rowA, rowB) in zip(a.basis, b.basis) {
            for (x, y) in zip(rowA, rowB) { #expect(abs(x - y) < 1e-6) }
        }
    }

    /// λ scales the removal of a subspace, whose orthonormalized rows no
    /// longer correspond to the concepts that produced them — so two λ at one
    /// layer has no defensible meaning and is refused rather than guessed.
    @Test func twoStrengthsAtOneLayerRefuse() {
        #expect(throws: InterventionPlan.AmbiguousStrength.self) {
            _ = try InterventionPlan.ablator([
                edit("a", [1, 0, 0], strength: 1, mode: .ablate),
                edit("b", [0, 1, 0], strength: 0.5, mode: .ablate),
            ])
        }
    }

    /// Different layers may of course differ.
    @Test func differentLayersMayCarryDifferentStrengths() throws {
        let ablator = try #require(
            try InterventionPlan.ablator([
                edit("a", layer: 0, [1, 0, 0], strength: 1, mode: .ablate),
                edit("b", layer: 1, [0, 1, 0], strength: 0.5, mode: .ablate),
            ]))
        #expect(ablator.ablation(at: 0)?.strength == 1)
        #expect(ablator.ablation(at: 1)?.strength == 0.5)
    }

    @Test func aConditionThatAblatesNothingBuildsNoAblator() throws {
        #expect(try InterventionPlan.ablator([]) == nil)
        #expect(
            try InterventionPlan.ablator([
                edit("a", [1, 0, 0], strength: 2, mode: .add)
            ]) == nil)
        // An all-zero vector cannot define a direction: no ablator rather
        // than a basis of noise.
        #expect(
            try InterventionPlan.ablator([
                edit("a", [0, 0, 0], mode: .ablate)
            ]) == nil)
    }

    /// The whole point of the ordering: the chain computes
    /// `h₀ − λ·P h₀ + Σαᵢvᵢ` regardless of how the injectors are listed.
    @Test func theBuiltChainIsOrderIndependent() throws {
        let edits: [InterventionPlan.Edit] = [
            edit("x", [1, 0, 0], mode: .ablate),
            edit("y", [0, 1, 0], strength: 2, mode: .add),
            edit("z", [0, 0, 1], strength: 3, mode: .add),
        ]
        let a = out(try InterventionPlan.interventions(edits), [5, 1, 1])
        let b = out(
            try InterventionPlan.interventions(edits.reversed()), [5, 1, 1])
        for (x, y) in zip(a, b) { #expect(abs(x - y) < 1e-5) }
        #expect(abs(a[0]) < 1e-5)
        #expect(abs(a[1] - 3) < 1e-5)
        #expect(abs(a[2] - 4) < 1e-5)
    }

    /// The same literal basis the server's `test_both_engines_build_the_same_basis`
    /// asserts. If the two orthonormalizations ever diverge, one side's test
    /// fails here rather than two studies quietly disagreeing about which
    /// subspace they removed.
    ///
    /// Note which direction seeds the basis: "anger" sorts before "fear", and
    /// that concept-name ordering is the only reason this value is stable at
    /// all — Gram-Schmidt would otherwise follow the caller's list.
    @Test func bothEnginesBuildTheSameBasis() throws {
        let ablator = try #require(
            try InterventionPlan.ablator([
                edit("fear", [1, 0, 0], mode: .ablate),
                edit("anger", [1, 0.2, 0], mode: .ablate),
            ]))
        let basis = try #require(ablator.ablation(at: 0)?.basis)
        let expected: [[Float]] = [
            [0.9805806756909201, 0.19611613513818402, 0.0],
            [0.19611613513818446, -0.98058067569092, 0.0],
        ]
        #expect(basis.count == expected.count)
        for (row, want) in zip(basis, expected) {
            for (x, y) in zip(row, want) { #expect(abs(x - y) < 1e-6) }
        }
    }

    /// A chain someone assembled by hand with the ablator late is caught.
    @Test func theInvariantCheckRejectsAMisorderedChain() {
        let ablator = SubspaceAblator(layers: [0], vector: [1, 0, 0])
        let injector = VectorInjector(layer: 0, vector: [0, 1, 0], alpha: 1)
        #expect(!InterventionPlan.satisfiesOrderingInvariant([injector, ablator]))
        #expect(
            !InterventionPlan.satisfiesOrderingInvariant([ablator, ablator]))
        #expect(InterventionPlan.satisfiesOrderingInvariant([ablator, injector]))
        #expect(InterventionPlan.satisfiesOrderingInvariant([injector]))
    }
}
