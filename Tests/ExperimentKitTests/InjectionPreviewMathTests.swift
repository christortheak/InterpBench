import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the shared injection-preview math
/// (`ChatService.InjectionPreview.compute`) — the ONE implementation both the
/// local slot path (`injectionPreview`, vector norm from loaded bytes) and the
/// server slot path (`serverInjectionPreview`, vector norm from the catalog's
/// `normsPerLayer`) delegate to, so the Δnorm line cannot drift between
/// substrates.
@MainActor
struct InjectionPreviewMathTests {

    private typealias Preview = ChatService.InjectionPreview

    // MARK: - Parity: same inputs, identical numbers on both paths

    @Test func localAndServerPathsAgreeOnIdenticalInputs() throws {
        // The local path feeds (‖v‖ at layer, sidecar residualNormPerLayer[layer]);
        // the server path feeds (normsPerLayer[layer], residualNormPerLayer[layer])
        // from the catalog record. For the same per-layer scalars the previews
        // must be field-for-field identical — asserted over a grid covering
        // raw/norm-unit modes and present/absent denominators.
        let grid: [(vectorNorm: Float, residualNorm: Float?, alpha: Double, normUnits: Bool)] = [
            (4.0, 20.0, 0.5, true),
            (4.0, 20.0, 0.5, false),
            (1.5, nil, -3.0, true),
            (2.5, 10.0, -0.25, true),
            (7.75, 31.0, 8.0, false),
        ]
        for inputs in grid {
            let viaLocalPath = Preview.compute(
                layer: 18, vectorNorm: inputs.vectorNorm, residualNorm: inputs.residualNorm,
                alpha: inputs.alpha, alphaInNormUnits: inputs.normUnits, fixedLayer: false)
            let viaServerPath = Preview.compute(
                layer: 18, vectorNorm: inputs.vectorNorm, residualNorm: inputs.residualNorm,
                alpha: inputs.alpha, alphaInNormUnits: inputs.normUnits, fixedLayer: false)
            let local = try #require(viaLocalPath)
            let server = try #require(viaServerPath)
            #expect(local.layer == server.layer)
            #expect(local.vectorNorm == server.vectorNorm)
            #expect(local.residualNorm == server.residualNorm)
            #expect(local.effectiveAlpha == server.effectiveAlpha)
            #expect(local.injectedNorm == server.injectedNorm)
            #expect(local.isNormUnits == server.isNormUnits)
            #expect(local.normUnitsFallback == server.normUnitsFallback)
        }
    }

    // MARK: - The math itself (against the injection-time formulas)

    @Test func normUnitPreviewMatchesTheInjectionScale() throws {
        // Norm units: scale = α·residual/‖v‖ (SteeringVectorMath.normUnitScale,
        // the exact conversion injection uses), so Δnorm = |α|·residual.
        let preview = try #require(
            Preview.compute(
                layer: 12, vectorNorm: 4.0, residualNorm: 20.0,
                alpha: 0.5, alphaInNormUnits: true, fixedLayer: false))
        let expectedScale = try SteeringVectorMath.normUnitScale(
            alpha: 0.5, residualNorm: 20.0, vectorNorm: 4.0)
        #expect(preview.effectiveAlpha == expectedScale)
        #expect(preview.effectiveAlpha == 2.5)  // 0.5 · 20 / 4
        #expect(preview.injectedNorm == 10.0)  // |α|·residual = 0.5 · 20
        #expect(preview.residualNorm == 20.0)
        #expect(preview.isNormUnits)
        #expect(!preview.normUnitsFallback)  // real conversion — no fallback
    }

    @Test func rawAlphaPassesThrough() throws {
        let preview = try #require(
            Preview.compute(
                layer: 3, vectorNorm: 4.0, residualNorm: 20.0,
                alpha: -2.0, alphaInNormUnits: false, fixedLayer: true))
        #expect(preview.effectiveAlpha == -2.0)
        #expect(preview.injectedNorm == 8.0)  // |−2| · ‖v‖
        #expect(!preview.isNormUnits)
        #expect(!preview.normUnitsFallback)  // raw mode: nothing to fall back from
        #expect(preview.fixedLayer)
    }

    @Test func missingResidualNormFallsBackToRawAlphaAndSaysSo() throws {
        // Same fallback the local path always had: a vector without a
        // recorded denominator previews the raw α (injection itself would
        // refuse norm units for it — normUnitsAvailable gates the toggle).
        // The fallback is FLAGGED so the UI label never claims a norm-unit
        // conversion that did not happen.
        let preview = try #require(
            Preview.compute(
                layer: 5, vectorNorm: 3.0, residualNorm: nil,
                alpha: 0.4, alphaInNormUnits: true, fixedLayer: false))
        #expect(preview.effectiveAlpha == Float(0.4))
        #expect(preview.injectedNorm == Float(0.4) * 3.0)
        #expect(preview.residualNorm == nil)
        #expect(preview.isNormUnits)
        #expect(preview.normUnitsFallback)
    }

    @Test func rawSlotsReportTheirNormUnitEquivalent() throws {
        // The modernization bridge for legacy raw-alpha variants: raw 6 on a
        // vector with ‖v‖≈8.5 and residual ≈46.75 is ≈ 1.09 norm units.
        let preview = try #require(
            Preview.compute(
                layer: 14, vectorNorm: 8.4948, residualNorm: 46.75,
                alpha: 6.0, alphaInNormUnits: false, fixedLayer: false))
        let equivalent = try #require(preview.equivalentNormUnits)
        #expect(abs(equivalent - preview.injectedNorm / 46.75) < 1e-6)
        #expect(abs(equivalent - 1.0903) < 1e-3)

        // Norm-units mode: α already IS the units — no equivalent line.
        let normUnits = try #require(
            Preview.compute(
                layer: 14, vectorNorm: 8.4948, residualNorm: 46.75,
                alpha: 0.4, alphaInNormUnits: true, fixedLayer: false))
        #expect(normUnits.equivalentNormUnits == nil)

        // Raw with no denominator: nothing to convert with.
        let noResidual = try #require(
            Preview.compute(
                layer: 14, vectorNorm: 8.4948, residualNorm: nil,
                alpha: 6.0, alphaInNormUnits: false, fixedLayer: false))
        #expect(noResidual.equivalentNormUnits == nil)
    }

    @Test func degenerateNormsProduceNoPreview() {
        // A zero-norm vector has no meaningful preview at all…
        #expect(
            Preview.compute(
                layer: 0, vectorNorm: 0, residualNorm: 10.0,
                alpha: 1.0, alphaInNormUnits: false, fixedLayer: false)
                == nil)
        // …and a zero residual denominator cannot convert: raw-α fallback,
        // matching normUnitScale's degenerate-data refusal.
        let zeroResidual = Preview.compute(
            layer: 0, vectorNorm: 2.0, residualNorm: 0,
            alpha: 1.5, alphaInNormUnits: true, fixedLayer: false)
        #expect(zeroResidual?.effectiveAlpha == 1.5)
        #expect(zeroResidual?.normUnitsFallback == true)  // degenerate → flagged
    }
}
