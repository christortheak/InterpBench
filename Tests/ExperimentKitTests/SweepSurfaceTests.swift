import Foundation
import Testing

@testable import ExperimentKit

/// E2 — everything a sweep's surface needs, resolved in one place.
struct SweepPanelModelTests {

    private func spec(
        fractions: [Double] = [0.5, 0.7],
        alphas: [Double] = [0.05, 0.1],
        margin: Double? = nil
    ) -> ExperimentManifest.SweepSpec {
        var spec = ExperimentManifest.SweepSpec(
            layerFractions: fractions, alphas: alphas,
            devPromptsFile: "prompts/dev/dev.jsonl",
            batteryFile: "prompts/batteries/basic.jsonl", maxTokens: 64)
        spec.selection = .init(
            objective: .init(metric: "markerDensity"),
            constraints: .init(capabilityTolerance: 0.15, coherenceFloor: 0.45),
            controls: margin.map { .init(matchedNormRandomMargin: $0) })
        return spec
    }

    private func resolved(
        fractions: [Double] = [0.5, 0.7], alphas: [Double] = [0.05, 0.1],
        margin: Double? = nil, layerCount: Int? = 62
    ) throws -> SweepPanelModel.Resolved {
        let s = spec(fractions: fractions, alphas: alphas, margin: margin)
        return SweepPanelModel.resolve(
            spec: s, criterion: try SweepSelectionRule.resolve(s.selection),
            layerCount: layerCount, files: [])
    }

    @Test func fractionsResolveToLayerIndicesWithoutLoadingAModel() throws {
        // Requiring a 27B load to answer "what is 0.7 of this network?" is
        // the reason that question went unanswered.
        let grid = try resolved()
        #expect(grid.layers == [31, 43])
        #expect(grid.cellCount == 4)
        #expect(grid.gridSummary.contains("layers 31, 43 of 62"))
    }

    @Test func anUnknownLayerCountSaysSoRatherThanGuessing() throws {
        let grid = try resolved(layerCount: nil)
        #expect(grid.layers.isEmpty)
        #expect(grid.gridSummary.contains("layer indices unknown"))
        #expect(grid.gridSummary.contains("no cached layer count"))
    }

    @Test func collidingFractionsAreCountedAndNamed() throws {
        // A grid of "three depths" that is really two is a silently smaller
        // sweep.
        let grid = try resolved(fractions: [0.5, 0.505, 0.7], layerCount: 62)
        #expect(grid.layers == [31, 43])
        #expect(grid.collapsedFractions == 1)
        #expect(grid.gridSummary.contains("collapsed"))
    }

    @Test func anAbsentControlIsStatedNotLeftBlank() throws {
        // An empty field and "no control declared" look identical and mean
        // very different things.
        let without = try resolved(margin: nil)
        #expect(without.control == .absent)
        #expect(without.control.detail.contains("NO matched-norm random control"))
        #expect(without.control.detail.contains("unmeasured"))

        let with = try resolved(margin: 0.02)
        #expect(with.control.isDeclared)
        #expect(with.control.detail.contains("0.02"))
    }

    @Test func aMissingInstrumentFileIsAFactNotAnError() {
        let file = SweepPanelModel.inspect(
            label: "dev prompts", path: "prompts/dev/never-authored.jsonl",
            pinnedHash: nil)
        #expect(!file.exists)
        #expect(file.detail.contains("MISSING"))
        #expect(file.detail.contains("refuses at start"))
    }

    @Test func aDriftedPinIsCalledOut() {
        var file = SweepPanelModel.InstrumentFile(
            label: "dev prompts", declaredPath: "p", resolvedPath: "/p",
            exists: true, rowCount: 30, sha256: String(repeating: "a", count: 64),
            pinnedHash: String(repeating: "b", count: 64))
        #expect(file.drifted)
        #expect(file.detail.contains("DRIFTED"))
        file.pinnedHash = file.sha256
        #expect(!file.drifted)
        #expect(file.detail.contains("pinned"))
    }

    @Test func choicePromptsAreListedOnlyWhenTheObjectiveReadsThem() {
        var s = spec()
        s.selection = .init(
            objective: .init(
                metric: "logprobShift",
                choicePromptsFile: "prompts/dev/choices.jsonl"))
        let withChoices = SweepPanelModel.declaredFiles(
            spec: s, objective: "logprobShift")
        #expect(withChoices.count == 3)
        #expect(withChoices.contains { $0.label == "choice prompts" })
        // A markerDensity sweep never opens it, so listing it would imply a
        // dependency that does not exist.
        let without = SweepPanelModel.declaredFiles(
            spec: s, objective: "markerDensity")
        #expect(without.count == 2)
    }
}

/// E3 — a sweep run's grid, arranged for display.
struct SweepGridPresentationTests {

    private func row(
        _ layer: Int, _ alpha: Double, objective: Double,
        distinct2: Double = 0.9, battery: Double = 0.9
    ) -> SweepRunCatalog.Row {
        SweepRunCatalog.Row(
            concept: "pw", layer: layer, alpha: alpha, markerDensity: 0,
            distinct2: distinct2, batteryAccuracy: battery, objective: objective)
    }

    private var baseline: SweepRunCatalog.Row {
        SweepRunCatalog.Row(
            concept: "pw", layer: -1, alpha: 0, markerDensity: 0,
            distinct2: 0.99, batteryAccuracy: 0.9, objective: 0)
    }

    @Test func constraintFailuresAreStruckThroughNotHidden() {
        // Hiding a measured cell misrepresents the sweep; showing it plain
        // misrepresents the result.
        let rows = [
            baseline,
            row(31, 0.05, objective: 0.4),
            row(41, 0.05, objective: 0.9, distinct2: 0.1),  // fails the floor
        ]
        let grid = SweepGridPresentation.grid(
            concept: "pw", rows: rows, recommendation: nil)
        let failed = grid.cell(layer: 41, alpha: 0.05)
        #expect(failed?.state == .failedConstraint)
        #expect(failed?.isStruckThrough == true)
        // Not shaded: a cell that can never be selected must not read as a
        // contender, and its score must not set the colour range.
        #expect(failed?.intensity == nil)
        #expect(grid.cell(layer: 31, alpha: 0.05)?.intensity != nil)
    }

    @Test func theRampSpansOnlyEligibleCells() {
        let rows = [
            baseline,
            row(31, 0.05, objective: 0.1),
            row(31, 0.1, objective: 0.2),
            row(41, 0.05, objective: 99, distinct2: 0.0),  // ineligible outlier
        ]
        let grid = SweepGridPresentation.grid(
            concept: "pw", rows: rows, recommendation: nil)
        // The ineligible 99 must not wash out the two real cells.
        #expect(grid.cell(layer: 31, alpha: 0.05)?.intensity == 0.0)
        #expect(grid.cell(layer: 31, alpha: 0.1)?.intensity == 1.0)
    }

    @Test func aSweepThatSelectedNothingReportsTheReasonAsAResult() {
        // The practicalwisdom case: a sweep with no winner is a RESULT, not
        // an empty grid.
        let grid = SweepGridPresentation.grid(
            concept: "pw", rows: [baseline, row(31, 0.05, objective: -0.35)],
            recommendation: .failure("no eligible cell beat the baseline"))
        #expect(grid.outcome == .noRecommendation(reason: "no eligible cell beat the baseline"))
        #expect(grid.outcome.detail.contains("no cell was selected"))
    }

    @Test func anAbsentControlIsStatedOnTheResult() {
        let grid = SweepGridPresentation.grid(
            concept: "pw", rows: [baseline, row(31, 0.05, objective: 0.4)],
            recommendation: nil)
        #expect(grid.control == .absent)
        #expect(grid.control.detail.contains("unmeasured"))
        #expect(grid.control.detail.contains("exploratory"))
    }

    @Test func theBaselineIsShownSeparatelyFromTheGrid() {
        let grid = SweepGridPresentation.grid(
            concept: "pw", rows: [baseline, row(31, 0.05, objective: 0.4)],
            recommendation: nil)
        // The no-injection anchor is not a candidate cell.
        #expect(grid.baseline?.state == .baseline)
        #expect(grid.layers == [31])
        #expect(grid.cells.count == 1)
    }
}
