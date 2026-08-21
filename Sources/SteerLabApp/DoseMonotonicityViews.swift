import ExperimentKit
import SwiftUI

// Surfacing dose-monotonicity where the promote decision is made
// (usability plan Phase 2, item 11): the Spearman-ρ monotonicity statistic
// (`StudyStatistics.doseMonotonicity`) previously had no UI consumer. This
// view renders the plain verdict line plus a small dose–response chart for
// one sweep-grid (concept, layer) — everything computed by pure, tested
// ExperimentKit functions (`EffectNarrative`).

/// One promotion-defensibility line + mini chart for a sweep grid cell's
/// (concept, layer), read across the sweep's strength (α) ladder under the
/// declared objective metric.
struct SweepDoseMonotonicityView: View {
    let run: SweepRunCatalog.SweepRun
    let concept: String
    let layer: Int
    /// The resolved objective metric ("markerDensity" reads the density
    /// column; anything else reads the recorded objective values).
    let metric: String
    /// Show the mini chart under the sentence (the recommendation row does;
    /// the transient selected-cell strip keeps just the sentence).
    var showsChart = true

    private var points: [EffectNarrative.DosePoint] {
        EffectNarrative.dosePoints(
            sweepRows: run.rows, concept: concept, layer: layer, metric: metric)
    }

    var body: some View {
        let points = self.points
        let dose = EffectNarrative.doseResponse(points: points)
        VStack(alignment: .leading, spacing: 4) {
            Label {
                Text(EffectNarrative.doseSentence(dose))
                    .font(.caption)
            } icon: {
                Image(systemName: symbol(dose))
                    .foregroundStyle(color(dose))
            }
            Text(
                "dose-monotonicity at layer L\(layer) across the sweep's "
                    + "strength ladder, on \(metric) — a promotion-rule criterion")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if showsChart, points.count >= 2 {
                DoseResponseChart(series: [
                    EffectNarrative.DoseSeries(
                        concept: concept, layer: layer, metric: metric,
                        points: points)
                ])
            }
        }
    }

    private func symbol(_ dose: StudyStatistics.DoseResponse?) -> String {
        guard let dose else { return "questionmark.circle" }
        return dose.isMonotone
            ? "chart.line.uptrend.xyaxis" : "exclamationmark.triangle"
    }

    private func color(_ dose: StudyStatistics.DoseResponse?) -> Color {
        guard let dose else { return .secondary }
        if dose.isMonotone {
            return dose.spearmanRho >= 0 ? .green : .orange
        }
        return .orange
    }
}
