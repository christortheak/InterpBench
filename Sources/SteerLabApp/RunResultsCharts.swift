import Charts
import ExperimentKit
import SwiftUI

// Real charts for the Results surface (usability plan Phase 2, item 10):
// a forest plot of per-condition effect sizes with CI whiskers and a zero
// line, and a dose–response chart for runs whose conditions form an alpha
// ladder. Pure rendering — all data preparation lives in ExperimentKit
// (`EffectNarrative`), and the numeric tables below the charts stay the
// source of truth.

/// The chart block rendered ABOVE the effect-sizes table: a metric picker
/// (effects of different metrics don't share an axis scale), the forest
/// plot for the chosen metric, and — when the run's conditions ladder a
/// concept across strengths — the dose–response line chart.
struct EffectChartsSection: View {
    let rows: [RunResults.EffectSizeRow]
    let interventions: [String: String]

    @State private var selectedMetric: String?

    /// Metrics in first-appearance order (matches the table).
    private var metrics: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for row in rows where seen.insert(row.metric).inserted {
            ordered.append(row.metric)
        }
        return ordered
    }

    private var activeMetric: String? {
        if let selectedMetric, metrics.contains(selectedMetric) {
            return selectedMetric
        }
        return metrics.first
    }

    private var activeRows: [RunResults.EffectSizeRow] {
        guard let activeMetric else { return [] }
        return rows.filter { $0.metric == activeMetric }
    }

    private var doseSeries: [EffectNarrative.DoseSeries] {
        guard let activeMetric else { return [] }
        return EffectNarrative.doseSeries(
            effectSizes: rows, interventions: interventions
        ).filter { $0.metric == activeMetric }
    }

    var body: some View {
        if !rows.isEmpty {
            GroupBox("Effect charts") {
                VStack(alignment: .leading, spacing: 10) {
                    if metrics.count > 1 {
                        Picker("Measure", selection: metricBinding) {
                            ForEach(metrics, id: \.self) { metric in
                                Text(EffectNarrative.metricPhrase(metric)).tag(metric)
                            }
                        }
                        .pickerStyle(.menu)
                        .controlSize(.small)
                        .fixedSize()
                    }
                    if let activeMetric {
                        EffectForestChart(rows: activeRows)
                        Text(forestCaption(activeMetric))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if !doseSeries.isEmpty {
                            DoseResponseChart(series: doseSeries)
                            Text(doseCaption)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var metricBinding: Binding<String> {
        Binding(
            get: { activeMetric ?? "" },
            set: { selectedMetric = $0 })
    }

    private func forestCaption(_ metric: String) -> String {
        "each row: the condition's shift in \(EffectNarrative.metricPhrase(metric)) "
            + "vs its paired baseline, with the 95% bootstrap CI — whiskers "
            + "crossing the zero line are consistent with no effect"
    }

    private var doseCaption: String {
        "dose–response: the same shift plotted against steering strength (α) "
            + "for conditions that ladder one concept and layer across strengths"
    }
}

/// Forest plot: one horizontal row per condition, CI as a whisker, the mean
/// as a point, a vertical rule at zero. Conditions whose CI excludes zero
/// draw in the accent color; the rest stay secondary (theme-friendly on
/// light and dark).
struct EffectForestChart: View {
    let rows: [RunResults.EffectSizeRow]

    var body: some View {
        Chart {
            RuleMark(x: .value("no effect", 0))
                .foregroundStyle(.tertiary)
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            ForEach(rows) { row in
                if row.ciLower.isFinite, row.ciUpper.isFinite {
                    RuleMark(
                        xStart: .value("CI lower", row.ciLower),
                        xEnd: .value("CI upper", row.ciUpper),
                        y: .value("Condition", row.condition)
                    )
                    .foregroundStyle(whiskerStyle(row))
                    .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round))
                }
                if row.meanDiff.isFinite {
                    PointMark(
                        x: .value("Shift vs baseline", row.meanDiff),
                        y: .value("Condition", row.condition)
                    )
                    .foregroundStyle(pointStyle(row))
                    .symbolSize(60)
                }
            }
        }
        .chartXAxisLabel("shift vs paired baseline (Δ)")
        .chartYAxis {
            AxisMarks { _ in
                AxisValueLabel()
            }
        }
        .frame(height: max(80, CGFloat(rows.count) * 28 + 40))
        .accessibilityLabel(
            "Forest plot of per-condition effect sizes with 95% confidence "
                + "intervals around a zero line")
    }

    private func whiskerStyle(_ row: RunResults.EffectSizeRow) -> Color {
        row.ciExcludesZero ? Color.accentColor : Color.secondary.opacity(0.55)
    }

    private func pointStyle(_ row: RunResults.EffectSizeRow) -> Color {
        row.ciExcludesZero ? Color.accentColor : Color.secondary
    }
}

/// Dose–response: effect vs strength (α), one line per (concept, layer)
/// series, CI whiskers per point where the source carries them, and a
/// horizontal rule at zero. Reused by the Optimizations surface with
/// sweep-grid series (no CIs there).
struct DoseResponseChart: View {
    let series: [EffectNarrative.DoseSeries]

    var body: some View {
        Chart {
            RuleMark(y: .value("no effect", 0))
                .foregroundStyle(.tertiary)
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            ForEach(series) { line in
                ForEach(line.points, id: \.alpha) { point in
                    if let lower = point.ciLower, let upper = point.ciUpper {
                        RuleMark(
                            x: .value("Strength (α)", point.alpha),
                            yStart: .value("CI lower", lower),
                            yEnd: .value("CI upper", upper)
                        )
                        .foregroundStyle(.secondary.opacity(0.45))
                        .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                    }
                    LineMark(
                        x: .value("Strength (α)", point.alpha),
                        y: .value("Effect", point.effect)
                    )
                    .foregroundStyle(by: .value("Series", line.label))
                    PointMark(
                        x: .value("Strength (α)", point.alpha),
                        y: .value("Effect", point.effect)
                    )
                    .foregroundStyle(by: .value("Series", line.label))
                }
            }
        }
        .chartXAxisLabel("steering strength (α, residual-norm units)")
        .chartYAxisLabel("effect")
        .chartLegend(series.count > 1 ? .visible : .hidden)
        .frame(height: 180)
        .accessibilityLabel(
            "Dose–response chart: effect size versus steering strength")
    }
}
