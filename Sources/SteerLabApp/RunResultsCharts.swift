import Charts
import ExperimentKit
import Foundation
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
                        .help(
                            "choose which measure the charts plot — effects of "
                                + "different measures do not share an axis "
                                + "scale, so only one is drawn at a time")
                    }
                    if let activeMetric {
                        EffectForestChart(
                            rows: activeRows,
                            metricLabel: axisMetricLabel(activeMetric))
                        Text(forestCaption(activeMetric))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        if !doseSeries.isEmpty {
                            DoseResponseChart(
                                series: doseSeries,
                                metricLabel: axisMetricLabel(activeMetric))
                            Text(doseCaption)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
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

/// The AXIS form of a metric phrase. `EffectNarrative.metricPhrase` reads
/// "lexical variety (distinct2)" — plain words first, engine term second —
/// and an axis label has no room for both, so the trailing engine term is
/// dropped here. The captions under each chart still carry the full phrase,
/// and an axis that says only "effect" (the previous label) does not say
/// what changed by how much, which changes with the Measure picker.
func axisMetricLabel(_ metric: String) -> String {
    let phrase = EffectNarrative.metricPhrase(metric)
    guard let open = phrase.lastIndex(of: "(") else { return phrase }
    let trimmed = phrase[phrase.startIndex..<open]
        .trimmingCharacters(in: .whitespaces)
    return trimmed.isEmpty ? phrase : trimmed
}

/// Forest plot: one horizontal row per condition, CI as a whisker, the mean
/// as a point, a vertical rule at zero. Conditions whose CI excludes zero
/// draw in the accent color; the rest stay secondary (theme-friendly on
/// light and dark).
struct EffectForestChart: View {
    let rows: [RunResults.EffectSizeRow]
    /// What the Δ axis is measuring, from the active Measure picker.
    var metricLabel: String?
    @State private var readout: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            chart
            // A constant one-line slot: a readout that appeared on hover
            // would move everything under it.
            Text(readout ?? " ")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    private var xAxisLabel: String {
        guard let metricLabel else { return "shift vs paired baseline (Δ)" }
        return "shift in \(metricLabel) vs paired baseline (Δ)"
    }

    private var chart: some View {
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
        .chartXAxisLabel(xAxisLabel)
        .chartYAxis {
            AxisMarks { _ in
                AxisValueLabel()
            }
        }
        .frame(height: max(80, CGFloat(rows.count) * 28 + 40))
        .chartOverlay { proxy in
            GeometryReader { geometry in
                Rectangle()
                    .fill(Color.clear)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            readout = readoutText(
                                at: location, proxy: proxy, geometry: geometry)
                        case .ended:
                            readout = nil
                        }
                    }
            }
        }
        .accessibilityLabel(
            "Forest plot of per-condition effect sizes with 95% confidence "
                + "intervals around a zero line")
    }

    /// The hovered condition's numbers, so a value can be read off the
    /// picture instead of only out of the table below it.
    private func readoutText(
        at location: CGPoint, proxy: ChartProxy, geometry: GeometryProxy
    ) -> String? {
        guard let plotFrame = proxy.plotFrame else { return nil }
        let frame = geometry[plotFrame]
        guard frame.contains(location) else { return nil }
        guard
            let condition = proxy.value(
                atY: location.y - frame.minY, as: String.self),
            let match = rows.first(where: { $0.condition == condition })
        else { return nil }
        return String(
            format: "%@ · Δ %+.4g [%.4g, %.4g]", match.condition,
            match.meanDiff, match.ciLower, match.ciUpper)
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
    /// What the effect axis is measuring. nil where the caller has no
    /// metric to name; the axis then reads "effect" as it always did.
    var metricLabel: String?
    @State private var readout: String?

    /// A dose point with its position in the ladder. `ForEach(points,
    /// id: \.alpha)` collided whenever two conditions sat at the SAME
    /// strength (repeats, seeds), and duplicate ForEach ids drop marks.
    private struct IndexedPoint: Identifiable {
        let id: Int
        let point: EffectNarrative.DosePoint
    }

    private func indexed(_ points: [EffectNarrative.DosePoint]) -> [IndexedPoint] {
        points.enumerated().map { IndexedPoint(id: $0.offset, point: $0.element) }
    }

    private var yAxisLabel: String {
        guard let metricLabel else { return "effect" }
        return "shift in \(metricLabel)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            chart
            // A constant one-line slot, so the hover readout never moves
            // the rows under the chart.
            Text(readout ?? " ")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    private var chart: some View {
        Chart {
            RuleMark(y: .value("no effect", 0))
                .foregroundStyle(.tertiary)
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            ForEach(series) { line in
                ForEach(indexed(line.points)) { entry in
                    if let lower = entry.point.ciLower,
                        let upper = entry.point.ciUpper
                    {
                        RuleMark(
                            x: .value("Strength (α)", entry.point.alpha),
                            yStart: .value("CI lower", lower),
                            yEnd: .value("CI upper", upper)
                        )
                        .foregroundStyle(.secondary.opacity(0.45))
                        .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                    }
                    LineMark(
                        x: .value("Strength (α)", entry.point.alpha),
                        y: .value("Effect", entry.point.effect)
                    )
                    .foregroundStyle(by: .value("Series", line.label))
                    PointMark(
                        x: .value("Strength (α)", entry.point.alpha),
                        y: .value("Effect", entry.point.effect)
                    )
                    .foregroundStyle(by: .value("Series", line.label))
                }
            }
        }
        .chartXAxisLabel("steering strength (α, residual-norm units)")
        .chartYAxisLabel(yAxisLabel)
        // ALWAYS visible: the legend is the only place a series'
        // "<concept> L<layer>" identity appears, and a one-ladder chart
        // used to hide it — leaving a picture that never says what it plots.
        .chartLegend(.visible)
        .frame(height: 180)
        .chartOverlay { proxy in
            GeometryReader { geometry in
                Rectangle()
                    .fill(Color.clear)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            readout = nearest(
                                to: location, proxy: proxy, geometry: geometry)
                        case .ended:
                            readout = nil
                        }
                    }
            }
        }
        .accessibilityLabel(
            "Dose–response chart: effect size versus steering strength")
    }

    /// The ladder point nearest the pointer, so a value can be read off the
    /// picture instead of only out of the table.
    private func nearest(
        to location: CGPoint, proxy: ChartProxy, geometry: GeometryProxy
    ) -> String? {
        guard let plotFrame = proxy.plotFrame else { return nil }
        let frame = geometry[plotFrame]
        guard frame.contains(location) else { return nil }
        guard
            let alpha = proxy.value(
                atX: location.x - frame.minX, as: Double.self)
        else { return nil }
        var best: String?
        var bestDistance = Double.infinity
        for line in series {
            for point in line.points where abs(point.alpha - alpha) < bestDistance {
                bestDistance = abs(point.alpha - alpha)
                best = String(
                    format: "%@ · α %.4g → %+.4g", line.label, point.alpha,
                    point.effect)
            }
        }
        return best
    }
}
