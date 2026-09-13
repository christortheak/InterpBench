import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct ResultReviewSheet: Identifiable {
    enum Mode {
        case generations
        case judgments
    }

    let id = UUID()
    let mode: Mode
    let detail: StudyRunDetail

    var title: String {
        switch mode {
        case .generations: "Generated Responses"
        case .judgments: "Judge Responses"
        }
    }
}

struct ResultReviewWindow: View {
    let sheet: ResultReviewSheet
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(sheet.title)
                        .font(.title2.weight(.semibold))
                    Text(sheet.detail.item.directoryName)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                // Audit headline 12: the sheet had no way out — no Close
                // button and no cancelAction, so Escape was at best
                // undiscoverable.
                Button("Close", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    switch sheet.mode {
                    case .generations:
                        ForEach(sheet.detail.generations) { generation in
                            generationCard(generation)
                        }
                    case .judgments:
                        ForEach(sheet.detail.judgments) { judgment in
                            judgmentCard(judgment)
                        }
                    }
                }
                .padding(.trailing, 8)
            }
        }
        .padding(18)
        .frame(minWidth: 760, minHeight: 560)
    }

    private func generationCard(_ generation: StudyGenerationPreview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(generation.condition) · \(generation.promptID)")
                    .font(.headline)
                Spacer()
                Text(
                    "\(generation.wordCount) words · distinct-2 "
                        + generation.distinct2.formatted(.number.precision(.fractionLength(3)))
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            if let decisions = generation.interventionDecisions {
                DisclosureGroup("Intervention decisions") {
                    Text("These records identify the policy, the consumed token position, the probe scores, and the requested strengths. Partial or failed responses are not completed evidence.").font(.caption)
                    ScrollView([.horizontal, .vertical]) {
                        PolicyJSONView(value: decisions)
                    }.frame(maxHeight: 300)
                }
            }
            if let readings = generation.probeMeasurements { ProbeMeasurementResultsView(value: readings) }
            Text(generation.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Text(generation.output + (generation.truncated ? "\n…" : ""))
                .font(.body.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.35))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(12)
        .background(.background)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }

    private func judgmentCard(_ judgment: StudyJudgePreview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(judgment.condition) · \(judgment.promptID)")
                    .font(.headline)
                Spacer()
                Text(
                    "winner \(judgment.winner) · \(judgment.conditionResult) · confidence "
                        + judgment.confidence.formatted(.number.precision(.fractionLength(2)))
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Text("baseline \(judgment.baselineWas) · condition \(judgment.conditionWas)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(judgment.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Text(judgment.briefReason)
                .textSelection(.enabled)
            if let scores = scoreSummary(judgment) {
                Text(scores)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let structured = structuredFieldsSummary(judgment) {
                Text(structured)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            DisclosureGroup("Raw judge JSON") {
                Text(judgment.rawJSON)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
            .help("the judge's verbatim response for this pair, as recorded in judgments.jsonl")
        }
        .padding(12)
        .background(.background)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }

    private func scoreSummary(_ judgment: StudyJudgePreview) -> String? {
        let a = (judgment.aScores ?? [:]).map { "\($0.key): \($0.value)" }.sorted().joined(
            separator: ", ")
        let b = (judgment.bScores ?? [:]).map { "\($0.key): \($0.value)" }.sorted().joined(
            separator: ", ")
        guard !a.isEmpty || !b.isEmpty else { return nil }
        return "A [\(a.isEmpty ? "no scores" : a)] · B [\(b.isEmpty ? "no scores" : b)]"
    }

    private func structuredFieldsSummary(_ judgment: StudyJudgePreview) -> String? {
        guard let fields = judgment.structuredFields, !fields.isEmpty else { return nil }
        let summary = fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
        return "structured_fields [\(summary)]"
    }

    private func structuredSummaryText(_ summary: StructuredFieldSummaryView) -> String {
        var parts = ["n \(summary.count)"]
        if let mean = summary.numericMean {
            parts.append("mean \(mean.formatted(.number.precision(.fractionLength(3))))")
        }
        if let trueCount = summary.trueCount, let falseCount = summary.falseCount {
            parts.append("true \(trueCount)")
            parts.append("false \(falseCount)")
        }
        if let stringCounts = summary.stringCounts, !stringCounts.isEmpty {
            let counts = stringCounts.map { "\($0.key): \($0.value)" }
                .sorted()
                .joined(separator: ", ")
            parts.append(counts)
        }
        return parts.joined(separator: " · ")
    }
}

/// Import JSONL… sheet for the Input Data section: paste (text area) or
/// choose a file, watch the live parse preview — record count, how many
/// records carry `options`/`target`, or the FIRST error with its line
/// number — and import only when every line parses. Garbage is refused,
/// never coerced into prompt text. All parsing rules live in
/// `TaskPromptsImport` (ExperimentKit, unit-tested); this sheet renders
/// them.
