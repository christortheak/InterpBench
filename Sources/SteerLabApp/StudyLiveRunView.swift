import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct StudyLiveRunView: View {
    @Bindable var jobs: StudyLocalJobController
    var body: some View { liveRunViewer(jobs: jobs) }

    @ViewBuilder
    private func liveRunViewer(jobs: StudyLocalJobController) -> some View {
        let hasLiveContent =
            jobs.isRunning || jobs.isEvaluating
            || jobs.liveRunDirectory != nil
            || jobs.liveEvaluationDirectory != nil
            || jobs.liveActiveGeneration != nil
            || jobs.liveActiveJudgment != nil
            || !jobs.liveGenerations.isEmpty
            || !jobs.liveJudgments.isEmpty

        if hasLiveContent {
            Section("Run Viewer") {
                if let directory = jobs.liveRunDirectory {
                    LabeledContent("Run", value: URL(filePath: directory).lastPathComponent)
                        .font(.caption)
                        .help("the immutable run artifact directory being written")
                }
                if let directory = jobs.liveEvaluationDirectory {
                    LabeledContent("Judge", value: URL(filePath: directory).lastPathComponent)
                        .font(.caption)
                        .help("the immutable paired-judge artifact directory being written")
                }

                if let active = jobs.liveActiveGeneration {
                    liveGenerationCard(active)
                    // A1: the progress row carries its own Stop (the same
                    // cooperative flag as the Run/Judge buttons' Stop).
                    if jobs.isRunning {
                        Button("Stop Run", role: .destructive) {
                            jobs.cancelStudyRun()
                        }
                        .controlSize(.small)
                        .disabled(jobs.studyRunCancelRequested)
                        .help("stops after this generation; partial artifacts stay")
                    }
                }

                if !jobs.liveGenerations.isEmpty {
                    DisclosureGroup("Generated Responses (\(jobs.liveGenerations.count))") {
                        ForEach(jobs.liveGenerations.prefix(12)) { generation in
                            compactGenerationCard(generation)
                        }
                    }
                    .help("completed generations appear here as soon as each prompt finishes")
                }

                if let active = jobs.liveActiveJudgment {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Judging \(active.condition) · \(active.promptID)")
                            .font(.callout.weight(.medium))
                        if jobs.isEvaluating {
                            Button("Stop", role: .destructive) {
                                jobs.cancelPairedJudge()
                            }
                            .controlSize(.small)
                            .disabled(jobs.evaluationCancelRequested)
                            .help("stops after this judgment; completed judgments stay")
                        }
                    }
                    .padding(.vertical, 4)
                    .help("the paired judge is comparing this condition response against baseline")
                }

                if !jobs.liveJudgments.isEmpty {
                    DisclosureGroup("Judge Results (\(jobs.liveJudgments.count))") {
                        ForEach(jobs.liveJudgments.prefix(20)) { judgment in
                            compactJudgmentCard(judgment)
                        }
                    }
                    .help(
                        "completed paired-judge decisions, highlighted separately from raw generations"
                    )
                }

                if !jobs.isRunning && !jobs.isEvaluating {
                    Button("Clear Viewer") { jobs.clearLiveViewer() }
                        .help("clears only this live display; run artifacts remain on disk")
                }
            }
        }
    }

    private func liveGenerationCard(_ active: LiveStudyGeneration) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ProgressView()
                    .controlSize(.small)
                Text("Generating \(active.condition) · \(active.promptID)")
                    .font(.callout.weight(.medium))
                Spacer()
                Text("\(active.output.split(whereSeparator: { $0.isWhitespace }).count) words")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !active.prompt.isEmpty {
                Text(active.prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .textSelection(.enabled)
            }
            Text(active.output.isEmpty ? "waiting for first tokens…" : active.output)
                .font(.body.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.blue.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(.vertical, 4)
    }

    private func compactGenerationCard(_ generation: StudyGenerationPreview) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(generation.condition) · \(generation.promptID)")
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(
                    "\(generation.wordCount) words · distinct-2 "
                        + generation.distinct2.formatted(.number.precision(.fractionLength(3)))
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Text(generation.output + (generation.truncated ? "\n…" : ""))
                .font(.caption.monospaced())
                .lineLimit(10)
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.25))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(.vertical, 4)
    }

    private func compactJudgmentCard(_ judgment: StudyJudgePreview) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(judgment.condition) · \(judgment.promptID)")
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(judgment.conditionResult)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(judgmentHighlight(judgment).opacity(0.18))
                    .clipShape(Capsule())
            }
            Text(
                "winner \(judgment.winner) · confidence "
                    + judgment.confidence.formatted(.number.precision(.fractionLength(2)))
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            Text(judgment.briefReason)
                .font(.caption)
                .textSelection(.enabled)
            if let structured = structuredFieldsSummary(judgment) {
                Text(structured)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 5)
    }

    private func judgmentHighlight(_ judgment: StudyJudgePreview) -> Color {
        switch judgment.conditionResult {
        case "condition": .blue
        case "baseline": .orange
        default: .secondary
        }
    }

    private func structuredFieldsSummary(_ judgment: StudyJudgePreview) -> String? {
        guard let fields = judgment.structuredFields, !fields.isEmpty else { return nil }
        let summary = fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
        return "structured_fields [\(summary)]"
    }
}
