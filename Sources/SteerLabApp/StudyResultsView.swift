import AppKit
import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct StudyResultsView<JudgeControls: View>: View {
    @Bindable var service: ChatService
    @Bindable var results: StudyResultsState
    var refresh: () -> Void
    @ViewBuilder var judgeControls: () -> JudgeControls
    @State private var reviewSheet: ResultReviewSheet?
    var body: some View {
        resultsView().sheet(item: $reviewSheet) { ResultReviewWindow(sheet: $0) }
    }

    @ViewBuilder
    private func resultsView() -> some View {
        if results.resultRuns.isEmpty {
            Text("No run artifacts for this study yet.")
                .font(.caption)
                .foregroundStyle(.secondary)
            judgeControls()
        } else {
            Picker(
                "Run",
                selection: Binding<String?>(
                    get: { results.selectedResultID },
                    set: { results.selectedResultID = $0 })
            ) {
                ForEach(results.resultRuns) { item in
                    Text(resultPickerLabel(item))
                        .tag(String?.some(item.id))
                }
            }
            .help(
                "which immutable run directory this pane reads — run, "
                    + "validation, judge and other artifacts of this study, "
                    + "newest first")
            Button("Refresh Results") { refresh() }
                .help(
                    "re-scans this study's runs/ tree and reloads the "
                        + "selected run's artifacts")

            if let detail = results.selectedResult {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Run artifact: \(detail.item.path)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .help(detail.item.path)
                    if let judgeArtifactDirectory = detail.judgeArtifactDirectory {
                        Text("Judge artifact: \(judgeArtifactDirectory)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .help(judgeArtifactDirectory)
                    }
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [URL(filePath: detail.item.path)])
                    }
                    .font(.caption)
                    .help(
                        "opens this run directory in Finder — the artifacts "
                            + "are immutable, so this is a read-only look")
                }

                EvidenceCustodyView(runDirectory: URL(fileURLWithPath: detail.item.path))
                    .id(detail.item.path)

                artifactLinks(detail)

                // F10: the browser item is built ONCE per selection by the
                // panel (config.json read + sweep.csv probe) — never inline
                // here, where the body re-evaluates on every live progress
                // note.
                if let browserItem = results.selectedResultBrowserItem {
                    RunSemanticSectionsView(service: service, item: browserItem)
                }

                judgeControls()

                if let judge = detail.pairedJudgeReport {
                    DisclosureGroup("Paired Judge Report") {
                        LabeledContent("Judge", value: judge.judgeModel)
                        Text(judge.sourceRunDirectory)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        ForEach(judge.conditions, id: \.name) { condition in
                            LabeledContent(condition.name) {
                                Text(judgeConditionLine(condition))
                            }
                            if !condition.structuredSummaries.isEmpty {
                                ForEach(condition.structuredSummaries.keys.sorted(), id: \.self) {
                                    field in
                                    if let summary = condition.structuredSummaries[field] {
                                        LabeledContent(field) {
                                            Text(structuredSummaryText(summary))
                                        }
                                        .font(.caption)
                                    }
                                }
                            }
                        }
                        if !detail.judgments.isEmpty {
                            Button("Review Judge Responses") {
                                reviewSheet = ResultReviewSheet(mode: .judgments, detail: detail)
                            }
                            .help(
                                "opens every recorded judgment in a sheet — "
                                    + "prompt, both outputs' scores, the brief "
                                    + "reason and the raw judge JSON")
                        }
                    }
                    .help(
                        "the paired judge's per-condition tallies from "
                            + "judge-report.json — wins, ties and mean confidence")
                }

                if !detail.robustnessReports.isEmpty {
                    DisclosureGroup("Agent Robustness") {
                        ForEach(detail.robustnessReports.keys.sorted(), id: \.self) { name in
                            if let report = detail.robustnessReports[name] {
                                robustnessReportView(name: name, report: report)
                            }
                        }
                    }
                    .help(
                        "capability and coherence checks per agent from "
                            + "robustness-report.json — does the intervention "
                            + "cost the model its answers?")
                }

                if let validation = detail.validationReportText {
                    DisclosureGroup("Validation Report") {
                        Text(validation)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                    }
                    .help(
                        "the convergent-validity evidence this run wrote — the "
                            + "same text freeze's validation gate reads")
                }

                if !detail.generations.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Button("Review Responses (\(detail.generations.count))") {
                            reviewSheet = ResultReviewSheet(mode: .generations, detail: detail)
                        }
                        .help(
                            "opens every generated response in a sheet — "
                                + "prompt, condition and the full output text")
                        ForEach(detail.generations.prefix(5)) { generation in
                            LabeledContent("\(generation.condition) · \(generation.promptID)") {
                                Text("\(generation.wordCount) words")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every link here hands the file to whatever app owns its extension —
    /// it LEAVES SteerLab, which the labels alone never said (audit 10). The
    /// help on each says so, and Reveal in Finder above is the stay-put
    /// alternative.
    @ViewBuilder
    private func artifactLinks(_ detail: StudyRunDetail) -> some View {
        HStack {
            if !detail.generations.isEmpty {
                Link("generations.jsonl", destination: artifactURL(detail, "generations.jsonl"))
                    .help(Self.artifactLinkHelp("one JSON line per generated response"))
            }
            if !detail.judgments.isEmpty {
                Link("judgments.jsonl", destination: artifactURL(detail, "judgments.jsonl"))
                    .help(
                        Self.artifactLinkHelp(
                            "one JSON line per judgment, noncompliant rows included"))
            }
            if detail.report != nil || detail.validationReportText != nil {
                Link("report.json", destination: artifactURL(detail, "report.json"))
                    .help(Self.artifactLinkHelp("this run's summary statistics and manifest stamp"))
            }
            if detail.pairedJudgeReport != nil {
                Link("judge-report.json", destination: artifactURL(detail, "judge-report.json"))
                    .help(Self.artifactLinkHelp("the paired judge's per-condition tallies"))
            }
            if !detail.robustnessReports.isEmpty {
                Link(
                    "robustness-report.json",
                    destination: artifactURL(detail, "robustness-report.json"))
                    .help(Self.artifactLinkHelp("per-agent capability and coherence checks"))
            }
        }
        .font(.caption)
    }

    private static func artifactLinkHelp(_ what: String) -> String {
        "opens this file OUTSIDE SteerLab, in whichever app owns its type — "
            + what
    }

    private func artifactURL(_ detail: StudyRunDetail, _ filename: String) -> URL {
        let judgeFiles = Set(["judgments.jsonl", "judge-report.json"])
        if judgeFiles.contains(filename), let judgeArtifactDirectory = detail.judgeArtifactDirectory
        {
            return URL(filePath: judgeArtifactDirectory).appending(component: filename)
        }
        return URL(filePath: detail.item.path).appending(component: filename)
    }

    private func resultPickerLabel(_ item: StudyRunListItem) -> String {
        switch item.kind {
        case .run:
            return "run · \(item.directoryName)"
                + (item.generationCount > 0 ? " · \(item.generationCount) outputs" : "")
        case .validate:
            return "validation · \(item.directoryName)"
        case .evaluate:
            return "judge · \(item.directoryName)"
        case .other:
            return "artifact · \(item.directoryName)"
        }
    }

    private func judgeConditionLine(_ condition: PairedJudgeReportView.Condition) -> String {
        let confidence = condition.meanConfidence.formatted(.number.precision(.fractionLength(2)))
        return "condition \(condition.conditionWins) · baseline \(condition.baselineWins)"
            + " · ties \(condition.ties) · confidence \(confidence)"
    }

    private func robustnessReportView(name: String, report: VariantRobustnessReport) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(name)
                .font(.callout.weight(.semibold))
            LabeledContent(
                "Capability",
                value:
                    percent(report.variantBatteryAccuracy) + " agent · "
                    + percent(report.baselineBatteryAccuracy) + " baseline")
            LabeledContent(
                "Distinct-2",
                value:
                    report.meanVariantDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " agent · "
                    + report.meanBaselineDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " baseline")
            if report.judgeModel != nil {
                let counts = Dictionary(grouping: report.coherenceItems.compactMap(\.judgeResult)) {
                    $0
                }
                .mapValues(\.count)
                // The record vocabulary keeps saying "variant"; the reader
                // reads "agent" (audit 10 polish — the artifact keys are
                // untouched).
                Text(
                    "Judge: baseline \(counts["baseline"] ?? 0) · agent \(counts["variant"] ?? 0) · ties \(counts["tie"] ?? 0)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(report.coherenceItems.filter { $0.judge != nil }, id: \.index) { item in
                    if let judge = item.judge {
                        DisclosureGroup("Judge \(item.index): \(item.judgeResult ?? judge.winner)")
                        {
                            Text(judge.briefReason)
                                .font(.caption)
                                .textSelection(.enabled)
                        }
                        .help("the coherence judge's reason for this item, as recorded")
                    }
                }
            }
            if report.warnings.isEmpty {
                Label("No robustness warnings", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                ForEach(report.warnings, id: \.self) { warning in
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding(.vertical, 6)
    }

    private func percent(_ value: Float) -> String {
        value.formatted(.percent.precision(.fractionLength(0)))
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
