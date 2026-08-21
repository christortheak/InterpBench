import ExperimentKit
import SwiftUI

// Semantic Results views (app gaps A3/A6/A13; team findings P3/P4): the
// classification header, the categorical study view, the analyze action,
// the statistics tables, and the structured validation report. All render
// `RunResults` state — no parsing or science logic in this file.

// MARK: - Classification header (P4)

struct RunClassificationHeaderView: View {
    let classification: RunResults.Classification

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(classification.label)
                    .font(.callout.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 3)
                    .background(badgeColor.opacity(0.18), in: Capsule())
                    .overlay(Capsule().strokeBorder(badgeColor.opacity(0.5)))
                // Frozen classes bake the epoch into the label — a chip
                // there could only repeat or contradict it (suppressed).
                if classification.showsEpochChips {
                    if classification.epoch == .failed {
                        chip("epoch mismatch", color: .red)
                    }
                    if classification.epoch == .unknown {
                        chip("epoch unverified", color: .secondary)
                    }
                }
                Spacer()
            }
            Text(classification.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
            // F8: a snapshot the strict decoder cannot read is a CAUTION
            // state, never a silently missing banner.
            if let warning = classification.snapshotWarning {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let warning = classification.revisionWarning {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var badgeColor: Color {
        switch classification.runClass {
        case .frozenEvidenceGrade:
            .green
        case .frozenEpochMismatch: .red
        case .frozenUnverified: .orange
        case .forcedFreeze: .red
        case .validatedDraft: .blue
        case .draftPilot: .gray
        // Same red as the other non-citable classes: an incomplete run is
        // not a milder form of "draft", it is a failure record.
        case .partialFailedRun: .red
        }
    }

    private func chip(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

// MARK: - Semantic sections orchestrator

/// The ONE semantic-section stack shared by all three Results surfaces —
/// the global browser detail, the study-scoped Results detail, and the
/// remote (unpaired-server) detail (F9). Every section renders from the
/// injected `RunResults.Model`, absent-artifact tolerant, so a section
/// added here appears on every surface at once. Only the analyze action
/// differs per surface and is injected.
struct RunSemanticSectionsContent<AnalyzeRow: View>: View {
    let model: RunResults.Model
    var selectFile: ((String) -> Void)?
    @ViewBuilder let analyzeRow: () -> AnalyzeRow

    init(
        model: RunResults.Model,
        selectFile: ((String) -> Void)? = nil,
        @ViewBuilder analyzeRow: @escaping () -> AnalyzeRow
    ) {
        self.model = model
        self.selectFile = selectFile
        self.analyzeRow = analyzeRow
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if model.hasManifestSnapshot {
                RunClassificationHeaderView(classification: model.classification)
            }
            analyzeRow()
            recordCoverageCaptions
            // F2: the per-condition summary renders for EVERY study run —
            // whole-run means (marker density included) live in report.json
            // regardless of whether the run is option-bearing.
            ConditionSummaryTableView(model: model)
            // Declared exclusions (manifest `exclusionRules`): what the
            // stamped rules dropped before the paired statistics — including
            // the explicit zero case (silence would be ambiguous). Reads the
            // run's own stamp (report.json / analysis.json / exclusions.json);
            // runs without declared rules have no stamp and no section.
            RunExclusionsSectionView(runDirectory: model.runDirectory)
            if model.isCategorical {
                CategoricalStudySectionView(model: model, selectFile: selectFile)
            }
            if let effectSizes = model.effectSizes, !effectSizes.isEmpty {
                // Phase 2: sentences and pictures first, dense tables second —
                // the charts render ABOVE the numeric table, never instead.
                EffectChartsSection(
                    rows: effectSizes,
                    interventions: model.conditionInterventions)
                EffectSizesTableView(
                    rows: effectSizes,
                    interventions: model.conditionInterventions)
            }
            if let residuals = model.alienResiduals, !residuals.isEmpty {
                AlienResidualsTableView(rows: residuals)
            }
            if let panelEffects = model.panelEffects, !panelEffects.isEmpty {
                PanelEffectsTableView(rows: panelEffects)
            }
            if !model.panelTranscripts.isEmpty {
                PanelTranscriptView(transcripts: model.panelTranscripts)
            }
            if let movers = model.promotedMovers {
                PromotedMoversView(movers: movers)
            }
            if let validation = model.validationReport {
                ValidationReportSectionView(
                    report: validation, cosine: model.cosineMatrix)
            }
            if !model.isCategorical, let summaries = model.summaries {
                SummariesTableView(table: summaries)
            }
        }
    }

    /// F5's labeling rule, coverage half: whatever is derived from the
    /// decoded records head says so whenever completeness is unproven —
    /// stamped report.json numbers are the whole-run half.
    @ViewBuilder
    private var recordCoverageCaptions: some View {
        if model.generationsTruncated {
            Text(
                "generations.jsonl may exceed the bounded read — item-level "
                    + "tables cover the decoded head only; report.json numbers "
                    + "are whole-run")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        if model.skippedRecordLines > 0 {
            Text("\(model.skippedRecordLines) undecodable record line(s) skipped")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }
}

extension RunSemanticSectionsContent where AnalyzeRow == EmptyView {
    init(model: RunResults.Model, selectFile: ((String) -> Void)? = nil) {
        self.init(model: model, selectFile: selectFile) { EmptyView() }
    }
}

/// Local-run wrapper: loads the `RunResults.Model` for the selected LOCAL
/// run off the main actor and renders the shared section stack with the
/// local analyze action.
struct RunSemanticSectionsView: View {
    @Bindable var service: ChatService
    let item: RunBrowser.Item
    @State private var model: RunResults.Model?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let model {
                RunSemanticSectionsContent(
                    model: model, selectFile: select(fileNamed:)
                ) {
                    RunAnalyzeRow(service: service, item: item)
                }
            } else {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("reading run artifacts…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .task(id: item.id) {
            let url = item.url
            model = nil
            // Bounded reads + pure parsing off the main actor.
            let loaded = await Task.detached(priority: .userInitiated) {
                RunResults.load(runDirectory: url)
            }.value
            model = loaded
        }
    }

    /// "Direct access to raw responses": focus a run file so the activity
    /// viewer renders its bounded preview.
    private func select(fileNamed name: String) {
        guard
            let file = RunBrowser.files(in: item.url).first(where: {
                $0.name == name
            })
        else { return }
        service.experiments.selectedResultsFile = file
    }
}

// MARK: - Per-condition summary table (F2 — every study run)

/// Whole-run per-condition summary from report.json: generations, mean
/// words, mean distinct-2 (the categorical N/A rule applies), and — the
/// manipulation-check DIAGNOSTIC — one marker-density column per concept.
/// Categorical readout columns (choice readouts, target rate, stamped
/// agreement) appear only on option-bearing runs. These are STAMPED
/// whole-run numbers, never head-derived (F5's labeling rule).
struct ConditionSummaryTableView: View {
    let model: RunResults.Model

    var body: some View {
        if !conditionNames.isEmpty, hasContent {
            GroupBox("Per-condition summary") {
                VStack(alignment: .leading, spacing: 4) {
                    ScrollView(.horizontal) {
                        Grid(
                            alignment: .leading, horizontalSpacing: 12,
                            verticalSpacing: 2
                        ) {
                            headerRow
                            ForEach(conditionNames, id: \.self) { name in
                                dataRow(name)
                            }
                        }
                        .font(.caption2.monospaced())
                        .padding(.vertical, 2)
                    }
                    caption
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var conditionNames: [String] {
        if let report = model.report, !report.conditions.isEmpty {
            return RunResults.orderedConditions(report.conditions.keys)
        }
        if !model.completion.perCondition.isEmpty {
            return model.completion.perCondition.map(\.condition)
        }
        return model.choiceMatrix.conditions
    }

    /// Without a report there are no stamped means, and without intervention
    /// summaries the rows would be all dashes — show nothing rather than an
    /// empty grid (absence is shown by the sections that DO have data).
    private var hasContent: Bool {
        model.report?.conditions.isEmpty == false
            || !model.conditionInterventions.isEmpty
    }

    /// Marker-density columns: the union of stamped concepts (sorted).
    private var markerConcepts: [String] {
        guard let report = model.report else { return [] }
        var concepts: Set<String> = []
        for summary in report.conditions.values {
            if let densities = summary.meanMarkerDensity {
                concepts.formUnion(densities.keys)
            }
        }
        return concepts.sorted()
    }

    private var showsCategoricalColumns: Bool { model.isCategorical }

    private var showsBattery: Bool {
        model.report?.conditions.values.contains {
            $0.capabilityBatteryAccuracy != nil
        } ?? false
    }

    /// Ordinal columns appear only when the report stamped an ordinalScale
    /// summary (ordinalMean/ordinalSD) for at least one condition.
    private var showsOrdinal: Bool {
        model.report?.conditions.values.contains {
            $0.ordinalMean != nil
        } ?? false
    }

    private var headerRow: some View {
        GridRow {
            Text("condition")
            Text("intervention")
            Text("generations")
            Text("mean words")
            Text("mean distinct-2")
            ForEach(markerConcepts, id: \.self) { concept in
                Text("\(concept) markers")
            }
            if showsCategoricalColumns {
                Text("choice readouts")
                Text("target choice rate")
                Text("agreement (stamped)")
            }
            if showsOrdinal {
                Text("ordinal mean ± SD")
            }
            if showsBattery {
                Text("battery")
            }
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    private func dataRow(_ name: String) -> some View {
        let summary = model.report?.conditions[name]
        let categorical = model.categoricalConditions.contains(name)
        return GridRow {
            Text(name)
            Text(model.conditionInterventions[name] ?? "—")
                .foregroundStyle(.secondary)
            Text(summary.map { "\($0.generations)" } ?? "—")
            Text(
                RunResults.textMetricDisplay(
                    summary?.meanWordCount, categorical: categorical,
                    fractionDigits: 1))
            Text(
                RunResults.textMetricDisplay(
                    summary?.meanDistinct2, categorical: categorical,
                    fractionDigits: 3))
            ForEach(markerConcepts, id: \.self) { concept in
                Text(
                    summary?.meanMarkerDensity?[concept]
                        .map { String(format: "%.3f", $0) } ?? "—")
            }
            if showsCategoricalColumns {
                Text(summary?.choiceReadouts.map { "\($0)" } ?? "—")
                Text(
                    summary?.choiceRate.map {
                        String(format: "%.1f%%", $0 * 100)
                    } ?? "—")
                Text(agreementText(summary))
            }
            if showsOrdinal {
                Text(ordinalText(summary))
            }
            if showsBattery {
                Text(batteryText(summary))
            }
        }
    }

    private func ordinalText(_ summary: RunResults.ConditionSummary?) -> String {
        guard let mean = summary?.ordinalMean else { return "—" }
        let sd = summary?.ordinalSD.map { String(format: " ± %.2f", $0) } ?? ""
        return String(format: "%.2f", mean) + sd
    }

    private func agreementText(_ summary: RunResults.ConditionSummary?) -> String {
        guard let fraction = summary?.agreementWithBaseline else { return "—" }
        let count = summary?.agreementTotal.map { " (n=\($0))" } ?? ""
        return String(format: "%.0f%%", fraction * 100) + count
    }

    private func batteryText(_ summary: RunResults.ConditionSummary?) -> String {
        guard let accuracy = summary?.capabilityBatteryAccuracy else { return "—" }
        let count = summary?.capabilityBatteryItemCount
        return String(format: "%.0f%%", accuracy * 100)
            + (count.map { " of \($0)" } ?? "")
    }

    private var caption: some View {
        Text(
            model.report != nil
                ? "whole-run numbers stamped in report.json — marker density "
                    + "is the manipulation-check diagnostic, never a promotion "
                    + "objective"
                : "no report.json in this run — conditions listed from the "
                    + "manifest snapshot only")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

// MARK: - Declared exclusions (Phase-4 item 19 — the stamp made readable)

/// Renders a run's `exclusions` stamp in plain language: the one summary
/// sentence ("12 of 240 answers were excluded: …"), the active rules, a
/// per-condition table with per-rule counts (zeros included) and surviving
/// N, and the pairwise-deletion note. A run whose study declared rules but
/// excluded nothing says so explicitly — silence would be ambiguous.
/// Loading is absent-artifact tolerant: no stamp (legacy runs, no declared
/// rules, or a remote directory this pane cannot read) renders nothing.
struct RunExclusionsSectionView: View {
    let runDirectory: URL
    @State private var stamp: ExclusionStamp?

    var body: some View {
        Group {
            if let stamp {
                GroupBox("Excluded answers (declared rules)") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            Text(ExclusionRulesUI.summary(of: stamp))
                                .font(.caption.weight(.semibold))
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                            InfoButton(text: StudyInfo.runExclusions)
                            Spacer(minLength: 0)
                        }
                        rulesList(stamp)
                        conditionTable(stamp)
                        Text(stamp.note)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .task(id: runDirectory) {
            let directory = runDirectory
            stamp = await Task.detached(priority: .utility) {
                ExclusionRulesUI.loadStamp(runDirectory: directory)
            }.value
        }
    }

    /// The active rules as stamped — the stamp's own plain-language
    /// descriptions plus the declared parameters (endpoint, bounds, how
    /// many items carried a check).
    private func rulesList(_ stamp: ExclusionStamp) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(stamp.rules, id: \.rule) { rule in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(ExclusionRulesUI.shortLabel(forRule: rule.rule))
                        .font(.caption2.monospaced())
                    Text(ruleDetail(rule))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func ruleDetail(_ rule: ExclusionStamp.ResolvedRule) -> String {
        var text = rule.description
        if let checked = rule.checkedItems {
            text += " — \(checked) item\(checked == 1 ? "" : "s") declared a check"
        }
        return text
    }

    /// condition | answers | one column per rule | surviving — stamped
    /// whole-run numbers (never head-derived), zeros rendered as zeros.
    private func conditionTable(_ stamp: ExclusionStamp) -> some View {
        let conditions = RunResults.orderedConditions(stamp.consideredN.keys)
        let ruleIDs = stamp.rules.map(\.rule)
        return ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                GridRow {
                    Text("condition")
                    Text("answers")
                    ForEach(ruleIDs, id: \.self) { id in
                        Text(ExclusionRulesUI.shortLabel(forRule: id))
                    }
                    Text("surviving")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
                ForEach(conditions, id: \.self) { condition in
                    GridRow {
                        Text(condition)
                        Text("\(stamp.consideredN[condition] ?? 0)")
                        ForEach(ruleIDs, id: \.self) { id in
                            let count = stamp.excludedByRule[condition]?[id] ?? 0
                            Text("\(count)")
                                .foregroundStyle(
                                    count > 0
                                        ? AnyShapeStyle(.orange)
                                        : AnyShapeStyle(.secondary))
                        }
                        Text("\(stamp.survivingN[condition] ?? 0)")
                    }
                    .font(.caption2.monospaced())
                }
            }
            .padding(.vertical, 2)
        }
    }
}

// MARK: - Analyze action (A3)

@MainActor
private enum RunAnalyzeState {
    case idle
    case running
    case finished(String)
    case failed(String)
}

/// The in-app `analyze` trigger for a completed study run. Routing is
/// `RunResults.analyzeRoute` (tested): local swift-mlx runs call the landed
/// `ExperimentTasks.analyze` in-process; python-hf runs submit the server's
/// analyze verb as a durable job. Refusals (epoch guard, missing runs)
/// surface verbatim.
struct RunAnalyzeRow: View {
    @Bindable var service: ChatService
    let item: RunBrowser.Item
    @State private var state: RunAnalyzeState = .idle

    private var serverAvailable: Bool {
        service.cluster.client != nil && service.cluster.remoteState != nil
    }

    private var route: RunResults.AnalyzeRoute {
        RunResults.analyzeRoute(
            runType: item.runType,
            substrate: item.substrate,
            experiment: item.experiment,
            serverAvailable: serverAvailable)
    }

    var body: some View {
        switch route {
        case .local(let experiment):
            actionRow(
                title: "Analyze (local)",
                caption: "paired effect sizes (bootstrap CI + Wilcoxon) over the "
                    + "newest completed run of '\(experiment)' — pure CPU, writes "
                    + "a new immutable analyze run"
            ) { runLocal(experiment: experiment) }
        case .server(let experiment):
            actionRow(
                title: "Analyze on \(service.cluster.substrateLabel)",
                caption: "submits the server's analyze verb for '\(experiment)' "
                    + "as a durable job (per-engine epoch guard: a server run is "
                    + "analyzed on the server)"
            ) { runServer(experiment: experiment) }
        case .unavailable(let reason):
            if item.runType == "run" {
                // Only explain refusals for study runs — an analyze row on
                // every extract/validate directory would be noise.
                Text("Analyze unavailable: \(reason)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func actionRow(
        title: String, caption: String, action: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Button(action: action) {
                    Label(title, systemImage: "function")
                }
                .controlSize(.small)
                .disabled(isRunning)
                if isRunning { ProgressView().controlSize(.small) }
                statusText
                Spacer()
            }
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var isRunning: Bool {
        if case .running = state { return true }
        return false
    }

    @ViewBuilder
    private var statusText: some View {
        switch state {
        case .idle, .running:
            EmptyView()
        case .finished(let message):
            Text(message)
                .font(.caption)
                .foregroundStyle(.green)
                .textSelection(.enabled)
                .lineLimit(2)
        case .failed(let message):
            Text(message)
                .font(.caption)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .lineLimit(4)
        }
    }

    private func runLocal(experiment: String) {
        state = .running
        Task {
            do {
                let directory = try await Task.detached(priority: .userInitiated) {
                    try ExperimentTasks.analyze(experimentName: experiment)
                }.value
                state = .finished(
                    "analysis → \(directory.lastPathComponent) — refresh the "
                        + "run list to browse it")
            } catch {
                // The epoch guard's and data checks' refusals, verbatim.
                state = .failed("\(error)")
            }
        }
    }

    private func runServer(experiment: String) {
        state = .running
        Task {
            guard let client = service.cluster.client else {
                state = .failed("no server client — connect in Compute first")
                return
            }
            do {
                let jobID = try await client.submitExperimentJob(
                    experiment: experiment, verb: "analyze")
                state = .finished(
                    "submitted analyze job \(jobID) — monitor in Compute, then "
                        + "refresh the server runs list")
            } catch {
                state = .failed("\(error)")
            }
        }
    }
}

/// The remote (unpaired-server) sibling: same routing rule with
/// `browsingRemote`, submitting through the active server's client.
struct RemoteRunAnalyzeRow: View {
    @Bindable var service: ChatService
    let runType: String?
    let substrate: String?
    let experiment: String?
    @State private var state: RunAnalyzeState = .idle

    private var route: RunResults.AnalyzeRoute {
        RunResults.analyzeRoute(
            runType: runType, substrate: substrate, experiment: experiment,
            serverAvailable: service.cluster.client != nil
                && service.cluster.remoteState != nil,
            browsingRemote: true)
    }

    var body: some View {
        switch route {
        case .server(let experiment):
            HStack(spacing: 8) {
                Button {
                    submit(experiment: experiment)
                } label: {
                    Label(
                        "Analyze on \(service.cluster.substrateLabel)",
                        systemImage: "function")
                }
                .controlSize(.small)
                .disabled(isRunning)
                .help(
                    "submit the server's analyze verb for '\(experiment)' as a "
                        + "durable job (newest completed run of the study)")
                if isRunning { ProgressView().controlSize(.small) }
                statusText
                Spacer()
            }
        case .local, .unavailable:
            // Remote browsing never analyzes locally; refusal reasons for
            // non-study runs stay silent here (the row would be noise).
            EmptyView()
        }
    }

    private var isRunning: Bool {
        if case .running = state { return true }
        return false
    }

    @ViewBuilder
    private var statusText: some View {
        switch state {
        case .idle, .running:
            EmptyView()
        case .finished(let message):
            Text(message).font(.caption).foregroundStyle(.green)
                .textSelection(.enabled).lineLimit(2)
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(.red)
                .textSelection(.enabled).lineLimit(4)
        }
    }

    private func submit(experiment: String) {
        state = .running
        Task {
            guard let client = service.cluster.client else {
                state = .failed("no server client — connect in Compute first")
                return
            }
            do {
                let jobID = try await client.submitExperimentJob(
                    experiment: experiment, verb: "analyze")
                state = .finished("submitted analyze job \(jobID) — monitor in Compute")
            } catch {
                state = .failed("\(error)")
            }
        }
    }
}

// MARK: - Categorical study view (P3)

struct CategoricalStudySectionView: View {
    let model: RunResults.Model
    var selectFile: ((String) -> Void)? = nil

    var body: some View {
        GroupBox("Categorical study") {
            VStack(alignment: .leading, spacing: 10) {
                completionRow
                if !model.agreements.isEmpty {
                    agreementBox
                }
                if !model.choiceMatrix.isEmpty {
                    ChoiceMatrixTableView(matrix: model.choiceMatrix)
                }
                if !model.armSummaries.isEmpty {
                    ArmAggregationTableView(summaries: model.armSummaries)
                }
                if !model.parseFailures.isEmpty {
                    ParseFailuresView(failures: model.parseFailures)
                }
                if let summaries = model.summaries {
                    SummariesTableView(table: summaries)
                }
                rawLinksRow
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var completionRow: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(model.completion.summaryLine)
                .font(.caption.weight(.semibold))
            ForEach(model.completion.perCondition) { condition in
                HStack(spacing: 6) {
                    Text(condition.condition)
                        .font(.caption2.monospaced())
                    Text("\(condition.completed) of \(condition.planned)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(
                            condition.completed < condition.planned
                                ? AnyShapeStyle(.orange)
                                : AnyShapeStyle(.secondary))
                    if let error = condition.error {
                        Text(error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
        }
    }

    /// The agreement contract (F9): the STAMPED readout-level number is THE
    /// "agreement with baseline"; the local item-level recount is a
    /// FALLBACK and must never render under the same label — each row says
    /// which one it is.
    private var agreementBox: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Exact agreement with baseline")
                .font(.caption.weight(.semibold))
            ForEach(model.agreements) { agreement in
                HStack(spacing: 6) {
                    Text(agreement.condition)
                        .font(.caption2.monospaced())
                    Text("matches baseline \(agreement.matches)/\(agreement.comparable)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(
                            agreement.matches == agreement.comparable
                                ? AnyShapeStyle(.secondary)
                                : AnyShapeStyle(.primary))
                    Text(
                        agreement.fromServerStamp
                            ? "stamped, readout-level"
                            : "recomputed, item-level")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var rawLinksRow: some View {
        HStack(spacing: 8) {
            if let selectFile {
                Button("View raw records") { selectFile("generations.jsonl") }
                    .controlSize(.small)
                    .help("focus generations.jsonl in the viewer pane")
                if model.summaries != nil {
                    Button("View summaries.csv") { selectFile("summaries.csv") }
                        .controlSize(.small)
                }
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([model.runDirectory])
                } label: {
                    Label("Reveal artifacts", systemImage: "folder")
                }
                .controlSize(.small)
                .help("reveal the immutable run directory in Finder")
            }
            Spacer()
        }
    }
}

/// The item × condition comparison table: generated/parsed choice per cell,
/// color-coded against the item's target, with choice probability and log
/// odds when the answer-token instrument ran.
struct ChoiceMatrixTableView: View {
    let matrix: RunResults.ChoiceMatrix

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Choices by item × condition")
                .font(.caption.weight(.semibold))
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 3) {
                    GridRow {
                        Text("item")
                        Text("target")
                        ForEach(matrix.conditions, id: \.self) { condition in
                            Text(condition)
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    ForEach(matrix.items) { item in
                        GridRow {
                            itemLabel(item)
                            Text(item.target ?? "—")
                                .font(.caption2.monospaced())
                            ForEach(matrix.conditions, id: \.self) { condition in
                                cellView(item.cells[condition])
                            }
                        }
                    }
                }
                .padding(.vertical, 2)
            }
            legend
        }
    }

    private func itemLabel(_ item: RunResults.ItemRow) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(item.promptID)
                .font(.caption2.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            if let arm = item.arm {
                Text(arm)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func cellView(_ cell: RunResults.ChoiceCell?) -> some View {
        if let cell {
            VStack(alignment: .leading, spacing: 0) {
                Text(choiceText(cell))
                    .font(.caption2.monospaced().weight(.semibold))
                    .foregroundStyle(choiceColor(cell))
                if let detail = detailText(cell) {
                    Text(detail)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        } else {
            Text("—")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func choiceText(_ cell: RunResults.ChoiceCell) -> String {
        if let choice = cell.choice {
            if cell.sampleCount > 1 { return "\(choice) (majority)" }
            return choice
        }
        return cell.parseFailures > 0 ? "parse failure" : "—"
    }

    private func choiceColor(_ cell: RunResults.ChoiceCell) -> Color {
        if cell.choice == nil, cell.parseFailures > 0 { return .red }
        switch cell.matchesTarget {
        case .some(true): return .green
        case .some(false): return .orange
        case .none: return .primary
        }
    }

    private func detailText(_ cell: RunResults.ChoiceCell) -> String? {
        var parts: [String] = []
        if let probability = cell.targetProbability {
            parts.append(String(format: "P %.3f", probability))
        }
        if let logOdds = cell.targetLogOdds {
            parts.append(String(format: "lo %+.2f", logOdds))
        }
        if cell.parseFailures > 0, cell.sampleCount > 0 {
            parts.append("\(cell.parseFailures)/\(cell.sampleCount) unparsed")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private var legend: some View {
        Text(
            "green = matches target · orange = differs from target · red = "
                + "parse failure · P/lo = choice probability / log odds of the "
                + "target (answer-token instrument)")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

struct ArmAggregationTableView: View {
    let summaries: [RunResults.ArmSummary]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("By experimental arm")
                .font(.caption.weight(.semibold))
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                    GridRow {
                        Text("arm")
                        Text("condition")
                        Text("items")
                        Text("target-match rate")
                        Text("mean P(target)")
                        Text("mean log-odds")
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    ForEach(summaries) { summary in
                        GridRow {
                            Text(summary.arm)
                            Text(summary.condition)
                            Text("\(summary.items)")
                            Text(percent(summary.targetMatchRate))
                            Text(number(summary.meanTargetProbability, "%.3f"))
                            Text(number(summary.meanTargetLogOdds, "%+.3f"))
                        }
                        .font(.caption2.monospaced())
                    }
                }
            }
        }
    }

    private func percent(_ value: Double?) -> String {
        value.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
    }

    private func number(_ value: Double?, _ format: String) -> String {
        value.map { String(format: format, $0) } ?? "—"
    }
}

struct ParseFailuresView: View {
    let failures: [RunResults.ParseFailure]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(
                "\(failures.count) parse failure\(failures.count == 1 ? "" : "s")",
                systemImage: "exclamationmark.triangle")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.red)
            ForEach(failures.prefix(20)) { failure in
                VStack(alignment: .leading, spacing: 0) {
                    Text("\(failure.condition) · \(failure.promptID) · \(failure.endpoint)")
                        .font(.caption2.monospaced())
                    if !failure.outputExcerpt.isEmpty {
                        Text(failure.outputExcerpt)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
            }
            if failures.count > 20 {
                Text("first 20 shown — the rest are in generations.jsonl")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Statistics tables (A3)

struct EffectSizesTableView: View {
    let rows: [RunResults.EffectSizeRow]
    /// Per-condition intervention summaries (from the manifest snapshot) —
    /// lets the narrative lead with "Steering 'fear' at layer 12 …" instead
    /// of the bare condition name.
    var interventions: [String: String] = [:]

    var body: some View {
        GroupBox("Effect sizes (paired to baseline)") {
            VStack(alignment: .leading, spacing: 4) {
                // A sentence per effect (Phase 2, item 9): the generated
                // plain-language read of each row, above the numbers, never
                // instead of them. Pure ExperimentKit function — tested.
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(rows) { row in
                        Text(EffectNarrative.sentence(
                            for: row, intervention: interventions[row.condition]))
                            .font(.caption)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(.bottom, 2)
                ScrollView(.horizontal) {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                        header
                        ForEach(rows) { row in
                            dataRow(row)
                        }
                    }
                    .padding(.vertical, 2)
                }
                caption
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        GridRow {
            Text("condition")
            Text("metric")
            Text("n")
            Text("Δ mean [95% CI]")
            Text("CI")
            Text("p (Wilcoxon)")
            Text("p (corrected)")
            Text("sig.")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    private func dataRow(_ row: RunResults.EffectSizeRow) -> some View {
        GridRow {
            Text(row.condition)
            Text(row.metric)
            Text("\(row.n)")
            Text(
                String(
                    format: "%+.4g  [%.4g, %.4g]", row.meanDiff, row.ciLower,
                    row.ciUpper))
            EffectCIBar(row: row)
            Text(pText(row.wilcoxonP))
            Text(correctedText(row))
            significanceMark(row)
        }
        .font(.caption2.monospaced())
    }

    private func pText(_ p: Double?) -> String {
        p.map { String(format: "%.4f", $0) } ?? "—"
    }

    private func correctedText(_ row: RunResults.EffectSizeRow) -> String {
        guard let adjusted = row.adjustedP else { return "—" }
        let method = row.correction.map { " (\($0))" } ?? ""
        return String(format: "%.4f", adjusted) + method
    }

    @ViewBuilder
    private func significanceMark(_ row: RunResults.EffectSizeRow) -> some View {
        switch row.significantAfterCorrection {
        case .some(true):
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .imageScale(.small)
        case .some(false):
            Text("n.s.")
                .foregroundStyle(.secondary)
        case .none:
            Text("—")
                .foregroundStyle(.secondary)
        }
    }

    private var caption: some View {
        Text(
            "significance uses the corrected p when the run carries one "
                + "(adjustedP — both engines stamp it), else the raw Wilcoxon "
                + "p — bars show the bootstrap CI around zero")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

/// A compact CI bar on a shared symmetric scale: the zero line in the
/// middle, the CI as a capsule, the mean as a tick. Table stays the source
/// of truth; the bar is a scanning aid.
private struct EffectCIBar: View {
    let row: RunResults.EffectSizeRow

    private static let width: CGFloat = 90
    private static let height: CGFloat = 10

    var body: some View {
        let magnitude = max(abs(row.ciLower), abs(row.ciUpper), abs(row.meanDiff), 1e-12)
        let scale = (Self.width / 2) / magnitude
        ZStack(alignment: .leading) {
            Rectangle()
                .fill(.quaternary)
                .frame(width: 1, height: Self.height)
                .offset(x: Self.width / 2)
            Capsule()
                .fill(row.ciExcludesZero ? Color.accentColor : Color.secondary.opacity(0.5))
                .frame(
                    width: max(2, (row.ciUpper - row.ciLower) * scale),
                    height: 4)
                .offset(x: Self.width / 2 + row.ciLower * scale)
            Rectangle()
                .fill(.primary)
                .frame(width: 1.5, height: Self.height)
                .offset(x: Self.width / 2 + row.meanDiff * scale)
        }
        .frame(width: Self.width, height: Self.height, alignment: .leading)
        .accessibilityLabel(
            String(
                format: "CI %.4g to %.4g, mean %.4g", row.ciLower, row.ciUpper,
                row.meanDiff))
    }
}

struct AlienResidualsTableView: View {
    let rows: [RunResults.AlienResidualRow]

    var body: some View {
        GroupBox("Alien-stance residuals (R = Δmodel − Δhuman)") {
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                    GridRow {
                        Text("condition")
                        Text("endpoint")
                        Text("Δ model")
                        Text("Δ human")
                        Text("R [95% CI]")
                        Text("region")
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    ForEach(rows) { row in
                        GridRow {
                            Text(row.condition)
                            Text(row.endpoint)
                            Text(String(format: "%+.4g", row.deltaModel))
                            Text(String(format: "%+.4g", row.deltaHuman))
                            Text(rText(row))
                            Text(row.region)
                                .foregroundStyle(regionColor(row.region))
                                .fontWeight(.semibold)
                        }
                        .font(.caption2.monospaced())
                    }
                }
                .padding(.vertical, 2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func rText(_ row: RunResults.AlienResidualRow) -> String {
        var text = String(format: "%+.4g", row.r)
        if let lower = row.ciRLower, let upper = row.ciRUpper {
            text += String(format: "  [%.4g, %.4g]", lower, upper)
        }
        return text
    }

    private func regionColor(_ region: String) -> Color {
        switch region {
        case "alien", "inverted": .red
        case "hyperHuman", "hypoHuman": .orange
        case "humanAligned": .green
        default: .secondary
        }
    }
}

/// A14: the multi-agent panel-effect decomposition (panel-effects.csv) as a
/// semantic table — direct / spillover / group effects with their Ns, plus
/// the transmission and amplification ratios, per endpoint. Empty estimand
/// cells (the writer's NaN convention) render as "—", never as a number.
struct PanelEffectsTableView: View {
    let rows: [RunResults.PanelEffectRow]

    var body: some View {
        GroupBox("Panel effects (configured vs stripped baseline)") {
            VStack(alignment: .leading, spacing: 4) {
                ScrollView(.horizontal) {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                        header
                        ForEach(rows) { row in
                            dataRow(row)
                        }
                    }
                    .padding(.vertical, 2)
                }
                caption
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        GridRow {
            Text("endpoint")
            Text("direct (n)")
            Text("spillover (n)")
            Text("group (n)")
            Text("transmission")
            Text("amplification")
            Text("dropped")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    private func dataRow(_ row: RunResults.PanelEffectRow) -> some View {
        GridRow {
            Text(row.endpoint)
            Text(effect(row.direct, n: row.directN))
            Text(effect(row.spillover, n: row.spilloverN))
            Text(effect(row.group, n: row.groupN))
            Text(ratio(row.transmissionRatio))
            Text(ratio(row.amplification))
            Text("\(row.droppedTurns)")
                .foregroundStyle(
                    row.droppedTurns > 0
                        ? AnyShapeStyle(.orange) : AnyShapeStyle(.secondary))
        }
        .font(.caption2.monospaced())
    }

    private func effect(_ value: Double?, n: Int) -> String {
        guard let value else { return "— (\(n))" }
        return String(format: "%+.4g (%d)", value, n)
    }

    private func ratio(_ value: Double?) -> String {
        value.map { String(format: "%.3g", $0) } ?? "—"
    }

    private var caption: some View {
        Text(
            "direct = treated seats vs their own baseline turns · spillover = "
                + "untreated seats sharing the panel · group = the designated "
                + "panel-outcome turn · transmission = spillover/direct · "
                + "amplification = group/direct · — = no parseable paired turns")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

struct PromotedMoversView: View {
    let movers: RunResults.PromotedMovers

    var body: some View {
        GroupBox("Promoted movers (screen funnel)") {
            VStack(alignment: .leading, spacing: 6) {
                moverList("Promoted", movers.promoted, color: .green)
                moverList("Rejected", movers.rejected, color: .secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func moverList(
        _ title: String, _ entries: [RunResults.PromotedMover], color: Color
    ) -> some View {
        if !entries.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(title) (\(entries.count))")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(color)
                ForEach(entries) { mover in
                    VStack(alignment: .leading, spacing: 0) {
                        HStack(spacing: 6) {
                            Text(mover.concept)
                                .font(.caption2.monospaced().weight(.semibold))
                            if let endpoint = mover.endpoint {
                                Text(endpoint)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            if let effect = mover.effectEstimate {
                                Text(String(format: "Δ %+.4g", effect))
                                    .font(.caption2.monospacedDigit())
                            }
                            if let p = mover.adjustedP {
                                Text(String(format: "p̃ %.4f", p))
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if !mover.reasons.isEmpty {
                            Text(mover.reasons.joined(separator: "; "))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }
}

struct SummariesTableView: View {
    let table: RunResults.SummariesTable

    var body: some View {
        GroupBox("Per-item summaries (summaries.csv)") {
            VStack(alignment: .leading, spacing: 4) {
                ScrollView(.horizontal) {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                        GridRow {
                            ForEach(table.header.indices, id: \.self) { index in
                                Text(table.header[index])
                                    .foregroundStyle(.secondary)
                            }
                        }
                        ForEach(table.rows.indices, id: \.self) { rowIndex in
                            GridRow {
                                ForEach(table.rows[rowIndex].indices, id: \.self) { column in
                                    Text(cellText(table.rows[rowIndex][column]))
                                        .textSelection(.enabled)
                                }
                            }
                        }
                    }
                    .font(.caption2.monospaced())
                    .padding(.vertical, 2)
                }
                if table.truncated {
                    Text("first \(table.rows.count) rows — open the file for the full table")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func cellText(_ value: String) -> String {
        value.isEmpty ? "—" : value
    }
}

// MARK: - Validation report (A13)

struct ValidationReportSectionView: View {
    let report: RunResults.ValidationReport
    let cosine: RunResults.CosineMatrix?

    var body: some View {
        GroupBox("Validation report") {
            VStack(alignment: .leading, spacing: 10) {
                convergentTable
                if let cosine {
                    CosineMatrixGridView(matrix: cosine)
                } else if let worst = report.worstCosinePair {
                    Text("worst cross-concept |cosine|: \(worst)")
                        .font(.caption2.monospaced())
                }
                if !report.logitLens.isEmpty {
                    logitLensBox
                }
                if !report.capabilityBattery.isEmpty {
                    batteryBox
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var convergentTable: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Convergent validity (held-out scenarios, chance = 50%)")
                .font(.caption.weight(.semibold))
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                GridRow {
                    Text("concept")
                    Text("layer")
                    Text("scenarios")
                    Text("transfer")
                    Text("calibrated")
                    Text("AUC")
                    Text("")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
                ForEach(report.concepts) { row in
                    conceptRow(row)
                }
            }
            // Review 2026-08-01: the diagnostics existed only in the JSON —
            // the app showed transfer accuracy alone, the one number the
            // fair incident proved can indict the threshold, not the vector.
            Text(
                "transfer = accuracy at the extraction-derived threshold; "
                    + "calibrated = at the held-out classes' own midpoint; "
                    + "AUC = threshold-free ranking. ⚠︎ marks a one-sided "
                    + "transfer read (every item on one side — that accuracy "
                    + "measured the threshold, not the vector). Colors only "
                    + "beyond ±0.10 of chance: 0.51 is not a finding.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func conceptRow(_ row: RunResults.ValidationConceptRow) -> some View {
        GridRow {
            Text(row.concept)
            Text(row.layer.map { "L\($0)" } ?? "—")
            Text(row.scenarios.map { "\($0)" } ?? "—")
            accuracyText(row)
            calibratedText(row)
            aucText(row)
            noteText(row)
        }
        .font(.caption2.monospaced())
    }

    /// Verdict color with a NEUTRAL band around chance (review 2026-08-02:
    /// an AUC of 0.51 colored green is false precision — without
    /// uncertainty or a declared threshold, near-chance is just
    /// near-chance). Green/red only beyond ±0.10 of the 0.5 line.
    private func chanceBandColor(_ value: Double) -> Color? {
        if value >= 0.6 { return .green }
        if value <= 0.4 { return .red }
        return nil
    }

    @ViewBuilder
    private func accuracyText(_ row: RunResults.ValidationConceptRow) -> some View {
        if let accuracy = row.accuracy {
            // The one-sided flag rides the TRANSFER column: that is the
            // number it indicts.
            Text(String(format: "%.0f%%", accuracy * 100)
                + (row.oneSided == true ? " ⚠︎" : ""))
                .foregroundStyle(
                    row.oneSided == true
                        ? Color.orange
                        : chanceBandColor(accuracy) ?? Color.primary)
                .fontWeight(.semibold)
        } else if let fraction = row.fractionAboveMidpoint {
            Text(String(format: "%.0f%% above midpoint (unlabeled)", fraction * 100))
                .foregroundStyle(.orange)
        } else {
            Text("not run")
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func calibratedText(_ row: RunResults.ValidationConceptRow) -> some View {
        if let calibrated = row.calibratedAccuracy {
            Text(String(format: "%.0f%%", calibrated * 100))
                .foregroundStyle(chanceBandColor(calibrated) ?? Color.primary)
                .fontWeight(.semibold)
        } else {
            Text("—").foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func aucText(_ row: RunResults.ValidationConceptRow) -> some View {
        if let auc = row.auc {
            Text(String(format: "%.2f", auc))
                .foregroundStyle(chanceBandColor(auc) ?? Color.primary)
        } else {
            Text("—").foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func noteText(_ row: RunResults.ValidationConceptRow) -> some View {
        if let note = row.note {
            Text(note)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        } else {
            Text("")
        }
    }

    private var logitLensBox: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Logit lens (top tokens through the output head)")
                .font(.caption.weight(.semibold))
            ForEach(report.logitLens) { row in
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 6) {
                        Text(row.concept)
                            .font(.caption2.monospaced().weight(.semibold))
                        if let layer = row.layer {
                            Text("L\(layer)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    if let note = row.note {
                        Text(note)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    } else {
                        if !row.topPositive.isEmpty {
                            tokenLine("+", row.topPositive)
                        }
                        if !row.topNegative.isEmpty {
                            tokenLine("−", row.topNegative)
                        }
                    }
                }
            }
        }
    }

    private func tokenLine(_ sign: String, _ tokens: [String]) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(sign)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
            Text(tokens.joined(separator: "  "))
                .font(.caption2.monospaced())
                .textSelection(.enabled)
                .lineLimit(2)
        }
    }

    private var batteryBox: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Capability battery (evidence)")
                .font(.caption.weight(.semibold))
            ForEach(report.capabilityBattery) { row in
                HStack(spacing: 6) {
                    Text(row.condition)
                        .font(.caption2.monospaced())
                    if let accuracy = row.accuracy {
                        Text(String(format: "%.0f%%", accuracy * 100))
                            .font(.caption2.monospacedDigit())
                    }
                    if let correct = row.correct, let total = row.total {
                        Text("(\(correct)/\(total))")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}

/// The discriminant-validity grid: cross-concept cosines with high
/// off-diagonals flagged (distinct concepts must not collapse into one
/// direction).
struct CosineMatrixGridView: View {
    let matrix: RunResults.CosineMatrix

    @ViewBuilder
    private var matrixLayerLine: some View {
        if matrix.hasMixedLayers {
            // Should be impossible since the one-layer invariant landed;
            // surfaced rather than averaged over.
            Label(
                "rows record DIFFERENT layers — this matrix is asymmetric and "
                    + "has no defined reading; re-validate",
                systemImage: "exclamationmark.triangle.fill")
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else if let layer = matrix.layer {
            Text("measured at layer \(layer)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Text("layer not recorded — this matrix predates the layer stamp, "
                + "so its depth is unknown")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Discriminant validity (cross-concept cosines)")
                .font(.caption.weight(.semibold))
            // The depth travels with the number. A cosine without its layer
            // cannot be compared to another: the residual stream drifts, so
            // the same concept a few layers apart can be near-orthogonal to
            // itself, and a matrix read at an unstated depth invites reading
            // depth drift as concept distinctness.
            matrixLayerLine
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 2) {
                    GridRow {
                        Text("")
                        ForEach(matrix.concepts.indices, id: \.self) { index in
                            Text(matrix.concepts[index])
                                .foregroundStyle(.secondary)
                        }
                    }
                    ForEach(matrix.concepts.indices, id: \.self) { row in
                        GridRow {
                            Text(matrix.concepts[row])
                                .foregroundStyle(.secondary)
                            ForEach(matrix.concepts.indices, id: \.self) { column in
                                cellText(row: row, column: column)
                            }
                        }
                    }
                }
                .font(.caption2.monospaced())
                .padding(.vertical, 2)
            }
            if flaggedCount > 0 {
                Label(
                    "\(flaggedCount) off-diagonal pair(s) above "
                        + String(
                            format: "|%.1f|", RunResults.CosineMatrix.flagThreshold)
                        + " — check for collapsed directions",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var flaggedCount: Int {
        var count = 0
        for row in matrix.concepts.indices {
            for column in matrix.concepts.indices
            where row < column && matrix.isFlagged(row: row, column: column) {
                count += 1
            }
        }
        return count
    }

    @ViewBuilder
    private func cellText(row: Int, column: Int) -> some View {
        let value = matrix.values[row][column]
        let flagged = matrix.isFlagged(row: row, column: column)
        Text(value.map { String(format: "%.2f", $0) } ?? "nan")
            .foregroundStyle(
                row == column
                    ? AnyShapeStyle(.tertiary)
                    : flagged ? AnyShapeStyle(.orange) : AnyShapeStyle(.primary))
            .fontWeight(flagged ? .semibold : .regular)
    }
}

/// C2: the panel deliberation, read end to end.
///
/// Rebuilt from the run's per-turn records, never from `transcript.md` — the
/// records are the measurement artifact, and reading them here keeps one
/// source of truth. Arms sit side by side at the same play-through, because
/// the question a reader has is almost always "what did the steered panel do
/// differently HERE", and paging is by replicate, because a replicate is one
/// independent transcript and also the unit the statistics use.
struct PanelTranscriptView: View {
    let transcripts: [PanelTranscript.Transcript]
    @State private var replicate: Int = 0
    @State private var expanded: Set<String> = []

    private var replicates: [Int] { PanelTranscript.replicates(in: transcripts) }
    private var arms: [PanelTranscript.Transcript] {
        PanelTranscript.arms(in: transcripts, replicate: replicate)
    }

    var body: some View {
        Section("Panel transcript") {
            if replicates.count > 1 {
                Picker("Play-through", selection: $replicate) {
                    ForEach(replicates, id: \.self) { index in
                        Text("Replicate \(index + 1) of \(replicates.count)").tag(index)
                    }
                }
                .pickerStyle(.segmented)
                Text(
                    "Each play-through is an independent transcript — the unit the "
                        + "statistics treat as one observation.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .top, spacing: 16) {
                ForEach(arms) { transcript in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Text(transcript.condition)
                                .font(.headline)
                            Text("\(transcript.turns.count) turns · \(transcript.totalWords) words")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        ForEach(transcript.turns) { turn in
                            turnRow(turn)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .onAppear {
            if !replicates.contains(replicate) { replicate = replicates.first ?? 0 }
        }
    }

    private func turnRow(_ turn: PanelTranscript.Turn) -> some View {
        DisclosureGroup(
            isExpanded: Binding(
                get: { expanded.contains(turn.id) },
                set: { isExpanded in
                    if isExpanded {
                        expanded.insert(turn.id)
                    } else {
                        expanded.remove(turn.id)
                    }
                })
        ) {
            VStack(alignment: .leading, spacing: 6) {
                Text(turn.output)
                    .font(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                // Visibility IS the propagation claim — who saw this turn is
                // what makes a downstream shift attributable.
                if let routed = turn.routedTo {
                    Text(
                        routed.isEmpty
                            ? "Visible to: nobody (private turn)"
                            : "Visible to: \(routed.count) agent\(routed.count == 1 ? "" : "s")")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 2)
        } label: {
            VStack(alignment: .leading, spacing: 1) {
                Text("\(turn.index). \(turn.title)")
                    .font(.subheadline)
                Text(turn.speaker)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
