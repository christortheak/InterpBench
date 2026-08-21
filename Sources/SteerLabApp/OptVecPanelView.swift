import ExperimentKit
import SteeringKit
import SwiftUI

/// Data → OptVec: the read-only v1 surface over OptVec campaign bundles and
/// their server-produced results, plus the ONE action — attaching a trained
/// OptVec artifact to a draft study. This file decides nothing: every fact
/// renders from `OptVecPanel` (which reads the two stateless stores), and
/// the attach button calls the panel's action. Bundles and results are
/// WORKSPACE truth; where live scheduler state is unknowable from this Mac,
/// the label says so instead of guessing.
struct OptVecPanelView: View {
    @Bindable var service: ChatService

    private var panel: OptVecPanel { service.optvec }

    var body: some View {
        Form {
            bundlesSection
            runsSection
            attachSection
            if let status = panel.status {
                Section {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .onAppear { panel.refresh() }
    }

    // MARK: - Bundles

    private var bundlesSection: some View {
        Section {
            if panel.bundles.isEmpty {
                Text(
                    "No OptVec dataset bundles under prompts/optvec/ in this "
                        + "workspace. A bundle is one folder holding the nine "
                        + "hashed dataset files, bundle.json, and REPORT.md — "
                        + "see the authoring spec "
                        + "(\(OptVecBundleStore.authoringSpec)).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(panel.bundles) { entry in
                bundleRow(entry)
            }
            if let entry = panel.selectedBundle,
                let report = panel.reports[entry.name]
            {
                bundleDetail(entry: entry, report: report)
            }
        } header: {
            HStack {
                Text("Dataset bundles")
                Spacer()
                Button {
                    panel.refresh()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)
                .help("re-scan prompts/optvec/ and re-run the data check")
            }
        } footer: {
            Text(
                "verdicts are the same checks as `steerlab-server data check "
                    + "optvec` — directives present, all nine files pinned, "
                    + "pinned hashes agreeing with the bytes on disk")
                .font(.caption2)
        }
    }

    private func bundleRow(_ entry: OptVecBundleStore.Entry) -> some View {
        let report = panel.reports[entry.name]
        let isSelected = panel.selectedBundleName == entry.name
        return Button {
            panel.selectedBundleName = isSelected ? nil : entry.name
        } label: {
            HStack(spacing: 8) {
                Image(
                    systemName: report?.ready == true
                        ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(report?.ready == true ? .green : .orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.name)
                        .font(.callout.monospaced().weight(
                            isSelected ? .semibold : .regular))
                    Text(bundleCaption(entry, report: report))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(
            report?.ready == true
                ? "all data checks pass — hashes agree with the bytes on disk"
                : "data check found blockers — select for the requirement list")
    }

    private func bundleCaption(
        _ entry: OptVecBundleStore.Entry, report: OptVecBundleStore.Report?
    ) -> String {
        var parts: [String] = []
        if let issue = entry.bundle?.targetIssue?.lines.first {
            parts.append(issue)
        }
        if let report {
            parts.append(
                report.ready
                    ? "ready — \(report.requirements.count) checks pass"
                    : "\(report.blockers.count) blocker(s)")
        }
        if let failure = entry.decodeFailure {
            parts.append("bundle.json undecodable: \(failure)")
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func bundleDetail(
        entry: OptVecBundleStore.Entry, report: OptVecBundleStore.Report
    ) -> some View {
        if let bundle = entry.bundle {
            VStack(alignment: .leading, spacing: 4) {
                if let direction = bundle.shiftDirection?.lines.first {
                    Text("Shift: \(direction)")
                        .font(.caption)
                        .textSelection(.enabled)
                }
                if let families = bundle.caseFamilies?.lines, !families.isEmpty {
                    Text("Case families: \(families.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .help(families.joined(separator: "\n"))
                }
                if let anchors = bundle.anchorIssues?.lines, !anchors.isEmpty {
                    Text("Anchor issues: \(anchors.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .help(anchors.joined(separator: "\n"))
                }
            }
        }
        ForEach(report.requirements) { requirement in
            requirementRow(requirement)
        }
    }

    private func requirementRow(
        _ requirement: OptVecBundleStore.Requirement
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: requirementIcon(requirement.status))
                .foregroundStyle(requirementColor(requirement.status))
                .imageScale(.small)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(requirement.name)
                        .font(.caption.monospaced().weight(.semibold))
                    Text(requirement.status.rawValue)
                        .font(.caption2)
                        .foregroundStyle(requirementColor(requirement.status))
                    if let rows = requirement.rows {
                        Text("\(rows) rows")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Text(requirement.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                if let sha = requirement.sha256 {
                    Text("sha256 \(sha)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(
                            "paste-able pin — emitted only for files that "
                                + "passed every check")
                }
            }
            Spacer()
        }
    }

    private func requirementIcon(
        _ status: OptVecBundleStore.Requirement.Status
    ) -> String {
        switch status {
        case .invalid: "xmark.octagon.fill"
        case .missing: "questionmark.circle.fill"
        case .partial: "circle.bottomhalf.filled"
        case .present: "checkmark.circle.fill"
        }
    }

    private func requirementColor(
        _ status: OptVecBundleStore.Requirement.Status
    ) -> Color {
        switch status {
        case .invalid: .red
        case .missing: .orange
        case .partial: .yellow
        case .present: .green
        }
    }

    // MARK: - Runs

    private var runsSection: some View {
        Section {
            if panel.runs.isEmpty {
                Text(
                    "No optvec-* runs in this workspace's runs/ tree yet. "
                        + "Training and eval run on the Python server "
                        + "(`steerlab-server optvec …`); imported or paired "
                        + "run directories appear here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(panel.runs) { run in
                runRow(run)
            }
            runDetail
        } header: {
            Text("Campaigns & results")
        } footer: {
            Text(
                "campaign cells are raw sbatch submissions — status here is "
                    + "derived offline from campaign-state.json and COMPLETED "
                    + "markers; live queued/running state needs squeue on the "
                    + "cluster")
                .font(.caption2)
        }
    }

    private func runRow(_ run: OptVecRunStore.RunItem) -> some View {
        let isSelected = panel.selectedRunName == run.name
        return Button {
            panel.selectedRunName = isSelected ? nil : run.name
            panel.loadSelectedRunDetail()
        } label: {
            HStack(spacing: 8) {
                Text(run.kind.label)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
                VStack(alignment: .leading, spacing: 1) {
                    Text(run.name)
                        .font(.caption.monospaced().weight(
                            isSelected ? .semibold : .regular))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let stage = run.stage, stage != "complete" {
                        Text("stage: \(stage)")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("config.json runType \(run.kind.rawValue)")
    }

    @ViewBuilder
    private var runDetail: some View {
        if let status = panel.campaignStatus {
            campaignDetail(status)
        }
        if let progress = panel.trainProgress {
            trainDetail(progress)
        }
        if let report = panel.evalReport {
            evalDetail(report)
        }
        if let report = panel.interpretReport {
            interpretDetail(report)
        }
        if let report = panel.familyReport {
            familyDetail(report)
        }
        if let report = panel.jspaceReport {
            jspaceDetail(report)
        }
        if let report = panel.geometryReport {
            geometryDetail(report)
        }
    }

    private func campaignDetail(
        _ status: OptVecRunStore.CampaignStatus
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            let totals = status.totals
            Text(
                ["completed", "submitted", "planned", "failed", "exhausted"]
                    .compactMap { key -> String? in
                        guard
                            let state = OptVecRunStore.OfflineCellStatus(
                                rawValue: key),
                            let count = totals[state], count > 0
                        else { return nil }
                        return "\(count) \(key)"
                    }
                    .joined(separator: " · "))
                .font(.caption.weight(.semibold))
            ForEach(status.cells) { cell in
                HStack(spacing: 8) {
                    Text(cell.cell.cellID)
                        .font(.caption.monospaced())
                    Text(cell.status.label)
                        .font(.caption2)
                        .foregroundStyle(
                            cell.status == .completed
                                ? .green
                                : cell.status == .failed
                                    || cell.status == .exhausted
                                    ? .red : .secondary)
                    if cell.attempts > 0 {
                        Text("attempt \(cell.attempts)/\(cell.attemptBudget)")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    if let job = cell.lastJobID {
                        Text("job \(job)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                    }
                    Spacer()
                }
            }
        }
    }

    private func trainDetail(
        _ progress: OptVecRunStore.TrainProgress
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            if let stage = progress.stage {
                Text("stage: \(stage)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(stage == "complete" ? .green : .orange)
                    .help(
                        "config.json notes.stage — anything but 'complete' on "
                            + "a dead run is where it stopped")
            }
            if let metrics = progress.lastMetrics {
                Text(trainMetricsLine(metrics))
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
            if let reference = progress.artifactReference {
                HStack(spacing: 6) {
                    Text("artifact: \(reference)")
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Use in attach") {
                        panel.attachArtifactReference = reference
                    }
                    .controlSize(.mini)
                    .help("fill the attach form below with this artifact")
                }
            }
        }
    }

    private func trainMetricsLine(
        _ metrics: OptVecRunStore.TrainMetricsLine
    ) -> String {
        var parts: [String] = []
        if let step = metrics.step { parts.append("step \(step)") }
        if let loss = metrics.loss {
            parts.append(String(format: "loss %.3f", loss))
        }
        if let val = metrics.valShiftRate {
            parts.append(String(format: "val shift %.2f", val))
        }
        if let composite = metrics.valComposite {
            parts.append(String(format: "val composite %.2f", composite))
        }
        return parts.joined(separator: " · ")
    }

    private func evalDetail(_ report: OptVecRunStore.EvalReport) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(
                "claim: \(report.claim ?? "?") · split: "
                    + (report.split ?? "?"))
                .font(.caption.weight(.semibold))
            if let firewall = report.firewall {
                // The engine's own screen-grade disclaimer, verbatim —
                // eval.json is never citable evidence.
                Text(firewall)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
            ForEach(report.doseResponse ?? []) { dose in
                doseRow(dose)
            }
            if let library = report.library {
                libraryRows(library)
            }
        }
    }

    private func doseRow(_ dose: OptVecRunStore.EvalReport.Dose) -> some View {
        var parts: [String] = []
        if let multiple = dose.alphaMultiple {
            parts.append(
                dose.isBaseline == true
                    ? "baseline (α×0)" : String(format: "α×%.2g", multiple))
        }
        if let shift = dose.target?.shiftRate {
            parts.append(String(format: "shift %.2f", shift))
        }
        if let movement = dose.target?.meanLogOddsMovement {
            parts.append(String(format: "Δlogodds %.2f", movement))
        }
        if let flip = dose.anchor?.flipRate {
            parts.append(String(format: "anchor flip %.2f", flip))
        }
        if let kl = dose.anchor?.meanKLFromBaseline {
            parts.append(String(format: "anchor KL %.3f", kl))
        }
        if let delta = dose.capability?.accuracyDelta {
            parts.append(String(format: "cap Δacc %+.2f", delta))
        }
        if let fluency = dose.fluency?.deltaFromBaseline {
            parts.append(String(format: "fluency Δ %+.2f", fluency))
        }
        return Text(parts.joined(separator: " · "))
            .font(.caption.monospaced())
            .textSelection(.enabled)
    }

    @ViewBuilder
    private func libraryRows(_ library: OptVecRunStore.LibraryBlock) -> some View {
        Text(
            "library comparison — \(library.comparedCount ?? 0) vector(s) at "
                + "layer \(library.layer.map(String.init) ?? "?")")
            .font(.caption.weight(.semibold))
        ForEach(
            Array((library.topK ?? []).enumerated()), id: \.offset
        ) { _, entry in
            libraryEntryRow(entry)
        }
    }

    private func libraryEntryRow(
        _ entry: OptVecRunStore.LibraryEntry
    ) -> some View {
        var parts: [String] = []
        if let name = entry.concept ?? entry.name { parts.append(name) }
        if let cosine = entry.cosine {
            parts.append(String(format: "cos %.3f", cosine))
        }
        // Cosines render only beside their null percentile — a bare cosine
        // against a 5376-dim random basis reads as signal when it is noise.
        if let percentile = entry.nullPercentile {
            parts.append(String(format: "null pct %.1f", percentile))
        }
        return Text(parts.joined(separator: " · "))
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
    }

    private func interpretDetail(
        _ report: OptVecRunStore.InterpretReport
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(
                "condition: \(report.condition ?? "unknown") · claim: "
                    + (report.claim ?? "?"))
                .font(.caption.weight(.semibold))
            if let lens = report.stages?.logitLens {
                if let skipped = lens.skipped {
                    stageSkipped("logit lens", skipped)
                } else {
                    stageLine(
                        "logit lens (+v)",
                        (lens.positive?.promoted ?? [])
                            .compactMap(\.piece).prefix(8)
                            .joined(separator: " "))
                    stageLine(
                        "logit lens (−v)",
                        (lens.negative?.promoted ?? [])
                            .compactMap(\.piece).prefix(8)
                            .joined(separator: " "))
                }
            }
            if let sae = report.stages?.sae {
                if let skipped = sae.skipped {
                    stageSkipped("SAE", skipped)
                } else {
                    stageLine(
                        "SAE top features",
                        (sae.topByDecoderCosine ?? [])
                            .compactMap { feature in
                                feature.feature.map {
                                    String(
                                        format: "%d (%.2f)", $0,
                                        feature.cosine ?? 0)
                                }
                            }
                            .prefix(6).joined(separator: ", "))
                }
            }
            if let jlens = report.stages?.jlensSupport {
                if let skipped = jlens.skipped {
                    stageSkipped("J-lens support", skipped)
                } else {
                    stageLine("J-lens support", jlens.lensID ?? "computed")
                }
            }
            if let library = report.stages?.library {
                libraryRows(library)
            }
        }
    }

    private func stageLine(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .trailing)
            Text(value.isEmpty ? "—" : value)
                .font(.caption2.monospaced())
                .textSelection(.enabled)
        }
    }

    private func stageSkipped(_ label: String, _ reason: String) -> some View {
        stageLine(label, "skipped — \(reason)")
    }

    private func familyDetail(
        _ report: OptVecRunStore.FamilyReport
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(familySummaryLine(report))
                .font(.caption.weight(.semibold))
            ForEach(
                Array((report.solutions ?? []).enumerated()), id: \.offset
            ) { _, solution in
                familySolutionRow(solution)
            }
            if let contrast = report.contrastS1S2 {
                if let skipped = contrast.skipped {
                    stageSkipped("S1-vs-S2", skipped)
                } else {
                    stageLine(
                        "S1-vs-S2 Δ",
                        (contrast.deltaS1MinusS2 ?? [:])
                            .sorted { $0.key < $1.key }
                            .map { key, value in
                                value.map {
                                    String(format: "%@ %+.3f", key, $0)
                                } ?? "\(key) —"
                            }
                            .joined(separator: " · "))
                }
            }
        }
    }

    private func familySummaryLine(
        _ report: OptVecRunStore.FamilyReport
    ) -> String {
        var parts: [String] = []
        if let count = report.count { parts.append("\(count) solution(s)") }
        if let conditions = report.conditions, !conditions.isEmpty {
            parts.append(
                conditions.sorted { $0.key < $1.key }
                    .map { "\($0.value)×\($0.key)" }
                    .joined(separator: ", "))
        }
        if let distinct = report.distinctLibraryMatches {
            parts.append("\(distinct) distinct library match(es)")
        }
        if let none = report.noLibraryMatchCount, none > 0 {
            // "Matches nothing in the library" is a first-class (alien)
            // result, not an absence.
            parts.append("\(none) with no library match")
        }
        return parts.joined(separator: " · ")
    }

    private func familySolutionRow(
        _ solution: OptVecRunStore.FamilyReport.Solution
    ) -> some View {
        var parts: [String] = []
        if let condition = solution.condition { parts.append(condition) }
        if let match = solution.topLibraryMatch {
            var text = match.concept ?? match.name ?? "?"
            if let cosine = match.cosine {
                text += String(format: " cos %.3f", cosine)
            }
            if let percentile = match.nullPercentile {
                text += String(format: " (null pct %.1f)", percentile)
            }
            if match.clearsNull == false { text += " — below null" }
            parts.append(text)
        } else if solution.libraryMatchCategory == "no-library-match" {
            parts.append("no library match")
        }
        if let concentration = solution.saeConcentration {
            parts.append(String(format: "SAE conc %.2f", concentration))
        }
        return Text(parts.joined(separator: " · "))
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
    }

    private func jspaceDetail(
        _ report: OptVecRunStore.JSpaceReport
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(
                "claim: \(report.claim ?? "exploratory") · tier: "
                    + (report.evidenceTier ?? "?"))
                .font(.caption.weight(.semibold))
            if let qualification = report.qualification {
                // Mandatory rider: the lens is unqualified (Stage 4
                // unimplemented) — this stamp must accompany any J-space
                // rendering.
                Text(qualification)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
            ForEach(
                Array((report.vectors ?? []).enumerated()), id: \.offset
            ) { _, vector in
                jspaceVectorRows(vector)
            }
            if let family = report.family {
                stageLine(
                    "family PR (unit)",
                    family.raw?.participationRatioUnitNormalized.map {
                        String(format: "%.2f over %d vector(s)", $0,
                               family.count ?? 0)
                    } ?? "—")
            }
        }
    }

    @ViewBuilder
    private func jspaceVectorRows(
        _ vector: OptVecRunStore.JSpaceReport.VectorBlock
    ) -> some View {
        Text(vector.artifact?.name ?? vector.artifact?.reference ?? "vector")
            .font(.caption.monospaced().weight(.semibold))
            .lineLimit(1)
            .truncationMode(.middle)
        ForEach(vector.layers ?? []) { layer in
            jspaceLayerRow(layer)
        }
    }

    private func jspaceLayerRow(
        _ layer: OptVecRunStore.JSpaceReport.LayerAggregate
    ) -> some View {
        var parts: [String] = []
        if let index = layer.layer {
            parts.append(
                "L\(index)\(layer.isInjectionLayer == true ? " (inj)" : "")")
        }
        // Energies render ONLY as energy/null pairs (the writer's own rule).
        if let energy = layer.meanEmergentEnergy,
            let null = layer.meanEmergentEnergyNull
        {
            parts.append(
                String(format: "emergent %.3g (null %.3g", energy, null)
                    + (layer.meanEmergentEnergyNullRatio.map {
                        String(format: ", ×%.1f)", $0)
                    } ?? ")"))
        }
        if let energy = layer.directEnergy, let null = layer.directEnergyNull {
            parts.append(
                String(format: "direct %.3g (null %.3g)", energy, null))
        }
        if let cosine = layer.meanCosineDeltaDirect {
            parts.append(String(format: "cos(Δ,direct) %.2f", cosine))
        }
        return Text(parts.joined(separator: " · "))
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
    }

    private func geometryDetail(
        _ report: OptVecRunStore.GeometryReport
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            stageLine(
                "participation ratio",
                report.participationRatioUnitNormalized.map {
                    String(
                        format: "%.2f (unit-normalized) over %d vector(s) at "
                            + "layer %d",
                        $0, report.count ?? 0, report.layer ?? -1)
                } ?? "—")
            Text(
                "unit-normalized PR answers \"how many directions is this "
                    + "family?\" — the raw PR is dominated by the longest "
                    + "vector")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Attach (the one v1 action)

    private var attachSection: some View {
        @Bindable var panel = panel
        return Section {
            Picker("Draft study", selection: $panel.attachStudyName) {
                Text("select…").tag("")
                ForEach(panel.draftStudyNames, id: \.self) {
                    Text($0).tag($0)
                }
            }
            .help(
                "attach is draft-only — the pinned-artifact concept enters "
                    + "the study's manifest and freezes with it")
            Picker("Trained artifact", selection: $panel.attachArtifactReference) {
                Text("select…").tag("")
                ForEach(panel.attachableArtifacts, id: \.reference) { artifact in
                    Text(artifact.reference).tag(artifact.reference)
                }
            }
            .help(
                "saved artifacts from optvec-train runs in this workspace "
                    + "(norm-backfilled copies preferred — attach refuses a "
                    + "vector born without residual norms)")
            TextField("Concept name", text: $panel.attachConceptName)
                .help(
                    "the study-side name for this direction — an OptVec "
                        + "vector has no source concept, so this names it")
            TextField(
                "Eval run (optional)", text: $panel.attachEvalRun)
                .help(
                    "the optvec-eval run whose test-split eval.json stands "
                        + "in for validate evidence (freeze surfaces it as "
                        + "an advisory); defaults to the run recorded in the "
                        + "artifact's own provenance")
            HStack {
                Button("Attach to study") {
                    panel.attachSelectedArtifact()
                }
                .disabled(
                    panel.attachStudyName.isEmpty
                        || panel.attachConceptName.isEmpty
                        || panel.attachArtifactReference.isEmpty)
                .help(
                    "pin the artifact's bytes (both file hashes) into the "
                        + "draft manifest as a pinnedArtifact concept — the "
                        + "same contract as `steerlab-server experiment "
                        + "attach-artifact`")
                Spacer()
            }
            if let error = panel.attachError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        } header: {
            Text("Attach a trained vector to a study")
        } footer: {
            Text(
                "the artifact is the recipe: verify and freeze re-hash both "
                    + "files, and the confirm study runs through the standard "
                    + "lifecycle (server engine) with control-matrix "
                    + "conditions")
                .font(.caption2)
        }
    }
}
