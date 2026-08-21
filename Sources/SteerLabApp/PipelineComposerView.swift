import ExperimentKit
import SwiftUI

/// Pipeline Composer (stage 5, sixth review round): the app AUTHORS the
/// chain it runs. Declares `pipeline.stages` + `pipeline.gates` as manifest
/// data (draft-only — the chain is part of the preregistered object), shows
/// the frozen declaration read-only, and surfaces the primary Run Pipeline
/// action (submits the `pipeline` verb through the same bundle path as
/// every remote run; executor/resources come from Remote options).
struct PipelineComposerSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    /// Stages relevant to the current study focus (view filter): an agent
    /// comparison has nothing to extract or sweep, so its chain is
    /// run → evaluate → analyze. Stages already DECLARED in the manifest
    /// always render (a filter must never hide declared data).
    var relevantStages: [String] = PipelineDraft.allStages
    /// Concept studies also declare the promotion rule here — the
    /// screen→confirm gate the funnel's promote step must pass (moved from
    /// the dissolved Science Manifest; it belongs beside the chain whose
    /// promote stage it governs).
    var showsPromotionRule = false
    /// Submission override so the enclosing view can route the pipeline
    /// verb through the shared no-GPU submission gate (2026-07-21 incident,
    /// part 1) — pipelines execute the model like every other bundle verb.
    /// Nil falls back to direct submission.
    var submitAction: (@MainActor () -> Void)? = nil

    @State private var declared = false
    @State private var draft = PipelineDraft()
    @State private var syncedFor: String?

    private var isDraft: Bool { manifest.status == .draft }

    private var visibleStages: [String] {
        PipelineDraft.allStages.filter {
            relevantStages.contains($0) || draft.stages.contains($0)
        }
    }

    /// Live mirror-validation of the CURRENT draft (seventh round: the
    /// composer must not save a block freeze would refuse — order is safe
    /// by construction, but e.g. evaluate-without-run is not). Draft-level
    /// COMPLETENESS runs first (review 2026-08-02, P1): the encoding
    /// silently drops a metric-without-minimum floor, so validating only
    /// the encoded block blessed a save that wrote no gate.
    private var draftViolations: [String] {
        guard declared, isDraft else { return [] }
        return draft.completenessViolations
            + ExperimentStore.pipelineBlockViolations(draft.encoded())
    }

    /// The saved manifest block's violations — gates the Run action.
    private var savedViolations: [String] {
        ExperimentStore.pipelineBlockViolations(manifest.pipeline)
    }

    var body: some View {
        Section("Pipeline (chain runner)") {
            if isDraft {
                editor
            } else if let existing = PipelineDraft.parse(manifest.pipeline) {
                frozenSummary(existing)
            } else {
                Text("No pipeline declared. Duplicate the study to declare one "
                    + "— the chain is part of the preregistered object.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if showsPromotionRule {
                promotionRuleEditor
            }
            if manifest.pipeline != nil {
                runRow
            }
        }
        .onAppear { syncFromManifest() }
        .onChange(of: manifest.name) { syncFromManifest() }
        .onChange(of: manifest.pipeline) { syncFromManifest() }
        .onChange(of: relevantStages) { syncFromManifest() }
    }

    // The ⓘ affordance itself is the shared `InfoButton` (InfoPopover.swift)
    // — this composer is where the pattern originated (2026-07-19 feedback).

    private static let stageInfo = """
    The chain runs its stages in this fixed order, as ONE cluster job with \
    one model load:

    • extract — re-derive each concept's steering vector from its pinned \
    stimuli.
    • validate — score held-out validation scenarios and cross-concept \
    cosines (the artifacts the validation gates read).
    • sweep — try the layer×alpha grid on dev prompts; the declared \
    selection rule picks each concept's winning cell.
    • promote — mint each winning cell as a named agent with a birth \
    certificate.
    • run — the actual study: every task prompt × condition, paired to \
    baseline, with the capability battery per condition.
    • evaluate — blinded A/B judging of this chain's own run.
    • analyze — effect sizes, corrections, residuals over that run.

    A stage that isn't relevant to the current Study Focus is hidden here \
    (already-declared stages always show). Evaluate/analyze require run in \
    the same chain.
    """

    private static let gateInfo = """
    Gates are scientific stop conditions checked BETWEEN stages. A failed \
    gate is not a job failure: the chain stops deliberately, records \
    pipeline-abort.json (which gate, measured vs threshold, the evidence \
    run), and later stages never run.

    • Accuracy floor (0–1): after validate, EVERY concept's declared \
    metric must reach this. WHICH number gates is a declared, hashed part \
    of the study: transfer accuracy (held-out items at the \
    extraction-derived threshold — the legacy floor; a property of the \
    threshold as much as of the vector, structurally near chance for \
    pooled recipes), calibrated accuracy (separation at the held-out \
    classes' own midpoint), calibrated balanced accuracy, or AUC \
    (threshold-free ranking). A report that cannot produce the declared \
    metric fails the gate — an unmeasured number cannot pass a floor, \
    and there is never a fallback to a different metric.
    • Distinctness cap (0–1): no two concepts' vectors may exceed this \
    |cosine| similarity. Catches concepts collapsing into one generic \
    direction.
    • Sweep selection: every concept must have a winning cell under the \
    declared criterion — otherwise there is nothing defensible to promote \
    or run.

    No gates = the chain runs every stage to completion unconditionally \
    (fine for exploration, loud advisory at freeze).
    """

    private var runButtonLabel: String {
        isDraft ? "Run Draft Pipeline (exploratory)" : "Run Frozen Pipeline"
    }

    private static let runHelp = """
        submit the declared chain as ONE job on the active server (executor \
        and resources from Remote options): extract through run with one \
        model load, gate-checked between stages; progress and aborts appear \
        under Pipelines. Draft chains are exploratory — their runs are \
        stamped manifestStatus: draft; freeze first for the preregistered \
        chain
        """

    /// A draft chain is legal EXPLORATORY work — the label and the run's
    /// ledger stamp (manifestStatus) both say which kind this is.
    @ViewBuilder
    private var runRow: some View {
        // Finding 11c, the other direction: this button forces the verb to
        // `pipeline` (`ExperimentPanel.runPipelineRemotely`), so the echo
        // states the chain regardless of what Remote options' verb picker
        // currently shows.
        ExecutionPlanEchoRow(
            plan: ExecutionPlanEcho.describe(
                verb: "pipeline",
                target: .server,
                serverLabel: "the active server",
                dryRun: panel.remoteDryRun,
                declaredPipelineStages: ShardedSubmission.declaredPipelineStages(
                    manifest.pipeline)))
        Button(runButtonLabel) {
            if let submitAction {
                submitAction()
            } else {
                Task { await panel.runPipelineRemotely() }
            }
        }
        .disabled(!savedViolations.isEmpty)
        .help(Self.runHelp)
        // The submission's live outcome, INLINE where the button was
        // pressed (2026-07-19 paper cut: failures previously surfaced
        // only inside the collapsed Remote options).
        if let status = panel.remoteStatus, !status.isEmpty {
            Text(status)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private var editor: some View {
        Toggle("Declare pipeline", isOn: declareBinding)
            .help(
                "the chain runs as one cluster submission and is preregistered "
                    + "with the study — stages and gates are manifest data, "
                    + "declared before anything runs")
        if declared {
            HStack(spacing: 10) {
                ForEach(visibleStages, id: \.self) { stage in
                    Toggle(stage, isOn: stageBinding(stage))
                        .toggleStyle(.checkbox)
                        .font(.caption)
                }
                InfoButton(text: Self.stageInfo)
            }
            .help(
                "stages run in this fixed order; evaluate/analyze need run "
                    + "in the chain (a chain judges its own run)")
            // Gates as PROPER labeled rows (2026-07-19 screenshot: the old
            // single HStack rendered the TextField titles as stray form
            // labels — "e.g. 0.6" floating outside tiny fields). One row
            // per gate, hint INSIDE the field, explanation underneath.
            if draft.stages.contains("validate") || draft.stages.contains("sweep") {
                HStack(spacing: 6) {
                    Text("Gates — scientific stop conditions")
                        .font(.caption.bold())
                    InfoButton(text: Self.gateInfo)
                    Spacer()
                }
                .padding(.top, 2)
            }
            if draft.stages.contains("validate") {
                accuracyFloorRows
                gateRow(
                    label: "Distinctness cap",
                    hint: "0.8",
                    binding: thresholdBinding(\.maxCrossConceptCosine),
                    caption: "abort if any two concepts' vectors exceed this "
                        + "|cosine| similarity (0–1; empty = no cap)")
            }
            if draft.stages.contains("sweep") {
                Toggle(
                    "Selection required — the sweep must pick a winning cell "
                        + "for every concept",
                    isOn: sweepGateBinding)
                    .toggleStyle(.checkbox)
                    .font(.caption)
                    .help(
                        "abort after sweep when any concept has no defensible "
                            + "cell under the declared criterion — nothing to "
                            + "promote or run")
            }
            // "No gates" explains itself (2026-07-21, issue 8), and is calm
            // when gates are legitimately absent: gates only exist for the
            // validate and sweep stages, so e.g. a run → analyze chain
            // never needs one.
            if draft.isGateless {
                let gateable = draft.stages.contains("validate")
                    || draft.stages.contains("sweep")
                Label(
                    gateable
                        ? "No gates declared. Gates are optional stop "
                            + "conditions between chained stages — e.g. a "
                            + "minimum validation score before the sweep "
                            + "runs. Without them this chain runs every "
                            + "stage to completion; fine for exploration, "
                            + "and evidence-grade studies get a freeze note "
                            + "recommending them."
                        : "No gates declared — none apply to this chain. "
                            + "Gates are optional stop conditions between "
                            + "chained stages and exist for validate and "
                            + "sweep; a chain like run → analyze needs none.",
                    systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(draftViolations, id: \.self) { violation in
                Text(violation)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
            Button("Save Pipeline") {
                panel.savePipelineDeclaration(draft)
            }
            .disabled(!draftViolations.isEmpty)
        }
    }

    private static let promotionInfo = """
    The promotion rule is the declared screen→confirm gate: what a \
    concept's screening result must satisfy before its promoted agent \
    enters a confirmation study.

    • FDR threshold (0–1): the concept's screening q-value (BH-FDR across \
    the screen) must fall below this — e.g. 0.05.
    • Dose-monotone: the effect must grow with α across the sweep's \
    ladder, not appear at one cell only.
    • Exceeds random floor: the effect must beat the matched-norm random \
    control's — proof it is the DIRECTION, not perturbation energy.
    • Capability gate (free text): the capability-battery condition the \
    promoted agent must hold, e.g. "battery within 0.05 of baseline".

    Declared here as manifest data, before results exist — the analysis \
    layer reads it; it changes no prompt and is never sent to any model.
    """

    /// The screen→confirm promotion rule (from the dissolved Science
    /// Manifest) — fields disabled when frozen, save offered on drafts.
    @ViewBuilder
    private var promotionRuleEditor: some View {
        @Bindable var panel = panel
        HStack(spacing: 8) {
            TextField(
                "promotion FDR threshold (e.g. 0.05)",
                text: $panel.promotionFDRText)
                .frame(maxWidth: 220)
            Toggle("dose-monotone", isOn: $panel.promotionDoseMonotone)
                .toggleStyle(.checkbox)
            Toggle(
                "exceeds random floor",
                isOn: $panel.promotionExceedsRandomFloor)
                .toggleStyle(.checkbox)
            InfoButton(text: Self.promotionInfo)
        }
        .disabled(!isDraft)
        TextField(
            "capability gate (free text, e.g. battery within 0.05 of baseline)",
            text: $panel.promotionCapabilityGateText)
            .disabled(!isDraft)
        if isDraft {
            Button("Save Promotion Rule") { panel.savePromotionRule() }
                .help(
                    "writes the screen→confirm promotion gate into the "
                        + "manifest (draft-only; the analysis layer reads it)")
        }
    }

    @ViewBuilder
    private func frozenSummary(_ existing: PipelineDraft) -> some View {
        Text("Stages: " + existing.stages.joined(separator: " → "))
            .font(.caption)
        Text(gateSummary(existing))
            .font(.caption2)
            .foregroundStyle(existing.isGateless ? .orange : .secondary)
    }

    private func gateSummary(_ draft: PipelineDraft) -> String {
        var parts: [String] = []
        if let floor = draft.minScenarioAccuracy {
            parts.append("transfer accuracy ≥ \(floor.formatted())")
        }
        if let metric = draft.accuracyFloorMetric,
            let minimum = draft.accuracyFloorMinimum
        {
            parts.append("\(metric) ≥ \(minimum.formatted())")
        }
        if let cap = draft.maxCrossConceptCosine {
            parts.append("cross-concept |cosine| ≤ \(cap.formatted())")
        }
        if draft.requireSelectionForEveryConcept {
            parts.append("every concept must select a cell")
        }
        return parts.isEmpty
            ? "no gates — runs to completion unconditionally"
            : "Gates: " + parts.joined(separator: " · ")
    }

    /// One gate as a labeled Form row: name leading, compact numeric field
    /// trailing (hint INSIDE as its placeholder), plain-language caption
    /// underneath.
    @ViewBuilder
    private func gateRow(
        label: String, hint: String, binding: Binding<String>, caption: String
    ) -> some View {
        LabeledContent(label) {
            TextField(hint, text: binding)
                .textFieldStyle(.roundedBorder)
                .frame(width: 64)
                .multilineTextAlignment(.trailing)
        }
        .font(.caption)
        Text(caption)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var declareBinding: Binding<Bool> {
        Binding(
            get: { declared },
            set: { enabled in
                declared = enabled
                if !enabled { panel.savePipelineDeclaration(nil) }
            })
    }

    private func stageBinding(_ stage: String) -> Binding<Bool> {
        Binding(
            get: { draft.stages.contains(stage) },
            set: { draft.setStage(stage, enabled: $0) })
    }

    private var sweepGateBinding: Binding<Bool> {
        Binding(
            get: { draft.requireSelectionForEveryConcept },
            set: { draft.requireSelectionForEveryConcept = $0 })
    }

    /// The accuracy floor: WHICH number gates, then the minimum. The metric
    /// is manifest data (review 2026-08-02: the gate briefly preferred the
    /// calibrated accuracy implicitly, which made the same frozen manifest
    /// pass or fail by engine version — declaring the metric is the fix).
    /// Selecting a named metric clears the legacy field and vice versa —
    /// the resolver refuses both, so the composer never produces both.
    @ViewBuilder private var accuracyFloorRows: some View {
        Picker("Floor metric", selection: floorMetricBinding) {
            Text("Transfer accuracy (legacy minScenarioAccuracy)").tag("")
            Text("Calibrated accuracy (held-out midpoint)")
                .tag("calibratedAccuracy")
            Text("Calibrated balanced accuracy")
                .tag("calibratedBalancedAccuracy")
            Text("AUC (threshold-free ranking)").tag("auc")
            Text("Transfer accuracy (declared)").tag("transferAccuracy")
        }
        .font(.caption)
        .help(
            "which number the validation floor reads — a declared, hashed "
                + "part of the study. A report that cannot produce the "
                + "declared metric FAILS the gate (older engine, unlabeled "
                + "set, or a degenerate one-class set with no calibration) "
                + "— never a silent fallback to a different metric. For "
                + "pooled recipes (designatedReference) gate on calibrated "
                + "accuracy or AUC: their transfer accuracy sits near "
                + "chance for threshold reasons that say nothing about the "
                + "vector")
        if draft.accuracyFloorMetric == nil {
            gateRow(
                label: "Validation floor",
                hint: "0.6",
                binding: thresholdBinding(\.minScenarioAccuracy),
                caption: "abort after validate unless EVERY concept's "
                    + "TRANSFER accuracy (held-out items at the "
                    + "extraction-derived threshold) reaches this (0–1; "
                    + "empty = no floor)")
        } else {
            gateRow(
                label: "Declared minimum",
                hint: "0.7",
                binding: thresholdBinding(\.accuracyFloorMinimum),
                caption: "abort after validate unless EVERY concept's "
                    + "declared metric reaches this (0–1; required while a "
                    + "metric is selected)")
        }
    }

    private var floorMetricBinding: Binding<String> {
        Binding(
            get: { draft.accuracyFloorMetric ?? "" },
            set: { selected in
                if selected.isEmpty {
                    draft.accuracyFloorMetric = nil
                    draft.accuracyFloorMinimum = nil
                } else {
                    draft.accuracyFloorMetric = selected
                    // Exactly one floor: the named metric replaces the
                    // legacy transfer floor (its value seeds the minimum).
                    if draft.accuracyFloorMinimum == nil {
                        draft.accuracyFloorMinimum = draft.minScenarioAccuracy
                    }
                    draft.minScenarioAccuracy = nil
                }
            })
    }

    private func thresholdBinding(
        _ keyPath: WritableKeyPath<PipelineDraft, Double?>
    ) -> Binding<String> {
        Binding(
            get: { draft[keyPath: keyPath].map { $0.formatted() } ?? "" },
            set: { text in
                draft[keyPath: keyPath] = Double(
                    text.replacingOccurrences(of: ",", with: "."))
            })
    }

    private func syncFromManifest() {
        // Re-seed the local draft whenever the SELECTION, the saved block,
        // or the study type changes — never mid-edit for unrelated view
        // updates. A fresh declaration seeds only the stages the study
        // type makes relevant (a compare-agents chain starts as `run`, not
        // extract → …); type changes reseed an UNSAVED draft.
        let marker = "\(manifest.name)|\(manifest.pipeline != nil)|"
            + relevantStages.joined(separator: ",")
        guard syncedFor != marker else { return }
        syncedFor = marker
        if let existing = PipelineDraft.parse(manifest.pipeline) {
            draft = existing
            declared = true
        } else {
            draft = PipelineDraft(
                stages: PipelineDraft.seedStages(relevant: relevantStages))
            declared = false
        }
    }
}
