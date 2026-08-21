import ExperimentKit
import SteeringKit
import SwiftUI

/// Agents → New Agent → "Optimize from concept vector": the one-form
/// Optimization Composer. Objective first (explicitly chosen, no default),
/// vector RECIPES pinned from the active workspace's catalog, instruments as
/// pickers over real workspace files (with free-text escape hatches), a
/// verbatim-JSON escape hatch for the criterion, and a Declare that creates
/// the real optimization manifest via `OptimizationComposer.declare` — the
/// study then opens in Agents → Optimizations.
struct OptimizationComposerView: View {
    @Bindable var service: ChatService
    var navigate: (WorkbenchSection) -> Void
    /// Called with the created (sanitized) study name — the parent lands on
    /// Agents → Optimizations with that run selected.
    var onDeclared: (String) -> Void
    /// Opens the Optimizations region (kept affordance + the "declare on an
    /// existing draft study" footer, which opens the existing declare flow
    /// there).
    var openOptimizations: () -> Void

    // Catalog + workspace scans (computed on appear/workspace change, never
    // per frame). In a server workspace the catalog is the active server's
    // `/api/vectors`, not the Mac's runs tree.
    @State private var assessments: [ComposerAssessment] = []
    @State private var selectedArtifactIDs: Set<String> = []
    @State private var showUnavailableVectors = false
    @State private var instrumentFiles: [String] = []
    @State private var rubricOptions: [String] = []

    // Objective — starts UNSELECTED; Declare is disabled until chosen.
    @State private var metric: String?
    @State private var choicePromptsFile = ""
    /// Per-concept choice instruments (choicePromptsFiles, 2026-08-02):
    /// with several concepts selected, each gets its OWN row — the map is
    /// what Declare writes, so a multi-concept sweep never scores one
    /// concept's cells on another's items. Keys persist across selection
    /// changes; only the currently selected concepts are composed.
    @State private var choiceFilesByConcept: [String: String] = [:]
    @State private var toleranceText = "\(SweepSelectionRule.defaultCapabilityTolerance)"
    @State private var floorText = "\(SweepSelectionRule.defaultCoherenceFloor)"
    @State private var marginText = ""
    @State private var controlApplyTo = "winner"
    @State private var controlTopKText = ""
    @State private var rubricFile: String?
    @State private var judgeDrafts: [JudgeDraft] = [JudgeDraft()]

    // Criterion-as-JSON escape hatch. `appliedSelection` retains everything
    // the type round-trips (verbatim manifest data — never hashed).
    @State private var criterionJSONExpanded = false
    @State private var criterionJSONText = ""
    @State private var criterionJSONError: String?
    @State private var appliedSelection: ExperimentManifest.SweepSelection?

    // Sweep grid + instruments (defaults mirror SweepSpec()).
    @State private var layerFractionsText = SweepSpecForm.numberListText(
        ExperimentManifest.SweepSpec().layerFractions)
    @State private var alphasText = SweepSpecForm.numberListText(
        ExperimentManifest.SweepSpec().alphas)
    @State private var devPromptsFile = ExperimentManifest.SweepSpec().devPromptsFile
    @State private var batteryFile = ExperimentManifest.SweepSpec().batteryFile
    @State private var maxTokensText = String(ExperimentManifest.SweepSpec().maxTokens)

    // Declare.
    @State private var nameText = ""
    @State private var nameEdited = false
    /// Last programmatic suggestion — user edits are detected as "text no
    /// longer equals the suggestion", so auto-suggest keeps tracking the
    /// selection until the user actually types.
    @State private var lastSuggestedName = ""
    @State private var descriptionText = ""
    @State private var formError: String?
    /// Item 2 (cluster-testing): a Declare & Optimize sweep submission
    /// parked while the shared no-GPU-session dialog asks.
    @State private var pendingModelJob: PendingModelJob?

    private var panel: ExperimentPanel { service.experiments }

    var body: some View {
        Group {
            objectiveSection
            vectorSection
            gridSection
            declareSection
        }
        // Item 2: no-GPU-session warning before the Declare & Optimize
        // server sweep submission.
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        // Moving from one selected concept to several switches the UI to
        // per-concept map rows — migrate a non-empty singular into the
        // previously-single concept's row first, or its instrument renders
        // blank and must be re-picked (review 2026-08-03 round 2, P2).
        .onChange(of: selectedConceptNames) { previous, current in
            let singular = choicePromptsFile
                .trimmingCharacters(in: .whitespaces)
            guard current.count > 1, previous.count <= 1,
                !singular.isEmpty else { return }
            for concept in previous where choiceFilesByConcept[concept] == nil {
                choiceFilesByConcept[concept] = singular
            }
        }
        .task(id: service.cluster.activeWorkspace) {
            if service.cluster.computeTarget == .server {
                await service.catalog.refreshRemoteVectors()
            }
            refreshScans()
        }
    }

    private func refreshScans() {
        if service.cluster.computeTarget == .server {
            assessments = service.catalog.remoteVectors
                .filter {
                    WorkspaceScoping.offerableForServerSteering(
                        substrate: $0.substrate)
                }
                .map { record in
                    ComposerAssessment(
                        id: record.id,
                        label: [
                            record.concept, record.name, record.resolvedMethod,
                        ].compactMap { $0 }.joined(separator: " · "),
                        modelID: record.modelID,
                        verdict: OptimizationComposer.assess(
                            OptimizationComposer.facts(for: record)),
                        // Server catalogs send the sidecar's full timestamp
                        // (2026-08-03); shorten to minutes exactly like the
                        // local rows so same-day re-extractions still read
                        // apart.
                        extracted: record.extracted.map {
                            String($0.prefix(16))
                                .replacingOccurrences(of: "T", with: " ")
                        })
                }
        } else {
            let artifacts = VectorCatalog.scan()
            // Re-extracting one concept in an afternoon yields several
            // artifacts with the same name whose run directories differ only
            // in milliseconds. Label them by what actually differs — or, when
            // nothing does, say they are interchangeable.
            let labels = Dictionary(
                uniqueKeysWithValues: ArtifactDisambiguation.labels(
                    ArtifactDisambiguation.vectorCandidates(artifacts)
                ).map { ($0.id, $0) })
            assessments = artifacts.map { artifact in
                ComposerAssessment(
                    id: artifact.id,
                    label: labels[artifact.id]?.text ?? artifact.label,
                    modelID: artifact.sidecar.modelID,
                    verdict: OptimizationComposer.assess(
                        OptimizationComposer.facts(for: artifact)),
                    extracted: String(
                        artifact.sidecar.extractionDate.prefix(16))
                        .replacingOccurrences(of: "T", with: " "))
            }
        }
        instrumentFiles = OptimizationComposer.scanInstrumentFiles()
        rubricOptions = JudgeRubricStore.list()
    }

    // MARK: Derived state

    /// The picker's choice, or a metric applied verbatim via JSON that the
    /// picker cannot represent.
    private var effectiveMetric: String? {
        metric ?? appliedSelection?.objective?.metric
    }

    /// Substrate scoping (pure rule in
    /// `OptimizationComposer.partitionByModelAvailability`): artifacts for
    /// installed models pick normally; the rest are listed behind a
    /// disclosure, un-selectable — sweeping them here would fail at model
    /// load.
    private var modelPartition:
        (available: [ComposerAssessment], unavailable: [ComposerAssessment])
    {
        OptimizationComposer.partitionByModelAvailability(
            assessments,
            availableModels: service.workspaceModelOptions,
            modelID: \.modelID)
    }

    /// Where "here" is, for the unavailable-model explanation.
    private var substrateLocationLabel: String {
        service.cluster.computeTarget == .server
            ? service.cluster.substrateLabel : "this Mac"
    }

    /// Selection resolves through the AVAILABLE half only, so an artifact
    /// whose model disappears (workspace switch mid-compose) drops out of
    /// the plan instead of declaring a study this substrate cannot sweep.
    private var selectedPins: [OptimizationComposer.ConceptPin] {
        modelPartition.available.compactMap { assessment in
            guard selectedArtifactIDs.contains(assessment.id),
                case .pinnable(let pin) = assessment.verdict
            else { return nil }
            return pin
        }
    }

    private var planProblem: String? {
        OptimizationComposer.planProblem(selectedPins)
    }

    /// The study model a declared optimization will sweep on — known once
    /// concept vectors are picked (it comes from the artifacts). Local
    /// judges with a blank model resolve to THIS model at sweep start.
    private var planModelID: String? {
        planProblem == nil ? selectedPins.first?.modelID : nil
    }

    private var builtJudges: [ExperimentManifest.JudgeRef] {
        judgeDrafts
            .map { draft in
                let provider = draft.provider
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return ExperimentManifest.JudgeRef(
                    name: draft.name.trimmingCharacters(in: .whitespacesAndNewlines),
                    kind: draft.kind,
                    model: draft.model.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty ? nil : draft.model.trimmingCharacters(
                            in: .whitespacesAndNewlines),
                    provider: draft.kind == "openrouter" && !provider.isEmpty
                        ? provider : nil)
            }
            .filter { !$0.name.isEmpty }
    }

    private var declareDisabledReason: String? {
        if effectiveMetric == nil {
            return "choose the selection objective first — the criterion is "
                + "pre-declared data, never a default"
        }
        if let planProblem { return planProblem }
        if effectiveMetric == "judgeScore" {
            if rubricFile == nil {
                return "judgeScore needs a rubric file to pin (prompts/rubrics/)"
            }
            if builtJudges.isEmpty {
                return "add at least one judge — Declare stays disabled "
                    + "until one exists"
            }
        }
        if effectiveMetric == "logprobShift" {
            let concepts = selectedConceptNames
            if concepts.count > 1 {
                let missing = concepts.filter {
                    (choiceFilesByConcept[$0] ?? "")
                        .trimmingCharacters(in: .whitespaces).isEmpty
                }
                if !missing.isEmpty {
                    return "logprobShift needs a choice file per selected "
                        + "concept — missing: "
                        + missing.joined(separator: ", ")
                }
            } else if singleConceptChoiceFile(concepts.first ?? "")
                .trimmingCharacters(in: .whitespaces).isEmpty
            {
                return "logprobShift needs a choice prompts file"
            }
        }
        if nameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "name the optimization study"
        }
        return nil
    }

    // MARK: Objective section

    @ViewBuilder
    private var objectiveSection: some View {
        Section("Optimize from Concept Vector — objective") {
            Text(
                "The evidence-grade path: sweep layer×alpha on dev prompts, "
                    + "select the winning cell by a pre-declared criterion "
                    + "under capability/coherence constraints, then Create "
                    + "Agent from the recommended cell — its birth certificate "
                    + "carries the run, criterion, dev-split hash, and metrics. "
                    + "Declare the objective FIRST; it is hashed manifest data "
                    + "that freeze pins.")
                .font(.caption)
                .foregroundStyle(.secondary)
            ObjectiveMetricPicker(metric: $metric)
            objectiveCaptions
            objectiveInstrumentFields
            constraintFields
            criterionJSONEditor
        }
    }

    @ViewBuilder
    private var objectiveCaptions: some View {
        if metric == "markerDensity" {
            Text(
                "marker density is a smoke-test / manipulation check — never "
                    + "the promotion objective when the claim is about a "
                    + "substantive outcome (the surface-prose confound)")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        if metric == nil, let applied = appliedSelection?.objective?.metric {
            Text(
                "criterion JSON declares objective '\(applied)' (not one of "
                    + "this form's options) — kept verbatim; the engine "
                    + "validates it at Declare")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    /// The concepts a declared sweep would select for, in stable order —
    /// drives one choice-instrument row each under logprobShift.
    private var selectedConceptNames: [String] {
        selectedPins.map(\.concept).sorted()
    }

    @ViewBuilder
    private var objectiveInstrumentFields: some View {
        if effectiveMetric == "logprobShift" {
            let concepts = selectedConceptNames
            if concepts.count > 1 {
                Text(
                    "one choice instrument per selected concept — scoring "
                        + "one concept's cells on another's items dilutes "
                        + "the objective, so the map form is required for "
                        + "multi-concept sweeps; every file is pinned by "
                        + "hash at freeze")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(concepts, id: \.self) { concept in
                    InstrumentPathField(
                        label: "Choice prompts — \(concept)",
                        options: instrumentFiles,
                        text: Binding(
                            get: { choiceFilesByConcept[concept] ?? "" },
                            set: { choiceFilesByConcept[concept] = $0 }),
                        help: "choice rows (prompt + ≥2 options, optional "
                            + "target) scored for '\(concept)' only — the "
                            + "shift objective is mean Δ logP(target) vs "
                            + "baseline")
                    choicePromptsAdvisory(
                        for: choiceFilesByConcept[concept] ?? "")
                }
            } else if let concept = concepts.first {
                // One concept: one row, labeled with the concept, reading
                // whichever representation holds a value (a one-concept map
                // loaded via criterion JSON shows here rather than being
                // silently dropped — review 2026-08-02 round 4, P2; Declare
                // migrates it to the singular field explicitly).
                InstrumentPathField(
                    label: "Choice prompts — \(concept) (JSONL, required)",
                    options: instrumentFiles,
                    text: Binding(
                        get: { singleConceptChoiceFile(concept) },
                        set: {
                            choicePromptsFile = $0
                            choiceFilesByConcept[concept] = $0
                        }),
                    help: "choice rows (prompt + ≥2 options, optional target) — "
                        + "the shift objective is mean Δ logP(target) vs "
                        + "baseline; pinned by hash at freeze")
                choicePromptsAdvisory(for: singleConceptChoiceFile(concept))
            } else {
                InstrumentPathField(
                    label: "Choice prompts (JSONL, required)",
                    options: instrumentFiles,
                    text: $choicePromptsFile,
                    help: "choice rows (prompt + ≥2 options, optional target) — "
                        + "the shift objective is mean Δ logP(target) vs "
                        + "baseline; pinned by hash at freeze")
                choicePromptsAdvisory(for: choicePromptsFile)
            }
        }
        if effectiveMetric == "judgeScore" {
            rubricPicker
            judgesEditor
        }
    }

    /// The one-concept value: the concept's MAP entry when the key exists,
    /// else the singular field. The map is the single source of truth —
    /// every edit path writes it (the single-concept binding writes both),
    /// so a present key is always at least as fresh as the singular, while
    /// the reverse preference resurrected a stale singular after an
    /// A → A+B (edit A's row) → A round trip (review 2026-08-03, P2).
    private func singleConceptChoiceFile(_ concept: String) -> String {
        choiceFilesByConcept[concept] ?? choicePromptsFile
    }

    @ViewBuilder
    private func choicePromptsAdvisory(for file: String) -> some View {
        switch SweepSpecForm.previewChoicePrompts(file: file) {
        case .noFile:
            EmptyView()
        case .ok(let preview):
            Text(SweepSpecForm.choicePromptsSummary(preview))
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .problem(let reason):
            Text(reason + " — advisory preview from the engine's own loader; "
                + "the sweep applies the same check for real at start")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private var rubricPicker: some View {
        Picker("Judge rubric", selection: $rubricFile) {
            Text("choose…").tag(String?.none)
            ForEach(rubricOptions, id: \.self) { path in
                Text(path).tag(String?.some(path))
            }
        }
        .help(
            "versioned rubric under prompts/rubrics/, pinned by hash into the "
                + "study manifest at Declare")
        if let rubricFile {
            FileReferenceRow(label: "judge rubric", path: rubricFile)
        }
        if rubricOptions.isEmpty {
            Text("no rubric files under prompts/rubrics/ in this workspace — "
                + "judgeScore needs one to pin")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private var judgesEditor: some View {
        ForEach($judgeDrafts) { $draft in
            JudgeDraftRow(
                draft: $draft,
                installedModels: service.workspaceModelOptions,
                studyModel: planModelID,
                remove: { id in judgeDrafts.removeAll { $0.id == id } })
        }
        HStack(spacing: 8) {
            Button("Add judge") { judgeDrafts.append(JudgeDraft()) }
                .controlSize(.small)
            judgesCaption
        }
        localJudgeRuleCaptions
        sameModelJudgeHonestyCaption
    }

    /// True when the declared panel is ENTIRELY same-model: every judge is
    /// local and resolves to the study model (blank — which resolves to the
    /// study model at sweep start — or naming it outright).
    private var allJudgesResolveToStudyModel: Bool {
        guard !builtJudges.isEmpty else { return false }
        return builtJudges.allSatisfy { judge in
            guard judge.kind == "local" else { return false }
            let model = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if model.isEmpty { return true }
            guard let planModelID else { return false }
            return model == planModelID
        }
    }

    /// Honesty note, informational not alarming: an all-same-model panel is
    /// fine for shakedown but weak evidence (the model grades its own
    /// steered output).
    @ViewBuilder
    private var sameModelJudgeHonestyCaption: some View {
        if allJudgesResolveToStudyModel {
            Text("same-model judging is convenient for shakedown; for "
                + "evidence runs prefer a Claude judge or a separately "
                + "declared judge panel — the rubric pins support both")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var judgesCaption: some View {
        if builtJudges.isEmpty {
            Text("add at least one judge — Declare stays disabled until one exists")
                .font(.caption2)
                .foregroundStyle(.orange)
        } else {
            Text("the judge panel is pinned into the manifest "
                + "(freeze asks ≥2 judges for evidence)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Local-judge model semantics, stated where the blank default lives:
    /// blank = the study model (the same model that generated the steered
    /// output); a DIFFERENT local model warns about the local engine's
    /// sweep-start refusal (one loaded model).
    @ViewBuilder
    private var localJudgeRuleCaptions: some View {
        if judgeDrafts.contains(where: { $0.kind == "local" }) {
            if let model = planModelID {
                Text("blank = the study model — judges the steered output "
                    + "with the same model that generated it (\(model))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                if let warning = SweepSpecForm.localJudgeSlotWarning(
                    judges: builtJudges, studyModelID: model)
                {
                    Text(warning)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            } else {
                Text("blank = the study model — judges the steered output "
                    + "with the same model that generated it; which model "
                    + "that is comes from the concept vectors picked below")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var constraintFields: some View {
        TextField("Capability tolerance (0–1)", text: $toleranceText)
            .textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.leading)
            .help(
                "the CAPABILITY guardrail: every cell re-runs the capability "
                    + "battery (simple factual/arithmetic checks), and a cell "
                    + "whose battery accuracy drops more than this far below "
                    + "the un-steered baseline is ineligible no matter how "
                    + "well it scores the objective. 0.1 = steering may cost "
                    + "at most 10 points of battery accuracy. Guards against "
                    + "promoting a strength that moves the objective by "
                    + "breaking the model")
        TextField("Coherence floor (distinct-2, 0–1)", text: $floorText)
            .textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.leading)
            .help(
                "the DEGENERATION guardrail: distinct-2 is the fraction of "
                    + "distinct word bigrams in a cell's dev generations — "
                    + "looping, repetitive text scores low. Cells below the "
                    + "floor are ineligible. Guards against strengths that "
                    + "\"win\" by collapsing into repetition (0.45 is the "
                    + "engine default)")
        TextField("Matched-norm random control margin (empty = none)", text: $marginText)
            .textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.leading)
            .help(
                "the SPECIFICITY control: the winning cell is re-run with a "
                    + "RANDOM direction of the same norm at the same "
                    + "layer/strength, and the concept vector must beat that "
                    + "control's objective score by at least this margin or "
                    + "no agent is recommended. 0 = must merely beat it. "
                    + "This is the difference between \"this direction moves "
                    + "the endpoint\" and \"any vector this long moves the "
                    + "endpoint\". Empty skips the control (screens only — "
                    + "evidence-grade promotion should declare one)")
        ControlScopeControls(
            applyTo: $controlApplyTo, topKText: $controlTopKText,
            marginText: marginText)
        Text("tolerance and floor are guardrails (cells failing them are "
            + "struck from selection); the margin is a control the WINNER "
            + "must additionally beat. Hover any field for the full rule.")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    // MARK: Criterion JSON escape hatch

    @ViewBuilder
    private var criterionJSONEditor: some View {
        DisclosureGroup("Edit criterion as JSON", isExpanded: $criterionJSONExpanded) {
            TextEditor(text: $criterionJSONText)
                .font(.caption.monospaced())
                .frame(minHeight: 120)
            HStack(spacing: 8) {
                Button("Apply JSON") { applyCriterionJSON() }
                    .controlSize(.small)
                Button("Seed from form") {
                    criterionJSONText = encodedSelectionJSON()
                    criterionJSONError = nil
                }
                .controlSize(.small)
            }
            if let criterionJSONError {
                Text(criterionJSONError)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
            Text("the criterion is VERBATIM manifest data — never hashed; "
                + "fields this form does not show are kept as long as the "
                + "manifest type round-trips them")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .onChange(of: criterionJSONExpanded) {
            if criterionJSONExpanded, criterionJSONText.isEmpty {
                criterionJSONText = encodedSelectionJSON()
            }
        }
    }

    private func encodedSelectionJSON() -> String {
        let selection = composedSelection()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(selection) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    private func applyCriterionJSON() {
        do {
            let decoded = try JSONDecoder().decode(
                ExperimentManifest.SweepSelection.self,
                from: Data(criterionJSONText.utf8))
            appliedSelection = decoded
            // Overwrite form fields from the decoded block.
            let decodedMetric = decoded.objective?.metric
            metric = decodedMetric.flatMap {
                SweepSelectionRule.knownMetrics.contains($0) ? $0 : nil
            }
            choicePromptsFile = decoded.objective?.choicePromptsFile ?? ""
            choiceFilesByConcept = decoded.objective?.choicePromptsFiles ?? [:]
            // A JSON singular with several concepts already selected means
            // the SAME instrument for every concept — seed the map rows so
            // the per-concept UI shows the declaration instead of blanks
            // (review 2026-08-03 round 2, P2).
            let appliedSingular = choicePromptsFile
                .trimmingCharacters(in: .whitespaces)
            if !appliedSingular.isEmpty, selectedConceptNames.count > 1 {
                for concept in selectedConceptNames
                    where choiceFilesByConcept[concept] == nil
                {
                    choiceFilesByConcept[concept] = appliedSingular
                }
            }
            toleranceText = decoded.constraints?.capabilityTolerance
                .map { "\($0)" } ?? ""
            floorText = decoded.constraints?.coherenceFloor.map { "\($0)" } ?? ""
            marginText = decoded.controls?.matchedNormRandomMargin
                .map { "\($0)" } ?? ""
            controlApplyTo = decoded.controls?.applyTo ?? "winner"
            controlTopKText = decoded.controls?.topK.map(String.init) ?? ""
            criterionJSONError = nil
        } catch {
            criterionJSONError = "\(error)"
        }
    }

    /// The criterion block Declare writes: the last applied JSON (verbatim
    /// content the type round-trips), overlaid with the form's fields.
    private func composedSelection() -> ExperimentManifest.SweepSelection {
        var selection = appliedSelection ?? ExperimentManifest.SweepSelection()
        let effective = effectiveMetric ?? ""
        var objective = selection.objective
            ?? ExperimentManifest.SweepSelection.Objective(metric: effective)
        objective.metric = effective
        let concepts = selectedConceptNames
        if effective == "logprobShift", concepts.count > 1 {
            // The per-concept map — one instrument per selected concept;
            // rows for unselected concepts are simply not composed.
            var map: [String: String] = [:]
            for concept in concepts {
                let value = (choiceFilesByConcept[concept] ?? "")
                    .trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { map[concept] = value }
            }
            objective.choicePromptsFiles = map.isEmpty ? nil : map
            objective.choicePromptsFile = nil
        } else {
            // One concept (or none yet): the singular representation,
            // resolved through the same map-first rule the field displays
            // (review 2026-08-02 round 4 migrated JSON-loaded maps; review
            // 2026-08-03 P2 made the map win so a stale singular can never
            // be what gets declared).
            let choice = (concepts.first.map { singleConceptChoiceFile($0) }
                ?? choicePromptsFile)
                .trimmingCharacters(in: .whitespaces)
            objective.choicePromptsFile =
                (effective == "logprobShift" && !choice.isEmpty) ? choice : nil
            objective.choicePromptsFiles = nil
        }
        selection.objective = objective
        let tolerance = Double(toleranceText.trimmingCharacters(in: .whitespaces))
        let floor = Double(floorText.trimmingCharacters(in: .whitespaces))
        if tolerance != nil || floor != nil {
            var constraints = selection.constraints
                ?? ExperimentManifest.SweepSelection.Constraints()
            constraints.capabilityTolerance = tolerance
            constraints.coherenceFloor = floor
            selection.constraints = constraints
        } else {
            selection.constraints = nil
        }
        let margin = Double(marginText.trimmingCharacters(in: .whitespaces))
        // The controls block is rebuilt whole from form state (which the
        // JSON apply path seeds), so applyTo/topK survive a save instead of
        // silently reverting to winner-only (review 2026-08-03, P1). A
        // topK scope without a margin composes as declared — the shared
        // resolver refuses it with the engine's own message.
        let scopedTopK = controlApplyTo == "topK"
        if margin != nil || scopedTopK {
            selection.controls = ExperimentManifest.SweepSelection.Controls(
                matchedNormRandomMargin: margin,
                applyTo: scopedTopK ? "topK" : nil,
                topK: scopedTopK
                    ? Int(controlTopKText.trimmingCharacters(in: .whitespaces))
                    : nil)
        } else {
            selection.controls = nil
        }
        return selection
    }

    // MARK: Vector section

    @ViewBuilder
    private var vectorSection: some View {
        Section("Concept vectors — recipes to pin") {
            Text(
                "Pick the vector artifact(s) whose recipes this optimization "
                    + "pins: concept + CURRENT stimulus hash + extraction "
                    + "options from each sidecar. The study pins the recipe, "
                    + "never vector bytes — the sweep re-derives vectors from "
                    + "the pinned data. The base model comes from the "
                    + "artifacts; mixing models refuses.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if assessments.isEmpty {
                Text("no vector artifacts in the active workspace catalog — "
                    + "extract one in Data first")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Open Data") { navigate(.data) }
                    .controlSize(.small)
            } else {
                vectorRows
            }
            vectorSelectionCaptions
        }
    }

    /// The available half, tamed (field report 2026-08-02: "a ton of
    /// concept vectors, many duplicates, most unselectable — a mess"):
    /// pinnable artifacts first with interchangeable duplicates (same
    /// concept + model + recipe + stimulus bytes) collapsed to one row,
    /// and un-pinnable artifacts behind a disclosure instead of shuffled
    /// through the pickable list.
    private var pickableRows: (
        pickable: [ComposerAssessment], duplicateCount: Int,
        blocked: [ComposerAssessment]
    ) {
        var pickable: [ComposerAssessment] = []
        var blocked: [ComposerAssessment] = []
        var duplicates = 0
        var keptIndexByIdentity: [String: Int] = [:]
        for assessment in modelPartition.available {
            guard case .pinnable(let pin) = assessment.verdict else {
                blocked.append(assessment)
                continue
            }
            // The CANONICAL full recipe identity — never a hand-selected
            // field subset (review 2026-08-02 round 4, P1). Preference
            // among interchangeable rows: a selected one stays visible,
            // and an uncautioned artifact represents the group over a
            // drift-cautioned one (both pin identically; the caution is
            // about the hidden artifact's history, not the recipe).
            let identity = pin.recipeIdentity
            if let keptIndex = keptIndexByIdentity[identity] {
                let kept = pickable[keptIndex]
                let keptSelected = selectedArtifactIDs.contains(kept.id)
                let newSelected = selectedArtifactIDs.contains(assessment.id)
                if newSelected, keptSelected {
                    // Both picked: both stay visible — a selection must
                    // never be invisible.
                    pickable.append(assessment)
                    continue
                }
                let keptCaution: Bool =
                    if case .pinnable(let keptPin) = kept.verdict {
                        keptPin.caution != nil
                    } else { false }
                if !keptSelected, newSelected || (keptCaution && pin.caution == nil) {
                    pickable[keptIndex] = assessment
                }
                duplicates += 1
                continue
            }
            keptIndexByIdentity[identity] = pickable.count
            pickable.append(assessment)
        }
        return (pickable, duplicates, blocked)
    }

    /// Pickable artifacts grouped by concept (field report 2026-08-02:
    /// "each unique vector should display exactly once — different recipes
    /// for the same concept underneath that concept").
    private var pickableByConcept: [(concept: String, rows: [ComposerAssessment])] {
        let groups = Dictionary(grouping: pickableRows.pickable) { assessment in
            if case .pinnable(let pin) = assessment.verdict {
                return pin.concept
            }
            return "?"
        }
        return groups.keys.sorted().map { ($0, groups[$0] ?? []) }
    }

    @ViewBuilder
    private var vectorRows: some View {
        let partition = modelPartition
        let rows = pickableRows
        if partition.available.isEmpty {
            Text("no vectors for models installed on \(substrateLocationLabel) "
                + "— extract one in Data for an installed model, or switch "
                + "the substrate this workspace runs on")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else if rows.pickable.isEmpty {
            Text("none of the \(partition.available.count) vector artifacts "
                + "here can be pinned — an optimization pins the RECIPE "
                + "(concept + stimulus bytes + extraction options), so the "
                + "concept's stimulus data must exist in THIS workspace's "
                + "prompts/. Extract in this workspace, or import the "
                + "stimulus data alongside the vectors")
                .font(.caption)
                .foregroundStyle(.orange)
        }
        ForEach(pickableByConcept, id: \.concept) { group in
            Text(group.concept)
                .font(.caption.weight(.semibold))
            ForEach(group.rows) { assessment in
                ArtifactPickRow(
                    assessment: assessment,
                    verdict: assessment.verdict,
                    isSelected: selectedArtifactIDs.contains(assessment.id),
                    toggle: { toggleArtifact(assessment.id) })
                    .padding(.leading, 12)
            }
        }
        if rows.duplicateCount > 0 {
            Text("\(rows.duplicateCount) interchangeable duplicate(s) hidden "
                + "— same concept, model, recipe, and stimulus bytes; any "
                + "one of them pins identically")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if !rows.blocked.isEmpty {
            DisclosureGroup {
                ForEach(rows.blocked) { assessment in
                    ArtifactPickRow(
                        assessment: assessment,
                        verdict: assessment.verdict,
                        isSelected: false,
                        toggle: {})
                }
            } label: {
                Text("Vectors that can't be pinned in this workspace "
                    + "(\(rows.blocked.count))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .help(
                "these artifacts' stimulus data is not in this workspace's "
                    + "prompts/, so the recipe cannot be pinned here — the "
                    + "sweep re-derives vectors from pinned stimulus bytes, "
                    + "never from the artifact's tensor bytes")
        }
        if !partition.unavailable.isEmpty {
            DisclosureGroup(isExpanded: $showUnavailableVectors) {
                ForEach(partition.unavailable) { assessment in
                    UnavailableArtifactRow(
                        assessment: assessment,
                        location: substrateLocationLabel)
                }
            } label: {
                Text("Vectors for models not available here "
                    + "(\(partition.unavailable.count))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .help(
                "vectors extracted from models this workspace cannot load — "
                    + "an optimization sweeps on the vector's own base model, "
                    + "so these cannot be selected here")
        }
    }

    private func toggleArtifact(_ id: String) {
        if selectedArtifactIDs.contains(id) {
            selectedArtifactIDs.remove(id)
        } else {
            selectedArtifactIDs.insert(id)
        }
        if !nameEdited {
            let suggestion = OptimizationComposer.suggestedName(
                firstConcept: selectedPins.first?.concept)
            lastSuggestedName = suggestion
            nameText = suggestion
        }
    }

    @ViewBuilder
    private var vectorSelectionCaptions: some View {
        if let planProblem, !selectedArtifactIDs.isEmpty {
            Text(planProblem)
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        ForEach(
            Array(selectedPins.filter { $0.caution != nil }.enumerated()),
            id: \.offset
        ) { _, pin in
            StimulusDriftBadge(
                text: "'\(pin.concept)': stimulus data changed since extraction",
                detail: pin.caution)
        }
        if let model = selectedPins.first?.modelID, planProblem == nil {
            Text("base model (from the artifacts): \(model)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Grid + instruments section

    @ViewBuilder
    private var gridSection: some View {
        Section("Sweep grid & instruments") {
            TextField("Layer fractions (0–1, comma-separated)", text: $layerFractionsText)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .help("network-depth fractions the sweep maps to layers, e.g. 0.5, 0.7, 0.85")
            TextField("Alphas (norm units, comma-separated)", text: $alphasText)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .help(
                    "steering strengths in residual-norm units — the "
                        + "no-injection baseline cell is always implied")
            AlphaMagnitudeWarning(alphasText: alphasText)
            TextField("Max tokens per generation", text: $maxTokensText)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
            InstrumentPathField(
                label: "Dev prompts (JSONL)",
                options: instrumentFiles,
                text: $devPromptsFile,
                help: "open-ended prompts every cell GENERATES on — these "
                    + "texts feed the coherence floor (and marker density, "
                    + "and judge pairing under judgeScore). They measure "
                    + "whether steered generation stays sane. Distinct from "
                    + "the choice prompts, which are two-option items the "
                    + "logprobShift objective SCORES without generating — "
                    + "those measure whether the vector moves the target "
                    + "disposition")
            InstrumentPathField(
                label: "Capability battery (JSONL)",
                options: instrumentFiles,
                text: $batteryFile,
                help: "short tasks with known answers (arithmetic, recall) "
                    + "generated under every cell — the capability "
                    + "tolerance compares each cell's battery accuracy to "
                    + "baseline. It measures whether steering broke the "
                    + "model, not whether it expressed the concept")
            instrumentRolesCaption
            sweepInputPinNote
        }
    }

    /// The three instruments in one breath — each answers a different
    /// question, which is why a sweep needs all of them (field report
    /// 2026-08-02: "how are dev prompts used differently from the choice
    /// prompts? We need explanations here").
    private var instrumentRolesCaption: some View {
        Text(
            "three instruments, three questions — choice prompts: did the "
                + "vector move the concept? (the objective, scored, no "
                + "generation) · dev prompts: is steered text still "
                + "coherent? (the coherence floor, generated) · battery: "
                + "can the model still do basic tasks? (the capability "
                + "tolerance, generated)")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    /// Honest-pin note (STATUS §3): these two files are packaged into run
    /// bundles and covered by the workspace git-cleanliness gate, but they
    /// are NOT hash-pinned in the manifest — the sweep hashes dev prompts
    /// into selection provenance only ex post, and the sweep battery carries
    /// no hash at all. There is no manifest field to record a pin into
    /// (`measurementPins` is the reserved cross-engine gate id for when they
    /// join the verify() surface), so no "Pin now" button is offered — a
    /// button that pinned nothing would be worse than the truth.
    /// Truthful pin note. Its predecessor said these files were "not
    /// hash-pinned" — false since 2026-07-20 (sweep.devPromptsHash +
    /// batteryHash) and doubly false since 2026-08-02 (choice instruments
    /// pin too). A caption claiming the evidence is weaker than it is ages
    /// as badly as one claiming it is stronger.
    private var sweepInputPinNote: some View {
        Text(
            "every instrument file is packaged with the study and "
                + "hash-pinned at freeze (dev prompts, battery, and choice "
                + "prompts alike); drift from the pinned bytes refuses "
                + "sweep start")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    // MARK: Declare section

    @ViewBuilder
    private var declareSection: some View {
        Section("Declare") {
            TextField("Study name (required)", text: $nameText)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .onChange(of: nameText) { _, newValue in
                    if newValue != lastSuggestedName { nameEdited = true }
                }
                .help("the optimization manifest's name under experiments/ — "
                    + "auto-suggested from the first concept, editable")
            TextField("Question or purpose (optional)", text: $descriptionText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .lineLimit(1 ... 3)
            HStack(spacing: 8) {
                Button("Declare & Optimize") { declare(andOptimize: true) }
                    .buttonStyle(.borderedProminent)
                    .disabled(declareDisabledReason != nil)
                    .help(
                        "declare the optimization and start its sweep "
                            + "immediately — watch it in Optimizations and "
                            + "the Activity pane")
                Button("Declare Optimization") { declare(andOptimize: false) }
                    .disabled(declareDisabledReason != nil)
                    .help(
                        "declare only — run the sweep later from "
                            + "Optimizations")
                Button("Open Optimizations") { openOptimizations() }
                    .controlSize(.small)
                    .help("existing optimization runs — grids, recommendations, Create Agent")
            }
            declareCaptions
        }
    }

    @ViewBuilder
    private var declareCaptions: some View {
        if let formError {
            Text(formError)
                .font(.caption2)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        } else if let declareDisabledReason {
            Text(declareDisabledReason)
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Text("creates the draft study with its concept and corpus pins, "
                + "declares the criterion verbatim, and opens the run in "
                + "Optimizations — grid and criterion are manifest data that "
                + "freeze pins; the dev-prompts and battery files are named by "
                + "path only (see the note above)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        Text("or declare a criterion on an existing draft study: the Declare "
            + "an Optimization flow in Agents → Optimizations")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    /// Declare (optionally then start the sweep) and land on Optimizations
    /// with the new run selected. `andOptimize` runs
    /// `ExperimentPanel.runSweep` — the exact button path Optimizations
    /// offers — so progress shows in Activity/Optimizations; its busy-chat
    /// preflight and status surfacing apply unchanged.
    private func declare(andOptimize: Bool) {
        formError = nil
        guard let fractions = SweepSpecForm.parseNumberList(layerFractionsText) else {
            formError = "layer fractions: enter comma-separated numbers, e.g. 0.5, 0.7, 0.85"
            return
        }
        guard let alphas = SweepSpecForm.parseNumberList(alphasText) else {
            formError = "alphas: enter comma-separated numbers, e.g. 0.05, 0.08, 0.13"
            return
        }
        guard let maxTokens = Int(maxTokensText.trimmingCharacters(in: .whitespaces)) else {
            formError = "max tokens: enter a whole number"
            return
        }
        do {
            let plan = try OptimizationComposer.makePlan(selectedPins)
            var spec = ExperimentManifest.SweepSpec(
                layerFractions: fractions,
                alphas: alphas,
                devPromptsFile: devPromptsFile.trimmingCharacters(in: .whitespaces),
                batteryFile: batteryFile.trimmingCharacters(in: .whitespaces),
                maxTokens: maxTokens)
            spec.selection = composedSelection()
            let isJudge = effectiveMetric == "judgeScore"
            let created = try OptimizationComposer.declare(
                name: nameText,
                description: descriptionText,
                plan: plan,
                spec: spec,
                judgeRubricFile: isJudge ? rubricFile : nil,
                judges: isJudge ? builtJudges : [],
                panel: panel)
            if andOptimize {
                // Item 2: same no-GPU-session gate as Optimizations' own
                // Optimize button (local sweeps pass straight through).
                let panel = panel
                ModelJobGPUGate.submit(
                    "optimization sweep", service: service,
                    pending: $pendingModelJob
                ) { await panel.runSweep(experimentName: created) }
            }
            onDeclared(created)
        } catch {
            formError = "\(error)"
        }
    }
}

// MARK: - Assessment + judge draft models

/// One catalog artifact with its recipe verdict — file-scope so the small
/// row subviews can name it.
private struct ComposerAssessment: Identifiable {
    let id: String
    let label: String
    let modelID: String
    let verdict: OptimizationComposer.ArtifactVerdict
    /// Extraction timestamp for display (field report 2026-08-02: dates
    /// distinguish otherwise-similar rows). Local sidecars carry date+time;
    /// the server catalog exposes the date.
    var extracted: String?
}

private struct JudgeDraft: Identifiable, Equatable {
    let id = UUID()
    var name: String = ""
    /// Defaults to `openrouter` (2026-07-24): external judging standardises
    /// there, where the serving backend is pinned and verified.
    var kind: String = "openrouter"
    var model: String = ""
    /// OpenRouter judges only: the pinned serving provider (required for
    /// that kind — no default exists).
    var provider: String = ""
    /// Local-kind escape hatch: the user chose "custom…" to type a model id
    /// not in the installed list.
    var customModel: Bool = false
}

// MARK: - Alpha magnitude warning

/// Non-blocking warning for alphas ≥ 1 (field incident 2026-08-02: "2.2"
/// and "2.5" rode into a real sweep). Alphas are residual-norm FRACTIONS —
/// 1.0 injects a vector as large as the entire residual stream, which has
/// no plausible science reading; ≥ 1 is almost always a typo for a value
/// ten times smaller. A warning, never a refusal: the grid stays the
/// researcher's declaration.
/// Control-scope authoring (review 2026-08-03, P1: the engine grew
/// `controls.applyTo`/`topK` but neither editor could author them, and a
/// save clobbered JSON-applied values back to winner-only). Shared by the
/// composer and the existing-optimization editor.
struct ControlScopeControls: View {
    @Binding var applyTo: String
    @Binding var topKText: String
    let marginText: String

    var body: some View {
        Picker("Control scope", selection: $applyTo) {
            Text("Winner only").tag("winner")
            Text("Top candidates").tag("topK")
        }
        .help(
            "how the margin is applied. Winner only (historical): the argmax "
                + "cell alone is controlled — one disruption-artifact corner "
                + "can veto a grid holding a legitimate winner. Top "
                + "candidates: the top K promotable cells are walked in "
                + "objective order and the FIRST to beat its OWN matched-norm "
                + "control is promoted; every evaluated control is stamped "
                + "into the recommendation's provenance. Either way this is "
                + "a deterministic screening rule — the confirm stage "
                + "establishes direction specificity.")
        if applyTo == "topK" {
            TextField("Candidates to control (K)", text: $topKText)
                .textFieldStyle(.roundedBorder)
                .help(
                    "how many top promotable cells may be tried against "
                        + "their own controls before the concept gets no "
                        + "recommendation. Each try is one more comparison "
                        + "against one random draw, so keep K small (3 is "
                        + "plenty) — a large K raises the chance a noisy "
                        + "control lets a weak cell through the screen.")
            if marginText.trimmingCharacters(in: .whitespaces).isEmpty {
                Label(
                    "top-candidates scope needs a control margin — declare "
                        + "one above (0 = must merely beat the control) or "
                        + "the engine refuses the criterion",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            // Empty K warns too (review 2026-08-03 round 2, P3): it is
            // exactly the state a user pauses in, and the engine will
            // refuse the declare — say so before they hit the button.
            if Int(topKText.trimmingCharacters(in: .whitespaces))
                .map({ $0 < 1 }) ?? true
            {
                Label(
                    "K must be an integer ≥ 1 — how many top candidates "
                        + "may be tried against their own controls",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }
}

struct AlphaMagnitudeWarning: View {
    let alphasText: String

    var body: some View {
        if let alphas = SweepSpecForm.parseNumberList(alphasText) {
            let oversized = alphas.filter { $0 >= 1 }
            if !oversized.isEmpty {
                Label(
                    "alpha \(oversized.map { "\($0)" }.joined(separator: ", "))"
                        + " ≥ 1: alphas are residual-norm fractions, so this "
                        + "injects a vector at least as large as the entire "
                        + "residual stream — almost certainly a typo (0.22, "
                        + "not 2.2). Cells like these can also fool the "
                        + "objective by flattening the model's choices.",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Objective picker

/// The criterion picker starts UNSELECTED — an optimization's selection
/// rule is declared data, so the form never implies a default. judgeScore
/// and logprobShift are the outcome instruments; markerDensity is badged as
/// the smoke-test it is.
private struct ObjectiveMetricPicker: View {
    @Binding var metric: String?

    var body: some View {
        Picker("Selection objective", selection: $metric) {
            Text("choose…").tag(String?.none)
            Text("judge score — paired judging vs baseline "
                + "(outcome instrument; recommended when the claim is about a "
                + "substantive outcome)")
                .tag(String?.some("judgeScore"))
            Text("logprob shift — Δ logP(target) on choice prompts "
                + "(outcome instrument; recommended when the claim is about a "
                + "substantive outcome)")
                .tag(String?.some("logprobShift"))
            Text("marker density — smoke-test / manipulation check")
                .tag(String?.some("markerDensity"))
        }
        .help(
            "the declared selection objective — no default: judgeScore and "
                + "logprobShift are the outcome instruments; markerDensity is "
                + "a diagnostic/manipulation check, never the promotion "
                + "objective when the claim is about a substantive outcome")
    }
}

// MARK: - Instrument file field

/// Picker over real workspace files + free-text path escape hatch + the
/// shared FileReferenceRow inspection affordance. Display only — never a
/// gate; the engine validates instruments at declare/sweep start.
private struct InstrumentPathField: View {
    let label: String
    let options: [String]
    @Binding var text: String
    var help: String = ""

    private var pickerSelection: Binding<String?> {
        Binding<String?>(
            get: { options.contains(text) ? text : nil },
            set: { picked in
                if let picked { text = picked }
            })
    }

    var body: some View {
        Picker(label, selection: pickerSelection) {
            Text("choose…").tag(String?.none)
            ForEach(options, id: \.self) { path in
                Text(path).tag(String?.some(path))
            }
        }
        .help(help)
        TextField("\(label) — workspace-relative path", text: $text)
            .textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.leading)
        if !text.trimmingCharacters(in: .whitespaces).isEmpty {
            FileReferenceRow(
                label: label,
                path: text.trimmingCharacters(in: .whitespaces))
        }
    }
}

// MARK: - Judge draft row

/// One judge in the composer's panel editor. The name is required (empty
/// rows are dropped at Declare). A LOCAL judge scores pairs on an MLX model
/// loaded on this substrate; its model DEFAULTS to the study model (blank =
/// the study model — the same model that generated the steered output),
/// with the installed-models picker and a "custom…" free-text escape hatch
/// for overriding. A Claude judge's model is an Anthropic API id (free
/// text, blank = the default). `JudgeRef` semantics are untouched — this is
/// entry UI only.
private struct JudgeDraftRow: View {
    @Binding var draft: JudgeDraft
    let installedModels: [String]
    /// The plan's model id (from the picked vectors) — what a blank local
    /// judge model resolves to. Nil until vectors are picked.
    let studyModel: String?
    let remove: (UUID) -> Void

    private static let customTag = "custom…"

    var body: some View {
        HStack(spacing: 8) {
            TextField("Judge name (required)", text: $draft.name)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
            Picker("kind", selection: $draft.kind) {
                // Not offered for new judges (2026-07-24) — see
                // `JudgingSectionView.judgeRow`. Listed only when it is
                // already this draft's kind, so an imported panel renders
                // instead of being silently rewritten.
                if draft.kind == "claude" {
                    Text("claude (legacy)").tag("claude")
                }
                Text("openrouter").tag("openrouter")
                Text("local").tag("local")
            }
            .labelsHidden()
            .frame(width: 110)
            modelField
            Button {
                remove(draft.id)
            } label: {
                Image(systemName: "minus.circle")
            }
            .buttonStyle(.plain)
            .help("remove this judge")
        }
    }

    @ViewBuilder
    private var modelField: some View {
        if draft.kind == "local" {
            localModelPicker
            if showsCustomField {
                TextField("Judge model id (blank = the study model)", text: $draft.model)
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.leading)
                    .help("a model id outside the installed list — on this "
                        + "Mac the sweep holds one loaded model, so a local "
                        + "judge model other than the study model refuses at "
                        + "sweep start")
            }
        } else if draft.kind == "openrouter" {
            // No defaults (server rule, 2026-07-19): slug + provider are
            // both pins; the engine refuses a blank at declare/sweep start.
            TextField("Model slug (required)", text: $draft.model)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .help("OpenRouter model slug, e.g. anthropic/claude-opus-4.8 "
                    + "— required; there is no default to resolve")
            TextField("Provider (required)", text: $draft.provider)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .help("the pinned serving provider (allow_fallbacks is off): "
                    + "the same slug served by another backend is a "
                    + "different judge")
        } else {
            TextField("Claude model (optional — blank = default)", text: $draft.model)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .help("Anthropic API model id for the Claude judge; leave "
                    + "blank for the default judge model")
        }
    }

    /// The typed model is shown as free text when the user picked "custom…"
    /// or when a loaded value is not in the installed list.
    private var showsCustomField: Bool {
        draft.customModel
            || (!draft.model.isEmpty && !installedModels.contains(draft.model))
    }

    /// The blank (default) choice names what blank MEANS: the study model
    /// when known, or its dependency on the vector picks when not.
    private var defaultChoiceLabel: String {
        studyModel.map { "study model — \($0)" }
            ?? "study model (known once vectors are picked)"
    }

    private var localModelPicker: some View {
        Picker("model", selection: pickerSelection) {
            Text(defaultChoiceLabel).tag("")
            ForEach(installedModels, id: \.self) { model in
                Text(model).tag(model)
            }
            Text(Self.customTag).tag(Self.customTag)
        }
        .labelsHidden()
        .help("the local MLX model that scores baseline-vs-condition pairs "
            + "against the rubric — blank = the study model (judges the "
            + "steered output with the same model that generated it); "
            + "custom… reveals a free-text field for a model not in the list")
    }

    private var pickerSelection: Binding<String> {
        Binding(
            get: {
                if draft.customModel { return Self.customTag }
                if installedModels.contains(draft.model) { return draft.model }
                return draft.model.isEmpty ? "" : Self.customTag
            },
            set: { picked in
                if picked == Self.customTag {
                    draft.customModel = true
                } else {
                    draft.customModel = false
                    draft.model = picked
                }
            })
    }
}

// MARK: - Artifact row

/// One catalog artifact: selectable when its recipe is pinnable, disabled
/// with the honest refusal when not, and loudly cautioned on stimulus drift
/// (still declarable — the optimization re-derives from current data).
private struct ArtifactPickRow: View {
    let assessment: ComposerAssessment
    let verdict: OptimizationComposer.ArtifactVerdict
    let isSelected: Bool
    let toggle: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Toggle(isOn: Binding(get: { isSelected }, set: { _ in toggle() })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(assessment.label)
                        .font(.callout)
                    Text(
                        [
                            assessment.extracted.map { "extracted \($0)" },
                            assessment.modelID,
                        ].compactMap { $0 }.joined(separator: " · "))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    verdictCaption
                }
            }
            .disabled(!isPinnable)
            Spacer()
        }
    }

    private var isPinnable: Bool {
        if case .pinnable = verdict { return true }
        return false
    }

    @ViewBuilder
    private var verdictCaption: some View {
        switch verdict {
        case .pinnable(let pin):
            if pin.caution != nil {
                StimulusDriftBadge(
                    text: "stimulus data changed since extraction",
                    detail: pin.caution)
            } else {
                Text("pins '\(pin.concept)' @ \(pin.currentStimulusHash.prefix(12))… "
                    + "(\(pin.method.rawValue), \(pin.readingPosition.label))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .refused(let refusal):
            Text(refusal.message)
                .font(.caption2)
                .foregroundStyle(.orange)
                .help(refusal.detail ?? refusal.message)
        }
    }
}

// MARK: - Stimulus drift badge

/// Compact drift caution: icon + short label; the tooltip carries the
/// declare-is-safe explanation plus the precise hash detail. Advisory only
/// — drift never gates declaring.
private struct StimulusDriftBadge: View {
    let text: String
    let detail: String?

    private static let explanation =
        "The concept's stimulus file was edited after this vector was "
        + "extracted. Declaring is safe — the optimization re-derives "
        + "vectors from the current stimulus data and pins the current "
        + "version; results may differ from this saved vector."

    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle")
            .font(.caption2)
            .foregroundStyle(.orange)
            .help(detail.map { Self.explanation + "\n\n" + $0 } ?? Self.explanation)
    }
}

// MARK: - Unavailable artifact row

/// A catalog artifact whose base model is not installed on the active
/// workspace: listed for honesty, disabled for selection — an optimization
/// sweeps on the artifact's own base model, so declaring it here would just
/// fail at model load.
private struct UnavailableArtifactRow: View {
    let assessment: ComposerAssessment
    let location: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(assessment.label)
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(assessment.modelID)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("model not installed on \(location)")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        .help(
            "an optimization pins this vector's base model, and the sweep "
                + "loads that model to re-derive vectors and generate — "
                + "\(location) cannot load it, so the sweep would fail. "
                + "Switch to a substrate with the model installed, or pick a "
                + "vector extracted from an installed model.")
    }
}
