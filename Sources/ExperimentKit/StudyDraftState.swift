import Foundation
import Observation
import SteeringKit

/// Editable study fields. Selection synchronization consumes supplied values; writes stay in study commands.
@Observable @MainActor
public final class StudyDraftState {
    public var newName = ""
    public var newDescription = ""
    /// Optional exact HF snapshot commit for Create Draft (App gap A7):
    /// empty = auto-pin from the local HF cache at first extract/validate.
    public var newRevision = ""
    public var protocolDescription = ""
    public var taskDescription = ""
    public var outcomeMeasures = ""
    public var studyKind: ExperimentManifest.StudyKind = .modelOutput
    public var studyBaseModelID = ""
    public var selectedVariantToAddID: ModelVariantRecord.ID?
    public var selectedMultiAgentScenarioID: MultiAgentScenarioRecord.ID?
    public var multiAgentIncludeBaseline = true
    public var taskPromptsFile = "prompts/dev/dev-prompts.jsonl"
    public var taskPromptsText = ""
    public var promptMode: ExperimentManifest.PromptMode = .chatAssistant
    public var systemPrompt = ""
    /// The declared reasoning effort (off | low | medium | xhigh) and the
    /// reasoning block's own token cap — the study protocol's
    /// `reasoningEffort`/`reasoningMaxTokens` (2026-09-03). The legacy
    /// `qwenThinkingEnabled` boolean survives below as a derived view for the
    /// route and the toggle that still speak it: on ≡ the template's default
    /// effort, off ≡ off.
    public var reasoningEffort: String = ReasoningEffort.off.rawValue
    public var reasoningMaxTokens: Int?
    public var qwenThinkingEnabled: Bool {
        get { (ReasoningEffort(rawValue: reasoningEffort) ?? .off).isOn }
        set {
            reasoningEffort = ReasoningEffort.legacy(qwenThinkingEnabled: newValue).rawValue
            if !newValue { reasoningMaxTokens = nil }
        }
    }
    public var evaluationPrompt = ""
    public var evaluationStructuredPrompt = ""
    public var judgeModel = ""
    /// Selected rubric file under prompts/rubrics/ ("" = inline draft text
    /// only — freezing a judge-evaluated study requires a pinned file).
    public var judgeRubricFile = ""
    /// Editable judge panel (kind/model rows). Saved into the manifest's
    /// "judges"; >=2 required at freeze for judge-evaluated studies.
    public var judges: [ExperimentManifest.JudgeRef] = []
    public var runTemperature: Double = 0.7
    public var runMaxTokens: Int = 2048
    public var conditionName = ""
    // Native condition editor (App gap A4): the add-vector-condition row's
    // fields. Concept options come from `conditionConceptOptions`.
    public var conditionConcept = ""
    public var conditionLayerText = ""
    public var conditionAlphaText = ""
    public var conditionAlphaInNormUnits = true
    /// Steer or ablate for the add-condition form. Switching resets the
    /// strength: α is typically 1–3 and λ = 2 is already a reflection, so
    /// carrying the number across would silently change what the condition
    /// does.
    public var conditionMode: InterventionPlan.Mode = .add {
        didSet {
            guard oldValue != conditionMode else { return }
            conditionAlphaText = conditionMode == .ablate ? "1" : ""
            clearFormError(.addCondition)
        }
    }
    // Direct concept-attach picker (App gap A8): one-step attach on the
    // Studies draft — concept, method, reading position, grand-mean corpus.
    // Writes ONLY through ExperimentStore.attachConcept (the CLI-attach twin).
    public var attachConceptName = ""
    public var attachMethod: ExtractionMethod = .meanDifference
    /// designatedReference only: the reference stories concept to subtract.
    public var attachReferenceName: String = ""
    /// WHERE the residual stream is read, as a picker holds it. `.recipeDefault`
    /// declares NOTHING — the method keeps its own reading position (last token
    /// for paired methods, mean-from-token-50 for grand mean and designated
    /// reference) and the manifest keeps its bytes. This SUPERSEDES the old
    /// pool-from field: `--pool-from K` is the legacy spelling of exactly
    /// `mean from token K`, the two may never be declared together, and one
    /// coherent control is what a person can reason about.
    public var attachReadingPositionChoice: ReadingPositionChoice = .recipeDefault {
        didSet {
            guard oldValue != attachReadingPositionChoice else { return }
            // Carry the number into the new kind's range. A convenience, not
            // a validation — the field still accepts anything, and the store
            // still answers for what is typed there.
            attachReadingPositionParameter =
                attachReadingPositionChoice
                .steppedParameter(from: attachReadingPositionParameter)
        }
    }
    /// The K/k/i/n beside the position, when it takes one.
    public var attachReadingPositionParameter = 0
    /// HOW the stimulus reaches the model. Raw declares nothing — absent IS
    /// the legacy raw rendering, so an undeclared attach writes what it always
    /// wrote.
    public var attachRendering = ExtractionRenderingChoice()
    /// emotionGrandMean only: extra corpus members (comma-separated) beyond
    /// the attached targets, which are always members.
    public var attachCorpusText = ""
    // Science-manifest editor fields (App gap A2), synced from the selected
    // draft and written back ONLY through ExperimentStore setters.
    public var phaseField = ""
    public var caseFamilyField = ""
    public var samplesPerItemField = 1
    public var seedPolicyField = ""
    /// The study's pinned numeric precision ("" = let the device decide).
    /// Server-honored; the Mac validates it at freeze because this is the
    /// AUTHORING surface (see `ExperimentManifest.dtype`).
    public var studyDtypeField = ""
    public var acknowledgeUnequalOptionLengthsField = false
    public var humanBaselinePathField = ""
    public var promotionFDRText = ""
    public var promotionDoseMonotone = false
    public var promotionExceedsRandomFloor = false
    public var promotionCapabilityGateText = ""
    // Confirmation flow (the concept study's CONFIRM phase): declared
    // perturbation policy inputs; ConfirmationStudy.attach does the
    // expansion + refusals.
    public var confirmAgentID: ModelVariantRecord.ID?
    public var confirmDeltasText = "0.2"
    public var confirmIncludeControl = true
    /// The last refusal each form produced, for rendering beside the control
    /// that produced it (finding 11a). `note(_:severity:)` alone routes a
    /// refusal to the notice feed at the TOP of a long panel, which a
    /// researcher editing a field far below never sees: observed twice on
    /// 2026-07-26, where an α = 0 refusal read as "Add Condition did
    /// nothing" and a control margin was believed saved for days. The notice
    /// feed still gets every message — this is an addition, not a move.
    public var formErrors: [FormField: String] = [:]
    public internal(set) var taskPromptsStatus: String?
    /// Non-editable badge: how many loaded items carry `options`/instrument
    /// fields (preserved verbatim on save; nil when none do).
    public internal(set) var taskPromptsInstrumentSummary: String?
    /// The full loaded records backing the text editor (see
    /// `TaskPromptsDocument`) and which file they came from — save pairs
    /// edited blocks against these so per-item instrument fields survive.
    var taskPromptsDocument: TaskPromptsDocument?
    var taskPromptsDocumentFile: String?
    var taskPromptsReview: TaskPromptsFileReview?
    var syncedSelection: String?
    /// Session-only stash of judge fields per row and per kind (field bug
    /// 2026-08-07): switching a judge's kind swaps the row to that kind's
    /// own field set, and the outgoing kind's values land here so toggling
    /// back restores them — a hand-discovered OpenRouter provider slug must
    /// survive an exploratory toggle to local. Never serialized: the
    /// manifest write keeps only kind-owned fields
    /// (`JudgeRef.keepingKindOwnedFields`), and the stash belongs to the
    /// study it was made on (cleared on selection sync, like seat edits).
    public internal(set) var judgeKindStashes: [Int: [String: JudgeKindStash]] = [:]
    /// Unsaved per-seat edits, keyed by the scenario's seat id.
    ///
    /// An OVERLAY, not the casting: what a study is cast as lives in the
    /// scenario file it pins, and this holds only what the researcher has
    /// changed since it was read (`SeatCasting.state`). Cleared when the
    /// selection changes and when a casting is saved — a seat edit belongs to
    /// the study it was made on.
    public var seatCastingEdits: [String: SeatOccupant] = [:]
    /// Declared control field for the picker.
    public var controlConcept = ""
    /// The control's OWN extraction method — never inherited from a study
    /// concept, which is the fault C2 removed.
    public var controlMethod: ExtractionMethod = .meanDifference
    /// The manual-pieces notes from the last scaffold (rendered under the
    /// button so the scaffold never pretends to be the whole Step-5 matrix).
    public internal(set) var lastControlMatrixNotes: [String] = []

    /// Capture setup as independent values for the shared authoring command.
    func protocolFields(inlineJudgeModel: String) -> StudyProtocolFields {
        var fields = StudyProtocolFields()
        fields.protocolDescription = protocolDescription
        fields.taskDescription = taskDescription
        fields.outcomeMeasures = outcomeMeasures
        fields.studyKind = studyKind
        fields.baseModelID = studyBaseModelID
        fields.promptMode = promptMode
        fields.systemPrompt = systemPrompt
        fields.reasoningEffort = reasoningEffort
        fields.reasoningMaxTokens = reasoningMaxTokens
        fields.dtype = studyDtypeField
        fields.judgeRubricFile = judgeRubricFile
        fields.judges = judges
        fields.evaluationPrompt = evaluationPrompt
        fields.evaluationStructuredPrompt = evaluationStructuredPrompt
        fields.inlineJudgeModel = inlineJudgeModel
        fields.temperature = runTemperature
        fields.maxTokens = runMaxTokens
        fields.seatCastingEdits = seatCastingEdits
        fields.multiAgentIncludeBaseline = multiAgentIncludeBaseline
        fields.taskPromptsFile = taskPromptsFile
        fields.phase = phaseField
        fields.caseFamily = caseFamilyField
        fields.samplesPerItem = samplesPerItemField
        fields.seedPolicy = seedPolicyField
        fields.acknowledgeUnequalOptionLengths = acknowledgeUnequalOptionLengthsField
        return fields
    }

    public enum FormField: String, Sendable, Hashable, CaseIterable {
        case addCondition
        case sweepSpec
        case validationControl
        case promotion
        case rename
        case template
    }

    public struct JudgeKindStash: Sendable, Equatable {
        public var model: String?
        public var provider: String?
        public var revision: String?
        public var dtype: String?
    }

    public func clearFormError(_ field: FormField) {
        formErrors[field] = nil
    }

    public func setJudgeKind(at index: Int, to newKind: String) {
        guard judges.indices.contains(index) else { return }
        let current = judges[index]
        guard current.kind != newKind else { return }
        var stash = judgeKindStashes[index] ?? [:]
        stash[current.kind] = JudgeKindStash(
            model: current.model, provider: current.provider,
            revision: current.revision, dtype: current.dtype)
        judgeKindStashes[index] = stash
        let restored = stash[newKind]
        judges[index].kind = newKind
        judges[index].model = restored?.model
        judges[index].provider = restored?.provider
        judges[index].revision = restored?.revision
        judges[index].dtype = restored?.dtype
    }

    public struct Defaults {
        public var baseModelID: String
        public var judgeModel: String
        public var variantID: ModelVariantRecord.ID?
        public var confirmAgentID: ModelVariantRecord.ID?
        public var scenarioID: MultiAgentScenarioRecord.ID?

        public init(
            baseModelID: String = "", judgeModel: String = "",
            variantID: ModelVariantRecord.ID? = nil,
            confirmAgentID: ModelVariantRecord.ID? = nil,
            scenarioID: MultiAgentScenarioRecord.ID? = nil
        ) {
            self.baseModelID = baseModelID
            self.judgeModel = judgeModel
            self.variantID = variantID
            self.confirmAgentID = confirmAgentID
            self.scenarioID = scenarioID
        }
    }

    public init() {}

    public func synchronize(
        _ selected: ExperimentManifest?, defaults: Defaults, force: Bool = false
    ) -> Bool {
        guard let manifest = selected else {
            syncedSelection = nil
            protocolDescription = ""
            taskDescription = ""
            outcomeMeasures = ""
            studyKind = .modelOutput
            taskPromptsFile = "prompts/dev/dev-prompts.jsonl"
            taskPromptsText = ""
            taskPromptsStatus = nil
            taskPromptsInstrumentSummary = nil
            taskPromptsDocument = nil
            taskPromptsDocumentFile = nil
            taskPromptsReview = nil
            // Workspace-scoped default: the active workspace's model choice
            // (server target → the selected/loaded SERVER model), falling
            // back to that workspace's inventory — never a local MLX id
            // seeding a server study.
            studyBaseModelID =
                defaults.baseModelID
            selectedVariantToAddID = nil
            selectedMultiAgentScenarioID = defaults.scenarioID
            seatCastingEdits = [:]
            multiAgentIncludeBaseline = true
            promptMode = .chatAssistant
            systemPrompt = ""
            reasoningEffort = ReasoningEffort.off.rawValue
            reasoningMaxTokens = nil
            evaluationPrompt = ""
            evaluationStructuredPrompt = ""
            judgeModel = defaults.judgeModel
            judgeRubricFile = ""
            judges = []
            judgeKindStashes = [:]
            runTemperature = 0
            runMaxTokens = 2048
            phaseField = ""
            caseFamilyField = ""
            samplesPerItemField = 1
            seedPolicyField = ""
            studyDtypeField = ""
            acknowledgeUnequalOptionLengthsField = false
            humanBaselinePathField = ""
            promotionFDRText = ""
            promotionDoseMonotone = false
            promotionExceedsRandomFloor = false
            promotionCapabilityGateText = ""
            conditionConcept = ""
            conditionLayerText = ""
            conditionAlphaText = ""
            conditionAlphaInNormUnits = true
            lastControlMatrixNotes = []
            return true
        }
        guard force || syncedSelection != manifest.name else { return false }
        syncedSelection = manifest.name
        protocolDescription = manifest.experimentDescription
        taskDescription = manifest.taskDescription ?? ""
        outcomeMeasures = manifest.outcomeMeasures ?? ""
        studyKind = manifest.studyKind
        // The study type needs no sync: `studyFocus` derives it from the
        // manifest (perturbation policy → confirm; concepts → concept
        // study; …) unless the user overrides via the top-of-page picker.
        studyBaseModelID = manifest.modelID
        selectedVariantToAddID = defaults.variantID
        confirmAgentID = defaults.confirmAgentID
        // The picker names the scenario a researcher CHOSE. For a cast study
        // that is the semantic scenario it was compiled from — the compiled
        // file is deliberately outside the library and would leave the picker
        // reading "select…" on a study that is fully configured.
        selectedMultiAgentScenarioID = defaults.scenarioID
        // Seat edits belong to the study they were made on.
        seatCastingEdits = [:]
        multiAgentIncludeBaseline = manifest.multiAgentIncludeBaseline
        taskPromptsFile = manifest.taskPromptsFile ?? "prompts/dev/dev-prompts.jsonl"
        if manifest.studyKind != .modelOutput {
            taskPromptsText = ""
            taskPromptsStatus = nil
            taskPromptsInstrumentSummary = nil
            taskPromptsDocument = nil
            taskPromptsDocumentFile = nil
            taskPromptsReview = nil
        }
        promptMode = manifest.promptMode ?? .chatAssistant
        systemPrompt = manifest.systemPrompt ?? ""
        reasoningEffort = manifest.resolvedReasoningEffort.rawValue
        reasoningMaxTokens = manifest.reasoningMaxTokens
        evaluationPrompt = manifest.evaluation?.judgePrompt ?? ""
        evaluationStructuredPrompt = manifest.evaluation?.structuredPrompt ?? ""
        judgeModel = manifest.evaluation?.judgeModel ?? defaults.judgeModel
        judgeRubricFile = manifest.judgeRubricFile ?? ""
        judges = manifest.judges ?? []
        // The kind stash belongs to the study it was made on (like seat
        // edits above).
        judgeKindStashes = [:]
        runTemperature = manifest.temperature
        runMaxTokens = manifest.maxTokens
        // Science-manifest editor fields (A2).
        phaseField = manifest.phase ?? ""
        caseFamilyField = manifest.caseFamily ?? ""
        samplesPerItemField = manifest.samplesPerItem ?? 1
        seedPolicyField = manifest.seedPolicy ?? ""
        studyDtypeField = manifest.dtype ?? ""
        acknowledgeUnequalOptionLengthsField =
            manifest.acknowledgeUnequalOptionLengths ?? false
        humanBaselinePathField = manifest.humanBaseline?.path ?? ""
        promotionFDRText = manifest.promotionRule?.fdrThreshold.map { "\($0)" } ?? ""
        promotionDoseMonotone = manifest.promotionRule?.doseMonotone ?? false
        promotionExceedsRandomFloor =
            manifest.promotionRule?.exceedsRandomFloor ?? false
        promotionCapabilityGateText = manifest.promotionRule?.capabilityGate ?? ""
        // Condition editor defaults (A4).
        conditionConcept = manifest.concepts.first?.name ?? ""
        conditionLayerText = ""
        conditionAlphaText = ""
        conditionAlphaInNormUnits = true
        lastControlMatrixNotes = []
        return true
    }
}
