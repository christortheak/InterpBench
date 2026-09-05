import Foundation
import SteeringKit

/// Source-compatible bindings for existing callers. Storage and observable
/// mutation live in the independent state/controller owners.
extension ExperimentPanel {
    public var newName: String {
        get { draft.newName }
        set { draft.newName = newValue }
    }

    public var newDescription: String {
        get { draft.newDescription }
        set { draft.newDescription = newValue }
    }

    public var newRevision: String {
        get { draft.newRevision }
        set { draft.newRevision = newValue }
    }

    public var protocolDescription: String {
        get { draft.protocolDescription }
        set { draft.protocolDescription = newValue }
    }

    public var taskDescription: String {
        get { draft.taskDescription }
        set { draft.taskDescription = newValue }
    }

    public var outcomeMeasures: String {
        get { draft.outcomeMeasures }
        set { draft.outcomeMeasures = newValue }
    }

    public var studyKind: ExperimentManifest.StudyKind {
        get { draft.studyKind }
        set { draft.studyKind = newValue }
    }

    public var studyBaseModelID: String {
        get { draft.studyBaseModelID }
        set { draft.studyBaseModelID = newValue }
    }

    public var selectedVariantToAddID: ModelVariantRecord.ID? {
        get { draft.selectedVariantToAddID }
        set { draft.selectedVariantToAddID = newValue }
    }

    public var selectedMultiAgentScenarioID: MultiAgentScenarioRecord.ID? {
        get { draft.selectedMultiAgentScenarioID }
        set { draft.selectedMultiAgentScenarioID = newValue }
    }

    public var multiAgentIncludeBaseline: Bool {
        get { draft.multiAgentIncludeBaseline }
        set { draft.multiAgentIncludeBaseline = newValue }
    }

    public var taskPromptsFile: String {
        get { draft.taskPromptsFile }
        set { draft.taskPromptsFile = newValue }
    }

    public var taskPromptsText: String {
        get { draft.taskPromptsText }
        set { draft.taskPromptsText = newValue }
    }

    public var promptMode: ExperimentManifest.PromptMode {
        get { draft.promptMode }
        set { draft.promptMode = newValue }
    }

    public var systemPrompt: String {
        get { draft.systemPrompt }
        set { draft.systemPrompt = newValue }
    }

    public var reasoningEffort: String {
        get { draft.reasoningEffort }
        set { draft.reasoningEffort = newValue }
    }

    public var reasoningMaxTokens: Int? {
        get { draft.reasoningMaxTokens }
        set { draft.reasoningMaxTokens = newValue }
    }

    public var qwenThinkingEnabled: Bool {
        get { draft.qwenThinkingEnabled }
        set { draft.qwenThinkingEnabled = newValue }
    }

    public var evaluationPrompt: String {
        get { draft.evaluationPrompt }
        set { draft.evaluationPrompt = newValue }
    }

    public var evaluationStructuredPrompt: String {
        get { draft.evaluationStructuredPrompt }
        set { draft.evaluationStructuredPrompt = newValue }
    }

    public var judgeModel: String {
        get { draft.judgeModel }
        set { draft.judgeModel = newValue }
    }

    public var judgeRubricFile: String {
        get { draft.judgeRubricFile }
        set { draft.judgeRubricFile = newValue }
    }

    public var judges: [ExperimentManifest.JudgeRef] {
        get { draft.judges }
        set { draft.judges = newValue }
    }

    public var runTemperature: Double {
        get { draft.runTemperature }
        set { draft.runTemperature = newValue }
    }

    public var runMaxTokens: Int {
        get { draft.runMaxTokens }
        set { draft.runMaxTokens = newValue }
    }

    public var conditionName: String {
        get { draft.conditionName }
        set { draft.conditionName = newValue }
    }

    public var conditionConcept: String {
        get { draft.conditionConcept }
        set { draft.conditionConcept = newValue }
    }

    public var conditionLayerText: String {
        get { draft.conditionLayerText }
        set { draft.conditionLayerText = newValue }
    }

    public var conditionAlphaText: String {
        get { draft.conditionAlphaText }
        set { draft.conditionAlphaText = newValue }
    }

    public var conditionAlphaInNormUnits: Bool {
        get { draft.conditionAlphaInNormUnits }
        set { draft.conditionAlphaInNormUnits = newValue }
    }

    public var conditionMode: InterventionPlan.Mode {
        get { draft.conditionMode }
        set { draft.conditionMode = newValue }
    }

    public var attachConceptName: String {
        get { draft.attachConceptName }
        set { draft.attachConceptName = newValue }
    }

    public var attachMethod: ExtractionMethod {
        get { draft.attachMethod }
        set { draft.attachMethod = newValue }
    }

    public var attachReferenceName: String {
        get { draft.attachReferenceName }
        set { draft.attachReferenceName = newValue }
    }

    public var attachReadingPositionChoice: ReadingPositionChoice {
        get { draft.attachReadingPositionChoice }
        set { draft.attachReadingPositionChoice = newValue }
    }

    public var attachReadingPositionParameter: Int {
        get { draft.attachReadingPositionParameter }
        set { draft.attachReadingPositionParameter = newValue }
    }

    public var attachRendering: ExtractionRenderingChoice {
        get { draft.attachRendering }
        set { draft.attachRendering = newValue }
    }

    public var attachCorpusText: String {
        get { draft.attachCorpusText }
        set { draft.attachCorpusText = newValue }
    }

    public var phaseField: String {
        get { draft.phaseField }
        set { draft.phaseField = newValue }
    }

    public var caseFamilyField: String {
        get { draft.caseFamilyField }
        set { draft.caseFamilyField = newValue }
    }

    public var samplesPerItemField: Int {
        get { draft.samplesPerItemField }
        set { draft.samplesPerItemField = newValue }
    }

    public var seedPolicyField: String {
        get { draft.seedPolicyField }
        set { draft.seedPolicyField = newValue }
    }

    public var studyDtypeField: String {
        get { draft.studyDtypeField }
        set { draft.studyDtypeField = newValue }
    }

    public var acknowledgeUnequalOptionLengthsField: Bool {
        get { draft.acknowledgeUnequalOptionLengthsField }
        set { draft.acknowledgeUnequalOptionLengthsField = newValue }
    }

    public var humanBaselinePathField: String {
        get { draft.humanBaselinePathField }
        set { draft.humanBaselinePathField = newValue }
    }

    public var promotionFDRText: String {
        get { draft.promotionFDRText }
        set { draft.promotionFDRText = newValue }
    }

    public var promotionDoseMonotone: Bool {
        get { draft.promotionDoseMonotone }
        set { draft.promotionDoseMonotone = newValue }
    }

    public var promotionExceedsRandomFloor: Bool {
        get { draft.promotionExceedsRandomFloor }
        set { draft.promotionExceedsRandomFloor = newValue }
    }

    public var promotionCapabilityGateText: String {
        get { draft.promotionCapabilityGateText }
        set { draft.promotionCapabilityGateText = newValue }
    }

    public var confirmAgentID: ModelVariantRecord.ID? {
        get { draft.confirmAgentID }
        set { draft.confirmAgentID = newValue }
    }

    public var confirmDeltasText: String {
        get { draft.confirmDeltasText }
        set { draft.confirmDeltasText = newValue }
    }

    public var confirmIncludeControl: Bool {
        get { draft.confirmIncludeControl }
        set { draft.confirmIncludeControl = newValue }
    }

    public var formErrors: [FormField: String] {
        get { draft.formErrors }
        set { draft.formErrors = newValue }
    }

    public internal(set) var taskPromptsStatus: String? {
        get { draft.taskPromptsStatus }
        set { draft.taskPromptsStatus = newValue }
    }

    public internal(set) var taskPromptsInstrumentSummary: String? {
        get { draft.taskPromptsInstrumentSummary }
        set { draft.taskPromptsInstrumentSummary = newValue }
    }

    var taskPromptsDocument: TaskPromptsDocument? {
        get { draft.taskPromptsDocument }
        set { draft.taskPromptsDocument = newValue }
    }

    var taskPromptsDocumentFile: String? {
        get { draft.taskPromptsDocumentFile }
        set { draft.taskPromptsDocumentFile = newValue }
    }

    public internal(set) var isValidating: Bool {
        get { localJobs.isValidating }
        set { localJobs.isValidating = newValue }
    }

    public internal(set) var isRunning: Bool {
        get { localJobs.isRunning }
        set { localJobs.isRunning = newValue }
    }

    public internal(set) var isEvaluating: Bool {
        get { localJobs.isEvaluating }
        set { localJobs.isEvaluating = newValue }
    }

    public internal(set) var isExtracting: Bool {
        get { localJobs.isExtracting }
        set { localJobs.isExtracting = newValue }
    }

    public internal(set) var lastExtractDirectory: String? {
        get { localJobs.lastExtractDirectory }
        set { localJobs.lastExtractDirectory = newValue }
    }

    public internal(set) var isSweeping: Bool {
        get { localJobs.isSweeping }
        set { localJobs.isSweeping = newValue }
    }

    public internal(set) var lastValidationDirectory: String? {
        get { localJobs.lastValidationDirectory }
        set { localJobs.lastValidationDirectory = newValue }
    }

    public internal(set) var lastRunDirectory: String? {
        get { localJobs.lastRunDirectory }
        set { localJobs.lastRunDirectory = newValue }
    }

    public internal(set) var lastEvaluationDirectory: String? {
        get { localJobs.lastEvaluationDirectory }
        set { localJobs.lastEvaluationDirectory = newValue }
    }

    public internal(set) var liveRunDirectory: String? {
        get { localJobs.liveRunDirectory }
        set { localJobs.liveRunDirectory = newValue }
    }

    public internal(set) var liveEvaluationDirectory: String? {
        get { localJobs.liveEvaluationDirectory }
        set { localJobs.liveEvaluationDirectory = newValue }
    }

    public internal(set) var liveActiveGeneration: LiveStudyGeneration? {
        get { localJobs.liveActiveGeneration }
        set { localJobs.liveActiveGeneration = newValue }
    }

    public internal(set) var liveActiveJudgment: LiveStudyJudgment? {
        get { localJobs.liveActiveJudgment }
        set { localJobs.liveActiveJudgment = newValue }
    }

    public internal(set) var liveGenerations: [StudyGenerationPreview] {
        get { localJobs.liveGenerations }
        set { localJobs.liveGenerations = newValue }
    }

    public internal(set) var liveJudgments: [StudyJudgePreview] {
        get { localJobs.liveJudgments }
        set { localJobs.liveJudgments = newValue }
    }

    public internal(set) var resultRuns: [StudyRunListItem] {
        get { results.resultRuns }
        set { results.resultRuns = newValue }
    }

    public var selectedResultID: String? {
        get { results.selectedResultID }
        set { results.selectedResultID = newValue }
    }

    public internal(set) var selectedResult: StudyRunDetail? {
        get { results.selectedResult }
        set { results.selectedResult = newValue }
    }

    public internal(set) var selectedResultBrowserItem: RunBrowser.Item? {
        get { results.selectedResultBrowserItem }
        set { results.selectedResultBrowserItem = newValue }
    }

    var syncedSelection: String? {
        get { draft.syncedSelection }
        set { draft.syncedSelection = newValue }
    }

    public var remoteExecutor: String {
        get { submission.remoteExecutor }
        set { submission.remoteExecutor = newValue }
    }

    public var remoteVerb: String {
        get { submission.remoteVerb }
        set { submission.remoteVerb = newValue }
    }

    public var remoteDryRun: Bool {
        get { submission.remoteDryRun }
        set { submission.remoteDryRun = newValue }
    }

    public var remoteGres: String {
        get { submission.remoteGres }
        set { submission.remoteGres = newValue }
    }

    public var remoteWalltime: String {
        get { submission.remoteWalltime }
        set { submission.remoteWalltime = newValue }
    }

    public var remoteResumePolicy: RemoteResumePolicy {
        get { submission.remoteResumePolicy }
        set { submission.remoteResumePolicy = newValue }
    }

    public var remoteParallelJobs: Int {
        get { submission.remoteParallelJobs }
        set { submission.remoteParallelJobs = newValue }
    }

    public internal(set) var remoteStatus: String? {
        get { remoteJobs.remoteStatus }
        set { remoteJobs.remoteStatus = newValue }
    }

    public internal(set) var remoteProfileSummary: String? {
        get { remoteJobs.remoteProfileSummary }
        set { remoteJobs.remoteProfileSummary = newValue }
    }

    public internal(set) var remoteJobID: String? {
        get { remoteJobs.remoteJobID }
        set { remoteJobs.remoteJobID = newValue }
    }

    public internal(set) var remoteLogLines: [String] {
        get { remoteJobs.remoteLogLines }
        set { remoteJobs.remoteLogLines = newValue }
    }

    public internal(set) var remoteLastUploadedBundle: String? {
        get { remoteJobs.remoteLastUploadedBundle }
        set { remoteJobs.remoteLastUploadedBundle = newValue }
    }

    public internal(set) var remoteImportedRunDirectory: String? {
        get { remoteJobs.remoteImportedRunDirectory }
        set { remoteJobs.remoteImportedRunDirectory = newValue }
    }

    public internal(set) var recentServerJobs: [RecentServerJob] {
        get { remoteJobs.recentServerJobs }
        set { remoteJobs.recentServerJobs = newValue }
    }

    public internal(set) var activeServerJob: ActiveServerJob? {
        get { remoteJobs.activeServerJob }
        set { remoteJobs.activeServerJob = newValue }
    }

    public internal(set) var activeSweepJob: ActiveServerJob? {
        get { remoteJobs.activeSweepJob }
        set { remoteJobs.activeSweepJob = newValue }
    }

    public internal(set) var sweepCancelRequested: Bool {
        get { localJobs.sweepCancelRequested }
        set { localJobs.sweepCancelRequested = newValue }
    }

    public internal(set) var studyRunCancelRequested: Bool {
        get { localJobs.studyRunCancelRequested }
        set { localJobs.studyRunCancelRequested = newValue }
    }

    public internal(set) var validationCancelRequested: Bool {
        get { localJobs.validationCancelRequested }
        set { localJobs.validationCancelRequested = newValue }
    }

    public internal(set) var evaluationCancelRequested: Bool {
        get { localJobs.evaluationCancelRequested }
        set { localJobs.evaluationCancelRequested = newValue }
    }

    public internal(set) var extractCancelRequested: Bool {
        get { localJobs.extractCancelRequested }
        set { localJobs.extractCancelRequested = newValue }
    }

    public internal(set) var lastServerRunDirectory: String? {
        get { remoteJobs.lastServerRunDirectory }
        set { remoteJobs.lastServerRunDirectory = newValue }
    }

    public internal(set) var judgeKindStashes: [Int: [String: JudgeKindStash]] {
        get { draft.judgeKindStashes }
        set { draft.judgeKindStashes = newValue }
    }

    public var seatCastingEdits: [String: SeatOccupant] {
        get { draft.seatCastingEdits }
        set { draft.seatCastingEdits = newValue }
    }

    public internal(set) var remoteResultsRuns: [RemoteStampedRunRecord] {
        get { results.remoteResultsRuns }
        set { results.remoteResultsRuns = newValue }
    }

    public internal(set) var remoteResultsStatus: String? {
        get { results.remoteResultsStatus }
        set { results.remoteResultsStatus = newValue }
    }

    public internal(set) var isLoadingRemoteResults: Bool {
        get { results.isLoadingRemoteResults }
        set { results.isLoadingRemoteResults = newValue }
    }

    public var selectedRemoteResultsRun: RemoteStampedRunRecord? {
        get { results.selectedRemoteResultsRun }
        set { results.selectedRemoteResultsRun = newValue }
    }

    public var selectedResultsFile: RunBrowser.FileEntry? {
        get { results.selectedResultsFile }
        set { results.selectedResultsFile = newValue }
    }

    public var controlConcept: String {
        get { draft.controlConcept }
        set { draft.controlConcept = newValue }
    }

    public var controlMethod: ExtractionMethod {
        get { draft.controlMethod }
        set { draft.controlMethod = newValue }
    }

    public internal(set) var lastControlMatrixNotes: [String] {
        get { draft.lastControlMatrixNotes }
        set { draft.lastControlMatrixNotes = newValue }
    }
}
