import Foundation
import SteeringKit

/// Complete setup values captured by an authoring surface. This is a replacement
/// command, not a sparse patch; no observable state or global selection is retained.
public struct StudyProtocolFields: Sendable {
    public var protocolDescription: String = ""
    public var taskDescription: String = ""
    public var outcomeMeasures: String = ""
    public var studyKind: ExperimentManifest.StudyKind = .modelOutput
    public var baseModelID: String = ""
    public var promptMode: ExperimentManifest.PromptMode = .chatAssistant
    public var systemPrompt: String = ""
    public var reasoningEffort: String = "off"
    public var reasoningMaxTokens: Int? = nil
    public var dtype: String = ""
    public var judgeRubricFile: String = ""
    public var judges: [ExperimentManifest.JudgeRef] = []
    public var evaluationPrompt: String = ""
    public var evaluationStructuredPrompt: String = ""
    public var inlineJudgeModel: String = ""
    public var temperature: Double = 0.7
    public var maxTokens: Int = 2048
    public var seatCastingEdits: [String: SeatOccupant] = [:]
    public var multiAgentIncludeBaseline: Bool = true
    public var taskPromptsFile: String = ""
    public var phase: String = ""
    public var caseFamily: String = ""
    public var samplesPerItem: Int = 1
    public var seedPolicy: String = ""
    public var acknowledgeUnequalOptionLengths: Bool = false

    public init() {}
}

/// One decoded input and the hash of exactly those bytes. A caller can capture
/// this alongside a reviewed draft without handing the service an editor object.
public struct StudyProtocolScenario: Sendable {
    public let workspaceRoot: URL
    public let path: String
    public let scenario: MultiAgentScenario
    public let hash: String

    public init(path: String, workspaceRoot: URL) throws {
        let data = try Data(contentsOf: ExperimentStore.resolveProjectPath(path, root: workspaceRoot))
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.path = path
        scenario = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        hash = MultiAgentScenarioStore.hash(data)
    }
}

/// Protocol authorship is shared application work. The caller supplies a reviewed
/// document, complete fields and an optional scenario selection; results contain
/// presentation facts, never callbacks into a panel. Admission and publication
/// use the same cross-process manifest lock as CLI and HTTP draft writers.
public enum StudyProtocolAuthoring {
    public enum Result: Sendable {
        case saved(DraftAuthoringSnapshot, didCompileSeats: Bool, advisories: [String])
        case requiresScenario
    }

    public static func save(
        reviewed: DraftAuthoringSnapshot, fields: StudyProtocolFields,
        scenario selection: StudyProtocolScenario? = nil
    ) throws -> Result {
        let root = reviewed.workspaceRoot
        let repository = ExperimentRepository(workspaceRoot: root)
        return try ManifestFileTransaction.withLock(
            manifestURL: repository.manifestURL(reviewed.manifest.name), workspaceRoot: root
        ) {
            try ManifestFileTransaction.requireCurrent(
                .sha256(reviewed.file.sha256), at: repository.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            if let selection,
                try ManifestFileTransaction.canonicalPath(selection.workspaceRoot)
                    != ManifestFileTransaction.canonicalPath(root)
            {
                throw ExperimentError.refusing(.staleManifest,
                    "The selected scenario belongs to another workspace.",
                    repair: "Select the scenario in the reviewed study's workspace and apply the edit again.")
            }
            return try publish(reviewed: reviewed, fields: fields, selection: selection)
        }
    }

    private static func publish(
        reviewed: DraftAuthoringSnapshot, fields: StudyProtocolFields,
        selection: StudyProtocolScenario?
    ) throws -> Result {
        let root = reviewed.workspaceRoot
        var manifest = reviewed.manifest
        var advisories: [String] = []
        var didCompileSeats = false
        // Validate every protocol field before pinning or publishing. These
        // are the same field policies used by the named store setters, but
        // the setup edit publishes once instead of leaving a partial edit
        // behind if a later policy refuses it.
        let name = manifest.name
        try ManifestDraftEdits.setPhase(
            nilIfEmpty(fields.phase), experimentName: name, manifest: &manifest)
        try ManifestDraftEdits.setCaseFamily(
            nilIfEmpty(fields.caseFamily), experimentName: name, manifest: &manifest)
        try ManifestDraftEdits.setSamplingPolicy(
            samplesPerItem: fields.samplesPerItem,
            seedPolicy: nilIfEmpty(fields.seedPolicy),
            experimentName: name, manifest: &manifest)
        try ManifestDraftEdits.setSamplingProtocol(
            temperature: fields.temperature, maxTokens: fields.maxTokens,
            experimentName: name, manifest: &manifest)
        try ManifestDraftEdits.setAcknowledgeUnequalOptionLengths(
            fields.acknowledgeUnequalOptionLengths, experimentName: name, manifest: &manifest)
        manifest.experimentDescription = fields.protocolDescription
        manifest.taskDescription = nilIfEmpty(fields.taskDescription)
        manifest.outcomeMeasures = nilIfEmpty(fields.outcomeMeasures)
        manifest.studyKind = fields.studyKind
        let baseModelChanged = applyBaseModelChoice(fields.baseModelID, to: &manifest)
        manifest.promptMode = fields.promptMode
        manifest.systemPrompt = nilIfEmpty(fields.systemPrompt)
        // The panel writes the effort spelling and drops the legacy
        // boolean, exactly as `setSamplingProtocol` does; the joint
        // rules (budget beside a non-off effort, on a family with a
        // thinking mode) are refused HERE with the store's sentences so
        // a draft the run would refuse is never saved.
        let reasoningProblems = ReasoningEffort.protocolViolations(
            effort: fields.reasoningEffort, reasoningMaxTokens: fields.reasoningMaxTokens,
            modelID: manifest.modelID)
        guard reasoningProblems.isEmpty else {
            throw ExperimentError.malformed(
                reasoningProblems.joined(separator: "; "),
                repair: "declare a reasoning budget beside a non-off "
                    + "effort (or set the effort to off)")
        }
        manifest.reasoningEffort = fields.reasoningEffort
        manifest.reasoningMaxTokens = fields.reasoningMaxTokens
        manifest.qwenThinkingEnabled = nil
        manifest.dtype = nilIfEmpty(fields.dtype)
        // Judge-rubric versioning: pin the selected rubric file at its
        // CURRENT hash ("" clears the pin — draft-only inline text).
        let rubricFile = fields.judgeRubricFile.trimmingCharacters(in: .whitespacesAndNewlines)
        if rubricFile.isEmpty {
            manifest.judgeRubricFile = nil
            manifest.judgeRubricHash = nil
        } else {
            try JudgeRubricStore.pin(rubricFile, into: &manifest, workspaceRoot: root)
        }
        let panelJudges = fields.judges
            .map {
                ExperimentManifest.JudgeRef(
                    name: $0.name.trimmingCharacters(in: .whitespacesAndNewlines),
                    kind: $0.kind,
                    model: nilIfEmpty($0.model ?? ""),
                    // The provider is a PIN (openrouter judges) — a save
                    // that drops it invalidates the judge (2026-07-19) —
                    // and so are the local-judge revision/dtype pins
                    // (2026-07-23), which this reconstruction previously
                    // dropped.
                    provider: nilIfEmpty($0.provider ?? ""),
                    revision: nilIfEmpty($0.revision ?? ""),
                    dtype: nilIfEmpty($0.dtype ?? ""))
                    // The write funnel serializes only the fields the
                    // judge's kind OWNS (field bug 2026-08-07): no UI
                    // path can leak a kind-foreign field — a local
                    // judge keeping "provider" from its OpenRouter
                    // past — into the manifest.
                    .keepingKindOwnedFields()
            }
            .filter { !$0.name.isEmpty }
        manifest.judges = panelJudges.isEmpty ? nil : panelJudges
        // The explicit declaration (2026-07-22 incident): pinned judges
        // + a chosen rubric file ARE paired judging, so the save WRITES
        // the `evaluation` block — new drafts carry one unambiguous
        // declaration instead of relying on the engines' pin-pair
        // synthesis. Removing the last judge or clearing the rubric
        // clears/updates it coherently on the same save.
        manifest.evaluation = ExperimentStore.evaluationDeclaration(
            judges: panelJudges,
            rubricFile: rubricFile,
            inlineRubric: fields.evaluationPrompt,
            structuredPrompt: nilIfEmpty(fields.evaluationStructuredPrompt),
            inlineJudgeModel: fields.inlineJudgeModel)
        // Saving as one study type NEVER deletes the other type's
        // configuration (the Study Type picker's "switching never
        // deletes anything" promise, enforced here where it was once
        // broken): a multi-agent save keeps concepts, injection
        // conditions, agents, and the task-prompts pin exactly as they
        // were; a model-output save keeps a previously pinned
        // scenario. Carried-but-hidden content surfaces through the
        // type section's hidden-content note.
        // Sampling settings are assigned BEFORE the scenario branch: a
        // compiled scenario binds them, so a recompile must read the values
        // this save is writing, not the previous ones.
        if fields.studyKind == .multiAgent {
            let casting = SeatCasting.state(
                of: manifest,
                selected: selection.map {
                    ($0.scenario, $0.path)
                },
                overlay: fields.seatCastingEdits, workspaceRoot: root)
            switch casting?.form {
            case .uncast, .cast:
                // The compile inputs are manifest fields, so the scenario is
                // (re-)compiled on every setup save: a study whose model,
                // temperature or token budget moved after it was cast would
                // otherwise go on pinning a scenario that binds the previous
                // ones — a silent disagreement between the manifest and the
                // file the run actually reads.
                if let casting {
                    let assignment = baseModelChanged
                        ? SeatCasting.resetToBaseline(casting.assignment)
                        : casting.assignment
                    if baseModelChanged, SeatCasting.isTreated(casting.assignment) {
                        advisories.append(
                            "the base model changed, so every seat was reset "
                                + "to baseline — agents built on the previous "
                                + "model cannot run in this panel. Recast the "
                                + "seats below.")
                    }
                    try SeatCasting.compile(
                        assignment, semantic: casting.semantic,
                        semanticPath: casting.semanticPath, into: &manifest, workspaceRoot: root)
                    // Provenance must describe the semantic bytes actually used,
                    // never a second read that might have changed during compilation.
                    // A saved casting keeps its original semantic provenance until
                    // a newly selected semantic document is explicitly supplied.
                    manifest.multiAgentSemanticScenarioHash = selection.flatMap {
                        PanelComposition.isSemantic($0.scenario) ? $0.hash : nil
                    } ?? reviewed.manifest.multiAgentSemanticScenarioHash
                    didCompileSeats = true
                }
            case .legacyBound:
                // A hand-bound scenario is pinned as it stands — its casting
                // is inside the file, and this save must not rewrite it.
                guard let scenario = selection else { break }
                manifest.multiAgentScenarioPath =
                    scenario.path
                manifest.multiAgentScenarioHash =
                    scenario.hash
                manifest.multiAgentSemanticScenarioPath = nil
                manifest.multiAgentSemanticScenarioHash = nil
            case nil:
                return .requiresScenario
            }
            manifest.multiAgentIncludeBaseline = fields.multiAgentIncludeBaseline
        } else if !fields.taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            _ = try ExperimentStore.pinTaskPrompts(
                fields.taskPromptsFile, into: &manifest, workspaceRoot: root)
        } else {
            // An EMPTIED prompts field on a model-output study is the
            // one explicit clear this save performs.
            manifest.taskPromptsFile = nil
            manifest.taskPromptsHash = nil
        }
        let saved = try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed)
        return .saved(saved, didCompileSeats: didCompileSeats, advisories: advisories)
    }

    /// A model change invalidates revision and variant pins. An empty field is
    /// no requested change, preserving headless callers that did not choose one.
    @discardableResult
    static func applyBaseModelChoice(_ choice: String, to manifest: inout ExperimentManifest) -> Bool {
        let requested = choice.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requested.isEmpty, requested != manifest.modelID else { return false }
        manifest.modelID = requested
        manifest.modelRevision = nil
        manifest.variantConditions.removeAll()
        return true
    }

    private static func nilIfEmpty(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
