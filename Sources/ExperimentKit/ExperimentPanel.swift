import CryptoKit
import Foundation
import Observation
import SteeringKit

/// Observable state for the domain-neutral study builder (both front ends):
/// create a draft protocol, attach concepts (pinned at their CURRENT hashes
/// and the concepts panel's current extraction options), capture baseline
/// and steered conditions, verify, freeze, duplicate, and launch measured
/// runs through the same ExperimentKit task path as the CLI.
@Observable @MainActor
public final class ExperimentPanel {

    public internal(set) weak var host: ChatService?

    public private(set) var experiments: [ExperimentManifest] = []
    public var selectedName: String? {
        didSet {
            if oldValue != selectedName {
                selectedResultID = nil
                selectedResult = nil
                selectedResultBrowserItem = nil
                // The last remote-freeze answer belongs to the previous
                // selection.
                remoteFreezeGateFailure = nil
                remoteFreezeAdvisories = []
                remoteFreezeIdentityWarning = nil
                remoteFreezeIdentityNote = nil
                remoteFreezeCanSyncDraft = false
                // Pipeline listings belong to the previous selection —
                // clear immediately, refresh in the background (sixth
                // round: stale chains must never render under the wrong
                // study).
                pipelineRuns = []
                localPipelineRuns = []
                // The focus override is per-study view state.
                studyFocusOverride = nil
                refresh()
                syncDraftFieldsFromSelection(force: true)
                Task { await refreshPipelineRuns() }
            }
        }
    }
    public var newName = ""
    public var newDescription = ""
    /// Optional exact HF snapshot commit for Create Draft (App gap A7):
    /// empty = auto-pin from the local HF cache at first extract/validate.
    public var newRevision = ""

    /// Display labels for the studies in `experiments`, read once per
    /// refresh — a picker body must never touch the disk.
    public private(set) var displayLabels: [String: String] = [:]

    /// A study the panel wants renamed right now: set when New Study mints a
    /// placeholder name, cleared when the UI opens the rename affordance.
    /// One click creating `new-study-2026-08-06` is only an improvement if
    /// the naming step follows immediately.
    public var renameInvitation: String?

    /// What to call this study in lists: its label when one is set, its
    /// canonical name otherwise. The canonical name stays the identity —
    /// directory, run stamps, CLI arguments — so surfaces render it as
    /// secondary text rather than dropping it.
    public func displayName(_ manifest: ExperimentManifest) -> String {
        displayLabels[manifest.name] ?? manifest.name
    }
    public var protocolDescription = ""
    public var taskDescription = ""
    public var outcomeMeasures = ""
    public var studyKind: ExperimentManifest.StudyKind = .modelOutput

    /// THE study-type setter (2026-07-19 second pass): one control answers
    /// "what kind of study is this?". Sets the view classification AND —
    /// on drafts — PERSISTS the declared type into the manifest
    /// immediately (durable across selection changes; an empty comparison
    /// no longer re-derives back to Concept study), keeping the
    /// engine-facing studyKind consistent. Frozen studies get the view
    /// lens only.
    public func setStudyType(_ type: StudyIntent) {
        studyFocusOverride = type
        guard let manifest = selected, manifest.status == .draft else { return }
        studyKind = type.mappedKind
        do {
            try ExperimentStore.setStudyType(type, experimentName: manifest.name)
            refresh()
        } catch {
            note(
                "Couldn't save the study-type change — the study file may be "
                    + "locked or the disk full. Details: \(error)",
                severity: .error)
        }
    }

    /// Cross-section entry to CONFIRM-phase authoring (Agents →
    /// Optimizations → "Open Studies — Confirm agent"). A confirmation is
    /// a NEW preregistered study (docs: iterate by duplicating, never
    /// editing), created here by ALLOWLIST from the screen study —
    /// `ExperimentStore.createConfirmationDraft(fromScreen:named:)` — and
    /// the screen study is left untouched: its phase never flips (P1 fix
    /// 2026-07-19; the previous behavior mutated the selected study in
    /// place), and none of its EXECUTION state (sweep, pipeline,
    /// conditions, agent arms, promotion rule, old perturbation policy)
    /// travels (second-pass P1 fix, same day: the earlier
    /// duplicate-then-patch inherited it all). The new draft:
    /// - is named "<screen>-confirm" (collision-suffixed),
    /// - declares phase "confirm" and type conceptStudy (confirmation is
    ///   the concept study's phase 2 since the 2026-07-19 fold-in),
    /// - pins `screenTaskPromptsHash` from the SCREEN study's
    ///   `taskPromptsHash` (the held-out-pool rule's reference),
    /// - carries ONLY the scientific pins (model identity, concepts,
    ///   generation settings, measurement declarations — the store
    ///   function's doc lists every field's disposition); its own task
    ///   prompts start EMPTY: confirm needs a held-out pool, so readiness
    ///   demands one and verify's disjointness rule can never be
    ///   satisfied by the inherited screen pool.
    /// The manual path is unchanged — a researcher may still set the
    /// Funnel phase by hand on a draft they choose to edit; this fixes the
    /// shortcut, it forbids nothing (tool, not police officer).
    public func createConfirmationDraft(from screenName: String) {
        guard let screen = try? ExperimentStore.load(name: screenName) else {
            note(
                "Couldn't find study '\(screenName)' in this workspace — "
                    + "a confirmation draft is built from the screen "
                    + "study's pins, so it must exist here first.",
                severity: .error)
            return
        }
        var candidate = "\(screenName)-confirm"
        var counter = 1
        while (try? ExperimentStore.load(name: candidate)) != nil {
            counter += 1
            candidate = "\(screenName)-confirm-\(counter)"
        }
        do {
            _ = try ExperimentStore.createConfirmationDraft(
                fromScreen: screenName, named: candidate)
            refresh()
            selectedName = candidate
            studyFocusOverride = .conceptStudy
            // Preselect the agent this screen study promoted when it is in
            // the library (the caller knows the optimization; the birth
            // certificate names the experiment).
            if let promoted = confirmableAgents.first(where: {
                $0.artifact.promotion?.experiment == screenName
            }) {
                confirmAgentID = promoted.id
            }
            let poolNote = screen.taskPromptsHash == nil
                ? " The screen study has no pinned task prompts, so no "
                    + "screen-pool reference could be recorded — pin the "
                    + "screen pool before freezing either study."
                : ""
            note(
                "Created confirmation draft '\(candidate)' — a confirmation "
                    + "is a new preregistered study, so '\(screenName)' is "
                    + "untouched. Pick held-out task prompts (the screen "
                    + "pool is pinned as the reference the confirm pool "
                    + "must stay disjoint from)." + poolNote,
                severity: .success)
        } catch {
            note(
                "Couldn't create the confirmation draft — nothing was "
                    + "changed. Details: \(error)",
                severity: .error)
        }
    }
    public var studyBaseModelID = ChatService.availableModels.first?.id ?? ""
    public var selectedVariantToAddID: ModelVariantRecord.ID?
    public var selectedMultiAgentScenarioID: MultiAgentScenarioRecord.ID?
    public var multiAgentIncludeBaseline = true
    public var taskPromptsFile = "prompts/dev/dev-prompts.jsonl"
    public var taskPromptsText = ""
    public var promptMode: ExperimentManifest.PromptMode = .chatAssistant
    public var systemPrompt = ""
    public var qwenThinkingEnabled = false
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
    /// Optional pooling token ("" = the method default reading position:
    /// last token for paired methods, mean-from-token-50 for grand mean).
    public var attachPoolFromText = ""
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
    public private(set) var status: String?
    /// Where notes persist (A15). The shared per-workspace feed in the app;
    /// tests inject a hermetic instance.
    public var notices: PanelNotices = .shared

    /// A15: the ONE way panel events speak — sets the legacy single-slot
    /// status string (unchanged UI) AND appends a persistent notice to the
    /// workspace feed, verbatim. High-frequency progress mirrors (per-line
    /// job logs, per-generation live updates) deliberately keep writing
    /// `status` directly: they are transient telemetry, and flooding the
    /// ring would evict the failures the feed exists to keep.
    public func note(_ message: String, severity: PanelNotice.Severity = .info) {
        status = message
        notices.record(source: "Studies", severity: severity, message: message)
    }

    /// A form whose refusals must render AT the control, not only in the
    /// panel-top notice area.
    public enum FormField: String, Sendable, Hashable, CaseIterable {
        case addCondition
        case sweepSpec
        case validationControl
        case promotion
        case rename
        case template
    }

    /// The last refusal each form produced, for rendering beside the control
    /// that produced it (finding 11a). `note(_:severity:)` alone routes a
    /// refusal to the notice feed at the TOP of a long panel, which a
    /// researcher editing a field far below never sees: observed twice on
    /// 2026-07-26, where an α = 0 refusal read as "Add Condition did
    /// nothing" and a control margin was believed saved for days. The notice
    /// feed still gets every message — this is an addition, not a move.
    public var formErrors: [FormField: String] = [:]

    /// Refuse a form action: record the message for inline rendering AND
    /// speak it through the normal notice path.
    public func refuse(
        _ field: FormField, _ message: String,
        severity: PanelNotice.Severity = .error
    ) {
        formErrors[field] = message
        note(message, severity: severity)
    }

    /// Clear a form's inline refusal — on success, or when the researcher
    /// edits the inputs that caused it.
    public func clearFormError(_ field: FormField) {
        formErrors[field] = nil
    }

    public private(set) var taskPromptsStatus: String?
    /// Non-editable badge: how many loaded items carry `options`/instrument
    /// fields (preserved verbatim on save; nil when none do).
    public private(set) var taskPromptsInstrumentSummary: String?
    /// The full loaded records backing the text editor (see
    /// `TaskPromptsDocument`) and which file they came from — save pairs
    /// edited blocks against these so per-item instrument fields survive.
    private var taskPromptsDocument: TaskPromptsDocument?
    private var taskPromptsDocumentFile: String?
    public private(set) var isValidating = false
    public private(set) var isRunning = false
    public private(set) var isEvaluating = false
    /// A local extraction is executing (A11). Extraction shares the GPU with
    /// runs/validation/sweeps, so all four flags gate each other.
    public private(set) var isExtracting = false
    public private(set) var lastExtractDirectory: String?
    /// A local sweep is executing (Optimizations' Optimize). Sweeps share the GPU
    /// with runs/validation, so all three flags gate each other.
    public private(set) var isSweeping = false
    public private(set) var lastValidationDirectory: String?
    public private(set) var lastRunDirectory: String?
    public private(set) var lastEvaluationDirectory: String?
    public private(set) var liveRunDirectory: String?
    public private(set) var liveEvaluationDirectory: String?
    public private(set) var liveActiveGeneration: LiveStudyGeneration?
    public private(set) var liveActiveJudgment: LiveStudyJudgment?
    public private(set) var liveGenerations: [StudyGenerationPreview] = []
    public private(set) var liveJudgments: [StudyJudgePreview] = []
    public private(set) var violations: [String] = []
    public private(set) var resultRuns: [StudyRunListItem] = []
    public var selectedResultID: String? {
        didSet {
            if oldValue != selectedResultID {
                loadSelectedResult()
            }
        }
    }
    public private(set) var selectedResult: StudyRunDetail?
    /// The selected run's Results-browser item, built ONCE per selection
    /// (F10): `RunBrowser.item(at:)` reads config.json and probes sweep.csv
    /// synchronously, and the study-detail body re-evaluates on every live
    /// progress note — that read must never sit inline in a SwiftUI body.
    public private(set) var selectedResultBrowserItem: RunBrowser.Item?
    @ObservationIgnored private var browserItemMemo = RunBrowser.MemoizedItem()
    private var syncedSelection: String?
    public var remoteExecutor = "local"
    // Defaults match the common intent — "run my study" — not the most
    // cautious combination: verify+dryRun defaults produced submissions that
    // appeared to do nothing. Dry run stays available as an explicit toggle.
    public var remoteVerb = "run"
    public var remoteDryRun = false
    /// One-shot request from a cross-link (e.g. Optimizations' "Submit Bundle:
    /// sweep") that the Studies view open its Run-on-Server disclosure so the
    /// preconfigured verb/study are visible, not hidden behind a collapsed
    /// group. The Studies view consumes (and clears) it on appear.
    public var pendingRevealRemoteControls = false
    public var remoteGres = "A100"
    /// 4 hours (2026-08-03): the old 30-minute default walltime-killed the
    /// first real 27B sweep twenty minutes in — 30m fit only smoke tests,
    /// and a killed job costs a full queue wait to retry. 4h covers a
    /// trimmed-grid multi-concept sweep or validate with headroom; the
    /// field stays editable for anything bigger.
    public var remoteWalltime = "04:00:00"
    /// Resume-on-checkpoint policy for Slurm submissions — DEFAULT ON with
    /// the server's shipped limit (2026-07-22 incident: a checkpointed run
    /// continuing is what the researcher asked for by submitting it; OFF is
    /// the surprising choice). Sent only for the slurm executor; what was
    /// sent is stamped into the submission transcript line.
    public var remoteResumePolicy = RemoteResumePolicy()
    /// "Parallel GPU jobs" (2026-07-22): shard a Slurm run across K sibling
    /// GPU jobs (default 1 = single job, the historical path). Encoded on
    /// the submission only when it applies (`ShardedSubmission` rule);
    /// execution logistics only — never in the manifest or content hash.
    public var remoteParallelJobs = 1
    public private(set) var remoteStatus: String?
    public private(set) var remoteProfileSummary: String?
    // Persisted so a researcher can reconnect to a running Slurm job after an
    // app restart (Phase C exit criterion).
    public private(set) var remoteJobID: String? = UserDefaults.standard.string(forKey: "SteerLabRemoteJobID") {
        didSet { UserDefaults.standard.set(remoteJobID, forKey: "SteerLabRemoteJobID") }
    }
    public private(set) var remoteLogLines: [String] = []
    public private(set) var remoteLastUploadedBundle: String?
    public private(set) var remoteImportedRunDirectory: String?
    /// The active server's `runs/` listing (read-only browse; refreshed on
    /// demand, cleared when no server workspace is active).
    public private(set) var remoteRuns: [RemoteRunRecord] = []
    /// Chain-runner runs for the selected experiment (stage 5 affordance).
    public private(set) var pipelineRuns: [ClusterClient.PipelineRunSummary] = []
    /// Imported/local chains for the selected experiment — read from the
    /// portable ledger (preferred) or the raw ledger in the local runs tree.
    public private(set) var localPipelineRuns: [ClusterClient.PipelineRunSummary] = []
    private var remoteLogTask: Task<Void, Never>?

    /// Server jobs submitted from THIS panel this session (run-verb jobs and
    /// bundle submissions), id-first so a researcher can always copy the id
    /// and reconnect later — jobs persist on the server across app restarts.
    public struct RecentServerJob: Identifiable, Sendable, Equatable {
        public let id: String
        public let verb: String
        public let study: String
        public var state: String

        public init(id: String, verb: String, study: String, state: String) {
            self.id = id
            self.verb = verb
            self.study = study
            self.state = state
        }
    }

    public private(set) var recentServerJobs: [RecentServerJob] = []

    /// The in-flight server experiment job (durable, server-side): drives the
    /// visible Cancel control next to the run status. Cleared on terminal
    /// state; a timed-out follow keeps it set — the job is still running and
    /// must stay cancellable.
    public struct ActiveServerJob: Sendable, Equatable {
        public let id: String
        public let verb: String
        public let study: String
    }

    public private(set) var activeServerJob: ActiveServerJob?

    /// The in-flight server SWEEP job, tracked in its own slot: the
    /// Optimizations Cancel button must never cancel a study-run or Submit
    /// Bundle job that happens to occupy `activeServerJob`/`remoteJobID`.
    /// Same lifecycle as `activeServerJob` (cleared on terminal state; a
    /// timed-out follow keeps it set — the job is still cancellable).
    public private(set) var activeSweepJob: ActiveServerJob?

    /// A local-sweep cancellation was requested (Optimizations' Cancel):
    /// `ExperimentTasks.sweep` polls this between generations and stops
    /// after the current one, keeping partial grid rows. Reset when the
    /// next local sweep starts.
    public private(set) var sweepCancelRequested = false

    /// Cancel the in-flight Optimize. Server route: cancel the tracked
    /// sweep job (its own slot — never another flow's job id). Local route:
    /// raise the flag the sweep observes between generations; partial rows
    /// stay in the run directory either way.
    public func cancelSweep() async {
        if let job = activeSweepJob {
            loadStoredRemoteToken()
            guard let client = remoteClient else {
                note("invalid server URL", severity: .error)
                return
            }
            do {
                try await client.cancelJob(job.id)
                note("cancel requested for server sweep job \(job.id) "
                    + "('\(job.study)') — cancelling…", severity: .warning)
            } catch {
                note("cancel failed for sweep job \(job.id): \(error)", severity: .error)
            }
            return
        }
        guard isSweeping, !sweepCancelRequested else { return }
        sweepCancelRequested = true
        note("cancelling optimization — stops after the current generation", severity: .warning)
        appendDisplayLog(
            "cancellation requested — the sweep stops after the current "
                + "generation; partial rows stay in the run directory")
    }

    // MARK: Local-operation cancellation (App gap A1)

    /// A local study-run cancellation was requested: `ExperimentTasks.run`
    /// polls this between generations/choice items/battery items and stops
    /// after the current one — partial artifacts stay, marked by a
    /// cancelled.txt note, and no report.json is written. Reset when the
    /// next local run starts. (Server-routed runs are durable jobs with
    /// their own Cancel Server Job control.)
    public private(set) var studyRunCancelRequested = false
    /// Same flag for `ExperimentTasks.validate` (polled between concepts,
    /// scenarios, control extractions, and battery items; a cancelled
    /// validation writes NO evidence).
    public private(set) var validationCancelRequested = false
    /// Same flag for `ExperimentTasks.evaluatePairedJudge` (polled between
    /// judgments; completed judgments stay, no judge report is written).
    public private(set) var evaluationCancelRequested = false
    /// Same flag for `ExperimentTasks.extract` (A11; polled between
    /// concepts — completed concepts keep their sidecar artifacts, and the
    /// run directory is marked cancelled).
    public private(set) var extractCancelRequested = false

    /// Stop the in-flight LOCAL extraction after the current concept.
    public func cancelExtract() {
        guard isExtracting, !extractCancelRequested else { return }
        extractCancelRequested = true
        note("cancelling extraction — stops after the current concept; "
            + "completed vectors stay in the run directory", severity: .warning)
        appendDisplayLog(
            "cancellation requested — extraction stops after the current "
                + "concept; completed vectors stay")
    }

    /// Stop the in-flight LOCAL study run after the current generation.
    public func cancelStudyRun() {
        guard isRunning, !studyRunCancelRequested else { return }
        studyRunCancelRequested = true
        note("cancelling study run — stops after the current generation; "
            + "partial artifacts stay in the run directory", severity: .warning)
        appendDisplayLog(
            "cancellation requested — the run stops after the current "
                + "generation; partial artifacts stay (no report.json)")
    }

    /// Stop the in-flight LOCAL validation after the current unit of work.
    public func cancelValidation() {
        guard isValidating, !validationCancelRequested else { return }
        validationCancelRequested = true
        note("cancelling validation — stops after the current unit; "
            + "no validation evidence will be written", severity: .warning)
        appendDisplayLog(
            "cancellation requested — validation stops after the current "
                + "unit; no evidence is written")
    }

    /// Stop the in-flight LOCAL paired-judge evaluation after the current
    /// judgment.
    public func cancelPairedJudge() {
        guard isEvaluating, !evaluationCancelRequested else { return }
        evaluationCancelRequested = true
        note("cancelling paired judge — stops after the current judgment; "
            + "completed judgments stay, no judge report is written", severity: .warning)
        appendDisplayLog(
            "cancellation requested — judging stops after the current "
                + "judgment; no judge report is written")
    }

    /// Cancel the in-flight server experiment job (the durable job keeps its
    /// cancelled record server-side; the follow loop unwinds on the terminal
    /// state).
    public func cancelActiveServerJob() async {
        guard let job = activeServerJob else { return }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note("invalid server URL", severity: .error)
            return
        }
        do {
            try await client.cancelJob(job.id)
            note("cancel requested for server \(job.verb) job \(job.id) ('\(job.study)')", severity: .warning)
        } catch {
            note("cancel failed for job \(job.id): \(error)", severity: .error)
        }
    }

    /// The server-side run directory produced by the last completed
    /// run-on-active-server job (a path in the SERVER's tree, not local).
    public private(set) var lastServerRunDirectory: String?
    /// Read-only freeze gate summary for the selected draft (nil for frozen/
    /// completed studies); recomputed on every refresh.
    public private(set) var freezeReadiness: ExperimentStore.FreezeReadiness?
    /// The server's verbatim gate refusal from the last remote-freeze attempt
    /// (the FastAPI `detail` — the server's self-naming "cannot freeze …"
    /// message). Rendered by the view in the same idiom as an unmet local
    /// freeze gate; cleared on selection change and on the next attempt.
    public private(set) var remoteFreezeGateFailure: String?
    /// Non-blocking advisories the server returned WITH a successful remote
    /// freeze (the additive "advisories" response key) — rendered like local
    /// readiness advisories, cross-substrate ones prominently.
    public private(set) var remoteFreezeAdvisories: [String] = []
    /// The manifest-identity BLOCK from the last remote-freeze attempt: the
    /// server's same-named copy is not the document on screen (field-level
    /// mismatch summary + remedy), or the identity was unverifiable on an
    /// unpaired workspace. Rendered prominently; cleared on selection change
    /// and at the next attempt. See `FreezeRouting.remoteFreezePrecheck`.
    public private(set) var remoteFreezeIdentityWarning: String?
    /// Informational identity note when the freeze PROCEEDED anyway: the
    /// server-only-copy statement (name/status/canonical-body hash) or the
    /// paired-workspace could-not-verify note.
    public private(set) var remoteFreezeIdentityNote: String?
    /// The identity BLOCK's one-click remedy is available (2026-07-21
    /// incident, part 3): the mismatch case with a LOCAL DRAFT on screen —
    /// "Update the server's copy" pushes it as the server's draft and
    /// re-runs the identity check. Rule in
    /// `FreezeRouting.canOfferServerDraftSync`; frozen local manifests
    /// never push (duplicate-never-edit).
    public private(set) var remoteFreezeCanSyncDraft = false
    /// True while the push+recheck round-trip runs (disables the button).
    public private(set) var isSyncingServerDraft = false
    /// Decision-time coherence check for a SERVER-routed freeze on a
    /// workspace-PAIRED server: the shared runs/ tree evaluated from the
    /// server's perspective (`WorkspaceScoping.serverSubstrate`), so
    /// validate-locally-then-freeze-on-the-server warns BEFORE the click,
    /// not only in the server's own freeze-time advisory. nil when the
    /// workspace is unpaired (the local tree says nothing about the server's
    /// evidence), the study is not a draft, or the evidence is coherent.
    /// Recomputed on refresh, like `freezeReadiness`.
    public private(set) var serverFreezeCrossSubstrateAdvisory: String?

    // Display-pane live log (same affordance LoRA training and vector builds
    // use): panel-initiated runs mirror their CLI-style progress lines into
    // the chat transcript so long runs are observable outside this panel.
    private var displayLogID: UUID?
    private var displayLogTitle = ""
    private var displayLogLines: [String] = []

    /// The workspace-global connection (URL, token, Keychain persistence)
    /// lives on the shared `ClusterConnectionStore`; the panel reaches it
    /// through its host so run-on-server always targets the same server the
    /// rest of the app is connected to.
    public var cluster: ClusterConnectionStore? { host?.cluster }

    private var remoteClient: ClusterClient? { cluster?.client }

    private func loadStoredRemoteToken() {
        cluster?.loadStoredToken()
    }

    /// True when the active workspace is a connected server — Run/Validate
    /// Study then submit durable server jobs instead of running in-process.
    public var isServerWorkspace: Bool {
        if case .server = cluster?.activeWorkspace { return true }
        return false
    }

    /// Mac-authority mode (2026-07-21): server Compute whose pairing is
    /// KNOWN to be a different tree — `.unpaired` (same machine, different
    /// root) or `.remoteAuthoritative` (SSH cluster). The Mac workspace is
    /// then the source of truth: Freeze runs locally, Validate travels as a
    /// bundle job whose evidence imports back home. Paired and UNKNOWN
    /// pairing keep the server-resident routes — the mode requires a
    /// confirmed answer, never a guess.
    public var isKnownUnpairedServerWorkspace: Bool {
        guard isServerWorkspace else { return false }
        switch cluster?.activeServerPairing {
        case .unpaired, .remoteAuthoritative: return true
        case .paired, .unknown, nil: return false
        }
    }

    /// The substrate the LOCAL freeze gate's evidence matcher keys on
    /// (`ExperimentStore.freeze(runSubstrate:)`): the server engine exactly
    /// in Mac-authority mode (known-unpaired server Compute — the study
    /// runs there, the freeze happens here), this engine otherwise. Paired
    /// server workspaces are deliberately NOT included: their freeze routes
    /// to the server, whose own gates evaluate server-substrate evidence,
    /// and the local readiness view keeps its historical local perspective
    /// plus the explicit paired-workspace coherence advisory
    /// (`serverFreezeCrossSubstrateAdvisory`) — keying readiness on the
    /// server there would render that same advisory twice.
    public var freezeEvidenceRunSubstrate: String {
        isKnownUnpairedServerWorkspace
            ? WorkspaceScoping.serverSubstrate
            : ExperimentStore.evidenceSubstrate
    }

    // MARK: Server residency preflight (Run Server Copy enablement)

    /// Whether the selected study exists in the ACTIVE server's own
    /// experiments/ tree. nil = unknown (no server workspace, not yet
    /// checked, or the listing failed — the button stays enabled and the
    /// in-run refusal remains the backstop). Direct run verbs execute the
    /// SERVER-RESIDENT copy only; false drives the visible callout that
    /// points at Submit Bundle, the portable path.
    public private(set) var serverHasSelectedStudy: Bool?
    /// Cache key (study + workspace) so selection-driven refreshes don't
    /// hammer `GET /api/experiments` for a selection already checked.
    private var serverResidencyKey: String?

    /// Refresh `serverHasSelectedStudy` for the current selection, cached per
    /// (study, server workspace). The view calls this when the selection or
    /// the active workspace changes.
    public func refreshServerResidency() async {
        guard isServerWorkspace, let name = selectedName else {
            serverHasSelectedStudy = nil
            serverResidencyKey = nil
            return
        }
        let substrate = cluster?.substrateLabel ?? "server"
        let key = "\(substrate)::\(name)"
        if key == serverResidencyKey, serverHasSelectedStudy != nil { return }
        serverResidencyKey = key
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            serverHasSelectedStudy = nil
            return
        }
        guard let names = try? await client.experimentNames() else {
            // Listing failed: unknown, not "missing" — don't disable the
            // button on a transient error; the run-path refusal backstops.
            serverHasSelectedStudy = nil
            return
        }
        let resident = names.contains(name)
        noteServerResidency(resident)
    }

    /// Records a residency answer (from the refresh above or the run-path
    /// backstop). On the transition to "not on the server", preselect the
    /// bundle controls for what the user actually meant — run, for real —
    /// so Submit Bundle does what Run Server Copy could not.
    private func noteServerResidency(_ resident: Bool) {
        let wasMissing = serverHasSelectedStudy == false
        serverHasSelectedStudy = resident
        if !resident, !wasMissing {
            remoteVerb = "run"
            remoteDryRun = false
        }
    }

    // MARK: Pure label/status helpers (unit-tested)

    /// Dynamic Submit Bundle label: the button says what it will do.
    public static func bundleSubmitLabel(verb: String, dryRun: Bool) -> String {
        dryRun ? "Submit Bundle: \(verb) (dry run)" : "Submit Bundle: \(verb)"
    }

    public var submitBundleButtonLabel: String {
        Self.bundleSubmitLabel(verb: remoteVerb, dryRun: remoteDryRun)
    }

    /// Post-submission status: what was submitted and where to watch it.
    public static func bundleSubmittedStatus(
        study: String, verb: String, dryRun: Bool, substrate: String, jobID: String
    ) -> String {
        let mode = dryRun ? "\(verb) (dry run — prepared only, nothing executes)" : verb
        return "bundled study '\(study)' submitted: \(mode) on \(substrate) — "
            + "job \(jobID), following in the activity pane"
    }

    /// The not-on-server callout body (rendered prominently by the view).
    public static func residencyCalloutMessage(study: String, substrate: String) -> String {
        "Study '\(study)' exists locally, not on \(substrate). Direct runs "
            + "execute the server-resident copy only. Submit Bundle sends a "
            + "portable copy — or pair the server to this workspace "
            + "(serve --root <workspace>)."
    }

    // MARK: Display-pane live log

    private func beginDisplayLog(title: String, initialLine: String) {
        guard let host else { return }
        displayLogTitle = title
        displayLogLines = [initialLine]
        displayLogID = host.startLiveLog(title: title, initialLine: initialLine)
    }

    private func appendDisplayLog(_ line: String) {
        guard let host, let id = displayLogID else { return }
        displayLogLines.append(line)
        if displayLogLines.count > 400 {
            displayLogLines.removeFirst(displayLogLines.count - 400)
        }
        host.updateLiveLog(id: id, title: displayLogTitle, lines: displayLogLines)
    }

    private func endDisplayLog(_ finalLine: String? = nil) {
        if let finalLine { appendDisplayLog(finalLine) }
        displayLogID = nil
        displayLogLines = []
    }

    public var judgeModelOptions: [String] {
        var seen = Set<String>()
        var options: [String] = []
        func append(_ model: String?) {
            guard let model else { return }
            let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !seen.contains(trimmed) else { return }
            seen.insert(trimmed)
            options.append(trimmed)
        }

        append(judgeModel)
        append(selected?.modelID)
        append(studyBaseModelID)
        append(host?.selectedModelID)
        for model in SteeredContainerLoader.localModelIDs() {
            append(model)
        }
        if options.isEmpty {
            for model in ChatService.availableModels.map(\.id) {
                append(model)
            }
        }
        append(ClaudePairedJudge.defaultModel)
        return options
    }

    /// Rubric files available for pinning (prompts/rubrics/*), refreshed on
    /// access so a freshly added rubric appears without an app restart.
    public var rubricFileOptions: [String] {
        JudgeRubricStore.list()
    }

    /// A new judge defaults to `openrouter` (2026-07-24). External judging
    /// standardises there: OpenRouter reaches Anthropic models via provider
    /// `anthropic`, and unlike the direct Claude path it reports which
    /// backend actually served the verdict, so the pin is verifiable. The
    /// `claude` kind still WORKS — it is simply no longer the default, and
    /// the app no longer offers it for new judges.
    public func addJudge() {
        let index = judges.count + 1
        judges.append(
            .init(
                name: "judge-\(index)",
                kind: index == 1 ? "openrouter" : "local",
                model: nil))
    }

    public func removeJudge(at index: Int) {
        guard judges.indices.contains(index) else { return }
        judges.remove(at: index)
        // The kind stash is keyed by row index — shift the entries above
        // the removed row down so each remaining row keeps ITS stash.
        var shifted: [Int: [String: JudgeKindStash]] = [:]
        for (row, stash) in judgeKindStashes where row != index {
            shifted[row > index ? row - 1 : row] = stash
        }
        judgeKindStashes = shifted
    }

    /// One kind's field set for one judge row, held while the row wears a
    /// different kind. See `judgeKindStashes`.
    public struct JudgeKindStash: Sendable, Equatable {
        public var model: String?
        public var provider: String?
        public var revision: String?
        public var dtype: String?
    }

    /// Session-only stash of judge fields per row and per kind (field bug
    /// 2026-08-07): switching a judge's kind swaps the row to that kind's
    /// own field set, and the outgoing kind's values land here so toggling
    /// back restores them — a hand-discovered OpenRouter provider slug must
    /// survive an exploratory toggle to local. Never serialized: the
    /// manifest write keeps only kind-owned fields
    /// (`JudgeRef.keepingKindOwnedFields`), and the stash belongs to the
    /// study it was made on (cleared on selection sync, like seat edits).
    public private(set) var judgeKindStashes: [Int: [String: JudgeKindStash]] = [:]

    /// The kind picker's write path (field bug 2026-08-07). Stashes the
    /// outgoing kind's fields and restores any previously entered for the
    /// incoming kind this session, so the row's live fields are always the
    /// CURRENT kind's own — a kind never renders (or saves) another kind's
    /// values, and nothing the researcher typed is lost to a toggle.
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

    public init() {
        refresh()
    }

    public var selected: ExperimentManifest? {
        experiments.first { $0.name == selectedName }
    }

    /// Pin a file the Data Readiness checklist just scaffolded, and persist it.
    ///
    /// Needed because some requirements are satisfied by the FILE existing
    /// while others are satisfied by the PIN. The panel script is the second
    /// kind — its requirement asks "is a panel pinned?" — so "Create from
    /// template" used to leave the blocker standing with the freshly created
    /// file sitting right beside it, which reads as the checklist being
    /// broken. Returns true when something was pinned and saved.
    @discardableResult
    public func pinScaffoldedFile(
        requirement: DataRequirement, createdPath: String
    ) -> Bool {
        guard var manifest = selected, manifest.status == .draft else { return false }
        // Paths are stored workspace-relative; the checklist hands back an
        // absolute one.
        let root = VectorCatalog.projectRoot.standardizedFileURL.path
        var relative = URL(filePath: createdPath).standardizedFileURL.path
        if relative.hasPrefix(root + "/") {
            relative = String(relative.dropFirst(root.count + 1))
        }
        do {
            guard try StudyDataReadiness.pinScaffolded(
                requirement: requirement, createdPath: relative,
                into: &manifest, workspaceRoot: VectorCatalog.projectRoot)
            else { return false }
            try ExperimentStore.save(manifest)
            refresh()
            note("pinned \(relative)", severity: .info)
            return true
        } catch {
            note("could not pin \(relative): \(error)", severity: .warning)
            return false
        }
    }

    public var multiAgentScenarioOptions: [MultiAgentScenarioRecord] {
        MultiAgentScenarioStore.scan()
    }

    // MARK: Seats — who sits where in the study's scenario

    /// Unsaved per-seat edits, keyed by the scenario's seat id.
    ///
    /// An OVERLAY, not the casting: what a study is cast as lives in the
    /// scenario file it pins, and this holds only what the researcher has
    /// changed since it was read (`SeatCasting.state`). Cleared when the
    /// selection changes and when a casting is saved — a seat edit belongs to
    /// the study it was made on.
    public var seatCastingEdits: [String: SeatOccupant] = [:]

    /// The Seats section's model: the seats of the scenario this study is
    /// running (or about to), and who occupies each one.
    ///
    /// Nil for anything that is not a multi-agent study, and for a multi-agent
    /// study with no scenario chosen yet — there is nothing to cast until a
    /// scenario names some seats.
    public var seatCasting: SeatCasting.State? {
        guard var manifest = selected else { return nil }
        // The editor's type wins over the stored one for READING, so the
        // section appears the moment the type picker says multi-agent rather
        // than one save later.
        manifest.studyKind = studyKind
        let record = selectedMultiAgentScenarioID.flatMap { id in
            multiAgentScenarioOptions.first { $0.id == id }
        }
        return SeatCasting.state(
            of: manifest,
            selected: record.map { ($0.scenario, relativeProjectPath(for: $0.url)) },
            overlay: seatCastingEdits)
    }

    /// Agents a seat may be cast with: the library filtered to the study's
    /// SAVED base model — the same eligibility rule the instantiation table
    /// applies, and the same one `attachAgent` enforces for comparison arms.
    ///
    /// The saved model rather than the editor's field on purpose: a casting is
    /// compiled against the manifest, so offering agents for a base model that
    /// has not been saved yet would offer a cast that the compile then binds to
    /// the previous model.
    public var availableAgentsForSeats: [ModelVariantRecord] {
        guard let model = selected?.modelID, !model.isEmpty else { return [] }
        return ModelVariantStore.scan().filter { $0.artifact.baseModelID == model }
    }

    /// The occupant a library agent contributes to a seat, pinned by path and
    /// artifact hash exactly as a variant condition is. (`SeatCasting.occupant`
    /// is THE derivation — shared with `TemplateInstantiation.occupant(for:)`
    /// and with `panel compile`, so a seat cast here, a seat cast in the
    /// instantiation table, and a seat cast headlessly are the same value.)
    public func seatOccupant(forAgentID id: String?) -> SeatOccupant {
        guard let id, let record = availableAgentsForSeats.first(where: { $0.id == id })
        else { return .baseline }
        return SeatCasting.occupant(for: record)
    }

    /// The library id currently occupying a seat, or nil for baseline — what a
    /// seat picker binds to.
    public func seatAgentID(for seat: String) -> String? {
        guard case .agent(_, let path, _) = seatCasting?.occupants[seat] ?? .baseline
        else { return nil }
        return availableAgentsForSeats.first {
            ModelVariantStore.relativePath(for: $0) == path
        }?.id
    }

    public func setSeatAgent(_ agentID: String?, seat: String) {
        seatCastingEdits[seat] = seatOccupant(forAgentID: agentID)
    }

    /// Why the seat casting cannot be saved right now, or nil.
    public func seatCastingRefusal(_ state: SeatCasting.State) -> String? {
        guard let manifest = selected else { return "no study selected" }
        guard manifest.status == .draft else {
            return "'\(manifest.name)' is \(manifest.status.rawValue) — its "
                + "scenario is part of the record. Duplicate it as a draft to "
                + "cast a different panel."
        }
        guard manifest.studyKind == .multiAgent else {
            return "save the study setup first — the casting is compiled "
                + "against this study's model and sampling settings"
        }
        guard state.isEditable else { return SeatCasting.legacyAdvisory }
        guard !manifest.modelID.isEmpty else {
            return "pick this study's base model first — a compiled scenario "
                + "binds every seat to it"
        }
        return nil
    }

    /// Compile this study's seat casting and pin the result as its scenario.
    ///
    /// The write is `SeatCasting.compile` — the same call the design
    /// instantiation path makes — so the study ends up pinning an ordinary
    /// bound scenario that the run loop, the freeze packager and the Python
    /// engine already understand.
    public func saveSeatCasting() {
        guard var manifest = selected, let state = seatCasting else { return }
        if let refusal = seatCastingRefusal(state) {
            note("Couldn't save the seats — " + refusal, severity: .error)
            return
        }
        do {
            let compiled = try SeatCasting.compile(
                state.assignment, semantic: state.semantic,
                semanticPath: state.semanticPath, into: &manifest)
            try ExperimentStore.save(manifest)
            seatCastingEdits = [:]
            refresh()
            let cast = state.assignment.ordered.filter { $0 != .baseline }.count
            note(
                "cast \(state.seats.count) seat(s) (\(cast) steered) and pinned "
                    + "\(compiled.path) — this study now runs that compiled "
                    + "scenario",
                severity: .success)
        } catch {
            note(
                "Couldn't save the seats — nothing was pinned. "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"),
                severity: .error)
        }
    }

    /// "Create permuted siblings…": every distinct re-seating of THIS study's
    /// casting, as sibling studies.
    ///
    /// Sibling studies rather than conditions because a manifest carries
    /// exactly one scenario on both engines and its panel arms are the fixed
    /// baseline/configured pair — N castings cannot be N conditions of one
    /// study without changing the run loop. So the study becomes its own design
    /// (`templateFromStudy`, which silently returns the existing one when this
    /// study is an unchanged instance of it) and the ordinary instantiation
    /// table opens on it, preloaded with the permutation rows. No new minting
    /// machinery: the table already costs the batch, isolates row failures and
    /// offers mint-only or mint-and-submit.
    public func startPermutedSiblings() {
        clearFormError(.template)
        guard let manifest = selected, let state = seatCasting else { return }
        guard state.form == .cast else {
            refuse(
                .template,
                state.form == .legacyBound
                    ? "Couldn't permute this panel — " + SeatCasting.legacyAdvisory
                    : "Save the seats first — permuted siblings re-seat the "
                        + "casting this study is running, and it has none yet")
            return
        }
        do {
            let mint = try StudyTemplateStore.templateFromStudy(
                experimentName: manifest.name)
            refreshTemplates()
            for warning in mint.warnings { note(warning, severity: .warning) }
            templateInstantiationInvitation = TemplateInstantiationInvitation(
                design: mint.template.name,
                permuting: state.assignment.ordered)
            note(
                "permuting this study's casting against design "
                    + "'\(mint.template.name)'\(mint.minted ? " (just saved)" : "") "
                    + "— each row below mints one sibling study",
                severity: .info)
        } catch {
            refuse(
                .template,
                "Couldn't open the permutation table — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    /// Baseline-model options for the study builder — the ACTIVE workspace's
    /// inventory (one rule: `WorkspaceScoping.studyBaselineModelOptions`).
    /// Server target: the SERVER's installed models (the same
    /// `workspaceModelOptions` source the chat's WorkspaceModelPicker reads)
    /// — never the local MLX tiers, whose repo ids the server cannot load.
    /// Local target: the local tiers plus the current/selected models and the
    /// bases of saved definitions (unchanged behavior). A current
    /// `studyBaseModelID` outside the returned inventory is the view's
    /// "(not installed)" row, not an extra pickable option here.
    public var modelOptions: [String] {
        WorkspaceScoping.studyBaselineModelOptions(
            workspaceIsServer: isServerWorkspace,
            localOptions: localModelOptions,
            serverOptions: host?.workspaceModelOptions ?? [])
    }

    private var localModelOptions: [String] {
        var seen = Set<String>()
        var options: [String] = []
        func append(_ value: String?) {
            guard let value else { return }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !seen.contains(trimmed) else { return }
            seen.insert(trimmed)
            options.append(trimmed)
        }
        append(studyBaseModelID)
        append(selected?.modelID)
        append(host?.selectedModelID)
        for model in ChatService.availableModels.map(\.id) { append(model) }
        for variant in ModelVariantStore.scan() { append(variant.artifact.baseModelID) }
        return options
    }

    public var availableVariantsForStudy: [ModelVariantRecord] {
        let attached = Set(selected?.variantConditions.map(\.artifactPath) ?? [])
        return ModelVariantStore.scan().filter {
            $0.artifact.baseModelID == studyBaseModelID
                && !attached.contains(ModelVariantStore.relativePath(for: $0))
        }
    }

    /// Concepts on disk not yet attached to the selected experiment.
    public var attachableConcepts: [String] {
        let attached = Set(selected?.concepts.map(\.name) ?? [])
        return VectorCatalog.conceptNames().filter { !attached.contains($0) }
    }

    /// Sweep-recommended conditions carrying selection provenance — the
    /// promotable cells (screening's outputs, confirmation's inputs).
    public var promotableRecommendations: [ExperimentManifest.Condition] {
        selected?.conditions.filter { $0.selection != nil } ?? []
    }

    /// Agents eligible for a confirmation study on the selected draft:
    /// vector-only, single-injection, matching the study base model —
    /// exactly what `ConfirmationStudy.attach` will accept. Sweep-promoted
    /// agents sort first (they are what confirmation is FOR; hand-created
    /// ones stay legal and get the freeze advisory).
    public var confirmableAgents: [ModelVariantRecord] {
        ModelVariantStore.scan()
            .filter {
                $0.artifact.baseModelID == studyBaseModelID
                    && $0.artifact.adapters.isEmpty
                    && $0.artifact.injections.count == 1
            }
            .sorted {
                ($0.artifact.promotion != nil ? 0 : 1, $0.artifact.name)
                    < ($1.artifact.promotion != nil ? 0 : 1, $1.artifact.name)
            }
    }

    /// Conditions the attached perturbation policy generated (anchor, ±δ,
    /// control) — shown so the condition machinery stays visible, never
    /// hidden behind the agent vocabulary.
    public var generatedConfirmationConditions: [ExperimentManifest.Condition] {
        guard let manifest = selected,
              let agent = manifest.perturbationPolicy?.sourceAgent.name
        else { return [] }
        return manifest.conditions.filter {
            ConfirmationStudy.isGeneratedName($0.name, agent: agent)
        }
    }

    /// Attach the declared perturbation policy to the selected draft — same
    /// code path as `steerlab-cli experiment confirm`.
    public func attachPerturbations() {
        guard let name = selectedName else { return }
        guard let record = confirmableAgents.first(where: { $0.id == confirmAgentID })
        else {
            note("select an agent to confirm", severity: .info)
            return
        }
        let deltas = confirmDeltasText
            .split(separator: ",")
            .compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard !deltas.isEmpty else {
            note("alpha deltas must be numbers, e.g. 0.2, 0.5", severity: .error)
            return
        }
        do {
            let manifest = try ConfirmationStudy.attach(
                experimentName: name,
                agent: record.url.path,
                deltas: deltas,
                includeControl: confirmIncludeControl,
                log: { _ in })
            let generated = manifest.conditions.filter {
                ConfirmationStudy.isGeneratedName(
                    $0.name, agent: record.artifact.name)
            }
            note("attached perturbation policy for '\(record.artifact.name)' "
                + "— \(generated.count) conditions (anchor, ±δ"
                + (confirmIncludeControl ? ", matched-norm control)" : ")"), severity: .success)
            refresh()
        } catch {
            note("confirm failed: \(error)", severity: .error)
        }
    }

    /// Mint an agent (variant artifact) from the concept's sweep-selected
    /// cell — same code path as `steerlab-cli experiment promote`.
    public func promoteRecommended(concept: String) {
        guard let name = selectedName else { return }
        promote(experimentName: name, concept: concept)
    }

    /// Where a promotion mints its agent. nil (the legacy default) follows
    /// the active workspace; the Optimizations surface passes it explicitly because
    /// an UNPAIRED server workspace still lists LOCAL optimizations, whose sweep
    /// provenance lives in local manifests — promoting one of those must
    /// stay a local mint even while the compute target is a server.
    public enum PromotionRoute: Sendable {
        case local
        case activeServer
    }

    /// Promotion for ANY experiment (the Optimizations surface keeps its own
    /// selection without disturbing the Studies panel's). Passing `cell`
    /// bypasses the declared selection and REQUIRES the override path:
    /// `overrideReason` is stamped into the birth certificate alongside
    /// promotedBy=manualOverride — the deviation documents itself.
    ///
    /// Substrate-aware: in a server workspace this routes through the server's
    /// own promote route (`promoteOnActiveServer`) — artifacts never cross
    /// substrates, so the agent is minted where the sweep provenance lives.
    /// `route` overrides that default (see `PromotionRoute`).
    /// The pinned contract for a promotion, built from the sweep run the
    /// caller is LOOKING AT.
    ///
    /// The run must be passed in, not discovered here. An earlier version
    /// scanned the local `runs/` tree — which is wrong in a server workspace,
    /// where the Optimizations view has separately downloaded the SERVER's
    /// run: it either found nothing (pins nil, and the server fell back to
    /// the ambient newest evidence this contract exists to remove) or found a
    /// same-named LOCAL run and sent its name and the local manifest hash to
    /// a server where neither may mean anything.
    ///
    /// The epoch hash comes from the same source as the run: a local run is
    /// checked against the local manifest, and a server run carries the
    /// server's own hash from its selection provenance.
    public static func promotionPins(
        experimentName: String,
        concept: String,
        sweepRun: SweepRunCatalog.SweepRun,
        localManifestHash: String?
    ) -> AgentPromotion.Pins? {
        guard let recommendation = sweepRun.recommendations[concept] else {
            return nil
        }
        guard case .selected(let provenance) = recommendation else {
            // A failure entry pins the RUN but has no cell to agree with;
            // promoting past it needs a loud override, which stays the
            // caller's decision.
            return .init(
                sweepRun: sweepRun.runName, experimentHash: localManifestHash)
        }
        return .init(
            sweepRun: sweepRun.runName,
            experimentHash: localManifestHash,
            winningCell: (provenance.winningCell.layer, provenance.winningCell.alpha))
    }

    /// The local manifest's epoch hash, when this workspace owns the study.
    /// Nil in a server workspace, where the server checks its own.
    public func localManifestHash(_ experimentName: String) -> String? {
        guard !isServerWorkspace else { return nil }
        return experiments.first { $0.name == experimentName }
            .map(ExperimentStore.manifestHash)
    }

    /// - Parameter pins: the contract built from the sweep run the CALLER is
    ///   displaying (`ExperimentPanel.promotionPins`). Passing nil lets the
    ///   engine resolve ambiently, which is legal for the CLI's unpinned
    ///   verb but must never happen from evidence-bearing UI — the surfaces
    ///   there refuse rather than promote unpinned.
    public func promote(
        experimentName: String,
        concept: String,
        cell: (layer: Int, alpha: Double)? = nil,
        overrideReason: String? = nil,
        route: PromotionRoute? = nil,
        pins: AgentPromotion.Pins? = nil
    ) {
        let onServer = route.map { $0 == .activeServer } ?? isServerWorkspace
        if onServer {
            Task {
                await promoteOnActiveServer(
                    experimentName: experimentName, concept: concept,
                    cell: cell, overrideReason: overrideReason, pins: pins)
            }
            return
        }
        do {
            let record = try AgentPromotion.promote(
                experimentName: experimentName, concept: concept,
                cell: cell, overrideReason: overrideReason, pins: pins)
            let suffix = cell == nil
                ? "— it carries the sweep-selection birth certificate and is "
                    + "ready in the Agents library"
                : "— stamped promotedBy: manualOverride (declared selection "
                    + "bypassed)"
            note("promoted '\(concept)' → agent '\(record.artifact.name)' "
                + suffix, severity: .success)
            refresh()
        } catch {
            note("promote failed: \(error)", severity: .error)
        }
    }

    // MARK: Optimizations — declare a sweep spec, run the sweep

    /// Declare (or update) the sweep spec on a DRAFT manifest — the Optimizations
    /// "Declare an Optimization" edge. The spec is hashed manifest data that freeze
    /// pins, so non-draft manifests refuse here (duplicate to iterate), and
    /// the declared selection is validated at SAVE time so a bad criterion is
    /// caught at declaration, not at sweep start. Returns false (with the
    /// reason in `status`) when nothing was written.
    @discardableResult
    public func setSweepSpec(
        _ spec: ExperimentManifest.SweepSpec, for name: String
    ) -> Bool {
        // Normalize BEFORE validating, so what is checked is what is
        // written: an absolute instrument path inside the workspace becomes
        // the portable workspace-relative form (both declare flows — the
        // composer and the Optimizations editor — funnel through here).
        let spec = SweepSpecForm.workspaceRelativeNormalized(spec)
        do {
            var manifest = try ExperimentStore.load(name: name)
            guard manifest.status == .draft else {
                refuse(
                    .sweepSpec,
                    "'\(name)' is \(manifest.status.rawValue) — the sweep "
                        + "spec is pinned manifest data; duplicate the study to "
                        + "change it", severity: .warning)
                return false
            }
            if let problem = SweepSpecForm.validate(spec) {
                refuse(.sweepSpec, "sweep spec not saved: \(problem)")
                return false
            }
            let outcome = SweepSpecForm.validateSelection(spec.selection)
            if case .invalid(let reason) = outcome {
                refuse(.sweepSpec, "sweep spec not saved: \(reason)")
                return false
            }
            if let problem = SweepSpecForm.validateObjectiveRequirements(
                spec.selection, manifest: manifest)
            {
                refuse(.sweepSpec, "sweep spec not saved: \(problem)")
                return false
            }
            manifest.sweep = spec
            try ExperimentStore.save(manifest)
            refresh()
            clearFormError(.sweepSpec)
            if case .declaredAhead(let metric) = outcome {
                note("declared sweep spec on '\(name)' — objective "
                    + "'\(metric)' is not implemented on this engine: the "
                    + "sweep will REFUSE at start (declaring ahead is allowed)", severity: .warning)
            } else {
                note("declared sweep spec on '\(name)' "
                    + "(\(spec.layerFractions.count) layer fraction"
                    + "\(spec.layerFractions.count == 1 ? "" : "s") × "
                    + "\(spec.alphas.count) alpha"
                    + "\(spec.alphas.count == 1 ? "" : "s"))", severity: .success)
                // judgeScore local-judge resolution, surfaced at save time:
                // blank-model judges are legal (study model); a non-study
                // local judge model warns about the local sweep-start
                // refusal.
                if spec.selection?.objective?.metric == "judgeScore" {
                    let judges = manifest.judges ?? []
                    if let warning = SweepSpecForm.localJudgeSlotWarning(
                        judges: judges, studyModelID: manifest.modelID)
                    {
                        note((status ?? "") + " — ⚠︎ \(warning)", severity: .warning)
                    } else if let defaultNote = SweepSpecForm.localJudgeDefaultNote(
                        judges: judges, studyModelID: manifest.modelID)
                    {
                        note((status ?? "") + " — \(defaultNote)", severity: .info)
                    }
                }
            }
            return true
        } catch {
            refuse(.sweepSpec, "declare optimization failed: \(error)")
            return false
        }
    }

    /// Run the layer×alpha sweep for an experiment — same task path as
    /// `steerlab-cli experiment sweep`. Local workspace: in-process through
    /// `ExperimentTasks.sweep` (which loads the pinned model itself — no
    /// preloaded chat model needed), progress mirrored into the display pane.
    /// Server workspace: submitted as a durable server job for the
    /// SERVER-RESIDENT copy, followed in the shared display.
    public func runSweep(experimentName name: String) async {
        guard !isSweeping, !isRunning, !isValidating else { return }
        // Busy-chat preflight (both routes): a sweep contends for the same
        // model slot as a live chat generation — locally the load itself,
        // on the server the loaded-slot registry (observed live as
        // "ModelLoadError: all loaded model slots are busy"). Refuse up
        // front with the fix; the server's own error surfacing stays for
        // in-flight streams this client cannot see.
        if host?.isGenerating == true {
            let refusal = "a chat generation is in flight — the model slot "
                + "is busy; stop or finish the Playground chat, then optimize"
            note(refusal, severity: .error)
            appendDisplayLog(refusal)
            return
        }
        if isServerWorkspace {
            await runExperimentVerbOnActiveServer(experimentName: name, verb: "sweep")
            return
        }
        isSweeping = true
        sweepCancelRequested = false
        note("sweeping '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Optimization sweep — \(name)",
            initialLine: "verifying pins — the sweep loads the pinned model itself…")
        defer {
            isSweeping = false
            refresh()
        }
        do {
            try await ExperimentTasks.sweep(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.sweepCancelRequested ?? false
                },
                log: { [weak self] line in
                    await MainActor.run {
                        self?.appendDisplayLog(line)
                        self?.status = line
                    }
                })
            refresh()
            if sweepCancelRequested {
                // The sweep returned normally with a partial grid (server
                // parity) — never report a cancelled run as complete.
                note("sweep cancelled for '\(name)' — partial grid rows "
                    + "kept in the run directory; no recommendation from an "
                    + "incomplete grid", severity: .warning)
                endDisplayLog("sweep cancelled for '\(name)' — partial rows kept")
            } else {
                note("sweep complete for '\(name)' — grid and recommendations updated", severity: .success)
                endDisplayLog("sweep complete for '\(name)'")
            }
        } catch {
            refresh()
            note("sweep failed: \(error)", severity: .error)
            endDisplayLog("sweep failed: \(error)")
        }
    }

    public func refresh() {
        experiments = ExperimentStore.list()
        displayLabels = ExperimentStore.displayLabels(experiments)
        refreshTemplates()
        if let selectedName, !experiments.contains(where: { $0.name == selectedName }) {
            self.selectedName = nil
        }
        violations = selected.map(ExperimentStore.verify) ?? []
        if let manifest = selected, manifest.status == .draft {
            // Readiness (and its cross-substrate advisory) keys the
            // validate-evidence match on the substrate the study will RUN
            // on — under server Compute an imported server validate run
            // counts and local-engine evidence does not (Mac-authority
            // mode, 2026-07-21); a local workspace keeps this engine.
            freezeReadiness = ExperimentStore.freezeReadiness(
                for: manifest, violations: violations,
                runSubstrate: freezeEvidenceRunSubstrate)
            serverFreezeCrossSubstrateAdvisory = computeServerFreezePerspectiveAdvisory(
                for: manifest)
        } else {
            freezeReadiness = nil
            serverFreezeCrossSubstrateAdvisory = nil
        }
        syncDraftFieldsFromSelection()
        refreshResults()
    }

    /// One-click study creation: mint a readable, unique placeholder name and
    /// invite the rename immediately. Naming a study before it exists is a
    /// decision the researcher cannot yet make — the draft can be renamed for
    /// as long as it stays a draft, so the name is not a gate on getting in.
    public func newStudy() {
        let placeholder = ExperimentStore.placeholderStudyName()
        newName = placeholder
        create()
        // Only invite the rename when the draft actually landed.
        if selectedName == placeholder { renameInvitation = placeholder }
    }

    public func create() {
        guard let host else { return }
        do {
            // Workspace-scoped fallback: a server-target draft must never be
            // silently pinned to a local MLX id the server can't load.
            // Optional up-front revision pin (App gap A7): the store always
            // accepted it; empty keeps the auto-pin-from-HF-cache behavior.
            let revision = newRevision.trimmingCharacters(in: .whitespacesAndNewlines)
            // The workspace fallback used to apply only when
            // studyBaseModelID was EMPTY. But loading any study sets it, so
            // after the first selection it was never empty and a new draft
            // silently inherited the previous study's model — including one
            // this workspace cannot offer, which then rendered as a
            // "(not installed)" row nobody could have chosen.
            let workspaceDefault = host.workspaceSelectedModelID ?? host.selectedModelID
            let inventory = modelOptions
            let carried = studyBaseModelID.trimmingCharacters(in: .whitespacesAndNewlines)
            let seed =
                (!carried.isEmpty && (inventory.isEmpty || inventory.contains(carried)))
                ? carried
                : workspaceDefault
            if !carried.isEmpty, seed != carried {
                note(
                    "new draft pinned to '\(seed)': the carried-over model "
                        + "'\(carried)' is not available in this workspace",
                    severity: .info)
            }
            var manifest = try ExperimentStore.create(
                name: newName,
                description: newDescription,
                modelID: seed,
                modelRevision: revision.isEmpty ? nil : revision)
            manifest.studyKind = studyKind
            manifest.temperature = 0
            manifest.maxTokens = 2048
            manifest.promptMode = .chatAssistant
            manifest.qwenThinkingEnabled = false
            try ExperimentStore.save(manifest)
            newName = ""
            newDescription = ""
            newRevision = ""
            refresh()
            selectedName = manifest.name
            note("created draft protocol '\(manifest.name)' (model \(manifest.modelID)"
                + (manifest.modelRevision.map { ", revision \($0.prefix(12))…)" } ?? ")"), severity: .success)
        } catch {
            note(
                "Couldn't create the draft — check the name isn't already in "
                    + "use and the workspace is writable. Details: \(error)",
                severity: .error)
        }
    }

    /// Rename the selected study. Two independent effects, both optional,
    /// applied in one action so the researcher sees one "Rename":
    ///
    /// - `canonicalName` (drafts only) moves `experiments/<old>/` to
    ///   `experiments/<new>/` and rewrites the manifest name. Runs already
    ///   recorded under the old name are immutable and are NOT touched — the
    ///   outcome's note says so, because they stop listing under this study.
    /// - `label` writes the hash-exempt display-label sidecar, allowed at
    ///   every status: it is not manifest content, so it cannot move the
    ///   content hash or re-epoch a frozen study's runs.
    ///
    /// The canonical rename runs FIRST so the label lands in the study's
    /// final directory.
    public func renameSelected(canonicalName: String?, label: String?) {
        guard let name = selectedName else { return }
        clearFormError(.rename)
        var current = name
        var messages: [String] = []
        if let canonicalName,
            ExperimentStore.resolvedRenameTarget(canonicalName) != name
        {
            do {
                let outcome = try ExperimentStore.rename(
                    experimentName: name, to: canonicalName)
                current = outcome.newName
                messages.append("renamed '\(outcome.oldName)' → '\(outcome.newName)'")
                if let runsNote = outcome.runsNote { messages.append(runsNote) }
            } catch {
                refuse(.rename, "Couldn't rename the study — nothing changed. \(error)")
                return
            }
        }
        if let label {
            do {
                try ExperimentStore.setDisplayLabel(label, experimentName: current)
                let normalized = ExperimentStore.normalizedDisplayLabel(label)
                messages.append(
                    normalized.isEmpty
                        ? "cleared the display label"
                        : "display label set to \"\(normalized)\"")
            } catch {
                refuse(
                    .rename,
                    "Couldn't save the display label — the study folder must be "
                        + "writable. Details: \(error)")
                return
            }
        }
        guard !messages.isEmpty else { return }
        refresh()
        selectedName = current
        note(messages.joined(separator: " — "), severity: .success)
    }

    // MARK: Study templates (the invariant half of a replication)

    /// The workspace's design library, refreshed with the study list.
    /// ("Template" is the artifact's stored name; the interface calls one a
    /// DESIGN, which is what it is.)
    public private(set) var templates: [StudyTemplate] = []
    public var selectedTemplateName: String?
    /// The template a just-completed "Load as Template" wants opened — the
    /// view consumes and clears it, exactly as `renameInvitation` works.
    /// Also the Templates tab's cross-section handoff: Instantiate sets it and
    /// navigates to Studies, which opens the flow on it (consumed on appear as
    /// well as on change — the Studies view is not on screen when it is set).
    /// Also carries the occupant pool "Create permuted siblings…" hands over,
    /// so the table opens already holding one row per distinct re-seating.
    public var templateInstantiationInvitation: TemplateInstantiationInvitation?
    /// The Templates tab's "New from Study" source. A study NAME, chosen from
    /// the whole list at any status.
    public var templateSourceStudyName: String?
    /// What the next new study starts from (the Studies tab's first control).
    public var newStudyDesign: StudyDesignChoice = .fromScratch

    public var selectedTemplate: StudyTemplate? {
        templates.first { $0.name == selectedTemplateName }
    }

    /// Live design lineage per minted study, keyed by study name — both facts
    /// (`agreement` + `designRevised`, see `StudyTemplateStore.DesignLineage`).
    ///
    /// Computed once per `refresh()` and never per frame: the check re-reads
    /// each minted study's compiled panel from disk, which a SwiftUI body must
    /// not do. Studies with no lineage are absent from the map.
    public private(set) var designLineage:
        [String: StudyTemplateStore.DesignLineage] = [:]

    public func refreshTemplates() {
        templates = StudyTemplateStore.list()
        if let selectedTemplateName,
            !templates.contains(where: { $0.name == selectedTemplateName })
        {
            self.selectedTemplateName = nil
        }
        newStudyDesign = StudyDesignChoice.resolve(newStudyDesign, designs: templates)
        if let templateSourceStudyName,
            !experiments.contains(where: { $0.name == templateSourceStudyName })
        {
            self.templateSourceStudyName = nil
        }
        designLineage = [:]
        for manifest in experiments where manifest.templateProvenance != nil {
            designLineage[manifest.name] = StudyTemplateStore.lineage(of: manifest)
        }
    }

    /// The design-summary rows for a template — the Templates tab's read-only
    /// body. On the panel so the view calls one function and formats nothing.
    public func designSummary(_ template: StudyTemplate) -> [StudyDesignSummary.Row] {
        StudyDesignSummary.rows(for: template)
    }

    /// "New from Study" in the Templates tab: mint a design from ANY study.
    ///
    /// Deliberately quieter than `loadSelectedStudyAsTemplate`. In the library
    /// the researcher is looking at the list, so the DEDUP case needs no
    /// sentence — selecting the design that already existed shows the result.
    /// A fresh mint and a divergence both say something, because both changed
    /// the library.
    public func newDesignFromStudy(named name: String) {
        clearFormError(.template)
        do {
            let mint = try StudyTemplateStore.templateFromStudy(experimentName: name)
            refreshTemplates()
            selectedTemplateName = mint.template.name
            for warning in mint.warnings { note(warning, severity: .warning) }
            guard mint.minted else { return }  // silent dedup: the selection IS the answer
            if let parent = mint.divergedFrom {
                note(
                    "created design '\(mint.template.name)' — '\(name)' had "
                        + "diverged from '\(parent)', so this is a new design "
                        + "rather than an edit of that one",
                    severity: .success)
            } else {
                note("created design '\(mint.template.name)'", severity: .success)
            }
        } catch {
            refuse(
                .template,
                "Couldn't load '\(name)' as a design — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    /// Edit a design's description. METADATA, not design: the description is
    /// excluded from the content hash (see `StudyTemplateStore.hash`), so this
    /// cannot make an instance diverge — which is exactly why it is the one
    /// field the read-only library lets you change.
    public func updateTemplateDescription(_ name: String, to description: String) {
        clearFormError(.template)
        do {
            var template = try StudyTemplateStore.load(name: name)
            guard template.templateDescription != description else { return }
            template.templateDescription = description
            try StudyTemplateStore.save(template)
            refreshTemplates()
        } catch {
            refuse(
                .template,
                "Couldn't save the description — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    // MARK: The design round trip (Edit design… → Studies → Save back)

    /// "Edit design…": mint the agentless scratch draft a design is REVISED
    /// through, and select it.
    ///
    /// The whole affordance is this one line of indirection. Revising a design
    /// means editing a manifest, there is exactly one manifest editor (Studies),
    /// and a second editor in the design library would drift from it the first
    /// time a field is added to only one — so the design is cast into an
    /// ordinary draft and the researcher edits THAT. The view navigates to
    /// Studies on a non-nil return; nothing here knows about tabs.
    @discardableResult
    public func editDesign(_ name: String) -> String? {
        clearFormError(.template)
        do {
            let draft = try StudyTemplateStore.mintEditDraft(templateName: name)
            refresh()
            selectedName = draft.name
            note(
                "opened '\(draft.name)' — an ordinary draft of design "
                    + "'\(name)'. Edit it here, then Save back to design to "
                    + "update '\(name)' in place",
                severity: .success)
            return draft.name
        } catch {
            refuse(
                .template,
                "Couldn't open design '\(name)' for editing — "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
            return nil
        }
    }

    /// "New Template": the blank version of the same loop — a scratch draft with
    /// no design behind it, which becomes a design via "Save as new design".
    ///
    /// Deliberately the ORDINARY from-scratch creation path (`newStudy`), not a
    /// second one: a design authored from nothing and a study authored from
    /// nothing are the same manifest, and the only difference is what the
    /// researcher does with it at the end.
    ///
    /// The draft is NOT pre-stamped with a lineage line saying "unsaved design".
    /// `templateProvenance` names a design and pins its hash; stamping a design
    /// that does not exist would make `agreement(of:)` report `.designMissing`
    /// and the lineage line say "no longer in the library" — a false statement
    /// about the library, to hint at an intention. The pointer is UI copy
    /// instead.
    @discardableResult
    public func newDesignDraft() -> String? {
        clearFormError(.template)
        newStudy()
        return selectedName
    }

    /// The design a study can be saved back ONTO, or nil.
    ///
    /// Nil is not an error state — most studies have no design behind them.
    /// Paired with `saveBackToDesignRefusal` so the control can be present and
    /// explain itself rather than vanish.
    public func saveBackToDesignTarget(
        for manifest: ExperimentManifest
    ) -> String? {
        guard let provenance = manifest.templateProvenance else { return nil }
        guard templates.contains(where: { $0.name == provenance.template })
        else { return nil }
        return provenance.template
    }

    /// Why the selected study cannot be saved back onto a design, or nil.
    ///
    /// TWO refusals, both about the design rather than the study: it has to
    /// exist, and this study has to name it.
    ///
    /// Status is deliberately NOT one of them (2026-08-06). A design carries no
    /// lifecycle stamps — `strippedBody` removes every one of them — so writing
    /// a frozen study's settings onto a design cannot make the design claim to
    /// be frozen, and it cannot touch the frozen study either: the write goes to
    /// `templates/<name>/`, the run record is untouched, and studies minted
    /// earlier keep the hash stamped at their own mint time. "New from Study"
    /// already accepts any status for exactly this reason; refusing here only
    /// forced the researcher to duplicate a frozen study into a draft to write
    /// the same bytes.
    public func saveBackToDesignRefusal(
        for manifest: ExperimentManifest
    ) -> String? {
        guard let provenance = manifest.templateProvenance else {
            return "'\(manifest.name)' was not minted from a design — use Save "
                + "as new design"
        }
        guard templates.contains(where: { $0.name == provenance.template })
        else {
            return "design '\(provenance.template)' is no longer in the library "
                + "(renamed or deleted) — use Save as new design"
        }
        return nil
    }

    /// Overwrite the design the selected draft names, in place.
    ///
    /// The counterpart to `newDesignFromStudy`: both are offered, both are
    /// worded for what they do, and neither is a default. The in-place write
    /// bumps the design's content hash; studies minted from it earlier keep the
    /// hash stamped at their own mint time, so their lineage lines go on saying
    /// what they were actually minted from (see
    /// `StudyTemplateStore.saveStudyBackToDesign`).
    public func saveSelectedStudyBackToDesign() {
        guard let manifest = selected else { return }
        clearFormError(.template)
        if let refusal = saveBackToDesignRefusal(for: manifest) {
            refuse(.template, "Couldn't save back to a design — " + refusal)
            return
        }
        do {
            let update = try StudyTemplateStore.saveStudyBackToDesign(
                experimentName: manifest.name)
            refreshTemplates()
            selectedTemplateName = update.design
            for warning in update.warnings { note(warning, severity: .warning) }
            guard update.changed else {
                note(
                    "design '\(update.design)' already matched "
                        + "'\(manifest.name)' — nothing about the recipe moved",
                    severity: .info)
                return
            }
            note(
                "updated design '\(update.design)' in place "
                    + "(\(update.hashBefore.prefix(12))… → "
                    + "\(update.hashAfter.prefix(12))…) — studies minted from it "
                    + "earlier keep their original lineage stamps",
                severity: .success)
        } catch {
            refuse(
                .template,
                "Couldn't update the design — nothing was written. "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    // `loadSelectedStudyAsTemplate` (the Studies tab's old "Load as Template"
    // button) is gone with the 2026-08-06 restructure: saving a study as a
    // design happens in the design library, where the result is visible, and
    // two near-identical mint paths differing only in how loudly they
    // announced themselves was one path too many. `newDesignFromStudy` above
    // is the single entry point.

    public func renameTemplate(_ oldName: String, to newName: String) {
        clearFormError(.template)
        do {
            let resolved = try StudyTemplateStore.rename(
                templateName: oldName, to: newName)
            refreshTemplates()
            selectedTemplateName = resolved
            guard resolved != oldName else { return }
            note(
                "renamed template '\(oldName)' → '\(resolved)' — studies already "
                    + "minted from it keep the old name in their lineage stamp",
                severity: .success)
        } catch {
            refuse(
                .template,
                "Couldn't rename the template — nothing changed. "
                    + ((error as? ExperimentError)?.reason ?? "\(error)"))
        }
    }

    public func deleteTemplate(_ name: String) {
        clearFormError(.template)
        do {
            try StudyTemplateStore.delete(name: name)
            if selectedTemplateName == name { selectedTemplateName = nil }
            refreshTemplates()
            note(
                "deleted template '\(name)' — studies minted from it are "
                    + "untouched ordinary drafts",
                severity: .success)
        } catch {
            refuse(.template, "Couldn't delete the template: \(error)")
        }
    }

    /// The lineage line for a minted study: which recipe it came from, and
    /// which batch of siblings it belongs to.
    ///
    /// Sibling counting is by `batchGroup` across the workspace, because
    /// panel castings are sibling STUDIES by necessity (one scenario per
    /// manifest on both engines) and the batch id is the only thing tying
    /// them together.
    /// Divergence is READ FROM THE CACHE built at refresh (`designLineage`),
    /// never recomputed here: this is called from a view body, and the check
    /// re-reads files.
    public func templateLineage(_ manifest: ExperimentManifest) -> String? {
        guard let provenance = manifest.templateProvenance else { return nil }
        // Editable-but-visibly-diverged is the policy: nothing here blocks an
        // edit, and the word changes so the lineage line stops claiming a
        // replication the study no longer is.
        let lineage =
            designLineage[manifest.name]
            ?? .init(agreement: .matches, designRevised: false)
        let agreement = lineage.agreement
        var line = agreement == .diverged
            ? "diverged from design '\(provenance.template)' "
            : "from design '\(provenance.template)' "
        line += "@ \(provenance.templateHash.prefix(12))…"
        if agreement == .designMissing {
            line += " (no longer in the library)"
        }
        // The SECOND, independent fact. Without it a study minted before a
        // design was revised reads as a plain "from design 'X'" forever, and
        // the researcher takes that as a replication of the recipe now
        // filed under X — which it is not. The study's own stamp comparison
        // is unchanged and still leads the line.
        if lineage.designRevised {
            line +=
                agreement == .diverged
                ? " · the design has since been revised too"
                : " · matches its design as minted · the design has since "
                    + "been revised"
        }
        if let batch = provenance.batchGroup {
            let siblings = experiments.filter {
                $0.templateProvenance?.batchGroup == batch
            }
            line += " · batch \(batch)"
            if siblings.count > 1 {
                line += " (\(siblings.count) studies minted together)"
            }
        }
        if agreement == .diverged {
            line += " — edited since minting; edits are allowed and the stamp "
                + "records where it started"
        }
        return line
    }

    /// Why the selected study cannot be deleted, or nil.
    ///
    /// Draft-only, and that is the STORE's rule, not a UI choice:
    /// `ExperimentStore.moveDraftToTrash` refuses anything frozen or complete
    /// with the immutability line, because a frozen manifest is what every run
    /// directory's stamp points at. Surfaced as a reason rather than a hidden
    /// button so the answer to "why can't I delete this?" is on the control.
    public var deleteSelectedStudyRefusal: String? {
        guard let manifest = selected else { return "select a study first" }
        guard manifest.status == .draft else {
            return "'\(manifest.name)' is \(manifest.status.rawValue) — frozen "
                + "and completed studies are immutable and cannot be deleted "
                + "(their runs stamp them); duplicate as a draft to iterate"
        }
        return nil
    }

    /// The display names of the studies minted in the same batch as this one,
    /// excluding itself. Empty when the study has no batch.
    public func batchSiblings(_ manifest: ExperimentManifest) -> [String] {
        guard let batch = manifest.templateProvenance?.batchGroup else { return [] }
        return experiments
            .filter {
                $0.templateProvenance?.batchGroup == batch
                    && $0.name != manifest.name
            }
            .map { displayName($0) }
    }

    /// Move the selected DRAFT to a `.trash-<timestamp>` sibling (App gap
    /// A12) — never a destructive delete; frozen/completed studies refuse
    /// inside the store with the immutability line.
    public func deleteSelectedDraft() {
        guard let name = selectedName else { return }
        do {
            let destination = try ExperimentStore.moveDraftToTrash(name: name)
            selectedName = nil
            refresh()
            note("moved draft '\(name)' to "
                + "experiments/\(destination.deletingLastPathComponent().lastPathComponent)/"
                + "\(destination.lastPathComponent) — recover it from there if needed", severity: .success)
        } catch {
            note(
                "Couldn't move the draft to trash — nothing was deleted; "
                    + "frozen studies can never be deleted, and the "
                    + "experiments/ folder must be writable. Details: \(error)",
                severity: .error)
        }
    }

    public func testRemoteConnection() async {
        loadStoredRemoteToken()
        guard let remoteClient else {
            remoteStatus = "invalid server URL"
            return
        }
        do {
            let caps = try await remoteClient.capabilities()
            cluster?.persistToken()
            remoteProfileSummary = Self.profileSummary(caps)
            remoteStatus = "connected: \(caps.engine ?? "server") \(caps.serverVersion ?? "")"
        } catch {
            remoteStatus = "remote connection failed: \(error)"
        }
    }

    /// One-line description of the backend so a client can tell a local dev
    /// server from a cluster, and whether it is a batch (allocation-scoped) box.
    static func profileSummary(_ caps: ClusterCapabilities) -> String {
        var parts: [String] = []
        if case .string(let profile)? = caps.profile?["profile"] { parts.append(profile) }
        if case .string(let executor)? = caps.profile?["executor"] { parts.append(executor) }
        if case .string(let topology)? = caps.profile?["launchTopology"], topology != "local" {
            parts.append(topology)
        }
        if caps.remoteStudy?.externalTransferRequired == true { parts.append("external-transfer") }
        return parts.isEmpty ? "local" : parts.joined(separator: " · ")
    }

    /// Reconnect to a running/finished job by id (e.g. after an app restart),
    /// resuming the live log tail and refreshing status.
    public func reconnectRemoteJob(_ id: String) async {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            remoteStatus = "enter a job id to reconnect"
            return
        }
        remoteJobID = trimmed
        remoteLogLines = []
        loadStoredRemoteToken()
        remoteLogTask?.cancel()
        remoteLogTask = Task { [weak self] in await self?.streamRemoteJobLog(jobID: trimmed) }
    }

    public func stopRemoteLogStream() {
        remoteLogTask?.cancel()
        remoteLogTask = nil
    }

    /// The no-GPU-allocation dialog's "Fix options" action (2026-07-21
    /// incident, part 1): snap the Remote options to a submission that
    /// actually requests a GPU — executor "slurm", and an empty gres
    /// prefilled with the active site's first GPU vocabulary entry. Pure
    /// rule in `ModelJobSubmissionPreflight.fixedOptions`; the view also
    /// reveals the Remote options so the researcher reviews before
    /// resubmitting (nothing is auto-submitted).
    public func applyGPUAllocationFix() {
        var siteGPUTypes: [String] = []
        if case .slurm(let slurm)? = cluster?.activeSite?.scheduler {
            siteGPUTypes = slurm.gpuTypes
        }
        let fixed = ModelJobSubmissionPreflight.fixedOptions(
            executor: remoteExecutor, gres: remoteGres, siteGPUTypes: siteGPUTypes)
        remoteExecutor = fixed.executor
        remoteGres = fixed.gres
        pendingRevealRemoteControls = true
        note(
            "Remote options now request a GPU (executor slurm"
                + (fixed.gres.isEmpty ? "" : ", gres \(fixed.gres)")
                + ") — review them, then Run again",
            severity: .info)
    }

    /// `verbOverride` submits a specific verb regardless of the Remote
    /// options' verb picker — the Mac-authority Validate path submits
    /// "validate" as a bundle job without touching the picker state. All
    /// other Remote options (executor, gres, walltime, dry run) apply
    /// as configured.
    public func submitSelectedStudyRemotely(verbOverride: String? = nil) async {
        // Every refusal is a NOTICE, not just a status line (2026-07-19
        // paper cut): remoteStatus renders inside the Remote options
        // disclosure, which may be collapsed — a submission that failed
        // must be loud wherever the user pressed the button.
        guard let manifest = selected else {
            remoteStatus = "select a study first"
            note("select a study first", severity: .info)
            return
        }
        _ = await submitStudyRemotely(
            manifest, verbOverride: verbOverride, followLog: true)
    }

    /// Whether a bundle submission has anywhere to go. Read by the batch UI so
    /// "Mint & Submit All" is unavailable-with-a-reason rather than six
    /// identical connection failures in a row.
    public var canSubmitBundles: Bool { remoteClient != nil }

    /// Submits ONE named study through the same bundle path the single-study
    /// button uses — the per-row action of a template batch.
    ///
    /// Log-following is OFF: `remoteLogTask` is a single slot and each new
    /// follower cancels the previous one, so a six-study batch that followed
    /// every job would end up streaming only the last. Every submission is
    /// still recorded in Recent Server Jobs, which is where a batch is watched
    /// from.
    public func submitStudyBundle(
        named name: String
    ) async -> Result<String, StudyBatchSubmission.Failure> {
        guard let manifest = experiments.first(where: { $0.name == name }) else {
            return .failure(
                StudyBatchSubmission.Failure(
                    reason: "study '\(name)' is no longer in this workspace"))
        }
        return await submitStudyRemotely(manifest, followLog: false)
    }

    /// The one bundle-submit implementation: package, upload, submit, stamp.
    ///
    /// Extracted from `submitSelectedStudyRemotely` unchanged so a batch row
    /// and a single click take the identical path — including every guard
    /// (frozen-on-server, stochastic-agent refusal) and the recent-jobs stamp.
    /// The only difference a batch asks for is that it does not seize the
    /// shared log stream.
    @discardableResult
    private func submitStudyRemotely(
        _ manifest: ExperimentManifest,
        verbOverride: String? = nil,
        followLog: Bool
    ) async -> Result<String, StudyBatchSubmission.Failure> {
        let submissionVerb = verbOverride ?? remoteVerb
        loadStoredRemoteToken()
        guard let remoteClient else {
            remoteStatus = "invalid server URL"
            let refusal =
                "remote submit refused: no server connection — connect a "
                + "server in the substrate selector first"
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
        // Frozen-on-server guard — SHARED with every bundle path
        // (`ClusterClient.frozenOnServerConflict`): a local draft must not
        // shadow the server's frozen same-named study.
        if let conflict = await remoteClient.frozenOnServerConflict(
            study: manifest.name, localStatus: manifest.status)
        {
            remoteStatus = conflict
            note(conflict, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: conflict))
        }
        // Old-server guard (2026-07-21): a stochastic saved-agent study on a
        // server without study-owned sampling would run the agents greedy
        // while the baseline samples — refuse BEFORE packaging/uploading
        // (same rule as UnifiedStudyRunner.submitBundle).
        if let refusal = SubstrateRouting.stochasticVariantSubmissionRefusal(
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem,
            variantConditionCount: manifest.variantConditions.count,
            verb: submissionVerb,
            capabilities: cluster?.capabilities)
        {
            remoteStatus = refusal
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
        // Scope-drift guard (2026-08-06 field incident) — same rule as
        // UnifiedStudyRunner.submitBundle: a stale outcomeInstrumentScope
        // pin refuses at SUBMIT, before packaging/upload, instead of on the
        // compute node after the model staged.
        if let refusal = ExperimentTasks.scopeDriftSubmitRefusal(
            for: manifest, verb: submissionVerb)
        {
            remoteStatus = refusal
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
        do {
            remoteStatus = "packaging \(manifest.name)..."
            remoteLogLines = []
            // Packaging copies files, hashes them, and shells out to tar; keep it
            // off the main actor so the UI doesn't freeze on a real study bundle.
            let bundle = try await Task.detached {
                try RunBundlePackager.packageExperiment(manifest)
            }.value
            remoteStatus = "uploading \(bundle.lastPathComponent)..."
            let uploaded = try await remoteClient.uploadBundle(bundle)
            remoteLastUploadedBundle = uploaded.path
            let resources = [
                "gres": remoteGres,
                "walltime": remoteWalltime,
            ].filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            remoteStatus = "submitting \(submissionVerb)..."
            // The resume policy applies to Slurm submissions only — sending
            // it on a local-executor submission would stamp a policy that
            // cannot act.
            let resumePolicy: RemoteResumePolicy? =
                remoteExecutor == "slurm" ? remoteResumePolicy : nil
            let submission = try await remoteClient.submitBundle(
                path: uploaded.path,
                verb: submissionVerb,
                executor: remoteExecutor,
                dryRun: remoteDryRun,
                resources: resources,
                resumePolicy: resumePolicy,
                parallelJobs: remoteParallelJobs)
            remoteJobID = submission.jobId
            let substrate = cluster?.substrateLabel ?? submission.executor
            var submitted = Self.bundleSubmittedStatus(
                study: manifest.name, verb: submissionVerb, dryRun: remoteDryRun,
                substrate: substrate, jobID: submission.jobId)
            if let resumePolicy {
                submitted += " — \(resumePolicy.transcriptStamp)"
            }
            // The sharding stamp derives from the server's RESPONSE (the
            // shard ids it actually created), never from the request — the
            // server may have ignored the fan-out (finding 5, 2026-07-22).
            if let stamp = ShardedSubmission.transcriptStamp(
                shardJobIDs: submission.shardJobIDs)
            {
                submitted += " — \(stamp)"
            }
            remoteStatus = submitted
            note(submitted, severity: .info)
            noteRecentServerJob(
                id: submission.jobId, verb: "\(submissionVerb) (bundle)",
                study: manifest.name, state: "submitted")
            if followLog {
                remoteLogTask?.cancel()  // don't interleave with a prior stream
                let verb = submissionVerb
                let dryRun = remoteDryRun
                let study = manifest.name
                remoteLogTask = Task { [weak self] in
                    await self?.followBundleJob(
                        jobID: submission.jobId, verb: verb, study: study, dryRun: dryRun)
                }
            }
            return .success(submission.jobId)
        } catch {
            remoteStatus = "remote submit failed: \(error)"
            let refusal =
                "Couldn't submit the bundle to the server — nothing is "
                + "running; check the connection in Compute and submit "
                + "again. Details: \(error)"
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
    }

    /// Follows a bundle-submission job in the SHARED display pane (the same
    // MARK: Two-phase sweep judgment (key-custody design 2026-07-18)

    /// Sweep AND evaluate runs of the selected study awaiting Mac-side
    /// external judgment (each record's `kind` says which). Refreshed
    /// alongside the remote surfaces; empty on non-server hosts.
    public private(set) var awaitingSweepJudgments:
        [ClusterClient.AwaitingSweepJudgment] = []
    /// True while a judging pass runs on this Mac (single-flight).
    public private(set) var isJudgingSweep = false

    public func refreshAwaitingSweepJudgments(study: String) async {
        guard let client = remoteClient else {
            awaitingSweepJudgments = []
            return
        }
        // Evaluate awaiting is best-effort separately: an older server
        // without the route must not hide judgable sweep work.
        let sweeps =
            (try? await client.awaitingSweepJudgments(experiment: study)) ?? []
        let evaluates =
            (try? await client.awaitingEvaluateJudgments(experiment: study))
            ?? []
        awaitingSweepJudgments = sweeps + evaluates
    }

    /// Phase 2 on THIS Mac: fetch the blinded packets (hash-verified), judge
    /// them with the Keychain key, and hand the judgments to the server's
    /// completion verb (which verifies every pin and replays the selection).
    public func judgeAwaitingSweep(
        study: String, awaiting: ClusterClient.AwaitingSweepJudgment
    ) async {
        guard let client = remoteClient else {
            remoteStatus = "no server connection for judging"
            return
        }
        guard !isJudgingSweep else { return }
        isJudgingSweep = true
        defer { isJudgingSweep = false }
        do {
            let judgmentRun = try await SweepJudgmentRunner.judgeAndComplete(
                client: client, experiment: study, awaiting: awaiting,
                onProgress: { [weak self] progress in
                    await MainActor.run { self?.remoteStatus = progress }
                })
            remoteStatus = awaiting.isEvaluate
                ? "evaluation judged on this Mac → judge report completed "
                    + "(\(judgmentRun))"
                : "sweep judged on this Mac → selection completed "
                    + "(\(judgmentRun)) — recommendations updated"
            await refreshAwaitingSweepJudgments(study: study)
            refresh()
        } catch {
            remoteStatus = "sweep judging failed: "
                + String(describing: error)
        }
    }

    /// live-log affordance direct server runs use), mirroring each line into
    /// `remoteLogLines` for the Studies disclosure. Bundle jobs must be
    /// visible in the activity pane, not only inside Compute.
    private func followBundleJob(
        jobID: String, verb: String, study: String, dryRun: Bool
    ) async {
        guard let client = remoteClient else { return }
        guard host != nil else {
            // No display host (headless/test) — fall back to the plain
            // remote-log stream so the lines still land somewhere.
            await streamRemoteJobLog(jobID: jobID)
            return
        }
        let label = dryRun ? "\(verb) (dry run)" : verb
        let job = await followServerJobInDisplay(
            jobID: jobID,
            client: client,
            title: "Bundled study \(label) — \(study) [job \(jobID)]",
            label: "bundle \(label)",
            mirrorToRemoteLog: true)
        guard let job else { return }
        noteRecentServerJob(
            id: jobID, verb: "\(verb) (bundle)", study: study, state: job.status)
        if job.status == "succeeded" {
            // Executing a bundle imports the study into the server's tree —
            // the residency preflight may now say yes; drop the cache and
            // re-check so Run Server Copy re-enables without reselecting.
            serverResidencyKey = nil
            await refreshServerResidency()
            // Evidence comes home (Mac-authority mode, 2026-07-21): a
            // bundled validate exists to mint freeze evidence for THIS
            // workspace — import its evidence bundle now and recompute
            // freeze readiness, instead of waiting for the auto-import
            // poll. Hash-verified by the same importer either way; a
            // failure surfaces in the status line and the run stays
            // importable manually.
            if verb == "validate" {
                await importEvidence(fromJobID: jobID)
                refresh()
            }
        }
        if job.status == "prepared" {
            remoteStatus = "job \(jobID) prepared — dry run staged the bundle; "
                + "nothing executed"
        } else if let error = job.error, !error.isEmpty {
            remoteStatus = "job \(jobID) \(job.status): \(error)"
        } else {
            remoteStatus = "job \(jobID) \(job.status)"
        }
    }

    // MARK: Run on the active server (durable jobs, no bundle transfer)

    /// Runs an experiment verb on the ACTIVE server workspace as a durable
    /// job (`POST /api/experiment/{name}/{verb}`), streams the job log into
    /// the display pane, refreshes the server run listing on completion, and
    /// surfaces the produced run directory. This is the connected-server
    /// path: the study must exist in the server's own experiments/ tree.
    /// The hash-pinned bundle upload (`submitSelectedStudyRemotely`) remains
    /// the remote-cluster path.
    public func runStudyOnActiveServer(verb: String = "run") async {
        guard let name = selectedName else {
            note("select a study first", severity: .info)
            return
        }
        await runExperimentVerbOnActiveServer(experimentName: name, verb: verb)
    }

    /// The parameterized core: Optimizations keeps its OWN selection, so its
    /// Optimize must be able to submit for an arbitrary experiment name without
    /// touching the Studies panel's `selectedName`.
    public func runExperimentVerbOnActiveServer(
        experimentName name: String, verb: String
    ) async {
        guard isServerWorkspace else {
            note("no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note("invalid server URL", severity: .error)
            return
        }
        let substrate = cluster?.substrateLabel ?? "server"
        do {
            // Preflight: direct experiment verbs run SERVER-RESIDENT studies
            // only. On a workspace-paired server this always passes; on an
            // unpaired (remote) server it refuses with the portable path
            // named, instead of a confusing job-side missing-file failure.
            if let names = try? await client.experimentNames() {
                let resident = names.contains(name)
                // The cached residency answer belongs to the Studies
                // selection; an Optimizations-initiated verb for another study must
                // not overwrite it.
                if name == selectedName { noteServerResidency(resident) }
                if !resident {
                    note("study '\(name)' is not in \(substrate)'s workspace — "
                        + "direct runs execute the server-resident copy only. Pair "
                        + "the server to this workspace (serve --root <workspace>) "
                        + "or use Submit Bundle, the portable path for remote engines", severity: .info)
                    return
                }
            }
            note("submitting \(verb) for '\(name)' to \(substrate)…", severity: .info)
            let jobID = try await client.submitExperimentJob(experiment: name, verb: verb)
            remoteJobID = jobID
            activeServerJob = ActiveServerJob(id: jobID, verb: verb, study: name)
            // Sweep jobs additionally occupy their own slot so Optimizations'
            // Cancel targets exactly this job, never another flow's.
            if verb == "sweep" { activeSweepJob = activeServerJob }
            noteRecentServerJob(id: jobID, verb: verb, study: name, state: "pending")
            note("server \(verb) job \(jobID) submitted for '\(name)' on \(substrate)", severity: .info)
            let job = await followServerJobInDisplay(
                jobID: jobID,
                client: client,
                title: "Server study \(verb) — \(name) [job \(jobID)]",
                label: "study \(verb)")
            if let job {
                // Terminal: the cancel affordance retires. A timed-out follow
                // (job == nil) deliberately KEEPS activeServerJob set — the
                // job is still running server-side and must stay cancellable.
                activeServerJob = nil
                if verb == "sweep" { activeSweepJob = nil }
                noteRecentServerJob(id: jobID, verb: verb, study: name, state: job.status)
                if let result = job.result,
                    let directory = Self.findString(
                        in: .object(result), keyPath: ["runDirectory"])
                {
                    lastServerRunDirectory = directory
                }
                if let error = job.error, !error.isEmpty {
                    note("server \(verb) job \(jobID) \(job.status): \(error)", severity: .error)
                } else {
                    note("server \(verb) job \(jobID) \(job.status)"
                        + (lastServerRunDirectory.map { " → \($0)" } ?? ""), severity: .info)
                }
            } else {
                note("server \(verb) job \(jobID) still running on \(substrate) — "
                    + "reconnect from Compute or the recent-jobs list", severity: .info)
            }
            await refreshRemoteRuns()
            await refreshRecentServerJobs()
        } catch {
            note(
                "Couldn't submit the \(verb) job to the server — nothing is "
                    + "running on the server; check the connection in Compute "
                    + "and submit again. Details: \(error)",
                severity: .error)
        }
    }

    /// Upserts a session-scoped recent-job row (newest first, capped).
    private func noteRecentServerJob(id: String, verb: String, study: String, state: String) {
        if let index = recentServerJobs.firstIndex(where: { $0.id == id }) {
            recentServerJobs[index].state = state
            return
        }
        recentServerJobs.insert(
            RecentServerJob(id: id, verb: verb, study: study, state: state), at: 0)
        if recentServerJobs.count > 15 {
            recentServerJobs.removeLast(recentServerJobs.count - 15)
        }
    }

    /// Refreshes recent-job states from the server (`client.jobs()`,
    /// filtered to experiment job kinds). Jobs this panel did not submit are
    /// appended too — they persist server-side and remain reconnectable.
    public func refreshRecentServerJobs() async {
        guard let client = remoteClient else { return }
        guard let jobs = try? await client.jobs() else { return }
        for job in jobs
        where job.kind.hasPrefix("experiment:") || job.kind.hasPrefix("study-submit") {
            if let index = recentServerJobs.firstIndex(where: { $0.id == job.id }) {
                recentServerJobs[index].state = job.status
            } else {
                let verb = job.kind.split(separator: ":").last.map(String.init) ?? job.kind
                noteRecentServerJob(id: job.id, verb: verb, study: "—", state: job.status)
            }
        }
    }

    /// Streams a server job's log into a display-pane live log until the
    /// stream ends, then resolves the job's terminal record (panel twin of
    /// `ConceptBuilder.followServerJobInDisplay`). Returns nil when the job
    /// is still running at timeout or the task was cancelled — the job keeps
    /// running server-side either way.
    private func followServerJobInDisplay(
        jobID: String,
        client: ClusterClient,
        title: String,
        label: String,
        maxLines: Int = 400,
        mirrorToRemoteLog: Bool = false
    ) async -> RemoteJobRecord? {
        guard let host else { return nil }
        var lines = ["queued job \(jobID)"]
        let logID = host.startLiveLog(title: title, initialLine: lines[0])

        do {
            try await client.streamJobLog(jobID: jobID) { line in
                await MainActor.run {
                    lines.append(line)
                    if lines.count > maxLines {
                        lines.removeFirst(lines.count - maxLines)
                    }
                    host.updateLiveLog(id: logID, title: title, lines: lines)
                    self.status = "\(label) job \(jobID): \(line)"
                    if mirrorToRemoteLog {
                        self.appendRemoteLogLine(line)
                    }
                }
            }
        } catch is CancellationError {
            return nil
        } catch {
            lines.append("log stream ended: \(error.localizedDescription)")
            host.updateLiveLog(id: logID, title: title, lines: lines)
        }

        if let job = try? await client.job(jobID),
            Self.terminalJobStatuses.contains(job.status) || job.finishedAt != nil
        {
            lines.append("job \(job.status)")
            if let error = job.error, !error.isEmpty {
                lines.append(error)
            }
            host.updateLiveLog(id: logID, title: title, lines: lines)
            return job
        }
        // Stream gone but the job is unresolved (transient fetch error, or a
        // non-terminal status like "cancelling"/"running" after a broken
        // stream): poll until terminal, mirroring ConceptBuilder's fallback.
        let deadline = Date().addingTimeInterval(600)
        while !Task.isCancelled, Date() < deadline {
            if let job = try? await client.job(jobID) {
                if Self.terminalJobStatuses.contains(job.status)
                    || job.finishedAt != nil
                {
                    return job
                }
                if RemoteJobStatusClass.classify(status: job.status) == .resumable {
                    // Actionable, not a dead end (2026-07-22 incident): the
                    // checkpoint line names the Resume button and the
                    // auto-resume toggle.
                    note(
                        RemoteJobStatusClass.checkpointGuidance(jobID: jobID),
                        severity: .warning)
                } else {
                    note("\(label) job \(jobID) \(job.status)"
                        + (job.logTail.last.map { ": \($0)" } ?? "…"), severity: .info)
                }
            }
            try? await Task.sleep(for: .seconds(2))
        }
        return nil
    }

    /// Terminal statuses of the server's durable job store. "prepared" is
    /// the terminal outcome of a dry-run submission (staged, nothing
    /// executed) and "parked" of a worker that stopped short with durable
    /// state and a recovery action; "cancelling" is deliberately NOT here —
    /// it is the non-terminal window between a cancel request and the
    /// worker's acknowledgement, and a follower must keep following through
    /// it.
    private static let terminalJobStatuses: Set<String> = [
        "succeeded", "failed", "cancelled", "prepared", "parked",
    ]

    /// Appends to the Studies-disclosure job log with the shared 400-line cap.
    private func appendRemoteLogLine(_ line: String) {
        remoteLogLines.append(line)
        if remoteLogLines.count > 400 {
            remoteLogLines.removeFirst(remoteLogLines.count - 400)
        }
    }

    public func streamRemoteJobLog(jobID: String? = nil) async {
        guard let remoteClient else { return }
        guard let id = jobID ?? remoteJobID else {
            remoteStatus = "no remote job selected"
            return
        }
        do {
            try await remoteClient.streamJobLog(jobID: id) { [weak self] line in
                await MainActor.run {
                    self?.appendRemoteLogLine(line)
                }
            }
            if let job = try? await remoteClient.job(id) {
                remoteStatus = "job \(id): \(job.status)"
            }
        } catch {
            remoteStatus = "remote log stream failed: \(error)"
        }
    }

    /// Fetch the active server's run-directory listing. Runs are per-substrate
    /// artifacts: this is a read-only browse of what exists server-side, not a
    /// merge into the local results list. Full result *detail* (reports,
    /// generations) still comes home via the evidence-bundle import flow —
    /// the server exposes per-run files (`GET /api/runs/{id}/file`) but no
    /// structured results API yet.
    public func refreshRemoteRuns() async {
        guard let cluster, case .server = cluster.activeWorkspace,
            let remoteClient
        else {
            remoteRuns = []
            return
        }
        do {
            remoteRuns = try await remoteClient.runs()
            remoteStatus = "listed \(remoteRuns.count) server run"
                + (remoteRuns.count == 1 ? "" : "s")
        } catch {
            remoteRuns = []
            remoteStatus = "could not list server runs: \(error)"
        }
    }

    /// Chain-runner runs for the selected experiment (stage 5): per-stage
    /// status, disposition, and the abort record's gate details. Older
    /// servers without the route read as "none". Local/imported chains are
    /// listed from the workspace runs tree (portable ledger preferred).
    /// The captured name is re-checked after the await (seventh round): a
    /// slow response for a PREVIOUS selection must never overwrite the
    /// current one's list.
    public func refreshPipelineRuns() async {
        guard let name = selectedName else {
            pipelineRuns = []
            localPipelineRuns = []
            return
        }
        localPipelineRuns = LocalPipelineCatalog.summaries(experiment: name)
        guard let cluster, case .server = cluster.activeWorkspace,
            let remoteClient
        else {
            pipelineRuns = []
            return
        }
        do {
            let runs = try await remoteClient.pipelineRuns(experiment: name)
            guard selectedName == name else { return }
            pipelineRuns = runs
        } catch {
            if selectedName == name { pipelineRuns = [] }
        }
    }

    // MARK: Remote Results browsing (unpaired server workspaces)

    /// Where the Results browser sources its run list from, per compute mode.
    /// - `local`: the Local (MLX) workspace — scan this workspace's `runs/`.
    /// - `pairedServer`: a server sharing this workspace's artifact root —
    ///   server runs land in the SAME tree, so the local scan already shows
    ///   them; browsing stays local (with a caption saying so) rather than
    ///   rendering a duplicate list.
    /// - `remoteServer`: an unpaired (or not-yet-verified) server — browse
    ///   ITS `runs/` read-only over the API, source-labeled like Optimizations.
    public enum ResultsSource: Sendable, Equatable {
        case local
        case pairedServer
        case remoteServer
    }

    public var resultsSource: ResultsSource {
        guard let cluster, case .server = cluster.activeWorkspace else { return .local }
        switch cluster.activeServerPairing {
        case .paired: return .pairedServer
        default: return .remoteServer
        }
    }

    /// The active server's enriched `runs/` listing (config.json stamps +
    /// file sizes) for the remote Results browser. Cleared when no server
    /// workspace is active.
    public private(set) var remoteResultsRuns: [RemoteStampedRunRecord] = []
    public private(set) var remoteResultsStatus: String?
    public private(set) var isLoadingRemoteResults = false

    /// The remote run selected in the Results pane — the remote sibling of
    /// `ChatService.selectedResultsRun` (which stays a LOCAL `RunBrowser.Item`
    /// and is owned elsewhere). The activity pane's summary column mirrors
    /// whichever selection matches the active results source.
    public var selectedRemoteResultsRun: RemoteStampedRunRecord?

    /// The FILE focused in the LOCAL Results run detail (lives beside
    /// `ChatService.selectedResultsRun`): the Results detail pane lists the
    /// run's files and sets this; the activity viewer's Results mode renders
    /// this file's bounded preview. Pure UI selection state — no run data.
    public var selectedResultsFile: RunBrowser.FileEntry?

    /// Refresh the remote Results listing from the active server. Keeps the
    /// current selection when its run id survives the refresh.
    public func refreshRemoteResultsRuns() async {
        guard let cluster, case .server = cluster.activeWorkspace else {
            remoteResultsRuns = []
            selectedRemoteResultsRun = nil
            remoteResultsStatus = nil
            return
        }
        // Not connected yet (fresh launch, server workspace persisted from a
        // prior session): a friendly empty state, never an attempted fetch
        // that surfaces a raw transport error.
        guard cluster.remoteState != nil else {
            remoteResultsRuns = []
            selectedRemoteResultsRun = nil
            remoteResultsStatus = "not connected to \(cluster.substrateLabel) "
                + "— connect in Compute to browse its runs"
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            remoteResultsRuns = []
            remoteResultsStatus = "invalid server URL"
            return
        }
        isLoadingRemoteResults = true
        defer { isLoadingRemoteResults = false }
        do {
            let runs = try await client.stampedRuns()
            remoteResultsRuns = runs
            if let selected = selectedRemoteResultsRun {
                selectedRemoteResultsRun = runs.first { $0.id == selected.id }
            }
            let substrate = cluster.substrateLabel
            remoteResultsStatus = "\(runs.count) run\(runs.count == 1 ? "" : "s") "
                + "on \(substrate)"
        } catch {
            remoteResultsRuns = []
            // Human-sized reason, not a raw Swift error dump.
            remoteResultsStatus = "could not reach \(cluster.substrateLabel) — "
                + "check the connection in Compute "
                + "(\(error.localizedDescription))"
        }
    }

    /// Everything the remote run detail renders, assembled from ONE bounded
    /// fetch pass: previews and the semantic model share each file's bytes
    /// (a tunnel must never pay for the same head twice).
    public struct RemoteRunDetailPayload: Sendable {
        public var previewed: [RemoteRunFilePreviewItem] = []
        public var other: [RemoteRunFileEntry] = []
        public var model: RunResults.Model?

        public init() {}
    }

    /// Byte cap for the remote report.json / manifest-snapshot / validation
    /// report head fetch — orders of magnitude above real reports, still
    /// BOUNDED (F5: the unbounded run-file route is never used here, even
    /// when the listed size is unknown). The server's own head ceiling
    /// (`RUN_FILE_HEAD_CAP`, 8 MiB) sits ABOVE this by contract, so a
    /// report under this cap arrives complete; a genuinely over-bound file
    /// arrives truncated WITH the server's file-size/truncated metadata
    /// headers and degrades to head-derived tables with the truncation
    /// caption — never to silently biased numbers presented as complete.
    public static let remoteReportByteLimit = 4_194_304

    /// Head-fetch caps for the files the semantic model reads. Superset of
    /// the preview needs for the same names, so ONE fetch serves both.
    static let remoteSemanticCaps: [String: Int] = [
        "generations.jsonl": RunBrowser.jsonPreviewByteLimit,
        "report.json": remoteReportByteLimit,
        "experiment.json": remoteReportByteLimit,
        "validation-report.json": remoteReportByteLimit,
        "promoted-movers.json": RunBrowser.jsonPreviewByteLimit,
        "summaries.csv": RunBrowser.jsonPreviewByteLimit,
        "effect-sizes.csv": RunBrowser.jsonPreviewByteLimit,
        "alien-residuals.csv": RunBrowser.jsonPreviewByteLimit,
        "cosine-matrix.csv": RunBrowser.jsonPreviewByteLimit,
        "panel-effects.csv": RunBrowser.jsonPreviewByteLimit,
    ]

    /// Fetch one remote run's detail: every listed file exactly ONCE,
    /// head-bounded (the `head=` param; JSON size-gated from the LISTED
    /// size before any fetch), concurrently across files. The same bytes
    /// feed the bounded previews AND the semantic `RunResults` model, whose
    /// parse runs off the main actor like the local path. Truncation is
    /// judged from the server's head metadata (actual file size + truncated
    /// flag) when present, falling back to the listed-size heuristic against
    /// older servers. Fetch failures degrade that file's preview AND surface
    /// in `remoteResultsStatus` — never a bare nil (and a PRIOR error there
    /// never outlives a successful load). The server remains the source of
    /// truth; nothing is written locally until the researcher chooses
    /// Import Evidence.
    ///
    /// `fetcher` is injectable for tests; nil uses the connected client.
    public func loadRemoteRunDetail(
        run: RemoteStampedRunRecord,
        fetcher: (@Sendable (_ name: String, _ maxBytes: Int) async throws -> RemoteRunFileHead)? =
            nil
    ) async -> RemoteRunDetailPayload {
        // A stale status from a previous run's failed load must not caption
        // THIS load — it re-appears below only if this load itself fails.
        remoteResultsStatus = nil
        var payload = RemoteRunDetailPayload()
        let fetch: @Sendable (String, Int) async throws -> RemoteRunFileHead
        if let fetcher {
            fetch = fetcher
        } else {
            loadStoredRemoteToken()
            guard let client = remoteClient else {
                payload.other = run.previewFileEntries
                remoteResultsStatus = "invalid server URL"
                return payload
            }
            let runID = run.id
            fetch = { name, maxBytes in
                try await client.runFileHead(
                    runID: runID, name: name, maxBytes: maxBytes)
            }
        }

        // One bounded fetch per listed file, concurrent across files. The
        // request size is the semantic cap for model-feeding files (already
        // ≥ the preview parser's need for that type), else the preview
        // plan's bytes; files with no plan and no semantic role move no
        // bytes at all.
        let entries = run.previewFileEntries
        var fetched: [String: (head: RemoteRunFileHead, requested: Int)] = [:]
        var failures: [String: String] = [:]
        await withTaskGroup(
            of: (name: String, requested: Int, result: Result<RemoteRunFileHead, any Error>).self
        ) { group in
            for file in entries {
                let requested: Int
                if let semanticCap = Self.remoteSemanticCaps[file.name] {
                    requested = semanticCap
                } else if let planBytes = RunBrowser.remoteFetchPlan(
                    name: file.name, size: file.size).requestBytes
                {
                    requested = planBytes
                } else {
                    continue
                }
                group.addTask {
                    do {
                        return (
                            file.name, requested,
                            .success(try await fetch(file.name, requested))
                        )
                    } catch {
                        return (file.name, requested, .failure(error))
                    }
                }
            }
            for await outcome in group {
                switch outcome.result {
                case .success(let head):
                    fetched[outcome.name] = (head, outcome.requested)
                case .failure(let error):
                    failures[outcome.name] = "\(error)"
                }
            }
        }

        // Previews, in listing order, from the shared bytes (pure parsers,
        // identical caps to local browsing).
        var artifacts = RunResults.ArtifactBytes()
        for file in entries {
            let preview: RunBrowser.FilePreview
            if let (fetchedHead, requested) = fetched[file.name] {
                // The server's actual file size (when stamped) supersedes the
                // listing for truncation captions — 0 there means "unknown".
                preview = RunBrowser.remotePreview(
                    name: file.name,
                    size: fetchedHead.fileSize ?? file.size,
                    data: fetchedHead.data)
                if RunResults.ArtifactBytes.fileNames.contains(file.name) {
                    let head = RunBrowser.remoteHead(
                        data: fetchedHead.data, listedSize: file.size,
                        requestedBytes: requested,
                        serverFileSize: fetchedHead.fileSize,
                        serverTruncated: fetchedHead.truncated)
                    artifacts.assign(
                        name: file.name, data: head.data,
                        truncated: head.truncated)
                }
            } else if let failure = failures[file.name] {
                preview = .unavailable(reason: "fetch failed: \(failure)")
            } else if case .none(let reason) = RunBrowser.remoteFetchPlan(
                name: file.name, size: file.size)
            {
                preview = .unavailable(reason: reason)
            } else {
                preview = .unavailable(reason: "no preview")
            }
            if case .unavailable = preview {
                payload.other.append(file)
            } else {
                payload.previewed.append(
                    RemoteRunFilePreviewItem(file: file, preview: preview))
            }
        }

        // Semantic model: pure parsing off the main actor (mirrors the
        // local detail's Task.detached load).
        if !artifacts.isEmpty {
            let runID = run.id
            let built = artifacts
            payload.model = await Task.detached(priority: .userInitiated) {
                RunResults.remoteModel(runID: runID, artifacts: built)
            }.value
        }

        // Surface fetch failures through the existing status line — a
        // missing semantic section must be attributable, never a bare nil.
        if !failures.isEmpty {
            let names = failures.keys.sorted()
            let shown = names.prefix(3).map {
                "\($0) (\(failures[$0] ?? "error"))"
            }
            remoteResultsStatus = "run \(run.id): could not fetch "
                + shown.joined(separator: "; ")
                + (names.count > 3 ? " — and \(names.count - 3) more" : "")
        }
        return payload
    }

    /// The evidence-bundle file inside a server run directory, when the run
    /// carries one (`<run_id>.evidence-bundle.tar.gz`, written by the
    /// server's `package_evidence`). Its presence is what lets the remote
    /// detail header offer a direct Import Evidence — the existing, only way
    /// to make a remote run durable in this workspace.
    public nonisolated static func evidenceBundleFileName(in files: [String]) -> String? {
        files.first { $0.hasSuffix(".evidence-bundle.tar.gz") }
    }

    /// Import the evidence bundle contained in a SERVER run directory:
    /// download it (path-contained server-side), verify + extract through
    /// the same `EvidenceBundleImporter` as the job-based import, and land
    /// it under this workspace's `runs/` as an immutable imported run.
    public func importEvidence(fromServerRun run: RemoteStampedRunRecord) async {
        loadStoredRemoteToken()
        guard let remoteClient else {
            remoteResultsStatus = "invalid server URL"
            return
        }
        guard let bundleName = Self.evidenceBundleFileName(in: run.files) else {
            remoteResultsStatus = "run \(run.id) carries no evidence bundle — "
                + "import from the producing job instead (Compute section)"
            return
        }
        do {
            remoteResultsStatus = "downloading evidence from \(run.id)..."
            let downloads = VectorCatalog.projectRoot
                .appending(components: ".steerlab", "downloads")
            let bundlePath = run.path.hasSuffix("/")
                ? run.path + bundleName : run.path + "/" + bundleName
            let localBundle = try await remoteClient.downloadArtifact(
                path: bundlePath, to: downloads)
            let imported = try await Task.detached {
                try EvidenceBundleImporter.importEvidenceBundle(localBundle)
            }.value
            remoteImportedRunDirectory = imported.path
            let importedMessage = "evidence from \(run.id) imported → "
                + "runs/\(imported.lastPathComponent) (hashes verified)"
            remoteResultsStatus = importedMessage
            note(importedMessage, severity: .success)
            noteEvidenceRevisionAdoption(forImportedRun: imported)
            refreshResults(selecting: imported.lastPathComponent)
            // An imported chain appears under Imported / local immediately
            // — the round trip is visible without a manual refresh.
            await refreshPipelineRuns()
        } catch {
            remoteResultsStatus = "evidence import failed: \(error)"
        }
    }

    /// After a VERIFIED evidence import: complete the researcher's declared
    /// intent by adopting the evidence snapshot's model revision into a
    /// still-unpinned local draft, or flag a conflicting pin loudly
    /// (external review 2026-07-22 — a server-resolved revision otherwise
    /// leaves local freeze readiness blocked on "revision not pinned").
    /// Decision + save live in `EvidenceRevisionAdoption` (unit-tested);
    /// this is the notice glue.
    public func noteEvidenceRevisionAdoption(forImportedRun imported: URL) {
        let outcome = EvidenceRevisionAdoption.adoptModelRevision(
            fromImportedRun: imported)
        guard let notice = EvidenceRevisionAdoption.notice(for: outcome) else {
            return
        }
        note(notice.message, severity: notice.isWarning ? .warning : .success)
        if case .adopted = outcome {
            // The draft changed on disk — reload so the editor and freeze
            // readiness see the pinned revision immediately.
            refresh()
        }
    }

    // MARK: Optimizations on the active server workspace

    /// Server experiments that qualify as OPTIMIZATIONS — a condition carrying
    /// sweep-selection provenance, or a sweep run on the server's runs/ tree.
    /// (The server's experiment detail does not expose the declared sweep
    /// spec, so a run-directory match stands in for the local lens's
    /// `manifest.sweep != nil` arm.) Cleared when no server workspace is
    /// active.
    public private(set) var remoteOptimizations: [RemoteExperimentRecord] = []

    /// Refresh the server-workspace Optimizations lens: fetch experiment summaries
    /// (with verbatim per-condition selection blocks) plus the run listing,
    /// and keep the experiments that screen.
    public func refreshRemoteOptimizations() async {
        guard isServerWorkspace else {
            remoteOptimizations = []
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            remoteOptimizations = []
            note("invalid server URL", severity: .error)
            return
        }
        do {
            async let summaries = client.experimentSummaries()
            async let runList = client.runs()
            let (experiments, runRecords) = try await (summaries, runList)
            remoteRuns = runRecords
            remoteOptimizations = experiments.filter { record in
                record.conditions?.contains { $0.selection != nil } == true
                    || SweepRunCatalog.newestRemoteSweepRunRecord(
                        experiment: record.name, in: runRecords) != nil
            }
        } catch {
            remoteOptimizations = []
            let substrate = cluster?.substrateLabel ?? "server"
            note("could not list optimizations on \(substrate): \(error)", severity: .error)
        }
    }

    /// Newest sweep run for a SERVER experiment: match the run id with the
    /// same naming rule as local discovery, fetch `sweep.csv` +
    /// `recommendations.json` over the run-file route, and parse with the
    /// same entry points. Returns nil (with a status line) when absent or
    /// unreadable — the grid renders its empty state either way.
    public func loadRemoteSweepRun(experiment: String) async -> SweepRunCatalog.SweepRun? {
        guard isServerWorkspace else { return nil }
        loadStoredRemoteToken()
        guard let client = remoteClient else { return nil }
        do {
            if remoteRuns.isEmpty {
                remoteRuns = try await client.runs()
            }
            guard
                let record = SweepRunCatalog.newestRemoteSweepRunRecord(
                    experiment: experiment, in: remoteRuns)
            else { return nil }
            let csv = try await client.runFile(runID: record.id, name: "sweep.csv")
            var recommendations: Data?
            if record.files.contains("recommendations.json") {
                recommendations = try await client.runFile(
                    runID: record.id, name: "recommendations.json")
            }
            return try SweepRunCatalog.remoteSweepRun(
                runPath: record.path,
                csvText: String(decoding: csv, as: UTF8.self),
                recommendationsData: recommendations)
        } catch {
            let substrate = cluster?.substrateLabel ?? "server"
            note("could not load sweep run for '\(experiment)' from \(substrate): \(error)", severity: .error)
            return nil
        }
    }

    /// Promote through the ACTIVE server (`POST /api/experiment/{name}/promote`):
    /// the study, its sweep provenance, and the minted agent all live in the
    /// SERVER's workspace — artifacts never cross substrates. Preflights
    /// server residency like `runStudyOnActiveServer`; a server refusal (400
    /// detail) surfaces verbatim in the status line.
    public func promoteOnActiveServer(
        experimentName: String,
        concept: String,
        cell: (layer: Int, alpha: Double)? = nil,
        overrideReason: String? = nil,
        pins: AgentPromotion.Pins? = nil
    ) async {
        guard isServerWorkspace else {
            note("no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note("invalid server URL", severity: .error)
            return
        }
        let substrate = cluster?.substrateLabel ?? "server"
        do {
            if let names = try? await client.experimentNames(),
                !names.contains(experimentName)
            {
                note("study '\(experimentName)' is not in \(substrate)'s workspace — "
                    + "promote mints from the server-resident copy only. Pair "
                    + "the server to this workspace (serve --root <workspace>) "
                    + "or use Submit Bundle, the portable path for remote engines", severity: .info)
                return
            }
            note("promoting '\(concept)' from '\(experimentName)' on \(substrate)…", severity: .info)
            let minted = try await client.promoteExperiment(
                name: experimentName, concept: concept,
                cell: cell, overrideReason: overrideReason, pins: pins)
            let stamp = cell == nil
                ? "it carries the sweep-selection birth certificate"
                : "stamped promotedBy: manualOverride (declared selection bypassed)"
            note("promoted '\(concept)' → agent '\(minted.variant.name)' "
                + "minted in \(substrate)'s workspace (\(stamp)) — "
                + "see the Agents section's server list", severity: .success)
            await cluster?.refreshRemoteVariants()
            // The minted agent references the sweep run's persisted vectors —
            // refresh the vector catalog too so applying it in Playground
            // resolves those refs immediately.
            await host?.catalog.refreshRemoteVectors()
        } catch {
            note("server promote failed: \(error)", severity: .error)
        }
    }

    public func cancelRemoteJob() async {
        guard let remoteClient, let remoteJobID else { return }
        do {
            try await remoteClient.cancelJob(remoteJobID)
            remoteStatus = "cancel requested for \(remoteJobID)"
        } catch {
            remoteStatus = "remote cancel failed: \(error)"
        }
    }

    public func downloadRemoteEvidence() async {
        guard let remoteJobID else {
            remoteStatus = "no remote job selected"
            return
        }
        await importEvidence(fromJobID: remoteJobID)
    }

    /// Manual resume of a checkpointed server job — the Resume button
    /// (2026-07-22 incident: a run checkpointed cleanly at the walltime
    /// margin and the app offered no way to continue it). The server
    /// re-sbatches the job's OWN `run.sbatch`; the run continues from its
    /// checkpoint. The follower needs no re-attach: it follows the JOB
    /// RECORD id (non-terminal through "checkpointed"), and the
    /// continuation's child record folds its completion back onto that same
    /// record.
    public func resubmitRemoteJob(_ id: String) async {
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note(
                "remote resume refused: no server connection — connect a "
                    + "server in the substrate selector first",
                severity: .error)
            return
        }
        do {
            let result = try await client.resubmitJob(id)
            let line = RemoteJobStatusClass.resumedStatusLine(
                jobID: id, slurmJobID: result.slurmJobID,
                continuationJobID: result.jobId)
            remoteStatus = line
            note(line, severity: .success)
            await refreshRecentServerJobs()
        } catch let error as ClusterClient.ClientError {
            // 409 details are the server's own plain-language refusal
            // (already resubmitted / still running / cancelled) — show its
            // words verbatim.
            let detail = ClusterClient.unwrappingDetail(error)
            remoteStatus = "resume failed: \(detail)"
            note("Couldn't resume job \(id) — \(detail)", severity: .error)
        } catch {
            note(
                "Couldn't resume job \(id) — \(error.localizedDescription)",
                severity: .error)
        }
    }

    /// Which job rows should offer the Import Evidence action: completed
    /// run-verb jobs (direct `run`, bundle `run (bundle)`, or a server-listed
    /// `study-submit`) plus bundled `validate` jobs — the kinds whose result
    /// can carry an evidence bundle (`execute_run_bundle` packages evidence
    /// for any verb that produces a run directory; validate bundles are the
    /// Mac-authority freeze-evidence path, 2026-07-21). Dry runs finish
    /// `prepared`, never `succeeded`, so they never qualify.
    public static func jobOffersEvidenceImport(verb: String, state: String) -> Bool {
        guard state == "succeeded" else { return false }
        return verb == "run" || verb == "run (bundle)" || verb == "study-submit"
            || verb == "validate (bundle)"
    }

    /// Kind-based twin for the Compute panel's job rows, which carry the
    /// server's raw `RemoteJobRecord.kind` strings rather than this panel's
    /// verb labels: direct run verbs list as `experiment:run`, study
    /// submissions as `study-submit` (direct) / `study-submit-bundle`
    /// (bundle upload). Same semantics — completed, run-verb,
    /// evidence-capable; dry runs finish `prepared` and never qualify, and
    /// non-run verbs (`experiment:sweep`, …) never carry an evidence bundle.
    public static func jobOffersEvidenceImport(kind: String, state: String) -> Bool {
        guard state == "succeeded" else { return false }
        // `pipeline-orphan-reconcile` (2026-08-06) is the daemon's startup
        // resume of an orphaned chain: on success it packages the chain's
        // evidence exactly like a bundle execution would have — the healed
        // results must flow home through the same auto-import, or the
        // incident is only half-fixed.
        return kind == "experiment:run" || kind == "study-submit"
            || kind == "study-submit-bundle"
            || kind == "pipeline-orphan-reconcile"
    }

    /// Which job rows should offer the *diagnostic* retrieval action:
    /// jobs that did NOT succeed but whose server-side failure path
    /// packaged whatever the run had produced (retention 2026-07-24).
    ///
    /// Deliberately a SEPARATE predicate from `jobOffersEvidenceImport`
    /// rather than a loosened state check on it. The two answer different
    /// questions — "are there results to import?" versus "is there a
    /// failure record to retrieve?" — and every evidence-grade gate in the
    /// app is wired to the first. Widening it would have quietly made
    /// partial evidence citable, which is the one thing partial evidence
    /// must never become. Any verb qualifies here: a failed sweep's
    /// diagnostics are as worth retrieving as a failed run's.
    public static func jobOffersPartialEvidenceImport(
        _ job: RemoteJobRecord
    ) -> Bool {
        guard job.status != "succeeded" else { return false }
        return job.partialEvidenceBundlePath != nil
    }

    /// Import the evidence bundle produced by ANY completed server job (the
    /// recent-jobs rows pass their own id; the Run-on-Server disclosure's
    /// button passes the panel's last job). Downloads the bundle, verifies
    /// the server-stamped SHA-256, and lands it under this workspace's
    /// `runs/` as an immutable imported run — the status line says exactly
    /// where, and the study result list refreshes so it appears immediately.
    public func importEvidence(fromJobID jobID: String) async {
        loadStoredRemoteToken()
        guard let remoteClient else {
            remoteStatus = "invalid server URL"
            return
        }
        do {
            let job = try await remoteClient.job(jobID)
            guard let result = job.result,
                let bundlePath = Self.findString(
                    in: .object(result), keyPath: ["runResult", "evidenceBundle", "bundlePath"])
                    ?? Self.findString(in: .object(result), keyPath: ["evidenceBundle", "bundlePath"])
            else {
                let pendingMessage = "job \(jobID) has no evidence bundle yet"
                remoteStatus = pendingMessage
                note(pendingMessage, severity: .info)
                return
            }
            // Cross-check the download against the server-stamped bundle hash when present.
            let expectedSHA = Self.findString(
                in: .object(result), keyPath: ["runResult", "evidenceBundle", "bundleSha256"])
                ?? Self.findString(in: .object(result), keyPath: ["evidenceBundle", "bundleSha256"])
            remoteStatus = "downloading evidence..."
            let downloads = VectorCatalog.projectRoot.appending(components: ".steerlab", "downloads")
            let localBundle = try await remoteClient.downloadArtifact(path: bundlePath, to: downloads)
            // Extraction + per-file hashing off the main actor.
            let imported = try await Task.detached {
                try EvidenceBundleImporter.importEvidenceBundle(localBundle, expectedSHA256: expectedSHA)
            }.value
            remoteImportedRunDirectory = imported.path
            let importedMessage = "evidence from job \(jobID) imported → "
                + "runs/\(imported.lastPathComponent) (hashes verified)"
            remoteStatus = importedMessage
            note(importedMessage, severity: .success)
            noteEvidenceRevisionAdoption(forImportedRun: imported)
            refreshResults(selecting: imported.lastPathComponent)
            // An imported chain appears under Imported / local immediately
            // — the round trip is visible without a manual refresh.
            await refreshPipelineRuns()
        } catch {
            let failureMessage = "evidence import failed: \(error)"
            remoteStatus = failureMessage
            note(failureMessage, severity: .error)
        }
    }

    private static func findString(in value: JSONValue, keyPath: [String]) -> String? {
        guard let first = keyPath.first else {
            if case .string(let value) = value { return value }
            return nil
        }
        guard case .object(let object) = value, let child = object[first] else { return nil }
        return findString(in: child, keyPath: Array(keyPath.dropFirst()))
    }

    /// The ONE place a draft's base model may change (open-issues §8,
    /// residual (b)). A model change invalidates every cast agent — an
    /// adapter or steering vector built on one model is not eligible for a
    /// seat (or an arm) running another — so it drops the revision pin and
    /// every variant condition; multi-agent seats fall back to baseline in
    /// `saveProtocol` for the same reason.
    ///
    /// An empty `studyBaseModelID` is "no choice made", never a request to
    /// change the model to the empty string: the field is a PANEL cache
    /// synced on selection, and a caller that never synced it (a headless
    /// route, a fresh panel) must not be able to clear a study's arms by
    /// omission. Returns whether the model actually changed.
    @discardableResult
    private func applyStudyBaseModelChoice(
        to manifest: inout ExperimentManifest
    ) -> Bool {
        let requested = studyBaseModelID.trimmingCharacters(
            in: .whitespacesAndNewlines)
        guard !requested.isEmpty, requested != manifest.modelID else {
            return false
        }
        manifest.modelID = requested
        manifest.modelRevision = nil
        manifest.variantConditions.removeAll()
        return true
    }

    /// Re-reads the panel's base-model choice from the selected manifest.
    /// Headless routes call this before a protocol save whose request names
    /// no model: such a save must never act as a base-model change, and
    /// without the resync it compares the manifest against whatever the
    /// panel last synced — which can silently drop every variant condition
    /// (open-issues §8, residual (b)).
    public func adoptSelectedManifestBaseModel() {
        if let modelID = selected?.modelID { studyBaseModelID = modelID }
    }

    public func saveProtocol() {
        guard var manifest = selected, manifest.status == .draft else { return }
        do {
            manifest.experimentDescription = protocolDescription
            manifest.taskDescription = nilIfEmpty(taskDescription)
            manifest.outcomeMeasures = nilIfEmpty(outcomeMeasures)
            manifest.studyKind = studyKind
            let baseModelChanged = applyStudyBaseModelChoice(to: &manifest)
            manifest.promptMode = promptMode
            manifest.systemPrompt = nilIfEmpty(systemPrompt)
            manifest.qwenThinkingEnabled = qwenThinkingEnabled
            manifest.dtype = nilIfEmpty(studyDtypeField)
            // Judge-rubric versioning: pin the selected rubric file at its
            // CURRENT hash ("" clears the pin — draft-only inline text).
            let rubricFile = judgeRubricFile.trimmingCharacters(in: .whitespacesAndNewlines)
            if rubricFile.isEmpty {
                manifest.judgeRubricFile = nil
                manifest.judgeRubricHash = nil
            } else {
                try JudgeRubricStore.pin(rubricFile, into: &manifest)
            }
            let panelJudges = judges
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
            manifest.evaluation = Self.evaluationDeclaration(
                judges: panelJudges,
                rubricFile: rubricFile,
                inlineRubric: evaluationPrompt,
                structuredPrompt: nilIfEmpty(evaluationStructuredPrompt),
                inlineJudgeModel: resolvedInlineJudgeModel())
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
            manifest.temperature = runTemperature
            manifest.maxTokens = runMaxTokens
            if studyKind == .multiAgent {
                let selection = selectedMultiAgentScenarioID.flatMap { id in
                    multiAgentScenarioOptions.first { $0.id == id }
                }
                let casting = SeatCasting.state(
                    of: manifest,
                    selected: selection.map {
                        ($0.scenario, relativeProjectPath(for: $0.url))
                    },
                    overlay: seatCastingEdits)
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
                            note(
                                "the base model changed, so every seat was reset "
                                    + "to baseline — agents built on the previous "
                                    + "model cannot run in this panel. Recast the "
                                    + "seats below.",
                                severity: .warning)
                        }
                        try SeatCasting.compile(
                            assignment, semantic: casting.semantic,
                            semanticPath: casting.semanticPath, into: &manifest)
                        seatCastingEdits = [:]
                    }
                case .legacyBound:
                    // A hand-bound scenario is pinned as it stands — its casting
                    // is inside the file, and this save must not rewrite it.
                    guard let scenario = selection else { break }
                    manifest.multiAgentScenarioPath =
                        relativeProjectPath(for: scenario.url)
                    manifest.multiAgentScenarioHash =
                        try MultiAgentScenarioStore.hash(scenario.url)
                    manifest.multiAgentSemanticScenarioPath = nil
                    manifest.multiAgentSemanticScenarioHash = nil
                case nil:
                    note("select a scenario first", severity: .info)
                    return
                }
                manifest.multiAgentIncludeBaseline = multiAgentIncludeBaseline
            } else if !taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                _ = try ExperimentStore.pinTaskPrompts(taskPromptsFile, into: &manifest)
            } else {
                // An EMPTIED prompts field on a model-output study is the
                // one explicit clear this save performs.
                manifest.taskPromptsFile = nil
                manifest.taskPromptsHash = nil
            }
            try ExperimentStore.save(manifest)
            // Formerly "Science Manifest" fields (that section is
            // dissolved), saved through the same store setters from their
            // new homes: funnel phase + sampling policy (Study Setup),
            // case family + option-length acknowledgment (Evaluation).
            // After save(manifest): these load-modify-save by name.
            try ExperimentStore.setPhase(
                nilIfEmpty(phaseField), experimentName: manifest.name)
            try ExperimentStore.setCaseFamily(
                nilIfEmpty(caseFamilyField), experimentName: manifest.name)
            try ExperimentStore.setSamplingPolicy(
                samplesPerItem: samplesPerItemField <= 1 ? nil : samplesPerItemField,
                seedPolicy: nilIfEmpty(seedPolicyField),
                experimentName: manifest.name)
            try ExperimentStore.setAcknowledgeUnequalOptionLengths(
                acknowledgeUnequalOptionLengthsField, experimentName: manifest.name)
            refresh()
            note("saved protocol notes and run defaults", severity: .success)
        } catch {
            note(
                "Couldn't save the study setup — check the study is still a "
                    + "draft and its file is writable, then save again. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    // MARK: Science-manifest editor (App gap A2) + instrument activation (P1)

    /// The current editor's outcome mode, read straight from the manifest
    /// (single source of truth — the picker writes through
    /// `setOutcomeMode`, never a shadow field).
    public var outcomeMode: InstrumentActivation.OutcomeMode {
        InstrumentActivation.OutcomeMode.from(selected?.outcomeInstruments)
    }

    /// Items in the pinned prompt set that carry categorical `options` —
    /// what the DATA supports, independent of what is enabled.
    public var detectedOptionsItemCount: Int {
        taskPromptsDocument?.optionsItemCount ?? 0
    }

    public var detectedCapabilitiesLine: String? {
        InstrumentActivation.detectedCapabilitiesLine(
            optionsItemCount: detectedOptionsItemCount)
    }

    /// Option-carrying items whose declared `responseFormat` the
    /// answer-token instruments cannot read.
    public var detectedUnscorableOptionItemCount: Int {
        taskPromptsDocument?.unscorableOptionItemCount ?? 0
    }

    /// The prominent pre-run warning (P1): options present, no categorical
    /// instrument declared. nil = nothing to warn about.
    public var instrumentActivationWarning: String? {
        InstrumentActivation.activationWarning(
            optionsItemCount: detectedOptionsItemCount,
            instruments: selected?.outcomeInstruments,
            unscorableOptionItemCount: detectedUnscorableOptionItemCount)
    }

    /// Writes the outcome-mode choice into `manifest.outcomeInstruments`
    /// through the store (draft-only; never auto-enabled — the method
    /// belongs in provenance).
    public func setOutcomeMode(_ mode: InstrumentActivation.OutcomeMode) {
        guard let name = selectedName else { return }
        guard mode != .notDeclared else {
            // "not declared" is the read-back of an ABSENT declaration; the
            // picker never clears an explicit one silently.
            return
        }
        do {
            let instruments = InstrumentActivation.applying(
                mode, to: selected?.outcomeInstruments)
            try ExperimentStore.setOutcomeInstruments(
                instruments, experimentName: name)
            refresh()
            note("declared outcome instruments: "
                + (instruments?.joined(separator: ", ") ?? "none"), severity: .success)
        } catch {
            note(
                "Couldn't declare the outcome mode — the study must still be "
                    + "a draft (frozen studies are read-only). Details: \(error)",
                severity: .error)
        }
    }

    /// Enabled auxiliary instruments (F3) — declared ids the Outcome Mode
    /// picker does not own (today: `repeReaderScore`), rendered as their
    /// own Evaluation rows with the sampling implication stated.
    public var auxiliaryOutcomeInstruments: [String] {
        InstrumentActivation.auxiliaryInstruments(of: selected?.outcomeInstruments)
    }

    /// The effective-record-kinds note for the pre-run warning area (F3):
    /// non-nil exactly when the mode reads answer-token-only but a declared
    /// auxiliary forces sampled generation anyway.
    public var effectiveRecordKindsNote: String? {
        InstrumentActivation.effectiveRecordKindsNote(
            instruments: selected?.outcomeInstruments)
    }

    /// Remove one auxiliary instrument from `outcomeInstruments` (F3) —
    /// draft-only by the store's gate, written through
    /// `setOutcomeInstruments` like every other instrument edit. Removing
    /// the reader is what makes a genuinely logprob-only run possible.
    public func removeAuxiliaryInstrument(_ id: String) {
        guard let name = selectedName else { return }
        do {
            let remaining = (selected?.outcomeInstruments ?? []).filter { $0 != id }
            try ExperimentStore.setOutcomeInstruments(
                remaining.isEmpty ? nil : remaining, experimentName: name)
            refresh()
            note(
                "removed auxiliary instrument \(id) — outcome instruments: "
                    + (remaining.isEmpty
                        ? "not declared (engine default, sampled text)"
                        : remaining.joined(separator: ", ")),
                severity: .success)
        } catch {
            note("\(error)", severity: .error)
        }
    }

    /// The reader instrument can be re-added only when the manifest
    /// actually pins reader artifacts (`readerRefs`) — declaring
    /// `repeReaderScore` with no pinned reader is an immediate verify
    /// violation, so the affordance hides instead of inviting one.
    public var canAddReaderInstrument: Bool {
        guard let manifest = selected, manifest.status == .draft else { return false }
        return !(manifest.readerRefs ?? []).isEmpty
            && !(manifest.outcomeInstruments ?? []).contains("repeReaderScore")
    }

    /// Add `repeReaderScore` back alongside the current mode (F3) —
    /// draft-only, through the store.
    public func addReaderInstrument() {
        guard let name = selectedName, canAddReaderInstrument else { return }
        do {
            let instruments = (selected?.outcomeInstruments ?? []) + ["repeReaderScore"]
            try ExperimentStore.setOutcomeInstruments(
                instruments, experimentName: name)
            refresh()
            note(
                "added reader instrument repeReaderScore — sampled generation "
                    + "will run and each response is scored by the pinned readers",
                severity: .success)
        } catch {
            note("\(error)", severity: .error)
        }
    }

    /// Save every science-manifest field through the store setters (A2).
    /// Draft-only by the store's gate; no view-side JSON anywhere.
    /// Save the promotion rule — the screen→confirm gate (FDR threshold on
    /// screening q-values, dose monotonicity, matched-norm random floor,
    /// capability gate) declared in the Pipeline section for concept
    /// studies. (The former Science Manifest's other fields save through
    /// `saveProtocol` from their new homes.)
    public func savePromotionRule() {
        guard let name = selectedName else { return }
        let fdrText = promotionFDRText.trimmingCharacters(in: .whitespaces)
        if !fdrText.isEmpty, Double(fdrText) == nil {
            note("promotion rule not saved: FDR threshold must be a number in (0, 1)", severity: .error)
            return
        }
        do {
            try ExperimentStore.setPromotionRule(
                ExperimentManifest.PromotionRule(
                    fdrThreshold: Double(fdrText),
                    doseMonotone: promotionDoseMonotone ? true : nil,
                    exceedsRandomFloor: promotionExceedsRandomFloor ? true : nil,
                    capabilityGate: nilIfEmpty(promotionCapabilityGateText)),
                experimentName: name)
            refresh()
            note("saved promotion rule (screen→confirm gate)", severity: .success)
        } catch {
            note(
                "Couldn't save the promotion rule — the study must still be a "
                    + "draft and its file writable. Details: \(error)",
                severity: .error)
        }
    }

    /// Unpin the human baseline (explicit action from Data & Prompts — the
    /// old path of emptying a text field and pressing a distant save was
    /// too easy to do by accident).
    public func clearHumanBaseline() {
        guard let name = selectedName else { return }
        do {
            try ExperimentStore.clearHumanBaseline(experimentName: name)
            humanBaselinePathField = ""
            refresh()
            note("human baseline unpinned", severity: .success)
        } catch {
            note(
                "Couldn't unpin the human baseline — the study must still be "
                    + "a draft. Details: \(error)",
                severity: .error)
        }
    }

    /// Re-pin the human baseline at its CURRENT bytes (explicit action —
    /// drift stays a visible finding otherwise).
    public func repinHumanBaseline() {
        guard let name = selectedName else { return }
        do {
            let pinned = try ExperimentStore.pinHumanBaseline(
                path: humanBaselinePathField, experimentName: name)
            refresh()
            note("pinned human baseline \(pinned.path) @ \(pinned.hash.prefix(12))…", severity: .success)
        } catch {
            note(
                "Couldn't pin the human baseline — check the path points at "
                    + "an existing CSV inside the workspace (e.g. "
                    + "prompts/baselines/…). Details: \(error)",
                severity: .error)
        }
    }

    // MARK: Native condition editor (App gap A4)

    /// Concepts a vector condition may reference: the draft's attached
    /// concepts (conditions must reference pinned concepts).
    public var conditionConceptOptions: [String] {
        selected?.concepts.map(\.name) ?? []
    }

    /// Add a single-slot vector condition from the editor row's fields —
    /// negative α is legal (a one-field direction control).
    public func addVectorCondition() {
        guard let name = selectedName else { return }
        let concept = conditionConcept.trimmingCharacters(in: .whitespaces)
        guard !concept.isEmpty else {
            refuse(.addCondition, "pick a concept for the condition", severity: .info)
            return
        }
        let isAblation = conditionMode == .ablate
        // Ablation does not take a layer: it covers the whole network, and the
        // form hides the field. Parse it only when steering, so a stale value
        // left in the box cannot refuse an ablation that never needed it.
        var layer = 0
        if !isAblation {
            guard let parsed = Int(
                conditionLayerText.trimmingCharacters(in: .whitespaces)),
                parsed >= 0
            else {
                refuse(.addCondition, "condition layer must be a non-negative integer")
                return
            }
            layer = parsed
        }
        guard let alpha = Double(conditionAlphaText.trimmingCharacters(in: .whitespaces)),
            alpha.isFinite, alpha != 0
        else {
            refuse(
                .addCondition,
                isAblation
                    ? "ablation strength λ must be a nonzero number — 1 removes "
                        + "the concept completely, 0.5 removes half of it, and 2 "
                        + "reflects it (flips the concept while keeping the "
                        + "residual stream's length). λ = 0 would be a condition "
                        + "that does nothing, which the baseline already covers"
                    : "condition α must be a nonzero number (negative = direction control)")
            return
        }
        let conditionTitle = conditionName.isEmpty
            ? (isAblation
                ? "\(concept)-ablate-l\(SweepSpecForm.numberListText([alpha]))"
                : "\(concept)-L\(layer)-a\(SweepSpecForm.numberListText([alpha]))")
            : conditionName
        do {
            try ExperimentStore.upsertCondition(
                .init(
                    name: conditionTitle,
                    slots: [
                        .init(
                            concept: concept, layer: layer, alpha: alpha,
                            mode: isAblation ? .ablate : nil)
                    ],
                    bandWidth: 1,
                    // λ is never in residual-norm units; recording the flag as
                    // true would claim a conversion the run loop does not do.
                    alphaInNormUnits: isAblation ? false : conditionAlphaInNormUnits),
                experimentName: name)
            conditionName = ""
            refresh()
            clearFormError(.addCondition)
            note(
                isAblation
                    ? "added condition '\(conditionTitle)' — ablates \(concept) "
                        + "at λ\(alpha) across every layer"
                    : "added condition '\(conditionTitle)' (\(concept) L\(layer) "
                        + "α\(alpha)\(conditionAlphaInNormUnits ? " norm-units" : ""))",
                severity: .success)
        } catch {
            refuse(
                .addCondition,
                "Couldn't add the condition — the study may be frozen, or "
                    + "the referenced concept is no longer pinned. "
                    + "Details: \(error)")
        }
    }

    // MARK: Validation controls + instrument scope (authoring)

    /// Concepts on disk that are neither study concepts nor already declared
    /// controls — the candidates for the picker.
    public var validationControlCandidates: [String] {
        let pinned = Set(selected?.concepts.map(\.name) ?? [])
        let declared = Set((selected?.validationControls ?? []).map(\.concept))
        return VectorCatalog.conceptNames()
            .filter { !pinned.contains($0) && !declared.contains($0) }
            .sorted()
    }

    /// Declared control field for the picker.
    public var controlConcept = ""
    /// The control's OWN extraction method — never inherited from a study
    /// concept, which is the fault C2 removed.
    public var controlMethod: ExtractionMethod = .meanDifference

    public func addValidationControl() {
        guard let name = selectedName else { return }
        let concept = controlConcept.trimmingCharacters(in: .whitespaces)
        guard !concept.isEmpty else {
            refuse(.validationControl, "pick a concept to use as a control",
                   severity: .info)
            return
        }
        do {
            try ExperimentStore.attachValidationControl(
                concept: concept,
                options: .init(method: controlMethod),
                experimentName: name)
            controlConcept = ""
            refresh()
            clearFormError(.validationControl)
            note("declared '\(concept)' as a discriminant control "
                + "(\(controlMethod.rawValue), stimulus hash pinned)",
                severity: .success)
        } catch {
            refuse(
                .validationControl,
                "Couldn't declare '\(concept)' as a control — the study must "
                    + "still be a draft, and the concept needs a readable "
                    + "stimulus set. Details: \(error)")
        }
    }

    public func removeValidationControl(_ concept: String) {
        guard let name = selectedName else { return }
        do {
            try ExperimentStore.removeValidationControl(
                concept: concept, experimentName: name)
            refresh()
            note("removed control '\(concept)'", severity: .info)
        } catch {
            refuse(.validationControl, "Couldn't remove the control: \(error)")
        }
    }

    /// Response formats present in the loaded task prompts, with row counts —
    /// what a scope can actually select over.
    public var availableResponseFormats: [(format: String, count: Int)] {
        guard let document = taskPromptsDocument else { return [] }
        var counts: [String: Int] = [:]
        for item in document.responseFormatItems where item.hasOptions {
            counts[item.format?.rawValue ?? "(undeclared)", default: 0] += 1
        }
        return counts.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    public func declareOutcomeInstrumentScope(_ formats: [String]) {
        guard let name = selectedName else { return }
        do {
            try ExperimentStore.declareOutcomeInstrumentScope(
                responseFormats: formats, experimentName: name)
            refresh()
            clearFormError(.validationControl)
            note(
                formats.isEmpty
                    ? "cleared the outcome-instrument scope — the instrument "
                        + "applies to every item again"
                    : "scoped the outcome instruments to \(formats.joined(separator: ", ")) "
                        + "rows (the selected row set is pinned)",
                severity: .success)
        } catch {
            refuse(
                .validationControl,
                "Couldn't declare the scope — load the study's task prompts "
                    + "first. Details: \(error)")
        }
    }

    /// One-click negative-α counterpart for an existing condition.
    public func addSignControl(for conditionNamed: String) {
        guard let name = selectedName,
            let source = selected?.conditions.first(where: { $0.name == conditionNamed })
        else { return }
        do {
            let control = ExperimentStore.signControlCondition(for: source)
            try ExperimentStore.upsertCondition(control, experimentName: name)
            refresh()
            note("added sign control '\(control.name)' (α negated — direction control)", severity: .success)
        } catch {
            note(
                "Couldn't add the sign control — the study must still be a "
                    + "draft. Details: \(error)",
                severity: .error)
        }
    }

    /// One-click matched-norm random control for an existing condition.
    public func addMatchedNormRandomControl(for conditionNamed: String) {
        guard let name = selectedName,
            let source = selected?.conditions.first(where: { $0.name == conditionNamed })
        else { return }
        do {
            let control = ExperimentStore.randomControlCondition(for: source)
            try ExperimentStore.upsertCondition(control, experimentName: name)
            refresh()
            note(
                control.controlType == "randomDirectionAblation"
                    ? "added random-direction ablation control "
                        + "'\(control.name)' — it removes a random direction "
                        + "instead of the concept's, so a difference between "
                        + "them is specific to this concept rather than to "
                        + "removing any direction"
                    : "added matched-norm random control '\(control.name)' "
                        + "(controlType: randomMatchedNorm)",
                severity: .success)
        } catch {
            note(
                "Couldn't add the random-direction control — the study must "
                    + "still be a draft. Details: \(error)",
                severity: .error)
        }
    }

    /// One-click Step-5 control-matrix scaffold; the result line names what
    /// was added, what already existed, and what stays manual.
    public func scaffoldControlMatrix() {
        guard let name = selectedName else { return }
        do {
            let result = try ExperimentStore.scaffoldControlMatrix(experimentName: name)
            refresh()
            var parts: [String] = []
            parts.append(
                result.added.isEmpty
                    ? "control matrix already complete — nothing added"
                    : "scaffolded: \(result.added.joined(separator: ", "))")
            if !result.skipped.isEmpty {
                parts.append("kept existing: \(result.skipped.joined(separator: ", "))")
            }
            clearFormError(.addCondition)
            note(parts.joined(separator: " · "), severity: .info)
            lastControlMatrixNotes = result.notes
        } catch {
            refuse(
                .addCondition,
                "Couldn't scaffold the control matrix — no conditions were "
                    + "changed; the study must still be a draft. "
                    + "Details: \(error)")
            lastControlMatrixNotes = []
        }
    }

    /// The manual-pieces notes from the last scaffold (rendered under the
    /// button so the scaffold never pretends to be the whole Step-5 matrix).
    public private(set) var lastControlMatrixNotes: [String] = []

    /// The Data Readiness "edit" affordance (2026-07-19 paper cut: the
    /// button's effect — populating the Input Data editor further down —
    /// was invisible, so it read as dead). Same load, plus a notice saying
    /// what happened and WHERE to look.
    public func loadTaskPromptsInteractively() {
        loadTaskPrompts()
        if taskPromptsDocument != nil {
            note(
                (taskPromptsStatus ?? "task prompts loaded")
                    + " — edit them in the Input Data section below",
                severity: .info)
        } else {
            note(
                "task prompts could not be loaded: "
                    + (taskPromptsStatus ?? "unknown error")
                    + " — if the file does not exist yet, use Create from "
                    + "template in Data Readiness",
                severity: .error)
        }
    }

    public func loadTaskPrompts() {
        let file = taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !file.isEmpty else {
            taskPromptsText = ""
            taskPromptsStatus = "choose a task prompts file first"
            taskPromptsInstrumentSummary = nil
            taskPromptsDocument = nil
            taskPromptsDocumentFile = nil
            return
        }
        do {
            let url = try VectorCatalog.projectFile(file)
            let data = try Data(contentsOf: url)
            let document = try TaskPromptsDocument.load(data)
            taskPromptsDocument = document
            taskPromptsDocumentFile = file
            taskPromptsText = document.editorText
            taskPromptsInstrumentSummary = document.instrumentSummary
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            taskPromptsStatus =
                "loaded \(document.count) prompt\(document.count == 1 ? "" : "s")"
                + " @ \(hash.prefix(12))…"
        } catch {
            taskPromptsStatus = "\(error)"
        }
    }

    public func saveTaskPrompts() {
        guard var manifest = selected, manifest.status == .draft else { return }
        let file = taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !file.isEmpty else {
            taskPromptsStatus = "choose a task prompts file first"
            return
        }
        let prompts = TaskPromptsDocument.editorBlocks(taskPromptsText)
        guard !prompts.isEmpty else {
            taskPromptsStatus = "add at least one prompt"
            return
        }
        do {
            let url = try VectorCatalog.projectFile(file)
            // Round-trip the FULL records: pair edited blocks with the loaded
            // document so `options`/`target`/unknown keys survive the save.
            // If the editor was never loaded for THIS file but the file
            // exists, load it now — never blind-overwrite an instrument file
            // with text-only lines.
            var document: TaskPromptsDocument
            if let loaded = taskPromptsDocument, taskPromptsDocumentFile == file {
                document = loaded
            } else if let data = try? Data(contentsOf: url),
                let loaded = try? TaskPromptsDocument.load(data)
            {
                document = loaded
            } else {
                document = TaskPromptsDocument.fromTexts([])
            }
            document = document.applyingEditedTexts(prompts)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try document.serialized().write(to: url, options: [.atomic])
            let hash = try ExperimentStore.pinTaskPrompts(file, into: &manifest)
            try ExperimentStore.save(manifest)
            refresh()
            taskPromptsDocument = document
            taskPromptsDocumentFile = file
            taskPromptsText = document.editorText
            taskPromptsInstrumentSummary = document.instrumentSummary
            // P1: report metadata-preserved and instruments-enabled as two
            // separate facts — preserved fields are NOT enabled measurement.
            let activation = InstrumentActivation.savePinSummary(
                optionsItemCount: document.optionsItemCount,
                itemCount: document.count,
                instruments: selected?.outcomeInstruments)
            taskPromptsStatus =
                "saved and pinned \(prompts.count) prompt\(prompts.count == 1 ? "" : "s")"
                + " @ \(hash.prefix(12))… — \(activation)"
            note("saved task prompts and pinned their hash — \(activation)", severity: .success)
        } catch {
            taskPromptsStatus = "\(error)"
            note(
                "Couldn't save the task prompts — nothing was pinned; check "
                    + "the file path stays inside the workspace and the study "
                    + "is a draft. Details: \(error)",
                severity: .error)
        }
    }

    /// The Import JSONL… action: validated full-record import (paste or
    /// file) that writes to the readiness scaffold's task-prompts
    /// destination, sets it as this study's prompts file, and pins the hash
    /// — one action from paste to pinned. The study-pack write rule
    /// applies: an existing destination with DIFFERING contents refuses
    /// unless `replacingExisting` (the sheet's explicit "Replace the
    /// existing file" checkbox) is set. Refusals (bad line, non-draft,
    /// differing file) surface on `taskPromptsStatus`; returns whether the
    /// import landed so the sheet can dismiss on success only. The
    /// manifest save runs INSIDE the import's transaction (its `persist`
    /// step), so a save refusal — the study frozen on disk since this
    /// panel loaded its draft copy — rolls back the written file instead
    /// of orphaning or clobbering it.
    @discardableResult
    public func importTaskPromptsJSONL(
        _ text: String, replacingExisting: Bool = false
    ) -> Bool {
        guard var manifest = selected, manifest.status == .draft else {
            taskPromptsStatus =
                "select a draft study first — import writes the file and pins "
                + "its hash into the draft manifest"
            return false
        }
        do {
            let result = try TaskPromptsImport.importIntoStudy(
                text: text, manifest: &manifest,
                replacingExisting: replacingExisting,
                persist: { try ExperimentStore.save($0) })
            taskPromptsFile = result.file
            refresh()
            // Re-read through the document loader so the editor, the
            // instrument badge, and the loaded-document pairing all reflect
            // the imported file.
            loadTaskPrompts()
            taskPromptsStatus =
                "imported \(result.recordCount) record\(result.recordCount == 1 ? "" : "s")"
                + " → \(result.file), pinned @ \(result.hash.prefix(12))…"
            note("imported task-prompt JSONL and pinned its hash", severity: .success)
            return true
        } catch {
            taskPromptsStatus = "\(error)"
            return false
        }
    }

    public func runStudy() async {
        guard let name = selectedName, !isRunning else { return }
        if isServerWorkspace {
            await runStudyOnActiveServer(verb: "run")
            return
        }
        isRunning = true
        studyRunCancelRequested = false
        resetLiveViewer()
        note("running study '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Study run — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isRunning = false
            refresh()
        }
        do {
            let runDirectory = try await ExperimentTasks.run(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.studyRunCancelRequested ?? false
                },
                progress: { [weak self] event in
                    await MainActor.run {
                        self?.handleStudyProgress(event)
                    }
                })
            lastRunDirectory = runDirectory.path
            refresh()
            refreshResults(selecting: runDirectory.lastPathComponent)
            if studyRunCancelRequested {
                // Cancelled by user: partial artifacts on disk, honestly
                // marked — never reported as an error or a completion.
                note("study run cancelled by user — partial artifacts kept "
                    + "in \(runDirectory.lastPathComponent) (no report.json; "
                    + "not a completed run)", severity: .warning)
                endDisplayLog(
                    "study run cancelled by user — partial artifacts kept in "
                        + runDirectory.lastPathComponent)
            } else {
                note("study run complete: \(runDirectory.lastPathComponent)", severity: .success)
                endDisplayLog("study run complete: \(runDirectory.lastPathComponent)")
            }
        } catch {
            refresh()
            note(
                "The study run failed — no report.json was written; any "
                    + "partial run directory remains on disk for inspection. "
                    + "Details: \(error)",
                severity: .error)
            endDisplayLog("study run failed: \(error)")
        }
    }

    public func validateStudy() async {
        guard let name = selectedName, !isValidating else { return }
        if isServerWorkspace {
            // Mac-authority mode (2026-07-21): on a KNOWN-unpaired server
            // the direct verb would execute whatever same-named copy the
            // server happens to hold (the researcher's real stale-draft
            // failure) — and even when it matched, a session-delegated
            // validate writes its run into the SERVER's tree with no
            // evidence bundle, invisible to the local freeze gate. Validate
            // travels as a hash-pinned BUNDLE job instead: it carries the
            // manifest on screen, executes under the Remote options'
            // resources, packages an evidence bundle, and the completion
            // hook imports that evidence back into THIS workspace, where
            // the local freeze gate matches it for the server run
            // substrate. Paired (and pairing-unknown) servers keep the
            // direct server-resident verb, GPU-session delegation included.
            if isKnownUnpairedServerWorkspace {
                await submitSelectedStudyRemotely(verbOverride: "validate")
                return
            }
            await runStudyOnActiveServer(verb: "validate")
            return
        }
        isValidating = true
        validationCancelRequested = false
        note("validating study '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Study validation — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isValidating = false
            refresh()
        }
        do {
            let runDirectory = try await ExperimentTasks.validate(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.validationCancelRequested ?? false
                },
                log: { [weak self] line in
                    await MainActor.run {
                        self?.appendDisplayLog(line)
                        self?.status = line
                    }
                })
            if validationCancelRequested {
                note("validation cancelled by user — partial artifacts kept "
                    + "in \(runDirectory.lastPathComponent); no validation "
                    + "evidence was written", severity: .warning)
                endDisplayLog(
                    "validation cancelled by user — no evidence written")
            } else {
                lastValidationDirectory = runDirectory.path
                note("validation complete: \(runDirectory.lastPathComponent)", severity: .success)
                endDisplayLog("validation complete: \(runDirectory.lastPathComponent)")
            }
        } catch {
            note(
                "Validation failed — no validation evidence was written, so "
                    + "freeze will still ask for a matching validate run. Fix "
                    + "the cause and validate again. Details: \(error)",
                severity: .error)
            endDisplayLog("validation failed: \(error)")
        }
    }

    // MARK: Explicit extraction (App gap A11)

    /// Why Extract is disabled, in one plain sentence — nil means runnable.
    /// Deliberately status-blind: draft AND frozen (and completed) studies
    /// may extract — re-derivation is deterministic from the pinned recipe,
    /// which is exactly what the freeze protects (the CLI extract verb gates
    /// only on `verify()`, mirrored here as the violations check).
    public static func extractDisabledReason(
        busy: Bool, hasViolations: Bool, missingOnServer: Bool
    ) -> String? {
        if busy {
            return "another study task is running — wait for it to finish"
        }
        if hasViolations {
            return "pinned inputs no longer verify — extraction would derive "
                + "vectors from drifted data"
        }
        if missingOnServer {
            return "the study is not in the active server's workspace — "
                + "Submit Bundle (verb: extract) is the portable path"
        }
        return nil
    }

    /// Explicit vector re-derivation from the pinned recipe (A11): the CLI
    /// `experiment extract` verb, in-panel. Server workspaces submit the
    /// server's extract verb as a durable job; locally the existing
    /// `ExperimentTasks.extract` runs in-process and the run directory is
    /// reported (recovered by newest-`-extract` lookup — the task API
    /// prints but does not return it).
    public func extractStudy() async {
        guard let name = selectedName, !isExtracting else { return }
        if isServerWorkspace {
            await runStudyOnActiveServer(verb: "extract")
            return
        }
        isExtracting = true
        extractCancelRequested = false
        note("extracting vectors for '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Vector extraction — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isExtracting = false
            refresh()
        }
        do {
            try await ExperimentTasks.extract(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.extractCancelRequested ?? false
                })
            let runDirectory = ExperimentStore.newestRunDirectory(
                experimentName: name, task: "extract")
            lastExtractDirectory = runDirectory?.path
            refresh()
            if let id = runDirectory?.lastPathComponent {
                refreshResults(selecting: id)
            }
            if extractCancelRequested {
                note(
                    "extraction cancelled by user — completed vectors kept in "
                        + (runDirectory?.lastPathComponent ?? "the run directory")
                        + " (marked cancelled)",
                    severity: .warning)
                endDisplayLog("extraction cancelled by user — partial vectors kept")
            } else {
                note(
                    "extraction complete: "
                        + (runDirectory?.lastPathComponent ?? "see runs/"),
                    severity: .success)
                endDisplayLog(
                    "extraction complete: "
                        + (runDirectory?.lastPathComponent ?? "see runs/"))
            }
        } catch {
            refresh()
            note(
                "Vector extraction failed — any completed vectors remain in "
                    + "the run directory; nothing pinned in the study changed. "
                    + "Details: \(error)",
                severity: .error)
            endDisplayLog("extraction failed: \(error)")
        }
    }

    /// The run the paired judge will evaluate: the selected completed run,
    /// or — sensible default — the study's LATEST completed run when the
    /// selection is empty or a non-run artifact (validation, judge output).
    public var pairedJudgeTarget: StudyRunListItem? {
        if let item = selectedResult?.item, item.kind == .run {
            return item
        }
        return resultRuns.first { $0.kind == .run }
    }

    /// Why Run Paired Judge is disabled, in one plain sentence — nil means
    /// runnable. The UI must always surface this next to the button: a
    /// silently gray button is a bug, not a state.
    public var pairedJudgeDisabledReason: String? {
        if isEvaluating || isRunning || isValidating || isExtracting {
            return "another study task is running — wait for it to finish"
        }
        guard selectedName != nil else { return "select a study first" }
        guard pairedJudgeTarget != nil else {
            return "no completed study run to judge yet — Run Study first"
        }
        let hasRubricFile = !judgeRubricFile
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasInlineRubric = !evaluationPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if !hasRubricFile, !hasInlineRubric {
            return "no judge rubric — pin a rubric file or enter inline rubric text above"
        }
        let panelJudges = judges.filter {
            !$0.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        if panelJudges.isEmpty {
            let model = judgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
            let effective = model.isEmpty ? ClaudePairedJudge.defaultModel : model
            if ClaudePairedJudge.isClaudeModel(effective), ClaudeStimulusGenerator.apiKey == nil {
                return "judge '\(effective)' needs a Claude API key — set "
                    + "ANTHROPIC_API_KEY or save a key in the Compute section "
                    + "(stored in the macOS Keychain)"
            }
        } else {
            // A local judge with no model is legal — it resolves to the
            // study model (manifest.modelID) at evaluation start.
            for judge in panelJudges {
                if judge.kind == "claude", ClaudeStimulusGenerator.apiKey == nil {
                    return "Claude judge '\(judge.name)' needs an API key — set "
                        + "ANTHROPIC_API_KEY or save a key in the Compute section "
                    + "(stored in the macOS Keychain)"
                }
                if judge.kind == "openrouter",
                    JudgeKeyStore.resolveKey(kind: "openrouter") == nil
                {
                    return "OpenRouter judge '\(judge.name)' needs an external "
                        + "judge key — save one in the Compute section or set "
                        + "OPENROUTER_API_KEY"
                }
            }
        }
        return nil
    }

    public func runPairedJudgeEvaluation() async {
        guard let name = selectedName, !isEvaluating else { return }
        guard let item = pairedJudgeTarget else {
            note("no completed study run to judge yet — Run Study first", severity: .info)
            return
        }
        // Make the defaulted target visible: judging always operates on the
        // run the Results picker shows.
        if selectedResultID != item.id {
            selectedResultID = item.id
        }
        isEvaluating = true
        evaluationCancelRequested = false
        liveEvaluationDirectory = nil
        liveActiveJudgment = nil
        liveJudgments = []
        note("running paired judge for '\(item.directoryName)'…", severity: .info)
        // A pinned rubric file makes inline draft text optional; with
        // neither, there is nothing to judge with.
        let evaluation = evaluationSpecFromDraft()
        if evaluation == nil,
            judgeRubricFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            note("enter a paired judge rubric or pin a rubric file first", severity: .info)
            isEvaluating = false
            return
        }
        beginDisplayLog(
            title: "Paired judge — \(name)",
            initialLine: "judging run \(item.directoryName)…")
        defer {
            isEvaluating = false
            refresh()
        }
        do {
            let url = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: name,
                sourceRunDirectory: URL(filePath: item.path),
                evaluation: evaluation,
                shouldCancel: { [weak self] in
                    await self?.evaluationCancelRequested ?? false
                },
                progress: { [weak self] event in
                    await MainActor.run {
                        self?.handleStudyProgress(event)
                    }
                })
            refresh()
            refreshResults(selecting: item.id)
            if evaluationCancelRequested {
                note("paired judge cancelled by user — completed judgments "
                    + "kept in \(url.lastPathComponent); no judge report written", severity: .warning)
                endDisplayLog(
                    "paired judge cancelled by user — no judge report written")
            } else {
                lastEvaluationDirectory = url.path
                note("paired judge complete: \(url.lastPathComponent)", severity: .success)
                endDisplayLog("paired judge complete: \(url.lastPathComponent)")
            }
        } catch {
            refresh()
            note(
                "Paired judging failed — no judge report was written; the "
                    + "source run is untouched, so judging can simply be run "
                    + "again. Details: \(error)",
                severity: .error)
            endDisplayLog("paired judge failed: \(error)")
        }
    }

    public func clearLiveViewer() {
        resetLiveViewer()
    }

    private func resetLiveViewer() {
        liveRunDirectory = nil
        liveEvaluationDirectory = nil
        liveActiveGeneration = nil
        liveActiveJudgment = nil
        liveGenerations = []
        liveJudgments = []
    }

    private func handleStudyProgress(_ event: ExperimentTasks.StudyTaskProgress) {
        switch event {
        case .runDirectory(let path):
            liveRunDirectory = path
            status = "writing study artifacts to \(URL(filePath: path).lastPathComponent)…"
            appendDisplayLog("run directory: \(URL(filePath: path).lastPathComponent)")
        case .generationStarted(let condition, let promptID, let prompt):
            liveActiveGeneration = LiveStudyGeneration(
                condition: condition,
                promptID: promptID,
                prompt: prompt,
                output: "")
            status = "generating \(condition) · \(promptID)…"
            appendDisplayLog("generating [\(condition)] \(promptID)…")
        case .generationChunk(let condition, let promptID, let output):
            // Per-token chunks update only the in-panel viewer — mirroring
            // every chunk would flood the display-pane log.
            liveActiveGeneration = LiveStudyGeneration(
                condition: condition,
                promptID: promptID,
                prompt: liveActiveGeneration?.prompt ?? "",
                output: output)
        case .generationCompleted(let generation):
            liveGenerations.insert(generation, at: 0)
            liveActiveGeneration = nil
            status = "generated \(generation.condition) · \(generation.promptID)"
            appendDisplayLog(
                "generated [\(generation.condition)] \(generation.promptID): "
                    + "\(generation.wordCount) words")
        case .evaluationDirectory(let path):
            liveEvaluationDirectory = path
            status = "writing judge artifacts to \(URL(filePath: path).lastPathComponent)…"
            appendDisplayLog("judge directory: \(URL(filePath: path).lastPathComponent)")
        case .judgmentStarted(let condition, let promptID):
            liveActiveJudgment = LiveStudyJudgment(condition: condition, promptID: promptID)
            status = "judging \(condition) · \(promptID)…"
        case .judgmentCompleted(let judgment):
            liveJudgments.insert(judgment, at: 0)
            liveActiveJudgment = nil
            status = "judged \(judgment.condition) · \(judgment.promptID): \(judgment.conditionResult)"
            appendDisplayLog(
                "judged [\(judgment.condition)] \(judgment.promptID): "
                    + "\(judgment.conditionResult)")
        case .codingCompleted(let coding):
            liveActiveJudgment = nil
            let codes = coding.codes.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value.displayString)" }
                .joined(separator: ", ")
            status = "coded \(coding.condition) · \(coding.promptID) "
                + "[\(coding.judge)]"
            appendDisplayLog(
                "coded [\(coding.condition)] \(coding.promptID) "
                    + "(\(coding.judge)): \(codes)")
        }
    }

    // MARK: Direct concept attach / detach (App gap A8)

    /// The attach picker's options: concepts on disk not yet attached to the
    /// selected study, with what their on-disk data supports (paired
    /// stimulus set, grand-mean stories, or both).
    public var attachableConceptSources: [ExperimentStore.ConceptSources] {
        attachableConcepts
            .map { ExperimentStore.conceptSources(name: $0) }
            .filter { !$0.supportedMethods.isEmpty }
    }

    /// One pin-status line per attached concept for the Studies list:
    /// stimulus hash, method, reading position, and the three-state
    /// validation pin. Pure; unit-tested.
    public static func conceptPinStatusLine(
        _ ref: ExperimentManifest.ConceptRef
    ) -> String {
        var parts = [
            "stimuli @ \(ref.stimulusSetHash.prefix(12))…",
            ref.options.method.rawValue,
            ref.options.readingPosition.label,
        ]
        if let hash = ref.validationHash {
            parts.append("validation @ \(hash.prefix(12))…")
        } else if ref.validationHashPinnedAbsent {
            parts.append("validation pinned absent")
        } else {
            parts.append("validation unpinned (legacy attach)")
        }
        return parts.joined(separator: " · ")
    }

    /// One-step attach from the Studies picker (A8): writes through
    /// `ExperimentStore.attachConcept` — the same pins as the CLI attach
    /// (stimulus hash + validationHash + neutral corpus + grand-mean
    /// corpus). Draft-only; the store's immutability refusal surfaces here.
    public func attachConceptFromPicker() {
        guard let experiment = selectedName else { return }
        let concept = attachConceptName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !concept.isEmpty else {
            note("pick a concept to attach", severity: .warning)
            return
        }
        var poolFrom: Int?
        let poolText = attachPoolFromText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !poolText.isEmpty {
            guard let token = Int(poolText), token >= 0 else {
                note(
                    "pool-from must be a token index ≥ 0 — got '\(poolText)'",
                    severity: .error)
                return
            }
            poolFrom = token
        }
        let corpus = attachCorpusText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        do {
            let manifest = try ExperimentStore.attachConcept(
                concept,
                method: attachMethod,
                poolFromToken: poolFrom,
                corpusConcepts: corpus,
                reference: attachReferenceName.isEmpty ? nil : attachReferenceName,
                experimentName: experiment)
            refresh()
            attachConceptName = ""
            if attachMethod == .emotionGrandMean {
                let hash = manifest.grandMeanCorpus?.hashes[concept] ?? ""
                note(
                    "pinned \(concept) @ \(hash.prefix(12))… (emotionGrandMean, "
                        + "corpus of \(manifest.grandMeanCorpus?.concepts.count ?? 0) "
                        + "concept(s))",
                    severity: .success)
            } else if let ref = manifest.concepts.first(where: { $0.name == concept }) {
                note(
                    "pinned \(concept) @ \(ref.stimulusSetHash.prefix(12))… "
                        + "(\(ref.options.method.rawValue), "
                        + "\(ref.options.readingPosition.label))",
                    severity: .success)
            }
        } catch {
            note(
                "Couldn't attach the concept — check its stimulus files "
                    + "exist on disk and the study is still a draft. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// Detach a pinned concept from the selected draft (store-refused for
    /// frozen studies and for concepts still referenced by conditions).
    public func detachConcept(_ concept: String) {
        guard let experiment = selectedName else { return }
        do {
            try ExperimentStore.detachConcept(concept, experimentName: experiment)
            refresh()
            note("detached concept '\(concept)' from '\(experiment)'")
        } catch {
            note("\(error)", severity: .error)
        }
    }

    /// Pins the concept at its current stimulus hash, using the concepts
    /// panel's current extraction options (method, pooling).
    public func attachConcept(_ name: String) {
        guard var manifest = selected, manifest.status == .draft else { return }
        do {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            let stimuli = try StimulusSet(directory: directory)
            let options = host?.concepts.extractionOptions ?? ExtractionOptions()
            manifest.concepts.removeAll { $0.name == name }
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: name, stimulusSetHash: stimuli.hash, options: options))
            ExperimentStore.pinNeutralCorpus(into: &manifest)  // norm denominator
            try ExperimentStore.save(manifest)
            refresh()
            note("pinned \(name) @ \(stimuli.hash.prefix(12))…", severity: .success)
        } catch {
            note(
                "Couldn't attach the concept — check its stimulus files "
                    + "under prompts/concepts/\(name)/ and that the study is "
                    + "still a draft. Details: \(error)",
                severity: .error)
        }
    }

    public func addVariantCondition(_ id: ModelVariantRecord.ID? = nil) {
        guard var manifest = selected, manifest.status == .draft else { return }
        // An unsaved picker change lands here too: the agent about to be
        // added must be checked against the model the researcher CHOSE, not
        // the one last saved (same reset rules as `saveProtocol`).
        applyStudyBaseModelChoice(to: &manifest)
        guard let id = id ?? selectedVariantToAddID,
            let record = ModelVariantStore.scan().first(where: { $0.id == id })
        else {
            note("select an agent to add", severity: .info)
            return
        }
        guard record.artifact.baseModelID == manifest.modelID else {
            note("agent uses \(record.artifact.baseModelID), not study base model \(manifest.modelID)", severity: .info)
            return
        }
        do {
            // The one agent → condition path, shared with template
            // instantiation (`ExperimentStore.attachAgent`).
            try ExperimentStore.attachAgent(record, into: &manifest)
            try ExperimentStore.save(manifest)
            selectedVariantToAddID = nil
            refresh()
            note("added agent '\(record.artifact.name)'", severity: .success)
        } catch {
            note(
                "Couldn't add the agent — check it uses this study's "
                    + "baseline model and the study is still a draft. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    public func removeVariantCondition(_ name: String) {
        guard var manifest = selected, manifest.status == .draft else { return }
        manifest.variantConditions.removeAll { $0.name == name }
        do {
            try ExperimentStore.save(manifest)
            refresh()
            note("removed agent '\(name)'", severity: .success)
        } catch {
            note("\(error)", severity: .error)
        }
    }

    // MARK: Study focus + study.json copy/paste (authoring streamline)

    /// View-layer override of the derived study intent — filters which
    /// sections render; NEVER stored in the manifest and never deletes
    /// anything. Cleared on selection change.
    public var studyFocusOverride: StudyIntent?

    /// The effective focus: the user's override, else derived from what
    /// the manifest actually contains.
    public var studyFocus: StudyIntent {
        studyFocusOverride
            ?? selected.map(StudyIntent.derive(from:))
            ?? .conceptStudy
    }

    /// The selected study as one pasteable JSON document (the same
    /// experiment.json every engine reads). Nil (with a notice) on failure.
    public func exportSelectedStudyJSON() -> String? {
        guard let manifest = selected else {
            note("select a study first", severity: .info)
            return nil
        }
        do {
            return try ExperimentStore.exportStudyJSON(manifest)
        } catch {
            note("could not export study JSON: \(error)", severity: .error)
            return nil
        }
    }

    /// Import pasted study JSON as a NEW DRAFT (freeze metadata stripped —
    /// pasted text cannot mint a preregistered object), select it, and
    /// surface its verify() result loudly.
    public func importStudyJSON(_ text: String) {
        do {
            let (manifest, violations, filesWritten) =
                try ExperimentStore.importStudyJSON(text)
            refresh()
            selectedName = manifest.name
            let filesNote = filesWritten.isEmpty
                ? ""
                : " (+ \(filesWritten.count) data file(s) written: "
                    + filesWritten.joined(separator: ", ") + ")"
            if violations.isEmpty {
                note("imported draft '\(manifest.name)'\(filesNote) — "
                    + "verify clean",
                     severity: .success)
            } else {
                note(
                    "imported draft '\(manifest.name)'\(filesNote) with "
                        + "\(violations.count) verification issue(s): "
                        + violations.joined(separator: "; "),
                    severity: .error)
            }
        } catch {
            note(
                "Couldn't import the study JSON — nothing was created; check "
                    + "it is valid JSON and that its \"name\" is not already "
                    + "in use. Details: \(error)",
                severity: .error)
        }
    }

    /// Save the Pipeline Composer's declaration into the manifest (stage 5,
    /// sixth round — the app must AUTHOR the chain it runs, not just submit
    /// it). `nil` removes the block. Draft-only, like every declaration.
    public func savePipelineDeclaration(_ draft: PipelineDraft?) {
        guard var manifest = selected, manifest.status == .draft else { return }
        manifest.pipeline = draft?.encoded()
        do {
            try ExperimentStore.save(manifest)
            refresh()
            note(
                draft == nil
                    ? "pipeline declaration removed"
                    : "pipeline declared — submit it with Run Pipeline",
                severity: .success)
        } catch {
            note(
                "Couldn't save the pipeline declaration — the study must "
                    + "still be a draft and its file writable. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// The primary action for a declared chain: submit the pipeline verb
    /// through the same bundle path as every remote run (executor and
    /// resources come from Remote options).
    public func runPipelineRemotely() async {
        remoteVerb = "pipeline"
        await submitSelectedStudyRemotely()
    }

    /// Stage-4 authoring affordance (stage 5): declare "the agent this
    /// study's sweep promotes for CONCEPT" as a condition — before the
    /// agent exists. Data-only: the SERVER resolves it at run time from the
    /// promotion birth certificate and pins path + hash as run evidence.
    public func addForwardReferencedCondition(concept: String) {
        guard var manifest = selected, manifest.status == .draft else { return }
        guard manifest.concepts.contains(where: { $0.name == concept }) else {
            note("attach concept '\(concept)' first", severity: .info)
            return
        }
        let conditionName = "\(concept)-agent"
        guard !manifest.variantConditions.contains(
            where: { $0.name == conditionName })
        else {
            note("condition '\(conditionName)' already declared", severity: .info)
            return
        }
        manifest.variantConditions.append(
            .init(
                name: conditionName, artifactPath: "", artifactHash: "",
                artifact: .init(
                    name: "", baseModelID: "", promptMode: "",
                    qwenThinkingEnabled: false, temperature: 0,
                    systemPrompt: ""),
                fromPromotion: .init(concept: concept)))
        do {
            try ExperimentStore.save(manifest)
            refresh()
            note(
                "declared '\(conditionName)' — the agent this study's sweep "
                    + "promotes for '\(concept)', resolved at run time",
                severity: .success)
        } catch {
            note("could not declare forward reference: \(error)", severity: .error)
        }
    }

    /// Serializes the live steering boxes into a named condition. Concepts
    /// not yet attached are pinned automatically at their current hashes.
    public func captureCondition() {
        guard let host, var manifest = selected else { return }
        let name = conditionName.isEmpty ? "condition-\(manifest.conditions.count + 1)"
            : conditionName
        var slots: [ExperimentManifest.Condition.Slot] = []
        do {
            for slot in host.slots where slot.enabled {
                guard let artifact = host.artifact(for: slot) else { continue }
                let concept = artifact.sidecar.concept
                if !manifest.concepts.contains(where: { $0.name == concept }) {
                    let directory = VectorCatalog.conceptsDirectory.appending(
                        component: concept)
                    let stimuli = try StimulusSet(directory: directory)
                    let options = host.concepts.extractionOptions
                    manifest.concepts.append(
                        ExperimentStore.makeConceptRef(
                            name: concept, stimulusSetHash: stimuli.hash,
                            options: options))
                }
                slots.append(
                    .init(concept: concept, layer: Int(slot.layer), alpha: slot.alpha))
            }
            guard !slots.isEmpty else {
                note("no enabled steering boxes with vectors to capture", severity: .info)
                return
            }
            manifest.conditions.removeAll { $0.name == name }
            let neutralBasis = host.removeNeutralDirectionsAtSteering
                ? host.selectedNeutralPCBasis
                : nil
            // Cross-engine pin rule (2026-07-13): neutralPCBasisHash is the
            // SHA-256 of the basis FILE BYTES (verify() checks it) — the
            // historical corpusHash stamp pinned only the corpus, not the
            // PCA output.
            let neutralBasisHash = neutralBasis.flatMap { record -> String? in
                guard let data = try? Data(contentsOf: record.url) else { return nil }
                return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            }
            manifest.conditions.append(
                .init(
                    name: name, slots: slots,
                    bandWidth: host.layerBandWidth,
                    alphaInNormUnits: host.alphaInNormUnits,
                    neutralPCBasisPath: neutralBasis.map(NeutralPCStore.relativePath),
                    neutralPCBasisLabel: neutralBasis?.label,
                    neutralPCBasisHash: neutralBasisHash))
            ExperimentStore.pinNeutralCorpus(into: &manifest)  // norm denominator
            try ExperimentStore.save(manifest)
            conditionName = ""
            refresh()
            note("captured '\(name)' (\(slots.count) slot\(slots.count == 1 ? "" : "s"))", severity: .success)
        } catch {
            note(
                "Couldn't capture the condition — the study file may be "
                    + "locked or the disk full. Details: \(error)",
                severity: .error)
        }
    }

    /// Adds an explicit no-intervention condition. Baseline is domain-neutral:
    /// it means "same task and sampling settings, no activation edits."
    public func addBaselineCondition() {
        guard var manifest = selected, manifest.status == .draft else { return }
        let name = conditionName.isEmpty ? "baseline" : conditionName
        do {
            manifest.conditions.removeAll { $0.name == name }
            manifest.conditions.append(
                .init(name: name, slots: [], bandWidth: 1, alphaInNormUnits: true))
            try ExperimentStore.save(manifest)
            conditionName = ""
            refresh()
            clearFormError(.addCondition)
            note("added no-steer baseline '\(name)'", severity: .success)
        } catch {
            refuse(
                .addCondition,
                "Couldn't add the baseline condition — the study file may be "
                    + "locked or read-only. Details: \(error)")
        }
    }

    public func removeCondition(_ name: String) {
        guard var manifest = selected, manifest.status == .draft else { return }
        manifest.conditions.removeAll { $0.name == name }
        do {
            try ExperimentStore.save(manifest)
            refresh()
        } catch {
            note("\(error)", severity: .error)
        }
    }

    public func freeze() {
        guard let name = selectedName else { return }
        do {
            // Mac-authority mode: a local freeze under server Compute
            // matches evidence for the RUN substrate (the server) — an
            // imported server validate run counts; this engine's own
            // evidence does not, because the study will not run here.
            let runSubstrate = freezeEvidenceRunSubstrate
            let frozen = try ExperimentStore.freeze(
                name: name, runSubstrate: runSubstrate)
            refresh()
            var line = "frozen @ \(frozen.freezeHash?.prefix(12) ?? "?")… "
                + "(git \(frozen.gitCommit?.prefix(8) ?? "uncommitted"))"
            if runSubstrate != ExperimentStore.evidenceSubstrate {
                line += " — evidence matched for the run substrate "
                    + "\(runSubstrate); submit the frozen study to the server "
                    + "as a bundle to run it"
            }
            note(line, severity: .success)
        } catch {
            note(
                "Freeze did not complete — the study is unchanged and still "
                    + "a draft; the detail names the gate or pin that "
                    + "stopped it. Details: \(error)",
                severity: .error)
        }
    }

    /// The perspective-shifted cross-substrate advisory backing
    /// `serverFreezeCrossSubstrateAdvisory` (see its doc): only meaningful
    /// when the freeze would route to a server that serves THIS workspace
    /// tree — an unpaired server's evidence is invisible to a local scan, so
    /// warning from it would cry wolf.
    private func computeServerFreezePerspectiveAdvisory(
        for manifest: ExperimentManifest
    ) -> String? {
        guard isServerWorkspace, cluster?.activeServerPairing == .paired,
            !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
                || !manifest.variantConditions.isEmpty
        else { return nil }
        return ExperimentStore.crossSubstrateValidationAdvisory(
            for: manifest, perspective: WorkspaceScoping.serverSubstrate)
    }

    /// Freeze the selected study THROUGH the active server's gated freeze
    /// authority (`POST /api/authoring/{name}/freeze`): the gates are
    /// evaluated server-side against SERVER-substrate evidence and the
    /// manifest is stamped `frozenBy: "server"` — completing the
    /// substrate-coherent order server-extract → server-validate →
    /// server-freeze → server-run without leaving the app. Deliberately no
    /// force parameter (parity with local freeze: forcing requires the CLI).
    /// On a workspace-PAIRED server the frozen manifest lands in the shared
    /// tree and `refresh()` re-reads it — verify() then re-checks it through
    /// the frozenBy:"server" freeze-canonical path; on an unpaired server the
    /// frozen manifest lives in the SERVER's tree and the status line says so.
    public func freezeOnActiveServer() async {
        guard let name = selectedName else {
            note("select a study first", severity: .info)
            return
        }
        guard isServerWorkspace else {
            note("no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note("invalid server URL", severity: .error)
            return
        }
        remoteFreezeGateFailure = nil
        remoteFreezeAdvisories = []
        remoteFreezeIdentityWarning = nil
        remoteFreezeIdentityNote = nil
        remoteFreezeCanSyncDraft = false
        let substrate = cluster?.substrateLabel ?? "server"
        // Residency + lifecycle preflight in one read (`GET
        // /api/experiment/{name}`): freeze stamps the SERVER-RESIDENT copy
        // only, and an already-frozen server copy refuses here with a clear
        // message instead of a job-side 400. A transient fetch failure
        // proceeds — the freeze call's own refusal is the backstop.
        var serverReportedStatus: String?
        do {
            let detail = try await client.experimentDetail(name: name)
            if name == selectedName { noteServerResidency(true) }
            serverReportedStatus = detail.status
            if let remoteState = detail.status, remoteState != "draft" {
                note("'\(name)' on \(substrate) is already \(remoteState) — "
                    + "duplicate to iterate", severity: .info)
                return
            }
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(404, _) = error {
                if name == selectedName { noteServerResidency(false) }
                note("study '\(name)' is not in \(substrate)'s workspace — "
                    + "freeze stamps the server-resident copy only. Pair the "
                    + "server to this workspace (serve --root <workspace>), or "
                    + "switch Compute to Local (MLX) to freeze the local copy", severity: .info)
                return
            }
        } catch {}
        // Manifest-identity guard: the freeze call names a study; the server
        // freezes WHICHEVER same-named copy it holds. Verify that copy IS the
        // document on screen (same-engine canonicalization of both documents,
        // volatile freeze stamps excluded — deliberately NOT an invented
        // cross-engine byte-canonical hash) before anything is stamped.
        // Block/proceed rules live in `FreezeRouting.remoteFreezePrecheck`.
        let identity = await remoteFreezeManifestIdentity(
            client: client, name: name, substrate: substrate,
            serverReportedStatus: serverReportedStatus)
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: identity,
            study: name,
            serverLabel: substrate,
            workspacePaired: cluster?.activeServerPairing == .paired)
        guard precheck.proceed else {
            remoteFreezeIdentityWarning = precheck.message
            // The one-click remedy (2026-07-21 incident, part 3): a
            // mismatched LOCAL DRAFT can be pushed as the server's draft
            // copy — the view offers "Update the server's copy".
            remoteFreezeCanSyncDraft = FreezeRouting.canOfferServerDraftSync(
                identity: identity, localIsDraft: selected?.status == .draft)
            note("freeze on \(substrate) refused before submission — "
                + "its copy of '\(name)' could not be confirmed as the "
                + "manifest on screen", severity: .error)
            return
        }
        remoteFreezeIdentityNote = precheck.message
        note("freezing '\(name)' on \(substrate)…", severity: .info)
        do {
            let result = try await client.freezeExperiment(name: name)
            remoteFreezeAdvisories = result.advisories
            // Paired server: the frozen manifest was written into the shared
            // tree — re-read it so the panel shows frozen/frozenBy:"server"
            // and local verify() re-checks via freeze-canonical.json.
            refresh()
            var line = "frozen on \(substrate) @ "
                + "\(result.manifest.freezeHash?.prefix(12) ?? "?")… "
                + "(frozenBy: server"
                + (result.manifest.gitCommit.map { ", git \($0.prefix(8))" } ?? "")
                + ")"
            if selected?.status != .frozen {
                line += " — the frozen manifest is in \(substrate)'s workspace; "
                    + "the local copy is untouched"
            }
            note(line, severity: .success)
        } catch let error as ClusterClient.ClientError {
            // The server's gate refusal IS the actionable text — keep it
            // verbatim and render it like an unmet local freeze gate.
            let unwrapped = ClusterClient.unwrappingDetail(error)
            if case .badResponse(_, let detail) = unwrapped {
                remoteFreezeGateFailure = detail
            }
            note(
                "\(substrate) declined to freeze — a gate or pin on the "
                    + "server side is unmet; the study is unchanged. "
                    + "Details: \(unwrapped)",
                severity: .error)
        } catch {
            note(
                "Freeze on \(substrate) did not complete — the study is "
                    + "unchanged; check the server connection and try again. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// Fetch-and-compare step of the remote-freeze identity guard: the
    /// server's raw manifest body vs the LOCAL manifest document (the file
    /// backing what the Studies list displays). Pure comparison lives in
    /// `ExperimentStore.compareManifestDocuments`; pure block/proceed rules
    /// in `FreezeRouting.remoteFreezePrecheck` — this only does the IO.
    private func remoteFreezeManifestIdentity(
        client: ClusterClient, name: String, substrate: String,
        serverReportedStatus: String?
    ) async -> FreezeRouting.RemoteManifestIdentity {
        let localData = ExperimentStore.manifestData(name: name)
        do {
            let serverBody = try await client.experimentManifestBody(name: name)
            guard let localData else {
                return .localMissing(
                    serverStatus: serverReportedStatus,
                    canonicalBodyHash: ExperimentStore.canonicalManifestBodyHash(
                        serverBody))
            }
            switch ExperimentStore.compareManifestDocuments(
                local: localData, server: serverBody)
            {
            case .equal:
                return .verifiedEqual
            case .different(let fields):
                return .mismatch(fields)
            case .unparseable:
                return .unverifiable("a manifest body is not a JSON object")
            }
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(404, _) = error {
                return .unverifiable(
                    "\(substrate) does not serve the manifest body — "
                        + "older server without GET /api/experiment/{name}/manifest")
            }
            return .unverifiable(
                "manifest body fetch failed: \(ClusterClient.unwrappingDetail(error))")
        } catch {
            return .unverifiable("manifest body fetch failed: \(error)")
        }
    }

    /// "Update the server's copy" (2026-07-21 incident, part 3): push the
    /// CURRENT local manifest document as the active server's DRAFT copy
    /// (`PUT /api/experiment/{name}/manifest`), then RE-RUN the identity
    /// check and report verified-equal — or the remaining difference,
    /// honestly. Draft manifests only, on both sides: this method refuses a
    /// frozen local document, and the server refuses to overwrite a frozen
    /// copy (freeze firewall — duplicate to iterate). Nothing is
    /// auto-frozen afterwards: the researcher clicks Freeze again with the
    /// identity verified.
    public func pushManifestToActiveServer() async {
        guard let name = selectedName, let manifest = selected else {
            note("select a study first", severity: .info)
            return
        }
        guard manifest.status == .draft else {
            note("only a DRAFT manifest can be pushed as the server's copy — "
                + "frozen studies are read-only; duplicate to iterate",
                severity: .info)
            return
        }
        loadStoredRemoteToken()
        guard let client = remoteClient else {
            note("invalid server URL", severity: .error)
            return
        }
        guard let localData = ExperimentStore.manifestData(name: name) else {
            note("could not read the local manifest file for '\(name)' — "
                + "nothing was pushed", severity: .error)
            return
        }
        let substrate = cluster?.substrateLabel ?? "server"
        isSyncingServerDraft = true
        defer { isSyncingServerDraft = false }
        do {
            note("updating \(substrate)'s copy of '\(name)'…", severity: .info)
            let result = try await client.replaceExperimentManifest(
                name: name, manifestBody: localData)
            // Merge semantics (2026-08-06): the server KEEPS auto-pins the
            // pushed document omitted (its resolved model revision, sweep
            // projections) and names them in `preserved`. Adopt the
            // revision into the local draft HERE — same rule as
            // EvidenceRevisionAdoption: completing the researcher's own
            // declared state — so the identity re-check below converges
            // instead of re-flagging a difference the researcher never
            // authored. Anything else preserved is reported honestly.
            adoptPreservedServerPins(result.preserved, study: name,
                                     substrate: substrate)
            // Re-run the SAME identity check the freeze precheck uses — the
            // affordance's claim is "now verified equal", never "pushed, so
            // it must match".
            let identity = await remoteFreezeManifestIdentity(
                client: client, name: name, substrate: substrate,
                serverReportedStatus: result.status)
            let outcome = FreezeRouting.serverDraftSyncOutcome(
                recheck: identity,
                study: name,
                serverLabel: substrate,
                canonicalBodyHash: result.canonicalBodyHash)
            if outcome.resolved {
                remoteFreezeIdentityWarning = nil
                remoteFreezeCanSyncDraft = false
                remoteFreezeIdentityNote = outcome.message
                note(outcome.message, severity: .success)
            } else {
                remoteFreezeIdentityWarning = outcome.message
                note(outcome.message, severity: .error)
            }
        } catch let error as ClusterClient.ClientError {
            // The server's refusal (e.g. its copy is frozen) is the
            // actionable text — verbatim.
            let unwrapped = ClusterClient.unwrappingDetail(error)
            note(
                "\(substrate) declined the manifest push — its copy is "
                    + "unchanged. Details: \(unwrapped)",
                severity: .error)
        } catch {
            note(
                "Manifest push to \(substrate) did not complete — its copy "
                    + "may be unchanged; check the connection and try again. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// The client half of the push-merge rule (2026-08-06): when the server
    /// KEPT auto-pins the pushed document omitted, adopt the model revision
    /// into the still-unpinned local draft (the same completing-declared-
    /// intent rule as `EvidenceRevisionAdoption`) and name everything else
    /// loudly — preserved sweep projections come home through sweep-run
    /// discovery (`SweepConditionAdoption`), not through this response,
    /// which carries only their names.
    private func adoptPreservedServerPins(
        _ preserved: ClusterClient.RemoteManifestReplaceResult.PreservedPins?,
        study name: String,
        substrate: String
    ) {
        guard let preserved else { return }
        if let revision = preserved.modelRevision, !revision.isEmpty {
            let local = try? ExperimentStore.load(name: name)
            if let local, local.modelRevision == nil, local.status == .draft {
                do {
                    try ExperimentStore.updateDraft(name: name) {
                        $0.modelRevision = revision
                    }
                    refresh()
                    note(
                        "\(substrate) kept its auto-pinned model revision "
                            + "\(revision) (the push omitted one) — adopted "
                            + "into the local draft so both copies agree",
                        severity: .info)
                } catch {
                    note(
                        "\(substrate) kept its auto-pinned model revision "
                            + "\(revision), but adopting it into the local "
                            + "draft failed: \(error)",
                        severity: .error)
                }
            } else if let localRevision = local?.modelRevision,
                localRevision != revision
            {
                note(
                    "\(substrate) kept model revision \(revision), but the "
                        + "local draft pins \(localRevision) — the copies "
                        + "genuinely disagree; re-validate at one revision "
                        + "or duplicate the study",
                    severity: .warning)
            }
        }
        if let conditions = preserved.conditions, !conditions.isEmpty {
            note(
                "\(substrate) kept its sweep-projected condition(s) "
                    + conditions.joined(separator: ", ")
                    + " — the push omitted them; adopt them locally via the "
                    + "sweep run's Adopt Projections before comparing copies",
                severity: .info)
        }
        if preserved.capabilityBattery != nil {
            note(
                "\(substrate) kept its capability-battery pin — the push "
                    + "omitted one; local freeze re-derives its own pin",
                severity: .info)
        }
    }

    public func duplicateSelected() {
        guard let name = selectedName else { return }
        var candidate = "\(name)-2"
        var counter = 2
        while (try? ExperimentStore.load(name: candidate)) != nil {
            counter += 1
            candidate = "\(name)-\(counter)"
        }
        do {
            let copy = try ExperimentStore.duplicate(name: name, as: candidate)
            refresh()
            selectedName = copy.name
            note("created draft '\(copy.name)'", severity: .success)
        } catch {
            note(
                "Couldn't duplicate the study — nothing was created; check "
                    + "the experiments/ folder is writable. Details: \(error)",
                severity: .error)
        }
    }

    private func syncDraftFieldsFromSelection(force: Bool = false) {
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
            // Workspace-scoped default: the active workspace's model choice
            // (server target → the selected/loaded SERVER model), falling
            // back to that workspace's inventory — never a local MLX id
            // seeding a server study.
            studyBaseModelID =
                host?.workspaceSelectedModelID
                ?? modelOptions.first
                ?? ChatService.availableModels.first?.id ?? ""
            selectedVariantToAddID = nil
            selectedMultiAgentScenarioID = multiAgentScenarioOptions.first?.id
            seatCastingEdits = [:]
            multiAgentIncludeBaseline = true
            promptMode = .chatAssistant
            systemPrompt = ""
            qwenThinkingEnabled = false
            evaluationPrompt = ""
            evaluationStructuredPrompt = ""
            judgeModel = defaultJudgeModel(for: nil)
            judgeRubricFile = ""
            judges = []
            judgeKindStashes = [:]
            resultRuns = []
            selectedResultID = nil
            selectedResult = nil
            selectedResultBrowserItem = nil
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
            return
        }
        guard force || syncedSelection != manifest.name else { return }
        syncedSelection = manifest.name
        protocolDescription = manifest.experimentDescription
        taskDescription = manifest.taskDescription ?? ""
        outcomeMeasures = manifest.outcomeMeasures ?? ""
        studyKind = manifest.studyKind
        // The study type needs no sync: `studyFocus` derives it from the
        // manifest (perturbation policy → confirm; concepts → concept
        // study; …) unless the user overrides via the top-of-page picker.
        studyBaseModelID = manifest.modelID
        selectedVariantToAddID = availableVariantsForStudy.first?.id
        confirmAgentID = confirmableAgents.first?.id
        // The picker names the scenario a researcher CHOSE. For a cast study
        // that is the semantic scenario it was compiled from — the compiled
        // file is deliberately outside the library and would leave the picker
        // reading "select…" on a study that is fully configured.
        let pickerPath = manifest.multiAgentSemanticScenarioPath
            ?? manifest.multiAgentScenarioPath
        if let pickerPath {
            // Symlinks resolved on both sides: a directory listing and a path
            // built from the workspace root can spell the same file two ways
            // (a workspace under /tmp is the everyday case), and a selection
            // that silently reads "select…" on a configured study is worse
            // than a slow comparison.
            let target = scenarioURL(from: pickerPath).resolvingSymlinksInPath().path
            selectedMultiAgentScenarioID = multiAgentScenarioOptions.first {
                $0.url.resolvingSymlinksInPath().path == target
            }?.id
        } else {
            selectedMultiAgentScenarioID = multiAgentScenarioOptions.first?.id
        }
        // Seat edits belong to the study they were made on.
        seatCastingEdits = [:]
        multiAgentIncludeBaseline = manifest.multiAgentIncludeBaseline
        taskPromptsFile = manifest.taskPromptsFile ?? "prompts/dev/dev-prompts.jsonl"
        if manifest.studyKind == .modelOutput {
            loadTaskPrompts()
        } else {
            taskPromptsText = ""
            taskPromptsStatus = nil
            taskPromptsInstrumentSummary = nil
            taskPromptsDocument = nil
            taskPromptsDocumentFile = nil
        }
        promptMode = manifest.promptMode ?? .chatAssistant
        systemPrompt = manifest.systemPrompt ?? ""
        qwenThinkingEnabled = manifest.qwenThinkingEnabled ?? false
        evaluationPrompt = manifest.evaluation?.judgePrompt ?? ""
        evaluationStructuredPrompt = manifest.evaluation?.structuredPrompt ?? ""
        judgeModel = manifest.evaluation?.judgeModel ?? defaultJudgeModel(for: manifest)
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
    }

    private func nilIfEmpty(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func relativeProjectPath(for url: URL) -> String {
        let root = VectorCatalog.projectRoot.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        if path == root { return "." }
        if path.hasPrefix(root + "/") {
            return String(path.dropFirst(root.count + 1))
        }
        return path
    }

    private func scenarioURL(from path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(filePath: path)
        }
        return VectorCatalog.projectRoot.appending(path: path)
    }

    private func evaluationSpecFromDraft() -> ExperimentManifest.EvaluationSpec? {
        let prompt = evaluationPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return nil }
        return ExperimentManifest.EvaluationSpec(
            kind: .pairedJudge,
            judgeModel: resolvedInlineJudgeModel(),
            judgePrompt: prompt,
            structuredPrompt: nilIfEmpty(evaluationStructuredPrompt))
    }

    /// The ad-hoc judge model the inline (scratchpad) evaluation spec
    /// carries: the panel field, else the study-model default.
    private func resolvedInlineJudgeModel() -> String {
        let trimmed = judgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultJudgeModel(for: selected) : trimmed
    }

    /// The explicit `evaluation` block a draft save writes (2026-07-22
    /// incident: the rubric-FILE + judges path never wrote one, and a
    /// frozen study died at the evaluate stage). Static and pure so the
    /// rule is testable without a live panel.
    ///
    /// Pinned judges + a chosen rubric file ARE a paired-judge declaration
    /// — written explicitly with the same shape the engines synthesize for
    /// legacy manifests (kind pairedJudge; judgeModel and judgePrompt
    /// empty: the panel carries the judges, the pinned file the rubric),
    /// plus the panel's structured-fields declaration, which the file path
    /// used to silently drop. Without the pin pair, the scratchpad rule is
    /// unchanged: inline text declares a draft-only inline evaluation,
    /// nothing declares none — so removing the last judge or clearing the
    /// rubric clears/updates the block coherently on the next save.
    ///
    /// The RULE now lives in `ExperimentStore.evaluationDeclaration` (WP0
    /// step 5½): the headless `experiment pin-rubric` verb writes the same
    /// declaration, and a second implementation of "what did the researcher
    /// declare" would drift from the one the panel wrote. This stays as the
    /// panel's name for it — its own tests call it — and forwards.
    nonisolated static func evaluationDeclaration(
        judges: [ExperimentManifest.JudgeRef],
        rubricFile: String,
        inlineRubric: String,
        structuredPrompt: String?,
        inlineJudgeModel: String
    ) -> ExperimentManifest.EvaluationSpec? {
        ExperimentStore.evaluationDeclaration(
            judges: judges, rubricFile: rubricFile, inlineRubric: inlineRubric,
            structuredPrompt: structuredPrompt,
            inlineJudgeModel: inlineJudgeModel)
    }

    private func defaultJudgeModel(for manifest: ExperimentManifest?) -> String {
        manifest?.modelID
            ?? host?.selectedModelID
            ?? SteeredContainerLoader.localModelIDs().first
            ?? ChatService.availableModels.first?.id
            ?? ClaudePairedJudge.defaultModel
    }

    public func refreshResults(selecting preferredID: String? = nil) {
        guard let name = selectedName else {
            resultRuns = []
            selectedResultID = nil
            selectedResult = nil
            selectedResultBrowserItem = nil
            return
        }
        resultRuns = StudyResultStore.list(experimentName: name)
        if let preferredID, resultRuns.contains(where: { $0.id == preferredID }) {
            selectedResultID = preferredID
            loadSelectedResult()
            return
        }
        if let selectedResultID, !resultRuns.contains(where: { $0.id == selectedResultID }) {
            self.selectedResultID = nil
        }
        if selectedResultID == nil {
            selectedResultID = resultRuns.first(where: { $0.kind == .run })?.id
                ?? resultRuns.first?.id
        } else {
            loadSelectedResult()
        }
    }

    private func loadSelectedResult() {
        guard let id = selectedResultID,
            let item = resultRuns.first(where: { $0.id == id })
        else {
            selectedResult = nil
            selectedResultBrowserItem = nil
            return
        }
        selectedResult = StudyResultStore.detail(for: item)
        // F10: build the browser item here, once per selection — never in a
        // view body. Runs are immutable, so the memo may serve repeats.
        selectedResultBrowserItem = browserItemMemo.item(
            at: URL(filePath: item.path))
    }

}

/// Pure form helpers for the Optimizations sweep-spec editor — parsing, formatting,
/// save-time validation, and the instrument-file advisory preview.
/// Unit-tested; no UI. The instrument checks read the workspace strictly
/// through the ENGINE's own loaders, never a parallel parser.
public enum SweepSpecForm {
    /// Parse a comma-separated number list ("0.35, 0.5, 0.65"). Returns nil
    /// for an empty list or any non-finite / unparseable entry.
    public static func parseNumberList(_ text: String) -> [Double]? {
        let parts = text.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !parts.isEmpty else { return nil }
        var values: [Double] = []
        for part in parts {
            guard let value = Double(part), value.isFinite else { return nil }
            values.append(value)
        }
        return values
    }

    /// Locale-independent inverse of `parseNumberList` (Swift's default
    /// Double description always uses "." — never a locale decimal comma).
    public static func numberListText(_ values: [Double]) -> String {
        values.map { value in
            value == value.rounded() && abs(value) < 1e15
                ? String(Int(value))
                : "\(value)"
        }
        .joined(separator: ", ")
    }

    /// Structural problems with a spec that no engine could run — nil means
    /// sound. (Selection-criterion validity is `validateSelection`'s job.)
    public static func validate(_ spec: ExperimentManifest.SweepSpec) -> String? {
        if spec.layerFractions.isEmpty {
            return "layer fractions must name at least one layer"
        }
        if let bad = spec.layerFractions.first(where: { $0 < 0 || $0 > 1 }) {
            return "layer fractions are depth fractions in [0, 1] — got \(bad)"
        }
        if spec.alphas.isEmpty {
            return "alphas must name at least one steering strength"
        }
        if let bad = spec.alphas.first(where: { $0 <= 0 }) {
            return "alphas are residual-norm units > 0 (0 is the implied "
                + "baseline cell) — got \(bad)"
        }
        if spec.maxTokens <= 0 {
            return "max tokens must be positive"
        }
        if spec.devPromptsFile.trimmingCharacters(in: .whitespaces).isEmpty {
            return "dev prompts file is required"
        }
        if spec.batteryFile.trimmingCharacters(in: .whitespaces).isEmpty {
            return "capability battery file is required"
        }
        return nil
    }

    /// EVERY path-bearing instrument field in a sweep spec is
    /// workspace-relative when it names something inside this workspace —
    /// the portable shape the manifest pins and both engines resolve.
    /// Enforced at the write funnel (`ExperimentPanel.setSweepSpec`) for the
    /// same reason `ModelVariantStore.save` normalizes agents there: the
    /// writers keep re-learning this rule one field at a time (field gap
    /// 2026-08-07: the composer's free-text choice-prompts field let an
    /// absolute Mac path ride into the manifest, where it would resolve on
    /// this machine and die on any other). A path OUTSIDE the workspace
    /// passes through untouched: it is already broken for portability, and
    /// rewriting it would only hide that — the engine's own file checks
    /// name it honestly.
    public static func workspaceRelativeNormalized(
        _ spec: ExperimentManifest.SweepSpec
    ) -> ExperimentManifest.SweepSpec {
        var normalized = spec
        normalized.devPromptsFile = ArtifactIdentity.workspaceRelative(
            spec.devPromptsFile)
        normalized.batteryFile = ArtifactIdentity.workspaceRelative(
            spec.batteryFile)
        if var objective = normalized.selection?.objective {
            objective.choicePromptsFile = objective.choicePromptsFile.map(
                ArtifactIdentity.workspaceRelative)
            objective.choicePromptsFiles = objective.choicePromptsFiles?
                .mapValues(ArtifactIdentity.workspaceRelative)
            normalized.selection?.objective = objective
        }
        return normalized
    }

    public enum SelectionValidation: Equatable, Sendable {
        case valid
        /// Legal manifest data whose instrument has not landed on this
        /// engine: saving is allowed (declare-ahead is the point), the sweep
        /// itself refuses at start.
        case declaredAhead(metric: String)
        case invalid(String)
    }

    /// Save-time criterion validation, reusing `SweepSelectionRule.resolve`
    /// for the range checks so declaration and sweep start agree. Unknown
    /// metrics and out-of-range numbers refuse; known-but-unimplemented
    /// metrics save loudly as declared-ahead.
    public static func validateSelection(
        _ selection: ExperimentManifest.SweepSelection?
    ) -> SelectionValidation {
        let metric = selection?.objective?.metric ?? "markerDensity"
        guard SweepSelectionRule.knownMetrics.contains(metric) else {
            return .invalid(
                "unknown selection metric '\(metric)' — known metrics: "
                    + SweepSelectionRule.knownMetrics.joined(separator: ", "))
        }
        // Range-check through resolve() with an implemented metric substituted,
        // so a declared-ahead objective still gets its numbers validated.
        var probe = selection ?? ExperimentManifest.SweepSelection()
        probe.objective = .init(metric: SweepSelectionRule.implementedMetrics[0])
        do {
            _ = try SweepSelectionRule.resolve(probe)
        } catch let error as ExperimentError {
            return .invalid(error.reason)
        } catch {
            return .invalid("\(error)")
        }
        return SweepSelectionRule.implementedMetrics.contains(metric)
            ? .valid
            : .declaredAhead(metric: metric)
    }

    /// Save-time check of the objective's INSTRUMENT requirements, so a
    /// declaration that could never sweep is caught at declaration:
    /// judgeScore needs the manifest's rubric + judge pins; logprobShift
    /// needs a readable, parseable choice-prompt file. Returns the problem,
    /// or nil when the objective can arm. (The Claude-credential check stays
    /// a sweep-START gate — a credential is a runtime fact, not manifest
    /// data. Likewise a LOCAL judge with no model is LEGAL — it resolves to
    /// the study model at sweep start — and a local judge naming a NON-study
    /// model refuses at sweep start on the local engine, where which engine
    /// runs the sweep is known; neither is a save-time refusal. See
    /// `localJudgeDefaultNote` / `localJudgeSlotWarning` for the save-time
    /// messaging.)
    public static func validateObjectiveRequirements(
        _ selection: ExperimentManifest.SweepSelection?,
        manifest: ExperimentManifest,
        root: URL? = nil
    ) -> String? {
        switch selection?.objective?.metric ?? "markerDensity" {
        case "judgeScore":
            if manifest.judgeRubricFile == nil || manifest.judgeRubricHash == nil {
                return "judgeScore objective needs a pinned judge rubric "
                    + "(judgeRubricFile + judgeRubricHash) in the manifest — "
                    + "pin one under prompts/rubrics/ in Studies › Evaluation "
                    + "before declaring"
            }
            if (manifest.judges ?? []).isEmpty {
                return "judgeScore objective needs at least one judge pinned "
                    + "in manifest.judges"
            }
        case "logprobShift":
            do {
                // Through the SHARED resolver (review 2026-08-02, P1): the
                // per-concept map form validates here exactly as sweep
                // start will — coverage of every attached concept, no
                // unattached names, exactly one declaration shape, and
                // every file loading through the real loader.
                let criterion = try SweepSelectionRule.resolve(selection)
                _ = try SweepSelectionRule.resolveObjective(
                    criterion: criterion, spec: selection, manifest: manifest,
                    hasClaudeCredential: true, hasOpenRouterCredential: true,
                    root: root)
            } catch let error as ExperimentError {
                return error.reason
            } catch {
                return "\(error)"
            }
        default:
            break
        }
        return nil
    }

    /// Save-time NOTE (never a refusal) for judgeScore panels: names the
    /// local judges with no declared model and states the resolution rule —
    /// blank is LEGAL and means the STUDY model (cross-engine rule,
    /// 2026-07-08). Nil when every local judge declares a model.
    public static func localJudgeDefaultNote(
        judges: [ExperimentManifest.JudgeRef], studyModelID: String
    ) -> String? {
        let defaulted = judges
            .filter { judge in
                judge.kind == "local"
                    && (judge.model ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            .map { "'\($0.name)'" }
        guard !defaulted.isEmpty else { return nil }
        let names = defaulted.joined(separator: ", ")
        return "local judge \(names): no model set — blank is legal and "
            + "means the study model ('\(studyModelID)'), resolved and "
            + "logged at sweep start"
    }

    /// Save-time WARNING (never a refusal — the manifest may sweep on an
    /// engine with a second model slot): a LOCAL judge naming a model other
    /// than the study model will refuse at sweep start on the LOCAL engine,
    /// which holds one loaded model. Nil when no local judge does so.
    public static func localJudgeSlotWarning(
        judges: [ExperimentManifest.JudgeRef], studyModelID: String
    ) -> String? {
        for judge in judges where judge.kind == "local" {
            let model = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !model.isEmpty, model != studyModelID else { continue }
            return "local judge '\(judge.name)' uses model '\(model)', not "
                + "the study model '\(studyModelID)' — the local sweep holds "
                + "one loaded model and will refuse at start; use the study "
                + "model as judge or a claude judge"
        }
        return nil
    }

    /// What a logprobShift choice-prompts file would measure — the advisory
    /// the editor shows under the path field while the researcher types.
    public struct ChoicePromptsPreview: Equatable, Sendable {
        public var rowCount: Int
        /// Options-per-row range across the file (min == max when uniform).
        public var minOptions: Int
        public var maxOptions: Int
        /// Rows carrying an explicit non-empty "target" key, vs rows whose
        /// target defaults to options[0] — the engine's resolution rule.
        public var explicitTargetRows: Int
        public var defaultedTargetRows: Int

        public init(
            rowCount: Int, minOptions: Int, maxOptions: Int,
            explicitTargetRows: Int, defaultedTargetRows: Int
        ) {
            self.rowCount = rowCount
            self.minOptions = minOptions
            self.maxOptions = maxOptions
            self.explicitTargetRows = explicitTargetRows
            self.defaultedTargetRows = defaultedTargetRows
        }
    }

    public enum ChoicePromptsPreviewOutcome: Equatable, Sendable {
        /// No path entered yet — the editor renders nothing.
        case noFile
        case ok(ChoicePromptsPreview)
        /// The ENGINE loader's refusal reason, verbatim (missing file, no
        /// rows, malformed JSONL naming the line, a row with <2 options,
        /// a target outside its options).
        case problem(String)
    }

    /// Advisory preview of a logprobShift choice-prompts file for the
    /// authoring UI. NEVER a save gate — the editor renders the outcome as
    /// a caption and saving keeps its existing behavior (the sweep
    /// validates the instrument for real at start). Loading, validation,
    /// and path resolution go through `SweepSelectionRule.loadChoiceRows` —
    /// the engine's own loader — so preview and sweep can never disagree,
    /// and problems carry the engine's row-/line-indexed messages verbatim.
    public static func previewChoicePrompts(
        file: String?, root: URL? = nil
    ) -> ChoicePromptsPreviewOutcome {
        let trimmed = (file ?? "").trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return .noFile }
        let root = root ?? VectorCatalog.projectRoot
        do {
            let (rows, loadedFile, _) = try SweepSelectionRule.loadChoiceRows(
                file: trimmed, root: root)
            // Explicit-vs-defaulted targets need the RAW rows: a resolved
            // ChoiceRow whose target equals options[0] could be either.
            // Re-read through the SAME parser the loader used — reuse,
            // never a forked grammar.
            let url = SweepSelectionRule.choicePromptsURL(
                file: loadedFile, root: root)
            let data = try Data(contentsOf: url)
            let raw = try ExperimentTasks.parseTaskPrompts(data)
            let explicit = raw.filter { $0.target?.isEmpty == false }.count
            let optionCounts = rows.map { $0.options.count }
            return .ok(
                ChoicePromptsPreview(
                    rowCount: rows.count,
                    minOptions: optionCounts.min() ?? 0,
                    maxOptions: optionCounts.max() ?? 0,
                    explicitTargetRows: explicit,
                    defaultedTargetRows: rows.count - explicit))
        } catch let error as ExperimentError {
            return .problem(error.reason)
        } catch {
            return .problem("\(error)")
        }
    }

    /// One-line caption for an `.ok` preview (the view renders it verbatim;
    /// pure so the wording is unit-testable).
    public static func choicePromptsSummary(
        _ preview: ChoicePromptsPreview
    ) -> String {
        let options = preview.minOptions == preview.maxOptions
            ? "\(preview.minOptions)"
            : "\(preview.minOptions)–\(preview.maxOptions)"
        let rows = preview.rowCount == 1 ? "row" : "rows"
        return "\(preview.rowCount) \(rows) · \(options) options per row · "
            + "\(preview.explicitTargetRows) with explicit target, "
            + "\(preview.defaultedTargetRows) defaulting to options[0]"
    }
}

/// Optimization lifecycle derivation for the Optimizations surface strip:
/// Declared → Swept → Recommended → Promoted. Pure and unit-tested; the view
/// only renders the result.
public enum OptimizationLifecycle {
    public struct States: Equatable, Sendable {
        /// The manifest declares a sweep spec. nil = not knowable on this
        /// substrate (the server's experiment detail does not expose the spec).
        public var declared: Bool?
        public var swept: Bool
        public var recommended: Bool
        /// An agent artifact carries `promotion.experiment == this study`.
        /// nil = not derivable (the server's variant listing carries no
        /// promotion block).
        public var promoted: Bool?

        public init(declared: Bool?, swept: Bool, recommended: Bool, promoted: Bool?) {
            self.declared = declared
            self.swept = swept
            self.recommended = recommended
            self.promoted = promoted
        }
    }

    /// A stamped recommendation is itself proof a sweep ran, even if the run
    /// directory has since been pruned — `swept` folds that in.
    public static func derive(
        hasSweepSpec: Bool?,
        hasSweepRun: Bool,
        hasRecommendation: Bool,
        hasPromotedAgent: Bool?
    ) -> States {
        States(
            declared: hasSweepSpec,
            swept: hasSweepRun || hasRecommendation,
            recommended: hasRecommendation,
            promoted: hasPromotedAgent)
    }

    /// True when any artifact's birth certificate names this study.
    public static func hasPromotedAgent(
        experiment: String, in artifacts: [ModelVariantArtifact]
    ) -> Bool {
        artifacts.contains { $0.promotion?.experiment == experiment }
    }

    /// The one next action the strip points at — empty-state-with-affordance
    /// as a rule, not a caption.
    public static func nextStep(_ states: States) -> String {
        if states.declared == false {
            return "next: declare the optimization (sweep) spec below"
        }
        if !states.swept {
            return "next: optimize — run the declared sweep"
        }
        if !states.recommended {
            return "no recommendation — inspect the grid (constraints or the "
                + "matched-norm control refused every cell), adjust the "
                + "declared criterion, and re-optimize"
        }
        if states.promoted == false {
            return "next: create the agent from the winning cell"
        }
        if states.promoted == true {
            return "next: build a confirmation study in Studies "
                + "(Confirm agent stage)"
        }
        return "promotion state is not derivable from this server's listing — "
            + "check the Agents section's server list"
    }
}

public struct StudyRunListItem: Identifiable, Codable, Sendable, Equatable {
    public enum Kind: String, Codable, Sendable {
        case run
        case validate
        case evaluate
        case other
    }

    public var id: String { directoryName }
    public let directoryName: String
    public let path: String
    public let kind: Kind
    public let createdAt: String
    public let generationCount: Int
    public let hasReport: Bool
}

public struct StudyRunDetail: Codable, Sendable, Equatable {
    public let item: StudyRunListItem
    public let judgeArtifactDirectory: String?
    public let report: StudyRunReportView?
    public let validationReportText: String?
    public let pairedJudgeReport: PairedJudgeReportView?
    public let robustnessReports: [String: VariantRobustnessReport]
    public let generations: [StudyGenerationPreview]
    public let judgments: [StudyJudgePreview]
}

public struct LiveStudyGeneration: Codable, Sendable, Equatable {
    public let condition: String
    public let promptID: String
    public let prompt: String
    public let output: String
}

public struct LiveStudyJudgment: Codable, Sendable, Equatable {
    public let condition: String
    public let promptID: String
}

public struct PairedJudgeReportView: Codable, Sendable, Equatable {
    public struct Condition: Codable, Sendable, Equatable {
        public let name: String
        public let pairs: Int
        public let conditionWins: Int
        public let baselineWins: Int
        public let ties: Int
        public let meanConfidence: Double
        public let structuredSummaries: [String: StructuredFieldSummaryView]
    }

    public let sourceRunDirectory: String
    public let judgeModel: String
    public let conditions: [Condition]
}

public struct StructuredFieldSummaryView: Codable, Sendable, Equatable {
    public let count: Int
    public let numericMean: Double?
    public let trueCount: Int?
    public let falseCount: Int?
    public let stringCounts: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case count
        case numericMean = "numeric_mean"
        case trueCount = "true_count"
        case falseCount = "false_count"
        case stringCounts = "string_counts"
    }
}

public struct StudyRunReportView: Codable, Sendable, Equatable {
    public struct Condition: Codable, Sendable, Equatable {
        public let name: String
        public let generations: Int
        public let meanWordCount: Float
        public let meanDistinct2: Float
        public let meanMarkerDensity: [String: Float]
    }

    public let experiment: String
    public let promptCount: Int?
    public let conditionCount: Int?
    public let seedCount: Int?
    public let taskPromptsFile: String?
    public let conditions: [Condition]
}

public struct StudyGenerationPreview: Identifiable, Codable, Sendable, Equatable {
    public var id: String { "\(condition)-\(promptID)" }
    public let condition: String
    public let promptID: String
    public let prompt: String
    public let output: String
    public let wordCount: Int
    public let distinct2: Float
    public let markerDensity: [String: Float]
    public let truncated: Bool
}

public struct StudyJudgePreview: Identifiable, Codable, Sendable, Equatable {
    public var id: String { "\(condition)-\(promptID)-\(sampleIndex)" }
    public let condition: String
    /// The pair's sample cell (the cross-engine pairing join key half);
    /// 0 for greedy single-sample runs.
    public let sampleIndex: UInt64
    /// Seed provenance for the two sides of the pair (cross-engine keys);
    /// nil on rows loaded from legacy judgment files.
    public let baselineSeed: UInt64?
    public let variantSeed: UInt64?
    public let promptID: String
    public let prompt: String
    public let baselineWas: String
    public let conditionWas: String
    public let winner: String
    public let conditionResult: String
    public let confidence: Double
    public let briefReason: String
    public let aScores: [String: Int]?
    public let bScores: [String: Int]?
    public let structuredFields: [String: JSONValue]?
    public let rawJSON: String
}

public enum StudyResultStore {
    public static func list(experimentName: String) -> [StudyRunListItem] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: ExperimentStore.runsDirectory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }

        return entries.compactMap { url in
            guard
                let data = try? Data(contentsOf: url.appending(component: "experiment.json")),
                let manifest = try? JSONDecoder().decode(ExperimentManifest.self, from: data),
                manifest.name == experimentName
            else { return nil }
            let name = url.lastPathComponent
            let kind: StudyRunListItem.Kind =
                name.contains("-run") ? .run
                : name.contains("-validate") ? .validate
                : name.contains("-evaluate") ? .evaluate
                : .other
            let generationsURL = url.appending(component: "generations.jsonl")
            return StudyRunListItem(
                directoryName: name,
                path: url.path,
                kind: kind,
                createdAt: String(name.prefix(24)),
                generationCount: lineCount(generationsURL),
                hasReport: fm.fileExists(atPath: url.appending(component: "report.json").path))
        }
        .filter { $0.kind != .evaluate }
        .sorted { lhs, rhs in lhs.directoryName > rhs.directoryName }
    }

    public static func detail(for item: StudyRunListItem) -> StudyRunDetail {
        let url = URL(filePath: item.path)
        let judgeArtifactURL =
            item.kind == .run
            ? latestEvaluationDirectory(forSourceRun: item.path)
            : (item.kind == .evaluate ? url : nil)
        return StudyRunDetail(
            item: item,
            judgeArtifactDirectory: judgeArtifactURL?.path,
            report: loadRunReport(url.appending(component: "report.json")),
            validationReportText: loadValidationReport(url.appending(component: "report.json")),
            pairedJudgeReport: loadPairedJudgeReport(
                (judgeArtifactURL ?? url).appending(component: "judge-report.json")),
            robustnessReports: loadRobustnessReports(url.appending(component: "robustness-report.json")),
            generations: loadGenerations(url.appending(component: "generations.jsonl")),
            judgments: loadJudgments((judgeArtifactURL ?? url).appending(component: "judgments.jsonl")))
    }

    private static func loadRobustnessReports(_ url: URL) -> [String: VariantRobustnessReport] {
        guard let data = try? Data(contentsOf: url),
            let reports = try? JSONDecoder().decode([String: VariantRobustnessReport].self, from: data)
        else { return [:] }
        return reports
    }

    private struct RawReport: Decodable {
        struct Condition: Decodable {
            let generations: Int
            let meanWordCount: Float
            let meanDistinct2: Float
            let meanMarkerDensity: [String: Float]
        }
        let experiment: String
        let promptCount: Int?
        let conditionCount: Int?
        let seedCount: Int?
        let taskPromptsFile: String?
        let conditions: [String: Condition]?
    }

    private static func loadRunReport(_ url: URL) -> StudyRunReportView? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONDecoder().decode(RawReport.self, from: data),
            let conditions = raw.conditions
        else { return nil }
        return StudyRunReportView(
            experiment: raw.experiment,
            promptCount: raw.promptCount,
            conditionCount: raw.conditionCount,
            seedCount: raw.seedCount,
            taskPromptsFile: raw.taskPromptsFile,
            conditions: conditions.map { name, condition in
                StudyRunReportView.Condition(
                    name: name,
                    generations: condition.generations,
                    meanWordCount: condition.meanWordCount,
                    meanDistinct2: condition.meanDistinct2,
                    meanMarkerDensity: condition.meanMarkerDensity)
            }.sorted { $0.name < $1.name })
    }

    private static func loadValidationReport(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            raw["validation"] != nil
        else { return nil }
        let pretty = (try? JSONSerialization.data(withJSONObject: raw, options: [.prettyPrinted, .sortedKeys]))
            ?? data
        return String(decoding: pretty, as: UTF8.self)
    }

    private struct RawJudgeReport: Decodable {
        struct Condition: Decodable {
            let pairs: Int
            let conditionWins: Int
            let baselineWins: Int
            let ties: Int
            let meanConfidence: Double
            let structuredSummaries: [String: StructuredFieldSummaryView]?

            enum CodingKeys: String, CodingKey {
                case pairs
                case conditionWins
                case baselineWins
                case ties
                case meanConfidence
                case structuredSummaries = "structuredSummaries"
            }
        }

        let sourceRunDirectory: String
        let judgeModel: String
        let conditions: [String: Condition]
    }

    private static func loadPairedJudgeReport(_ url: URL) -> PairedJudgeReportView? {
        guard let data = try? Data(contentsOf: url),
            let raw = try? JSONDecoder().decode(RawJudgeReport.self, from: data)
        else { return nil }
        return PairedJudgeReportView(
            sourceRunDirectory: raw.sourceRunDirectory,
            judgeModel: raw.judgeModel,
            conditions: raw.conditions.map { name, condition in
                PairedJudgeReportView.Condition(
                    name: name,
                    pairs: condition.pairs,
                    conditionWins: condition.conditionWins,
                    baselineWins: condition.baselineWins,
                    ties: condition.ties,
                    meanConfidence: condition.meanConfidence,
                    structuredSummaries: condition.structuredSummaries ?? [:])
            }.sorted { $0.name < $1.name })
    }

    private static func latestEvaluationDirectory(forSourceRun sourceRunPath: String) -> URL? {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: ExperimentStore.runsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey])
        else { return nil }

        return entries
            .filter { $0.lastPathComponent.contains("-evaluate") }
            .filter { url in
                guard
                    let data = try? Data(contentsOf: url.appending(component: "judge-report.json")),
                    let raw = try? JSONDecoder().decode(RawJudgeReport.self, from: data)
                else { return false }
                return raw.sourceRunDirectory == sourceRunPath
            }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .first
    }

    private struct RawGeneration: Decodable {
        let condition: String
        let promptID: String
        let prompt: String
        let output: String
        let wordCount: Int
        let distinct2: Float
        let markerDensity: [String: Float]
    }

    private static func loadGenerations(_ url: URL) -> [StudyGenerationPreview] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").prefix(80).compactMap { line in
            guard
                let raw = try? JSONDecoder().decode(
                    RawGeneration.self, from: Data(line.utf8))
            else { return nil }
            let limit = 1_800
            let truncated = raw.output.count > limit
            let output = truncated ? String(raw.output.prefix(limit)) : raw.output
            return StudyGenerationPreview(
                condition: raw.condition,
                promptID: raw.promptID,
                prompt: raw.prompt,
                output: output,
                wordCount: raw.wordCount,
                distinct2: raw.distinct2,
                markerDensity: raw.markerDensity,
                truncated: truncated)
        }
    }

    private struct RawJudgment: Decodable {
        let condition: String
        /// New rows carry the pair cell + both sides' seeds; legacy rows
        /// (pre sample-cell join) carried a single "seed". All optional so
        /// both generations of judgment files stay viewable.
        let sampleIndex: UInt64?
        let baselineSeed: UInt64?
        let variantSeed: UInt64?
        let seed: UInt64?
        let promptID: String
        let prompt: String
        let baselineWas: String
        let conditionWas: String
        let judgment: PairedJudgeResponse
        let conditionResult: String
    }

    private static func loadJudgments(_ url: URL) -> [StudyJudgePreview] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").prefix(200).compactMap { line in
            let data = Data(line.utf8)
            guard let raw = try? JSONDecoder().decode(RawJudgment.self, from: data) else {
                return nil
            }
            let pretty =
                (try? JSONSerialization.jsonObject(with: data))
                .flatMap {
                    try? JSONSerialization.data(
                        withJSONObject: $0, options: [.prettyPrinted, .sortedKeys])
                }
                .map { String(decoding: $0, as: UTF8.self) }
                ?? String(line)
            return StudyJudgePreview(
                condition: raw.condition,
                // Legacy rows keyed by seed: reuse it as the row id's cell
                // so multi-seed legacy files keep distinct preview rows.
                sampleIndex: raw.sampleIndex ?? raw.seed ?? 0,
                baselineSeed: raw.baselineSeed,
                variantSeed: raw.variantSeed,
                promptID: raw.promptID,
                prompt: raw.prompt,
                baselineWas: raw.baselineWas,
                conditionWas: raw.conditionWas,
                winner: raw.judgment.winner,
                conditionResult: raw.conditionResult,
                confidence: raw.judgment.confidence,
                briefReason: raw.judgment.briefReason,
                aScores: raw.judgment.aScores,
                bScores: raw.judgment.bScores,
                structuredFields: raw.judgment.structuredFields,
                rawJSON: pretty)
        }
    }

    private static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(separator: "\n").count
    }
}

// MARK: - File-picker pin + tabular import (Usability Plan Phase 3, 12–13)

/// New methods only (kill typed paths and raw JSONL from the critical
/// path): the workspace file picker's pin-on-selection, and the two
/// tabular-import flows. Each goes through the SAME validating pin path as
/// the typed-path route — nothing here weakens a pin.
extension ExperimentPanel {

    /// Pin a picker-chosen task-prompts file: set the path field, pin its
    /// hash through `ExperimentStore.pinTaskPrompts` (the run loop's own
    /// parser re-checks the records), persist, and load it into the
    /// editor. No file is written — choosing an existing JSONL pins it as
    /// it is.
    public func pinChosenTaskPromptsFile(_ relativePath: String) {
        guard var manifest = selected, manifest.status == .draft else {
            note(
                "select a draft study first — choosing a prompts file pins "
                    + "it into the draft manifest",
                severity: .info)
            return
        }
        do {
            let hash = try ExperimentStore.pinTaskPrompts(
                relativePath, into: &manifest)
            try ExperimentStore.save(manifest)
            taskPromptsFile = relativePath
            refresh()
            loadTaskPrompts()
            note(
                "pinned task prompts \(relativePath) @ \(hash.prefix(12))…",
                severity: .success)
        } catch {
            note(
                "Couldn't pin the chosen prompts file — it must be JSONL "
                    + "the run loop can parse (one {\"text\": …} object per "
                    + "line; Import table… converts spreadsheets). "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// The Import table… flow for task prompts: convert the mapped table
    /// to JSONL, write it to the study's task-prompts destination (never
    /// overwriting differing bytes), pin, and reload the editor. Returns
    /// the plain problem to show in the mapping sheet, or nil when the
    /// import landed (dismiss). The manifest save happens INSIDE the
    /// import's transaction (its `persist` step), so a save refusal — the
    /// study frozen on disk since this panel loaded its draft copy — rolls
    /// back the written file instead of orphaning it.
    public func importTaskPromptsTable(
        table: TabularImport.Table, mapping: [String: String]
    ) -> String? {
        guard var manifest = selected, manifest.status == .draft else {
            return "select a draft study first — import writes the file and "
                + "pins its hash into the draft manifest"
        }
        do {
            let result = try TabularImport.importTaskPrompts(
                table: table, mapping: mapping, manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            taskPromptsFile = result.file
            refresh()
            loadTaskPrompts()
            taskPromptsStatus =
                "imported \(result.recordCount) row\(result.recordCount == 1 ? "" : "s")"
                + " → \(result.file), pinned @ \(result.hash.prefix(12))…"
            note(
                "imported task-prompt table and pinned its hash",
                severity: .success)
            return nil
        } catch {
            return "\(error)"
        }
    }

    /// The Import table… flow for the human baseline: convert the mapped
    /// table to the analyze loader's CSV, write it to
    /// prompts/baselines/<study>-human-baseline.csv (never overwriting
    /// differing bytes), and pin through the shape-validating
    /// `pinHumanBaseline`. Returns the plain problem, or nil on success.
    /// Unlike the task-prompts flow there is NO panel-side save here:
    /// `pinHumanBaseline` persists internally (via `updateDraft`), inside
    /// the import's rollback — the whole import is already atomic.
    public func importHumanBaselineTable(
        table: TabularImport.Table, mapping: [String: String]
    ) -> String? {
        guard let name = selectedName else {
            return "select a study first — the baseline pins into the "
                + "selected draft's manifest"
        }
        do {
            let pinned = try TabularImport.importHumanBaseline(
                table: table, mapping: mapping, experimentName: name)
            humanBaselinePathField = pinned.path
            refresh()
            note(
                "imported human baseline → \(pinned.path), pinned @ "
                    + "\(pinned.hash.prefix(12))…",
                severity: .success)
            return nil
        } catch {
            return "\(error)"
        }
    }
}
