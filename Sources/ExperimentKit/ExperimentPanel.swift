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
    public let draft = StudyDraftState()
    public let results = StudyResultsState()
    public let localJobs = StudyLocalJobController()
    public let remoteJobs = StudyRemoteJobController()
    public let submission = StudySubmissionOptions()
    public let management: StudyManagementController
    public let freezeCoordinator = StudyFreezeController()
    public let bundleSubmission: StudyBundleSubmissionController
    public let serverExecution: StudyServerJobCoordinator
    public let pipelines = StudyPipelineController()

    public internal(set) weak var host: ChatService?

    /// Workspace model choices passed explicitly to the management owner.
    public var studyCreationContext: StudyCreationContext? {
        guard let host else { return nil }
        return StudyCreationContext(
            workspaceDefaultModelID: host.workspaceSelectedModelID ?? host.selectedModelID,
            modelOptions: modelOptions)
    }

    /// THE study-type setter (2026-07-19 second pass): one control answers
    /// "what kind of study is this?". Sets the view classification AND —
    /// on drafts — PERSISTS the declared type into the manifest
    /// immediately (durable across selection changes; an empty comparison
    /// no longer re-derives back to Concept study), keeping the
    /// engine-facing studyKind consistent. Frozen studies get the view
    /// lens only.
    public func setStudyType(_ type: StudyIntent) {
        studyFocusOverride = type
        guard let manifest = management.selected, manifest.status == .draft else { return }
        draft.studyKind = type.mappedKind
        do {
            try management.editReviewed(named: manifest.name) { reviewedName in
                try ExperimentStore.setStudyType(type, experimentName: reviewedName)
            }
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
            management.selectedName = candidate
            studyFocusOverride = .conceptStudy
            // Preselect the agent this screen study promoted when it is in
            // the library (the caller knows the optimization; the birth
            // certificate names the experiment).
            if let promoted = confirmableAgents.first(where: {
                $0.artifact.promotion?.experiment == screenName
            }) {
                draft.confirmAgentID = promoted.id
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
    public typealias FormField = StudyDraftState.FormField

    /// Refuse a form action: record the message for inline rendering AND
    /// speak it through the normal notice path.
    public func refuse(
        _ field: FormField, _ message: String,
        severity: PanelNotice.Severity = .error
    ) {
        draft.formErrors[field] = message
        note(message, severity: severity)
    }

    /// Clear a form's inline refusal — on success, or when the researcher
    /// edits the inputs that caused it.
    public func clearFormError(_ field: FormField) {
        draft.clearFormError(field)
    }

    public private(set) var violations: [String] = []

    /// One-shot request from a cross-link (e.g. Optimizations' "Submit Bundle:
    /// sweep") that the Studies view open its Run-on-Server disclosure so the
    /// preconfigured verb/study are visible, not hidden behind a collapsed
    /// group. The Studies view consumes (and clears) it on appear.
    public var pendingRevealRemoteControls = false

    /// The active server's `runs/` listing (read-only browse; refreshed on
    /// demand, cleared when no server workspace is active).
    public private(set) var remoteRuns: [RemoteRunRecord] = []

    /// Server jobs submitted from THIS panel this session (run-verb jobs and
    /// bundle submissions), id-first so a researcher can always copy the id
    /// and reconnect later — jobs persist on the server across app restarts.
    public typealias RecentServerJob = StudyRemoteJobController.RecentServerJob

    /// The in-flight server experiment job (durable, server-side): drives the
    /// visible Cancel control next to the run status. Cleared on terminal
    /// state; a timed-out follow keeps it set — the job is still running and
    /// must stay cancellable.
    public typealias ActiveServerJob = StudyRemoteJobController.ActiveServerJob

    /// Cancel the in-flight Optimize. Server route: cancel the tracked
    /// sweep job (its own slot — never another flow's job id). Local route:
    /// raise the flag the sweep observes between generations; partial rows
    /// stay in the run directory either way.
    public func cancelSweep() async {
        if let job = remoteJobs.activeSweepJob {
            loadStoredRemoteToken()
            guard let client = remoteClient else {
                note("invalid server URL", severity: .error)
                return
            }
            do {
                let client = try remoteJobs.clientForJob(job.id, connected: client)
                try await client.cancelJob(job.id)
                note("cancel requested for server sweep job \(job.id) "
                    + "('\(job.study)') — cancelling…", severity: .warning)
            } catch {
                note("cancel failed for sweep job \(job.id): \(error)", severity: .error)
            }
            return
        }
        localJobs.cancelSweep()
    }

    // MARK: Local-operation cancellation (App gap A1)

    /// Stop the in-flight LOCAL extraction after the current concept.
    public func cancelExtract() {
        localJobs.cancelExtract()
    }

    /// Stop the in-flight LOCAL study run after the current generation.
    public func cancelStudyRun() {
        localJobs.cancelStudyRun()
    }

    /// Stop the in-flight LOCAL validation after the current unit of work.
    public func cancelValidation() {
        localJobs.cancelValidation()
    }

    /// Stop the in-flight LOCAL paired-judge evaluation after the current
    /// judgment.
    public func cancelPairedJudge() {
        localJobs.cancelPairedJudge()
    }

    /// Cancel the in-flight server experiment job (the durable job keeps its
    /// cancelled record server-side; the follow loop unwinds on the terminal
    /// state).
    public func cancelActiveServerJob() async {
        loadStoredRemoteToken()
        await remoteJobs.cancelActiveServerJob(client: remoteClient)
    }

    // Display-pane live log (same affordance LoRA training and vector builds
    // use): panel-initiated runs mirror their CLI-style progress lines into
    // the chat transcript so long runs are observable outside this panel.

    /// The workspace-global connection (URL, token, Keychain persistence)
    /// lives on the shared `ClusterConnectionStore`; the panel reaches it
    /// through its host so run-on-server always targets the same server the
    /// rest of the app is connected to.
    public var cluster: ClusterConnectionStore? { host?.cluster }

    /// Surface context for focused controllers. Reading it never prompts for credentials.
    public var operationEnvironment: StudyOperationEnvironment {
        StudyOperationEnvironment(
            current: { [weak self] in
                guard let self else { return nil }
                return StudyOperationContext(
                    workspaceRoot: ExperimentStore.workspaceRoot,
                    selectedName: self.management.selectedName,
                    selectedIsDraft: self.management.selected?.status == .draft,
                    serverURL: self.cluster?.serverURL, isServer: self.isServerWorkspace,
                    pairing: self.cluster?.activeServerPairing,
                    substrate: self.cluster?.substrateLabel ?? "server",
                    capabilities: self.cluster?.capabilities, client: self.cluster?.client,
                    hasDisplay: self.host != nil, serverOrigin: self.cluster?.evidenceImportOrigin)
            },
            connect: { [weak self] in
                self?.cluster?.loadStoredToken()
                return self?.cluster?.client
            })
    }

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
    private var serverResidencyContext: StudyOperationContext?
    private var residencyGeneration = UUID()
    private var remoteRunsGeneration = UUID()
    private var remoteOptimizationsGeneration = UUID()
    private var awaitingJudgmentGeneration = UUID()
    private var awaitingJudgmentContext: StudyOperationContext?

    /// Refresh `serverHasSelectedStudy` for the current selection, cached per
    /// (study, server workspace). The view calls this when the selection or
    /// the active workspace changes.
    public func refreshServerResidency() async {
        let environment = operationEnvironment
        let generation = UUID()
        residencyGeneration = generation
        guard let context = environment.current(), context.isServer, let name = context.selectedName else {
            serverHasSelectedStudy = nil
            serverResidencyContext = nil
            return
        }
        if let previous = serverResidencyContext, previous.matches(context, selection: true),
            serverHasSelectedStudy != nil { return }
        serverResidencyContext = context
        guard let client = context.client else { serverHasSelectedStudy = nil; return }
        let names = try? await client.experimentNames()
        guard generation == residencyGeneration, environment.isCurrent(context, selection: true) else { return }
        guard let names else { serverHasSelectedStudy = nil; return }
        noteServerResidency(names.contains(name))
    }

    /// Records a residency answer (from the refresh above or the run-path
    /// backstop). On the transition to "not on the server", preselect the
    /// bundle controls for what the user actually meant — run, for real —
    /// so Submit Bundle does what Run Server Copy could not.
    private func noteServerResidency(_ resident: Bool) {
        let wasMissing = serverHasSelectedStudy == false
        serverHasSelectedStudy = resident
        if !resident, !wasMissing {
            submission.remoteVerb = "run"
            submission.remoteDryRun = false
        }
    }

    // MARK: Pure label/status helpers (unit-tested)

    /// Dynamic Submit Bundle label: the button says what it will do.
    public static func bundleSubmitLabel(verb: String, dryRun: Bool) -> String {
        dryRun ? "Submit Bundle: \(verb) (dry run)" : "Submit Bundle: \(verb)"
    }

    public var submitBundleButtonLabel: String {
        Self.bundleSubmitLabel(verb: submission.remoteVerb, dryRun: submission.remoteDryRun)
    }

    /// Post-submission status: what was submitted and where to watch it.
    public static func bundleSubmittedStatus(
        study: String, verb: String, dryRun: Bool, substrate: String, jobID: String
    ) -> String {
        StudySubmissionPresentation.bundleSubmittedStatus(study: study, verb: verb, dryRun: dryRun, substrate: substrate, jobID: jobID)
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
        localJobs.beginDisplayLog(title: title, initialLine: initialLine)
    }

    private func appendDisplayLog(_ line: String) {
        localJobs.appendDisplayLog(line)
    }

    private func endDisplayLog(_ finalLine: String? = nil) {
        localJobs.endDisplayLog(finalLine)
    }

    /// Test seams for the ad-hoc judge picker: the KEY one is a presence
    /// boolean, so no test ever holds, reads, or renders a credential.
    @ObservationIgnored public var judgeKeyPresenceOverrideForTesting: Bool?
    @ObservationIgnored
    public var judgeCapabilityOverrideForTesting: JudgeModelOffers.CapabilityCheck?
    /// Is-installed seam, so a test's picker never depends on which weights
    /// this developer's Mac happens to hold.
    @ObservationIgnored
    public var judgeInstalledOverrideForTesting: JudgeModelOffers.InstalledCheck?
    /// Claude key PRESENCE seam for the readiness gate — a boolean, never a
    /// credential.
    @ObservationIgnored public var claudeKeyPresenceOverrideForTesting: Bool?
    @ObservationIgnored public var localModelScanOverrideForTesting: [String]?

    /// What the ad-hoc judge picker offers. Same composition as the
    /// Robustness Check's — one rule, two panes: the cache scan is
    /// capability-filtered, curated entries are checked for PRESENCE (a model
    /// tier is a candidate, not an inventory) and flagged when absent, and a
    /// stored selection that fails is flagged rather than dropped.
    public var judgeModelOffers: JudgeModelOffers.Offers {
        var candidates: [JudgeModelOffers.Candidate] = []
        if let model = management.selected?.modelID { candidates.append(.cached(model)) }
        candidates.append(.cached(draft.studyBaseModelID))
        if let model = host?.selectedModelID { candidates.append(.cached(model)) }
        let scanned = localModelScanOverrideForTesting
            ?? SteeredContainerLoader.localModelIDs()
        candidates += scanned.map { .cached($0) }
        candidates += ChatService.availableModels.map { .curated($0.id) }
        candidates.append(.curated(ClaudePairedJudge.defaultModel))
        return JudgeModelOffers.compose(
            selected: draft.judgeModel,
            candidates: candidates,
            openRouterCredentialState: judgeKeyPresenceOverrideForTesting.map(CredentialObservation.State.init(present:))
                ?? CredentialObservation.openRouter(),
            capability: judgeCapabilityOverrideForTesting
                ?? JudgeModelOffers.liveCapability,
            installed: judgeInstalledOverrideForTesting
                ?? JudgeModelOffers.liveInstalled)
    }

    /// The flat id list the web surface's `<select>` consumes — the same
    /// offers, flattened, so the wire contract is unchanged by the filter.
    public var judgeModelOptions: [String] {
        let offers = judgeModelOffers
        return (offers.models + offers.openRouter).map(\.id)
    }

    /// The ad-hoc judge selection parsed — what the pane keys its OpenRouter
    /// fields off, so the pane never re-implements the spelling.
    public var adHocJudgeSelection: JudgeModelSpelling.Selection? {
        JudgeModelSpelling.parse(draft.judgeModel)
    }

    public var adHocOpenRouterModel: String {
        get {
            switch adHocJudgeSelection {
            case .openRouter(let model, _), .openRouterUnpinned(let model): model
            default: ""
            }
        }
        set {
            draft.judgeModel = JudgeModelSpelling.spellOpenRouter(
                model: newValue, provider: adHocOpenRouterProvider)
        }
    }

    public var adHocOpenRouterProvider: String {
        get {
            if case .openRouter(_, let provider) = adHocJudgeSelection {
                return provider
            }
            return ""
        }
        set {
            draft.judgeModel = JudgeModelSpelling.spellOpenRouter(
                model: adHocOpenRouterModel, provider: newValue)
        }
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
        let index = draft.judges.count + 1
        draft.judges.append(
            .init(
                name: "judge-\(index)",
                kind: index == 1 ? "openrouter" : "local",
                model: nil))
    }

    public func removeJudge(at index: Int) {
        guard draft.judges.indices.contains(index) else { return }
        draft.judges.remove(at: index)
        // The kind stash is keyed by row index — shift the entries above
        // the removed row down so each remaining row keeps ITS stash.
        var shifted: [Int: [String: JudgeKindStash]] = [:]
        for (row, stash) in draft.judgeKindStashes where row != index {
            shifted[row > index ? row - 1 : row] = stash
        }
        draft.judgeKindStashes = shifted
    }

    /// One kind's field set for one judge row, held while the row wears a
    /// different kind. See `judgeKindStashes`.
    public typealias JudgeKindStash = StudyDraftState.JudgeKindStash

    /// The kind picker's write path (field bug 2026-08-07). Stashes the
    /// outgoing kind's fields and restores any previously entered for the
    /// incoming kind this session, so the row's live fields are always the
    /// CURRENT kind's own — a kind never renders (or saves) another kind's
    /// values, and nothing the researcher typed is lost to a toggle.
    public func setJudgeKind(at index: Int, to newKind: String) {
        draft.setJudgeKind(at: index, to: newKind)
    }

    public init() {
        management = StudyManagementController(draft: draft)
        bundleSubmission = StudyBundleSubmissionController(jobs: remoteJobs)
        serverExecution = StudyServerJobCoordinator(jobs: remoteJobs)
        serverExecution.presentation = StudyServerJobPresentation(
            residency: { [weak self] name, resident in
                guard let self, self.management.selectedName == name else { return }
                self.noteServerResidency(resident)
            },
            refreshResidency: { [weak self] in
                self?.serverResidencyContext = nil
                await self?.refreshServerResidency()
            },
            refreshRuns: { [weak self] in await self?.refreshRemoteRuns() },
            importEvidence: { [weak self] id in await self?.importEvidence(fromJobID: id) },
            refresh: { [weak self] in self?.refresh() })
        pipelines.presentation = StudyPipelinePresentation(
            note: { [weak self] text, severity in self?.note(text, severity: severity) },
            refresh: { [weak self] in self?.refresh() })
        management.presentation = StudyManagementPresentation(
            note: { [weak self] text, severity in self?.note(text, severity: severity) },
            selectionChanged: { [weak self] in self?.managementSelectionChanged() },
            refreshed: { [weak self] in self?.refreshStudyDetails() })
        freezeCoordinator.presentation = StudyFreezePresentation(
            note: { [weak self] text, severity in self?.note(text, severity: severity) },
            refresh: { [weak self] in self?.refresh() },
            residency: { [weak self] name, resident in
                guard let self, self.management.selectedName == name else { return }
                self.noteServerResidency(resident)
            })
        let presentation = StudyJobPresentation(
            note: { [weak self] text, severity in self?.note(text, severity: severity) },
            status: { [weak self] text in self?.status = text },
            refresh: { [weak self] in self?.refresh() },
            selectResult: { [weak self] study, id in
                guard let self, self.management.selectedName == study else { return }
                self.refreshResults(selecting: id)
            },
            startLog: { [weak self] title, line in self?.host?.startLiveLog(title: title, initialLine: line) },
            updateLog: { [weak self] id, title, lines in self?.host?.updateLiveLog(id: id, title: title, lines: lines) })
        localJobs.presentation = presentation
        remoteJobs.presentation = presentation
        refresh()
    }

    private func managementSelectionChanged() {
        results.selectedResultID = nil
        results.selectedResult = nil
        results.selectedResultBrowserItem = nil
        freezeCoordinator.resetSelection()
        // Pipeline listings belong to the previous selection —
        // clear immediately, refresh in the background (sixth
        // round: stale chains must never render under the wrong
        // study).
        pipelines.resetSelection()
        // The focus override is per-study view state.
        studyFocusOverride = nil
        refresh()
        syncDraftFieldsFromSelection(force: true)
        Task { await pipelines.refresh(in: operationEnvironment) }
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
        guard var manifest = management.selected, manifest.status == .draft else { return false }
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
            try management.persistReviewedDraft(manifest)
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

    /// The Seats section's model: the seats of the scenario this study is
    /// running (or about to), and who occupies each one.
    ///
    /// Nil for anything that is not a multi-agent study, and for a multi-agent
    /// study with no scenario chosen yet — there is nothing to cast until a
    /// scenario names some seats.
    public var seatCasting: SeatCasting.State? {
        guard var manifest = management.selected else { return nil }
        // The editor's type wins over the stored one for READING, so the
        // section appears the moment the type picker says multi-agent rather
        // than one save later.
        manifest.studyKind = draft.studyKind
        let record = draft.selectedMultiAgentScenarioID.flatMap { id in
            multiAgentScenarioOptions.first { $0.id == id }
        }
        return SeatCasting.state(
            of: manifest,
            selected: record.map { ($0.scenario, relativeProjectPath(for: $0.url)) },
            overlay: draft.seatCastingEdits)
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
        guard let model = management.selected?.modelID, !model.isEmpty else { return [] }
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
        draft.seatCastingEdits[seat] = seatOccupant(forAgentID: agentID)
    }

    /// Why the seat casting cannot be saved right now, or nil.
    public func seatCastingRefusal(_ state: SeatCasting.State) -> String? {
        guard let manifest = management.selected else { return "no study selected" }
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
        guard let manifest = management.selected, let state = seatCasting else { return }
        if let refusal = seatCastingRefusal(state) {
            note("Couldn't save the seats — " + refusal, severity: .error)
            return
        }
        do {
            let reviewed = try management.reviewedDraft(named: manifest.name)
            let saved = try StudyPanelAuthoring.saveAssignment(state.assignment, semantic: state.semantic,
                semanticPath: state.semanticPath, reviewed: reviewed)
            management.acceptAuthoringResult(saved)
            let compiledPath = saved.manifest.multiAgentScenarioPath ?? ""
            draft.seatCastingEdits = [:]
            refresh()
            let cast = state.assignment.ordered.filter { $0 != .baseline }.count
            note(
                "cast \(state.seats.count) seat(s) (\(cast) steered) and pinned "
                    + "\(compiledPath) — this study now runs that compiled "
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
        guard let manifest = management.selected, let state = seatCasting else { return }
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
            let source = try management.reviewEditorDesignSource()
            let saved = try StudyDesignSaving.create(from: source)
            let mint = StudyTemplateStore.Mint(template: saved.snapshot.template, hash: StudyTemplateStore.hash(saved.snapshot.template),
                minted: saved.created, divergedFrom: saved.created ? manifest.templateProvenance?.template : nil, warnings: saved.warnings)
            management.refreshTemplates()
            for warning in mint.warnings { note(warning, severity: .warning) }
            management.designs.templateInstantiationInvitation = TemplateInstantiationInvitation(
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
        append(draft.studyBaseModelID)
        append(management.selected?.modelID)
        append(host?.selectedModelID)
        for model in ChatService.availableModels.map(\.id) { append(model) }
        for variant in ModelVariantStore.scan() { append(variant.artifact.baseModelID) }
        return options
    }

    public var availableVariantsForStudy: [ModelVariantRecord] {
        let attached = Set(management.selected?.variantConditions.map(\.artifactPath) ?? [])
        return ModelVariantStore.scan().filter {
            $0.artifact.baseModelID == draft.studyBaseModelID
                && !attached.contains(ModelVariantStore.relativePath(for: $0))
        }
    }

    /// Concepts on disk not yet attached to the selected experiment.
    public var attachableConcepts: [String] {
        let attached = Set(management.selected?.concepts.map(\.name) ?? [])
        return VectorCatalog.conceptNames().filter { !attached.contains($0) }
    }

    /// Sweep-recommended conditions carrying selection provenance — the
    /// promotable cells (screening's outputs, confirmation's inputs).
    public var promotableRecommendations: [ExperimentManifest.Condition] {
        management.selected?.conditions.filter { $0.selection != nil } ?? []
    }

    /// Agents eligible for a confirmation study on the selected draft:
    /// vector-only, single-injection, matching the study base model —
    /// exactly what `ConfirmationStudy.attach` will accept. Sweep-promoted
    /// agents sort first (they are what confirmation is FOR; hand-created
    /// ones stay legal and get the freeze advisory).
    public var confirmableAgents: [ModelVariantRecord] {
        ModelVariantStore.scan()
            .filter {
                $0.artifact.baseModelID == draft.studyBaseModelID
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
        guard let manifest = management.selected,
              let agent = manifest.perturbationPolicy?.sourceAgent.name
        else { return [] }
        return manifest.conditions.filter {
            ConfirmationStudy.isGeneratedName($0.name, agent: agent)
        }
    }

    /// Attach the declared perturbation policy to the selected draft — same
    /// code path as `steerlab-cli experiment confirm`.
    public func attachPerturbations() {
        guard let name = management.selectedName else { return }
        guard let record = confirmableAgents.first(where: { $0.id == draft.confirmAgentID })
        else {
            note("select an agent to confirm", severity: .info)
            return
        }
        let deltas = draft.confirmDeltasText
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
                includeControl: draft.confirmIncludeControl,
                log: { _ in })
            let generated = manifest.conditions.filter {
                ConfirmationStudy.isGeneratedName(
                    $0.name, agent: record.artifact.name)
            }
            note("attached perturbation policy for '\(record.artifact.name)' "
                + "— \(generated.count) conditions (anchor, ±δ"
                + (draft.confirmIncludeControl ? ", matched-norm control)" : ")"), severity: .success)
            refresh()
        } catch {
            note("confirm failed: \(error)", severity: .error)
        }
    }

    /// Mint an agent (variant artifact) from the concept's sweep-selected
    /// cell — same code path as `steerlab-cli experiment promote`.
    public func promoteRecommended(concept: String) {
        guard let name = management.selectedName else { return }
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
        return management.experiments.first { $0.name == experimentName }
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
    ///
    /// The write itself is the two fully-gated store verbs the CLI and the
    /// HTTP route use — `ExperimentStore.setSweepGrid` for the grid block,
    /// `setSweepSelection` for the rule — the same routing that closed the
    /// Studies remove button's divergence (`detachConcept`). Before that,
    /// this panel could save `--alphas 0.1, 0.05` where the CLI refuses it
    /// (`sweepGridRule`): a grid the sweep would sort into something other
    /// than its declaration. The selection is validated BEFORE the grid
    /// write so a refusal on either half leaves the manifest untouched.
    @discardableResult
    public func setSweepSpec(
        _ spec: ExperimentManifest.SweepSpec, reviewed: DraftAuthoringSnapshot,
        onSaved: ((DraftAuthoringSnapshot) -> Void)? = nil
    ) -> Bool {
        // Normalize BEFORE validating, so what is checked is what is
        // written: an absolute instrument path inside the workspace becomes
        // the portable workspace-relative form (both declare flows — the
        // composer and the Optimizations editor — funnel through here).
        let name = reviewed.manifest.name
        let spec = SweepSpecForm.workspaceRelativeNormalized(spec)
        do {
            let manifest = reviewed.manifest
            guard manifest.status == .draft else {
                refuse(
                    .sweepSpec,
                    "'\(name)' is \(manifest.status.rawValue) — the sweep "
                        + "spec is pinned manifest data; duplicate the study to "
                        + "change it", severity: .warning)
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
            let (grid, saved) = try DraftAuthoringTransaction.perform(reviewed: reviewed) { name in
                let grid = try ExperimentStore.setSweepGrid(
                    experimentName: name, layerFractions: spec.layerFractions,
                    alphas: spec.alphas, devPromptsFile: spec.devPromptsFile,
                    batteryFile: spec.batteryFile, maxTokens: spec.maxTokens,
                    selectionUpdate: .some(spec.selection))
                return (grid, try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: name))
            }
            management.acceptAuthoringResult(saved)
            onSaved?(saved)
            refresh()
            clearFormError(.sweepSpec)
            if case .declaredAhead(let metric) = outcome {
                note("declared sweep spec on '\(name)' — objective "
                    + "'\(metric)' is not implemented on this engine: the "
                    + "sweep will REFUSE at start (declaring ahead is allowed)", severity: .warning)
            } else {
                note("declared sweep spec on '\(name)' "
                    + "(\(Self.gridDeclarationSummary(grid)))", severity: .success)
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
        } catch let error as ExperimentError {
            // A store refusal (sweepGridRule, statusImmutable, …) — the
            // engine's own words, inline beside the Save button.
            refuse(.sweepSpec, "sweep spec not saved: \(error.reason)")
            return false
        } catch {
            refuse(.sweepSpec, "declare optimization failed: \(error)")
            return false
        }
    }

    /// The success-note grid summary — §4.15(b)'s side-by-side: the depth
    /// fractions the manifest stores AND the absolute layers they resolve to
    /// at this model's depth, when a vector for it has stated that depth.
    private static func gridDeclarationSummary(
        _ grid: ExperimentStore.SweepGridOutcome
    ) -> String {
        let spec = grid.manifest.sweep ?? .init()
        let alphaPart = "\(spec.alphas.count) alpha"
            + "\(spec.alphas.count == 1 ? "" : "s")"
        guard grid.layerCount != nil, !grid.resolvedLayers.isEmpty else {
            return "\(spec.layerFractions.count) layer fraction"
                + "\(spec.layerFractions.count == 1 ? "" : "s") × \(alphaPart)"
        }
        return "fractions \(SweepSpecForm.numberListText(spec.layerFractions)) → "
            + SweepSpecForm.resolvedLayersText(
                fractions: spec.layerFractions, layerCount: grid.layerCount)
            + " × \(alphaPart)"
    }

    /// Run the layer×alpha sweep for an experiment — same task path as
    /// `steerlab-cli experiment sweep`. Local workspace: in-process through
    /// `ExperimentTasks.sweep` (which loads the pinned model itself — no
    /// preloaded chat model needed), progress mirrored into the display pane.
    /// Server workspace: submitted as a durable server job for the
    /// SERVER-RESIDENT copy, followed in the shared display.
    public func runSweep(experimentName name: String) async {
        guard !localJobs.isSweeping, !localJobs.isRunning, !localJobs.isValidating else { return }
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
            await serverExecution.run(experimentName: name, verb: "sweep", in: operationEnvironment)
            return
        }
        await localJobs.runSweep(experimentName: name)
    }

    public func refresh() {
        management.refresh()
    }

    private func refreshStudyDetails() {
        violations = management.selected.map(ExperimentStore.verify) ?? []
        freezeCoordinator.refreshReadiness(
            manifest: management.selected, violations: violations,
            runSubstrate: freezeEvidenceRunSubstrate,
            serverPaired: isServerWorkspace && cluster?.activeServerPairing == .paired)
        syncDraftFieldsFromSelection()
        refreshResults()
    }

    public func testRemoteConnection() async {
        loadStoredRemoteToken()
        guard let remoteClient else {
            remoteJobs.remoteStatus = "invalid server URL"
            return
        }
        do {
            let caps = try await remoteClient.capabilities()
            cluster?.persistToken()
            remoteJobs.remoteProfileSummary = Self.profileSummary(caps)
            remoteJobs.remoteStatus = "connected: \(caps.engine ?? "server") \(caps.serverVersion ?? "")"
        } catch {
            remoteJobs.remoteStatus = "remote connection failed: \(error)"
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
        loadStoredRemoteToken()
        await remoteJobs.reconnectRemoteJob(id, client: remoteClient)
    }

    public func stopRemoteLogStream() {
        remoteJobs.stopRemoteLogStream()
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
            executor: submission.remoteExecutor, gres: submission.remoteGres, siteGPUTypes: siteGPUTypes)
        submission.remoteExecutor = fixed.executor
        submission.remoteGres = fixed.gres
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
        guard let manifest = management.selected else {
            remoteJobs.remoteStatus = "select a study first"
            note("select a study first", severity: .info)
            return
        }
        _ = await bundleSubmission.submit(
            manifest, request: submission.snapshot(verb: verbOverride), followLog: true,
            execution: serverExecution, in: operationEnvironment)
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
        guard let manifest = management.experiments.first(where: { $0.name == name }) else {
            return .failure(
                StudyBatchSubmission.Failure(
                    reason: "study '\(name)' is no longer in this workspace"))
        }
        return await bundleSubmission.submit(manifest, request: submission.snapshot, followLog: false,
            execution: serverExecution, in: operationEnvironment)
    }

    // MARK: Two-phase sweep judgment (key-custody design 2026-07-18)

    /// Sweep AND evaluate runs of the selected study awaiting Mac-side
    /// external judgment (each record's `kind` says which). Refreshed
    /// alongside the remote surfaces; empty on non-server hosts.
    public private(set) var awaitingSweepJudgments:
        [ClusterClient.AwaitingSweepJudgment] = []
    /// True while a judging pass runs on this Mac (single-flight).
    public private(set) var isJudgingSweep = false

    public func refreshAwaitingSweepJudgments(study: String) async {
        let environment = operationEnvironment
        let generation = UUID()
        awaitingJudgmentGeneration = generation
        awaitingJudgmentContext = nil
        guard let context = environment.current(), context.isServer, context.selectedName == study, let client = context.client else {
            awaitingSweepJudgments = []
            return
        }
        let sweeps = (try? await client.awaitingSweepJudgments(experiment: study)) ?? []
        guard generation == awaitingJudgmentGeneration, environment.isCurrent(context, selection: true) else { return }
        // Older servers may not have the evaluation route; keep valid sweep work.
        let evaluates = (try? await client.awaitingEvaluateJudgments(experiment: study)) ?? []
        guard generation == awaitingJudgmentGeneration, environment.isCurrent(context, selection: true) else { return }
        awaitingSweepJudgments = sweeps + evaluates
        awaitingJudgmentContext = context
    }

    /// Phase 2 on THIS Mac: fetch the blinded packets (hash-verified), judge
    /// them with the Keychain key, and hand the judgments to the server's
    /// completion verb (which verifies every pin and replays the selection).
    public func judgeAwaitingSweep(
        study: String, awaiting: ClusterClient.AwaitingSweepJudgment
    ) async {
        let environment = operationEnvironment
        guard let context = awaitingJudgmentContext,
            context.selectedName == study, environment.isCurrent(context, selection: true), let client = context.client,
            awaitingSweepJudgments.contains(where: {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                guard let first = try? encoder.encode($0), let second = try? encoder.encode(awaiting) else { return false }
                return first == second
            }) else {
            remoteJobs.remoteStatus = "no server connection for judging"
            return
        }
        guard !isJudgingSweep else { return }
        isJudgingSweep = true
        defer { isJudgingSweep = false }
        do {
            let judgmentRun = try await SweepJudgmentRunner.judgeAndComplete(
                client: client, experiment: study, awaiting: awaiting,
                onProgress: { [weak self] progress in
                    await MainActor.run {
                        guard environment.isCurrent(context, selection: true) else { return }
                        self?.remoteJobs.remoteStatus = progress
                    }
                })
            guard environment.isCurrent(context, selection: true) else { return }
            remoteJobs.remoteStatus = awaiting.isEvaluate
                ? "evaluation judged on this Mac → judge report completed "
                    + "(\(judgmentRun))"
                : "sweep judged on this Mac → selection completed "
                    + "(\(judgmentRun)) — recommendations updated"
            await refreshAwaitingSweepJudgments(study: study)
            refresh()
        } catch {
            guard environment.isCurrent(context, selection: true) else { return }
            remoteJobs.remoteStatus = "sweep judging failed: "
                + String(describing: error)
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
        guard let name = management.selectedName else {
            note("select a study first", severity: .info)
            return
        }
        await serverExecution.run(experimentName: name, verb: verb, in: operationEnvironment)
    }

    /// Refreshes recent-job states from the server (`client.jobs()`,
    /// filtered to experiment job kinds). Jobs this panel did not submit are
    /// appended too — they persist server-side and remain reconnectable.
    public func refreshRecentServerJobs() async {
        await remoteJobs.refreshRecentServerJobs(client: remoteClient)
    }

    public func streamRemoteJobLog(jobID: String? = nil) async {
        await remoteJobs.streamRemoteJobLog(jobID: jobID, client: remoteClient)
    }

    /// Fetch the active server's run-directory listing. Runs are per-substrate
    /// artifacts: this is a read-only browse of what exists server-side, not a
    /// merge into the local results list. Full result *detail* (reports,
    /// generations) still comes home via the evidence-bundle import flow —
    /// the server exposes per-run files (`GET /api/runs/{id}/file`) but no
    /// structured results API yet.
    public func refreshRemoteRuns() async {
        let environment = operationEnvironment
        let generation = UUID()
        remoteRunsGeneration = generation
        guard let context = environment.current(), context.isServer, let client = context.client else {
            remoteRuns = []
            return
        }
        do {
            let records = try await client.runs()
            guard generation == remoteRunsGeneration, environment.isCurrent(context) else { return }
            remoteRuns = records
            remoteJobs.remoteStatus = "listed \(records.count) server run" + (records.count == 1 ? "" : "s")
        } catch {
            guard generation == remoteRunsGeneration, environment.isCurrent(context) else { return }
            remoteRuns = []
            remoteJobs.remoteStatus = "could not list server runs: \(error)"
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

    /// Refresh the remote Results listing from the active server. Keeps the
    /// current selection when its run id survives the refresh.
    public func refreshRemoteResultsRuns() async {
        await results.refreshRemoteResultsRuns(cluster: cluster)
    }

    /// Everything the remote run detail renders, assembled from ONE bounded
    /// fetch pass: previews and the semantic model share each file's bytes
    /// (a tunnel must never pay for the same head twice).
    public typealias RemoteRunDetailPayload = StudyResultsState.RemoteRunDetailPayload

    /// Byte cap for the remote report.json / manifest-snapshot / validation
    /// report head fetch — orders of magnitude above real reports, still
    /// BOUNDED (F5: the unbounded run-file route is never used here, even
    /// when the listed size is unknown). The server's own head ceiling
    /// (`RUN_FILE_HEAD_CAP`, 8 MiB) sits ABOVE this by contract, so a
    /// report under this cap arrives complete; a genuinely over-bound file
    /// arrives truncated WITH the server's file-size/truncated metadata
    /// headers and degrades to head-derived tables with the truncation
    /// caption — never to silently biased numbers presented as complete.
    public static let remoteReportByteLimit = StudyResultsState.remoteReportByteLimit
    static let remoteSemanticCaps = StudyResultsState.remoteSemanticCaps

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
        return await results.loadRemoteRunDetail(run: run, client: remoteClient, fetcher: fetcher)
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
            results.remoteResultsStatus = "invalid server URL"
            return
        }
        guard let bundleName = Self.evidenceBundleFileName(in: run.files) else {
            results.remoteResultsStatus = "run \(run.id) carries no evidence bundle — "
                + "import from the producing job instead (Compute section)"
            return
        }
        let workspaceRoot = ExperimentStore.workspaceRoot
        do {
            results.remoteResultsStatus = "downloading evidence from \(run.id)..."
            let downloads = workspaceRoot
                .appending(components: ".steerlab", "downloads", UUID().uuidString)
            let bundlePath = run.path.hasSuffix("/")
                ? run.path + bundleName : run.path + "/" + bundleName
            let localBundle = try await remoteClient.downloadArtifact(
                path: bundlePath, to: downloads)
            let imported = try await Task.detached {
                try EvidenceBundleImporter.importEvidenceBundle(localBundle, workspaceRoot: workspaceRoot).runDirectory
            }.value
            remoteJobs.remoteImportedRunDirectory = imported.path
            let importedMessage = "evidence from \(run.id) imported → "
                + "runs/\(imported.lastPathComponent) (hashes verified)"
            results.remoteResultsStatus = importedMessage
            note(importedMessage, severity: .success)
            noteEvidenceRevisionAdoption(forImportedRun: imported, workspaceRoot: workspaceRoot)
            refreshResults(selecting: imported.lastPathComponent)
            // An imported chain appears under Imported / local immediately
            // — the round trip is visible without a manual refresh.
            await pipelines.refresh(in: operationEnvironment)
        } catch {
            results.remoteResultsStatus = "evidence import failed: \(error)"
        }
    }

    /// After a VERIFIED evidence import: complete the researcher's declared
    /// intent by adopting the evidence snapshot's model revision into a
    /// still-unpinned local draft, or flag a conflicting pin loudly
    /// (external review 2026-07-22 — a server-resolved revision otherwise
    /// leaves local freeze readiness blocked on "revision not pinned").
    /// Decision + save live in `EvidenceRevisionAdoption` (unit-tested);
    /// this is the notice glue.
    public func noteEvidenceRevisionAdoption(forImportedRun imported: URL, workspaceRoot: URL) {
        let outcome = EvidenceRevisionAdoption.adoptModelRevision(
            fromImportedRun: imported, workspaceRoot: workspaceRoot)
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
        let environment = operationEnvironment
        let generation = UUID()
        remoteOptimizationsGeneration = generation
        guard let context = environment.current(), context.isServer else {
            remoteOptimizations = []
            return
        }
        guard let client = context.client else {
            remoteOptimizations = []
            note("invalid server URL", severity: .error)
            return
        }
        do {
            async let summaries = client.experimentSummaries()
            async let runList = client.runs()
            let (experiments, runRecords) = try await (summaries, runList)
            guard generation == remoteOptimizationsGeneration, environment.isCurrent(context) else { return }
            remoteRuns = runRecords
            remoteOptimizations = experiments.filter { record in
                record.conditions?.contains { $0.selection != nil } == true
                    || SweepRunCatalog.newestRemoteSweepRunRecord(
                        experiment: record.name, in: runRecords) != nil
            }
        } catch {
            guard generation == remoteOptimizationsGeneration, environment.isCurrent(context) else { return }
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
        let environment = operationEnvironment
        guard let context = environment.current(), context.isServer, let client = context.client else { return nil }
        do {
            let records = try await client.runs()
            guard environment.isCurrent(context) else { return nil }
            guard
                let record = SweepRunCatalog.newestRemoteSweepRunRecord(
                    experiment: experiment, in: records)
            else { return nil }
            let csv = try await client.runFile(runID: record.id, name: "sweep.csv")
            var recommendations: Data?
            if record.files.contains("recommendations.json") {
                recommendations = try await client.runFile(
                    runID: record.id, name: "recommendations.json")
            }
            guard environment.isCurrent(context) else { return nil }
            return try SweepRunCatalog.remoteSweepRun(
                runPath: record.path,
                csvText: String(decoding: csv, as: UTF8.self),
                recommendationsData: recommendations)
        } catch {
            guard environment.isCurrent(context) else { return nil }
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
        let environment = operationEnvironment
        guard let context = environment.current(), context.isServer else {
            note("no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        guard environment.isCurrent(context) else { return }
        guard let client = environment.connect() else {
            note("invalid server URL", severity: .error)
            return
        }
        guard environment.isCurrent(context), client.profile == context.client?.profile else { return }
        let substrate = context.substrate
        do {
            let names = try? await client.experimentNames()
            guard environment.isCurrent(context) else { return }
            if let names, !names.contains(experimentName) {
                note("study '\(experimentName)' is not in \(substrate)'s workspace — "
                    + "promote mints from the server-resident copy only. Pair "
                    + "the server to this workspace (serve --root <workspace>) "
                    + "or use Submit Bundle, the portable path for remote engines", severity: .info)
                return
            }
            guard environment.isCurrent(context) else { return }
            note("promoting '\(concept)' from '\(experimentName)' on \(substrate)…", severity: .info)
            let minted = try await client.promoteExperiment(
                name: experimentName, concept: concept,
                cell: cell, overrideReason: overrideReason, pins: pins)
            guard environment.isCurrent(context) else { return }
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
            guard environment.isCurrent(context) else { return }
            await host?.catalog.refreshRemoteVectors()
        } catch {
            guard environment.isCurrent(context) else { return }
            note("server promote failed: \(error)", severity: .error)
        }
    }

    public func cancelRemoteJob() async {
        await remoteJobs.cancelRemoteJob(client: remoteClient)
    }

    public func downloadRemoteEvidence() async {
        guard let remoteJobID = remoteJobs.remoteJobID else {
            remoteJobs.remoteStatus = "no remote job selected"
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
        await remoteJobs.resume(id, client: remoteClient)
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
            remoteJobs.remoteStatus = "invalid server URL"
            return
        }
        do {
            let remoteClient = try remoteJobs.clientForJob(jobID, connected: remoteClient)
            let workspaceRoot = try remoteJobs.origin(for: jobID).workspaceRoot
            let job = try await remoteClient.job(jobID)
            guard let result = job.result,
                let bundlePath = Self.findString(
                    in: .object(result), keyPath: ["runResult", "evidenceBundle", "bundlePath"])
                    ?? Self.findString(in: .object(result), keyPath: ["evidenceBundle", "bundlePath"])
            else {
                let pendingMessage = "job \(jobID) has no evidence bundle yet"
                remoteJobs.remoteStatus = pendingMessage
                note(pendingMessage, severity: .info)
                return
            }
            // Cross-check the download against the server-stamped bundle hash when present.
            let expectedSHA = Self.findString(
                in: .object(result), keyPath: ["runResult", "evidenceBundle", "bundleSha256"])
                ?? Self.findString(in: .object(result), keyPath: ["evidenceBundle", "bundleSha256"])
            remoteJobs.remoteStatus = "downloading evidence..."
            let downloads = workspaceRoot.appending(components: ".steerlab", "downloads", UUID().uuidString)
            let localBundle = try await remoteClient.downloadArtifact(path: bundlePath, to: downloads)
            // Extraction + per-file hashing off the main actor.
            let imported = try await Task.detached {
                try EvidenceBundleImporter.importEvidenceBundle(localBundle, expectedSHA256: expectedSHA, workspaceRoot: workspaceRoot).runDirectory
            }.value
            remoteJobs.remoteImportedRunDirectory = imported.path
            let importedMessage = "evidence from job \(jobID) imported → "
                + "runs/\(imported.lastPathComponent) (hashes verified)"
            remoteJobs.remoteStatus = importedMessage
            note(importedMessage, severity: .success)
            noteEvidenceRevisionAdoption(forImportedRun: imported, workspaceRoot: workspaceRoot)
            refreshResults(selecting: imported.lastPathComponent)
            // An imported chain appears under Imported / local immediately
            // — the round trip is visible without a manual refresh.
            await pipelines.refresh(in: operationEnvironment)
        } catch {
            let failureMessage = "evidence import failed: \(error)"
            remoteJobs.remoteStatus = failureMessage
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
        StudyProtocolAuthoring.applyBaseModelChoice(draft.studyBaseModelID, to: &manifest)
    }

    /// Re-reads the panel's base-model choice from the selected manifest.
    /// Headless routes call this before a protocol save whose request names
    /// no model: such a save must never act as a base-model change, and
    /// without the resync it compares the manifest against whatever the
    /// panel last synced — which can silently drop every variant condition
    /// (open-issues §8, residual (b)).
    public func adoptSelectedManifestBaseModel() {
        if let modelID = management.selected?.modelID { draft.studyBaseModelID = modelID }
    }

    /// Explicitly discard the editor's unsaved fields and start a new review.
    /// A normal inventory refresh deliberately does neither.
    public func reloadSelectedDraft() {
        refresh()
        syncDraftFieldsFromSelection(force: true)
        note("reloaded the saved study setup", severity: .info)
    }

    public func saveProtocol() {
        guard let manifest = management.selected, manifest.status == .draft else { return }
        do {
            let reviewed = try management.reviewedDraft(named: manifest.name)
            let selection = draft.selectedMultiAgentScenarioID.flatMap { id in
                multiAgentScenarioOptions.first { $0.id == id }
            }
            let scenario = try (draft.studyKind == .multiAgent ? selection : nil).map {
                try StudyProtocolScenario(
                    path: relativeProjectPath(for: $0.url), workspaceRoot: reviewed.workspaceRoot)
            }
            let result = try StudyProtocolAuthoring.save(
                reviewed: reviewed, fields: draft.protocolFields(inlineJudgeModel: resolvedInlineJudgeModel()),
                scenario: scenario)
            switch result {
            case .requiresScenario:
                note("select a scenario first", severity: .info)
            case .saved(let saved, let didCompileSeats, let advisories):
                management.acceptAuthoringResult(saved)
                if didCompileSeats { draft.seatCastingEdits = [:] }
                refresh()
                for advisory in advisories { note(advisory, severity: .warning) }
                note("saved protocol notes and run defaults", severity: .success)
            }
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
        InstrumentActivation.OutcomeMode.from(management.selected?.outcomeInstruments)
    }

    /// Items in the pinned prompt set that carry categorical `options` —
    /// what the DATA supports, independent of what is enabled.
    public var detectedOptionsItemCount: Int {
        draft.taskPromptsDocument?.optionsItemCount ?? 0
    }

    public var detectedCapabilitiesLine: String? {
        InstrumentActivation.detectedCapabilitiesLine(
            optionsItemCount: detectedOptionsItemCount)
    }

    /// Option-carrying items whose declared `responseFormat` the
    /// answer-token instruments cannot read.
    public var detectedUnscorableOptionItemCount: Int {
        draft.taskPromptsDocument?.unscorableOptionItemCount ?? 0
    }

    /// The prominent pre-run warning (P1): options present, no categorical
    /// instrument declared. nil = nothing to warn about.
    public var instrumentActivationWarning: String? {
        InstrumentActivation.activationWarning(
            optionsItemCount: detectedOptionsItemCount,
            instruments: management.selected?.outcomeInstruments,
            unscorableOptionItemCount: detectedUnscorableOptionItemCount)
    }

    /// Writes the outcome-mode choice into `manifest.outcomeInstruments`
    /// through the store (draft-only; never auto-enabled — the method
    /// belongs in provenance).
    public func setOutcomeMode(_ mode: InstrumentActivation.OutcomeMode) {
        guard let name = management.selectedName else { return }
        guard mode != .notDeclared else {
            // "not declared" is the read-back of an ABSENT declaration; the
            // picker never clears an explicit one silently.
            return
        }
        do {
            let instruments = InstrumentActivation.applying(
                mode, to: management.selected?.outcomeInstruments)
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.setOutcomeInstruments(
                    instruments, experimentName: reviewedName)
            }
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
        InstrumentActivation.auxiliaryInstruments(of: management.selected?.outcomeInstruments)
    }

    /// The effective-record-kinds note for the pre-run warning area (F3):
    /// non-nil exactly when the mode reads answer-token-only but a declared
    /// auxiliary forces sampled generation anyway.
    public var effectiveRecordKindsNote: String? {
        InstrumentActivation.effectiveRecordKindsNote(
            instruments: management.selected?.outcomeInstruments)
    }

    /// Remove one auxiliary instrument from `outcomeInstruments` (F3) —
    /// draft-only by the store's gate, written through
    /// `setOutcomeInstruments` like every other instrument edit. Removing
    /// the reader is what makes a genuinely logprob-only run possible.
    public func removeAuxiliaryInstrument(_ id: String) {
        guard let name = management.selectedName else { return }
        do {
            let remaining = (management.selected?.outcomeInstruments ?? []).filter { $0 != id }
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.setOutcomeInstruments(
                    remaining.isEmpty ? nil : remaining, experimentName: reviewedName)
            }
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
        guard let manifest = management.selected, manifest.status == .draft else { return false }
        return !(manifest.readerRefs ?? []).isEmpty
            && !(manifest.outcomeInstruments ?? []).contains("repeReaderScore")
    }

    /// Add `repeReaderScore` back alongside the current mode (F3) —
    /// draft-only, through the store.
    public func addReaderInstrument() {
        guard let name = management.selectedName, canAddReaderInstrument else { return }
        do {
            let instruments = (management.selected?.outcomeInstruments ?? []) + ["repeReaderScore"]
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.setOutcomeInstruments(
                    instruments, experimentName: reviewedName)
            }
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
        guard let name = management.selectedName else { return }
        let fdrText = draft.promotionFDRText.trimmingCharacters(in: .whitespaces)
        if !fdrText.isEmpty, Double(fdrText) == nil {
            note("promotion rule not saved: FDR threshold must be a number in (0, 1)", severity: .error)
            return
        }
        do {
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.setPromotionRule(
                    ExperimentManifest.PromotionRule(
                        fdrThreshold: Double(fdrText),
                        doseMonotone: draft.promotionDoseMonotone ? true : nil,
                        exceedsRandomFloor: draft.promotionExceedsRandomFloor ? true : nil,
                        capabilityGate: nilIfEmpty(draft.promotionCapabilityGateText)),
                    experimentName: reviewedName)
            }
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
        guard let name = management.selectedName else { return }
        do {
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.clearHumanBaseline(experimentName: reviewedName)
            }
            draft.humanBaselinePathField = ""
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
        guard let name = management.selectedName else { return }
        do {
            let pinned = try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.pinHumanBaseline(
                    path: draft.humanBaselinePathField, experimentName: reviewedName)
            }
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
        management.selected?.concepts.map(\.name) ?? []
    }

    /// Add a single-slot vector condition from the editor row's fields —
    /// negative α is legal (a one-field direction control).
    public func addVectorCondition() {
        guard let name = management.selectedName else { return }
        let concept = draft.conditionConcept.trimmingCharacters(in: .whitespaces)
        guard !concept.isEmpty else {
            refuse(.addCondition, "pick a concept for the condition", severity: .info)
            return
        }
        let isAblation = draft.conditionMode == .ablate
        // Ablation does not take a layer: it covers the whole network, and the
        // form hides the field. Parse it only when steering, so a stale value
        // left in the box cannot refuse an ablation that never needed it.
        var layer = 0
        if !isAblation {
            guard let parsed = Int(
                draft.conditionLayerText.trimmingCharacters(in: .whitespaces)),
                parsed >= 0
            else {
                refuse(.addCondition, "condition layer must be a non-negative integer")
                return
            }
            layer = parsed
        }
        guard let alpha = Double(draft.conditionAlphaText.trimmingCharacters(in: .whitespaces)),
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
        let conditionTitle = draft.conditionName.isEmpty
            ? (isAblation
                ? "\(concept)-ablate-l\(SweepSpecForm.numberListText([alpha]))"
                : "\(concept)-L\(layer)-a\(SweepSpecForm.numberListText([alpha]))")
            : draft.conditionName
        do {
            try management.editReviewed(named: name) { reviewedName in
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
                        alphaInNormUnits: isAblation ? false : draft.conditionAlphaInNormUnits),
                    experimentName: reviewedName)
            }
            draft.conditionName = ""
            refresh()
            clearFormError(.addCondition)
            note(
                isAblation
                    ? "added condition '\(conditionTitle)' — ablates \(concept) "
                        + "at λ\(alpha) across every layer"
                    : "added condition '\(conditionTitle)' (\(concept) L\(layer) "
                        + "α\(alpha)\(draft.conditionAlphaInNormUnits ? " norm-units" : ""))",
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
        let pinned = Set(management.selected?.concepts.map(\.name) ?? [])
        let declared = Set((management.selected?.validationControls ?? []).map(\.concept))
        return VectorCatalog.conceptNames()
            .filter { !pinned.contains($0) && !declared.contains($0) }
            .sorted()
    }

    public func addValidationControl() {
        guard let name = management.selectedName else { return }
        let concept = draft.controlConcept.trimmingCharacters(in: .whitespaces)
        guard !concept.isEmpty else {
            refuse(.validationControl, "pick a concept to use as a control",
                   severity: .info)
            return
        }
        do {
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.attachValidationControl(
                    concept: concept,
                    options: .init(method: draft.controlMethod),
                    experimentName: reviewedName)
            }
            draft.controlConcept = ""
            refresh()
            clearFormError(.validationControl)
            note("declared '\(concept)' as a discriminant control "
                + "(\(draft.controlMethod.rawValue), stimulus hash pinned)",
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
        guard let name = management.selectedName else { return }
        do {
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.removeValidationControl(
                    concept: concept, experimentName: reviewedName)
            }
            refresh()
            note("removed control '\(concept)'", severity: .info)
        } catch {
            refuse(.validationControl, "Couldn't remove the control: \(error)")
        }
    }

    /// Response formats present in the loaded task prompts, with row counts —
    /// what a scope can actually select over.
    public var availableResponseFormats: [(format: String, count: Int)] {
        guard let document = draft.taskPromptsDocument else { return [] }
        var counts: [String: Int] = [:]
        for item in document.responseFormatItems where item.hasOptions {
            counts[item.format?.rawValue ?? "(undeclared)", default: 0] += 1
        }
        return counts.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    public func declareOutcomeInstrumentScope(_ formats: [String]) {
        guard let name = management.selectedName else { return }
        do {
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.declareOutcomeInstrumentScope(
                    responseFormats: formats, experimentName: reviewedName)
            }
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
        guard let name = management.selectedName,
            let source = management.selected?.conditions.first(where: { $0.name == conditionNamed })
        else { return }
        do {
            let control = ExperimentStore.signControlCondition(for: source)
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.upsertCondition(control, experimentName: reviewedName)
            }
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
        guard let name = management.selectedName,
            let source = management.selected?.conditions.first(where: { $0.name == conditionNamed })
        else { return }
        do {
            let control = ExperimentStore.randomControlCondition(for: source)
            try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.upsertCondition(control, experimentName: reviewedName)
            }
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
        guard let name = management.selectedName else { return }
        do {
            let result = try management.editReviewed(named: name) { reviewedName in
                try ExperimentStore.scaffoldControlMatrix(experimentName: reviewedName)
            }
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
            draft.lastControlMatrixNotes = result.notes
        } catch {
            refuse(
                .addCondition,
                "Couldn't scaffold the control matrix — no conditions were "
                    + "changed; the study must still be a draft. "
                    + "Details: \(error)")
            draft.lastControlMatrixNotes = []
        }
    }

    /// The Data Readiness "edit" affordance (2026-07-19 paper cut: the
    /// button's effect — populating the Input Data editor further down —
    /// was invisible, so it read as dead). Same load, plus a notice saying
    /// what happened and WHERE to look.
    public func loadTaskPromptsInteractively() {
        loadTaskPrompts()
        if draft.taskPromptsDocument != nil {
            note(
                (draft.taskPromptsStatus ?? "task prompts loaded")
                    + " — edit them in the Input Data section below",
                severity: .info)
        } else {
            note(
                "task prompts could not be loaded: "
                    + (draft.taskPromptsStatus ?? "unknown error")
                    + " — if the file does not exist yet, use Create from "
                    + "template in Data Readiness",
                severity: .error)
        }
    }

    public func loadTaskPrompts() {
        let file = draft.taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !file.isEmpty else {
            draft.taskPromptsText = ""
            draft.taskPromptsStatus = "choose a task prompts file first"
            draft.taskPromptsInstrumentSummary = nil
            draft.taskPromptsDocument = nil
            draft.taskPromptsDocumentFile = nil
            draft.taskPromptsReview = nil
            return
        }
        do {
            let review = try TaskPromptsFileReview(path: file, workspaceRoot: ExperimentStore.workspaceRoot)
            let data = review.file.data
            draft.taskPromptsReview = review
            let document = try TaskPromptsDocument.load(data)
            draft.taskPromptsDocument = document
            draft.taskPromptsDocumentFile = file
            draft.taskPromptsText = document.editorText
            draft.taskPromptsInstrumentSummary = document.instrumentSummary
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            draft.taskPromptsStatus =
                "loaded \(document.count) prompt\(document.count == 1 ? "" : "s")"
                + " @ \(hash.prefix(12))…"
        } catch {
            draft.taskPromptsDocument = nil
            draft.taskPromptsDocumentFile = nil
            draft.taskPromptsReview = nil
            draft.taskPromptsInstrumentSummary = nil
            draft.taskPromptsStatus = "\(error)"
        }
    }

    public func saveTaskPrompts() {
        guard let manifest = management.selected, manifest.status == .draft else { return }
        let file = draft.taskPromptsFile.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !file.isEmpty else {
            draft.taskPromptsStatus = "choose a task prompts file first"
            return
        }
        let prompts = TaskPromptsDocument.editorBlocks(draft.taskPromptsText)
        guard !prompts.isEmpty else {
            draft.taskPromptsStatus = "add at least one prompt"
            return
        }
        do {
            let reviewed = try management.reviewedDraft(named: manifest.name)
            let result = try TaskPromptsAuthoring.save(
                reviewed: reviewed, path: file, source: draft.taskPromptsReview,
                editorText: draft.taskPromptsText)
            management.acceptAuthoringResult(result.study)
            refresh()
            let document = try TaskPromptsDocument.load(result.prompts.file.data)
            let hash = result.prompts.file.sha256
            draft.taskPromptsFile = result.prompts.path
            draft.taskPromptsReview = result.prompts
            draft.taskPromptsDocument = document
            draft.taskPromptsDocumentFile = result.prompts.path
            draft.taskPromptsText = document.editorText
            draft.taskPromptsInstrumentSummary = document.instrumentSummary
            // P1: report metadata-preserved and instruments-enabled as two
            // separate facts — preserved fields are NOT enabled measurement.
            let activation = InstrumentActivation.savePinSummary(
                optionsItemCount: document.optionsItemCount,
                itemCount: document.count,
                instruments: management.selected?.outcomeInstruments)
            draft.taskPromptsStatus =
                "saved and pinned \(prompts.count) prompt\(prompts.count == 1 ? "" : "s")"
                + " @ \(hash.prefix(12))… — \(activation)"
            note("saved a new prompt version and pinned its hash — \(activation)", severity: .success)
        } catch {
            draft.taskPromptsStatus = "\(error)"
            note(
                "Couldn't save the task prompts — nothing was pinned; check "
                    + "the file path stays inside the workspace and the study "
                    + "is a draft. Details: \(error)",
                severity: .error)
        }
    }

    /// The Import JSONL… action: validated full-record import (paste or
    /// file) that publishes a new task-prompts version, sets it as this
    /// study's prompts file, and pins the hash
    /// — one action from paste to a pinned immutable input version. The sheet
    /// retains its study review; changing study/workspace or file bytes refuses.
    @discardableResult
    public func importTaskPromptsJSONL(
        _ text: String, reviewed: DraftAuthoringSnapshot
    ) -> Bool {
        guard management.selectedName == reviewed.manifest.name,
            reviewed.workspaceRoot == ExperimentStore.workspaceRoot.standardizedFileURL,
            reviewed.manifest.status == .draft else {
            draft.taskPromptsStatus =
                "select a draft study first — import writes the file and pins "
                + "its hash into the draft manifest"
            return false
        }
        do {
            let saved = try TaskPromptsAuthoring.importJSONL(text, reviewed: reviewed)
            management.acceptAuthoringResult(saved.study)
            let count = try TaskPromptsDocument.load(saved.prompts.file.data).count
            draft.taskPromptsFile = saved.prompts.path
            refresh()
            // Re-read through the document loader so the editor, the
            // instrument badge, and the loaded-document pairing all reflect
            // the imported file.
            loadTaskPrompts()
            draft.taskPromptsStatus =
                "imported \(count) record\(count == 1 ? "" : "s")"
                + " → \(saved.prompts.path), pinned @ \(saved.prompts.file.sha256.prefix(12))…"
            note("imported task-prompt JSONL and pinned its hash", severity: .success)
            return true
        } catch {
            draft.taskPromptsStatus = "\(error)"
            return false
        }
    }

    public func runStudy() async {
        guard let name = management.selectedName, !localJobs.isRunning else { return }
        if isServerWorkspace {
            await runStudyOnActiveServer(verb: "run")
            return
        }
        await localJobs.runStudy(experimentName: name)
    }

    public func validateStudy() async {
        guard let name = management.selectedName, !localJobs.isValidating else { return }
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
        await localJobs.validateStudy(experimentName: name)
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
        guard let name = management.selectedName, !localJobs.isExtracting else { return }
        if isServerWorkspace {
            await runStudyOnActiveServer(verb: "extract")
            return
        }
        await localJobs.extractStudy(experimentName: name)
    }

    /// The run the paired judge will evaluate: the selected completed run,
    /// or — sensible default — the study's LATEST completed run when the
    /// selection is empty or a non-run artifact (validation, judge output).
    public var pairedJudgeTarget: StudyRunListItem? {
        if let item = results.selectedResult?.item, item.kind == .run {
            return item
        }
        return results.resultRuns.first { $0.kind == .run }
    }

    /// Why Run Paired Judge is disabled, in one plain sentence — nil means
    /// runnable. The UI must always surface this next to the button: a
    /// silently gray button is a bug, not a state.
    ///
    /// The judge half runs through `JudgeReadiness` — the SHARED precondition
    /// list the picker flags with and the executing route enforces. This gate
    /// predated `JudgeModelSpelling` and knew one shape, Claude-plus-a-key
    /// (review round 7, finding 5): an `openrouter:` spelling with no pinned
    /// provider passed here and threw at resolve; OpenRouter with no key
    /// passed here and refused inside the client; a local pick this Mac does
    /// not hold passed here and reached the loader, which downloads. A green
    /// button followed by a refusal is this gate lying about the run.
    public var pairedJudgeDisabledReason: String? {
        if localJobs.isEvaluating || localJobs.isRunning || localJobs.isValidating || localJobs.isExtracting {
            return "another study task is running — wait for it to finish"
        }
        guard management.selectedName != nil else { return "select a study first" }
        guard pairedJudgeTarget != nil else {
            return "no completed study run to judge yet — Run Study first"
        }
        let hasRubricFile = !draft.judgeRubricFile
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasInlineRubric = !draft.evaluationPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if !hasRubricFile, !hasInlineRubric {
            return "no judge rubric — pin a rubric file or enter inline rubric text above"
        }
        return Self.judgeDisabledReason(
            judges: draft.judges,
            adHocJudgeModel: draft.judgeModel,
            claudeKeyPresent: claudeKeyPresenceOverrideForTesting
                ?? (ClaudeStimulusGenerator.apiKey != nil),
            openRouterKeyPresent: judgeKeyPresenceOverrideForTesting
                ?? (JudgeKeyStore.resolveKey(kind: "openrouter") != nil),
            installed: judgeInstalledOverrideForTesting
                ?? JudgeReadiness.liveInstalled,
            capability: judgeCapabilityOverrideForTesting
                ?? JudgeReadiness.liveCapability)
    }

    /// The JUDGE half of `pairedJudgeDisabledReason` — why the declared judge
    /// (or, with no panel, the ad-hoc single-string one) cannot run. Static
    /// and pure so the rule is asserted directly, rather than only through a
    /// panel that first needs a study, a completed run, and a rubric on disk.
    ///
    /// Key state arrives as PRESENCE booleans; no credential is read here.
    nonisolated static func judgeDisabledReason(
        judges: [ExperimentManifest.JudgeRef],
        adHocJudgeModel: String,
        claudeKeyPresent: Bool,
        openRouterKeyPresent: Bool,
        installed: JudgeReadiness.InstalledCheck = JudgeReadiness.liveInstalled,
        capability: JudgeReadiness.CapabilityCheck = JudgeReadiness.liveCapability
    ) -> String? {
        func refusal(_ raw: String) -> String? {
            JudgeReadiness.refusal(
                for: raw, claudeKeyPresent: claudeKeyPresent,
                openRouterKeyPresent: openRouterKeyPresent,
                installed: installed, capability: capability)
        }
        let panelJudges = judges.filter {
            !$0.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard !panelJudges.isEmpty else {
            // Blank means the default Claude judge — the same default
            // `ExperimentTasks.resolvedJudges` applies — so the gate asks
            // about what would actually run, not about the empty string.
            let model = adHocJudgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
            return refusal(model.isEmpty ? ClaudePairedJudge.defaultModel : model)
        }
        // A local judge with no model is legal — it resolves to the study
        // model (manifest.modelID) at evaluation start.
        for judge in panelJudges {
            if judge.kind == "claude", !claudeKeyPresent {
                return "Claude judge '\(judge.name)' needs an API key — set "
                    + "ANTHROPIC_API_KEY or save a key in the Compute section "
                    + "(stored in the macOS Keychain)"
            }
            if judge.kind == "openrouter", !openRouterKeyPresent {
                return "OpenRouter judge '\(judge.name)' needs an external "
                    + "judge key — save one in the Compute section or set "
                    + "OPENROUTER_API_KEY"
            }
            // A named local judge model is loaded at evaluation start; one
            // this Mac does not hold reaches the hub. The same precondition
            // the ad-hoc branch and the executing loop use.
            let model = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if judge.kind == "local", !model.isEmpty,
                let reason = refusal(model)
            {
                return reason
            }
        }
        return nil
    }

    public func runPairedJudgeEvaluation() async {
        guard let name = management.selectedName, !localJobs.isEvaluating else { return }
        guard let item = pairedJudgeTarget else {
            note("no completed study run to judge yet — Run Study first", severity: .info)
            return
        }
        // Make the defaulted target visible: judging always operates on the
        // run the Results picker shows.
        if results.selectedResultID != item.id {
            results.selectedResultID = item.id
        }
        await localJobs.runPairedJudgeEvaluation(experimentName: name, sourceRun: item,
            evaluation: evaluationSpecFromDraft(),
            hasPinnedRubric: !draft.judgeRubricFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    public func clearLiveViewer() {
        localJobs.clearLiveViewer()
    }

    private func resetLiveViewer() {
        localJobs.resetLiveViewer()
    }

    private func handleStudyProgress(_ event: ExperimentTasks.StudyTaskProgress) {
        localJobs.handleStudyProgress(event)
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
    /// stimulus hash, method, reading position, a non-raw extraction
    /// rendering, and the three-state validation pin. Pure; unit-tested.
    public static func conceptPinStatusLine(
        _ ref: ExperimentManifest.ConceptRef
    ) -> String {
        var parts = [
            "stimuli @ \(ref.stimulusSetHash.prefix(12))…",
            ref.options.method.rawValue,
            ref.options.readingPosition.label,
        ]
        // A raw rendering stays UNMENTIONED, exactly as an absent declaration
        // does — the two are the same recipe, and naming one and not the other
        // would read as a difference. Only a declared chat template shows.
        if let rendering = ref.options.extractionRendering, !rendering.isRaw {
            parts.append(rendering.label)
        }
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
        guard let experiment = management.selectedName else { return }
        let concept = draft.attachConceptName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !concept.isEmpty else {
            note("pick a concept to attach", severity: .warning)
            return
        }
        // HOW the stimulus reaches the model, declared BEFORE anything is
        // loaded — the same order the CLI attach takes, and the same parser,
        // so the engine asymmetries (the assistant voice, addGenerationPrompt
        // false) are answered here in the engine's own words rather than
        // hours later on a run. Raw declares nothing at all.
        let declaredRendering: ExtractionRendering?
        do {
            declaredRendering = try draft.attachRendering.declared()
        } catch let error as ExtractionRendering.DeclarationError {
            note("\(error.reason) — repair: \(error.repair)", severity: .error)
            return
        } catch {
            note("\(error)", severity: .error)
            return
        }
        // A non-off reasoning effort on a family without a thinking mode is
        // answered here, where the study's model id is known (the parser
        // cannot see one). Server twin: `experiment_store.attach`.
        if let problem = ExtractionRendering.thinkingModeProblem(
            declaredRendering, modelID: management.selected?.modelID ?? "")
        {
            note(problem.message, severity: .error)
            return
        }
        // WHERE it is read, as the cross-engine label. nil for the recipe
        // default, which keeps the manifest byte-identical.
        let declaredPosition = draft.attachReadingPositionChoice.declarationLabel(
            parameter: draft.attachReadingPositionParameter)
        let corpus = draft.attachCorpusText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        do {
            let manifest = try management.editReviewed(named: experiment) { reviewedName in
                try ExperimentStore.attachConcept(
                    concept,
                    method: draft.attachMethod,
                    corpusConcepts: corpus,
                    reference: draft.attachReferenceName.isEmpty ? nil : draft.attachReferenceName,
                    extractionRendering: declaredRendering,
                    readingPosition: declaredPosition,
                    experimentName: reviewedName)
            }
            refresh()
            draft.attachConceptName = ""
            if draft.attachMethod == .emotionGrandMean {
                let hash = manifest.grandMeanCorpus?.hashes[concept] ?? ""
                note(
                    "pinned \(concept) @ \(hash.prefix(12))… (emotionGrandMean, "
                        + "corpus of \(manifest.grandMeanCorpus?.concepts.count ?? 0) "
                        + "concept(s))",
                    severity: .success)
            } else if let ref = manifest.concepts.first(where: { $0.name == concept }) {
                var pins = [
                    ref.options.method.rawValue, ref.options.readingPosition.label,
                ]
                // Raw stays unmentioned, exactly like absent.
                if let rendering = ref.options.extractionRendering, !rendering.isRaw {
                    pins.append(rendering.label)
                }
                note(
                    "pinned \(concept) @ \(ref.stimulusSetHash.prefix(12))… "
                        + "(\(pins.joined(separator: ", ")))",
                    severity: .success)
            }
        } catch let error as ExperimentError where error.malformedInvocation != nil {
            // A typed declaration refusal — the reading position, the
            // rendering, the two spellings of one position. It reaches the
            // notice VERBATIM, with its repair: wrapping it in "check your
            // stimulus files" would send a person to look at the wrong thing.
            note(
                "\(error.reason) — repair: "
                    + "\(error.malformedInvocation?.repairAction ?? "")",
                severity: .error)
        } catch {
            note(
                "Couldn't attach the concept — check its stimulus files "
                    + "exist on disk and the study is still a draft. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// Detach a pinned concept from the selected draft — the same store call
    /// `steerlab-cli experiment detach` makes, so a click and a command author
    /// the same manifest. Store-refused for frozen studies and for a concept
    /// any declaration still names (see `ExperimentStore.conceptDependents`).
    public func detachConcept(_ concept: String) {
        guard let experiment = management.selectedName else { return }
        do {
            try management.editReviewed(named: experiment) { reviewedName in
                try ExperimentStore.detachConcept(concept, experimentName: reviewedName)
            }
            refresh()
            note("detached concept '\(concept)' from '\(experiment)'")
        } catch {
            note("\(error)", severity: .error)
        }
    }

    /// Pins the concept at its current stimulus hash, using the concepts
    /// panel's current extraction options (method, pooling).
    public func attachConcept(_ name: String) {
        guard var manifest = management.selected, manifest.status == .draft else { return }
        do {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            let stimuli = try StimulusSet(directory: directory)
            let options = host?.concepts.extractionOptions ?? ExtractionOptions()
            manifest.concepts.removeAll { $0.name == name }
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: name, stimulusSetHash: stimuli.hash, options: options))
            ExperimentStore.pinNeutralCorpus(into: &manifest)  // norm denominator
            try management.persistReviewedDraft(manifest)
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

    public func addVariantCondition(reviewedAgent artifact: AgentArtifactSnapshot) {
        guard let manifest = management.selected, manifest.status == .draft else { return }
        do {
            let reviewed = try management.reviewedDraft(named: manifest.name)
            let saved = try StudyAgentAuthoring.attach(artifact, reviewed: reviewed, baseModelChoice: draft.studyBaseModelID)
            management.acceptAuthoringResult(saved)
            draft.selectedVariantToAddID = nil
            refresh()
            note("added agent '\(artifact.record.artifact.name)'", severity: .success)
        } catch {
            note(
                "Couldn't add the agent — check it uses this study's "
                    + "baseline model and the study is still a draft. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    public func removeVariantCondition(_ name: String) {
        guard var manifest = management.selected, manifest.status == .draft else { return }
        manifest.variantConditions.removeAll { $0.name == name }
        do {
            try management.persistReviewedDraft(manifest)
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
            ?? management.selected.map(StudyIntent.derive(from:))
            ?? .conceptStudy
    }

    /// The selected study as one pasteable JSON document (the same
    /// experiment.json every engine reads). Nil (with a notice) on failure.
    public func exportSelectedStudyJSON() -> String? {
        guard let manifest = management.selected else {
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

    /// The selected study's manifest as one pretty-printed JSON document,
    /// for DISPLAY (the Studies display pane). Same document and same
    /// encoder as `exportSelectedStudyJSON` — `ExperimentStore.exportStudyJSON`
    /// over the manifest this panel already decoded, never a second read of
    /// experiment.json — but SILENT: a display pane re-reads on every render,
    /// and "select a study first" is the empty state there, not a notice.
    public var selectedStudyJSON: String? {
        guard let manifest = management.selected else { return nil }
        return try? ExperimentStore.exportStudyJSON(manifest)
    }

    /// Import pasted study JSON as a NEW DRAFT (freeze metadata stripped —
    /// pasted text cannot mint a preregistered object), select it, and
    /// surface its verify() result loudly.
    @discardableResult
    public func importStudyJSON(_ text: String, reviewed: StudyPackAuthoring.Preview) -> Bool {
        do {
            let imported = try StudyPackAuthoring.apply(Data(text.utf8),
                workspaceRoot: URL(fileURLWithPath: reviewed.workspaceRoot), expectedReviewSHA256: reviewed.reviewSHA256)
            let manifest = imported.study.manifest
            let violations = imported.violations
            let filesWritten = imported.filesWritten
            refresh()
            management.selectedName = manifest.name
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
            return true
        } catch {
            note(
                "Couldn't complete the study import. Inspect the named destination before retrying. Details: \(error)",
                severity: .error)
            return false
        }
    }

    /// The primary action for a declared chain: submit the pipeline verb
    /// through the same bundle path as every remote run (executor and
    /// resources come from Remote options).
    public func runPipelineRemotely() async {
        submission.remoteVerb = "pipeline"
        await submitSelectedStudyRemotely()
    }

    /// Stage-4 authoring affordance (stage 5): declare "the agent this
    /// study's sweep promotes for CONCEPT" as a condition — before the
    /// agent exists. Data-only: the SERVER resolves it at run time from the
    /// promotion birth certificate and pins path + hash as run evidence.
    public func addForwardReferencedCondition(concept: String) {
        guard var manifest = management.selected, manifest.status == .draft else { return }
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
            try management.persistReviewedDraft(manifest)
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
        guard let host, var manifest = management.selected else { return }
        let name = draft.conditionName.isEmpty ? "condition-\(manifest.conditions.count + 1)"
            : draft.conditionName
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
            try management.persistReviewedDraft(manifest)
            draft.conditionName = ""
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
        guard var manifest = management.selected, manifest.status == .draft else { return }
        let name = draft.conditionName.isEmpty ? "baseline" : draft.conditionName
        do {
            manifest.conditions.removeAll { $0.name == name }
            manifest.conditions.append(
                .init(name: name, slots: [], bandWidth: 1, alphaInNormUnits: true))
            try management.persistReviewedDraft(manifest)
            draft.conditionName = ""
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
        guard var manifest = management.selected, manifest.status == .draft else { return }
        manifest.conditions.removeAll { $0.name == name }
        do {
            try management.persistReviewedDraft(manifest)
            refresh()
        } catch {
            note("\(error)", severity: .error)
        }
    }

    private func syncDraftFieldsFromSelection(force: Bool = false) {
        let manifest = management.selected
        if let manifest, !force, draft.syncedSelection == manifest.name { return }
        // Resolve workspace/library facts here; the editor owns only their values.
        let scenarioID: MultiAgentScenarioRecord.ID?
        if let path = manifest?.multiAgentSemanticScenarioPath ?? manifest?.multiAgentScenarioPath {
            let target = scenarioURL(from: path).resolvingSymlinksInPath().path
            scenarioID = multiAgentScenarioOptions.first {
                $0.url.resolvingSymlinksInPath().path == target
            }?.id
        } else { scenarioID = multiAgentScenarioOptions.first?.id }
        let changed = draft.synchronize(manifest, defaults: .init(
            baseModelID: host?.workspaceSelectedModelID ?? modelOptions.first
                ?? ChatService.availableModels.first?.id ?? "",
            judgeModel: defaultJudgeModel(for: manifest),
            variantID: availableVariantsForStudy.first?.id,
            confirmAgentID: confirmableAgents.first?.id, scenarioID: scenarioID), force: force)
        guard changed else { return }
        management.beginAuthoringReview(named: manifest?.name)
        if manifest == nil { results.clearSelectionAndRuns() }
        if manifest?.studyKind == .modelOutput { loadTaskPrompts() }
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
        let prompt = draft.evaluationPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return nil }
        return ExperimentManifest.EvaluationSpec(
            kind: .pairedJudge,
            judgeModel: resolvedInlineJudgeModel(),
            judgePrompt: prompt,
            structuredPrompt: nilIfEmpty(draft.evaluationStructuredPrompt))
    }

    /// The ad-hoc judge model the inline (scratchpad) evaluation spec
    /// carries: the panel field, else the study-model default.
    private func resolvedInlineJudgeModel() -> String {
        let trimmed = draft.judgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultJudgeModel(for: management.selected) : trimmed
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
        results.refresh(experimentName: management.selectedName, repository: StudyResultRepository(workspaceRoot: ExperimentStore.workspaceRoot), selecting: preferredID)
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
    ///
    /// A thin name over `ExperimentStore.sweepGridProblem`, THE grid audit —
    /// the one whose strings are the cross-engine refusal contract. This
    /// form used to keep its own copy of these checks, minus the two ascent
    /// rules, which is how the Optimizations panel could save
    /// `alphas 0.1, 0.05` where `set-sweep-grid` refuses it. One definition
    /// now answers the CLI, the HTTP route, and every form.
    public static func validate(_ spec: ExperimentManifest.SweepSpec) -> String? {
        ExperimentStore.sweepGridProblem(
            layerFractions: spec.layerFractions, alphas: spec.alphas,
            devPromptsFile: spec.devPromptsFile,
            batteryFile: spec.batteryFile, maxTokens: spec.maxTokens)
    }

    /// The absolute layers a fraction axis resolves to — §4.15(b)'s
    /// side-by-side, as one caption phrase ("layers 17, 23, 28 of 34").
    /// A fraction set that collapses at this depth says so: a grid of "four
    /// depths" that is really three is a silently smaller sweep. With no
    /// known depth (nothing extracted for the model yet), that fact is the
    /// caption.
    public static func resolvedLayersText(
        fractions: [Double], layerCount: Int?
    ) -> String {
        guard let layerCount, layerCount > 0 else {
            return "layer indices unknown until a vector for this model "
                + "states its depth"
        }
        let layers = ExperimentManifest.SweepSpec(layerFractions: fractions)
            .resolvedLayers(layerCount: layerCount)
        var text = "layers " + layers.map(String.init).joined(separator: ", ")
            + " of \(layerCount)"
        let collapsed = max(0, Set(fractions).count - layers.count)
        if collapsed > 0 {
            text += " (\(collapsed) fraction\(collapsed == 1 ? "" : "s") "
                + "collapsed onto a layer already in the grid)"
        }
        return text
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

    /// The coherence rule as an EDITOR holds it: the ratio that decides the
    /// form, and the one absolute number the editor's single field edits.
    public struct EditorCoherenceForm: Equatable, Sendable {
        /// Non-nil ⇒ the baseline-relative form. Carried through the editor
        /// untouched so a save re-declares the form that was loaded.
        public var ratio: Double?
        /// The backstop under the relative rule, the floor under the legacy
        /// one — whichever absolute number this criterion's rule uses.
        public var floor: Double

        public init(ratio: Double?, floor: Double) {
            self.ratio = ratio
            self.floor = floor
        }
    }

    /// Read a declared criterion into the editor's two coherence values.
    ///
    /// The presence rule is the whole point (`SweepSelectionRule.resolve`,
    /// both engines): EITHER relative field selects the relative form. A
    /// criterion carrying only `coherenceAbsoluteBackstop` is therefore a
    /// RELATIVE criterion whose ratio was left to the default — and the
    /// editor used to load it with a nil ratio, which made the field say
    /// "floor" and made a save write `coherenceFloor`, converting a declared
    /// relative rule into the legacy absolute one that no baseline can move
    /// (review round 9, finding 4). Resolving the default ratio HERE means a
    /// load-then-save round-trips the declared FORM, and the number the
    /// editor shows is the number the sweep would gate on either way.
    public static func editorCoherenceForm(
        _ constraints: ExperimentManifest.SweepSelection.Constraints?
    ) -> EditorCoherenceForm {
        let ratio = constraints?.coherenceRatioToBaseline
        let backstop = constraints?.coherenceAbsoluteBackstop
        guard ratio != nil || backstop != nil else {
            return EditorCoherenceForm(
                ratio: nil,
                floor: constraints?.coherenceFloor
                    ?? SweepSelectionRule.defaultCoherenceFloor)
        }
        return EditorCoherenceForm(
            ratio: ratio ?? SweepSelectionRule.defaultCoherenceRatio,
            floor: backstop ?? SweepSelectionRule.defaultCoherenceBackstop)
    }

    /// The editor's two coherence values back into a constraints block — the
    /// exact inverse of `editorCoherenceForm`, so the pair is a round trip
    /// and not two rules written twice.
    public static func editorCoherenceConstraints(
        capabilityTolerance: Double?, form: EditorCoherenceForm
    ) -> ExperimentManifest.SweepSelection.Constraints {
        guard let ratio = form.ratio else {
            return .init(
                capabilityTolerance: capabilityTolerance,
                coherenceFloor: form.floor)
        }
        return .init(
            capabilityTolerance: capabilityTolerance,
            coherenceRatioToBaseline: ratio,
            coherenceAbsoluteBackstop: form.floor)
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
        guard var manifest = management.selected, manifest.status == .draft else {
            note(
                "select a draft study first — choosing a prompts file pins "
                    + "it into the draft manifest",
                severity: .info)
            return
        }
        do {
            let hash = try ExperimentStore.pinTaskPrompts(
                relativePath, into: &manifest)
            try management.persistReviewedDraft(manifest)
            draft.taskPromptsFile = relativePath
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

    /// Convert the reviewed table to full records, then use the same immutable
    /// input publication as JSONL import. A dialog retains its original target.
    public func importTaskPromptsTable(
        table: TabularImport.Table, mapping: [String: String], reviewed: DraftAuthoringSnapshot
    ) -> String? {
        do {
            let text = try TabularImport.taskPromptsJSONL(table: table, mapping: mapping)
            guard importTaskPromptsJSONL(text, reviewed: reviewed) else {
                return draft.taskPromptsStatus ?? "Prompt import was refused; review the study and retry."
            }
            return nil
        } catch { return "\(error)" }
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
        table: TabularImport.Table, mapping: [String: String], reviewed: DraftAuthoringSnapshot
    ) -> String? {
        guard management.selectedName == reviewed.manifest.name else {
            return "select a study first — the baseline pins into the "
                + "selected draft's manifest"
        }
        do {
            let (pinned, saved) = try DraftAuthoringTransaction.perform(reviewed: reviewed) { name in
                let pinned = try TabularImport.importHumanBaseline(table: table, mapping: mapping, experimentName: name)
                return (pinned, try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: name))
            }
            management.acceptAuthoringResult(saved)
            draft.humanBaselinePathField = pinned.path
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
