import Foundation
import Observation
import PDFKit

@Observable @MainActor
public final class FineTuningPanel {
    public internal(set) weak var host: ChatService?

    public enum FineTuneType: String, Codable, Sendable, CaseIterable, Identifiable {
        case lora
        case dora

        public var id: String { rawValue }
        public var label: String {
            switch self {
            case .lora: "LoRA"
            case .dora: "DoRA"
            }
        }
    }

    public static let defaultFineTuneType: FineTuneType = .lora
    public static let defaultRank = 8
    public static let defaultScale = 10.0
    public static let defaultAdaptedLayers = 16
    public static let defaultBatchSize = 4
    public static let defaultIterations = 1000
    public static let defaultLearningRate = 0.00001

    public struct DatasetProfile: Sendable, Equatable {
        public var path: String
        public var exampleCount: Int
        public var estimatedTokens: Int
        public var medianCharacters: Int
        public var instructionRows: Int
        public var textRows: Int
        public var unreadableRows: Int

        public var kind: String {
            if exampleCount == 0 { return "empty" }
            if instructionRows > textRows { return "instruction/chat" }
            return medianCharacters > 900 ? "document adaptation" : "text examples"
        }
    }

    public struct TrainingPlan: Sendable, Equatable {
        public var modelSizeBillions: Double?
        public var dataScale: String
        public var train: DatasetProfile
        public var validation: DatasetProfile
        public var recommendedRank: Int
        public var recommendedScale: Double
        public var recommendedLayers: Int
        public var recommendedBatchSize: Int
        public var recommendedIterations: Int
        public var recommendedLearningRate: Double
        public var rationale: String
        public var warnings: [String]
    }

    public struct LiveRobustnessOutput: Identifiable, Sendable, Equatable {
        public var id: String { "\(kind)-\(index)-\(side)" }
        public var kind: String
        public var index: Int
        public var total: Int
        public var prompt: String
        public var side: String
        public var output: String
        public var isComplete: Bool

        public var title: String {
            "\(kind) \(index)/\(total) · \(side)"
        }
    }

    public struct LiveRobustnessJudgment: Identifiable, Sendable, Equatable {
        public var id: String { "judge-\(index)" }
        public var index: Int
        public var prompt: String
        public var result: String
        public var response: PairedJudgeResponse
    }

    public private(set) var adapters: [FineTuneArtifactRecord] = []
    public private(set) var variants: [ModelVariantRecord] = []
    public private(set) var status: String?

    // MARK: Agent Library list state (see `AgentLibraryIndex`)

    /// Row-ready summaries of `variants` — what the Agent Library list
    /// renders. Kept here rather than in the view so it SURVIVES a section
    /// switch: the view's `@State` does not, which is why re-entering the
    /// Agents tab used to repeat the whole scan before drawing anything.
    public private(set) var agentIndex: [AgentLibraryIndex.Entry] = []
    /// Latest robustness evidence per agent index id — the deferred overlay,
    /// filled in after the list is on screen.
    public private(set) var robustnessByAgentID:
        [String: AgentEvidence.RobustnessEvidence] = [:]
    /// An off-main library scan is in flight (the list shows its previous
    /// rows meanwhile; the section header says it is refreshing).
    public private(set) var isScanningAgents = false
    /// True once a scan has completed, so "no agents yet" and "not scanned
    /// yet" are distinguishable empty states.
    public private(set) var hasScannedAgents = false
    /// The workspace root `agentIndex` describes — nil before the first scan
    /// completes. Rendered so a stale list can never masquerade as the
    /// workspace the toolbar names.
    public private(set) var scannedAgentRoot: URL?
    /// Latest-wins guard: a workspace switch during an in-flight scan must
    /// not let the previous root's rows land afterwards. Same mechanism as
    /// `DatasetInventoryModel.generation`.
    private var agentScanGeneration = 0
    /// The evidence overlay's OWN latest-wins counter, separate from the
    /// library scan's (review round 6, finding 6).
    ///
    /// The two passes were sharing `agentScanGeneration`, and the Library
    /// starts evidence TWICE by design: once in `.task`, for the rows carried
    /// over from a previous visit, and again from `onChange(of: agentIndex)`
    /// when the rescan reports new ones. Both captured the same token, so the
    /// token could not tell them apart, and whichever `runs/` walk happened to
    /// finish last won — including the older one, over evidence computed for a
    /// NEWER snapshot. Its own counter makes "a second evidence pass started"
    /// invalidate the first, which is what latest-wins means.
    private var agentEvidenceGeneration = 0
    /// Where notes persist (A15). The shared per-workspace feed in the app;
    /// tests inject a hermetic instance.
    public var notices: PanelNotices = .shared

    /// A15: sets the legacy single-slot status string (unchanged UI) AND
    /// appends a persistent notice to the workspace feed, verbatim.
    /// High-frequency progress mirrors (training-progress echoes, per-line
    /// job logs) keep writing `status` directly — transient telemetry must
    /// not evict the failures the feed exists to keep.
    func note(_ message: String, severity: PanelNotice.Severity = .info) {
        status = message
        notices.record(source: "Agents", severity: severity, message: message)
    }

    public private(set) var trainingPlan: TrainingPlan?
    public private(set) var isTraining = false
    public private(set) var trainingProgress: String?
    public private(set) var trainingLog: [String] = []
    public private(set) var isRobustnessRunning = false
    /// A local robustness-check cancellation was requested (App gap A1):
    /// `VariantRobustness.run` polls this between generations and stops
    /// after the current one; a partial battery is never scored or written.
    /// Reset when the next local check starts.
    public private(set) var robustnessCancelRequested = false
    public private(set) var robustnessReport: VariantRobustnessReport?
    public private(set) var lastRobustnessDirectory: String?
    public private(set) var liveRobustnessOutputs: [LiveRobustnessOutput] = []
    public private(set) var liveRobustnessJudgments: [LiveRobustnessJudgment] = []
    public private(set) var liveRobustnessJudgeStatus: String?
    public private(set) var liveRobustnessFailure: String?

    public var newAdapterName = "my-concept-lora"
    public var newAdapterBaseModelID = ChatService.availableModels.first?.id ?? ""
    public var newAdapterProjectDirectory = ""

    public var adapterName = ""
    public var adapterBaseModelID = ""
    public var adapterProjectDirectory = ""
    public var adapterDirectory = ""
    public var trainingWorkspacePath = ""
    public var trainingDataPath = ""
    public var validationDataPath = ""
    public var trainingMode: FineTuneTrainingMode = .document
    public var fineTuneType: FineTuneType = defaultFineTuneType
    public var rank = defaultRank
    public var scale = defaultScale
    public var adaptedLayers = defaultAdaptedLayers
    public var batchSize = defaultBatchSize
    public var iterations = defaultIterations
    public var learningRate = defaultLearningRate
    public var notes = ""
    public var selectedAdapterID: FineTuneArtifactRecord.ID?
    private var trainingTask: Task<Void, Never>?
    private var serverTrainingLogTask: Task<Void, Never>?
    /// The active server's durable fine-tune job id while server training is
    /// in flight — the honest cancel target (cluster-testing item 3: Cancel
    /// used to stop only the local log stream and the cluster job ran on).
    /// Set when `fineTuneTrain` answers; cleared when the log stream ends.
    public private(set) var serverTrainingJobID: String?
    /// Single-flight guard for the server cancel request.
    private var serverCancelInFlight = false

    public var variantName = "variant-1"
    public var selectedVariantID: ModelVariantRecord.ID?
    public var robustnessPresetID = VariantRobustness.defaultPreset.id
    public var robustnessBatteryFile = VariantRobustness.defaultPreset.batteryFile
    public var robustnessPromptsFile = VariantRobustness.defaultPreset.coherencePromptsFile
    public var robustnessMaxPrompts = VariantRobustness.defaultPreset.maxCoherencePrompts
    public var robustnessMaxTokens = VariantRobustness.defaultPreset.maxTokens
    public var robustnessJudgeModel = ""
    public var robustnessUseJudge = false
    /// The agent the Robustness Check targets — chosen by its OWN picker,
    /// deliberately not inherited from the browser/editor selection
    /// (live-testing finding: the implicit coupling read as "it somehow
    /// checks whatever the editor has open"). Under a server target the
    /// picker also offers the SERVER's stored agents (the same source the
    /// agents list shows — one scoping rule), identified by server path.
    public enum RobustnessTarget: Hashable, Sendable {
        /// A local definition record (runs in-process locally, or as an
        /// inline spec through the active server).
        case local(ModelVariantRecord.ID)
        /// A variant stored on the ACTIVE server, by its server path (runs
        /// as a stored-variant generate with the fetched spec + hash pinned).
        case server(path: String)
    }

    public var robustnessTarget: RobustnessTarget?

    /// Legacy local-record view of `robustnessTarget`, kept so existing call
    /// sites ("Run robustness" on a library row) stay source-compatible.
    public var robustnessTargetVariantID: ModelVariantRecord.ID? {
        get {
            if case .local(let id) = robustnessTarget { return id }
            return nil
        }
        set { robustnessTarget = newValue.map { .local($0) } }
    }

    public init() {
        refresh()
        adapterBaseModelID = newAdapterBaseModelID
    }

    public var selectedAdapter: FineTuneArtifactRecord? {
        guard let selectedAdapterID else { return nil }
        return adapters.first { $0.id == selectedAdapterID }
    }

    public var selectedVariant: ModelVariantRecord? {
        guard let selectedVariantID else { return nil }
        return variants.first { $0.id == selectedVariantID }
    }

    /// THE model-layer seam for "open this agent in the Agents section"
    /// (WP-Data phase 4): re-scan the library so an agent minted outside this
    /// app session is present, then select it.
    ///
    /// It exists so the Derived inventory's `DerivedArtifactRoute.agents`
    /// route can PRESELECT the artifact it names instead of switching
    /// sections and asking the researcher to find the row again. The
    /// selection was already model-side (`selectedVariantID`, which the
    /// Library browser binds to) — only the refresh-then-set rule was
    /// open-coded at each call site.
    ///
    /// `id` is the agent artifact's file path — `ModelVariantRecord.id`, the
    /// same string `DerivedArtifactEntry.selectionKey` carries. Returns false
    /// when no such record is in the library, leaving the selection alone
    /// rather than setting it to something the browser cannot show.
    @discardableResult
    public func selectAgent(id: ModelVariantRecord.ID) -> Bool {
        refresh()
        guard variants.contains(where: { $0.id == id }) else { return false }
        selectedVariantID = id
        return true
    }

    public var robustnessTargetVariant: ModelVariantRecord? {
        guard let robustnessTargetVariantID else { return nil }
        return variants.first { $0.id == robustnessTargetVariantID }
    }

    /// The server-stored target's record in the active server's listing
    /// (nil for local targets, or when the listing no longer has the path).
    public var robustnessTargetServerAgent: RemoteVariantRecord? {
        guard case .server(let path) = robustnessTarget else { return nil }
        return host?.cluster.remoteVariants.first { $0.path == path }
    }

    /// Display name for the chosen robustness target, either source.
    public var robustnessTargetName: String? {
        if let record = robustnessTargetVariant { return record.artifact.name }
        if case .server(let path) = robustnessTarget {
            return robustnessTargetServerAgent?.name
                ?? (path as NSString).lastPathComponent
        }
        return nil
    }

    /// "The Run button may fire": the chosen target still resolves in its
    /// source list (local record present, or server path in the active
    /// server's listing).
    public var hasResolvableRobustnessTarget: Bool {
        switch robustnessTarget {
        case .local: return robustnessTargetVariant != nil
        case .server: return robustnessTargetServerAgent != nil
        case nil: return false
        }
    }

    /// Adapters offerable for in-process application: filtered to the loaded
    /// model AND to this substrate — an explicitly foreign-stamped adapter
    /// (e.g. "hf-peft-lora" on "python-hf-transformers") never appears in an
    /// application picker; unstamped legacy records stay visible. The full
    /// `adapters` list (management/editing) is intentionally unfiltered.
    public var compatibleAdapters: [FineTuneArtifactRecord] {
        let loadable = adapters.filter {
            !AdapterSubstrateGate.isExplicitlyForeign(
                substrate: $0.artifact.substrate,
                adapterFormat: $0.artifact.adapterFormat)
        }
        guard let modelID = host?.loadedModelID ?? host?.selectedModelID else { return loadable }
        return loadable.filter { $0.artifact.baseModelID == modelID }
    }

    /// The Agents section's refresh: the same scan as `refresh()`, but OFF
    /// the main actor, so switching to the tab draws immediately and the
    /// library lands when it lands.
    ///
    /// Follows `DatasetInventoryModel.refresh` verbatim — the house pattern
    /// for a catalog that a section's appearance re-reads: capture the roots
    /// on the main actor, scan in a detached task, apply under a
    /// latest-wins generation token. Previously-scanned rows stay on screen
    /// while a rescan runs, which is what makes RE-entering the tab instant;
    /// it is invalidated exactly like the other catalogs, by calling this
    /// again (`WorkspaceControls.resetCatalogs`, a save, a promote).
    ///
    /// Cheap to call. Nothing here is a new invalidation mechanism.
    public func refreshAgentLibraryAsync() {
        let library = ModelVariantStore.directory
        let runs = VectorCatalog.runsDirectory
        let root = VectorCatalog.projectRoot
        agentScanGeneration &+= 1
        let token = agentScanGeneration
        isScanningAgents = true
        Task.detached(priority: .userInitiated) {
            let scannedAdapters = FineTuneStore.scan()
            let records = ModelVariantStore.scan(
                directory: library, importedRoot: runs)
            let entries = AgentLibraryIndex.summarize(records)
            await MainActor.run { [weak self] in
                guard let self, token == self.agentScanGeneration else { return }
                self.adapters = scannedAdapters
                self.applyScannedVariants(records, entries: entries)
                self.scannedAgentRoot = root
                self.hasScannedAgents = true
                self.isScanningAgents = false
                self.reconcileSelectionsAfterScan()
            }
        }
    }

    /// Second phase: the robustness overlay for the rows already on screen.
    /// Split from the scan because it costs a `runs/` walk plus a full-file
    /// hash PER AGENT — the part that actually scaled with the number of
    /// promoted agents — and no row needs it to draw. Callers run it only
    /// for the region that shows it (the Library).
    ///
    /// Two guards, because the token alone was not enough (review round 6,
    /// finding 6): its OWN generation counter, so a second evidence pass
    /// invalidates the first rather than racing it; and the captured snapshot
    /// — the root and the exact set of entry ids the walk was computed for —
    /// checked against the live index before anything is applied. The second
    /// is what makes a late result harmless even if a counter is ever bumped
    /// somewhere this method does not know about: evidence keyed on rows that
    /// are no longer the rows on screen renders under the wrong agent, and
    /// that is the failure worth being paranoid about.
    public func refreshAgentEvidenceAsync() {
        let entries = agentIndex
        guard !entries.isEmpty else {
            robustnessByAgentID = [:]
            return
        }
        let runs = ExperimentStore.runsDirectory
        let root = VectorCatalog.projectRoot
        let entryIDs = Set(entries.map(\.id))
        let scanToken = agentScanGeneration
        agentEvidenceGeneration &+= 1
        let evidenceToken = agentEvidenceGeneration
        Task.detached(priority: .utility) {
            let evidence = AgentLibraryIndex.evidence(
                for: entries, runsDirectory: runs)
            await MainActor.run { [weak self] in
                guard let self,
                    Self.evidenceMayLand(
                        scanToken: scanToken,
                        liveScanToken: self.agentScanGeneration,
                        evidenceToken: evidenceToken,
                        liveEvidenceToken: self.agentEvidenceGeneration,
                        root: root, liveRoot: self.scannedAgentRoot,
                        entryIDs: entryIDs,
                        liveEntryIDs: Set(self.agentIndex.map(\.id)))
                else { return }
                self.robustnessByAgentID = evidence.byEntryID
            }
        }
    }

    /// Whether a finished evidence walk still describes what is on screen.
    ///
    /// Pure and static so the rule is testable without racing two detached
    /// tasks: every input is what the walk CAPTURED beside what the panel
    /// holds NOW. All four must agree — a superseded library scan, a
    /// superseded evidence pass, a workspace that moved, or a row set that
    /// changed each make the result evidence about something else.
    nonisolated static func evidenceMayLand(
        scanToken: Int, liveScanToken: Int,
        evidenceToken: Int, liveEvidenceToken: Int,
        root: URL?, liveRoot: URL?,
        entryIDs: Set<String>, liveEntryIDs: Set<String>
    ) -> Bool {
        scanToken == liveScanToken
            && evidenceToken == liveEvidenceToken
            && root == liveRoot
            && entryIDs == liveEntryIDs
    }

    private func applyScannedVariants(
        _ records: [ModelVariantRecord],
        entries: [AgentLibraryIndex.Entry]
    ) {
        variants = records
        agentIndex = entries
        // Evidence keyed on ids that no longer exist would render under the
        // wrong row after a delete/rename; drop it rather than show it.
        let live = Set(entries.map(\.id))
        robustnessByAgentID = robustnessByAgentID.filter { live.contains($0.key) }
    }

    private func reconcileSelectionsAfterScan() {
        if let selectedVariantID,
            !variants.contains(where: { $0.id == selectedVariantID })
        {
            self.selectedVariantID = nil
        }
        if let robustnessTargetVariantID,
            !variants.contains(where: { $0.id == robustnessTargetVariantID })
        {
            self.robustnessTargetVariantID = nil
        }
        if let selectedAdapterID,
            !adapters.contains(where: { $0.id == selectedAdapterID })
        {
            self.selectedAdapterID = nil
        }
        if selectedAdapterID == nil {
            selectedAdapterID = compatibleAdapters.first?.id
        }
        if let selectedAdapter {
            loadTrainer(from: selectedAdapter)
        }
    }

    public func refresh() {
        adapters = FineTuneStore.scan()
        let records = ModelVariantStore.scan()
        applyScannedVariants(records, entries: AgentLibraryIndex.summarize(records))
        hasScannedAgents = true
        scannedAgentRoot = VectorCatalog.projectRoot
        // A synchronous refresh is a fresh truth: an async scan already in
        // flight must not land on top of it.
        agentScanGeneration &+= 1
        reconcileSelectionsAfterScan()
    }

    /// View-facing status entry point. A non-nil message is a notice like
    /// any other panel event; nil just clears the single-slot line.
    public func setStatus(_ message: String?) {
        if let message {
            note(message)
        } else {
            status = nil
        }
    }

    /// `setStatus` at WARNING severity — for outcomes that succeeded but
    /// carry something the researcher has to know (an agent saved around a
    /// vector the server could not vouch for, 2026-08-06). It must survive
    /// the next status write, which is exactly what the persistent feed is
    /// for; an info-severity line would read as routine progress.
    public func setWarningStatus(_ message: String) {
        note(message, severity: .warning)
    }

    /// Creating a named adapter mints its workspace home —
    /// `adapters/<slug>/` with `training/` and `validation/` data folders —
    /// and defaults the adapter directory + data folder selections to those
    /// paths (repointable via the pickers). Idempotent: an existing home is
    /// adopted, never overwritten. A custom project directory homes the
    /// adapter under `<custom>/<slug>/` with the same layout.
    public func createAdapterProject() {
        let name = newAdapterName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            note("name the adapter first", severity: .info)
            return
        }
        let modelID = newAdapterBaseModelID.isEmpty
            ? (host?.loadedModelID ?? host?.selectedModelID ?? "")
            : newAdapterBaseModelID
        guard !modelID.isEmpty else {
            note("choose a base model first", severity: .info)
            return
        }
        let slug = FineTuneStore.slugify(name)
        let customParent = newAdapterProjectDirectory
            .trimmingCharacters(in: .whitespacesAndNewlines)

        do {
            let home = try FineTuneStore.createAdapterHome(
                slug: slug,
                under: customParent.isEmpty
                    ? nil : FineTuneStore.absoluteURL(customParent))
            let artifact = FineTuneArtifact(
                name: name,
                baseModelID: modelID,
                adapterDirectory: FineTuneStore.relativePath(for: home.root),
                adapterHash: FineTuneStore.hashFile(home.root.appending(component: "adapters.safetensors")),
                configHash: FineTuneStore.hashFile(home.root.appending(component: "adapter_config.json")),
                fineTuneType: Self.defaultFineTuneType.rawValue,
                rank: Self.defaultRank,
                scale: Float(Self.defaultScale),
                adaptedLayers: Self.defaultAdaptedLayers,
                trainingWorkspacePath: nil,
                trainingDataPath: FineTuneStore.relativePath(for: home.training),
                trainingDataHash: FineTuneStore.hashFileOrDirectory(home.training),
                validationDataPath: FineTuneStore.relativePath(for: home.validation),
                validationDataHash: FineTuneStore.hashFileOrDirectory(home.validation),
                trainingMode: FineTuneTrainingMode.document.rawValue,
                batchSize: Self.defaultBatchSize,
                iterations: Self.defaultIterations,
                learningRate: Self.defaultLearningRate,
                notes: "")
            let record = try FineTuneStore.save(artifact)
            refresh()
            // Adopt the SCANNED record for the new artifact (record ids are
            // URL paths; the scan's enumerator canonicalizes symlinked path
            // components, so compare symlink-resolved).
            let saved = adapters.first {
                $0.url.resolvingSymlinksInPath().path
                    == record.url.resolvingSymlinksInPath().path
            } ?? record
            selectedAdapterID = saved.id
            loadTrainer(from: saved)
            note(
                "created adapter \(name) at \(home.root.path) — drop files onto "
                    + "the Training data and Validation data rows to fill "
                    + "training/ and validation/",
                severity: .success)
        } catch {
            note("could not create adapter project: \(error)", severity: .error)
        }
    }

    /// The two per-adapter data folders drops can land in.
    public enum AdapterDataFolder: String, Sendable {
        case training
        case validation
    }

    /// Drag-and-drop entry point: COPY the dropped files into the adapter's
    /// training/validation folder in the workspace (never referenced in
    /// place), with the house same-name rule — identical bytes are quietly
    /// fine, differing bytes refuse with a plain-language message. An empty
    /// selection lazily mints the adapter's workspace home and points the
    /// panel at it.
    public func importDroppedFiles(_ urls: [URL], to target: AdapterDataFolder) {
        guard selectedAdapter != nil else {
            note(
                "choose an adapter before dropping \(target.rawValue) files",
                severity: .info)
            return
        }
        guard !urls.isEmpty else { return }
        let folder: URL
        do {
            folder = try resolvedDataFolder(for: target)
        } catch {
            note(
                "could not prepare the \(target.rawValue) folder: \(error)",
                severity: .error)
            return
        }
        let result = FineTuneStore.copyDroppedFiles(urls, into: folder)
        var parts: [String] = []
        if !result.copied.isEmpty {
            parts.append(
                "added \(result.copied.count) file\(result.copied.count == 1 ? "" : "s") "
                    + "to \(target.rawValue)/")
        }
        if !result.identical.isEmpty {
            parts.append(
                "\(result.identical.count) identical file\(result.identical.count == 1 ? "" : "s") already present")
        }
        if parts.isEmpty && result.refusals.isEmpty {
            parts.append("nothing to add to \(target.rawValue)/")
        }
        note(
            (parts + result.refusals).joined(separator: "; "),
            severity: result.refusals.isEmpty ? .success : .warning)
        refreshTrainingPlanQuietly()
    }

    /// Resolve the folder a drop lands in: the stored selection when it is a
    /// directory, the containing folder for a legacy `<folder>/train.jsonl`
    /// selection, or a freshly minted `adapters/<slug>/` home when nothing
    /// is set yet (the panel's selections are updated to point there).
    private func resolvedDataFolder(for target: AdapterDataFolder) throws -> URL {
        let stored = (target == .training ? trainingDataPath : validationDataPath)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if stored.isEmpty {
            let home = try FineTuneStore.createAdapterHome(
                slug: FineTuneStore.slugify(adapterName))
            if trainingDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                trainingDataPath = FineTuneStore.relativePath(for: home.training)
            }
            if validationDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                validationDataPath = FineTuneStore.relativePath(for: home.validation)
            }
            return target == .training ? home.training : home.validation
        }
        let resolved = FineTuneStore.absoluteURL(stored).standardizedFileURL
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return resolved
        }
        // Legacy convention stored "<folder>/train.jsonl"; a path with a file
        // extension means the containing folder is the data folder.
        return resolved.pathExtension.isEmpty
            ? resolved
            : resolved.deletingLastPathComponent()
    }

    /// After a drop, the visible data summary (the training plan, when one
    /// is showing) is recomputed without emitting a notice.
    private func refreshTrainingPlanQuietly() {
        guard trainingPlan != nil else { return }
        trainingPlan = recommendTrainingPlan(
            modelID: adapterBaseModelID,
            train: profileDataset(path: trainingDataPath, defaultFilename: "train.jsonl"),
            validation: profileDataset(
                path: validationDataPath, defaultFilename: "validation.jsonl"))
    }

    public func chooseAdapter(_ record: FineTuneArtifactRecord) {
        selectedAdapterID = record.id
        loadTrainer(from: record)
    }

    public func saveSelectedAdapter() {
        do {
            let updated = try persistSelectedAdapter()
            refresh()
            selectedAdapterID = updated.id
            loadTrainer(from: updated)
            note("saved adapter \(updated.artifact.name)", severity: .success)
        } catch {
            note("could not save adapter: \(error)", severity: .error)
        }
    }

    @discardableResult
    private func persistSelectedAdapter() throws -> FineTuneArtifactRecord {
        guard let record = selectedAdapter else {
            note("choose an adapter to edit", severity: .info)
            throw ChatServiceError(reason: "choose an adapter to edit")
        }
        let adapterURL = FineTuneStore.absoluteURL(adapterDirectory)
        let trainURL = trainingDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil
            : FineTuneStore.absoluteURL(trainingDataPath)
        let validURL = validationDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil
            : FineTuneStore.absoluteURL(validationDataPath)
        var artifact = record.artifact
        artifact.name = adapterName.trimmingCharacters(in: .whitespacesAndNewlines)
        artifact.baseModelID = adapterBaseModelID
        artifact.adapterDirectory = FineTuneStore.relativePath(for: adapterURL)
        artifact.adapterHash = FineTuneStore.hashFile(adapterURL.appending(component: "adapters.safetensors"))
        artifact.configHash = FineTuneStore.hashFile(adapterURL.appending(component: "adapter_config.json"))
        artifact.fineTuneType = fineTuneType.rawValue
        artifact.rank = rank
        artifact.scale = Float(scale)
        artifact.adaptedLayers = adaptedLayers
        artifact.trainingWorkspacePath = trainingWorkspacePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil
            : FineTuneStore.relativePath(for: FineTuneStore.absoluteURL(trainingWorkspacePath))
        artifact.trainingDataPath = trainURL.map(FineTuneStore.relativePath)
        artifact.trainingDataHash = trainURL.flatMap { datasetHash(for: $0, defaultFilename: "train.jsonl") }
        artifact.validationDataPath = validURL.map(FineTuneStore.relativePath)
        artifact.validationDataHash = validURL.flatMap { datasetHash(for: $0, defaultFilename: "validation.jsonl") }
        artifact.trainingMode = trainingMode.rawValue
        artifact.batchSize = batchSize
        artifact.iterations = iterations
        artifact.learningRate = learningRate
        artifact.notes = notes

        return try FineTuneStore.update(artifact, at: record.url)
    }

    public func beginTraining() {
        guard !isTraining else {
            note("adapter training is already running", severity: .info)
            return
        }
        guard selectedAdapter != nil else {
            note("choose an adapter to train", severity: .info)
            return
        }
        do {
            let record = try persistSelectedAdapter()
            refresh()
            selectedAdapterID = record.id
            let request = FineTuneTrainingRequest(
                name: adapterName,
                baseModelID: adapterBaseModelID,
                trainingMode: trainingMode,
                fineTuneType: fineTuneType.rawValue,
                rank: rank,
                scale: Float(scale),
                adaptedLayers: adaptedLayers,
                batchSize: batchSize,
                iterations: iterations,
                learningRate: learningRate,
                adapterDirectory: FineTuneStore.absoluteURL(adapterDirectory),
                trainingDataPath: trainingDataPath,
                validationDataPath: validationDataPath)
            startTraining(request: request, recordURL: record.url)
        } catch {
            note("could not start adapter training: \(error)", severity: .error)
        }
    }

    /// Which cancel semantics apply — pure so it is unit-testable: a
    /// recorded server job id means Cancel must ask the SERVER to cancel the
    /// durable job (cancelling only the local log stream would show
    /// "cancelled" while the cluster job runs on — the confirmed item-3
    /// bug); local in-process training is the only case where immediate
    /// cancellation is true.
    public enum TrainingCancelPath: Equatable, Sendable {
        case none
        case localTask
        case serverJob(id: String)
    }

    public nonisolated static func trainingCancelPath(
        isTraining: Bool, serverJobID: String?
    ) -> TrainingCancelPath {
        guard isTraining else { return .none }
        if let id = serverJobID, !id.isEmpty { return .serverJob(id: id) }
        return .localTask
    }

    /// "cancel requested" is the strongest claim a server cancel may make.
    public nonisolated static func serverCancelRequestedMessage(jobID: String) -> String {
        "cancel requested for job \(jobID) — the cluster decides when it "
            + "stops; streaming logs until it reaches a terminal state"
    }

    /// A failed cancel request never reads as cancelled.
    public nonisolated static func serverCancelFailureMessage(
        jobID: String, detail: String
    ) -> String {
        "cancel request failed: \(detail) — job \(jobID) is still running "
            + "on the cluster"
    }

    public func cancelTraining() {
        switch Self.trainingCancelPath(
            isTraining: isTraining, serverJobID: serverTrainingJobID)
        {
        case .none:
            return
        case .localTask:
            // In-process training: the task really does stop at the next
            // checkpoint, so "cancelling" is true here — and only here.
            note("cancelling adapter training...", severity: .warning)
            trainingProgress = "cancelling..."
            trainingTask?.cancel()
        case .serverJob(let jobID):
            Task { await requestServerTrainingCancel(jobID: jobID) }
        }
    }

    /// Server training cancel = the same durable job-cancel the Compute tab
    /// submits (`POST /api/jobs/{id}/cancel`), reported honestly: the log
    /// stream KEEPS running until the job reaches a terminal state — that
    /// ending is what flips `isTraining` off with the job's real final
    /// status. A failed request says so; nothing here ever claims
    /// "cancelled" for a job that may still be running.
    private func requestServerTrainingCancel(jobID: String) async {
        guard !serverCancelInFlight else { return }
        guard let client = host?.cluster.client else {
            note(
                "cancel failed: no server connection — job \(jobID) may still "
                    + "be running on the cluster; reconnect and cancel it from "
                    + "Compute",
                severity: .error)
            return
        }
        serverCancelInFlight = true
        defer { serverCancelInFlight = false }
        recordTrainingEvent("requesting cancel for server job \(jobID)...")
        do {
            try await client.cancelJob(jobID)
            let message = Self.serverCancelRequestedMessage(jobID: jobID)
            recordTrainingEvent(message)
            note(message, severity: .warning)
        } catch let error as ClusterClient.ClientError {
            // A 502 here means scancel itself failed — the allocation may
            // still be running; show the server's words (same rule as the
            // Compute tab), never a JSON blob.
            let message = Self.serverCancelFailureMessage(
                jobID: jobID,
                detail: ClusterClient.unwrappingDetail(error).description)
            recordTrainingEvent(message)
            note(message, severity: .error)
        } catch {
            let message = Self.serverCancelFailureMessage(
                jobID: jobID, detail: error.localizedDescription)
            recordTrainingEvent(message)
            note(message, severity: .error)
        }
    }

    // MARK: - Server training (two routes, chosen by the server's contract)

    /// A normalized server training plan waiting for the researcher's yes.
    /// Nothing is scheduled while this is set: the plan is what the server
    /// WOULD run (resolved revision, per-file hashes and row counts,
    /// schedule, dtype), and `planHash` rides back as `expectedPlanHash` so
    /// the plan confirmed and the plan run are provably the same one.
    public struct PendingServerTrainingPlan: Sendable, Identifiable {
        public var id = UUID()
        public var plan: RemoteFineTunePlan
        public var planHash: String
        /// The request to send on confirmation — the planned one with
        /// `expectedPlanHash` filled in.
        public var request: RemoteFineTuneRequest
        /// Display lines, in confirmation order.
        public var summary: [String]
        public var trainFileCount: Int
        public var validationFileCount: Int
        public var trainingResolvedPath: String
        public var validationResolvedPath: String
    }

    /// Set by `beginServerTraining` on a structured (explicit-split) route;
    /// cleared by `confirmServerTrainingPlan` / `cancelServerTrainingPlan`.
    /// A view renders it as a confirmation sheet; headless callers confirm
    /// directly. Non-nil means "nothing has been scheduled yet".
    public private(set) var pendingServerTrainingPlan: PendingServerTrainingPlan?

    /// The one-line truth about a server that predates
    /// `docs/CLUSTER-LORA-READINESS.md` §3.
    public nonisolated static let legacySplitServerNote =
        "this server predates explicit train/validation splits (no "
        + "remoteFineTune capability): the corpus uploads as one inline text "
        + "stream, document boundaries are joined, and the server labels a "
        + "trailing fraction \"validation\" without evaluating it — "
        + "exploratory only, never evidence"

    /// Refusal wording when the server has splits but not assistant-only
    /// loss masking: training instruction rows there would weight the prompt
    /// tokens too, which is a different (and unclaimed) intervention.
    public nonisolated static func instructionChatUnmaskedMessage(
        missing: [String]
    ) -> String {
        "this server does not announce assistant-only loss masking "
            + "(missing: \(missing.joined(separator: ", "))) — instruction/chat "
            + "rows would train on prompt tokens as well; run instruction "
            + "tuning in the Local workspace, or update the server"
    }

    /// The confirmation summary a researcher approves before anything is
    /// scheduled. Deliberately terse and in plan order: resolved revision,
    /// split sizes, schedule, dtype, plan hash.
    public nonisolated static func serverTrainingPlanSummary(
        plan: RemoteFineTunePlan,
        planHash: String,
        trainFileCount: Int,
        validationFileCount: Int
    ) -> [String] {
        var lines: [String] = []
        lines.append(
            "base revision: "
                + (plan.resolvedRevision ?? "not resolved by the server"))
        func split(_ label: String, _ files: Int, _ rows: Int?) -> String {
            let rowText = rows.map { " (\($0) rows)" } ?? ""
            return "\(label): \(files) file\(files == 1 ? "" : "s")\(rowText)"
        }
        lines.append(split("train", trainFileCount, plan.dataset?.rows(role: "train")))
        lines.append(
            split(
                "validation", validationFileCount,
                plan.dataset?.rows(role: "validation")))
        if let schedule = plan.schedule {
            let steps = schedule.totalSteps.map(String.init) ?? "?"
            let epochs = schedule.epochs.map(String.init) ?? "?"
            let batch = schedule.effectiveBatchSize.map(String.init) ?? "?"
            lines.append(
                "schedule: \(steps) steps over \(epochs) epoch(s), effective "
                    + "batch \(batch), \(schedule.lrSchedule ?? "?") LR schedule")
        }
        lines.append("dtype: " + (plan.dtype ?? "server default"))
        lines.append("training mode: " + (plan.trainingMode ?? "unstated"))
        lines.append("plan hash: \(planHash)")
        return lines
    }

    /// Queue LoRA training on the active server as a durable job.
    ///
    /// Two routes, decided by the server's `remoteFineTune` capability block
    /// (`docs/CLUSTER-LORA-READINESS.md` §3) and by what the researcher
    /// actually selected:
    ///
    /// - **Structured** (server announces `schemaVersion >= 2` +
    ///   `explicitSplits`, and both selections are JSONL): both folders
    ///   upload file-by-file with raw bytes + SHA-256, example boundaries
    ///   and the frozen split intact. The server answers a normalized plan,
    ///   which lands in `pendingServerTrainingPlan` — nothing is scheduled
    ///   until `confirmServerTrainingPlan()`.
    /// - **Legacy inline**: the old lossy `corpus.txt` upload, for servers
    ///   without the capability block and for mixed-document corpora, with a
    ///   note in the log saying exactly what is lost.
    ///
    /// Either way the adapter stays a server-side artifact (it never merges
    /// into the local library). This panel always sends
    /// `evidenceGrade: false`: it is the exploratory daemon route, and
    /// evidence-grade adapters run as Slurm jobs through the submission path.
    public func beginServerTraining() async {
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            note("connect a server workspace first", severity: .info)
            return
        }
        let base = adapterBaseModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else {
            note("pick a server base model first", severity: .info)
            return
        }
        // Capabilities decide the route. The cached snapshot is the
        // connect-time fetch; when it is absent, ask rather than assume the
        // old contract (assuming would silently pick the lossy route).
        var capabilities = host.cluster.capabilities
        if capabilities == nil {
            capabilities = try? await client.capabilities()
        }
        if let capabilities, capabilities.supportsStructuredFineTuneUpload {
            await beginStructuredServerTraining(
                client: client, host: host, base: base, capabilities: capabilities)
        } else {
            guard trainingMode == .document else {
                note(
                    "server fine-tuning on this server supports document "
                        + "adaptation only — instruction/chat tuning runs in "
                        + "the Local workspace", severity: .info)
                return
            }
            await beginLegacyInlineServerTraining(
                client: client, host: host, base: base,
                reason: Self.legacySplitServerNote)
        }
    }

    /// Structured route: build the byte-faithful two-folder payload, ask the
    /// server to normalize it into a plan, and park that plan for
    /// confirmation.
    private func beginStructuredServerTraining(
        client: ClusterClient,
        host: ChatService,
        base: String,
        capabilities: ClusterCapabilities
    ) async {
        if trainingMode == .instructionChat,
            !capabilities.supportsFineTuneInstructionChat
        {
            note(
                Self.instructionChatUnmaskedMessage(
                    missing: ["remoteFineTune.instructionChatAssistantMask"]),
                severity: .warning)
            return
        }

        let payload: FineTuneTrainingData.StructuredPayload
        do {
            payload = try FineTuneTrainingData.structuredPayload(
                trainingPath: trainingDataPath,
                validationPath: validationDataPath)
        } catch let problem as FineTuneTrainingData.Problem
            where problem.kind == .notStructured
        {
            // Mixed documents are what the legacy payload is FOR — but only
            // for document adaptation, and only out loud.
            guard trainingMode == .document else {
                note(
                    "\(problem.message) — instruction/chat tuning needs JSONL "
                        + "rows with user/assistant fields", severity: .warning)
                return
            }
            note(problem.message, severity: .warning)
            await beginLegacyInlineServerTraining(
                client: client, host: host, base: base,
                reason: "the selection is a mixed-document corpus, so it "
                    + "uploads as one inline text stream with the server's "
                    + "own invented split — exploratory only")
            return
        } catch {
            // A structured dataset that is broken or missing its held-out
            // half is refused, never downgraded to the lossy route.
            note("\(error)", severity: .warning)
            return
        }

        var request = structuredRequest(base: base, payload: payload)
        guard capabilities.supportsFineTunePlanEndpoint else {
            // Splits without a plan endpoint: the upload is still honest,
            // but nothing was confirmed — say so and run it as exploratory.
            recordTrainingEvent(
                "server announces explicit splits but no /api/finetune/plan — "
                    + "training starts without a confirmed plan (exploratory)")
            await runServerTraining(
                client: client, host: host, route: .structured(request),
                sent: "sent \(payload.trainFiles.count) train + "
                    + "\(payload.validationFiles.count) validation JSONL "
                    + "file(s) from \(payload.trainingResolvedPath)")
            return
        }

        trainingProgress = "requesting server training plan..."
        status = trainingProgress
        do {
            let planned = try await client.fineTunePlan(request)
            request.expectedPlanHash = planned.planHash
            let summary = Self.serverTrainingPlanSummary(
                plan: planned.plan,
                planHash: planned.planHash,
                trainFileCount: payload.trainFiles.count,
                validationFileCount: payload.validationFiles.count)
            pendingServerTrainingPlan = PendingServerTrainingPlan(
                plan: planned.plan,
                planHash: planned.planHash,
                request: request,
                summary: summary,
                trainFileCount: payload.trainFiles.count,
                validationFileCount: payload.validationFiles.count,
                trainingResolvedPath: payload.trainingResolvedPath,
                validationResolvedPath: payload.validationResolvedPath)
            note(
                "server training plan ready — confirm to queue it: "
                    + summary.joined(separator: " · "), severity: .info)
        } catch {
            trainingProgress = nil
            note("server training plan failed: \(error)", severity: .error)
        }
    }

    /// The v2 request this panel's controls describe. `evidenceGrade` is
    /// deliberately false: the panel targets the exploratory daemon route.
    private func structuredRequest(
        base: String,
        payload: FineTuneTrainingData.StructuredPayload
    ) -> RemoteFineTuneRequest {
        var hyperparameters = RemoteFineTuneRequest.Hyperparameters()
        hyperparameters.rank = rank
        hyperparameters.alpha = scale
        hyperparameters.learningRate = learningRate
        hyperparameters.batchSize = batchSize
        hyperparameters.maxSteps = iterations > 0 ? iterations : nil
        let trimmedName = adapterName.trimmingCharacters(in: .whitespacesAndNewlines)
        return RemoteFineTuneRequest(
            baseModelID: base,
            revision: selectedAdapter?.artifact.baseRevision,
            name: trimmedName.isEmpty ? nil : trimmedName,
            trainingMode: RemoteFineTuneRequest.wireTrainingMode(trainingMode),
            evidenceGrade: false,
            dataset: RemoteFineTuneRequest.Dataset(
                files: payload.files.map(RemoteFineTuneRequest.DatasetFile.init)),
            hyperparameters: hyperparameters)
    }

    /// Queue the plan the researcher just approved.
    public func confirmServerTrainingPlan() async {
        guard let pending = pendingServerTrainingPlan else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            note(
                "the server connection went away before the plan was "
                    + "confirmed — nothing was scheduled", severity: .warning)
            return
        }
        pendingServerTrainingPlan = nil
        await runServerTraining(
            client: client, host: host, route: .structured(pending.request),
            sent: "sent \(pending.trainFileCount) train + "
                + "\(pending.validationFileCount) validation JSONL file(s) from "
                + "\(pending.trainingResolvedPath); plan \(pending.planHash)")
    }

    /// Drop an unconfirmed plan. Nothing was scheduled, so this cancels
    /// nothing on the cluster — and says so.
    public func cancelServerTrainingPlan() {
        guard pendingServerTrainingPlan != nil else { return }
        pendingServerTrainingPlan = nil
        trainingProgress = nil
        note(
            "server training plan discarded — nothing was scheduled",
            severity: .info)
    }

    /// Legacy route: the lossy inline corpus, with `reason` recorded in the
    /// log so a run's provenance never has to guess why it took this path.
    private func beginLegacyInlineServerTraining(
        client: ClusterClient,
        host: ChatService,
        base: String,
        reason: String
    ) async {
        let payload: FineTuneTrainingData.Payload
        do {
            payload = try FineTuneTrainingData.inlineText(storedPath: trainingDataPath)
        } catch {
            note("\(error)", severity: .warning)
            return
        }
        await runServerTraining(
            client: client, host: host,
            route: .legacyInline(text: payload.text, note: reason),
            sent: "sent \(payload.fileCount) training "
                + "file\(payload.fileCount == 1 ? "" : "s") from \(payload.resolvedPath)")
    }

    private enum ServerTrainingRoute {
        case legacyInline(text: String, note: String)
        case structured(RemoteFineTuneRequest)
    }

    /// Submit on the chosen route, then stream the durable job's log. The
    /// submission and streaming are identical for both routes — only the
    /// request differs.
    private func runServerTraining(
        client: ClusterClient,
        host: ChatService,
        route: ServerTrainingRoute,
        sent: String
    ) async {
        serverTrainingLogTask?.cancel()
        isTraining = true
        trainingLog = []
        trainingProgress = "submitting server LoRA training..."
        status = trainingProgress
        let logTitle = "Server LoRA training"
        let displayLogID = host.startLiveLog(
            title: logTitle,
            initialLine: "submitting server LoRA training for "
                + "\(adapterBaseModelID)...")
        if case .legacyInline(_, let reason) = route {
            recordTrainingEvent(reason)
            host.updateLiveLog(id: displayLogID, title: logTitle, lines: trainingLog)
        }
        do {
            let jobID: String
            switch route {
            case .legacyInline(let text, _):
                let trimmedName = adapterName
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                jobID = try await client.fineTuneTrain(
                    baseModelID: adapterBaseModelID
                        .trimmingCharacters(in: .whitespacesAndNewlines),
                    text: text,
                    name: trimmedName.isEmpty ? nil : trimmedName,
                    rank: rank,
                    alpha: Double(scale),
                    iterations: iterations,
                    learningRate: learningRate)
            case .structured(let request):
                jobID = try await client.fineTuneTrainV2(request)
            }
            serverTrainingJobID = jobID
            recordTrainingEvent("server fine-tune queued as job \(jobID) — \(sent)")
            host.updateLiveLog(id: displayLogID, title: logTitle, lines: trainingLog)
            note("server fine-tune queued as job \(jobID) — \(sent); "
                + "streaming logs in the display pane", severity: .info)
            await host.cluster.refreshRemoteState()
            serverTrainingLogTask = Task { [weak self, weak host] in
                await self?.streamServerTrainingLog(
                    client: client,
                    jobID: jobID,
                    displayLogID: displayLogID,
                    logTitle: logTitle,
                    host: host)
            }
        } catch {
            isTraining = false
            let message = "server fine-tune failed: \(error)"
            recordTrainingEvent(message)
            host.updateLiveLog(id: displayLogID, title: logTitle, lines: trainingLog)
            note("server fine-tune failed: \(error)", severity: .error)
        }
    }

    private func streamServerTrainingLog(
        client: ClusterClient,
        jobID: String,
        displayLogID: UUID,
        logTitle: String,
        host: ChatService?
    ) async {
        do {
            try await client.streamJobLog(jobID: jobID) { [weak self, weak host] line in
                await MainActor.run {
                    guard let self else { return }
                    self.recordTrainingEvent(line)
                    host?.updateLiveLog(id: displayLogID, title: logTitle, lines: self.trainingLog)
                }
            }
            if let job = try? await client.job(jobID) {
                let message = "server fine-tune job \(jobID): \(job.status)"
                recordTrainingEvent(message)
                note(message, severity: job.status == "failed" ? .error : .info)
            } else {
                recordTrainingEvent("server fine-tune job \(jobID): log stream ended")
            }
            isTraining = false
            trainingProgress = trainingLog.last
            serverTrainingLogTask = nil
            if serverTrainingJobID == jobID { serverTrainingJobID = nil }
            host?.updateLiveLog(id: displayLogID, title: logTitle, lines: trainingLog)
            await host?.cluster.refreshRemoteState()
            await host?.refreshServerSteeringArtifacts()
        } catch {
            if Task.isCancelled {
                // The STREAM was cancelled (a new training superseded it) —
                // the job itself may still be running; say which.
                recordTrainingEvent(
                    "server fine-tune log stream cancelled — job \(jobID) "
                        + "may still be running (check Compute)")
            } else {
                recordTrainingEvent("server fine-tune log stream failed: \(error)")
            }
            isTraining = false
            trainingProgress = trainingLog.last
            serverTrainingLogTask = nil
            if serverTrainingJobID == jobID { serverTrainingJobID = nil }
            status = trainingProgress
            host?.updateLiveLog(id: displayLogID, title: logTitle, lines: trainingLog)
        }
    }

    public func analyzeTrainingPlan() {
        guard selectedAdapter != nil else {
            note("choose an adapter before planning training", severity: .info)
            return
        }
        let train = profileDataset(path: trainingDataPath, defaultFilename: "train.jsonl")
        let validation = profileDataset(path: validationDataPath, defaultFilename: "validation.jsonl")
        let plan = recommendTrainingPlan(
            modelID: adapterBaseModelID,
            train: train,
            validation: validation)
        trainingPlan = plan
        note("planned adapter training: \(plan.dataScale), \(train.exampleCount) train examples, ~\(train.estimatedTokens) train tokens", severity: .success)
    }

    public func applyTrainingPlan() {
        guard let plan = trainingPlan else {
            note("analyze training data before applying a plan", severity: .info)
            return
        }
        fineTuneType = .lora
        rank = plan.recommendedRank
        scale = plan.recommendedScale
        adaptedLayers = plan.recommendedLayers
        batchSize = plan.recommendedBatchSize
        iterations = plan.recommendedIterations
        learningRate = plan.recommendedLearningRate
        note("applied recommended adapter parameters", severity: .success)
    }

    public func registerAdapter() {
        guard let host else {
            note("load app state before registering an adapter", severity: .info)
            return
        }
        let name = adapterName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            note("name the adapter artifact first", severity: .info)
            return
        }
        let directory = FineTuneStore.absoluteURL(
            adapterDirectory.trimmingCharacters(in: .whitespacesAndNewlines))
        guard FileManager.default.fileExists(atPath: directory.path) else {
            note("adapter directory not found", severity: .info)
            return
        }

        let adapterURL = directory.appending(component: "adapters.safetensors")
        let configURL = directory.appending(component: "adapter_config.json")
        let trainURL = trainingDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil
            : FineTuneStore.absoluteURL(trainingDataPath)
        let validURL = validationDataPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil
            : FineTuneStore.absoluteURL(validationDataPath)
        let artifact = FineTuneArtifact(
            name: name,
            baseModelID: host.loadedModelID ?? host.selectedModelID,
            adapterDirectory: FineTuneStore.relativePath(for: directory),
            adapterHash: FineTuneStore.hashFile(adapterURL),
            configHash: FineTuneStore.hashFile(configURL),
            fineTuneType: fineTuneType.rawValue,
            rank: rank,
            scale: Float(scale),
            adaptedLayers: adaptedLayers,
            trainingWorkspacePath: trainingWorkspacePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? nil
                : FineTuneStore.relativePath(for: FineTuneStore.absoluteURL(trainingWorkspacePath)),
            trainingDataPath: trainURL.map(FineTuneStore.relativePath),
            trainingDataHash: trainURL.flatMap { datasetHash(for: $0, defaultFilename: "train.jsonl") },
            validationDataPath: validURL.map(FineTuneStore.relativePath),
            validationDataHash: validURL.flatMap { datasetHash(for: $0, defaultFilename: "validation.jsonl") },
            trainingMode: trainingMode.rawValue,
            batchSize: batchSize,
            iterations: iterations,
            learningRate: learningRate,
            notes: notes)

        do {
            let record = try FineTuneStore.save(artifact)
            refresh()
            selectedAdapterID = record.id
            note("registered adapter \(record.artifact.name)", severity: .success)
        } catch {
            note("could not register adapter: \(error)", severity: .error)
        }
    }

    public func captureVariant() async {
        guard let host else {
            note("load app state before capturing a variant", severity: .info)
            return
        }
        let name = variantName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            note("name the model variant first", severity: .info)
            return
        }
        // Server workspace: the definition is composed from the SAME control
        // state an inline send uses (server base model, server vector/adapter
        // refs) and saved into the local variant store — definitions are
        // git-versioned recipes, visible in every workspace; only the refs
        // are per-substrate.
        if host.cluster.computeTarget == .server {
            // Honest capture: any enabled slot that fails to resolve against
            // the server catalog would silently vanish from the definition —
            // a provenance hazard (an empty-injections "steered" recipe).
            // Close the staleness window with one catalog refresh, then
            // REFUSE if anything still drops; never save a lying definition.
            var resolution = host.serverControlResolution()
            if !resolution.unresolvedVectorIDs.isEmpty {
                await host.catalog.refreshRemoteVectors()
                resolution = host.serverControlResolution()
            }
            if let refusal = InlineVariantComposer.unresolvedSlotRefusal(
                unresolvedVectorIDs: resolution.unresolvedVectorIDs)
            {
                note(refusal, severity: .error)
                return
            }
            guard
                let variant = InlineVariantComposer.captureArtifact(
                    state: resolution.state,
                    name: name,
                    loadedServerModelID: host.cluster.remoteState?.loadedModel)
            else {
                note("select (or load) a server model before capturing a variant", severity: .info)
                return
            }
            do {
                let record = try ModelVariantStore.save(variant)
                refresh()
                selectedVariantID = record.id
                note("captured model variant \(record.artifact.name) "
                    + "(server base \(record.artifact.baseModelID))", severity: .success)
            } catch {
                note("could not capture model variant: \(error)", severity: .error)
            }
            return
        }
        let adapterRefs: [ModelVariantArtifact.AdapterRef] = (
            host.adaptersEnabled ? host.selectedRuntimeAdapter : nil
        ).map { record in
            [
                .init(
                    name: record.artifact.name,
                    artifactPath: FineTuneStore.relativePath(for: record.url),
                    adapterDirectory: record.artifact.adapterDirectory,
                    adapterHash: record.artifact.adapterHash)
            ]
        } ?? []
        let injectionRefs: [ModelVariantArtifact.InjectionRef] = host.slots.compactMap { slot in
            guard slot.enabled,
                let artifact = host.artifact(for: slot)
            else { return nil }
            return .init(
                concept: artifact.sidecar.concept,
                vectorArtifactID: artifact.id,
                layer: Int(slot.layer),
                alpha: slot.alpha)
        }
        let neutralBasis = host.removeNeutralDirectionsAtSteering
            ? host.selectedNeutralPCBasis
            : nil
        let variant = ModelVariantArtifact(
            name: name,
            baseModelID: host.loadedModelID ?? host.selectedModelID,
            adapters: adapterRefs,
            injections: injectionRefs,
            bandWidth: host.layerBandWidth,
            alphaInNormUnits: host.alphaInNormUnits,
            neutralPCBasisPath: neutralBasis.map(NeutralPCStore.relativePath),
            neutralPCBasisLabel: neutralBasis?.label,
            promptMode: host.promptMode.rawValue,
            qwenThinkingEnabled: host.qwenThinkingEnabled,
            temperature: host.temperature,
            systemPrompt: host.systemPrompt)

        do {
            let record = try ModelVariantStore.save(variant)
            refresh()
            selectedVariantID = record.id
            note("captured model variant \(record.artifact.name)", severity: .success)
        } catch {
            note("could not capture model variant: \(error)", severity: .error)
        }
    }

    public func deleteSelectedVariant() {
        guard let selectedVariant else {
            note("select a model variant to delete", severity: .info)
            return
        }
        do {
            try ModelVariantStore.delete(selectedVariant)
            refresh()
            note("deleted model variant \(selectedVariant.artifact.name)", severity: .success)
        } catch {
            note("could not delete model variant: \(error)", severity: .error)
        }
    }

    /// Cancel the in-flight LOCAL robustness check (App gap A1): raises the
    /// cooperative flag the run polls between generations. Server-routed
    /// checks are durable jobs with their own cancel affordance.
    public func cancelRobustnessCheck() {
        guard isRobustnessRunning, !robustnessCancelRequested else { return }
        robustnessCancelRequested = true
        note("cancelling robustness check — stops after the current generation", severity: .warning)
    }

    public func runRobustnessCheck() {
        guard !isRobustnessRunning else {
            note("agent robustness check is already running", severity: .info)
            return
        }
        if case .server(let path) = robustnessTarget {
            // A server-stored agent exists only on its server: it can never
            // run in-process, and switching to Local clears nothing silently —
            // the refusal names the scoping rule.
            guard let host, case .server = host.cluster.activeWorkspace else {
                note("the chosen agent is stored on a server — switch the "
                    + "Compute workspace to that server to check it", severity: .info)
                return
            }
            runServerStoredRobustnessCheck(path: path, host: host)
            return
        }
        guard let target = robustnessTargetVariant else {
            note("choose an agent in the Robustness Check section first", severity: .info)
            return
        }
        if let host, case .server = host.cluster.activeWorkspace {
            runServerVariantRobustnessCheck(for: target, host: host)
            return
        }
        isRobustnessRunning = true
        robustnessCancelRequested = false
        robustnessReport = nil
        lastRobustnessDirectory = nil
        liveRobustnessOutputs = []
        liveRobustnessJudgments = []
        liveRobustnessJudgeStatus = nil
        liveRobustnessFailure = nil
        note("running robustness check for \(target.artifact.name)…", severity: .info)
        let judge = robustnessUseJudge
            ? robustnessJudgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
            : ""
        let batteryFile = robustnessBatteryFile
        let promptsFile = robustnessPromptsFile
        let maxPrompts = robustnessMaxPrompts
        let maxTokens = robustnessMaxTokens
        let presetID = robustnessPresetID == VariantRobustness.customPresetID
            ? nil
            : robustnessPresetID
        Task { [weak self] in
            do {
                let result = try await VariantRobustness.run(
                    record: target,
                    batteryFile: batteryFile,
                    coherencePromptsFile: promptsFile,
                    maxCoherencePrompts: maxPrompts,
                    maxTokens: maxTokens,
                    presetID: presetID,
                    judgeModel: judge.isEmpty ? nil : judge,
                    shouldCancel: { [weak self] in
                        await self?.robustnessCancelRequested ?? false
                    },
                    progress: { [weak self] event in
                        await MainActor.run {
                            self?.handleRobustnessProgress(event)
                        }
                    })
                await MainActor.run {
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessJudgeStatus = nil
                    if let result {
                        self?.robustnessReport = result.report
                        self?.lastRobustnessDirectory = result.directory.path
                        self?.note("robustness check complete: \(result.directory.lastPathComponent)", severity: .success)
                    } else {
                        // Cancelled by user: never an error, and never a
                        // report — a partial battery is not evidence.
                        self?.note("robustness check cancelled by user — "
                            + "no report written (a partial battery is never scored)", severity: .warning)
                    }
                }
            } catch {
                await MainActor.run {
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessFailure = "\(error)"
                    self?.note("robustness check failed: \(error)", severity: .error)
                }
            }
        }
    }

    /// Server-workspace robustness: batteries and coherence prompts stay
    /// local recipe data and scoring stays the same pure code — only
    /// generation goes through the ACTIVE server, as non-stream inline
    /// variant-generates of the variant under test (baseline arm =
    /// `stripInterventions`, greedy). Pre-flight refuses loudly when the
    /// variant's refs don't resolve server-side (same applicability seam the
    /// Variants tab uses), instead of failing item-by-item mid-run.
    private func runServerVariantRobustnessCheck(
        for selectedVariant: ModelVariantRecord, host: ChatService
    ) {
        guard let client = host.cluster.client else {
            note("invalid server URL", severity: .error)
            return
        }
        let substrateLabel = host.cluster.substrateLabel
        let applicability = host.serverApplicability(of: selectedVariant.artifact)
        if case .blocked(let reasons) = applicability {
            let message =
                "robustness check cannot run on \(substrateLabel): "
                + reasons.joined(separator: "; ")
            note(message, severity: .error)
            liveRobustnessFailure = message
            return
        }
        isRobustnessRunning = true
        robustnessReport = nil
        lastRobustnessDirectory = nil
        liveRobustnessOutputs = []
        liveRobustnessJudgments = []
        liveRobustnessJudgeStatus = nil
        liveRobustnessFailure = nil
        note("running robustness check for \(selectedVariant.artifact.name) "
            + "on \(substrateLabel)…", severity: .info)
        let judge = robustnessUseJudge
            ? robustnessJudgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
            : ""
        let batteryFile = robustnessBatteryFile
        let promptsFile = robustnessPromptsFile
        let maxPrompts = robustnessMaxPrompts
        let maxTokens = robustnessMaxTokens
        let presetID = robustnessPresetID == VariantRobustness.customPresetID
            ? nil
            : robustnessPresetID
        let artifact = selectedVariant.artifact
        Task { [weak self] in
            do {
                let result = try await VariantRobustness.runViaServer(
                    record: selectedVariant,
                    batteryFile: batteryFile,
                    coherencePromptsFile: promptsFile,
                    maxCoherencePrompts: maxPrompts,
                    maxTokens: maxTokens,
                    presetID: presetID,
                    judgeModel: judge.isEmpty ? nil : judge,
                    generate: { request in
                        // The variant under test rides along as an INLINE spec
                        // (the artifact IS the upload-payload schema); the
                        // server stamps it source:"inline" — no stored-variant
                        // identity is claimed for a local definition.
                        try await client.variantGenerate(
                            selection: .inline(artifact),
                            messages: request.messages.map {
                                ChatWireMessage(
                                    role: $0["role"] ?? "user",
                                    content: $0["content"] ?? "")
                            },
                            maxTokens: request.maxTokens,
                            temperature: request.temperature,
                            promptMode: artifact.promptMode,
                            systemPrompt: nil,  // the spec's own system prompt rules
                            stripInterventions: request.stripInterventions)
                    },
                    scoreBattery: { request in
                        // Format-2 batteries only: one call per side, scored
                        // by the server under the BATTERY's arming (§23). The
                        // same inline spec identifies the variant; the pin
                        // makes the far side prove it holds our bytes.
                        try await client.variantBatteryEvaluate(
                            selection: .inline(artifact),
                            battery: request.batteryFile,
                            batteryHash: request.batteryHash,
                            stripInterventions: request.stripInterventions)
                    },
                    progress: { [weak self] event in
                        await MainActor.run {
                            self?.handleRobustnessProgress(event)
                        }
                    })
                await MainActor.run {
                    self?.robustnessReport = result.report
                    self?.lastRobustnessDirectory = result.directory.path
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessJudgeStatus = nil
                    self?.note("robustness check complete on \(substrateLabel): "
                        + result.directory.lastPathComponent, severity: .success)
                }
            } catch {
                await MainActor.run {
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessFailure = "\(error)"
                    self?.note("robustness check failed: \(error)", severity: .error)
                }
            }
        }
    }

    /// Robustness check for a SERVER-STORED agent: fetch its spec + content
    /// hash from the active server (`variantDetail`), then run the same
    /// server-generation flow with the STORED selection — the server resolves
    /// the artifact from its own tree, the hash pins exact provenance, and
    /// `stripInterventions: true` is the baseline arm. Batteries, coherence
    /// prompts, and scoring stay local recipe data + pure code; the report
    /// lands in this app workspace's runs/ tree (evidence comes home).
    private func runServerStoredRobustnessCheck(path: String, host: ChatService) {
        guard let client = host.cluster.client else {
            note("invalid server URL", severity: .error)
            return
        }
        let substrateLabel = host.cluster.substrateLabel
        isRobustnessRunning = true
        robustnessReport = nil
        lastRobustnessDirectory = nil
        liveRobustnessOutputs = []
        liveRobustnessJudgments = []
        liveRobustnessJudgeStatus = nil
        liveRobustnessFailure = nil
        note("fetching stored agent spec from \(substrateLabel)…", severity: .info)
        let judge = robustnessUseJudge
            ? robustnessJudgeModel.trimmingCharacters(in: .whitespacesAndNewlines)
            : ""
        let batteryFile = robustnessBatteryFile
        let promptsFile = robustnessPromptsFile
        let maxPrompts = robustnessMaxPrompts
        let maxTokens = robustnessMaxTokens
        let presetID = robustnessPresetID == VariantRobustness.customPresetID
            ? nil
            : robustnessPresetID
        Task { [weak self] in
            do {
                let detail = try await client.variantDetail(path: path)
                await MainActor.run {
                    self?.note("running robustness check for "
                        + "\(detail.variant.name) on \(substrateLabel)…", severity: .info)
                }
                let artifact = detail.variant
                let hash = detail.hash
                let result = try await VariantRobustness.runViaServer(
                    variant: artifact,
                    variantPath: path,
                    variantHash: hash,
                    batteryFile: batteryFile,
                    coherencePromptsFile: promptsFile,
                    maxCoherencePrompts: maxPrompts,
                    maxTokens: maxTokens,
                    presetID: presetID,
                    judgeModel: judge.isEmpty ? nil : judge,
                    generate: { request in
                        try await client.variantGenerate(
                            selection: .stored(path: path, hash: hash),
                            messages: request.messages.map {
                                ChatWireMessage(
                                    role: $0["role"] ?? "user",
                                    content: $0["content"] ?? "")
                            },
                            maxTokens: request.maxTokens,
                            temperature: request.temperature,
                            promptMode: artifact.promptMode,
                            systemPrompt: nil,  // the stored spec's own system prompt rules
                            stripInterventions: request.stripInterventions)
                    },
                    scoreBattery: { request in
                        // Format-2 batteries only (§23): the stored artifact
                        // identifies the variant (path + hash), the battery
                        // rides as its own pin, and the arming is the
                        // battery's on both sides.
                        try await client.variantBatteryEvaluate(
                            selection: .stored(path: path, hash: hash),
                            battery: request.batteryFile,
                            batteryHash: request.batteryHash,
                            stripInterventions: request.stripInterventions)
                    },
                    progress: { [weak self] event in
                        await MainActor.run {
                            self?.handleRobustnessProgress(event)
                        }
                    })
                await MainActor.run {
                    self?.robustnessReport = result.report
                    self?.lastRobustnessDirectory = result.directory.path
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessJudgeStatus = nil
                    self?.note("robustness check complete on \(substrateLabel): "
                        + result.directory.lastPathComponent, severity: .success)
                }
            } catch {
                await MainActor.run {
                    self?.isRobustnessRunning = false
                    self?.liveRobustnessFailure = "\(error)"
                    self?.note("robustness check failed: \(error)", severity: .error)
                }
            }
        }
    }

    public func applyRobustnessPreset(_ presetID: String) {
        robustnessPresetID = presetID
        guard let preset = VariantRobustness.preset(id: presetID) else { return }
        robustnessBatteryFile = preset.batteryFile
        robustnessPromptsFile = preset.coherencePromptsFile
        robustnessMaxPrompts = preset.maxCoherencePrompts
        robustnessMaxTokens = preset.maxTokens
    }

    public func clearRobustnessViewer() {
        liveRobustnessOutputs = []
        liveRobustnessJudgments = []
        liveRobustnessJudgeStatus = nil
        liveRobustnessFailure = nil
        robustnessReport = nil
        lastRobustnessDirectory = nil
    }

    private func handleRobustnessProgress(_ event: VariantRobustnessProgress) {
        switch event {
        case .outputStarted(let kind, let index, let total, let prompt, let side):
            upsertLiveRobustnessOutput(
                .init(
                    kind: kind,
                    index: index,
                    total: total,
                    prompt: prompt,
                    side: side,
                    output: "",
                    isComplete: false))
            // Per-item progress echo — deliberately NOT a notice (A15).
            status = "robustness \(kind.lowercased()) \(side.lowercased()) \(index)/\(total)…"
        case .outputChunk(let kind, let index, let side, let output):
            if let position = liveRobustnessOutputs.firstIndex(where: {
                $0.kind == kind && $0.index == index && $0.side == side
            }) {
                liveRobustnessOutputs[position].output = output
            }
        case .outputCompleted(let kind, let index, let prompt, let side, let output):
            let existingTotal = liveRobustnessOutputs.first {
                $0.kind == kind && $0.index == index && $0.side == side
            }?.total ?? index
            upsertLiveRobustnessOutput(
                .init(
                    kind: kind,
                    index: index,
                    total: existingTotal,
                    prompt: prompt,
                    side: side,
                    output: output,
                    isComplete: true))
        case .judgeStarted(let index, let total, _):
            liveRobustnessJudgeStatus = "judging coherence \(index)/\(total)…"
            status = liveRobustnessJudgeStatus
        case .judgeCompleted(let index, let prompt, let result, let response):
            upsertLiveRobustnessJudgment(
                .init(index: index, prompt: prompt, result: result, response: response))
            liveRobustnessJudgeStatus = "judged coherence \(index): \(result)"
            status = liveRobustnessJudgeStatus
        }
    }

    private func upsertLiveRobustnessOutput(_ output: LiveRobustnessOutput) {
        if let index = liveRobustnessOutputs.firstIndex(where: { $0.id == output.id }) {
            liveRobustnessOutputs[index] = output
        } else {
            liveRobustnessOutputs.append(output)
        }
    }

    private func upsertLiveRobustnessJudgment(_ judgment: LiveRobustnessJudgment) {
        if let index = liveRobustnessJudgments.firstIndex(where: { $0.id == judgment.id }) {
            liveRobustnessJudgments[index] = judgment
        } else {
            liveRobustnessJudgments.append(judgment)
        }
    }

    private func loadTrainer(from record: FineTuneArtifactRecord) {
        adapterName = record.artifact.name
        adapterBaseModelID = record.artifact.baseModelID
        adapterDirectory = record.artifact.adapterDirectory
        adapterProjectDirectory = FineTuneStore.relativePath(
            for: FineTuneStore.absoluteURL(record.artifact.adapterDirectory).deletingLastPathComponent())
        trainingWorkspacePath = record.artifact.trainingWorkspacePath ?? ""
        trainingDataPath = record.artifact.trainingDataPath ?? ""
        validationDataPath = record.artifact.validationDataPath ?? ""
        trainingMode = FineTuneTrainingMode(rawValue: record.artifact.trainingMode ?? "")
            ?? .document
        fineTuneType = FineTuneType(rawValue: record.artifact.fineTuneType) ?? .lora
        rank = record.artifact.rank
        scale = Double(record.artifact.scale)
        adaptedLayers = record.artifact.adaptedLayers
        batchSize = record.artifact.batchSize
        iterations = record.artifact.iterations
        learningRate = record.artifact.learningRate
        notes = record.artifact.notes
        trainingPlan = nil
    }

    private func startTraining(request: FineTuneTrainingRequest, recordURL: URL) {
        isTraining = true
        trainingProgress = "starting adapter training..."
        trainingLog = []
        note("started adapter training for \(request.name)", severity: .info)

        trainingTask = Task {
            do {
                let result = try await FineTuneTrainer.train(request) { event in
                    Task { @MainActor in
                        self.recordTrainingEvent(event.description)
                    }
                }
                try finishTraining(result: result, recordURL: recordURL)
            } catch {
                finishTraining(error: error)
            }
        }
    }

    private func recordTrainingEvent(_ message: String) {
        trainingProgress = message
        trainingLog.append(message)
        if trainingLog.count > 80 {
            trainingLog.removeFirst(trainingLog.count - 80)
        }
        status = message
    }

    private func finishTraining(result: FineTuneTrainingResult, recordURL: URL) throws {
        guard let data = try? Data(contentsOf: recordURL),
            var artifact = try? JSONDecoder().decode(FineTuneArtifact.self, from: data)
        else {
            throw ChatServiceError(reason: "training finished, but adapter sidecar could not be reloaded")
        }
        artifact.adapterHash = FineTuneStore.hashFile(result.adapterURL)
        artifact.configHash = FineTuneStore.hashFile(result.configURL)
        // Substrate/format stamp at training save — the pinned cross-engine
        // contract (the server twin writes "python-hf-transformers" /
        // "hf-peft-lora"). Application paths refuse explicitly-foreign stamps.
        artifact.substrate = AdapterSubstrateGate.localSubstrate
        artifact.adapterFormat = AdapterSubstrateGate.localAdapterFormat
        if let trainingDataPath = artifact.trainingDataPath {
            artifact.trainingDataHash = datasetHash(
                for: FineTuneStore.absoluteURL(trainingDataPath),
                defaultFilename: "train.jsonl")
        }
        if let validationDataPath = artifact.validationDataPath {
            artifact.validationDataHash = datasetHash(
                for: FineTuneStore.absoluteURL(validationDataPath),
                defaultFilename: "validation.jsonl")
        }
        let updated = try FineTuneStore.update(artifact, at: recordURL)
        refresh()
        selectedAdapterID = updated.id
        loadTrainer(from: updated)
        isTraining = false
        trainingTask = nil
        let message = "training complete: \(result.trainExamples) train examples, \(result.validationExamples) validation examples"
        trainingProgress = message
        trainingLog.append(message)
        note(message, severity: .success)
    }

    private func finishTraining(error: Error) {
        isTraining = false
        trainingTask = nil
        let message: String
        if Task.isCancelled || String(describing: error).contains("cancelled") {
            message = "adapter training cancelled"
        } else {
            message = "adapter training failed: \(error)"
        }
        trainingProgress = message
        trainingLog.append(message)
        note(message, severity: message.contains("failed") ? .error : .warning)
    }

    private func datasetHash(for url: URL, defaultFilename: String) -> String? {
        let folder = datasetFolderURL(path: url.path, defaultFilename: defaultFilename)
        return FineTuneStore.hashFileOrDirectory(folder) ?? FineTuneStore.hashFileOrDirectory(url)
    }

    private func profileDataset(path: String, defaultFilename: String) -> DatasetProfile {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return .init(
                path: "", exampleCount: 0, estimatedTokens: 0, medianCharacters: 0,
                instructionRows: 0, textRows: 0, unreadableRows: 0)
        }
        let url = datasetURL(path: trimmed, defaultFilename: defaultFilename)
        let folder = datasetFolderURL(path: trimmed, defaultFilename: defaultFilename)
        var examples: [TrainingExample] = []
        var unreadableSources = 0

        if FileManager.default.fileExists(atPath: url.path) {
            let result = parseTrainingDataFile(url)
            examples.append(contentsOf: result.examples)
            unreadableSources += result.unreadableSources
        }

        if let urls = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            let primaryPath = url.standardizedFileURL.path
            for sourceURL in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard sourceURL.standardizedFileURL.path != primaryPath,
                    Self.isSupportedTrainingSource(sourceURL),
                    !sourceURL.lastPathComponent.lowercased().hasPrefix("readme")
                else { continue }
                let result = parseTrainingDataFile(sourceURL)
                examples.append(contentsOf: result.examples)
                unreadableSources += result.unreadableSources
            }
        } else if !FileManager.default.fileExists(atPath: url.path) {
            unreadableSources += 1
        }

        let lengths = examples.map { $0.text.count }.sorted()
        let median = lengths.isEmpty ? 0 : lengths[lengths.count / 2]
        let estimatedTokens = examples.reduce(0) { $0 + Self.estimateTokens($1.text) }
        return .init(
            path: FineTuneStore.relativePath(for: url),
            exampleCount: examples.count,
            estimatedTokens: estimatedTokens,
            medianCharacters: median,
            instructionRows: examples.filter(\.isInstruction).count,
            textRows: examples.filter { !$0.isInstruction }.count,
            unreadableRows: unreadableSources + examples.filter(\.isUnreadable).count)
    }

    private func datasetURL(path: String, defaultFilename: String) -> URL {
        var url = FineTuneStore.absoluteURL(path)
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            url = url.appending(component: defaultFilename)
        }
        return url
    }

    private func datasetFolderURL(path: String, defaultFilename: String) -> URL {
        let url = FineTuneStore.absoluteURL(path)
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return url
        }
        if url.lastPathComponent == defaultFilename {
            return url.deletingLastPathComponent()
        }
        return url.deletingLastPathComponent()
    }

    private struct TrainingExample {
        var text: String
        var isInstruction: Bool
        var isUnreadable: Bool
    }

    private func parseTrainingExamples(_ content: String) -> [TrainingExample] {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if let data = trimmed.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        {
            return array.compactMap(Self.trainingExample(from:))
        }
        return trimmed.components(separatedBy: .newlines).compactMap { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { return nil }
            guard let data = line.data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                return .init(text: line, isInstruction: false, isUnreadable: false)
            }
            return Self.trainingExample(from: object)
        }
    }

    private func parseTrainingDataFile(_ url: URL) -> (examples: [TrainingExample], unreadableSources: Int) {
        switch url.pathExtension.lowercased() {
        case "json", "jsonl", "":
            guard let content = try? String(contentsOf: url, encoding: .utf8) else {
                return ([], 1)
            }
            return (parseTrainingExamples(content), 0)
        case "txt", "md":
            guard let content = try? String(contentsOf: url, encoding: .utf8) else {
                return ([], 1)
            }
            let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return ([], 0) }
            return ([.init(text: text, isInstruction: false, isUnreadable: false)], 0)
        case "pdf":
            guard let text = Self.extractPDFText(url), !text.isEmpty else {
                return ([], 1)
            }
            return ([.init(text: text, isInstruction: false, isUnreadable: false)], 0)
        default:
            return ([], 0)
        }
    }

    private static func isSupportedTrainingSource(_ url: URL) -> Bool {
        ["json", "jsonl", "txt", "md", "pdf"].contains(url.pathExtension.lowercased())
    }

    private static func extractPDFText(_ url: URL) -> String? {
        guard let document = PDFDocument(url: url) else { return nil }
        let text = (0 ..< document.pageCount).compactMap { index in
            document.page(at: index)?.string?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { !$0.isEmpty }
        .joined(separator: "\n\n")
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func trainingExample(from object: [String: Any]) -> TrainingExample? {
        if let text = object["text"] as? String,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return .init(text: text, isInstruction: false, isUnreadable: false)
        }
        let system = object["system"] as? String ?? ""
        let user = object["user"] as? String ?? object["prompt"] as? String ?? ""
        let assistant = object["assistant"] as? String ?? object["completion"] as? String ?? ""
        let joined = [system, user, assistant]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        guard !joined.isEmpty else { return nil }
        return .init(text: joined, isInstruction: !assistant.isEmpty || !user.isEmpty, isUnreadable: false)
    }

    private static func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        let charEstimate = Int((Double(text.count) / 4.0).rounded(.up))
        let wordEstimate = Int((Double(words) * 1.35).rounded(.up))
        return max(1, max(charEstimate, wordEstimate))
    }

    private func recommendTrainingPlan(
        modelID: String,
        train: DatasetProfile,
        validation: DatasetProfile
    ) -> TrainingPlan {
        let modelSize = Self.modelSizeBillions(modelID)
        let tokens = train.estimatedTokens
        let scaleName: String
        switch tokens {
        case 0: scaleName = "empty data"
        case ..<25_000: scaleName = "tiny data"
        case ..<150_000: scaleName = "small data"
        case ..<1_000_000: scaleName = "medium data"
        default: scaleName = "large corpus"
        }

        let size = modelSize ?? 4
        let baseLR: Double = size >= 24 ? 0.000005 : (size >= 10 ? 0.000007 : 0.00001)
        let lrMultiplier: Double = tokens < 25_000 ? 0.5 : (tokens > 1_000_000 ? 1.0 : 1.0)
        let batch = size >= 24 ? 2 : 4
        let rank: Int
        if tokens < 25_000 {
            rank = 4
        } else if tokens < 150_000 {
            rank = 8
        } else if tokens < 1_000_000 {
            rank = 16
        } else {
            rank = 24
        }
        let layers: Int
        if size >= 24 {
            layers = tokens < 150_000 ? 16 : 24
        } else if size >= 10 {
            layers = tokens < 150_000 ? 12 : 20
        } else {
            layers = tokens < 25_000 ? 8 : 16
        }
        let epochs: Double
        if tokens < 25_000 {
            epochs = 2.0
        } else if tokens < 150_000 {
            epochs = 2.5
        } else {
            epochs = 1.5
        }
        let stepsPerEpoch = max(1, Int(ceil(Double(max(train.exampleCount, 1)) / Double(batch))))
        let rawIterations = Int((Double(stepsPerEpoch) * epochs).rounded(.up))
        let iterations: Int
        if tokens == 0 {
            iterations = Self.defaultIterations
        } else if tokens < 25_000 {
            iterations = min(max(rawIterations, 200), 800)
        } else if tokens < 150_000 {
            iterations = min(max(rawIterations, 600), 1_500)
        } else if tokens < 1_000_000 {
            iterations = min(max(rawIterations, 1_000), 3_000)
        } else {
            iterations = min(max(rawIterations, 1_500), 5_000)
        }

        var warnings: [String] = []
        if train.exampleCount == 0 {
            warnings.append("Training data folder is empty or unreadable; recommendations fall back to conservative defaults.")
        }
        if validation.exampleCount == 0 {
            warnings.append("Validation data folder is empty or unreadable; add held-out examples before relying on loss curves.")
        } else if train.exampleCount > 0 {
            let validationRatio = Double(validation.exampleCount) / Double(train.exampleCount)
            if validationRatio < 0.05 {
                warnings.append("Validation set is under 5% of training examples; loss estimates may be noisy.")
            }
        }
        if train.unreadableRows > 0 || validation.unreadableRows > 0 {
            warnings.append("Some rows or files could not be parsed as structured data and were counted as raw text or skipped.")
        }
        if train.kind != validation.kind, validation.exampleCount > 0 {
            warnings.append("Training and validation data look like different regimes; use matching held-out data when possible.")
        }

        let sizeText = modelSize.map { "\($0.formatted(.number.precision(.fractionLength(0 ... 1))))B" } ?? "unknown-size"
        let rationale = [
            "\(scaleName.capitalized) for a \(sizeText) model.",
            "Detected \(train.kind) training data with \(train.exampleCount) examples and ~\(tokens) tokens.",
            "Recommendation favors a conservative LoRA run: enough capacity to move behavior, but low learning rate and bounded steps to reduce overfitting.",
        ].joined(separator: " ")

        return .init(
            modelSizeBillions: modelSize,
            dataScale: scaleName,
            train: train,
            validation: validation,
            recommendedRank: rank,
            recommendedScale: Self.defaultScale,
            recommendedLayers: layers,
            recommendedBatchSize: batch,
            recommendedIterations: iterations,
            recommendedLearningRate: baseLR * lrMultiplier,
            rationale: rationale,
            warnings: warnings)
    }

    private static func modelSizeBillions(_ modelID: String) -> Double? {
        let pattern = #"(?i)(\d+(?:\.\d+)?)\s*b"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(
                in: modelID, range: NSRange(modelID.startIndex..., in: modelID)),
            let range = Range(match.range(at: 1), in: modelID)
        else { return nil }
        return Double(modelID[range])
    }

}
