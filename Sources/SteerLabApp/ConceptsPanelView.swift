import AppKit
import ExperimentKit
import SteeringKit
import SwiftUI

private enum VectorLibrarySort: String, CaseIterable, Identifiable {
    case date = "Date"
    case concept = "Concept"
    case recipe = "Recipe"
    case layers = "Layer count"
    case norm = "Vector norm"

    var id: String { rawValue }
}

/// Display-neutral projection of a local sidecar or active-server catalog
/// record. Vector bytes stay on their substrate; only catalog metadata is
/// unified so Data does not silently become a local-only browser.
private struct VectorLibraryEntry: Identifiable {
    var id: String
    var label: String
    var name: String
    var concept: String
    var modelID: String
    var layerCount: Int
    var recipe: String
    var reading: String
    var vectorNorm: Float
    var stimulusHash: String
    var extractionDate: String
    var sourceLabel: String
    var localArtifact: VectorArtifact?
}

private enum ConceptDatasetPreview: String, Identifiable {
    case contrastive
    case stories
    case probe

    var id: String { rawValue }

    var title: String {
        switch self {
        case .contrastive: "Contrastive Dataset"
        case .stories: "Grand Mean Story Corpus"
        case .probe: "Probe Training Data"
        }
    }
}

/// Concept construction panel: paste contrasting stimuli, rebuild the
/// direction live, watch design stats, save through the canonical pipeline.
struct ConceptsPanelView: View {
    @Bindable var service: ChatService
    @State private var showImporter = false
    @State private var showDeleteConceptWarning = false
    @State private var vectorFilterConcepts: Set<String> = []
    @State private var vectorFilterShowsAll = true
    @State private var vectorSort: VectorLibrarySort = .date
    @State private var vectorSortAscending = false
    @State private var vectorSearchText = ""
    @State private var datasetPreview: ConceptDatasetPreview?
    @State private var independenceScreenRunning = false
    /// Neutral-PC capture defaults to the middle-third layer band; all layers
    /// is the explicit, expensive opt-in (see `ChatService.buildNeutralPCBasis`).
    @State private var neutralPCAllLayers = false
    @State private var independenceVerdict: String?
    /// The concept the current `independenceVerdict` describes — the verdict
    /// hides when the selection moves so it can never be read against the
    /// wrong concept.
    @State private var independenceVerdictConcept: String?

    private var builder: ConceptBuilder { service.concepts }

    var body: some View {
        @Bindable var builder = service.concepts
        Form {
            // ONE scoping rule (WorkspaceScoping.artifactListPresentation):
            // any server target shows the SERVER's concept catalog — the
            // datasets its extraction/probe jobs actually read — labeled
            // with the workspace they live in, with the mismatch banner when
            // the server serves a different tree than the app's selected
            // workspace. Browsing it is read-only; editing stays local, with
            // explicit Upload/Fetch sync and per-concept drift badges.
            if service.cluster.artifactListPresentation.showsServerArtifacts {
                if service.cluster.artifactListPresentation.showsMismatchBanner {
                    Section {
                        WorkspaceMismatchBanner(cluster: service.cluster)
                    }
                }
                serverConceptCatalogSection
            }

            Section("Concept Index") {
                HStack(spacing: 8) {
                    Picker("Concept", selection: $builder.selectedExisting) {
                        if builder.selectedExisting == nil {
                            Text("No concept selected").tag(String?.none)
                        }
                        ForEach(builder.existingConcepts, id: \.self) { name in
                            Text(name).tag(String?.some(name))
                        }
                    }
                    .help("select the concept whose datasets and vector artifacts you want to browse")

                    newDatasetButton

                    Button(role: .destructive) {
                        showDeleteConceptWarning = true
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(builder.currentConceptName.isEmpty || builder.selectedExisting == nil)
                    .help("delete this concept's editable datasets; saved vector artifacts remain")
                }
                if builder.selectedExisting == nil {
                    newConceptHandoffRow
                }

                conceptSyncStatusRow
                localEditNoticeRow

                conceptDatasetBrowser
                independenceScreenRow
                conceptArtifactBrowser
                if service.cluster.computeTarget == .server {
                    serverVectorNormBackfillRows
                }
            }

            Section("Dataset Builder") {
                conceptPickerRow(label: "Dataset concept", allowDelete: true)
                localEditNoticeRow
                Picker("Vector recipe", selection: $builder.recipeFamily) {
                    ForEach(ConceptBuilder.RecipeFamily.allCases) { family in
                        Text(family.label).tag(family)
                    }
                }
                .help(
                    "choose the data contract first: CAA pairs, RepE/LAT paired "
                        + "reader data, or grand-mean multi-concept stories")
                datasetBrowseDisclosure
                LabeledContent(
                    builder.recipeFamily.isPaired ? "Paired rows" : "Story rows",
                    value: builder.recipeFamily.isPaired
                        ? "\(builder.positives.count)+ / \(builder.negatives.count)-"
                        : "\(builder.multiConceptRows.count)"
                )
                if builder.recipeFamily.isPaired {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Paired JSONL")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextEditor(text: $builder.pairedJSONLDraft)
                            .font(.callout.monospaced())
                            .frame(height: 100)
                            .help(
                                "paste the copied LLM response here: one "
                                    + "{\"positive\",\"negative\"} object per line; "
                                    + "SteerLab splits every object into matched + and - rows")
                        Text(
                            "Paste the generated response here. Stable IDs and reader "
                                + "train/held-out splits are assigned by SteerLab when the dataset is built."
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                        DisclosureGroup("Add pairs manually (+ and - line by line)") {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(
                                    builder.recipeFamily == .pairedDifferencePCA
                                        ? "Positive reader prompt (concept-present)"
                                        : "Positive (expresses the concept)"
                                )
                                .font(.caption).foregroundStyle(.secondary)
                                TextEditor(text: $builder.positiveDraft)
                                    .font(.callout)
                                    .frame(height: 70)
                                    .help(
                                        "one item per line. For RepE/LAT, this may be a matched "
                                            + "reader prompt or scaffold whose positive side is concept-present")
                                Text(
                                    builder.recipeFamily == .pairedDifferencePCA
                                        ? "Negative reader prompt (matched control)"
                                        : "Negative (content-matched, without the concept)"
                                )
                                .font(.caption).foregroundStyle(.secondary)
                                TextEditor(text: $builder.negativeDraft)
                                    .font(.callout)
                                    .frame(height: 70)
                                    .help(
                                        "line i should match positive line i: same topic, length, "
                                            + "and scaffold, minus the concept")
                            }
                            .padding(.top, 4)
                        }
                    }
                } else {
                    emotionStoryEntry
                }

                HStack {
                    Button(builder.recipeFamily.isPaired ? "Add to set" : "Add story row(s)") {
                        Task { await builder.addDrafts() }
                    }
                    .disabled(builder.isWorking)
                    .help(
                        "append the pasted lines to the working set; stats go stale "
                            + "until you rebuild (one forward pass per new stimulus)")

                    if builder.recipeFamily.isPaired {
                        rebuildButton
                    } else {
                        Text("Grand Mean corpus balance updates automatically as rows and selections change.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    if builder.recipeFamily.isPaired {
                        Button("Import file…") { showImporter = true }
                            .disabled(builder.isWorking)
                            .help(
                                "import pairs made elsewhere: JSONL ({\"positive\",\"negative\"} "
                                    + "per line), a JSON array, or two-column CSV")
                    }

                    workingIndicator(task: "Rebuilding stats", fallback: "Rebuilding stats...")
                }

                copyLLMPromptRow

                if let prompt = builder.lastCopiedPrompt {
                    DisclosureGroup("Last copied prompt") {
                        ScrollView {
                            Text(prompt)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 220)
                    }
                }
            }

            if builder.recipeFamily == .repeReaderLAT {
                repeReaderSection
            }

            probeTrainingSection

            neutralNormCorpusSection

            buildErrorBanner

            if let status = builder.status {
                Section {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(status.lowercased().contains("saved") ? .green : .secondary)
                }
            }

            Section("Concept Vector Builder") {
                Picker("Vector Type", selection: $builder.recipeFamily) {
                    ForEach(ConceptBuilder.RecipeFamily.allCases) { family in
                        // Server-only families are absent in Local rather than
                        // present-and-refusing: a lens direction derived on this
                        // machine would be meaningless steering that no existing
                        // check would catch.
                        if !family.isServerOnly
                            || service.cluster.computeTarget == .server {
                            Text(family.label).tag(family)
                        }
                    }
                }
                .help("choose the vector recipe to build for the loaded model")

                // A derived direction has no concept, no stimuli, and nothing to
                // pool — showing those controls would imply provenance it does
                // not have.
                if builder.recipeFamily.extractsFromStimuli,
                    builder.recipeFamily != .emotionGrandMean {
                    vectorBuilderConceptPickerRow
                }

                // Workspace-scoped model selector: local tiers in Local, the
                // server's installed models in a server workspace. Extraction
                // stamps this model into the artifact's sidecar.
                WorkspaceModelPicker(service: service, label: "Model")
                if service.cluster.computeTarget == .server {
                    HStack {
                        InstallModelButton(cluster: service.cluster)
                        Spacer()
                    }
                } else {
                    // Same install affordance on Local: the builder's selector
                    // drives the model, so it needs the same way to get one.
                    HStack {
                        AddLocalModelButton(service: service)
                        Spacer()
                    }
                }
                if service.cluster.computeTarget == .local,
                    let caption = ChatService.localModelSelectionCaption(
                        selectedModelID: service.selectedModelID,
                        loadedModelID: service.loadedModelID,
                        isInstalled: service.catalog.isInstalled(
                            service.selectedModelID, in: .local))
                {
                    // Not an instruction any more: this selector DRIVES the
                    // model, so building swaps the resident one itself (the
                    // same load path the Playground's Load button uses) — and
                    // loads it from a cold start when nothing is resident.
                    Text(caption)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if builder.recipeFamily == .emotionGrandMean {
                    if !builder.emotionConceptOptions.isEmpty {
                        emotionConceptSelector
                    }
                    if !builder.emotionBuildTopics.isEmpty {
                        topicSelector
                    }
                }

                statusChip

                if builder.recipeFamily == .jlensTokenDirection {
                    jlensBuilderRows
                }

                if builder.recipeFamily.extractsFromStimuli {
                    LabeledContent("Reading position") {
                        ReadingPositionField(
                            choice: $builder.readingPositionChoice,
                            parameter: $builder.readingPositionParameter,
                            defaultCaption: nil,
                            help:
                                "WHERE the residual stream is read. The whole "
                                    + "cross-engine vocabulary, not just the "
                                    + "pooled pair this pane used to offer — the "
                                    + "content-side roles only exist inside a "
                                    + "rendered turn, so they need the chat "
                                    + "template below. Vector-build provenance, "
                                    + "not concept identity")
                    }
                    if let refusal = builder.readingPositionRefusal {
                        Text(refusal)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                    LabeledContent("Extraction rendering") {
                        ExtractionRenderingField(
                            choice: $builder.extractionRenderingChoice,
                            help:
                                "HOW each stimulus reaches the model. 'raw' is "
                                    + "the legacy rendering — the bare string "
                                    + "through the tokenizer. 'chat template' "
                                    + "renders it the way a measured generation "
                                    + "does; the two produce DIFFERENT "
                                    + "directions, so the artifact stamps which "
                                    + "one it read")
                    }
                    if let refusal = builder.extractionRenderingRefusal {
                        Text(refusal)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }

                derivationAdviceRow

                saveButton
            }

            if let stats = builder.stats {
                statsSection(stats)
            }

            vectorLibrarySection

            Section {
                Text(
                    builder.recipeFamily.isPaired
                        ? "Concepts are reusable primitives. Saving writes the selected recipe dataset and creates a model-specific per-layer vector artifact with provenance."
                        : "Grand-mean vectors use selected build rows only. Validation rows remain in the concept corpus but are held out from extraction."
                )
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onAppear { builder.presentRecipeGuide() }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.json, .commaSeparatedText, .plainText]
        ) { result in
            if case .success(let url) = result {
                Task { await builder.importPairs(from: url) }
            }
        }
        .popover(item: $datasetPreview, arrowEdge: .trailing) { preview in
            datasetPreviewSheet(preview)
        }
        .alert("Delete concept?", isPresented: $showDeleteConceptWarning) {
            Button("Delete datasets", role: .destructive) {
                builder.deleteSelectedConcept()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This deletes editable datasets for \(builder.currentConceptName) (stimuli, story corpus, probe items, RepE mirror). Reader pair sets are KEPT — a fitted reader's pinned training data outlives the concept's editable stimuli — and saved vector artifacts and runs remain for reproducibility."
            )
        }
        .alert(
            "Overwrite local concept?",
            isPresented: Binding(
                get: { builder.pendingServerFetchConcept != nil },
                set: { presented in
                    if !presented { builder.pendingServerFetchConcept = nil }
                })
        ) {
            Button("Overwrite local copy", role: .destructive) {
                if let name = builder.pendingServerFetchConcept {
                    builder.pendingServerFetchConcept = nil
                    Task { await builder.fetchConceptFromServer(name, overwrite: true) }
                }
            }
            Button("Cancel", role: .cancel) {
                builder.pendingServerFetchConcept = nil
            }
        } message: {
            Text(
                "This replaces the local stimulus files under prompts/concepts/"
                    + "\(builder.pendingServerFetchConcept ?? "…") with the server's "
                    + "copy. Local edits that were never uploaded will be lost."
            )
        }
    }

    // MARK: Server concept catalog (workspace scoping)

    /// The ACTIVE server's concept datasets — what its extraction and probe
    /// jobs actually read — listed READ-ONLY next to the local index (never
    /// merged into it). Rows carry drift badges against the local workspace
    /// and, on a possibly-different tree, the explicit sync affordances.
    @ViewBuilder
    private var serverConceptCatalogSection: some View {
        let presentation = service.cluster.artifactListPresentation
        let showSync = WorkspaceScoping.conceptSyncAffordancesVisible(presentation)
        Section(service.cluster.serverArtifactListTitle(kind: "Concept datasets")) {
            Text(
                "server extraction and probe jobs read THIS catalog — the "
                    + "server's own prompts/concepts tree. Browsing is read-only; "
                    + "editing stays in the local workspace and syncs explicitly."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
            if builder.serverConcepts.isEmpty {
                Text("no concept datasets on \(service.cluster.substrateLabel)'s tree yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help(
                        "the server's prompts/concepts directory has no datasets; "
                            + "Upload to server (Concept Index) publishes the "
                            + "selected local concept there")
            } else {
                ForEach(builder.serverConcepts) { record in
                    serverConceptRow(record, showSync: showSync)
                }
            }
        }
        .task(id: service.cluster.activeWorkspace) {
            await builder.refreshServerConcepts()
        }
    }

    private func serverConceptRow(
        _ record: ClusterClient.RemoteConceptRecord, showSync: Bool
    ) -> some View {
        let counts = "\(record.positiveCount)+/\(record.negativeCount)-"
            + (record.hasValidation == true ? " · validation" : "")
            + (record.hasMarkers == true ? " · markers" : "")
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(record.name)
                        .font(.caption)
                        .fontWeight(.semibold)
                    conceptDriftBadge(for: record.name)
                }
                Text(counts)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if showSync {
                Button("Fetch from server…") {
                    Task { await builder.fetchConceptFromServer(record.name) }
                }
                .font(.caption)
                .disabled(builder.isWorking)
                .help(
                    "download the server's copy of '\(record.name)' into the "
                        + "local workspace — asks before overwriting local stimuli")
                Button("Upload to server…") {
                    Task { await builder.uploadConceptToServer(record.name) }
                }
                .font(.caption)
                .disabled(
                    builder.isWorking
                        || builder.conceptDrift(for: record.name) == .serverOnly)
                .help(
                    "overwrite the server's copy of '\(record.name)' with the "
                        + "local workspace's stimulus files")
            }
        }
    }

    /// Drift badge for one concept: local workspace vs the active server's
    /// catalog, compared on cross-engine content hashes. Silent when the
    /// verdict is unknown (older server, catalog not yet listed).
    @ViewBuilder
    private func conceptDriftBadge(for name: String) -> some View {
        if let badge = WorkspaceScoping.conceptDriftBadge(builder.conceptDrift(for: name)) {
            Text(badge.label)
                .font(.caption2)
                .foregroundStyle(badge.isWarning ? .orange : .green)
                .help(
                    "compares the local workspace's stimulus texts against the "
                        + "server's (content hash, formatting-independent); "
                        + "differing copies mean a server job would run on data "
                        + "other than what this panel shows")
        }
    }

    /// Current concept's sync status + explicit sync affordances, rendered
    /// under any server target (drift needs the server catalog to compare).
    @ViewBuilder
    private var conceptSyncStatusRow: some View {
        if service.cluster.artifactListPresentation.showsServerArtifacts,
            !currentConceptName.isEmpty
        {
            LabeledContent("Server sync") {
                HStack(spacing: 8) {
                    if let badge = WorkspaceScoping.conceptDriftBadge(
                        builder.conceptDrift(for: currentConceptName))
                    {
                        Text(badge.label)
                            .foregroundStyle(badge.isWarning ? .orange : .green)
                    } else {
                        Text("not compared yet")
                            .foregroundStyle(.secondary)
                    }
                    if WorkspaceScoping.conceptSyncAffordancesVisible(
                        service.cluster.artifactListPresentation)
                    {
                        Button("Upload to server…") {
                            Task { await builder.uploadConceptToServer(currentConceptName) }
                        }
                        .disabled(builder.isWorking)
                        .help(
                            "write this concept's local stimulus files to the "
                                + "server's tree so server jobs see the data this "
                                + "panel shows")
                        Button("Fetch from server…") {
                            Task { await builder.fetchConceptFromServer(currentConceptName) }
                        }
                        .disabled(builder.isWorking)
                        .help(
                            "download the server's copy into the local workspace "
                                + "— asks before overwriting local stimuli")
                    }
                }
                .font(.caption)
            }
            .help(
                "local↔server drift for '\(currentConceptName)': server "
                    + "extraction reads the server's tree, so differing copies "
                    + "surface here instead of failing (or silently diverging) "
                    + "inside a job")
        }
    }

    /// Standing caption on every edit affordance under an UNPAIRED server
    /// target (one rule: WorkspaceScoping.conceptEditingNotice) — edits land
    /// in the LOCAL workspace, invisible to the server until uploaded.
    @ViewBuilder
    private var localEditNoticeRow: some View {
        if let notice = WorkspaceScoping.conceptEditingNotice(
            service.cluster.artifactListPresentation)
        {
            Label(notice, systemImage: "pencil.line")
                .font(.caption)
                .foregroundStyle(.orange)
        }
    }

    // MARK: Top action row

    @ViewBuilder
    private var conceptDatasetBrowser: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent("Datasets", value: conceptDatasetSummary)
                .help("recipe-specific datasets attached to this concept")
            if builder.positives.isEmpty && builder.negatives.isEmpty
                && builder.multiConceptRows.isEmpty && builder.probeExamples.isEmpty
            {
                Text("No editable datasets are attached to this concept yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                HStack(spacing: 8) {
                    if !builder.positives.isEmpty || !builder.negatives.isEmpty {
                        Button {
                            datasetPreview = .contrastive
                        } label: {
                            Label(
                                "\(builder.positives.count)+/\(builder.negatives.count)-",
                                systemImage: "rectangle.split.2x1")
                        }
                        .help("browse the contrastive CAA/RepE rows for this concept")
                    }
                    if !builder.multiConceptRows.isEmpty {
                        Button {
                            datasetPreview = .stories
                        } label: {
                            Label("\(builder.multiConceptRows.count) stories", systemImage: "text.book.closed")
                        }
                        .help("browse Grand Mean story rows for this concept")
                    }
                    if !builder.probeExamples.isEmpty {
                        Button {
                            datasetPreview = .probe
                        } label: {
                            Label(
                                "\(builder.probePositiveCount)+/\(builder.probeNegativeCount)- probe",
                                systemImage: "waveform.path.ecg")
                        }
                        .help("browse separate labeled probe examples for this concept")
                    }
                }
                .buttonStyle(.bordered)
            }
        }
    }

    /// Firewall check, one click: run the Python stimulus-independence
    /// screen (`steerlab_server.experiment.stimulus_screen`) on the selected
    /// concept's `prompts/concepts/<name>/` files. WHICH vocabulary is
    /// forbidden is workspace data — the screen resolves and names it. The
    /// science stays in the Python module — this button only launches it
    /// through the local venv, streams findings to the Activity pane, and
    /// states the verdict plainly.
    @ViewBuilder
    private var independenceScreenRow: some View {
        let concept = builder.currentConceptName
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Button {
                    runIndependenceScreen()
                } label: {
                    Label(
                        "Screen Stimuli for Forbidden Vocabulary",
                        systemImage: "checkmark.shield")
                }
                .disabled(independenceScreenRunning || concept.isEmpty)
                .help(
                    "circularity-firewall check: scans this concept's stimulus "
                        + "and validation text for the workspace's forbidden "
                        + "vocabulary — prompts/screens/forbidden-vocabulary"
                        + ".json when present (workspace-editable named term "
                        + "lists), else the shipped judicial-study default "
                        + "list (court, sentence, defendant, …). Stimuli "
                        + "that encode the "
                        + "study's task domain confound every downstream "
                        + "effect. The governing vocabulary and flagged lines "
                        + "stream to the Activity pane; flagged means review, "
                        + "not automatic rejection")
                if independenceScreenRunning {
                    ProgressView().controlSize(.small)
                }
            }
            if let independenceVerdict, independenceVerdictConcept == concept {
                Text(independenceVerdict)
                    .font(.caption)
                    .foregroundStyle(
                        independenceVerdict.hasPrefix("Clean") ? Color.secondary : .orange)
                    .textSelection(.enabled)
            }
        }
    }

    private func runIndependenceScreen() {
        let concept = builder.currentConceptName
        guard !concept.isEmpty else { return }
        let directory = VectorCatalog.conceptsDirectory.appending(component: concept)
        independenceScreenRunning = true
        independenceVerdict = nil
        independenceVerdictConcept = nil
        let service = service
        Task {
            let title = "Stimulus independence screen — \(concept)"
            var lines = ["screening \(directory.path)…"]
            let logID = service.startLiveLog(title: title, initialLine: lines[0])
            let outcome = await StimulusIndependenceScreen.run(
                conceptDirectory: directory
            ) { line in
                lines.append(line)
                if lines.count > 400 { lines.removeFirst(lines.count - 400) }
                service.updateLiveLog(id: logID, title: title, lines: lines)
            }
            lines.append(outcome.verdict)
            service.updateLiveLog(id: logID, title: title, lines: lines)
            independenceVerdict = outcome.verdict
            independenceVerdictConcept = concept
            independenceScreenRunning = false
        }
    }

    @ViewBuilder
    private var conceptArtifactBrowser: some View {
        if currentConceptArtifacts.isEmpty {
            LabeledContent("Vector artifacts", value: "0")
                .help("saved model-specific vectors for this concept, across recipes and models")
        } else {
            DisclosureGroup("Vector artifacts (\(currentConceptArtifacts.count))") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(currentConceptArtifacts) { artifact in
                        conceptArtifactRow(artifact)
                        if artifact.id != currentConceptArtifacts.last?.id {
                            Divider()
                        }
                    }
                }
                .padding(.top, 4)
            }
            .help("saved model-specific vectors for this concept, across recipes and models")
        }
    }

    private func conceptArtifactRow(_ artifact: VectorLibraryEntry) -> some View {
        let isCompatible = activeVectorLibraryEntries.contains(where: { $0.id == artifact.id })
        let isInSteering = service.slots.contains(where: { $0.vectorID == artifact.id })
        return VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(artifact.name)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                Spacer()
                if let localArtifact = artifact.localArtifact,
                    localArtifact.sidecar.residualNormPerLayer == nil
                {
                    Button("Measure norms") {
                        Task { await builder.backfillNormsLocally(localArtifact) }
                    }
                    .disabled(builder.isWorking)
                    .help(
                        "this sidecar lacks residualNormPerLayer (the norm-unit "
                            + "alpha denominator): measure typical residual norms on "
                            + "the pinned neutral corpus with "
                            + "\(artifact.modelID) loaded and write a NEW "
                            + "artifact into a fresh run directory — the original is "
                            + "never modified")
                }
                Button("Use") {
                    useVector(artifact)
                }
                .disabled(!isCompatible)
                .help(
                    isCompatible
                        ? "select this vector into the first steering slot"
                        : "select \(artifact.modelID) before steering with this vector")
            }
            HStack(spacing: 8) {
                Text(artifact.recipe)
                Text("\(artifact.layerCount) layers")
                Text(dateTimeLabel(artifact.extractionDate))
                if isCompatible {
                    Text("current model").foregroundStyle(.green)
                }
                if isInSteering {
                    Text("in steering").foregroundStyle(.orange)
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            Text("\(artifact.modelID) · \(artifact.sourceLabel)")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .textSelection(.enabled)
        }
    }

    /// The concept index, for deciding whether a vector record is ATTACHED to a
    /// concept or standing alone. Not a display list — it is the membership test
    /// behind "unattached", so a derived direction can be told from a concept's
    /// own vectors without special-casing any recipe.
    private var builderExistingConceptNames: [String] {
        service.concepts.existingConcepts
    }

    /// Server-catalog records for this concept that LACK residual norms —
    /// norm-unit alphas need the denominator, so each row offers the backfill
    /// job (`POST /api/vectors/backfill-norms`, durable, per-substrate). Rows
    /// with norms never appear: backfill never overwrites.
    @ViewBuilder
    private var serverVectorNormBackfillRows: some View {
        let workspaceName = service.cluster.substrateLabel
        // The residual-norm denominator is a property of the MODEL, LAYER, and
        // neutral corpus — it has nothing to do with the concept. Scoping this
        // list to the selected concept therefore hid it from every artifact with
        // no concept attachment, which is exactly the class that always lacks
        // norms: derived directions (J-lens, reader conversions) and imports.
        // The affordance existed and was unreachable for the artifacts that
        // need it most. Unattached records are listed alongside the selected
        // concept's own.
        let known = Set(builderExistingConceptNames)
        let records = service.catalog.remoteVectors.filter {
            guard $0.residualNormPerLayer == nil, $0.hasResidualNorms != true
            else { return false }
            return $0.concept == currentConceptName || !known.contains($0.concept)
        }
        Group {
            if records.isEmpty {
                LabeledContent(
                    "Server vectors missing norms", value: "none on \(workspaceName)")
                    .help(
                        "server artifacts for this concept whose sidecars lack "
                            + "residualNormPerLayer; a Measure norms job appears here "
                            + "when one exists")
            } else {
                // Expanded when there is something to do. Collapsed-by-default
                // was the second half of the "no obvious UI meaning" problem:
                // a refusal sends you here, and the fix was behind a closed
                // triangle whose title never used the word the refusal used.
                DisclosureGroup(
                    isExpanded: .constant(true)
                ) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("""
                             Norm-unit α needs a per-layer denominator measured                              on the neutral corpus. Measuring writes a NEW                              artifact — the original is never modified — and the                              copy is what you then steer with.
                             """)
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        // WHICH corpus, pinned by hash: the measurement
                        // becomes the new artifact's residualNormSource /
                        // neutralCorpusHash provenance, so it is named
                        // before the button, not discovered after the job.
                        Text(neutralCorpusProvenanceLine)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        if records.count > 1 {
                            Button("Measure all (\(records.count))") {
                                Task {
                                    await builder.backfillAllNormsOnActiveServer(records)
                                }
                            }
                            .font(.caption)
                            .disabled(builder.isWorking)
                            .help(
                                "queue one durable server job per listed vector, "
                                    + "sequentially, for the selected server model; "
                                    + "vectors extracted on other models are skipped "
                                    + "loudly (norms are a per-model measurement). "
                                    + "Each writes a NEW artifact; originals are "
                                    + "never modified")
                        }
                        ForEach(records) { record in
                            serverVectorNormRow(record)
                            if record.id != records.last?.id { Divider() }
                        }
                    }
                    .padding(.top, 4)
                } label: {
                    // The title carries the vocabulary the refusals use, so
                    // someone told to "measure norms" recognizes the
                    // destination on sight.
                    Label(
                        "Server vectors missing norms on \(workspaceName) "
                            + "(\(records.count)) — measure norms here",
                        systemImage: "ruler")
                        .font(.callout.bold())
                }
            }
        }
        .task(id: service.cluster.activeWorkspace) {
            await service.catalog.refreshRemoteVectors()
        }
    }

    /// The corpus (and hash) the backfill measures on — named at the
    /// affordance because it becomes provenance. The hash is computed from
    /// the WORKSPACE's file bytes (raw-bytes SHA-256, the shared single-file
    /// corpus convention); a paired server measures its checkout of the same
    /// workspace file.
    private var neutralCorpusProvenanceLine: String {
        let (path, hash) = ConceptBuilder.neutralCorpusProvenance()
        guard let hash else {
            return "corpus: \(path) — not found in this workspace; seed the "
                + "workspace's neutral corpus first"
        }
        return "corpus: \(path) (sha256 \(hash.prefix(12))…) → stamped as "
            + "residualNormSource: neutral-corpus"
    }

    private func serverVectorNormRow(_ record: RemoteVectorRecord) -> some View {
        let title = "\(record.name) · \(record.layerCount) layers"
        let subtitle = "\(record.modelID) — \(record.id)"
        let helpText = "queue a durable server job: measure residual norms on the "
            + "server's neutral corpus with \(record.modelID) and write a NEW "
            + "artifact into a fresh run directory on its tree — the original is "
            + "never modified"
        return HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            Button("Measure norms") {
                Task { await builder.backfillNormsOnActiveServer(record) }
            }
            .font(.caption)
            .disabled(builder.isWorking)
            .help(helpText)
        }
    }

    @ViewBuilder
    private func datasetPreviewSheet(_ preview: ConceptDatasetPreview) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(preview.title)
                .font(.title2)
                .fontWeight(.semibold)
            Text(currentConceptName.isEmpty ? "Unsaved concept" : currentConceptName)
                .font(.caption)
                .foregroundStyle(.secondary)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    switch preview {
                    case .contrastive:
                        let count = max(builder.positives.count, builder.negatives.count)
                        ForEach(0 ..< count, id: \.self) { index in
                            VStack(alignment: .leading, spacing: 4) {
                                if index < builder.positives.count {
                                    datasetPreviewLine(label: "+", text: builder.positives[index]) {
                                        builder.removeContrastiveStimulus(isPositive: true, index: index)
                                    }
                                }
                                if index < builder.negatives.count {
                                    datasetPreviewLine(label: "-", text: builder.negatives[index]) {
                                        builder.removeContrastiveStimulus(isPositive: false, index: index)
                                    }
                                }
                            }
                            Divider()
                        }
                    case .stories:
                        ForEach(Array(builder.multiConceptRows.enumerated()), id: \.offset) { _, row in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(
                                    "\(row.concept) · \(row.topic ?? "manual") · \(ConceptBuilder.canonicalSplit(row.split))"
                                )
                                .font(.caption)
                                .fontWeight(.semibold)
                                Text(row.text)
                                    .font(.callout)
                                    .textSelection(.enabled)
                                Button("Remove") {
                                    builder.removeEmotionRow(row)
                                }
                                .font(.caption)
                            }
                            Divider()
                        }
                    case .probe:
                        ForEach(Array(builder.probeExamples.enumerated()), id: \.element.id) { index, item in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(item.expresses ? "concept-present" : "control")
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                    if let topic = item.topic, !topic.isEmpty {
                                        Text(topic)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Text(ConceptBuilder.canonicalSplit(item.split))
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Button("Remove") {
                                        builder.removeProbeExample(index: index)
                                    }
                                    .font(.caption)
                                }
                                Text(item.text)
                                    .font(.callout)
                                    .textSelection(.enabled)
                            }
                            Divider()
                        }
                    }
                }
            }
        }
        .frame(minWidth: 360, idealWidth: 460, maxWidth: 560, minHeight: 260, idealHeight: 420, maxHeight: 560)
        .padding(16)
    }

    private func datasetPreviewLine(label: String, text: String, remove: @escaping () -> Void) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption.monospaced())
                .foregroundStyle(label == "+" ? .green : .secondary)
                .frame(width: 18, alignment: .trailing)
            Text(text)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Remove", action: remove)
                .font(.caption)
        }
    }

    private func conceptPickerRow(label: String, allowDelete: Bool) -> some View {
        @Bindable var builder = service.concepts
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Picker(label, selection: $builder.selectedExisting) {
                    if builder.selectedExisting == nil {
                        Text("No concept selected").tag(String?.none)
                    }
                    ForEach(builder.existingConcepts, id: \.self) { name in
                        Text(name).tag(String?.some(name))
                    }
                }
                .help("choose the concept for this panel's dataset work")

                newDatasetButton

                if allowDelete {
                    Button(role: .destructive) {
                        showDeleteConceptWarning = true
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(builder.currentConceptName.isEmpty || builder.selectedExisting == nil)
                    .help("delete this concept's editable datasets; saved vector artifacts remain")
                }
            }
            if builder.selectedExisting == nil {
                newConceptHandoffRow
            }
        }
    }

    /// The retired new-concept entry's replacement (WP-Data phase 4).
    ///
    /// This panel used to carry its own from-scratch creation form: a
    /// "Name (kebab-case)" field with a Save Concept button, plus a `+`
    /// beside each concept picker that emptied the editor into a
    /// nameless draft. It was a SECOND creation entry — it knew only
    /// `prompts/concepts/<name>/`, so a story corpus, a probe set, a
    /// paired mirror, or a held-out set typed in here landed under the
    /// wrong recipe's root or nowhere at all. That is precisely the
    /// class of mistake the role-first flow exists to close.
    ///
    /// So the builder keeps everything it can EDIT and DERIVE — every
    /// stimulus row, every recipe option, every build action — and hands
    /// creation to the one flow that computes destinations from the
    /// engine's own path authorities. `ConceptBuilder.saveNewConcept()`
    /// is untouched and still the thing that runs; the sheet drives it.
    private var newDatasetButton: some View {
        NewDatasetButton(
            service: service,
            onCreated: { _ in service.datasetInventory.refresh() },
            openInConceptBuilder: { name in
                guard let name else { return }
                service.concepts.selectConcept(name)
            },
            title: "New Dataset…",
            systemImage: "plus",
            help:
                "declare what the dataset is — concept stimuli, paired "
                + "stimuli, story corpus, probe items, validation set, "
                + "neutral corpus, battery — and it is filed in the one "
                + "place that recipe reads. New concepts start here; this "
                + "panel edits and builds them."
        )
    }

    private var newConceptHandoffRow: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(
                "No concept selected. New Dataset… creates one in the "
                    + "canonical place for its recipe and opens it here."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private var vectorBuilderConceptPickerRow: some View {
        @Bindable var builder = service.concepts
        return VStack(alignment: .leading, spacing: 4) {
            Picker("Concept", selection: $builder.vectorBuilderSelectedExisting) {
                ForEach(builder.existingConcepts, id: \.self) { name in
                    Text(name).tag(String?.some(name))
                }
            }
            .disabled(builder.existingConcepts.isEmpty)
            .help(
                "choose the saved concept dataset to extract into a vector. "
                    + "This is independent of the Concept Index browser above")
            Text("Build target; changing this does not change the Concept Index or Dataset Builder.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var currentConceptName: String {
        if let selected = builder.selectedExisting { return selected }
        return builder.conceptName.lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
    }

    /// Every vector in the ACTIVE compute workspace. This powers the
    /// concept-level artifact history; model compatibility is applied
    /// separately for the steering-oriented Vector Library.
    private var activeVectorCatalogEntries: [VectorLibraryEntry] {
        if service.cluster.computeTarget == .server {
            return service.catalog.remoteVectors
                .filter {
                    WorkspaceScoping.offerableForServerSteering(
                        substrate: $0.substrate)
                }
                .map { record in
                let norms = record.normsPerLayer ?? []
                let middle = norms.isEmpty ? 0 : norms[
                    min(norms.count - 1, norms.count / 2)]
                return VectorLibraryEntry(
                    id: record.id,
                    label: [record.concept, record.name, record.resolvedMethod]
                        .compactMap { $0 }.joined(separator: " · "),
                    name: record.name,
                    concept: record.concept,
                    modelID: record.modelID,
                    layerCount: record.layerCount,
                    recipe: record.resolvedMethod ?? "unknown",
                    reading: record.resolvedReadingPosition ?? "unknown",
                    vectorNorm: middle,
                    stimulusHash: record.stimulusSetHash ?? "unrecorded",
                    extractionDate: record.resolvedExtractionDate ?? "",
                    sourceLabel: service.cluster.substrateLabel,
                    localArtifact: nil)
            }
        }
        return service.vectors.map { artifact in
            let sidecar = artifact.sidecar
            let middle = sidecar.normsPerLayer.isEmpty ? 0 : sidecar.normsPerLayer[
                min(sidecar.normsPerLayer.count - 1, sidecar.normsPerLayer.count / 2)]
            return VectorLibraryEntry(
                id: artifact.id,
                label: artifact.label,
                name: artifact.name,
                concept: sidecar.concept,
                modelID: sidecar.modelID,
                layerCount: sidecar.layerCount,
                recipe: sidecar.recipeMethod ?? sidecar.extractionMethod ?? "unknown",
                reading: sidecar.readingPosition ?? "unknown",
                vectorNorm: middle,
                stimulusHash: sidecar.stimulusSetHash,
                extractionDate: sidecar.extractionDate,
                sourceLabel: "Local (MLX)",
                localArtifact: artifact)
        }
    }

    /// The active catalog narrowed to the model that can actually consume the
    /// vector. Server browsing does not require the model to be resident; a
    /// selected installed model is enough.
    private var activeVectorLibraryEntries: [VectorLibraryEntry] {
        if service.cluster.computeTarget == .server {
            guard let model = service.selectedRemoteModelID, !model.isEmpty
            else { return activeVectorCatalogEntries }
            return activeVectorCatalogEntries.filter { $0.modelID == model }
        }
        guard let model = service.loadedModelID else { return [] }
        return activeVectorCatalogEntries.filter {
            $0.modelID == model
                && ($0.localArtifact.map {
                    WorkspaceScoping.offerableForLocalSteering(
                        substrate: $0.sidecar.substrate)
                } ?? false)
        }
    }

    private var currentConceptArtifacts: [VectorLibraryEntry] {
        let name = currentConceptName
        guard !name.isEmpty else { return [] }
        return activeVectorCatalogEntries
            .filter { $0.concept == name }
            .sorted { $0.extractionDate > $1.extractionDate }
    }

    private var vectorLibraryConcepts: [String] {
        Array(Set(activeVectorLibraryEntries.map(\.concept))).sorted()
    }

    private var libraryArtifacts: [VectorLibraryEntry] {
        let filtered = activeVectorLibraryEntries.filter { artifact in
            let conceptMatches = vectorFilterShowsAll
                || vectorFilterConcepts.contains(artifact.concept)
            guard conceptMatches else { return false }
            let query = vectorSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !query.isEmpty else { return true }
            return vectorSearchHaystack(for: artifact).contains(query)
        }
        return filtered.sorted { lhs, rhs in
            let comparison = vectorSortComparison(lhs, rhs)
            if comparison == 0 { return lhs.id < rhs.id }
            return vectorSortAscending ? comparison < 0 : comparison > 0
        }
    }

    private var vectorFilterLabel: String {
        if vectorFilterShowsAll { return "All" }
        return vectorFilterConcepts.isEmpty ? "None" : "\(vectorFilterConcepts.count)"
    }

    private func vectorSearchHaystack(for artifact: VectorArtifact) -> String {
        vectorSearchHaystack(
            label: artifact.label,
            name: artifact.name,
            concept: artifact.sidecar.concept,
            modelID: artifact.sidecar.modelID,
            recipe: artifact.sidecar.recipeMethod ?? artifact.sidecar.extractionMethod,
            reading: artifact.sidecar.readingPosition,
            stimulusHash: artifact.sidecar.stimulusSetHash,
            id: artifact.id)
    }

    private func vectorSearchHaystack(for artifact: VectorLibraryEntry) -> String {
        vectorSearchHaystack(
            label: artifact.label,
            name: artifact.name,
            concept: artifact.concept,
            modelID: artifact.modelID,
            recipe: artifact.recipe,
            reading: artifact.reading,
            stimulusHash: artifact.stimulusHash,
            id: artifact.id)
    }

    private func vectorSearchHaystack(
        label: String,
        name: String,
        concept: String,
        modelID: String,
        recipe: String?,
        reading: String?,
        stimulusHash: String,
        id: String
    ) -> String {
        [
            label,
            name,
            concept,
            modelID,
            recipe,
            reading,
            stimulusHash,
            URL(fileURLWithPath: id).deletingLastPathComponent().lastPathComponent,
        ]
        .compactMap { $0 }
        .joined(separator: " ")
        .lowercased()
    }

    private func vectorSortComparison(
        _ lhs: VectorLibraryEntry, _ rhs: VectorLibraryEntry
    ) -> Int {
        func compare<T: Comparable>(_ left: T, _ right: T) -> Int {
            if left < right { return -1 }
            if left > right { return 1 }
            return 0
        }

        let primary: Int
        switch vectorSort {
        case .date:
            primary = compare(lhs.extractionDate, rhs.extractionDate)
        case .concept:
            primary = compare(lhs.concept, rhs.concept)
        case .recipe:
            primary = compare(lhs.recipe, rhs.recipe)
        case .layers:
            primary = compare(lhs.layerCount, rhs.layerCount)
        case .norm:
            primary = compare(lhs.vectorNorm, rhs.vectorNorm)
        }
        if primary != 0 { return primary }
        return compare(lhs.extractionDate, rhs.extractionDate)
    }

    private var conceptDatasetSummary: String {
        var parts: [String] = []
        if !builder.positives.isEmpty || !builder.negatives.isEmpty {
            parts.append("contrastive \(builder.positives.count)+/\(builder.negatives.count)-")
        }
        if !builder.multiConceptRows.isEmpty {
            parts.append("stories \(builder.multiConceptRows.count)")
        }
        if !builder.probeExamples.isEmpty {
            parts.append("probe \(builder.probePositiveCount)+/\(builder.probeNegativeCount)-")
        }
        return parts.isEmpty ? "none yet" : parts.joined(separator: " · ")
    }

    @ViewBuilder
    private var datasetBrowseDisclosure: some View {
        if builder.recipeFamily.isPaired,
            !builder.positives.isEmpty || !builder.negatives.isEmpty
        {
            DisclosureGroup(
                "Browse dataset "
                    + "(\(builder.positives.count)+ / \(builder.negatives.count)-)"
            ) {
                Text("Each row is a pair; the number is its decision margin along the current direction.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                let pairCount = max(builder.positives.count, builder.negatives.count)
                ForEach(0 ..< pairCount, id: \.self) { index in
                    VStack(alignment: .leading, spacing: 2) {
                        if index < builder.positives.count {
                            stimulusRow(isPositive: true, index: index)
                        }
                        if index < builder.negatives.count {
                            stimulusRow(isPositive: false, index: index)
                        }
                        if index < pairCount - 1 { Divider() }
                    }
                }
            }
        } else if builder.recipeFamily == .emotionGrandMean,
            !builder.selectedEmotionRows.isEmpty
        {
            DisclosureGroup("Browse corpus (\(builder.selectedEmotionRows.count) rows)") {
                ForEach(Array(builder.selectedEmotionRows.enumerated()), id: \.offset) { _, row in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(
                                "\(row.concept) · \(row.topic ?? "manual") · \(ConceptBuilder.canonicalSplit(row.split))"
                            )
                            .font(.caption)
                            .bold()
                            Spacer()
                            Button("Remove") {
                                builder.removeEmotionRow(row)
                            }
                            .font(.caption)
                        }
                        Text(row.text)
                            .font(.caption)
                            .textSelection(.enabled)
                        Divider()
                    }
                }
            }
        }
    }

    // MARK: RepE reader UI (thin: renders ConceptBuilder state; REPE-IMPLEMENTATION-BRIEF §8)

    @ViewBuilder
    private var repeReaderSection: some View {
        @Bindable var builder = service.concepts
        Section("RepE Reader") {
            Text(
                "Concept data → reader artifact → optional steering variant. "
                    + "The reader is a fitted measurement instrument (task template "
                    + "+ LAT token position + PCA training normalization + held-out "
                    + "accuracy); steering with it is an explicit, provenance-stamped "
                    + "conversion — not \"positive/negative text → vector\"."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
            .task(id: service.cluster.activeWorkspace) {
                // Server readers are substrate state: re-list on workspace
                // switch (no-op → empty in the Local workspace).
                await builder.refreshServerReaders()
            }

            Toggle("Custom template text", isOn: $builder.useCustomReaderTemplate)
                .help(
                    "off = pick a registry template from prompts/templates/ (shared "
                        + "with the Python server); on = write a one-off scaffold, "
                        + "persisted and hashed into the reader's run directory")
            if builder.useCustomReaderTemplate {
                TextEditor(text: $builder.customReaderTemplateText)
                    .font(.callout.monospaced())
                    .frame(height: 80)
                Text(
                    "must contain {{stimulus}}; include {{concept}} to name the concept "
                        + "in the scaffold (RepE-faithful) or omit it for an unnamed "
                        + "clean-room scaffold (stamped as a divergence)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Picker("Task template", selection: $builder.selectedReaderTemplateID) {
                    ForEach(builder.readerTemplates, id: \.id) { template in
                        Text(template.id).tag(String?.some(template.id))
                    }
                }
                .disabled(builder.readerTemplates.isEmpty)
                .help("registry templates under prompts/templates/ — data shared with the server, pinned by raw-byte hash")
                if let template = builder.selectedReaderTemplate {
                    Text(template.text)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .foregroundStyle(.secondary)
                    if let divergence = template.divergence {
                        Text("divergence: \(divergence)")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                    LabeledContent(
                        "LAT token",
                        value: template.latToken == "final"
                            ? "final scaffold token" : template.latToken)
                        .help("the token position whose hidden state the reader reads; rendering guarantees it is the scaffold's last token")
                } else if builder.readerTemplates.isEmpty {
                    Text("no templates found under prompts/templates/")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }

            Stepper(
                "Held-out pairs: \(builder.readerHeldOutPairCount)",
                value: $builder.readerHeldOutPairCount,
                in: 0 ... max(0, builder.positives.count))
                .help(
                    "the LAST k pairs are written with split \"test\" and score the "
                        + "fitted probe they did not train — keep some, or accuracies "
                        + "are train-only")

            if service.cluster.computeTarget == .server {
                Text(
                    "Fits on the server's loaded model as a durable job; reader "
                        + "artifacts stay on that substrate (readers are per-model, "
                        + "per-substrate measurement instruments).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Build reader") {
                    Task { await builder.buildReader() }
                }
                .disabled(builder.isWorking)
                .help(
                    "renders every pair through the template, captures the LAT token, "
                        + "fits PC1 + probe per layer, and writes one reader artifact "
                        + "per layer into a fresh run directory")
                workingIndicator(task: "Building reader", fallback: "Building reader...")
            }

            if !builder.readerLayerScores.isEmpty {
                DisclosureGroup(
                    "Fit scores by layer (\(builder.readerLayerScores.count) layers)"
                ) {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                        GridRow {
                            ForEach(Self.readerLayerColumns, id: \.self) { column in
                                Text(column).font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        ForEach(builder.readerLayerScores) { score in
                            GridRow {
                                ForEach(
                                    Array(Self.readerLayerCells(score).enumerated()),
                                    id: \.offset
                                ) { cell in
                                    Text(cell.element).font(.caption.monospaced())
                                }
                            }
                        }
                    }
                }
            }

            readerArtifactRows
        }
    }

    /// The fit-score grid's columns, in the order `readerLayerCells` returns.
    /// "★" on a layer marks the fit's RECOMMENDED layer (argmax held-out
    /// accuracy) — a recommendation, not a selection: which layer a study
    /// reads is declared in its manifest.
    static let readerLayerColumns = [
        "layer", "train", "held-out", "PC1 var (diffs)", "sign from",
    ]

    /// One reader-layer row's cells, as plain strings.
    ///
    /// Split out of the grid deliberately: the cells are now four formatted
    /// values with two optionals among them, and SwiftUI's type checker gives
    /// up on the inline form ("unable to type-check this expression in
    /// reasonable time"). Formatting is not layout, and it is testable here.
    ///
    /// An absent explained variance prints "—", not "0.00": PC1's share of a
    /// difference cloud with no variance is undefined, and 0 would read as
    /// "PC1 explains nothing" — the opposite of what an all-identical cloud
    /// means. The sign cell says whether the HELD-OUT split fixed this layer's
    /// direction (the RepE paper's step 4) or the train labels did.
    static func readerLayerCells(
        _ score: ConceptBuilder.ReaderLayerScore
    ) -> [String] {
        let layer = score.isRecommendedLayer
            ? "\(score.layer) ★" : "\(score.layer)"
        let train = String(format: "%.0f%%", score.trainAccuracy * 100)
        let held =
            score.heldOutAccuracy.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
        let variance =
            score.differenceCloudVarianceLabel
        let sign: String
        switch score.signConvention {
        case .heldOutPairAgreement:
            sign = score.signHeldOutAccuracy
                .map { String(format: "held-out %.0f%%", $0 * 100) } ?? "held-out"
        case .trainMajority:
            sign = "train"
        }
        return [layer, train, held, variance, sign]
    }

    @ViewBuilder
    private var readerArtifactRows: some View {
        localReaderArtifactRows
        if service.cluster.computeTarget == .server {
            serverReaderRows
        }
    }

    @ViewBuilder
    private var serverReaderRows: some View {
        // Readers fitted on the active server (its runs/ tree), listed
        // read-only: substrate-specific instruments, never merged with the
        // local rows above.
        let workspaceName = service.cluster.substrateLabel
        let readers = builder.serverReaders.filter {
            $0.concept == builder.currentConceptName
        }
        if readers.isEmpty {
            LabeledContent("Server readers", value: "none yet on \(workspaceName)")
                .help(
                    "readers fitted on the server for this concept — they stay on "
                        + "that substrate; a queued fit job adds rows here when it "
                        + "finishes")
        } else {
            DisclosureGroup("Server readers on \(workspaceName) (\(readers.count))") {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(readers) { record in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(
                                "\(record.concept) · layer \(record.layer)"
                                    + (record.heldOutAccuracy.map {
                                        String(format: " · held-out %.0f%%", $0 * 100)
                                    } ?? "")
                            )
                            .font(.caption)
                            .fontWeight(.semibold)
                            Text("\(record.modelID) — \(record.id)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        if record.id != readers.last?.id { Divider() }
                    }
                }
                .padding(.top, 4)
            }
        }
    }

    @ViewBuilder
    private var localReaderArtifactRows: some View {
        let readers = builder.readerArtifacts.filter {
            $0.artifact.concept == builder.currentConceptName
        }
        if readers.isEmpty {
            LabeledContent("Reader artifacts", value: "none yet")
                .help("fitted readers for this concept, discovered under runs/")
        } else {
            DisclosureGroup("Reader artifacts (\(readers.count))") {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(readers) { record in
                        HStack(alignment: .firstTextBaseline) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(record.label)
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                Text(record.fileName)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Button("Derive steering vector") {
                                builder.deriveSteeringVectorFromReader(record)
                            }
                            .font(.caption)
                            .help(
                                "explicit conversion: unit reading direction at the "
                                    + "reader's layer; the sidecar stamps source, reader "
                                    + "hash, and controlMode \"reading-vector activation "
                                    + "addition\"")
                        }
                        if record.id != readers.last?.id { Divider() }
                    }
                }
                .padding(.top, 4)
            }
        }
    }

    /// One clipboard rule for every prompt-copy button. On pasteboard
    /// failure the prompt is still recorded (`recordCopyFailure`) — the
    /// "Last copied prompt" disclosure is the manual-recovery path, so it
    /// must hold the prompt precisely when the clipboard does not.
    private func copyToClipboard(_ prompt: String, successMessage: String) {
        NSPasteboard.general.clearContents()
        if NSPasteboard.general.setString(prompt, forType: .string) {
            builder.recordCopiedPrompt(prompt, message: successMessage)
        } else {
            builder.recordCopyFailure(
                prompt,
                message: "could not write the prompt to the macOS clipboard — it is still available under Last copied prompt")
        }
    }

    @ViewBuilder
    private var copyLLMPromptRow: some View {
        HStack {
            Button("Copy LLM prompt") {
                if let prompt = builder.generationPrompt() {
                    copyToClipboard(
                        prompt,
                        successMessage: builder.recipeFamily.isPaired
                            ? "prompt copied — run it in any LLM, then paste the response into Paired JSONL and click Add to set"
                            : "prompt copied — run it in any LLM, then paste the JSONL into the story box and click Add story rows")
                }
            }
            .help(
                "copies a recipe-specific prompt for any LLM subscription. The app parses the JSONL reply when you paste it back")

            Button("Copy Claude Cowork prompt") {
                if let prompt = builder.coworkGenerationPrompt() {
                    copyToClipboard(
                        prompt,
                        successMessage: "Claude Cowork prompt copied — paste the merged JSONL corpus into the Grand Mean story box")
                }
            }
            .disabled(builder.recipeFamily != .emotionGrandMean)
            .help(
                "copies instructions for Claude Cowork to spawn parallel agents, generate a balanced concept × topic story grid, and return one JSONL corpus")

            Text("paste JSONL back into the visible dataset box")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// The neutral-PC build ensures its own model locally
    /// (`buildNeutralPCBasis` → `ensureSelectedLocalModelLoaded`), so a cold
    /// start is enabled and only a genuine conflict disables it. In a SERVER
    /// workspace that ensure step is skipped and the build needs a resident
    /// local container, so the historical residency gate stands there.
    private var neutralPCBuildDisabled: Bool {
        if service.cluster.computeTarget == .server {
            return service.isBuildingNeutralPCBasis || service.state != .ready
        }
        return service.localBuildBlockedReason(
            isBuilding: service.isBuildingNeutralPCBasis) != nil
    }

    @ViewBuilder
    private var neutralNormCorpusSection: some View {
        @Bindable var builder = service.concepts
        let selectedCorpus = service.selectedNeutralCorpus
        let summary = builder.neutralCorpusSummary
        let normSummary = builder.normNeutralCorpusSummary
        Section("Neutral Corpora") {
            Picker(
                "Selected corpus",
                selection: Binding<String?>(
                    get: { service.selectedNeutralCorpusID },
                    set: { service.selectedNeutralCorpusID = $0 }))
            {
                ForEach(service.neutralCorpora) { corpus in
                    Text("\(corpus.label) · \(corpus.kind.label)")
                        .tag(String?.some(corpus.id))
                }
            }
            .help("choose the neutral corpus to import into or use for building neutral PCs")

            LabeledContent("Selected rows", value: "\(summary.count)")
            if let hash = summary.hash {
                LabeledContent("Hash", value: String(hash.prefix(12)))
                    .help("this hash is stamped into neutral PC basis artifacts")
            }
            LabeledContent("Norm rows", value: "\(normSummary.count)")
                .help("the broad calibration corpus remains separate from projection corpora")

            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 6) {
                GridRow {
                    Text("Projection name")
                    TextField("assistant-dialogue-neutral", text: $builder.projectionNeutralCorpusName)
                        .textFieldStyle(.roundedBorder)
                    Button("Select/create") {
                        service.selectProjectionNeutralCorpus(named: builder.projectionNeutralCorpusName)
                    }
                    .help("select or create a named projection corpus under prompts/neutral/projection/")
                }
                GridRow {
                    Text("Neutral against")
                    TextField("fear, joy, arousal, valence", text: $builder.projectionNeutralConceptsDraft)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(2)
                }
                GridRow {
                    Text("Matched domains")
                    TextField("workplace, household planning, technical help", text: $builder.projectionNeutralDomainsDraft)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(2)
                }
                GridRow {
                    Text("Avoid settings")
                    TextField("danger, illness, moral judgment", text: $builder.projectionNeutralExclusionsDraft)
                        .textFieldStyle(.roundedBorder)
                        .gridCellColumns(2)
                }
            }
            .font(.caption)

            HStack {
                Button("Copy neutral corpus prompt") {
                    if let prompt = builder.neutralCorpusPrompt() {
                        copyToClipboard(
                            prompt,
                            successMessage: "neutral corpus prompt copied — paste JSONL below and import")
                    }
                }
                .help(
                    "copies a prompt for long, domain-neutral passages suitable for token-50 residual norm calibration")

                Button("Copy projection prompt") {
                    if let prompt = builder.anthropicStyleNeutralDialoguePrompt() {
                        copyToClipboard(
                            prompt,
                            successMessage: "projection-neutral dialogue prompt copied — paste JSONL below and import")
                    }
                }
                .help(
                    "copies a prompt for Human/Assistant dialogues neutral with respect to the listed concepts and matched domains")

                Button("Import to selected") {
                    builder.importNeutralCorpusDraft()
                }
                .disabled(builder.neutralCorpusDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("replace the selected neutral corpus with the pasted rows")

                Button {
                    Task { await service.buildNeutralPCBasis(allLayers: neutralPCAllLayers) }
                } label: {
                    if service.isBuildingNeutralPCBasis {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("Build neutral PCs")
                    }
                }
                .disabled(neutralPCBuildDisabled || summary.count == 0)
                .help(
                    "run the selected model (loading it first if it is not resident) over the "
                        + "selected neutral corpus, estimate token-position PCs "
                        + "over the middle-third layer band, and store them for optional steering-time projection")

                Toggle("All layers (expensive)", isOn: $neutralPCAllLayers)
                    .toggleStyle(.checkbox)
                    .font(.caption)
                    .disabled(service.isBuildingNeutralPCBasis)
                    .help(
                        "capture every layer instead of the middle-third band: ~3× the memory and "
                            + "~3× the PCA time, and steering lives in the middle third anyway")
            }
            if let neutralPCStatus = service.neutralPCStatus {
                Text(neutralPCStatus)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            ZStack(alignment: .topLeading) {
                TextEditor(text: $builder.neutralCorpusDraft)
                    .font(.caption.monospaced())
                    .frame(height: 120)
                    .help("paste JSONL rows like {\"text\":\"...\"}; JSON arrays and plain paragraph blocks also work")
                if builder.neutralCorpusDraft.isEmpty {
                    Text("Paste JSONL for \(selectedCorpus.label).")
                        .font(.callout)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            Text(
                "Use the norm corpus for residual calibration. Use named projection corpora for steering-time or study-time nuisance removal; those can be matched to a concept family."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var probeTrainingSection: some View {
        @Bindable var builder = service.concepts
        Section("Probe Training Data") {
            localEditNoticeRow
            LabeledContent(
                "Separate examples",
                value: "\(builder.probePositiveCount)+ / \(builder.probeNegativeCount)-"
            )
            .help(
                "held-out labeled sentences for training or validating a reading probe; "
                    + "these are not used to build steering vectors")

            HStack {
                Stepper(
                    "Prompt count: \(builder.probeGenerationCount)",
                    value: $builder.probeGenerationCount,
                    in: 20 ... 600,
                    step: 20)
                    .help("how many labeled probe sentences to ask the LLM to generate")
                Spacer()
                Button("Copy probe LLM prompt") {
                    if let prompt = builder.probeGenerationPrompt() {
                        copyToClipboard(
                            prompt,
                            successMessage: "probe prompt copied — paste JSONL below, then import it as separate probe data")
                    }
                }
                .disabled(builder.currentConceptName.isEmpty)
            }

            ZStack(alignment: .topLeading) {
                TextEditor(text: $builder.probeDraft)
                    .font(.callout)
                    .frame(minHeight: 120)
                    .help(
                        "paste strict JSONL probe examples here. Expected fields: text, expresses, topic, split")
                if builder.probeDraft.isEmpty {
                    Text("Paste probe JSONL here.")
                        .font(.callout)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }

            HStack {
                Button("Import probe examples") {
                    Task { await builder.addProbeDrafts() }
                }
                .disabled(builder.probeDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("save pasted probe examples to prompts/probes/<concept>/items.jsonl")
                if service.cluster.computeTarget == .server {
                    Button("Train probe on server") {
                        Task { await builder.trainProbeOnActiveServer() }
                    }
                    .disabled(service.selectedRemoteModelID == nil)
                    .help(
                        "queue probe training as a durable server job over the "
                            + "server's checkout of prompts/probes/<concept>/items.jsonl "
                            + "on the selected server model; the probe artifact stays "
                            + "server-side")
                } else {
                    Button("Train chat probe") {
                        Task { await builder.trainReadingProbe() }
                    }
                    .disabled(builder.isWorking || builder.probePositiveCount < 4 || builder.probeNegativeCount < 4)
                    .help(
                        "record activations for the independent probe examples, train a scalar "
                            + "reading probe across layers, save the best layer, and make it "
                            + "available in chat highlighting")
                    workingIndicator(task: "Training probe", fallback: "Training probe...")
                }
                Text("Probe data stays separate from CAA, RepE, and Grand Mean vector data.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let costLine = builder.probeTrainingCostLine() {
                Text(costLine)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help(
                        "cost of probe training: one forward pass per labeled "
                            + "example, counted from the on-disk probe items")
            }

            if !builder.probeExamples.isEmpty {
                DisclosureGroup("Browse probe examples (\(builder.probeExamples.count))") {
                    ForEach(Array(builder.probeExamples.enumerated()), id: \.element.id) { index, item in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(item.expresses ? "concept-present" : "control")
                                    .font(.caption)
                                    .bold()
                                if let topic = item.topic, !topic.isEmpty {
                                    Text(topic).font(.caption).foregroundStyle(.secondary)
                                }
                                Text(ConceptBuilder.canonicalSplit(item.split))
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Button("Remove") {
                                    builder.removeProbeExample(index: index)
                                }
                                .font(.caption)
                            }
                            Text(item.text)
                                .font(.caption)
                                .textSelection(.enabled)
                            Divider()
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var emotionConceptSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Concepts")
                    .font(.headline)
                Spacer()
                Button("All") {
                    builder.includeAllEmotionConcepts()
                }
                .disabled(builder.includedEmotionConcepts.isEmpty && !builder.grandMeanBuildConceptsAreExplicit)
                .help("include and build every available Grand Mean concept")
            }
            Text("Build saves a vector for that concept. Include controls which concepts contribute rows to the grand mean.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            ScrollView {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    GridRow {
                        Text("")
                        Text("Build")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Include in Grand Mean")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    ForEach(builder.emotionConceptOptions, id: \.self) { concept in
                        GridRow {
                            Text(concept)
                                .font(.caption)
                                .lineLimit(1)
                            Toggle(
                                "Build \(concept)",
                                isOn: Binding(
                                    get: { builder.grandMeanConceptWillBuild(concept) },
                                    set: { builder.setGrandMeanBuildConcept(concept, build: $0) }))
                                .labelsHidden()
                            Toggle(
                                "Include \(concept)",
                                isOn: Binding(
                                    get: { builder.emotionConceptIsIncluded(concept) },
                                    set: { builder.setEmotionConcept(concept, included: $0) }))
                                .labelsHidden()
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 180)
        }
    }

    @ViewBuilder
    private var topicSelector: some View {
        DisclosureGroup("Topics included in this vector") {
            HStack {
                Button("Use all topics") { builder.includeAllEmotionTopics() }
                    .disabled(builder.includedEmotionTopics.isEmpty)
                Text("Unchecked topics stay in the corpus but are excluded from this vector artifact.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            ForEach(builder.emotionBuildTopics, id: \.self) { topic in
                Toggle(
                    topic,
                    isOn: Binding(
                        get: { builder.emotionTopicIsIncluded(topic) },
                        set: { builder.setEmotionTopic(topic, included: $0) }))
            }
        }
    }

    /// Panel state at a glance: are the stats current, and is the working
    /// set saved?
    private var statusChip: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(
                builder.statsStale
                    ? (builder.pendingPassCount > 0
                        ? "stats stale (\(builder.pendingPassCount) passes)" : "stats stale")
                    : "stats current",
                systemImage: builder.statsStale ? "clock.arrow.circlepath" : "checkmark.circle"
            )
            .foregroundStyle(builder.statsStale ? .orange : .secondary)
            Label(
                builder.unsavedChanges ? "unsaved changes" : "saved",
                systemImage: builder.unsavedChanges
                    ? "exclamationmark.circle" : "checkmark.circle"
            )
            .foregroundStyle(builder.unsavedChanges ? .orange : .secondary)
        }
        .font(.caption)
        .help(
            "stats: whether the numbers below describe the current stimuli and "
                + "options. saved: whether the working set and options match the files "
                + "on disk and the newest vector artifact")
    }

    // MARK: Emotion corpus UI

    @ViewBuilder
    private var emotionStoryEntry: some View {
        @Bindable var builder = service.concepts
        let selectedConcept = builder.conceptName.isEmpty ? "selected concept" : builder.conceptName
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Vector derived later")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("mean(\(selectedConcept) build stories) - mean(selected build stories)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Text("Corpus generation is symmetric; no concept is special until vector build time.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 12) {
                    emotionMetadataFields(selectedConcept: selectedConcept)
                }
                VStack(alignment: .leading, spacing: 10) {
                    emotionMetadataFields(selectedConcept: selectedConcept)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Story")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $builder.multiConceptDraft)
                        .font(.callout)
                        .frame(height: 180)
                        .help(
                            "type or paste one story, or separate multiple stories with blank lines. "
                                + "The app fills concept/topic/split metadata. JSONL still works for bulk imports")
                    if builder.multiConceptDraft.isEmpty {
                        Text("Write or paste one story. Separate multiple stories with blank lines.")
                            .font(.callout)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
            }
            Text(
                "For each topic, add rows for every concept: fear/commute, joy/commute, anger/commute; then repeat for dinner, work, travel, and so on."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func emotionMetadataFields(selectedConcept: String) -> some View {
        @Bindable var builder = service.concepts
        VStack(alignment: .leading, spacing: 3) {
            Text("Story concept")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(selectedConcept, text: $builder.multiConceptStoryConceptDraft)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 150)
                .help(
                    "leave blank for the selected concept; enter another concept "
                        + "when adding rows to the shared Grand Mean corpus")
            Text("blank = \(selectedConcept)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        VStack(alignment: .leading, spacing: 3) {
            Text("Topic")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField("e.g. commute", text: $builder.multiConceptTopicDraft)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 150)
                .help("reuse the same topic names across concepts")
            Text("reuse across concepts")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        VStack(alignment: .leading, spacing: 3) {
            Text("Split")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("Split", selection: $builder.multiConceptSplitDraft) {
                ForEach(["build", "validation"], id: \.self) { split in
                    Text(split).tag(split)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 125)
            .help("build rows are used to extract the vector; validation rows are held out")
            Text("default: build")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var emotionCorpusSection: some View {
        let summary = builder.emotionCorpusSummary
        Section("Corpus Balance") {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                GridRow {
                    corpusMetric("Build rows", "\(summary.rowCount)")
                    corpusMetric("Validation", "\(summary.validationRowCount)")
                    corpusMetric("Concepts", "\(summary.conceptCount)")
                    corpusMetric("Topics", "\(summary.topicCount)")
                }
                GridRow {
                    corpusMetric("Total rows", "\(summary.totalRowCount)")
                    corpusMetric("Draft/other", "\(summary.draftRowCount)")
                    corpusMetric("Target rows", "\(summary.targetRowCount)")
                    corpusMetric("Selected", builder.includedEmotionTopics.isEmpty ? "all topics" : "\(builder.includedEmotionTopics.count) topics")
                }
            }
            if summary.conceptCount < 2 {
                Label("Add at least one comparison concept before extracting.", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if summary.isBalanced {
                Label("Topic/concept cells are balanced.", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Label(
                    "Balance warning: \(summary.missingCells.count) empty topic/concept cells; cell counts range \(summary.minCellCount)-\(summary.maxCellCount).",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }
            DisclosureGroup("Topic x concept grid") {
                ForEach(summary.topicCounts.map(\.name), id: \.self) { topic in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(topic).font(.caption).bold()
                        HStack(alignment: .firstTextBaseline) {
                            ForEach(summary.conceptCounts.map(\.name), id: \.self) { concept in
                                let count = summary.cellCounts.first {
                                    $0.topic == topic && $0.concept == concept
                                }?.count ?? 0
                                Text("\(concept): \(count)")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(count == 0 ? .red : .secondary)
                            }
                        }
                    }
                    Divider()
                }
            }
        }
    }

    private func corpusMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.headline.monospacedDigit())
        }
    }

    @ViewBuilder
    private var vectorLibrarySection: some View {
        Section("Vector Library") {
            if service.cluster.computeTarget == .local && service.loadedModelID == nil {
                Text("Load a model to see compatible vector artifacts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                if service.cluster.computeTarget == .server,
                    let error = service.catalog.remoteVectorsError
                {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .textSelection(.enabled)
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        Text(vectorLibraryCountLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()

                        Menu {
                            ForEach(VectorLibrarySort.allCases) { sort in
                                Button {
                                    vectorSort = sort
                                } label: {
                                    if vectorSort == sort {
                                        Label(sort.rawValue, systemImage: "checkmark")
                                    } else {
                                        Text(sort.rawValue)
                                    }
                                }
                            }
                        } label: {
                            Text(vectorSort.rawValue)
                        }
                        .menuStyle(.button)
                        .fixedSize()
                        .help("choose the sort field")

                        Button {
                            vectorSortAscending.toggle()
                        } label: {
                            Image(systemName: vectorSortAscending ? "arrow.up" : "arrow.down")
                        }
                        .help(vectorSortAscending ? "ascending order; click for descending" : "descending order; click for ascending")

                        Menu {
                            Button("All") {
                                vectorFilterShowsAll = true
                                vectorFilterConcepts = []
                            }
                            .disabled(vectorFilterShowsAll)
                            Button("None") {
                                vectorFilterShowsAll = false
                                vectorFilterConcepts = []
                            }
                            .disabled(!vectorFilterShowsAll && vectorFilterConcepts.isEmpty)
                            Divider()
                            ForEach(vectorLibraryConcepts, id: \.self) { concept in
                                Toggle(
                                    concept,
                                    isOn: Binding(
                                        get: {
                                            vectorFilterShowsAll
                                                || vectorFilterConcepts.contains(concept)
                                        },
                                        set: { included in
                                            if vectorFilterShowsAll {
                                                vectorFilterConcepts = Set(vectorLibraryConcepts)
                                                vectorFilterShowsAll = false
                                            }
                                            if included {
                                                vectorFilterConcepts.insert(concept)
                                            } else {
                                                vectorFilterConcepts.remove(concept)
                                            }
                                        }))
                            }
                        } label: {
                            Label(vectorFilterLabel, systemImage: "line.3.horizontal.decrease.circle")
                        }
                        .menuStyle(.button)
                        .fixedSize()
                        .disabled(vectorLibraryConcepts.isEmpty)
                        .help("filter by concept")

                        TextField("Search", text: $vectorSearchText)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 180)
                        .help("search concept, recipe, model, hash, or run path")
                    }

                    if libraryArtifacts.isEmpty {
                        Text("No compatible saved vectors match the filter.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(libraryArtifacts) { artifact in
                                    let isCurrentConcept = artifact.concept == currentConceptName
                                    let isInSteering = service.slots.contains(where: { $0.vectorID == artifact.id })
                                    let isCurrentModel =
                                        service.cluster.computeTarget == .server
                                        ? service.selectedRemoteModelID == artifact.modelID
                                        : service.loadedModelID == artifact.modelID
                                    VStack(alignment: .leading, spacing: 5) {
                                        HStack(alignment: .firstTextBaseline) {
                                            Text(artifact.label)
                                                .font(.caption)
                                                .fontWeight(.semibold)
                                                .lineLimit(2)
                                            Spacer()
                                            Button("Use") {
                                                useVector(artifact)
                                            }
                                            .help("select this vector into the first steering slot")
                                        }
                                        HStack(spacing: 8) {
                                            if isCurrentConcept {
                                                Text("default concept")
                                                    .font(.caption2)
                                                    .foregroundStyle(.blue)
                                            }
                                            Text(
                                                isCurrentModel
                                                    ? "current model"
                                                    : "available on server")
                                            .font(.caption2)
                                            .foregroundStyle(
                                                isCurrentModel ? .green : .secondary)
                                            if isInSteering {
                                                Text("in steering")
                                                    .font(.caption2)
                                                    .foregroundStyle(.orange)
                                            }
                                            Text(artifact.sourceLabel)
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                                            GridRow {
                                                vectorMetric("concept", artifact.concept)
                                                vectorMetric("model", artifact.modelID)
                                            }
                                            GridRow {
                                                vectorMetric("layers", "\(artifact.layerCount)")
                                                vectorMetric("recipe", artifact.recipe)
                                            }
                                            GridRow {
                                                vectorMetric("read", artifact.reading)
                                                vectorMetric(
                                                    "norm",
                                                    String(format: "%.3f", artifact.vectorNorm))
                                            }
                                            GridRow {
                                                vectorMetric(
                                                    "data",
                                                    String(artifact.stimulusHash.prefix(12)))
                                                vectorMetric(
                                                    "made",
                                                    dateTimeLabel(artifact.extractionDate))
                                            }
                                            GridRow {
                                                vectorMetric(
                                                    "path",
                                                    URL(fileURLWithPath: artifact.id)
                                                        .deletingLastPathComponent()
                                                        .lastPathComponent)
                                            }
                                        }
                                    }
                                    .padding(.vertical, 5)
                                    Divider()
                                }
                            }
                        }
                        .frame(maxHeight: 360)
                    }
                }
            }
        }
        .task(id: service.cluster.activeWorkspace) {
            if service.cluster.computeTarget == .server {
                await service.catalog.refreshRemoteVectors()
            } else {
                service.catalog.refreshLocalVectors()
            }
        }
    }

    private func useVector(_ artifact: VectorLibraryEntry) {
        if service.cluster.computeTarget == .server,
            service.selectedRemoteModelID == nil
        {
            service.selectedRemoteModelID = artifact.modelID
        }
        service.selectVector(artifact.id)
        service.steeringEnabled = true
    }

    private var vectorLibraryCountLabel: String {
        let total = activeVectorLibraryEntries.count
        guard libraryArtifacts.count != total else {
            return "\(total) compatible vector artifacts"
        }
        return "\(libraryArtifacts.count) of \(total) compatible vector artifacts"
    }

    private func vectorMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption2.monospaced())
                .lineLimit(2)
                .textSelection(.enabled)
        }
    }

    private func dateTimeLabel(_ iso: String) -> String {
        let trimmed = iso.replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
        guard trimmed.count > 16 else { return trimmed }
        return String(trimmed.prefix(16))
    }

    /// Pre-derivation reuse/cost surfacing (`ExperimentKit.DerivationPlanner`
    /// via `ConceptBuilder.vectorDerivationAdvice`): before any build, say
    /// what the user is getting into. A fresh artifact offers "Use existing /
    /// Re-extract anyway"; a stale one names the classifier's reason verbatim
    /// and offers only re-extraction (never silent reuse); otherwise an
    /// honest cost line for the new extraction.
    @ViewBuilder
    private var derivationAdviceRow: some View {
        if let advice = builder.vectorDerivationAdvice() {
            switch advice {
            case .reusable(let record, let date):
                VStack(alignment: .leading, spacing: 4) {
                    Label(
                        "A fresh vector for \(record.concept) @ \(record.modelID) "
                            + "already exists"
                            + (date.map { " (extracted \(dateTimeLabel($0)))" } ?? ""),
                        systemImage: "checkmark.seal")
                        .font(.caption)
                        .foregroundStyle(.green)
                    HStack(spacing: 8) {
                        if service.cluster.computeTarget == .server {
                            Text(record.id)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .textSelection(.enabled)
                                .help(
                                    "the fresh artifact's id (run directory/name) in the "
                                        + "server's catalog — pick it from the server "
                                        + "artifact lists where selection lives")
                        } else {
                            Button("Use existing") {
                                service.selectVector(record.id)
                                service.steeringEnabled = true
                            }
                            .disabled(
                                !service.compatibleVectors.contains { $0.id == record.id }
                            )
                            .help(
                                service.compatibleVectors.contains { $0.id == record.id }
                                    ? "select the existing artifact into the first steering slot instead of re-extracting"
                                    : "load \(record.modelID) before steering with this vector")
                        }
                        Button("Re-extract anyway") { reExtract() }
                            .disabled(reExtractDisabled)
                            .help("spend the forward passes again even though a fresh artifact exists")
                    }
                }
            case .staleExists(let record, let reason):
                VStack(alignment: .leading, spacing: 4) {
                    Label(
                        "existing vector for \(record.concept) @ \(record.modelID) "
                            + "is stale: \(reason)",
                        systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    Button("Re-extract") { reExtract() }
                        .disabled(reExtractDisabled)
                        .help("stale artifacts are never reused — re-extract with the current recipe pins")
                }
            case .new(let costLine):
                Label(costLine, systemImage: "gauge.with.needle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help(
                        "no existing artifact for this concept + model on the active "
                            + "workspace: rough forward-pass arithmetic from the on-disk "
                            + "recipe files, no time estimate")
            }
        }
    }

    /// Same routing as the main build button: in-process MLX locally, a
    /// durable job on the active server workspace.
    private func reExtract() {
        if service.cluster.computeTarget == .server {
            Task { await builder.buildVectorOnActiveServer() }
        } else {
            Task { await builder.saveConceptAndExtract() }
        }
    }

    private var reExtractDisabled: Bool {
        if service.cluster.computeTarget == .server {
            return serverBuildDisabled
        }
        return localBuildDisabled
    }

    /// The server build's enablement. A standing declaration refusal turns it
    /// off for the same reason it turns the local build off — building under
    /// the last VALID declaration would produce a vector nobody asked for —
    /// but NOT the two refusals that are this engine's limit alone (the
    /// assistant voice, `addGenerationPrompt: false`), whose own repair text
    /// says to extract on the server engine. Those are exactly the
    /// declarations this button exists to reach.
    private var serverBuildDisabled: Bool {
        service.selectedRemoteModelID == nil
            || builder.hasRefusedServerExtractionDeclaration
    }

    /// The local build actions' enablement, from the model layer. NOT "a model
    /// is loaded": `saveConceptAndExtract` starts by loading the selected
    /// model itself, so a cold start builds — only another user of the
    /// in-process model (a running build, a streaming generation, a load in
    /// flight) disables the button.
    private var localBuildDisabled: Bool {
        service.localBuildBlockedReason(isBuilding: builder.isWorking) != nil
    }

    /// Lights up when there is something to publish; un-lights when edits
    /// are reversed. In a server workspace the action routes to the server's
    /// extraction API instead of the in-process MLX path.
    @ViewBuilder
    private var saveButton: some View {
        if service.cluster.computeTarget == .server {
            serverBuildButton
        } else {
            let button = Button("Save & generate vector") {
                Task { await builder.saveConceptAndExtract() }
            }
            .disabled(localBuildDisabled || !builder.canSaveAndExtract)
            .help(
                "write the recipe-specific dataset and extract a vector into an "
                    + "immutable run directory. The sidecar records recipe method, "
                    + "recipe hash, reading position, and source data hash")
            HStack(spacing: 8) {
                if builder.canSaveAndExtract {
                    button.buttonStyle(.borderedProminent)
                } else {
                    button
                }
                workingIndicator(task: "Generating vector", fallback: "Generating vector...")
            }
        }
    }

    /// Server-workspace build: queue extraction as a durable job on the
    /// active server (loading the selected server model first if needed).
    @ViewBuilder
    private var serverBuildButton: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button("Build vector on \(service.cluster.substrateLabel)") {
                Task { await builder.buildVectorOnActiveServer() }
            }
            .disabled(serverBuildDisabled)
            .help(
                "queue extraction as a durable server job with this panel's "
                    + "method, reading position, and extraction rendering — "
                    + "the whole declaration travels, and the server echoes "
                    + "what it applied. The server reads its own tree of the "
                    + "stimuli (the build preflights it against this panel's "
                    + "data, syncing when safe, refusing on drift) and the "
                    + "vector lands in the server's catalog, not the local "
                    + "runs tree")
            if let refusal = builder.serverExtractionDeclarationRefusal {
                Text(refusal)
                    .font(.caption2)
                    .foregroundStyle(.orange)
            } else if builder.extractionRenderingRefusalIsLocalEngineLimit {
                // The one place the asymmetry is GOOD news: this pane's local
                // build is off and this button is not, because the rendering
                // is one the server engine renders and this one cannot.
                Text(
                    "this rendering is unavailable on the local engine and is "
                        + "why the local build is off — the server engine "
                        + "renders it, so build it here")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(
                "runs on the server's copy of the stimuli and the selected "
                    + "server model; progress appears in Compute and the "
                    + "Activity pane")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Banner-style build-failure row: a failed vector/reader/probe build is
    /// an event the researcher must SEE, not a caption to hunt for. The same
    /// failure also lands in the Activity pane's live log (ConceptBuilder
    /// routes it through the shared host). Success stays quiet.
    @ViewBuilder
    private var buildErrorBanner: some View {
        if let error = builder.lastBuildError {
            Section {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Label(error, systemImage: "exclamationmark.octagon.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                    Spacer()
                    Button("Dismiss") { builder.clearBuildError() }
                        .controlSize(.small)
                }
                .padding(8)
                .background(errorBannerBackground)
            }
        }
    }

    private var errorBannerBackground: some ShapeStyle {
        Color.red.opacity(0.08)
    }

    @ViewBuilder
    private func workingIndicator(task: String, fallback: String) -> some View {
        // Prefix match so substrate-suffixed labels ("Building reader
        // (server)") light the same indicator as their local counterpart.
        if builder.isWorking && builder.activeTaskLabel?.hasPrefix(task) == true {
            ProgressView()
                .controlSize(.small)
            Text(builder.status ?? fallback)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }

    /// Costed rebuilds are explicit: the label shows exactly how many
    /// forward passes will run. Free rebuilds happen automatically.
    @ViewBuilder
    private var rebuildButton: some View {
        let button = Button(
            builder.pendingPassCount > 0
                ? "Rebuild stats (\(builder.pendingPassCount) passes)" : "Rebuild stats"
        ) {
            Task { await builder.rebuild() }
        }
        // NOT relaxed to `localBuildDisabled`: `ConceptBuilder.rebuild()` reads
        // `host.containerForExtraction` / `host.loadedModelID` directly and has
        // no `ensureSelectedLocalModelLoaded` step, so with nothing resident it
        // would fail with "load a model first". The residency gate is honest
        // here until the rebuild path ensures its own model.
        .disabled(service.state != .ready || builder.isWorking)
        .help(
            builder.recipeFamily.isPaired
                ? "recompute the paired direction and stats. The pass count is the forward passes needed for uncached stimuli"
                : "live pair-margin stats are not defined for emotion grand-mean corpora; save to extract")
        if builder.statsStale {
            button.buttonStyle(.borderedProminent)
        } else {
            button
        }
    }

    @ViewBuilder
    private func stimulusRow(isPositive: Bool, index: Int) -> some View {
        let text = isPositive ? builder.positives[index] : builder.negatives[index]
        let margin = builder.stats?.marginByStimulus["\(isPositive ? "pos" : "neg")-\(index)"]
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(isPositive ? "+" : "−")
                .bold().monospaced()
                .foregroundStyle(isPositive ? .green : .secondary)
            Text(text)
                .font(.caption)
                .lineLimit(2)
                .textSelection(.enabled)
                .help(text)
            Spacer()
            if let margin {
                Text(margin.formatted(.number.precision(.fractionLength(2))))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(margin < 0 ? .red : .secondary)
                    .help("decision margin — negative means wrong side of the direction")
            }
            Button {
                builder.removeStimulus(isPositive: isPositive, index: index)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .controlSize(.small)
            .foregroundStyle(.secondary)
            .help("remove from working set (files unchanged until save)")
        }
    }

    @ViewBuilder
    private func statsSection(_ stats: ConceptBuilder.Stats) -> some View {
        Section(
            "Direction stats @ layer \(stats.statsLayer)"
                + (builder.statsStale ? "  ⚠︎ stale — rebuild to update" : "")
        ) {
            if let heldOut = stats.heldOut {
                LabeledContent("Held-out accuracy") {
                    Text(
                        "\(Int((heldOut.accuracy * 100).rounded()))%  (n=\(heldOut.testCount))"
                    )
                    .fontWeight(.semibold)
                }
                .help(
                    "the headline stat: ~20% of stimuli are excluded from the direction "
                        + "and classified by it. Out-of-sample by construction — this is "
                        + "the number to optimize while iterating")
            } else {
                LabeledContent("Held-out accuracy", value: "needs more stimuli")
                    .help("requires at least ~10 stimuli per side to hold out a test set")
            }

            if let splitHalf = stats.splitHalf {
                LabeledContent("Split-half cosine", value: formatted(splitHalf))
                    .help(
                        "cosine between directions from disjoint halves of the set. "
                            + "High = your stimuli agree on what the concept is; low = the "
                            + "set is heterogeneous (mixing sub-concepts)")
            }
            if let stability = stats.stability {
                LabeledContent("Stability vs last rebuild", value: formatted(stability))
                    .help(
                        "cosine vs the previous rebuild. Near 1.00 across several "
                            + "additions = the direction has converged; more stimuli are polish")
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("Direction norm by layer").font(.caption).foregroundStyle(.secondary)
                NormSparkline(values: stats.normByLayer, markedIndex: stats.statsLayer)
                    .frame(height: 36)
            }
            .help(
                "where in the network the contrast lives (dot = stats layer). Peaks in "
                    + "the middle third are typical; start layer sweeps there")
        }

        if !stats.outliers.isEmpty {
            Section("Least-aligned stimuli") {
                Text("stimuli pulling weakest in the concept's direction — rewrite or prune candidates (red = wrong side of the boundary)")
                    .font(.caption2).foregroundStyle(.secondary)
                ForEach(stats.outliers) { outlier in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(outlier.classLabel).bold().monospaced()
                        Text(outlier.textPrefix + "…")
                            .lineLimit(1)
                            .font(.caption)
                        Spacer()
                        Text(formatted(outlier.margin))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(outlier.margin < 0 ? .red : .secondary)
                    }
                }
            }
        }

        if !stats.controlCosines.isEmpty {
            Section("Cosine vs other concepts") {
                ForEach(stats.controlCosines) { control in
                    LabeledContent(control.name) {
                        Text(formatted(control.cosine))
                            .foregroundStyle(abs(control.cosine) > 0.6 ? .red : .primary)
                    }
                    .help(
                        "discriminant check vs '\(control.name)'. Red above |0.6| — the "
                            + "two concepts are collapsing into one direction. The built-in "
                            + "negative-valence and arousal controls guard against affect "
                            + "vectors being mere unpleasantness or excitement")
                }
                Text("High |cosine| means the concepts share a direction.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func formatted(_ value: Float) -> String {
        value.formatted(.number.precision(.fractionLength(2)))
    }
}

/// Minimal line sparkline with a marker at the stats layer.
struct NormSparkline: View {
    let values: [Float]
    let markedIndex: Int

    var body: some View {
        Canvas { context, size in
            guard values.count > 1, let maxValue = values.max(), maxValue > 0 else { return }
            func point(_ index: Int) -> CGPoint {
                CGPoint(
                    x: CGFloat(index) / CGFloat(values.count - 1) * size.width,
                    y: size.height - CGFloat(values[index] / maxValue) * (size.height - 4) - 2)
            }
            var path = Path()
            path.move(to: point(0))
            for index in 1 ..< values.count {
                path.addLine(to: point(index))
            }
            context.stroke(path, with: .color(.accentColor), lineWidth: 1.5)

            if values.indices.contains(markedIndex) {
                let mark = point(markedIndex)
                context.fill(
                    Path(ellipseIn: CGRect(x: mark.x - 3, y: mark.y - 3, width: 6, height: 6)),
                    with: .color(.accentColor))
            }
        }
    }
}

// MARK: - J-lens token direction rows (Concept Vector Builder)

extension ConceptsPanelView {

    /// The J-lens pane: resolved lens provenance, word → exact token, label.
    ///
    /// Four rows, one of which is not interactive. There is no lens PICKER
    /// because the published artifacts are one lens per model — the fitting
    /// corpus is a path segment, so more could exist later, but offering a
    /// choice of one is ceremony. Selecting the model resolves the lens; this
    /// shows what it resolved to.
    @ViewBuilder
    var jlensBuilderRows: some View {
        // Same local projection `body` uses: `builder` is a computed property,
        // so the $-binding has to be made where it is needed.
        @Bindable var builder = service.concepts
        if let lens = builder.jlensLens {
            jlensLensProvenance(lens)
        } else {
            jlensMissingLensNotice
        }

        HStack {
            TextField("word or string, e.g. courage", text: $builder.jlensQuery)
                .onSubmit { Task { await builder.lookUpJLensTokens() } }
            Toggle("case variants", isOn: $builder.jlensIncludeCaseVariants)
                .toggleStyle(.checkbox)
            Button("Token options") { Task { await builder.lookUpJLensTokens() } }
                .disabled(builder.jlensQuery.isEmpty || builder.jlensLensID == nil)
        }

        if let options = builder.jlensTokenOptions {
            ForEach(options.candidates) { candidate in
                jlensCandidateRow(candidate)
            }
        }

        TextField("name, e.g. courage", text: $builder.jlensLabel)
            .help("your label; the token id is appended so two directions with "
                  + "the same label can never be confused")

        // Optional, and empty by default. Associating files this direction under
        // a concept in every concept-grouped view — useful when you mean it
        // (validating a lens token against that concept's held-out probe is a
        // real test), misleading when it happens by accident, so it is never
        // implied by the name.
        Picker("Associate with concept (optional)",
               selection: $builder.jlensAssociatedConcept) {
            Text("none — groups only with itself").tag("")
            ForEach(builderExistingConceptNames, id: \.self) { name in
                Text(name).tag(name)
            }
        }
        if !builder.jlensAssociatedConcept.isEmpty {
            Label(
                "will appear beside \(builder.jlensAssociatedConcept)'s "
                    + "stimulus-extracted vectors — a single token is not that "
                    + "concept, only a direction named for one of its words",
                systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let name = builder.jlensArtifactName {
            HStack(spacing: 6) {
                Text("saves as").foregroundStyle(.secondary)
                Text(name).font(.caption.monospaced()).textSelection(.enabled)
                if builder.jlensDuplicateExists(
                    in: service.catalog.vectorArtifacts(
                        for: service.cluster.activeWorkspace)) {
                    // Same name means same lens AND same token — a duplicate to
                    // reuse, not a clash to version around.
                    Label("already derived", systemImage: "checkmark.circle")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption)
        } else {
            Text("select an exact token to name the vector")
                .font(.caption).foregroundStyle(.secondary)
        }

        if !ConceptBuilder.RecipeFamily.jlensTokenDirection.arrivesWithResidualNorms {
            // Every other family produces norm-ready vectors; say so before
            // someone reaches for alpha-in-norm-units and gets a refusal later.
            Label(
                "norm-unit alpha needs the residual-norm backfill first — "
                    + "derivation measures no neutral-corpus denominator",
                systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Provenance, not a control. The fit revision is UNKNOWN for the published
    /// artifacts and stays visibly unknown — blank would invite filling it in
    /// from the runtime.
    @ViewBuilder
    private func jlensLensProvenance(_ lens: JLensRecord) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text("Lens").foregroundStyle(.secondary)
                Text(lens.lensID).font(.caption.monospaced())
                if let tier = builder.jlensLensTier {
                    TierBadge(tier: tier, expanded: false)
                }
            }
            Text("\(lens.converted?.layerCount ?? lens.sourceLayers?.count ?? 0) "
                 + "matrices · layers \(lens.layerSpan)")
            Text("\(lens.fit?.corpus ?? "?") · "
                 + "\(lens.fit?.promptsFitted.map(String.init) ?? "?") prompts · "
                 + "\(lens.fit?.dtype ?? "?") · fit revision "
                 + ((lens.fit?.revisionKnown ?? false)
                    ? (lens.fit?.revision ?? "?") : "unknown"))
            if lens.passingQualifications.isEmpty {
                Text("not qualified for this runtime — exploration only")
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var jlensMissingLensNotice: some View {
        HStack(spacing: 8) {
            Label("No lens imported for this model", systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.orange)
            Button("Import lens") {
                Task { await builder.importJLensForSelectedModel() }
            }
            .controlSize(.small)
        }
    }

    private func jlensCandidateRow(_ candidate: JLensTokenCandidate) -> some View {
        let isSelected = builder.jlensSelectedTokenID == candidate.tokenID
            && builder.jlensSelectedPiece == (candidate.decoded ?? candidate.piece)
        return VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                Text(String(candidate.tokenID)).font(.caption.monospaced())
                Text(candidate.form).font(.caption2).foregroundStyle(.secondary)
                if candidate.singleToken {
                    Text("single").font(.caption2).foregroundStyle(.green)
                } else {
                    Text("multi-token").font(.caption2).foregroundStyle(.orange)
                }
                Text(candidate.decoded.map { "\"\($0)\"" } ?? candidate.piece)
                    .font(.caption.monospaced())
                Spacer()
            }
            if let note = candidate.note {
                Text(note).font(.caption2).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { builder.selectJLensToken(candidate) }
    }
}
