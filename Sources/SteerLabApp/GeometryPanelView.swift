import ExperimentKit
import SteeringKit
import SwiftUI

struct GeometryPanelView: View {
    @Bindable var service: ChatService
    // The LOCAL vector selection now lives on the shared `GeometryPanel`
    // (phase 4): the computed tables render in the right-hand viewer, and a
    // cross-section route ("Open in Analysis" on a Derived row) has to be
    // able to preselect the vector it names — which a view-local @State
    // cannot be. The SERVER selection below stays view-local: nothing routes
    // to it, and it is scoped to a connection rather than to the workspace.
    @State private var gemmaScopeLayer = 0
    @State private var gemmaScopeStatus: String?
    @State private var isRunningGemmaScope = false
    @State private var logitLensLayer = 0
    @State private var isRunningLogitLens = false
    @State private var logitLensReport: LogitLensReport?
    @State private var logitLensStatus: String?
    @State private var gemmaScopeReports: [GemmaScopeReportArtifact] = []
    @State private var selectedGemmaScopeReportID: GemmaScopeReportArtifact.ID?
    @State private var gemmaScopeImportStatus: String?
    // Server-workspace geometry (POST /api/geometry): selection over the
    // server catalog, one layer per request. Compute state and the resulting
    // matrix live on the shared GeometryPanel; the table renders in the
    // right-hand viewer.
    @State private var serverSelectedIDs: Set<String> = []
    @State private var serverLayer = 0
    // Server-workspace Gemma Scope (POST /api/gemmascope/run): a durable
    // server job over a SERVER vector artifact, followed in the shared
    // Activity live log; reports and imported features live on the
    // server's tree, never merged locally.
    @State private var serverGemmaScopeLayer = 0
    @State private var isRunningServerGemmaScope = false
    @State private var serverGemmaScopeStatus: String?
    @State private var serverScopeReport: RemoteGemmaScopeReport?
    @State private var serverScopeReportPath: String?
    @State private var serverScopeReportRuns: [RemoteRunRecord] = []
    @State private var selectedServerScopeRunID: String?
    @State private var serverScopeImportStatus: String?

    private var selectedIDs: Set<VectorArtifact.ID> { service.geometry.selectedIDs }

    private var selectedArtifacts: [VectorArtifact] {
        service.compatibleVectors.filter { selectedIDs.contains($0.id) }
    }

    /// Vectors a route preselected that this pane cannot list — the local
    /// list is filtered to the LOADED model, so an artifact extracted against
    /// another model is selected and invisible. Said out loud rather than
    /// silently dropped.
    private var unlistableSelection: [String] {
        service.geometry.unlistableSelection(among: service.compatibleVectors)
    }

    /// Shared compute state — the viewer column renders its tables.
    private var geometry: GeometryPanel { service.geometry }

    // MARK: Substrate awareness (the Analysis surface owns its prerequisite)

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    /// Same vocabulary as Optimizations: where the listed vectors come from.
    private var sourceLabel: String {
        isServerWorkspace ? service.cluster.substrateLabel : "local workspace"
    }

    private var isLoadingModel: Bool {
        if case .loading = service.state { return true }
        return false
    }

    private var loadButtonTitle: String {
        switch service.cluster.computeTarget {
        case .local:
            switch service.localModelButtonAction {
            // Same rule as Playground: weights that are not here get a button
            // that says Download, never one that says Load.
            case .download: "Download…"
            case .load, .busy: service.state == .ready ? "Reload" : "Load"
            }
        case .server: "Load on Server"
        }
    }

    /// Local: Load loads, Download downloads. Shared with the "needs a model"
    /// empty state so both entry points behave identically.
    private func runLoadButtonAction() {
        guard service.cluster.computeTarget == .local else {
            Task { await service.loadWorkspaceModel() }
            return
        }
        switch service.localModelButtonAction {
        case .download(let model):
            Task { await service.installWorkspaceModel(model) }
        case .load:
            Task { await service.loadWorkspaceModel() }
        case .busy:
            break
        }
    }

    private var loadButtonDisabled: Bool {
        switch service.cluster.computeTarget {
        case .local: isLoadingModel || service.modelInstaller.isInstalling
        // Same eligibility rule as Playground (engineer review 2026-07-18:
        // Geometry's gate missed both the single-flight guard and the
        // memory-fit check, so an oversized stale selection stayed loadable
        // here after the picker grayed it out).
        case .server:
            service.selectedRemoteModelID == nil
                || service.isRemoteModelLoading
                || SessionModelFit.tooBigNote(
                    cluster: service.cluster,
                    model: service.selectedRemoteModelID) != nil
        }
    }

    private var modelStatusLine: String {
        if isServerWorkspace {
            return service.cluster.status ?? "not connected"
        }
        switch service.state {
        case .unloaded: return "no model loaded"
        case .loading(let percent): return "loading… \(percent)%"
        case .ready: return "loaded: \(service.loadedModelID ?? "?")"
        }
    }

    private var selectedServerRecords: [RemoteVectorRecord] {
        service.compatibleServerVectors.filter { serverSelectedIDs.contains($0.id) }
    }

    private var serverLayerLimit: Int {
        let counts = selectedServerRecords.map(\.layerCount)
        return max(0, (counts.max() ?? 1) - 1)
    }

    private var selectedSingleArtifact: VectorArtifact? {
        selectedArtifacts.count == 1 ? selectedArtifacts[0] : nil
    }

    private var gemmaScopeLayerLimit: Int {
        max(0, (selectedSingleArtifact?.sidecar.layerCount ?? 1) - 1)
    }

    private var gemmaScopeInfo: GemmaScopeInfo? {
        let layerCount = selectedSingleArtifact?.sidecar.layerCount
            ?? service.compatibleVectors.first?.sidecar.layerCount
        return GemmaScopeCatalog.info(
            for: service.loadedModelID ?? service.selectedModelID,
            layerCount: layerCount,
            preferredLayer: min(gemmaScopeLayer, gemmaScopeLayerLimit))
    }

    // MARK: Server Gemma Scope helpers

    private var selectedSingleServerRecord: RemoteVectorRecord? {
        selectedServerRecords.count == 1 ? selectedServerRecords[0] : nil
    }

    private var serverGemmaScopeLayerLimit: Int {
        max(0, (selectedSingleServerRecord?.layerCount ?? 1) - 1)
    }

    /// Same gate as local (`GemmaScopeCatalog.info` is nil off Gemma 3 —
    /// the server enforces the same rule server-side): suite info follows
    /// the selected SERVER vector's model, falling back to the selected
    /// server model so the section can explain itself before a selection.
    private var serverGemmaScopeInfo: GemmaScopeInfo? {
        guard let record = selectedSingleServerRecord else {
            return GemmaScopeCatalog.info(for: service.selectedRemoteModelID)
        }
        return GemmaScopeCatalog.info(
            for: record.modelID,
            layerCount: record.layerCount,
            preferredLayer: min(serverGemmaScopeLayer, serverGemmaScopeLayerLimit))
    }

    private var selectedGemmaScopeReport: GemmaScopeReportArtifact? {
        if let selectedGemmaScopeReportID,
            let selected = gemmaScopeReports.first(where: { $0.id == selectedGemmaScopeReportID })
        {
            return selected
        }
        return gemmaScopeReports.first
    }

    var body: some View {
        Form {
            modelSection
            if isServerWorkspace {
                serverGeometrySection
                serverVectorsSection
                // J-Space is mechanistic analysis over SERVER artifacts, so it
                // belongs here and only in the server branch: there is no local
                // equivalent to show.
                JSpacePanelSection(service: service)
            } else {
                localGeometrySection
                localVectorsSection
            }

            Section("Logit Lens") {
                if isServerWorkspace {
                    Text(
                        "Logit Lens runs on the local substrate only (it reads "
                            + "the vector through the locally loaded model's "
                            + "unembed) — switch Compute to Local (MLX).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if service.state != .ready {
                    needsModelPrompt(
                        "Logit Lens requires a loaded model (local loaded model only).")
                } else if let artifact = selectedSingleArtifact {
                    LabeledContent("Vector", value: artifact.sidecar.concept)
                    Stepper(
                        "Layer: \(min(logitLensLayer, gemmaScopeLayerLimit))",
                        value: $logitLensLayer,
                        in: 0 ... gemmaScopeLayerLimit)
                    HStack {
                        Button(isRunningLogitLens ? "Reading…" : "Read Through Unembed") {
                            runLogitLens(artifact: artifact)
                        }
                        .disabled(isRunningLogitLens || service.state != .ready)
                        if isRunningLogitLens {
                            ProgressView()
                                .controlSize(.small)
                            Text("Projecting vector through output head…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Text("Shows the tokens most upweighted and downweighted by the residual vector at the selected layer. Local loaded model only.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let logitLensReport {
                        LogitLensReportView(report: logitLensReport)
                    }
                    if let logitLensStatus {
                        Text(logitLensStatus)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                } else {
                    Text("Select exactly one compatible vector to read it through the model unembed.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Gemma Scope") {
                if isServerWorkspace {
                    serverGemmaScopeSectionContent
                } else if let info = gemmaScopeInfo {
                    LabeledContent("Suite", value: info.suiteName)
                    LabeledContent("Model", value: "\(info.modelSize) \(info.tuning.uppercased())")
                    LabeledContent("Site", value: info.recommendedSite)
                    LabeledContent("Release", value: info.recommendedRelease)
                    LabeledContent("SAE layer", value: "\(info.recommendedLayer)")
                    LabeledContent("SAE id", value: info.recommendedSAEID)

                    if let artifact = selectedSingleArtifact {
                        Stepper(
                            "Target layer: \(min(gemmaScopeLayer, gemmaScopeLayerLimit))",
                            value: $gemmaScopeLayer,
                            in: 0 ... gemmaScopeLayerLimit)
                        Button(isRunningGemmaScope ? "Analyzing…" : "Run Gemma Scope Analysis") {
                            runGemmaScopeAnalysis(artifact: artifact, info: info)
                        }
                        .disabled(isRunningGemmaScope)
                    } else {
                        Text("Select exactly one vector to run a Gemma Scope analysis.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    HStack {
                        if let url = URL(string: info.landingPageURL) {
                            Link(info.suiteName, destination: url)
                        }
                        if let url = URL(string: info.repositoryURL) {
                            Link(info.repository, destination: url)
                        }
                    }

                    DisclosureGroup("SAELens snippet") {
                        Text(info.saeLensSnippet)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.vertical, 4)
                    }

                    ForEach(info.notes, id: \.self) { note in
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if let gemmaScopeStatus {
                        Text(gemmaScopeStatus)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                } else {
                    Text("Load a Gemma 3 model to see Gemma Scope 2 repositories and SAELens snippets.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Gemma Scope Reports") {
                if isServerWorkspace {
                    serverGemmaScopeReportsContent
                } else {
                    localGemmaScopeReportsContent
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            refreshGemmaScopeReports()
            retireOrphanedGeometryTables()
        }
        .task {
            // Server workspace: make sure the vector catalog is populated so
            // the list isn't stale-empty on first visit (same shared catalog
            // the steering pickers read).
            if isServerWorkspace {
                await service.catalog.refreshRemoteVectors()
                refreshServerScopeReportRuns()
            }
        }
        .onChange(of: service.loadedModelID) {
            geometry.clearSelection()
            geometry.clearLocal()
            gemmaScopeLayer = 0
            logitLensLayer = 0
            logitLensReport = nil
            logitLensStatus = nil
            gemmaScopeStatus = nil
        }
        .onChange(of: selectedIDs) {
            gemmaScopeLayer = min(gemmaScopeLayer, gemmaScopeLayerLimit)
            logitLensLayer = min(logitLensLayer, gemmaScopeLayerLimit)
            gemmaScopeStatus = nil
            logitLensReport = nil
            logitLensStatus = nil
        }
        .onChange(of: serverSelectedIDs) { oldValue, newValue in
            // First selection defaults to the middle of the network (the
            // usual steering sweet spot); afterwards only clamp.
            if oldValue.isEmpty, !newValue.isEmpty {
                serverLayer = serverLayerLimit / 2
            }
            serverLayer = min(serverLayer, serverLayerLimit)
            serverGemmaScopeLayer = min(serverGemmaScopeLayer, serverGemmaScopeLayerLimit)
            serverGemmaScopeStatus = nil
        }
        .onChange(of: service.selectedRemoteModelID) {
            serverSelectedIDs.removeAll()
            geometry.clearServer()
            serverGemmaScopeStatus = nil
        }
        .onChange(of: selectedServerScopeRunID) {
            loadSelectedServerScopeReport()
        }
    }

    // MARK: Model section (the surface owns its prerequisite)

    /// The same picker + load action as Playground's Model section
    /// (`WorkspaceModelPicker` + `ChatService.loadWorkspaceModel`) — one load
    /// path, surfaced where the analysis needs it.
    private var modelSection: some View {
        Section("Model") {
            WorkspaceModelPicker(service: service)
            HStack(spacing: 8) {
                Button(loadButtonTitle) { runLoadButtonAction() }
                    .disabled(loadButtonDisabled)
                    .help(
                        isServerWorkspace
                            ? "load the selected model on \(service.cluster.substrateLabel)"
                            : "load the selected model from this Mac's model "
                                + "cache; models that are not downloaded offer "
                                + "Download instead")
                if isServerWorkspace {
                    InstallModelButton(cluster: service.cluster)
                        .controlSize(.small)
                } else {
                    AddLocalModelButton(service: service)
                        .controlSize(.small)
                }
                if service.modelInstaller.isInstalling {
                    Button("Cancel Download") { service.modelInstaller.cancel() }
                        .controlSize(.small)
                }
                Spacer()
            }
            if !isServerWorkspace, let status = service.modelInstaller.statusLine {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(
                        service.modelInstaller.isFailed ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(modelStatusLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    /// Actionable "needs a model" empty state: says what is missing and
    /// offers the SAME load action, instead of a bare instruction.
    private func needsModelPrompt(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(loadPromptTitle) { runLoadButtonAction() }
                .disabled(loadButtonDisabled)
        }
    }

    private var loadPromptTitle: String {
        if isLoadingModel { return "Loading…" }
        let model = service.workspaceSelectedModelID ?? service.selectedModelID
        if !isServerWorkspace, case .download = service.localModelButtonAction {
            return "Download \(model)…"
        }
        return "Load \(model)"
    }

    private var sourceCaption: some View {
        Text("source: \(sourceLabel)")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .help(
                isServerWorkspace
                    ? "vectors listed from the active server's catalog; geometry computes server-side"
                    : "vectors listed from this workspace's runs/, extracted from the loaded model")
    }

    // MARK: Local sections

    private var localGeometrySection: some View {
        Section("Geometry") {
            HStack {
                Button("Select All") {
                    geometry.select(
                        vectorIDs: Set(service.compatibleVectors.map(\.id)))
                }
                .disabled(service.compatibleVectors.isEmpty)

                Button("Clear") { geometry.clearSelection() }
                    .disabled(selectedIDs.isEmpty)

                Button("Compute") { geometry.compute(artifacts: selectedArtifacts) }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(selectedArtifacts.count < 2)
            }

            if !unlistableSelection.isEmpty {
                Text(
                    "\(unlistableSelection.count) selected vector"
                        + (unlistableSelection.count == 1 ? " is" : "s are")
                        + " not listed below: this list is filtered to the "
                        + "LOADED model. Load the model they were extracted "
                        + "against to include them."
                )
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .help(unlistableSelection.joined(separator: "\n"))
            }

            if let result = geometry.result {
                LabeledContent("Vectors", value: "\(result.artifacts.count)")
                LabeledContent("Common layers", value: "\(result.matrices.count)")
                Stepper(
                    "Layer: \(geometry.selectedLayer)",
                    value: Binding(
                        get: { geometry.selectedLayer },
                        set: { geometry.selectedLayer = $0 }),
                    in: 0 ... max(0, result.matrices.count - 1))
            } else {
                Text("Select at least two compatible vectors.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("cosine and RSA tables render in the viewer pane on the right")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let status = geometry.status {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var localVectorsSection: some View {
        Section("Vectors") {
            sourceCaption
            if service.compatibleVectors.isEmpty {
                if service.state == .ready {
                    Text("No vectors for this model in runs/.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    needsModelPrompt(
                        "This analysis requires a loaded model — vectors are "
                            + "model-specific, so the list follows the loaded model.")
                }
            } else {
                ForEach(service.compatibleVectors) { artifact in
                    localVectorToggle(artifact)
                }
            }
        }
    }

    private func localVectorToggle(_ artifact: VectorArtifact) -> some View {
        Toggle(
            artifact.label,
            isOn: Binding(
                get: { selectedIDs.contains(artifact.id) },
                set: { geometry.setSelected(artifact.id, $0) }))
        .help(
            "\(artifact.sidecar.concept) · "
                + "\(artifact.sidecar.modelID) · "
                + "\(artifact.sidecar.stimulusSetHash.prefix(12))…")
    }

    // MARK: Server sections (geometry runs server-side over the catalog)

    private var serverGeometrySection: some View {
        Section("Geometry") {
            HStack {
                Button("Select All") {
                    serverSelectedIDs = Set(service.compatibleServerVectors.map(\.id))
                }
                .disabled(service.compatibleServerVectors.isEmpty)

                Button("Clear") {
                    serverSelectedIDs.removeAll()
                    geometry.clearServer()
                }
                .disabled(serverSelectedIDs.isEmpty)

                Button(geometry.isServerComputing ? "Computing…" : "Compute on Server") {
                    geometry.computeOnServer(
                        records: selectedServerRecords,
                        layer: min(serverLayer, serverLayerLimit))
                }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(selectedServerRecords.count < 2 || geometry.isServerComputing)
            }

            Stepper(
                "Layer: \(min(serverLayer, serverLayerLimit))",
                value: $serverLayer,
                in: 0 ... serverLayerLimit)
            .disabled(selectedServerRecords.isEmpty)

            Text(
                "Computes on \(service.cluster.substrateLabel) over its vector "
                    + "catalog — no loaded model needed for geometry; the cosine "
                    + "table renders in the viewer pane on the right.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if selectedServerRecords.count < 2 {
                Text("Select at least two server vectors.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let serverStatus = geometry.serverStatus {
                Text(serverStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private var serverVectorsSection: some View {
        Section("Vectors") {
            sourceCaption
            if service.compatibleServerVectors.isEmpty {
                Text(
                    "No vectors on \(service.cluster.substrateLabel) for the "
                        + "selected model. Extract one in Data (it runs as a "
                        + "server job), or pick a different model above.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(service.compatibleServerVectors) { record in
                    serverVectorToggle(record)
                }
            }
        }
    }

    private func serverVectorToggle(_ record: RemoteVectorRecord) -> some View {
        Toggle(
            "\(record.concept) · \(record.name)",
            isOn: Binding(
                get: { serverSelectedIDs.contains(record.id) },
                set: { isOn in
                    if isOn {
                        serverSelectedIDs.insert(record.id)
                    } else {
                        serverSelectedIDs.remove(record.id)
                    }
                    geometry.clearServer()
                }))
        .help("\(record.concept) · \(record.modelID) · \(record.id)")
    }

    // MARK: Gemma Scope content (local + server branches)

    /// The pre-existing local reports browser, unchanged — extracted so the
    /// section can branch per substrate without growing the body.
    @ViewBuilder
    private var localGemmaScopeReportsContent: some View {
        HStack {
            Button("Refresh Reports") { refreshGemmaScopeReports() }
            if let count = gemmaScopeReports.isEmpty ? nil : gemmaScopeReports.count {
                Text("\(count) report\(count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }

        if gemmaScopeReports.isEmpty {
            Text("Run a prepared Gemma Scope job to create gemmascope-report.json, then refresh.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Picker("Report", selection: reportSelectionBinding) {
                ForEach(gemmaScopeReports) { report in
                    Text(report.label).tag(Optional(report.id))
                }
            }

            if let report = selectedGemmaScopeReport {
                GemmaScopeReportView(report: report) { row, source in
                    importGemmaScopeFeature(report: report, row: row, source: source)
                }
            }
        }

        if let gemmaScopeImportStatus {
            Text(gemmaScopeImportStatus)
                .font(.caption)
                .textSelection(.enabled)
        }
    }

    /// Server-workspace Gemma Scope: the same inputs as local (one selected
    /// vector + a target-layer preference snapped to the published SAE
    /// layers, Gemma 3 models only — the identical gate), run as a durable
    /// SERVER job followed in the Activity live log.
    @ViewBuilder
    private var serverGemmaScopeSectionContent: some View {
        if let info = serverGemmaScopeInfo {
            LabeledContent("Suite", value: info.suiteName)
            LabeledContent("Model", value: "\(info.modelSize) \(info.tuning.uppercased())")
            LabeledContent("Release", value: info.recommendedRelease)
            LabeledContent("SAE layer", value: "\(info.recommendedLayer)")
            LabeledContent("SAE id", value: info.recommendedSAEID)

            if let record = selectedSingleServerRecord {
                Stepper(
                    "Target layer: \(min(serverGemmaScopeLayer, serverGemmaScopeLayerLimit))",
                    value: $serverGemmaScopeLayer,
                    in: 0 ... serverGemmaScopeLayerLimit)
                Button(isRunningServerGemmaScope ? "Analyzing…" : "Run on Server") {
                    runServerGemmaScope(record: record, info: info)
                }
                .disabled(isRunningServerGemmaScope)
            } else {
                Text("Select exactly one server vector to run a Gemma Scope analysis.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(
                "Runs as a durable job on \(service.cluster.substrateLabel) "
                    + "(needs sae-lens in the server's environment); the job is "
                    + "followed in Activity and the report lands in the server's "
                    + "runs/ — source: \(sourceLabel).")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let serverGemmaScopeStatus {
                Text(serverGemmaScopeStatus)
                    .font(.caption)
                    .textSelection(.enabled)
            }
        } else {
            Text(
                "Gemma Scope 2 covers Gemma 3 models only — select a Gemma 3 "
                    + "server vector (or pick a Gemma 3 server model) to see "
                    + "repositories and run a server-side analysis.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Server-side reports: `gemmascope-report.json` files discovered in
    /// the SERVER's runs/ tree, fetched on demand and rendered with the
    /// same feature-row views as local reports (the row shape is a pinned
    /// cross-engine contract; the envelope carries no artifact sidecar).
    @ViewBuilder
    private var serverGemmaScopeReportsContent: some View {
        HStack {
            Button("Refresh Server Reports") { refreshServerScopeReportRuns() }
            if !serverScopeReportRuns.isEmpty {
                let count = serverScopeReportRuns.count
                Text("\(count) report\(count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        sourceCaption

        if serverScopeReportRuns.isEmpty {
            Text(
                "Run a server Gemma Scope analysis to create "
                    + "gemmascope-report.json under the server's runs/, then refresh.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Picker("Report", selection: $selectedServerScopeRunID) {
                ForEach(serverScopeReportRuns) { run in
                    Text(run.id).tag(Optional(run.id))
                }
            }
        }

        if let report = serverScopeReport {
            RemoteGemmaScopeReportView(report: report) { row, source in
                importServerScopeFeature(row: row, source: source)
            }
        }

        if let serverScopeImportStatus {
            Text(serverScopeImportStatus)
                .font(.caption)
                .textSelection(.enabled)
        }
    }

    // MARK: Server Gemma Scope actions

    private func runServerGemmaScope(record: RemoteVectorRecord, info: GemmaScopeInfo) {
        guard let client = service.cluster.client else {
            serverGemmaScopeStatus = "no server connection — check the Compute selector"
            return
        }
        // Same layer rule as the local run: rank the vector at the SAE's
        // layer, clamped to the artifact (the server clamps again).
        let layer = min(info.recommendedLayer, serverGemmaScopeLayerLimit)
        isRunningServerGemmaScope = true
        serverGemmaScopeStatus =
            "queuing Gemma Scope analysis on \(service.cluster.substrateLabel)…"
        Task {
            do {
                let jobID = try await client.gemmaScopeRun(
                    vectorRunDirectory: record.runDirectory,
                    name: record.name,
                    modelID: record.modelID,
                    layer: layer,
                    release: info.recommendedRelease,
                    saeID: info.recommendedSAEID)
                serverGemmaScopeStatus = "server job \(jobID) — live log in Activity"
                let job = await service.followServerJobInActivity(
                    jobID: jobID, client: client,
                    title: "Server Gemma Scope: \(record.concept)")
                guard let job else {
                    isRunningServerGemmaScope = false
                    serverGemmaScopeStatus =
                        "Gemma Scope job \(jobID) is still running server-side — "
                        + "open Compute to keep watching"
                    return
                }
                if job.status == "succeeded" {
                    let reportPath = job.result.flatMap { result -> String? in
                        if case .string(let path)? = result["reportPath"] { return path }
                        return nil
                    }
                    if let reportPath {
                        await loadServerScopeReport(serverPath: reportPath, client: client)
                    }
                    serverGemmaScopeStatus =
                        "Gemma Scope analysis finished on \(service.cluster.substrateLabel)"
                        + (reportPath.map { " → \($0)" } ?? "")
                    refreshServerScopeReportRuns()
                } else {
                    serverGemmaScopeStatus =
                        "server Gemma Scope job \(jobID) \(job.status): "
                        + (job.error ?? job.logTail.last ?? "no detail")
                }
                isRunningServerGemmaScope = false
            } catch {
                isRunningServerGemmaScope = false
                serverGemmaScopeStatus = "\(error)"
            }
        }
    }

    /// Fetch + decode a server report by its server-side path; remembers the
    /// path because the import route addresses the report by path.
    private func loadServerScopeReport(serverPath: String, client: ClusterClient) async {
        let url = URL(filePath: serverPath)
        let runID = url.deletingLastPathComponent().lastPathComponent
        do {
            let report = try await client.gemmaScopeReport(runID: runID)
            serverScopeReport = report
            serverScopeReportPath = serverPath
            selectedServerScopeRunID = runID
        } catch {
            serverScopeImportStatus = "could not load server report: \(error)"
        }
    }

    private func refreshServerScopeReportRuns() {
        guard let client = service.cluster.client else { return }
        Task {
            if let runs = try? await client.runs() {
                serverScopeReportRuns = runs.filter {
                    $0.files.contains("gemmascope-report.json")
                }
            }
        }
    }

    private func loadSelectedServerScopeReport() {
        guard let client = service.cluster.client,
            let runID = selectedServerScopeRunID,
            let run = serverScopeReportRuns.first(where: { $0.id == runID })
        else { return }
        let serverPath = run.path + "/gemmascope-report.json"
        guard serverPath != serverScopeReportPath else { return }
        Task { await loadServerScopeReport(serverPath: serverPath, client: client) }
    }

    private func importServerScopeFeature(row: GemmaScopeFeatureRow, source: String) {
        guard let client = service.cluster.client,
            let reportPath = serverScopeReportPath
        else {
            serverScopeImportStatus = "load a server report first"
            return
        }
        Task {
            do {
                let imported = try await client.gemmaScopeImport(
                    reportPath: reportPath, feature: row.feature)
                await service.catalog.refreshRemoteVectors()
                serverScopeImportStatus =
                    "imported feature \(row.feature) (\(source)) as \(imported.name) → "
                    + "\(imported.vectorPath) on \(service.cluster.substrateLabel) — "
                    + "raw decoder direction at the report layer (NOT rescaled to "
                    + "the analyzed vector's norm, unlike local import); it is now "
                    + "in the server vector catalog"
            } catch {
                serverScopeImportStatus = "\(error)"
            }
        }
    }

    private var reportSelectionBinding: Binding<GemmaScopeReportArtifact.ID?> {
        Binding(
            get: { selectedGemmaScopeReport?.id },
            set: { selectedGemmaScopeReportID = $0 })
    }

    private func runGemmaScopeAnalysis(artifact: VectorArtifact, info: GemmaScopeInfo) {
        let layer = min(info.recommendedLayer, gemmaScopeLayerLimit)
        isRunningGemmaScope = true
        gemmaScopeStatus = "Running Gemma Scope analysis at layer \(layer)…"
        Task {
            do {
                let result = try await GemmaScopeAnalysis.run(
                    artifact: artifact, layer: layer, info: info)
                await MainActor.run {
                    isRunningGemmaScope = false
                    refreshGemmaScopeReports()
                    if result.succeeded {
                        gemmaScopeStatus =
                            "\(result.summary)\n\(result.prepared.directory)"
                    } else {
                        gemmaScopeStatus =
                            "\(result.summary)\n\(result.stderr.isEmpty ? result.stdout : result.stderr)"
                    }
                }
            } catch {
                await MainActor.run {
                    isRunningGemmaScope = false
                    gemmaScopeStatus = "\(error)"
                }
            }
        }
    }

    private func runLogitLens(artifact: VectorArtifact) {
        let layer = min(logitLensLayer, gemmaScopeLayerLimit)
        isRunningLogitLens = true
        logitLensStatus = nil
        logitLensReport = nil
        Task {
            do {
                let report = try await service.logitLens(for: artifact, layer: layer, topK: 12)
                await MainActor.run {
                    isRunningLogitLens = false
                    logitLensReport = report
                    logitLensStatus = "Read \(artifact.sidecar.concept) through unembed at layer \(report.layer)."
                }
            } catch {
                await MainActor.run {
                    isRunningLogitLens = false
                    logitLensStatus = "\(error)"
                }
            }
        }
    }

    /// Viewer-state hygiene (2026-08-19): the computed cosine/RSA tables live
    /// on the shared `GeometryPanel` (they render in the right-hand viewer),
    /// but the vector SELECTION they were computed from is this view's own
    /// state and does not survive a section switch. Returning to Analysis
    /// therefore showed tables beside an empty selection list, describing
    /// vectors the pane no longer named. Orphaned content loses to the empty
    /// state: recompute is one click, and the same rule already governs the
    /// Results viewer's run selection.
    private func retireOrphanedGeometryTables() {
        geometry.retireOrphanedLocalTable()
        if serverSelectedIDs.isEmpty { geometry.clearServer() }
    }

    private func refreshGemmaScopeReports() {
        gemmaScopeReports = GemmaScopeReportCatalog.scan()
        if let selectedGemmaScopeReportID,
            gemmaScopeReports.contains(where: { $0.id == selectedGemmaScopeReportID })
        {
            return
        }
        selectedGemmaScopeReportID = gemmaScopeReports.first?.id
    }

    private func importGemmaScopeFeature(
        report: GemmaScopeReportArtifact,
        row: GemmaScopeFeatureRow,
        source: String
    ) {
        do {
            let artifact = try GemmaScopeReportCatalog.importFeature(
                report: report, row: row, source: source)
            service.refreshVectors()
            geometry.select(vectorIDs: [artifact.id])
            gemmaScopeImportStatus =
                "Imported feature \(row.feature) as \(artifact.sidecar.concept). It is now available for steering and studies."
        } catch {
            gemmaScopeImportStatus = "\(error)"
        }
    }
}

private struct GemmaScopeReportView: View {
    let report: GemmaScopeReportArtifact
    let importFeature: (GemmaScopeFeatureRow, String) -> Void

    var body: some View {
        LabeledContent("Vector", value: report.report.vector.concept)
        LabeledContent("Layer", value: "\(report.report.vector.layer)")
        LabeledContent("SAE", value: report.report.gemmaScope.recommendedSAEID)
        LabeledContent("Report", value: report.url.lastPathComponent)

        DisclosureGroup("Top positive") {
            GemmaScopeFeatureRowsView(
                rows: report.report.topPositive, source: "top-positive",
                importTitle: "Import", importFeature: importFeature)
        }
        DisclosureGroup("Top negative") {
            GemmaScopeFeatureRowsView(
                rows: report.report.topNegative, source: "top-negative",
                importTitle: "Import", importFeature: importFeature)
        }
        DisclosureGroup("Top absolute") {
            GemmaScopeFeatureRowsView(
                rows: report.report.topAbsolute, source: "top-absolute",
                importTitle: "Import", importFeature: importFeature)
        }
    }
}

/// A server-fetched Gemma Scope report: the SAME feature-row rendering as
/// the local report view (the rows are a pinned cross-engine shape), with
/// the narrower server envelope shown honestly — no vector/sidecar
/// metadata exists server-side, so none is displayed, and import mints a
/// SERVER-side artifact.
private struct RemoteGemmaScopeReportView: View {
    let report: RemoteGemmaScopeReport
    let importFeature: (GemmaScopeFeatureRow, String) -> Void

    var body: some View {
        LabeledContent("SAE", value: report.saeID)
        LabeledContent("Layer", value: "\(report.layer)")
        LabeledContent("Release", value: report.release)

        DisclosureGroup("Top positive") {
            GemmaScopeFeatureRowsView(
                rows: report.topPositive, source: "top-positive",
                importTitle: "Import on Server", importFeature: importFeature)
        }
        DisclosureGroup("Top negative") {
            GemmaScopeFeatureRowsView(
                rows: report.topNegative, source: "top-negative",
                importTitle: "Import on Server", importFeature: importFeature)
        }
        DisclosureGroup("Top absolute") {
            GemmaScopeFeatureRowsView(
                rows: report.topAbsolute, source: "top-absolute",
                importTitle: "Import on Server", importFeature: importFeature)
        }

        Text(
            "server report — carries no local artifact sidecar; Import on "
                + "Server mints the raw decoder direction as a vector in the "
                + "SERVER catalog (not rescaled to the analyzed vector's norm, "
                + "unlike local import)")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}

/// Shared row rendering for local and server Gemma Scope reports — one
/// visual vocabulary per feature row on both substrates.
private struct GemmaScopeFeatureRowsView: View {
    let rows: [GemmaScopeFeatureRow]
    let source: String
    let importTitle: String
    let importFeature: (GemmaScopeFeatureRow, String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(rows.prefix(8)) { row in
                HStack {
                    Text("F\(row.feature)")
                        .font(.caption.monospacedDigit())
                        .frame(width: 70, alignment: .leading)
                    Text(row.cosine.formatted(.number.precision(.fractionLength(3))))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(row.cosine >= 0 ? .blue : .red)
                    Spacer()
                    Button(importTitle) {
                        importFeature(row, source)
                    }
                    .disabled(row.decoderValues == nil)
                }
                .help(row.decoderValues == nil ? "rerun the Gemma Scope job to export decoder vectors" : "")
            }
        }
        .padding(.vertical, 4)
    }
}

private struct LogitLensReportView: View {
    let report: LogitLensReport

    var body: some View {
        LabeledContent("Layer", value: "\(report.layer)")
        DisclosureGroup("Top upweighted tokens", isExpanded: .constant(true)) {
            tokenRows(report.topPositive, positive: true)
        }
        DisclosureGroup("Top downweighted tokens") {
            tokenRows(report.topNegative, positive: false)
        }
    }

    private func tokenRows(_ rows: [LogitLensToken], positive: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(rows, id: \.tokenID) { row in
                HStack {
                    Text(row.token.debugDescription)
                        .font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Text(row.logit.formatted(.number.precision(.fractionLength(3))))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(positive ? .blue : .red)
                    Text("#\(row.tokenID)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

// GeometryMatrixView / GeometryRSAView / GeometryCell moved to
// GeometryViewerColumn.swift — computed tables render only in the viewer.
