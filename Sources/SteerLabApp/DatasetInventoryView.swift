import AppKit
import ExperimentKit
import SwiftUI

/// The Data section's landing region: an inventory TABLE of what this
/// workspace holds. The selected row's DETAIL renders in the workbench's
/// right-hand display pane (`DataInventoryDetailColumn`, below) — the same
/// split as Results, where the run list lives in the main pane and the
/// selection's contents in the viewer (2026-08-26; it used to render inline
/// underneath this table).
///
/// Two SCOPES (WP-Data phase 3): **Datasets** — the source data a recipe
/// reads — and **Derived** — the artifacts those recipes produced. They are
/// separate tables on purpose: a dataset has items and a size, an artifact
/// has a model, a method, and a birth date, and one column vocabulary
/// stretched over both would make every column mean less. What they share is
/// the region's shape (header, table) and the model (`DatasetInventoryModel`
/// scans both in one refresh, and owns the scope + selection so the display
/// pane can render them).
///
/// Layout note (macOS 27 beta hazard, project memory "split-view min-size
/// crashes"): this is a plain `VStack` inside the existing main pane — no new
/// split-view column, and every frame minimum below is a CONSTANT, never one
/// that varies with the selection, the empty state, or the scope.
struct DatasetInventoryView: View {
    @Bindable var service: ChatService
    /// Switch to the Concepts tool, optionally selecting a concept first.
    let openInConceptBuilder: (String?) -> Void

    @State private var kindFilter: DatasetKind?
    @State private var derivedKindFilter: DerivedArtifactKind?
    @State private var sortOrder = [
        KeyPathComparator(\DatasetInventoryEntry.name, order: .forward)
    ]
    @State private var derivedSortOrder = [
        KeyPathComparator(\DerivedArtifactEntry.sortableCreated, order: .reverse)
    ]
    /// The row a just-finished creation should land on. The scan is async, so
    /// the selection is applied when the entries arrive rather than guessed at
    /// dismissal time.
    @State private var pendingSelection: DatasetInventoryEntry.ID?

    /// Constant, never mode- or scope-dependent (see the layout note above).
    private static let tableMinimumHeight: CGFloat = 160

    private var model: DatasetInventoryModel { service.datasetInventory }

    /// Scope and selection live on the model — the display pane renders the
    /// selected row (see the type comment). The controls bind through it,
    /// the same shape the Analysis pane uses for the shared `GeometryPanel`.
    private var scope: DatasetInventoryModel.Scope { model.scope }

    private var scopeBinding: Binding<DatasetInventoryModel.Scope> {
        Binding(get: { model.scope }, set: { model.scope = $0 })
    }

    private var selectionBinding: Binding<DatasetInventoryEntry.ID?> {
        Binding(get: { model.selection }, set: { model.selection = $0 })
    }

    private var derivedSelectionBinding: Binding<DerivedArtifactEntry.ID?> {
        Binding(get: { model.derivedSelection }, set: { model.derivedSelection = $0 })
    }

    private var rows: [DatasetInventoryEntry] {
        let filtered = kindFilter.map { kind in
            model.entries.filter { $0.kind == kind }
        } ?? model.entries
        return filtered.sorted(using: sortOrder)
    }

    private var derivedRows: [DerivedArtifactEntry] {
        let filtered = derivedKindFilter.map { kind in
            model.derived.filter { $0.kind == kind }
        } ?? model.derived
        return filtered.sorted(using: derivedSortOrder)
    }

    var body: some View {
        VStack(spacing: 0) {
            scopePicker
            header
            Divider()
            Group {
                switch scope {
                case .datasets: tableRegion
                case .derived: derivedTableRegion
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .frame(minHeight: Self.tableMinimumHeight)
            Divider()
            detailPaneCaption
        }
        .task { model.refresh() }
        .onChange(of: model.entries) { _, entries in
            guard let pending = pendingSelection else { return }
            pendingSelection = nil
            // A creation that produced no dataset row yet (an empty concept
            // skeleton the builder will fill) simply has nothing to select.
            guard let entry = entries.first(where: { $0.id == pending })
            else { return }
            model.scope = .datasets
            if let kindFilter, kindFilter != entry.kind { self.kindFilter = nil }
            model.selection = entry.id
        }
    }

    /// One fixed-height line saying where the selected row's detail went —
    /// the same courtesy the Analysis pane pays its tables ("cosine and RSA
    /// tables render in the viewer pane on the right").
    private var detailPaneCaption: some View {
        Text(
            scope == .datasets
                ? "select a dataset — its path, hash, and the tools that open "
                    + "it render in the display pane on the right"
                : "select an artifact — its provenance (model, method, source "
                    + "data, hashes) renders in the display pane on the right")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.tail)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
    }

    // MARK: Scope

    private var scopePicker: some View {
        Picker("", selection: scopeBinding) {
            ForEach(DatasetInventoryModel.Scope.allCases) { item in
                Text(item.rawValue).tag(item)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .help(
            "Datasets: the source data a recipe reads. Derived: the vectors, "
                + "probes, adapters, neutral-PC bases, and agents those "
                + "recipes produced — read-only.")
    }

    // MARK: Header

    /// The one creation entry, shared with the Concepts & Vectors tool
    /// (`NewDatasetButton` owns the sheet). Both of this view's triggers
    /// route through it, and both land the researcher on the new row: the
    /// entry id is remembered and applied when the async re-scan arrives.
    private func creationButton(
        systemImage: String? = "plus", help: String? = nil
    ) -> some View {
        NewDatasetButton(
            service: service,
            onCreated: { entryID in
                pendingSelection = entryID
                model.refresh()
            },
            openInConceptBuilder: openInConceptBuilder,
            title: "New Dataset…",
            systemImage: systemImage,
            help: help
                ?? "declare what the dataset is, then file it in the one "
                    + "place its recipe reads")
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(scope == .datasets ? "Datasets" : "Derived artifacts")
                    .font(.headline)
                Text(summaryLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            if scope == .datasets {
                creationButton()

                Picker("Kind", selection: $kindFilter) {
                    Text("All kinds").tag(DatasetKind?.none)
                    ForEach(model.presentKinds) { kind in
                        Text(kind.label).tag(DatasetKind?.some(kind))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 190)
                .help("show one dataset family at a time")
            } else {
                Picker("Kind", selection: $derivedKindFilter) {
                    Text("All kinds").tag(DerivedArtifactKind?.none)
                    ForEach(model.presentDerivedKinds) { kind in
                        Text(kind.label).tag(DerivedArtifactKind?.some(kind))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 190)
                .help("show one artifact family at a time")
            }

            Button {
                model.refresh()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .disabled(model.isScanning)
            .help(
                "re-scan the workspace (both scopes). The inventory also "
                    + "refreshes on a workspace switch, alongside the other "
                    + "catalogs.")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var summaryLine: String {
        guard model.hasScanned else {
            return model.isScanning ? "scanning the workspace…" : "not scanned yet"
        }
        var parts: [String] = []
        switch scope {
        case .datasets:
            parts.append("\(model.entries.count) dataset\(model.entries.count == 1 ? "" : "s")")
            if model.issueCount > 0 {
                parts.append("\(model.issueCount) with issues")
            }
        case .derived:
            parts.append(
                "\(model.derived.count) artifact\(model.derived.count == 1 ? "" : "s")")
            parts.append("read-only")
        }
        if let root = model.scannedRoot {
            parts.append(root.lastPathComponent)
        }
        return parts.joined(separator: " · ")
    }

    // MARK: Datasets table

    @ViewBuilder
    private var tableRegion: some View {
        if model.hasScanned, model.entries.isEmpty {
            emptyState
        } else if model.hasScanned, rows.isEmpty {
            filteredEmptyState { kindFilter = nil }
        } else {
            Table(rows, selection: selectionBinding, sortOrder: $sortOrder) {
                TableColumn("Dataset", value: \.name) { entry in
                    HStack(spacing: 6) {
                        if entry.issue != nil {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .help(entry.issue ?? "")
                        }
                        Text(entry.name)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                TableColumn("Kind", value: \.kindLabel) { entry in
                    Text(entry.kindLabel)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                TableColumn("Items", value: \.sortableItemCount) { entry in
                    Text(entry.itemCountText)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(entry.itemCount == nil ? .secondary : .primary)
                        .help(
                            entry.itemCount == nil
                                ? "row count unavailable — see the detail pane"
                                : "rows on disk")
                }
                TableColumn("Size", value: \.byteSize) { entry in
                    Text(entry.sizeText)
                        .font(.callout.monospacedDigit())
                }
                TableColumn("Modified", value: \.sortableModified) { entry in
                    Text(entry.modifiedText)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(entry.modified == nil ? .secondary : .primary)
                }
            }
            .tableStyle(.inset)
        }
    }

    // MARK: Derived table (its OWN columns — see the type comment)

    @ViewBuilder
    private var derivedTableRegion: some View {
        if model.hasScanned, model.derived.isEmpty {
            derivedEmptyState
        } else if model.hasScanned, derivedRows.isEmpty {
            filteredEmptyState { derivedKindFilter = nil }
        } else {
            Table(
                derivedRows, selection: derivedSelectionBinding,
                sortOrder: $derivedSortOrder
            ) {
                TableColumn("Name", value: \.name) { entry in
                    Text(entry.name)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                TableColumn("Kind", value: \.kind.label) { entry in
                    Text(entry.kind.label)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                TableColumn("Model", value: \.modelText) { entry in
                    Text(entry.modelText)
                        .lineLimit(1)
                        .truncationMode(.head)
                        .foregroundStyle(entry.modelID == nil ? .secondary : .primary)
                        .help(entry.modelText)
                }
                TableColumn("Method / detail", value: \.detail) { entry in
                    Text(entry.detail)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .help(entry.detail)
                }
                TableColumn("Created", value: \.sortableCreated) { entry in
                    Text(entry.createdText)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(
                            entry.sortableCreated == .distantPast ? .secondary : .primary)
                        .help(entry.createdStamp)
                }
            }
            .tableStyle(.inset)
        }
    }

    private var emptyState: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Label("No datasets in this workspace", systemImage: "tray")
                    .font(.headline)
                Text(
                    "This inventory lists the data a study reads: concept "
                        + "stimuli, paired stimuli, story corpora, probe items, "
                        + "held-out validation sets, neutral corpora, capability "
                        + "batteries, and task/dev prompt sets. A new workspace "
                        + "starts concept-empty — nothing is created here until "
                        + "you author it."
                )
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                creationButton(
                    systemImage: nil,
                    help:
                        "declare the dataset's role first; the flow files it "
                        + "in the canonical place for that recipe")
                Text(
                    "Already have data? The same flow imports a file into the "
                        + "right destination. The builders in Concepts & "
                        + "Vectors are where rows are then edited and built "
                        + "from — creation is here."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 520, alignment: .leading)
            .padding(24)
        }
        .frame(maxWidth: .infinity)
    }

    private var derivedEmptyState: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Label("Nothing derived yet", systemImage: "shippingbox")
                    .font(.headline)
                Text(
                    "Vectors, reading probes, LoRA adapters, neutral-PC bases, "
                        + "and agents appear here once a build produces them. "
                        + "This scope only reports: artifacts live under runs/, "
                        + "which is immutable, and nothing is created, edited, "
                        + "or deleted from this table."
                )
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                Text(
                    "Start from a dataset: select one in the Datasets scope and "
                        + "use its Derive action to open the builder that "
                        + "produces the artifact."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 520, alignment: .leading)
            .padding(24)
        }
        .frame(maxWidth: .infinity)
    }

    private func filteredEmptyState(clear: @escaping () -> Void) -> some View {
        VStack(spacing: 8) {
            Text("Nothing of this kind")
                .font(.callout)
                .foregroundStyle(.secondary)
            Button("Show all kinds", action: clear)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// The display pane's Data mode: the DETAIL of the row selected in the
/// Inventory — its facts, its provenance, and the routes into the tool that
/// owns it. Read-only, exactly as the inline detail underneath the table was
/// (2026-08-26 — the table, its filters, and the selection stay in the Data
/// main pane; the selection's contents render here, the same split as
/// Results). Scope and selection come from the shared
/// `DatasetInventoryModel`, so this pane and the table can never disagree
/// about what is selected.
struct DataInventoryDetailColumn: View {
    @Bindable var service: ChatService
    /// The routing seam, performed by `DataSectionRouting` on the Data
    /// section's tool binding — this pane names a destination and never
    /// starts a build.
    let route: (DatasetRouteRequest) -> Void

    private var model: DatasetInventoryModel { service.datasetInventory }

    var body: some View {
        Group {
            switch model.scope {
            case .datasets: datasetDetail
            case .derived: derivedDetail
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Datasets detail

    @ViewBuilder
    private var datasetDetail: some View {
        if let entry = model.selectedEntry {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(entry.name)
                            .font(.headline)
                        Text(entry.kindLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    Text(entry.kind.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    if let note = entry.note {
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let issue = entry.issue {
                        Label(issue, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    detailRow("Path", entry.displayPath(), monospaced: true)
                    if entry.files.count > 1 {
                        detailRow(
                            "Files",
                            entry.files.map(\.lastPathComponent).joined(separator: ", "),
                            monospaced: true)
                    }
                    detailRow("Items", entry.itemCountText)
                    detailRow("Size", entry.sizeText)
                    detailRow("Modified", entry.modifiedText)
                    if let hash = entry.contentHash {
                        detailRow("sha256", hash, monospaced: true)
                    } else {
                        detailRow(
                            "sha256",
                            "—  (no store pins a hash for this family)")
                    }

                    deriveActions(for: entry)
                    actions(for: entry)
                        .padding(.top, 2)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            emptyState(
                "No dataset selected", systemImage: "square.stack.3d.up",
                description: "Select a dataset in the Data pane; its path, "
                    + "hash, and the tools that open it render here.")
        }
    }

    /// The third verb (Dataset → Check → **Derive**). Every button ROUTES to
    /// the tool that already owns the build; the gating and the destination
    /// sentence both come from `DatasetDerivation`, so the view decides
    /// nothing.
    @ViewBuilder
    private func deriveActions(for entry: DatasetInventoryEntry) -> some View {
        let derivations = DatasetDerivation.actions(for: entry)
        if !derivations.isEmpty {
            HStack(spacing: 8) {
                Text("Derive")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 72, alignment: .leading)
                ForEach(derivations) { action in
                    Button(action.title) {
                        route(
                            .conceptBuilder(
                                concept: action.concept,
                                grandMeanRecipe: action.setsGrandMeanRecipe))
                    }
                    .help(action.destination)
                }
                Spacer()
            }
            .controlSize(.small)
            .padding(.top, 2)
        } else if let reason = DatasetDerivation.noDerivationReason(for: entry) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("Derive")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 72, alignment: .leading)
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(.top, 2)
        }
    }

    // MARK: Derived detail

    @ViewBuilder
    private var derivedDetail: some View {
        if let entry = model.selectedDerived {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(entry.name)
                            .font(.headline)
                        Text(entry.kind.label)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    Text(entry.kind.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    detailRow("Path", entry.displayPath, monospaced: true)
                    ForEach(entry.facts) { fact in
                        detailRow(
                            fact.label, fact.value,
                            monospaced: fact.label.hasSuffix("sha256")
                                || fact.label.hasSuffix("sha256 (pinned)"))
                    }

                    derivedActions(for: entry)
                        .padding(.top, 2)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            emptyState(
                "No artifact selected", systemImage: "shippingbox",
                description: "Select an artifact in the Data pane; its "
                    + "provenance — model, method, source data, hashes — "
                    + "renders here.")
        }
    }

    private func emptyState(
        _ title: String, systemImage: String, description: String
    ) -> some View {
        ContentUnavailableView {
            Label(title, systemImage: systemImage)
        } description: {
            Text(description)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func detailRow(
        _ label: String, _ value: String, monospaced: Bool = false
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
            Text(value)
                .font(monospaced ? .caption.monospaced() : .caption)
                .textSelection(.enabled)
                .lineLimit(2)
                .truncationMode(.middle)
                .help(value)
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private func actions(for entry: DatasetInventoryEntry) -> some View {
        HStack(spacing: 8) {
            if let concept = entry.conceptName {
                Button("Open in Concept Builder") {
                    route(.conceptBuilder(concept: concept, grandMeanRecipe: false))
                }
                .help("switch to Concepts & Vectors with \(concept) selected")
            } else if entry.kind == .neutralCorpus {
                Button("Open in Concepts & Vectors") {
                    route(.conceptBuilder(concept: nil, grandMeanRecipe: false))
                }
                .help(
                    "neutral corpora are edited in the Concepts & Vectors tool's "
                        + "neutral-corpus section")
            }
            Button("Reveal in Finder") {
                let targets = entry.files.isEmpty ? [entry.directory] : entry.files
                NSWorkspace.shared.activateFileViewerSelecting(targets)
            }
            Button("Copy Path") {
                copy(entry.primaryURL.path)
            }
            .help("copy the absolute path")
            Spacer()
        }
        .controlSize(.small)
    }

    /// Read-only, by design: reveal, copy, and AT MOST ONE route to the tool
    /// that owns this artifact. No delete and no edit — runs/ is immutable,
    /// and even the mutable library subtrees are not mutated from here.
    @ViewBuilder
    private func derivedActions(for entry: DerivedArtifactEntry) -> some View {
        HStack(spacing: 8) {
            if let entryRoute = entry.route {
                Button(entryRoute.label) {
                    route(.derived(entryRoute, selection: entry.selectionKey))
                }
                .help(entryRoute.help)
            }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([entry.primaryURL])
            }
            Button("Copy Path") {
                copy(entry.primaryURL.path)
            }
            .help("copy the absolute path")
            Spacer()
        }
        .controlSize(.small)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// What the inventory asks the Data section to do. A closed vocabulary: the
/// inventory names a DESTINATION, and the section — the only view that knows
/// both its own tool tabs and the workbench's sections — performs it.
enum DatasetRouteRequest {
    /// Concepts & Vectors, optionally with a concept selected and the
    /// grand-mean recipe chosen (in that order — loading a concept resolves
    /// its recipe from disk, so the override must come after).
    case conceptBuilder(concept: String?, grandMeanRecipe: Bool)
    /// A derived artifact's single home, plus the id that home selects it by
    /// (`DerivedArtifactEntry.selectionKey`) where the route preselects.
    case derived(DerivedArtifactRoute, selection: String?)
}
