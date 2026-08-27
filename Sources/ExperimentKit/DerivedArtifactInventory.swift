import Foundation
import Observation
import SteeringKit

/// The Data section's DERIVED half (WP-Data phase 3).
///
/// Phase 1 deliberately listed source DATASETS only, because a derived
/// artifact's columns mean something else: a vector has no "items" and a
/// probe has no "size" worth reading — they have a model, a method, a layer,
/// and a birth date. So the Inventory region carries a scope switch, and this
/// is the other scope's engine. The two share the model, the refresh
/// lifecycle, and the detail-pane shape; they do NOT share a column
/// vocabulary.
///
/// Design rules, unchanged from phase 1 and phase 2:
///
/// - **Every kind is enumerated by the store that already owns it.**
///   `VectorCatalog.scan` (saved vectors), `ProbeCatalog.scan` (trained
///   reading probes), `FineTuneStore.scan` (LoRA adapters),
///   `NeutralPCStore.scan` (neutral-PC bases), `ModelVariantStore.scan`
///   (agents). No directory walk, no filename convention, and no sidecar
///   schema is re-implemented here — this file reads records and projects
///   them onto rows.
/// - **Provenance comes from the artifact's own sidecar.** Every fact in the
///   detail pane is a field the artifact itself carries. Where a sidecar has
///   no such field, the row says so rather than deriving one.
/// - **Read-only, and not just by convention.** `runs/` is immutable and the
///   three library subtrees (`model-variants/`, `neutral-pcs/`,
///   `jlens-lenses/`) are the only mutable ones — this scope offers no
///   delete, no rename, and no edit even there. Reveal, copy the path, and
///   at most one route to the tool that owns the artifact.
public enum DerivedArtifactInventory {

    /// Enumerate the workspace's derived artifacts. Pure and synchronous —
    /// the model runs it off the main actor, exactly like the dataset scan.
    ///
    /// `root` defaults to the RESOLVED workspace, so a workspace switch
    /// changes the result with no caller involvement.
    public static func scan(root: URL = VectorCatalog.projectRoot) -> [DerivedArtifactEntry] {
        let runs = VectorCatalog.runsDirectory(root: root)
        var entries: [DerivedArtifactEntry] = []
        entries.append(contentsOf: vectorEntries(runs: runs, root: root))
        entries.append(contentsOf: probeEntries(runs: runs, root: root))
        entries.append(contentsOf: adapterEntries(runs: runs, root: root))
        entries.append(contentsOf: neutralPCEntries(runs: runs, root: root))
        entries.append(contentsOf: agentEntries(runs: runs, root: root))

        return entries.sorted { left, right in
            if left.kind != right.kind {
                return left.kind.sortIndex < right.kind.sortIndex
            }
            if left.sortableCreated != right.sortableCreated {
                return left.sortableCreated > right.sortableCreated  // newest first
            }
            return left.name.localizedStandardCompare(right.name) == .orderedAscending
        }
    }

    // MARK: Saved vectors (VectorCatalog)

    private static func vectorEntries(runs: URL, root: URL) -> [DerivedArtifactEntry] {
        VectorCatalog.scan(runsDirectory: runs).map { artifact in
            let sidecar = artifact.sidecar
            // `VectorArtifact` names its pair by directory + base name; the
            // safetensors file is the artifact and the JSON is its sidecar.
            let vectorURL = artifact.directory.appending(component: "\(artifact.name).safetensors")
            let sidecarURL = artifact.directory.appending(component: "\(artifact.name).json")
            let method = sidecar.extractionMethod ?? sidecar.recipeMethod
            var facts: [DerivedArtifactFact] = [
                .init("Concept", sidecar.concept),
                .init("Model", sidecar.modelID),
            ]
            facts.appendIfPresent("Revision", sidecar.revision)
            facts.appendIfPresent("Method", method)
            facts.appendIfPresent("Recipe", sidecar.recipeName)
            facts.append(.init("Layers", "\(sidecar.layerCount)"))
            facts.append(.init("Hidden size", "\(sidecar.hiddenSize)"))
            facts.appendIfPresent("Reading position", sidecar.readingPosition)
            facts.appendIfPresent("Residual-norm source", sidecar.residualNormSource)
            facts.appendIfPresent("Neutral projection", sidecar.neutralProjection)
            facts.appendIfPresent("Substrate", sidecar.substrate)
            facts.append(.init("Stimulus sha256", sidecar.stimulusSetHash))
            // A mirrored pole inherits the stimulus hash above (same two files,
            // roles swapped), so the inventory row would otherwise be
            // indistinguishable from its source's while the two directions
            // point opposite ways.
            if let negation = sidecar.negatedFrom {
                facts.append(
                    .init(
                        "Mirrored from",
                        "\(negation.concept) — every layer × −1 "
                            + "(\(negation.path))"))
            }
            facts.appendIfPresent("Recipe sha256", sidecar.recipeHash)
            facts.appendIfPresent("Recipe identity sha256", sidecar.recipeIdentityHash)
            facts.appendIfPresent("Neutral corpus sha256", sidecar.neutralCorpusHash)
            facts.append(.init("Extracted", sidecar.extractionDate))

            return DerivedArtifactEntry(
                kind: .steeringVector,
                name: artifact.name,
                primaryURL: vectorURL,
                sidecarURL: sidecarURL,
                directory: artifact.directory,
                root: root,
                modelID: sidecar.modelID,
                detail: [method, "\(sidecar.layerCount) layers"]
                    .compactMap { $0 }.joined(separator: " · "),
                createdStamp: sidecar.extractionDate,
                concept: sidecar.concept,
                facts: facts,
                route: .analysis)
        }
    }

    // MARK: Trained reading probes (ProbeCatalog)

    private static func probeEntries(runs: URL, root: URL) -> [DerivedArtifactEntry] {
        ProbeCatalog.scan(runsDirectory: runs).map { record in
            let probe = record.artifact
            var facts: [DerivedArtifactFact] = [
                .init("Concept", probe.concept),
                .init("Model", probe.modelID),
            ]
            facts.appendIfPresent("Revision", probe.revision)
            facts.append(.init("Layer", "\(probe.layer)"))
            facts.append(.init("Recipe", probe.recipeName))
            facts.appendIfPresent("Recipe sha256", probe.recipeHash)
            facts.appendIfPresent("Calibration sha256", probe.calibrationHash)
            facts.appendIfPresent("Validation sha256", probe.validationHash)
            facts.append(.init("Created", probe.createdAt))
            facts.appendIfPresent("Notes", probe.notes)

            return DerivedArtifactEntry(
                kind: .readingProbe,
                name: record.url.deletingPathExtension().deletingPathExtension()
                    .lastPathComponent,
                primaryURL: record.url,
                sidecarURL: record.url,
                directory: record.url.deletingLastPathComponent(),
                root: root,
                modelID: probe.modelID,
                detail: "\(probe.recipeName) · layer \(probe.layer)",
                createdStamp: probe.createdAt,
                concept: probe.concept,
                facts: facts,
                route: .conceptsAndVectors)
        }
    }

    // MARK: Trained adapters (FineTuneStore — the Adapter Training panel's store)

    private static func adapterEntries(runs: URL, root: URL) -> [DerivedArtifactEntry] {
        FineTuneStore.scan(directory: runs.appending(component: "fine-tunes")).map { record in
            let artifact = record.artifact
            var facts: [DerivedArtifactFact] = [
                .init("Base model", artifact.baseModelID)
            ]
            facts.appendIfPresent("Base revision", artifact.baseRevision)
            facts.append(.init("Type", artifact.fineTuneType))
            facts.appendIfPresent("Adapter format", artifact.adapterFormat)
            facts.appendIfPresent("Substrate", artifact.substrate)
            facts.append(
                .init(
                    "Hyperparameters",
                    "rank \(artifact.rank) · scale \(artifact.scale) · "
                        + "\(artifact.adaptedLayers) layers · \(artifact.iterations) iters "
                        + "· batch \(artifact.batchSize) · lr \(artifact.learningRate)"))
            facts.append(.init("Adapter directory", artifact.adapterDirectory))
            facts.appendIfPresent("Training data", artifact.trainingDataPath)
            facts.appendIfPresent("Training data sha256", artifact.trainingDataHash)
            facts.appendIfPresent("Validation data", artifact.validationDataPath)
            facts.appendIfPresent("Adapter sha256", artifact.adapterHash)
            facts.appendIfPresent("Config sha256", artifact.configHash)
            facts.append(.init("Created", artifact.createdAt))
            facts.appendIfPresent(
                "Notes", artifact.notes.isEmpty ? nil : artifact.notes)

            return DerivedArtifactEntry(
                kind: .adapter,
                name: artifact.name,
                primaryURL: record.url,
                sidecarURL: record.url,
                directory: record.url.deletingLastPathComponent(),
                root: root,
                modelID: artifact.baseModelID,
                detail: "\(artifact.fineTuneType) · rank \(artifact.rank) · "
                    + "\(artifact.iterations) iters",
                createdStamp: artifact.createdAt,
                concept: nil,
                facts: facts,
                route: .adapterTraining)
        }
    }

    // MARK: Neutral-PC bases (NeutralPCStore — runs/neutral-pcs library)

    private static func neutralPCEntries(runs: URL, root: URL) -> [DerivedArtifactEntry] {
        NeutralPCStore.scan(directory: runs.appending(component: "neutral-pcs")).map { record in
            let basis = record.basis
            var facts: [DerivedArtifactFact] = [
                .init("Model", basis.modelID)
            ]
            facts.appendIfPresent("Revision", basis.modelRevision)
            facts.append(
                .init(
                    "Layers",
                    "\(basis.layers.count) captured"
                        + (basis.layers.isEmpty
                            ? ""
                            : " (\(basis.layers.min() ?? 0)–\(basis.layers.max() ?? 0))")))
            facts.append(.init("Components", basis.selectionDescription))
            facts.appendIfPresent("Layer band", basis.layerSelectionDescription)
            facts.append(.init("Reading position", basis.readingPosition))
            facts.append(.init("Token rows", "\(basis.tokenRowCount)"))
            facts.append(.init("Corpus", basis.corpusPath))
            facts.append(.init("Corpus sha256", basis.corpusHash))
            // THE pinned digest: `neutralPCBasisHash` is the SHA-256 of the
            // basis FILE BYTES (what `ExperimentStore.verify()` re-checks) —
            // not `corpusHash`, which pins only the corpus the PCA read. Both
            // are shown because a researcher chasing a pin mismatch needs to
            // know which one drifted.
            facts.appendIfPresent("Basis sha256 (pinned)", pinnedBasisHash(of: record.url))
            facts.append(.init("Created", basis.createdAt))

            return DerivedArtifactEntry(
                kind: .neutralPCBasis,
                name: record.url.deletingLastPathComponent().lastPathComponent,
                primaryURL: record.url,
                sidecarURL: record.url,
                directory: record.url.deletingLastPathComponent(),
                root: root,
                modelID: basis.modelID,
                detail: "\(basis.selectionDescription) · \(basis.layers.count) layers",
                createdStamp: basis.createdAt,
                concept: nil,
                facts: facts,
                route: .conceptsAndVectors)
        }
    }

    /// The digest the manifest pins and `verify()` re-checks, computed with
    /// the engine's own helper over the same bytes. Bounded by the dataset
    /// scan's parse budget so a pathological artifact cannot stall a UI
    /// refresh.
    static func pinnedBasisHash(of url: URL) -> String? {
        let stat = DatasetInventory.FileStat.total(of: [url])
        guard !stat.present.isEmpty, stat.bytes <= DatasetInventory.maximumParsedBytes,
            let data = try? Data(contentsOf: url)
        else { return nil }
        return ExperimentStore.sha256Hex(data)
    }

    // MARK: Agents (ModelVariantStore — a CROSS-REFERENCE, not a second home)

    private static func agentEntries(runs: URL, root: URL) -> [DerivedArtifactEntry] {
        // Both passes the store performs: the native `runs/model-variants/`
        // library AND the server's `runType: "variant-save"` run layout, so
        // the count here agrees with what the Agents section shows rather
        // than under-reporting server-promoted agents.
        ModelVariantStore.scan(
            directory: runs.appending(component: "model-variants"), importedRoot: runs
        ).map { record in
            let artifact = record.artifact
            let kind = AgentLibrary.kind(of: artifact)
            var facts: [DerivedArtifactFact] = [
                .init("Base model", artifact.baseModelID),
                .init("Kind", kind.label),
            ]
            facts.appendIfPresent("Base revision", artifact.baseRevision)
            facts.append(.init("Components", AgentLibrary.componentsSummary(artifact)))
            if let promotion = artifact.promotion {
                facts.append(.init("Promoted by", promotion.promotedBy))
                facts.append(.init("From study", promotion.experiment))
                facts.append(.init("Study sha256", promotion.experimentHash))
                facts.appendIfPresent(
                    "Winning cell",
                    promotion.winningCell.map { "layer \($0.layer) · α \($0.alpha)" })
                // The resolved criterion the sweep selected under — its
                // objective metric is the part a researcher reads at a
                // glance; the full block lives in the artifact.
                facts.appendIfPresent("Criterion", promotion.criterion?.objective?.metric)
                facts.appendIfPresent("Sweep run", promotion.sweepRun)
                facts.appendIfPresent("Override reason", promotion.overrideReason)
                facts.append(.init("Promoted at", promotion.promotedAt))
                facts.append(.init("Substrate", promotion.substrate))
            } else {
                facts.append(
                    .init(
                        "Promotion",
                        "none — hand-created, so it carries no birth certificate "
                            + "(a freeze advisory, not a refusal)"))
            }
            facts.append(.init("Created", artifact.createdAt))

            return DerivedArtifactEntry(
                kind: .agent,
                name: artifact.name,
                primaryURL: record.url,
                sidecarURL: record.url,
                directory: record.url.deletingLastPathComponent(),
                root: root,
                modelID: artifact.baseModelID,
                detail: kind.label,
                createdStamp: artifact.createdAt,
                concept: nil,
                facts: facts,
                route: .agents)
        }
    }
}

// MARK: - Kinds

/// The derived families this scope enumerates — each one a store that already
/// existed.
///
/// Two families are deliberately ABSENT. **Gemma Scope reports** and **J-lens
/// lenses** have artifacts on disk but their enumeration is scoped by a
/// selected model / remote engine rather than by the workspace alone
/// (`GemmaScopeReports`, `JLensRemote`), so a workspace-wide row would
/// misrepresent what the researcher can act on; J-lens is additionally
/// server-only by hard requirement, and nothing on this Mac may produce or
/// claim it. **Run directories** are not artifacts and already have the
/// Results section.
public enum DerivedArtifactKind: String, Sendable, CaseIterable, Identifiable, Codable {
    case steeringVector
    case readingProbe
    case adapter
    case neutralPCBasis
    case agent

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .steeringVector: "Vector"
        case .readingProbe: "Reading probe"
        case .adapter: "Adapter"
        case .neutralPCBasis: "Neutral-PC basis"
        case .agent: "Agent"
        }
    }

    /// One line of what the family IS and what produced it — the detail
    /// pane's orientation text, beside the kind so the view stays thin.
    public var detail: String {
        switch self {
        case .steeringVector:
            "a saved steering direction and its sidecar — extracted from a "
                + "concept's stimuli by one recipe, against one model at one "
                + "revision (runs/<run>/<name>.safetensors + .json)"
        case .readingProbe:
            "a fitted linear reading probe: a direction plus its calibration, "
                + "trained on a concept's labelled probe items "
                + "(runs/<run>/<name>.probe.json)"
        case .adapter:
            "a trained LoRA adapter and its training provenance "
                + "(runs/fine-tunes/<run>/fine-tune.json)"
        case .neutralPCBasis:
            "principal components of a neutral corpus's residual stream — the "
                + "projection basis a condition can pin; its FILE hash is the "
                + "pinned one (runs/neutral-pcs/<run>/neutral-pcs.json)"
        case .agent:
            "a model variant: a base model plus its adapters and injections, "
                + "with a promotion birth certificate when a sweep minted it. "
                + "Agents are managed in the Agents section — this row is a "
                + "cross-reference (runs/model-variants/<run>/model-variant.json)"
        }
    }

    var sortIndex: Int {
        switch self {
        case .steeringVector: 0
        case .readingProbe: 1
        case .neutralPCBasis: 2
        case .adapter: 3
        case .agent: 4
        }
    }
}

// MARK: - Routing

/// The ONE tool a derived row may open. Deliberately at most one per kind:
/// the inventory routes, it does not become a second workbench.
public enum DerivedArtifactRoute: String, Sendable, Equatable, Codable {
    /// Analysis (formerly Geometry) — cosines, RSA, Gemma Scope cross-checks.
    case analysis
    /// Data ▸ Adapter Training.
    case adapterTraining
    /// Data ▸ Concepts & Vectors.
    case conceptsAndVectors
    /// The Agents section.
    case agents

    public var label: String {
        switch self {
        case .analysis: "Open in Analysis"
        case .adapterTraining: "Open in Adapter Training"
        case .conceptsAndVectors: "Open in Concepts & Vectors"
        case .agents: "Open in Agents"
        }
    }

    /// What pressing it actually does — including what it does NOT do.
    ///
    /// Phase 4 upgraded two of these from honest to PRECISE. Analysis and
    /// Agents kept their selections in view-local `@State`, so the route
    /// could only switch sections and say "select it there"; both selections
    /// now live on their panel models (`GeometryPanel.select(vectorIDs:)`,
    /// `FineTuningPanel.selectAgent(id:)`) and the route preselects. The
    /// Analysis sentence still names its one remaining condition, because it
    /// is real: that pane's local list is filtered to the LOADED model.
    public var help: String {
        switch self {
        case .analysis:
            "switches to Analysis with this vector selected. Its local list "
                + "is filtered to the loaded model, so a vector extracted "
                + "against another model stays selected but unlisted until "
                + "that model is loaded — the pane says so."
        case .adapterTraining:
            "switches to Data ▸ Adapter Training, where this adapter is "
                + "listed with its training provenance"
        case .conceptsAndVectors:
            "switches to Data ▸ Concepts & Vectors, where this artifact's "
                + "source data and rebuild controls live"
        case .agents:
            "switches to the Agents section's Library with this agent "
                + "selected — agents are created, checked, and compared "
                + "there, not here"
        }
    }

    /// True where the route carries a selection the destination can apply
    /// (`DerivedArtifactEntry.selectionKey`). The two that cannot are not
    /// oversights: Adapter Training lists every adapter in one table with no
    /// selection to set, and Concepts & Vectors is reached by CONCEPT, which
    /// those rows carry separately.
    public var preselects: Bool {
        switch self {
        case .analysis, .agents: true
        case .adapterTraining, .conceptsAndVectors: false
        }
    }
}

// MARK: - Fact

/// One label/value line in the derived detail pane. A value is only ever a
/// field the artifact's own sidecar carries.
public struct DerivedArtifactFact: Identifiable, Sendable, Equatable, Hashable {
    public let label: String
    public let value: String

    public init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }

    public var id: String { label }
}

extension [DerivedArtifactFact] {
    /// Absent, never faked: a sidecar field that is nil or blank produces no
    /// row at all rather than an em-dash pretending to be information.
    mutating func appendIfPresent(_ label: String, _ value: String?) {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        append(.init(label, value))
    }
}

// MARK: - Entry

/// One derived row. Value type, `Sendable`, built entirely off the main actor
/// — the same contract as `DatasetInventoryEntry`, with the columns a derived
/// artifact actually has.
public struct DerivedArtifactEntry: Identifiable, Sendable, Equatable, Hashable {
    public let kind: DerivedArtifactKind
    public let name: String
    /// The artifact itself (the safetensors for a vector, the metadata JSON
    /// everywhere else) — what "Copy Path" copies.
    public let primaryURL: URL
    /// The JSON that carries the provenance. Equal to `primaryURL` for the
    /// families whose artifact IS its JSON.
    public let sidecarURL: URL
    /// The run directory that owns it — what "Reveal in Finder" opens.
    public let directory: URL
    /// The model the artifact was derived against, when its sidecar names one.
    public let modelID: String?
    /// The Method/Detail column: the one line that distinguishes two
    /// artifacts of the same kind for the same concept.
    public let detail: String
    /// The creation stamp AS WRITTEN by the producing store. Kept verbatim
    /// (never re-formatted into something the artifact does not say) and
    /// parsed separately for sorting.
    public let createdStamp: String
    /// The source concept, where the family has one.
    public let concept: String?
    /// The detail pane's provenance rows, in reading order.
    public let facts: [DerivedArtifactFact]
    public let route: DerivedArtifactRoute?
    /// Workspace-relative path of `primaryURL`, computed at construction.
    public let displayPath: String
    /// The id the DESTINATION tool selects this artifact by — nil where the
    /// route carries no selection (`DerivedArtifactRoute.preselects`).
    ///
    /// Each is the destination's OWN identity formula, not a new one:
    /// `VectorArtifact.id` is `<run dir>/<base name>` with no extension (so
    /// the `.safetensors` is dropped), and `ModelVariantRecord.id` is the
    /// artifact JSON's path. Computed here, once, off the main actor.
    public let selectionKey: String?
    /// The parsed creation instant, or nil where the stamp is not an ISO-8601
    /// one (legacy artifacts wrote free-form stamps). Parsed ONCE here: a
    /// table's sort comparator runs per comparison, and date parsing has no
    /// business happening there.
    public let created: Date?
    public let id: String

    public init(
        kind: DerivedArtifactKind,
        name: String,
        primaryURL: URL,
        sidecarURL: URL,
        directory: URL,
        root: URL,
        modelID: String?,
        detail: String,
        createdStamp: String,
        concept: String?,
        facts: [DerivedArtifactFact],
        route: DerivedArtifactRoute?
    ) {
        self.kind = kind
        self.name = name
        self.primaryURL = primaryURL
        self.sidecarURL = sidecarURL
        self.directory = directory
        self.modelID = modelID
        self.detail = detail
        self.createdStamp = createdStamp
        self.concept = concept
        self.facts = facts
        self.route = route
        self.displayPath = DatasetCreationPlanner.relativePath(of: primaryURL, root: root)
        self.created = Self.parse(createdStamp)
        self.id = Self.id(kind: kind, url: sidecarURL)
        self.selectionKey = Self.selectionKey(kind: kind, primaryURL: primaryURL)
    }

    /// The destination tool's own identity for this artifact — see
    /// `selectionKey`. A kind whose route does not preselect has none.
    static func selectionKey(kind: DerivedArtifactKind, primaryURL: URL) -> String? {
        switch kind {
        case .steeringVector:
            // `VectorArtifact.id` = directory + base name, extension dropped.
            return primaryURL.deletingPathExtension().path
        case .agent:
            // `ModelVariantRecord.id` = the artifact JSON's path.
            return primaryURL.path
        case .readingProbe, .adapter, .neutralPCBasis:
            return nil
        }
    }

    /// Stable across scans of the same workspace, and distinct between two
    /// kinds that could name one file. Same resolve-then-path formula as
    /// `DatasetInventoryEntry.id`, for the same reason (`/private/var` vs
    /// `/var`).
    public static func id(kind: DerivedArtifactKind, url: URL) -> String {
        "\(kind.rawValue):\(url.standardizedFileURL.resolvingSymlinksInPath().path)"
    }

    public var modelText: String { modelID ?? "—" }

    /// Sort key: `.distantPast` for an unparseable stamp, so legacy artifacts
    /// sort last under "newest first" instead of disappearing.
    public var sortableCreated: Date { created ?? .distantPast }

    /// Formatted where the stamp is a real instant; otherwise the stamp
    /// VERBATIM — an artifact is never shown a date it does not claim.
    public var createdText: String {
        guard let created else { return createdStamp.isEmpty ? "—" : createdStamp }
        return DateFormatter.localizedString(
            from: created, dateStyle: .short, timeStyle: .short)
    }

    /// Both ISO-8601 spellings the producing stores emit: `makeUniqueRunDirectory`
    /// stamps fractional seconds, the artifact writers mostly do not.
    /// Formatters are built per call rather than cached statically —
    /// `ISO8601DateFormatter` is not `Sendable`, and this runs once per row
    /// at construction, off the main actor.
    static func parse(_ stamp: String) -> Date? {
        guard !stamp.isEmpty else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: stamp) { return date }
        return ISO8601DateFormatter().date(from: stamp)
    }
}
