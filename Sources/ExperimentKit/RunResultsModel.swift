import Foundation

/// The loaded, derived view-state of one run directory for the Results
/// detail pane: records (including ChoiceRecords — A6), report, choice
/// matrix, arm aggregation, agreement, parse failures, completion,
/// classification (P4), analysis artifacts (A3), and the validation report
/// (A13). Loading is bounded: the generations read is capped (complete
/// lines only) and every artifact is optional — a run without one simply
/// has nil there.
extension RunResults {

    /// Byte cap for the generations.jsonl read. Categorical studies are far
    /// smaller than this; a multi-hundred-MB long-text run decodes its head
    /// and says so (`generationsTruncated`).
    public static let generationsByteLimit = 33_554_432
    /// Row cap for the summaries.csv semantic table.
    public static let summariesRowLimit = 300

    public struct SummariesTable: Sendable, Equatable {
        public var header: [String]
        public var rows: [[String]]
        public var truncated: Bool
    }

    public struct Model: Sendable, Equatable {
        public var runDirectory: URL

        // Raw layers.
        public var records: [Record] = []
        public var generationsTruncated = false
        public var skippedRecordLines = 0
        public var report: Report?
        public var summaries: SummariesTable?
        public var validationReport: ValidationReport?
        public var effectSizes: [EffectSizeRow]?
        public var alienResiduals: [AlienResidualRow]?
        public var promotedMovers: PromotedMovers?
        public var cosineMatrix: CosineMatrix?
        /// Multi-agent panel-effect decomposition (A14) — present only when
        /// the run wrote panel-effects.csv (the server's multi-agent path).
        public var panelEffects: [PanelEffectRow]?
        /// Panel transcripts rebuilt from the per-turn records (C2). Empty on
        /// every non-panel run — built from generations.jsonl, never from
        /// transcript.md, so the measurement artifact stays the one source of
        /// truth.
        public var panelTranscripts: [PanelTranscript.Transcript] = []

        // Derived layers (P3).
        /// True when the run carries a manifest snapshot (`experiment.json`)
        /// — the classification header is only meaningful then (a toy run or
        /// library directory has no evidence status to misread).
        public var hasManifestSnapshot = false
        /// Per-condition intervention summary derived from the manifest
        /// snapshot ("fear L14 α0.8", "matched-norm random control", variant
        /// artifact names) — P3's "conditions and intervention summaries".
        public var conditionInterventions: [String: String] = [:]
        public var choiceMatrix = ChoiceMatrix(conditions: [], items: [])
        public var armSummaries: [ArmSummary] = []
        public var parseFailures: [ParseFailure] = []
        public var agreements: [Agreement] = []
        public var completion = Completion(
            completed: 0, planned: nil, perCondition: [])
        public var categoricalConditions: Set<String> = []
        public var classification = Classification(
            runClass: .draftPilot, epoch: .unknown,
            unpinnedResolvedRevision: nil)

        /// The run carries any option-bearing item — in its decoded records
        /// OR stamped in its report's choice metrics — drives the
        /// categorical study view and the metric N/A rule.
        public var isCategorical: Bool { !categoricalConditions.isEmpty }
        public var hasGenerations: Bool { !records.isEmpty }
        public var hasAnalysisArtifacts: Bool {
            effectSizes != nil || alienResiduals != nil || promotedMovers != nil
        }

        public init(runDirectory: URL) {
            self.runDirectory = runDirectory
        }
    }

    // MARK: - One assembly pipeline (F9)

    /// Bounded raw artifact bytes of ONE run, wherever they came from
    /// (local disk heads or a remote server's bounded reads). Local `load`
    /// and the remote loader both fill this and go through the SAME
    /// `assemble` — the derivation tail (records → matrix → agreements →
    /// completion → effect-size lift → classification) cannot fork again.
    public struct ArtifactBytes: Sendable, Equatable {
        public var generationsText: String?
        public var generationsTruncated = false
        public var reportData: Data?
        public var summariesText: String?
        public var summariesTruncated = false
        public var validationReportData: Data?
        public var effectSizesText: String?
        public var alienResidualsText: String?
        public var promotedMoversData: Data?
        public var cosineMatrixText: String?
        public var panelEffectsText: String?
        public var snapshotData: Data?

        public init() {}

        public var isEmpty: Bool {
            generationsText == nil && reportData == nil && snapshotData == nil
                && summariesText == nil && validationReportData == nil
                && effectSizesText == nil && alienResidualsText == nil
                && promotedMoversData == nil && cosineMatrixText == nil
                && panelEffectsText == nil
        }

        /// The canonical run-file names the semantic model reads — the
        /// remote loader fetches exactly these (head-bounded) and routes
        /// the bytes here by name.
        public static let fileNames: Set<String> = [
            "generations.jsonl", "report.json", "experiment.json",
            "summaries.csv", "validation-report.json", "effect-sizes.csv",
            "alien-residuals.csv", "promoted-movers.json",
            "cosine-matrix.csv", "panel-effects.csv",
        ]

        /// Route one fetched file's bytes to its slot. `truncated` matters
        /// only for the line-oriented artifacts (a truncated JSON simply
        /// fails its decode, which is the honest outcome).
        public mutating func assign(name: String, data: Data, truncated: Bool = false) {
            switch name {
            case "generations.jsonl":
                generationsText = String(decoding: data, as: UTF8.self)
                generationsTruncated = truncated
            case "report.json":
                reportData = data
            case "experiment.json":
                snapshotData = data
            case "summaries.csv":
                summariesText = String(decoding: data, as: UTF8.self)
                summariesTruncated = truncated
            case "validation-report.json":
                validationReportData = data
            case "effect-sizes.csv":
                effectSizesText = String(decoding: data, as: UTF8.self)
            case "alien-residuals.csv":
                alienResidualsText = String(decoding: data, as: UTF8.self)
            case "promoted-movers.json":
                promotedMoversData = data
            case "cosine-matrix.csv":
                cosineMatrixText = String(decoding: data, as: UTF8.self)
            case "panel-effects.csv":
                panelEffectsText = String(decoding: data, as: UTF8.self)
            default:
                break
            }
        }
    }

    /// The single parse+derive pipeline behind `load` and `remoteModel`.
    /// Only the classification CONTEXT differs by surface (local runs can
    /// compare against the live workspace manifest; remote browsing cannot)
    /// — everything else is byte-identical by construction.
    static func assemble(
        runDirectory: URL,
        artifacts: ArtifactBytes,
        classify: (ManifestSnapshotReading?, [Record]) -> Classification
    ) -> Model {
        var model = Model(runDirectory: runDirectory)

        if let text = artifacts.generationsText {
            let parsed = records(fromJSONL: text)
            model.records = parsed.records
            model.skippedRecordLines = parsed.skippedLines
            model.generationsTruncated = artifacts.generationsTruncated
        }

        if let data = artifacts.reportData {
            model.report = report(fromJSON: data)
        }

        if let text = artifacts.summariesText, let table = csv(text) {
            let capped = Array(table.rows.prefix(summariesRowLimit))
            model.summaries = SummariesTable(
                header: table.header,
                rows: capped,
                truncated: artifacts.summariesTruncated
                    || table.rows.count > capped.count)
        }

        // A validate run's structured report (canonical filename first, then
        // the legacy byte-identical report.json copy).
        for data in [artifacts.validationReportData, artifacts.reportData] {
            guard let data, let parsed = validationReport(fromJSON: data)
            else { continue }
            model.validationReport = parsed
            break
        }

        if let text = artifacts.effectSizesText {
            model.effectSizes = effectSizes(fromCSV: text)
        }
        // Swift study runs also stamp effect sizes inline in report.json.
        if model.effectSizes == nil, let inline = model.report?.effectSizes,
            !inline.isEmpty
        {
            model.effectSizes = inline
        }

        if let text = artifacts.alienResidualsText {
            model.alienResiduals = alienResiduals(fromCSV: text)
        }
        if let data = artifacts.promotedMoversData {
            model.promotedMovers = promotedMovers(fromJSON: data)
        }
        if let text = artifacts.cosineMatrixText {
            model.cosineMatrix = cosineMatrix(fromCSV: text)
        }
        // A14: the multi-agent panel-effect decomposition — absent-file
        // tolerant like every other artifact (a run without it has nil).
        if let text = artifacts.panelEffectsText {
            model.panelEffects = panelEffects(fromCSV: text)
        }
        model.panelTranscripts = PanelTranscript.transcripts(from: model.records)

        model.choiceMatrix = choiceMatrix(records: model.records)
        model.armSummaries = armAggregation(matrix: model.choiceMatrix)
        model.parseFailures = parseFailures(records: model.records)
        model.agreements = baselineAgreement(
            matrix: model.choiceMatrix, report: model.report)
        model.completion = completion(records: model.records, report: model.report)
        // Records AND report evidence: stamped choice metrics keep the
        // categorical presentation alive when generation rows are missing
        // or truncated away.
        model.categoricalConditions = categoricalConditions(
            records: model.records, report: model.report)

        // F8: snapshot BYTES present drives the header; the strict decode
        // only gates the intervention summaries. Classification receives the
        // tolerant reading so an undecodable snapshot degrades loudly.
        let reading = artifacts.snapshotData.map(manifestSnapshotReading(from:))
        model.hasManifestSnapshot = artifacts.snapshotData != nil
        if let manifest = reading?.strict {
            model.conditionInterventions = interventionSummaries(manifest: manifest)
        }
        model.classification = classify(reading, model.records)
        return model
    }

    /// Read + derive everything for one LOCAL run directory. Pure reads —
    /// runs are immutable and this never writes a byte.
    public static func load(runDirectory: URL) -> Model {
        var artifacts = ArtifactBytes()

        if let head = RunBrowser.readHead(
            of: runDirectory.appending(component: "generations.jsonl"),
            maxBytes: generationsByteLimit)
        {
            artifacts.generationsText = head.text
            artifacts.generationsTruncated = head.truncated
        }
        artifacts.reportData = try? Data(
            contentsOf: runDirectory.appending(component: "report.json"))
        if let head = RunBrowser.readHead(
            of: runDirectory.appending(component: "summaries.csv"),
            maxBytes: RunBrowser.jsonPreviewByteLimit)
        {
            artifacts.summariesText = head.text
            artifacts.summariesTruncated = head.truncated
        }
        artifacts.validationReportData = try? Data(
            contentsOf: runDirectory.appending(component: "validation-report.json"))
        artifacts.effectSizesText = RunBrowser.readHead(
            of: runDirectory.appending(component: "effect-sizes.csv"),
            maxBytes: RunBrowser.jsonPreviewByteLimit)?.text
        artifacts.alienResidualsText = RunBrowser.readHead(
            of: runDirectory.appending(component: "alien-residuals.csv"),
            maxBytes: RunBrowser.jsonPreviewByteLimit)?.text
        artifacts.promotedMoversData = try? Data(
            contentsOf: runDirectory.appending(component: "promoted-movers.json"))
        artifacts.cosineMatrixText = RunBrowser.readHead(
            of: runDirectory.appending(component: "cosine-matrix.csv"),
            maxBytes: RunBrowser.jsonPreviewByteLimit)?.text
        artifacts.panelEffectsText = RunBrowser.readHead(
            of: runDirectory.appending(component: "panel-effects.csv"),
            maxBytes: RunBrowser.jsonPreviewByteLimit)?.text
        artifacts.snapshotData = try? Data(
            contentsOf: runDirectory.appending(component: "experiment.json"))

        return assemble(runDirectory: runDirectory, artifacts: artifacts) {
            reading, records in
            classification(
                reading: reading, runDirectory: runDirectory, records: records)
        }
    }

    /// Build the semantic Results model from bounded bytes fetched from a
    /// remote server. Remote browsing cannot compare the snapshot with a
    /// local live manifest, so frozen runs deliberately classify as epoch
    /// unverified until their evidence bundle is imported.
    public static func remoteModel(runID: String, artifacts: ArtifactBytes) -> Model {
        assemble(
            runDirectory: URL(filePath: "/remote-runs").appending(component: runID),
            artifacts: artifacts
        ) { reading, records in
            var result = classify(
                manifestStatus: reading?.statusRaw,
                freezeForced: reading?.freezeForced,
                hasMatchingValidateEvidence: nil,
                stampedHashMatchesLive: nil,
                manifestRevision: reading?.modelRevision,
                recordRevisions: Set(records.compactMap(\.modelRevision)))
            result.snapshotUnreadable = reading?.isUnreadable ?? false
            return result
        }
    }

    /// Convenience for the common three-artifact case (kept for tests and
    /// callers that predate `ArtifactBytes`).
    public static func remoteModel(
        runID: String,
        generationsText: String?,
        generationsTruncated: Bool,
        reportData: Data?,
        snapshotData: Data?
    ) -> Model {
        var artifacts = ArtifactBytes()
        artifacts.generationsText = generationsText
        artifacts.generationsTruncated = generationsTruncated
        artifacts.reportData = reportData
        artifacts.snapshotData = snapshotData
        return remoteModel(runID: runID, artifacts: artifacts)
    }

    /// One line per manifest condition saying what it injects: slot list in
    /// "concept L<layer> α<alpha>" form (α in residual-norm units by the
    /// manifest's own flag), the control type where declared, and variant
    /// conditions by their artifact. Baseline reads "no intervention".
    public static func interventionSummaries(
        manifest: ExperimentManifest
    ) -> [String: String] {
        var summaries: [String: String] = [
            baselineConditionName: "no intervention"
        ]
        for condition in manifest.conditions {
            var parts: [String] = []
            if condition.controlType == "randomMatchedNorm" {
                parts.append("matched-norm random control")
            } else if let controlType = condition.controlType {
                parts.append("control: \(controlType)")
            }
            let slots = condition.slots.map { slot in
                "\(slot.concept) L\(slot.layer) α\(formatAlpha(slot.alpha))"
            }
            if !slots.isEmpty {
                parts.append(slots.joined(separator: " + "))
            }
            summaries[condition.name] =
                parts.isEmpty ? "no slots" : parts.joined(separator: " · ")
        }
        for variant in manifest.variantConditions {
            summaries[variant.name] =
                "variant \((variant.artifactPath as NSString).lastPathComponent)"
        }
        return summaries
    }

    private static func formatAlpha(_ alpha: Double) -> String {
        alpha == alpha.rounded()
            ? String(Int(alpha))
            : String(format: "%g", alpha)
    }

    // MARK: - Analyze routing (A3)

    /// Where an in-app "Analyze…" for this run must execute. Per-engine
    /// epoch guards mean a run is analyzed on the engine that produced it
    /// (canonicalization differs across engines) — routing is by the run's
    /// substrate stamp, never by which browser it appears in.
    public enum AnalyzeRoute: Sendable, Equatable {
        /// `ExperimentTasks.analyze` in-process (pure CPU, no model load).
        case local(experiment: String)
        /// The server's analyze verb as a durable job
        /// (`POST /api/experiment/{name}/analyze`).
        case server(experiment: String)
        case unavailable(reason: String)
    }

    /// The Swift engine's substrate stamp (a python-hf run must round-trip
    /// to the server that produced it).
    public static let localSubstrate = "swift-mlx"

    public static func analyzeRoute(
        runType: String?,
        substrate: String?,
        experiment: String?,
        serverAvailable: Bool,
        browsingRemote: Bool = false
    ) -> AnalyzeRoute {
        guard runType == "run" else {
            return .unavailable(
                reason: "analyze consumes a completed study run (runType "
                    + "\"run\") — this run is "
                    + (runType.map { "a \($0) run" } ?? "unstamped"))
        }
        guard let experiment, !experiment.isEmpty else {
            return .unavailable(
                reason: "no experiment stamp on this run — analyze resolves "
                    + "the study from the run's config.json")
        }
        if browsingRemote {
            return serverAvailable
                ? .server(experiment: experiment)
                : .unavailable(
                    reason: "not connected to the server that owns this run — "
                        + "connect in Compute first")
        }
        // Local list (includes paired-server runs sharing this workspace).
        if let substrate, substrate != localSubstrate {
            return serverAvailable
                ? .server(experiment: experiment)
                : .unavailable(
                    reason: "this run was produced on \(substrate); the "
                        + "per-engine epoch guard means it must be analyzed "
                        + "there — connect the server in Compute")
        }
        return .local(experiment: experiment)
    }
}
