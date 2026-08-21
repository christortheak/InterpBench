import Foundation

// Chain-runner (pipeline) state, Mac side — stage 5 of the seamless
// pipeline (2026-07-18). The SERVER runs chains and owns the ledger; this
// engine renders their awaiting/aborted/completed states and the abort
// record's gate details. Cross-engine JSON keys match
// `tasks.list_pipeline_runs` exactly.

extension ClusterClient {

    /// One pipeline run's summary (`GET /api/experiment/{name}/pipelines`).
    public struct PipelineRunSummary: Codable, Sendable, Identifiable {
        /// One stage's ledger entry: `status` is "completed" | "started" |
        /// "pending" (never ran).
        public struct StageStatus: Codable, Sendable, Equatable {
            public var stage: String
            public var status: String
            public var runID: String?
        }

        /// One gate check from the abort record — `pipeline_spec.GateResult`
        /// verbatim (detail is researcher-facing prose; measured/threshold
        /// are present for numeric gates).
        public struct GateOutcome: Codable, Sendable, Equatable {
            public var passed: Bool?
            public var stage: String?
            public var gate: String?
            public var detail: String?
            public var measured: Double?
            public var threshold: Double?
        }

        public struct Abort: Codable, Sendable, Equatable {
            public var stage: String?
            public var gates: [GateOutcome]?
            public var evidenceRunID: String?
        }

        public struct WinningCell: Codable, Sendable, Equatable {
            public var layer: Int?
            public var alpha: Double?
        }

        /// A promoted-agent pin from the ledger (schema-2 chains; legacy
        /// schema-1 ledgers carry only the artifact name).
        public struct PromotedAgent: Codable, Sendable, Equatable {
            public var artifact: String?
            public var hash: String?
            public var sweepRun: String?
            public var winningCell: WinningCell?
        }

        /// The startup reconciler's orphan stamp (2026-08-06): the daemon
        /// found completed stages, a null disposition, and no live job, and
        /// could not resume the remainder in-process. `reason` is the
        /// researcher-facing text; the affordances are Resubmit (continue)
        /// and Import evidence (bring the finished stages home).
        public struct Parked: Codable, Sendable, Equatable {
            public var at: String?
            public var by: String?
            public var reason: String?
            public var completedStages: [String]?
            public var remainingStages: [String]?
        }

        /// One loud-and-stamped record of a resume that proceeded past
        /// manifest drift (`epochDriftAtContinuation`, 2026-08-05 policy:
        /// a submitted chain never dies on pinning drift — the drift is
        /// stamped and each remaining stage's own epoch guard decides what
        /// it may measure).
        public struct EpochDrift: Codable, Sendable, Equatable {
            public var ledgerHash: String?
            public var liveHash: String?
        }

        public var run: String
        public var schema: Int?
        /// The owning experiment — stamped by the cross-experiment listing
        /// (`GET /api/pipelines`); nil from older servers' per-experiment
        /// route, where the caller already knows the experiment.
        public var experiment: String?
        /// "completed" | "aborted" | nil (no terminal disposition).
        public var disposition: String?
        public var parked: Parked?
        public var epochDriftAtContinuation: [EpochDrift]?
        /// The manifest status when the chain STARTED ("draft" | "frozen")
        /// — a draft chain is legal exploratory work, and the label says so
        /// durably (seventh round).
        public var manifestStatus: String?
        public var experimentHash: String?
        /// Last ledger write (UTC ISO) — what lets a reader judge whether
        /// a non-terminal chain is plausibly still running or abandoned.
        public var updatedAt: String?
        public var stages: [StageStatus]
        public var abort: Abort?
        public var promotedAgents: [String: PromotedAgent]?

        public var id: String { run }

        /// The researcher-facing state label. An abort is a RECORDED
        /// scientific determination, not a failure — and a chain with no
        /// terminal disposition is honestly "unfinished": this listing
        /// cannot know whether a job still runs it (crashed/abandoned
        /// chains would otherwise read "in flight" forever); the
        /// updatedAt timestamp is the evidence to judge by.
        public var stateLabel: String {
            switch disposition {
            case "completed": return "completed"
            case "aborted": return "aborted (gate)"
            default: return parked == nil ? "unfinished" : "parked"
            }
        }

        /// Whether the chain has finished stage runs worth bringing home —
        /// the awaiting-import affordance's qualifying condition.
        public var hasCompletedStages: Bool {
            stages.contains { $0.status == "completed" }
        }

        /// Compact per-stage glyph line, e.g. "extract ✓ · validate ✓ ·
        /// sweep ✕ · promote – · run –" — pure, testable.
        public var stageSummaryLine: String {
            stages.map { entry in
                let glyph: String
                switch entry.status {
                case "completed": glyph = "✓"
                case "started": glyph = "▸"
                default:
                    glyph = (abort?.stage == entry.stage) ? "✕" : "–"
                }
                return "\(entry.stage) \(glyph)"
            }.joined(separator: " · ")
        }
    }

    /// Pipeline runs for an experiment, newest first. Older servers without
    /// the route surface as an error the caller treats as "none".
    public func pipelineRuns(
        experiment: String
    ) async throws -> [PipelineRunSummary] {
        struct Response: Decodable { var pipelines: [PipelineRunSummary] }
        let response: Response =
            try await get("/api/experiment/\(experiment)/pipelines")
        return response.pipelines
    }

    /// EVERY experiment's pipeline runs (`GET /api/pipelines`, 2026-08-06)
    /// — the Compute panel's orphan/awaiting-import surface. Rows carry
    /// `experiment` and, for chains the startup reconcile could not resume,
    /// the `parked` stamp. Older servers 404: callers treat it as "none".
    public func allPipelineRuns() async throws -> [PipelineRunSummary] {
        struct Response: Decodable { var pipelines: [PipelineRunSummary] }
        let response: Response = try await get("/api/pipelines")
        return response.pipelines
    }

    /// The server's receipt for an on-demand evidence packaging
    /// (`POST /api/bundles/evidence` — the route existed since the partial
    /// retention work; this is its first Swift caller, 2026-08-06).
    public struct EvidencePackageReceipt: Codable, Sendable {
        public var bundlePath: String?
        public var bundleSha256: String?
        public var runID: String?
        public var evidenceComplete: Bool?
        public var missingEvidence: [String]?
        /// Structured skip (2026-08-11): instead of packaging, the server
        /// answered "this run directory is a failure record with nothing
        /// to bundle" (a refused continuation's ledger-only pipeline dir).
        /// The importer notes it and moves on — a skip is never an error,
        /// and never an import.
        public var skipped: Bool? = nil
        public var reason: String? = nil
    }

    /// Package a server-resident run directory (a pipeline chain root
    /// included — the server walks the ledger to every completed stage run)
    /// as a hash-pinned evidence bundle, returning where it landed. The
    /// dead-pipeline import affordance packages first, then downloads and
    /// imports through the same verified path auto-import uses.
    public func packageEvidence(
        runDirectory: String
    ) async throws -> EvidencePackageReceipt {
        struct Body: Encodable { var runDirectory: String }
        return try await post("/api/bundles/evidence",
                              body: Body(runDirectory: runDirectory))
    }
}

/// Which server pipeline rows the Compute panel surfaces as "completed
/// stages awaiting import" (2026-08-06, a replication-run incident): chains with
/// finished stage runs whose evidence is not in this workspace yet, and
/// whose state is settled enough to act on — parked (the startup
/// reconciler's orphan stamp) or terminal (completed/aborted). An unparked
/// null-disposition chain is plausibly still running: the server's
/// reconcile parks dead ones, so this listing never has to guess. Pure and
/// unit-tested; the view only lays it out.
public enum PipelineImportTriage {
    public static func awaitingImport(
        _ rows: [ClusterClient.PipelineRunSummary],
        importedRunIDs: Set<String>,
        localRunExists: (String) -> Bool
    ) -> [ClusterClient.PipelineRunSummary] {
        rows.filter { row in
            guard row.hasCompletedStages else { return false }
            guard row.parked != nil || row.disposition != nil else {
                return false
            }
            guard !importedRunIDs.contains(row.run) else { return false }
            return !localRunExists(row.run)
        }
    }
}

/// Draft model for the app's Pipeline Composer (stage 5, sixth round — the
/// app must be able to AUTHOR the chain it runs, not just submit one
/// declared by hand-edited JSON). A typed view over the passthrough
/// `pipeline` JSONValue block: parsing is LENIENT (a malformed block reads
/// as a fresh draft — verify() still flags the manifest itself), encoding
/// writes exactly the cross-engine schema `pipeline_spec.resolve_pipeline`
/// consumes.
public struct PipelineDraft: Equatable, Sendable {
    public static let allStages = [
        "extract", "validate", "sweep", "promote", "run", "evaluate",
        "analyze",
    ]
    public static let defaultStages = [
        "extract", "validate", "sweep", "promote", "run",
    ]

    /// Seed stages for a FRESH declaration, derived from the study type's
    /// relevant stages (engineer finding 2026-07-19: a compare-agents
    /// chain must not default to extract → … — the funnel stages belong to
    /// concept studies). The historical default intersected with what is
    /// relevant; when nothing of it survives, the chain is just `run`
    /// (evaluate/analyze stay opt-in — they need judges/statistics the
    /// researcher declares deliberately).
    public static func seedStages(relevant: [String]) -> [String] {
        let seeded = defaultStages.filter { relevant.contains($0) }
        return seeded.isEmpty ? ["run"] : seeded
    }

    /// Selected stages in canonical order.
    public var stages: [String]
    /// The LEGACY floor — reads exactly the transfer accuracy
    /// (`scenarioAccuracy`), as it always has. Mutually exclusive with the
    /// declared floor below (the resolver refuses both).
    public var minScenarioAccuracy: Double?
    public var maxCrossConceptCosine: Double?
    /// Declared accuracy floor: gate on a NAMED metric
    /// (`ExperimentStore.pipelineAccuracyFloorMetrics`). The metric is
    /// manifest data, hashed with the study — an unmeasurable declared
    /// metric fails the gate rather than falling back (review 2026-08-02).
    public var accuracyFloorMetric: String?
    public var accuracyFloorMinimum: Double?
    public var requireSelectionForEveryConcept: Bool

    public init(
        stages: [String] = PipelineDraft.defaultStages,
        minScenarioAccuracy: Double? = nil,
        maxCrossConceptCosine: Double? = nil,
        accuracyFloorMetric: String? = nil,
        accuracyFloorMinimum: Double? = nil,
        requireSelectionForEveryConcept: Bool = false
    ) {
        self.stages = stages
        self.minScenarioAccuracy = minScenarioAccuracy
        self.maxCrossConceptCosine = maxCrossConceptCosine
        self.accuracyFloorMetric = accuracyFloorMetric
        self.accuracyFloorMinimum = accuracyFloorMinimum
        self.requireSelectionForEveryConcept = requireSelectionForEveryConcept
    }

    public static func parse(_ block: JSONValue?) -> PipelineDraft? {
        guard case .object(let pipeline)? = block else { return nil }
        var draft = PipelineDraft()
        if case .array(let items)? = pipeline["stages"] {
            let names: [String] = items.compactMap {
                if case .string(let name) = $0 { name } else { nil }
            }
            if !names.isEmpty { draft.stages = names }
        }
        if case .object(let gates)? = pipeline["gates"] {
            if case .object(let validate)? = gates["validate"] {
                if case .number(let value)? = validate["minScenarioAccuracy"] {
                    draft.minScenarioAccuracy = value
                }
                if case .number(let value)? = validate["maxCrossConceptCosine"] {
                    draft.maxCrossConceptCosine = value
                }
                if case .object(let floor)? = validate["accuracyFloor"] {
                    if case .string(let metric)? = floor["metric"] {
                        draft.accuracyFloorMetric = metric
                    }
                    if case .number(let minimum)? = floor["minimum"] {
                        draft.accuracyFloorMinimum = minimum
                    }
                }
            }
            if case .object(let sweep)? = gates["sweep"],
                case .bool(let required)? = sweep["requireSelectionForEveryConcept"]
            {
                draft.requireSelectionForEveryConcept = required
            }
        }
        return draft
    }

    /// Toggle one stage, keeping canonical order (the resolver refuses any
    /// other order — the composer never produces one).
    public mutating func setStage(_ stage: String, enabled: Bool) {
        var selected = Set(stages)
        if enabled { selected.insert(stage) } else { selected.remove(stage) }
        stages = Self.allStages.filter { selected.contains($0) }
    }

    public func encoded() -> JSONValue {
        var block: [String: JSONValue] = [
            "stages": .array(stages.map { .string($0) })
        ]
        var gates: [String: JSONValue] = [:]
        var validate: [String: JSONValue] = [:]
        if let minScenarioAccuracy {
            validate["minScenarioAccuracy"] = .number(minScenarioAccuracy)
        }
        if let maxCrossConceptCosine {
            validate["maxCrossConceptCosine"] = .number(maxCrossConceptCosine)
        }
        if let accuracyFloorMetric, let accuracyFloorMinimum {
            validate["accuracyFloor"] = .object([
                "metric": .string(accuracyFloorMetric),
                "minimum": .number(accuracyFloorMinimum),
            ])
        }
        if !validate.isEmpty { gates["validate"] = .object(validate) }
        if requireSelectionForEveryConcept {
            gates["sweep"] = .object(
                ["requireSelectionForEveryConcept": .bool(true)])
        }
        if !gates.isEmpty { block["gates"] = .object(gates) }
        return .object(block)
    }

    /// True when no gate is declared — the composer surfaces the same loud
    /// caution the freeze advisory gives.
    public var isGateless: Bool {
        minScenarioAccuracy == nil && maxCrossConceptCosine == nil
            && accuracyFloorMetric == nil
            && !requireSelectionForEveryConcept
    }

    /// Draft-level completeness the ENCODING cannot express (review
    /// 2026-08-02, P1: a metric selected with no minimum ENCODED as no
    /// floor at all — the stripped block validated clean, `isGateless`
    /// read the local metric and went quiet, and Save succeeded, so a
    /// researcher could believe a preregistered AUC gate existed while
    /// the manifest carried none). Checked BEFORE encoding; the composer
    /// refuses Save while any of these stand.
    public var completenessViolations: [String] {
        var violations: [String] = []
        if let metric = accuracyFloorMetric, accuracyFloorMinimum == nil {
            violations.append(
                "the accuracy floor declares metric '\(metric)' with no "
                    + "minimum — saving now would write NO accuracy gate at "
                    + "all; enter a minimum (0–1) or switch back to the "
                    + "legacy transfer floor")
        }
        if accuracyFloorMinimum != nil, accuracyFloorMetric == nil {
            violations.append(
                "the accuracy floor has a minimum but no metric — declare "
                    + "which number gates")
        }
        return violations
    }
}

/// Local (imported or locally recorded) pipeline runs — the CONSUMER of the
/// portable ledger (seventh round: "retained" is not "consumed"). Scans the
/// workspace runs tree for pipeline directories belonging to an experiment,
/// PREFERRING `pipeline-portable.json` (imported evidence — every reference
/// already a run ID) and falling back to the raw `pipeline.json` (locally
/// executed chains — path references reduced to basenames for display).
/// Pure file reading; no server involved.
public enum LocalPipelineCatalog {

    public static func summaries(
        experiment: String
    ) -> [ClusterClient.PipelineRunSummary] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: ExperimentStore.runsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .compactMap { directory in
                summary(runDirectory: directory, experiment: experiment)
            }
    }

    static func summary(
        runDirectory: URL, experiment: String
    ) -> ClusterClient.PipelineRunSummary? {
        let runID = runDirectory.lastPathComponent
        let portableURL = runDirectory.appending(
            component: "pipeline-portable.json")
        if let data = try? Data(contentsOf: portableURL),
            let portable = try? JSONDecoder().decode(Portable.self, from: data),
            portable.experiment == experiment
        {
            return summary(runID: runID, portable: portable)
        }
        let ledgerURL = runDirectory.appending(component: "pipeline.json")
        guard let data = try? Data(contentsOf: ledgerURL),
            let ledger = try? JSONDecoder().decode(RawLedger.self, from: data),
            ledger.experiment == experiment
        else { return nil }
        return summary(runID: runID, ledger: ledger)
    }

    // MARK: portable projection (imported evidence)

    struct Portable: Decodable {
        struct Agent: Decodable {
            var runID: String?
            var artifact: String?
            var hash: String?
            var sweepRun: String?
            var winningCell: ClusterClient.PipelineRunSummary.WinningCell?
        }
        struct Abort: Decodable {
            var stage: String?
            var gates: [ClusterClient.PipelineRunSummary.GateOutcome]?
            var evidenceRunID: String?
        }
        var experiment: String?
        var ledgerSchema: Int?
        var disposition: String?
        var updatedAt: String?
        var manifestStatus: String?
        var stages: [String]?
        var stageStatus: [String: String]?
        var stageRuns: [String: String]?
        var promotedAgents: [String: Agent]?
        var abort: Abort?
        var parked: ClusterClient.PipelineRunSummary.Parked?
        var epochDriftAtContinuation:
            [ClusterClient.PipelineRunSummary.EpochDrift]?
    }

    private static func summary(
        runID: String, portable: Portable
    ) -> ClusterClient.PipelineRunSummary {
        let stages = (portable.stages ?? []).map { stage in
            ClusterClient.PipelineRunSummary.StageStatus(
                stage: stage,
                status: portable.stageStatus?[stage] ?? "pending",
                runID: portable.stageRuns?[stage])
        }
        let abort = portable.abort.map {
            ClusterClient.PipelineRunSummary.Abort(
                stage: $0.stage, gates: $0.gates,
                evidenceRunID: $0.evidenceRunID)
        }
        let agents = portable.promotedAgents?.mapValues {
            ClusterClient.PipelineRunSummary.PromotedAgent(
                artifact: $0.artifact, hash: $0.hash, sweepRun: $0.sweepRun,
                winningCell: $0.winningCell)
        }
        return ClusterClient.PipelineRunSummary(
            run: runID, schema: portable.ledgerSchema,
            experiment: portable.experiment,
            disposition: portable.disposition,
            parked: portable.parked,
            epochDriftAtContinuation: portable.epochDriftAtContinuation,
            manifestStatus: portable.manifestStatus,
            experimentHash: nil, updatedAt: portable.updatedAt,
            stages: stages, abort: abort, promotedAgents: agents)
    }

    // MARK: raw ledger fallback (locally executed chains)

    struct RawLedger: Decodable {
        struct StageEntry: Decodable {
            var status: String?
            var runDirectory: String?
        }
        struct Abort: Decodable {
            var stage: String?
            var gates: [ClusterClient.PipelineRunSummary.GateOutcome]?
            var evidenceRunDirectory: String?
        }
        var schema: Int?
        var experiment: String?
        var disposition: String?
        var updatedAt: String?
        var manifestStatus: String?
        var stages: [String]?
        var stageResults: [String: StageEntry]?
        var abort: Abort?
        var parked: ClusterClient.PipelineRunSummary.Parked?
        var epochDriftAtContinuation:
            [ClusterClient.PipelineRunSummary.EpochDrift]?
    }

    private static func summary(
        runID: String, ledger: RawLedger
    ) -> ClusterClient.PipelineRunSummary {
        let stages = (ledger.stages ?? []).map { stage in
            let entry = ledger.stageResults?[stage]
            return ClusterClient.PipelineRunSummary.StageStatus(
                stage: stage,
                status: entry?.status ?? "pending",
                runID: entry?.runDirectory.map {
                    URL(filePath: $0).lastPathComponent
                })
        }
        let abort = ledger.abort.map {
            ClusterClient.PipelineRunSummary.Abort(
                stage: $0.stage, gates: $0.gates,
                evidenceRunID: $0.evidenceRunDirectory.map {
                    URL(filePath: $0).lastPathComponent
                })
        }
        return ClusterClient.PipelineRunSummary(
            run: runID, schema: ledger.schema,
            experiment: ledger.experiment,
            disposition: ledger.disposition,
            parked: ledger.parked,
            epochDriftAtContinuation: ledger.epochDriftAtContinuation,
            manifestStatus: ledger.manifestStatus,
            experimentHash: nil, updatedAt: ledger.updatedAt,
            stages: stages, abort: abort, promotedAgents: nil)
    }
}
