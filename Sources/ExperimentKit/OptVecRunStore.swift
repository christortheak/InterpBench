import Foundation
import SteeringKit

/// Read-only access to OptVec RUN artifacts in the workspace's `runs/` tree —
/// the results half of the app's OptVec surface (`OptVecBundleStore` holds
/// the dataset half; see its header for the v1/v2/v3 version contract, which
/// this store shares).
///
/// Everything here decodes files the PYTHON engine wrote (`optvec_train.py`,
/// `optvec_eval.py`, `optvec_interpret.py`, `optvec_jspace.py`,
/// `optvec_geometry.py`, `optvec_campaign.py`) into exactly the fields the
/// views render. Two rules are binding and carried as data, never dropped:
///
/// - **Claim grammar.** Every OptVec artifact certifies SUFFICIENCY (jspace:
///   exploratory). The stamps (`claim`, `firewall`, `qualification`,
///   `evidenceTier`) are surfaced verbatim; `eval.json` is screen-grade
///   evidence, never citable — its own `firewall` string says so.
/// - **No energy without its null.** jspace energies are only rendered
///   beside their `<key>Null` / `<key>NullRatio` companions (the writer
///   enforces the pairing; a reader that showed one alone would break the
///   schema's own rule). Same for library cosines and `nullPercentile`.
///
/// OptVec campaign cells NEVER enter the server's durable-jobs ledger — they
/// are raw `sbatch` submissions tracked by `campaign-state.json` + the
/// per-cell `COMPLETED` marker. So "progress" here is derived OFFLINE from
/// those files (the workspace is the source of truth); live scheduler state
/// (queued vs running) needs `squeue` on the cluster and is honestly
/// reported as unknown when this Mac cannot see it.
public enum OptVecRunStore {

    // MARK: - Run discovery

    /// The OptVec run types, exactly as stamped in each run's `config.json`
    /// (`runType` is the authority; the directory slug is only a hint).
    public enum Kind: String, Codable, Sendable, CaseIterable {
        case train = "optvec-train"
        case eval = "optvec-eval"
        case interpret = "optvec-interpret"
        case family = "optvec-family"
        case jspace = "optvec-jspace"
        case geometry = "optvec-geometry"
        case campaign = "optvec-campaign"

        public var label: String {
            switch self {
            case .train: "Training"
            case .eval: "Eval (test split)"
            case .interpret: "Interpretation"
            case .family: "Family summary"
            case .jspace: "J-space (exploratory)"
            case .geometry: "Geometry"
            case .campaign: "Slurm campaign"
            }
        }
    }

    public struct RunItem: Sendable, Equatable, Identifiable {
        public var name: String
        public var url: URL
        public var kind: Kind
        public var createdAt: String?
        public var modelID: String?
        /// `config.json` `notes.stage` — the crash indicator ("complete" is
        /// the only terminal success for train/eval runs).
        public var stage: String?

        public var id: String { name }
    }

    /// Every run directory whose `config.json` stamps an `optvec-*` runType,
    /// newest first (stamp-prefixed names sort chronologically). A missing
    /// `runs/` is an empty list, not an error.
    public static func list() -> [RunItem] {
        let fm = FileManager.default
        let runs = ExperimentStore.runsDirectory
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runs, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return entries
            .compactMap { url -> RunItem? in
                guard
                    let data = try? Data(
                        contentsOf: url.appending(component: "config.json")),
                    let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                    let runType = object["runType"] as? String,
                    let kind = Kind(rawValue: runType)
                else { return nil }
                let notes = object["notes"] as? [String: Any]
                return RunItem(
                    name: url.lastPathComponent, url: url, kind: kind,
                    createdAt: object["createdAt"] as? String,
                    modelID: object["modelID"] as? String,
                    stage: notes?["stage"] as? String)
            }
            .sorted { $0.name > $1.name }
    }

    // MARK: - Campaign (offline status derivation)

    public struct CampaignPlan: Codable, Sendable, Equatable {
        public struct Cell: Codable, Sendable, Equatable, Identifiable {
            public var cellID: String
            public var condition: String?
            public var layer: Int?
            public var seed: Int?

            public var id: String { cellID }
        }

        public struct Config: Codable, Sendable, Equatable {
            public struct Slurm: Codable, Sendable, Equatable {
                public var maxResubmits: Int?
            }

            public var name: String?
            public var slurm: Slurm?
        }

        public var schemaVersion: Int?
        public var name: String?
        public var config: Config?
        public var cells: [Cell]?
    }

    public struct CampaignState: Codable, Sendable, Equatable {
        public struct Attempt: Codable, Sendable, Equatable {
            public var jobID: String?
            public var outcome: String?  // "submitted" | "failed" | "adopted"
            public var exitCode: Int?
            public var detail: String?
        }

        public struct CellState: Codable, Sendable, Equatable {
            public var attempts: [Attempt]?
            public var lastJobID: String?
        }

        public var schemaVersion: Int?
        public var cells: [String: CellState]?
    }

    /// What this Mac can KNOW about a cell without the cluster's scheduler.
    /// `submitted` deliberately collapses the server's queued/running/
    /// unknown distinction — those need `squeue`, and inventing them here
    /// would present a guess as a fact.
    public enum OfflineCellStatus: String, Sendable {
        case planned
        case submitted
        case completed
        case failed
        case exhausted

        public var label: String {
            switch self {
            case .planned: "planned"
            case .submitted: "submitted (scheduler state unknown here)"
            case .completed: "completed"
            case .failed: "failed"
            case .exhausted: "exhausted (attempt budget spent)"
            }
        }
    }

    public struct CampaignCellStatus: Sendable, Equatable, Identifiable {
        public var cell: CampaignPlan.Cell
        public var status: OfflineCellStatus
        public var attempts: Int
        public var attemptBudget: Int
        public var lastJobID: String?

        public var id: String { cell.cellID }
    }

    public struct CampaignStatus: Sendable, Equatable {
        public var name: String
        public var cells: [CampaignCellStatus]

        public var totals: [OfflineCellStatus: Int] {
            var out: [OfflineCellStatus: Int] = [:]
            for cell in cells { out[cell.status, default: 0] += 1 }
            return out
        }
    }

    /// Offline mirror of `optvec_campaign.status`: the `COMPLETED` marker is
    /// the completion AUTHORITY; everything else derives from
    /// `campaign-state.json` attempts. `attemptBudget = 1 + maxResubmits`
    /// (default 2), the server's own arithmetic.
    public static func campaignStatus(runURL: URL) -> CampaignStatus? {
        guard
            let planData = try? Data(
                contentsOf: runURL.appending(component: "campaign.json")),
            let plan = try? JSONDecoder().decode(
                CampaignPlan.self, from: planData)
        else { return nil }
        let state =
            (try? Data(
                contentsOf: runURL.appending(component: "campaign-state.json")))
            .flatMap { try? JSONDecoder().decode(CampaignState.self, from: $0) }
        let budget = 1 + (plan.config?.slurm?.maxResubmits ?? 2)
        let fm = FileManager.default
        let cells = (plan.cells ?? []).map { cell -> CampaignCellStatus in
            let cellState = state?.cells?[cell.cellID]
            let attempts = cellState?.attempts ?? []
            let marker = runURL.appending(
                components: "cells", cell.cellID, "COMPLETED")
            let status: OfflineCellStatus
            if fm.fileExists(atPath: marker.path) {
                status = .completed
            } else if attempts.isEmpty {
                status = .planned
            } else if attempts.last?.outcome == "failed" {
                status = attempts.count >= budget ? .exhausted : .failed
            } else {
                status = .submitted
            }
            return CampaignCellStatus(
                cell: cell, status: status, attempts: attempts.count,
                attemptBudget: budget,
                lastJobID: cellState?.lastJobID)
        }
        return CampaignStatus(
            name: plan.name ?? runURL.lastPathComponent, cells: cells)
    }

    // MARK: - Training progress

    /// One `metrics.jsonl` row. The `val*` keys are present only on
    /// validation steps — always optionals.
    public struct TrainMetricsLine: Codable, Sendable, Equatable {
        public var step: Int?
        public var loss: Double?
        public var lossShift: Double?
        public var lossAnchor: Double?
        public var lossCap: Double?
        public var valShiftRate: Double?
        public var valAnchorKL: Double?
        public var valComposite: Double?
    }

    public struct TrainProgress: Sendable, Equatable {
        /// `config.json` `notes.stage` — "complete" is the only terminal
        /// success; anything else on a dead run is where it crashed.
        public var stage: String?
        public var lastMetrics: TrainMetricsLine?
        /// Extension-less workspace-relative locator of the saved artifact
        /// (present once the run reached "saving"/"complete").
        public var artifactReference: String?
    }

    /// Progress facts for one `optvec-train` run, read boundedly (the
    /// metrics stream is tailed, never slurped — a long campaign cell's
    /// curve can be large).
    public static func trainProgress(runURL: URL) -> TrainProgress {
        var progress = TrainProgress()
        if let data = try? Data(
            contentsOf: runURL.appending(component: "config.json")),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        {
            progress.stage = (object["notes"] as? [String: Any])?["stage"]
                as? String
        }
        if let line = tailLine(
            of: runURL.appending(component: "metrics.jsonl"))
        {
            progress.lastMetrics = try? JSONDecoder().decode(
                TrainMetricsLine.self, from: Data(line.utf8))
        }
        // The artifact is the sidecar/tensor pair at the run root (never in
        // checkpoints/): any `<name>.json` whose extractionMethod is optvec.
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: runURL, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
        {
            for url in entries where url.pathExtension == "json" {
                guard url.lastPathComponent != "config.json",
                    url.lastPathComponent != "eval.json",
                    let data = try? Data(contentsOf: url),
                    let sidecar = try? JSONDecoder().decode(
                        SteeringVectorSidecar.self, from: data),
                    sidecar.extractionMethod == ExtractionMethod.optvec.rawValue
                else { continue }
                let base = url.deletingPathExtension()
                progress.artifactReference =
                    ExperimentStore.workspaceRelativePath(base.path)
                break
            }
        }
        return progress
    }

    /// Last non-empty line of a file, reading only the final 16 KiB.
    static func tailLine(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let window: UInt64 = 16 * 1024
        let offset = size > window ? size - window : 0
        try? handle.seek(toOffset: offset)
        guard let data = try? handle.readToEnd() else { return nil }
        return String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .last.map(String.init)
    }

    // MARK: - eval.json

    public struct ArtifactBlock: Codable, Sendable, Equatable {
        public var reference: String?
        public var name: String?
        public var extractionMethod: String?
        public var modelID: String?
        public var revision: String?
        public var tensorSHA256: String?
    }

    public struct LibraryEntry: Codable, Sendable, Equatable {
        public var reference: String?
        public var name: String?
        public var concept: String?
        public var extractionMethod: String?
        public var cosine: Double?
        public var absCosine: Double?
        /// The matched-null percentile that makes the cosine readable —
        /// never render `cosine` without it.
        public var nullPercentile: Double?
    }

    public struct LibraryBlock: Codable, Sendable, Equatable {
        public struct NullStats: Codable, Sendable, Equatable {
            public var samples: Int?
            public var absCosineP95: Double?
            public var absCosineP99: Double?
        }

        public var layer: Int?
        public var comparedCount: Int?
        public var topK: [LibraryEntry]?
        public var null: NullStats?
    }

    public struct EvalReport: Codable, Sendable, Equatable {
        public struct Role: Codable, Sendable, Equatable {
            public var itemCount: Int?
            public var flipRate: Double?
            public var meanKLFromBaseline: Double?
            /// Over FLIPPABLE items only (items whose unsteered argmax was
            /// already the target are excluded).
            public var shiftRate: Double?
            public var meanLogOddsMovement: Double?
            public var accuracy: Double?
            public var baselineAccuracy: Double?
            public var accuracyDelta: Double?
        }

        public struct Fluency: Codable, Sendable, Equatable {
            public var textCount: Int?
            public var meanTokenLogprob: Double?
            public var deltaFromBaseline: Double?
            public var note: String?
        }

        public struct Dose: Codable, Sendable, Equatable, Identifiable {
            public var alphaMultiple: Double?
            public var alphaAbsolute: Double?
            public var isBaseline: Bool?
            public var target: Role?
            public var anchor: Role?
            public var capability: Role?
            public var fluency: Fluency?

            public var id: Double { alphaMultiple ?? -1 }
        }

        public var schemaVersion: Int?
        public var runID: String?
        /// Binding claim grammar: always "sufficiency".
        public var claim: String?
        public var split: String?
        /// The engine's own screen-grade disclaimer — render verbatim.
        public var firewall: String?
        public var artifact: ArtifactBlock?
        public var alphaMultiples: [Double]?
        public var doseResponse: [Dose]?
        public var library: LibraryBlock?
    }

    public static func evalReport(runURL: URL) -> EvalReport? {
        decode(EvalReport.self, at: runURL.appending(component: "eval.json"))
    }

    // MARK: - interpret.json

    public struct InterpretReport: Codable, Sendable, Equatable {
        public struct LogitLensSign: Codable, Sendable, Equatable {
            public struct Token: Codable, Sendable, Equatable {
                public var piece: String?
                public var logit: Double?
            }

            public var promoted: [Token]?
            public var suppressed: [Token]?
            public var promotedConcentration: Double?
        }

        /// Every stage may instead be `{"skipped": "<reason>"}` — a
        /// configured stage whose dependency is unavailable records the
        /// skip rather than aborting.
        public struct LogitLens: Codable, Sendable, Equatable {
            public var skipped: String?
            public var layer: Int?
            public var positive: LogitLensSign?
            public var negative: LogitLensSign?
        }

        public struct SAE: Codable, Sendable, Equatable {
            public struct Feature: Codable, Sendable, Equatable {
                public var feature: Int?
                public var cosine: Double?
            }

            public var skipped: String?
            public var release: String?
            public var saeID: String?
            public var topByDecoderCosine: [Feature]?
            public var decoderCosineConcentration: Double?
        }

        public struct JLensSupport: Codable, Sendable, Equatable {
            public var skipped: String?
            public var lensID: String?
            public var layer: Int?
        }

        public struct Stages: Codable, Sendable, Equatable {
            public var logitLens: LogitLens?
            public var sae: SAE?
            public var jlensSupport: JLensSupport?
            public var library: LibraryBlock?
        }

        public var schemaVersion: Int?
        public var runID: String?
        public var claim: String?
        public var claimNote: String?
        /// S0/S1/S2/S3 (or "unknown" — never guessed).
        public var condition: String?
        public var artifact: ArtifactBlock?
        public var promptCount: Int?
        public var alphaMultiples: [Double]?
        public var stages: Stages?
    }

    public static func interpretReport(runURL: URL) -> InterpretReport? {
        decode(
            InterpretReport.self,
            at: runURL.appending(component: "interpret.json"))
    }

    // MARK: - family.json

    public struct FamilyReport: Codable, Sendable, Equatable {
        public struct Solution: Codable, Sendable, Equatable {
            public struct Match: Codable, Sendable, Equatable {
                public var name: String?
                public var concept: String?
                public var cosine: Double?
                public var absCosine: Double?
                public var nullPercentile: Double?
                public var clearsNull: Bool?
            }

            public var runID: String?
            public var condition: String?
            public var layer: Int?
            public var topLibraryMatch: Match?
            /// "no-library-match" is a FIRST-CLASS result category (the
            /// alien result), not a null.
            public var libraryMatchCategory: String?
            public var libraryComparedCount: Int?
            public var saeConcentration: Double?
            public var logitLensConcentration: Double?
            public var logitLensTopTokens: [String]?
        }

        public struct Contrast: Codable, Sendable, Equatable {
            public struct GroupStats: Codable, Sendable, Equatable {
                public var count: Int?
                public var meanAbsTopLibraryCosine: Double?
                public var meanLogitLensConcentration: Double?
                public var meanSAEConcentration: Double?
                public var noLibraryMatchCount: Int?
            }

            public var skipped: String?
            public var s1: GroupStats?
            public var s2: GroupStats?
            /// Values may be JSON null when a group statistic was
            /// uncomputable — keep the key visible rather than dropping it.
            public var deltaS1MinusS2: [String: Double?]?
        }

        public var schemaVersion: Int?
        public var runID: String?
        public var claim: String?
        public var claimNote: String?
        public var count: Int?
        public var conditions: [String: Int]?
        public var solutions: [Solution]?
        public var distinctLibraryMatches: Int?
        public var noLibraryMatchCount: Int?
        /// Separates "compared and matched nothing" (the alien result) from
        /// "compared against nothing".
        public var noLibraryComparedCount: Int?
        public var contrastS1S2: Contrast?
    }

    public static func familyReport(runURL: URL) -> FamilyReport? {
        decode(
            FamilyReport.self, at: runURL.appending(component: "family.json"))
    }

    // MARK: - jspace.json (exploratory tier)

    public struct JSpaceReport: Codable, Sendable, Equatable {
        /// One observation-layer aggregate. Every energy is decoded WITH its
        /// null pair — rendering one without the other is a schema
        /// violation by the writer's own rule.
        public struct LayerAggregate: Codable, Sendable, Equatable, Identifiable {
            public var layer: Int?
            public var itemCount: Int?
            public var isInjectionLayer: Bool?
            public var meanDeltaEnergy: Double?
            public var meanDeltaEnergyNull: Double?
            public var meanDeltaEnergyNullRatio: Double?
            public var directEnergy: Double?
            public var directEnergyNull: Double?
            public var directEnergyNullRatio: Double?
            public var meanEmergentEnergy: Double?
            public var meanEmergentEnergyNull: Double?
            public var meanEmergentEnergyNullRatio: Double?
            public var meanCosineDeltaDirect: Double?

            public var id: Int { layer ?? -1 }
        }

        public struct VectorBlock: Codable, Sendable, Equatable {
            public var artifact: ArtifactBlock?
            public var alphaMultiple: Double?
            public var layers: [LayerAggregate]?
        }

        public struct FamilyBlock: Codable, Sendable, Equatable {
            public struct Stats: Codable, Sendable, Equatable {
                public var participationRatio: Double?
                /// "How many directions is this family?" — read THIS one;
                /// the raw PR is dominated by the longest vector.
                public var participationRatioUnitNormalized: Double?
            }

            public struct ObservationLayer: Codable, Sendable, Equatable {
                public var layer: Int?
                public var participationRatio: Double?
                public var participationRatioUnitNormalized: Double?
            }

            public var layer: Int?
            public var count: Int?
            public var raw: Stats?
            public var byObservationLayer: [ObservationLayer]?
        }

        public var schemaVersion: Int?
        public var runID: String?
        /// Always "exploratory" — jspace never certifies sufficiency.
        public var claim: String?
        public var evidenceTier: String?
        /// The mandatory un-qualified-lens disclaimer (Stage 4 is
        /// unimplemented) — must ride along with any J-space rendering.
        public var qualification: String?
        public var injectionLayer: Int?
        public var observationLayers: [Int]?
        public var vectors: [VectorBlock]?
        /// Null for a single artifact (shallow-vs-deep multiplicity needs a
        /// family).
        public var family: FamilyBlock?
    }

    public static func jspaceReport(runURL: URL) -> JSpaceReport? {
        decode(
            JSpaceReport.self, at: runURL.appending(component: "jspace.json"))
    }

    // MARK: - geometry.json

    public struct GeometryReport: Codable, Sendable, Equatable {
        public struct Entry: Codable, Sendable, Equatable {
            public var reference: String?
            public var name: String?
            public var extractionMethod: String?
            public var isOptVec: Bool?
            public var seed: Int?
            public var norm: Double?
        }

        public var schemaVersion: Int?
        public var runID: String?
        public var claim: String?
        public var layer: Int?
        public var count: Int?
        public var participationRatio: Double?
        public var participationRatioUnitNormalized: Double?
        public var entries: [Entry]?
        public var cosineMatrix: [[Double]]?
    }

    public static func geometryReport(runURL: URL) -> GeometryReport? {
        decode(
            GeometryReport.self,
            at: runURL.appending(component: "geometry.json"))
    }

    // MARK: - Helpers

    static func decode<T: Decodable>(_ type: T.Type, at url: URL) -> T? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}
