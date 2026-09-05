import Foundation
import SteeringKit

/// Minimal per-record view of generations.jsonl for analysis: sampled
/// records map onto MetricRow; choice-instrument records contribute
/// their ordinalPosition (when the run declared ordinalScale) and are
/// otherwise skipped.
struct AnalysisGeneration: Decodable {
    let instrument: String?
    let condition: String
    let seed: UInt64?
    let promptIndex: Int?
    let promptID: String
    let wordCount: Int?
    let distinct2: Float?
    let markerDensity: [String: Float]?
    /// Sampled output text — needed to RECOMPUTE reasoning-style values
    /// (they are derived, not stored on records).
    let output: String?
    /// The ordinalScale instrument's ladder position (instrument records
    /// of an ordinalScale run only) — one more paired numeric metric.
    let ordinalPosition: Double?
    /// Per-option joint logprobs (choice records only) — the input to the
    /// D3 distance-from-boundary diagnostics.
    let optionLogprobs: [String: Double]?
    /// Per-option log-odds against the rest of the option set, the
    /// per-option probabilities, the selected option and the item's
    /// target — the choice-deltas table's inputs (choice records only).
    let logOdds: [String: Double]?
    let choiceProbability: [String: Double]?
    let selected: String?
    let target: String?
    /// `"declared"` when the run stamped a DECLARED target (open-issues
    /// #6). Absent on every record written before the stamp existed —
    /// `ChoiceDeltas.targetIsDeclared` resolves those.
    let targetSource: String?
    /// Present only if a future sampled instrument writes one; today's
    /// answer-token readout is one per (condition, prompt).
    let sampleIndex: Int?
    /// Science-layer prompt metadata (stamped on sampled AND instrument
    /// records) — the stratification keys of the per-cell effect rows.
    let arm: String?
    let caseID: String?
    let factors: [String: String]?
}

extension ExperimentTasks {
    /// Cross-engine analyze output (`analysis.json`): epochUnverified is
    /// present ONLY when an unstamped run was accepted via
    /// --allow-unverified-epoch, and measurementDrift ONLY when a hash
    /// mismatch was tolerated because every drifted field was
    /// measurement-side (`RunEpoch.measurementFields`).
    struct AnalyzeReport: Codable {
        let experiment: String
        let experimentHash: String
        let sourceRun: String
        let sourceRunExperimentHash: String?
        let epochUnverified: Bool?
        let measurementDrift: String?
        let effectSizes: [EffectSizeEntry]
        /// Declared-exclusion stamp (cross-engine shape; also written as
        /// `exclusions.json`, the server's stamp file). nil ⇒ key omitted
        /// (no rules declared — analysis unchanged byte-for-byte).
        let exclusions: ExclusionStamp?
    }

    /// The cross-engine `reasoning-style.json` shape (sorted-keys JSON on
    /// both engines): source-run provenance + the pinned taxonomy identity +
    /// per-condition per-feature means. `epochUnverified` present ONLY when
    /// an unstamped run was accepted via --allow-unverified-epoch;
    /// `measurementDrift` ONLY when measurement-side drift was tolerated.
    struct RescoreStyleReport: Codable {
        struct ConditionBlock: Codable, Equatable {
            let features: [String: ReasoningStyleFeatureStat]
        }
        let experiment: String
        let experimentHash: String
        let sourceRun: String
        let sourceRunExperimentHash: String?
        let epochUnverified: Bool?
        let measurementDrift: String?
        let taxonomy: String
        let taxonomyHash: String
        /// The pinned taxonomy file, named beside its hash so the report is
        /// self-describing (same stamp as report.json's per-condition block).
        let taxonomyFile: String
        /// Style features are a diagnostic/manipulation check, never an
        /// outcome endpoint (docs/METHODS.md).
        let diagnosticOnly: Bool
        let conditions: [String: ConditionBlock]
    }
}

/// A captured source epoch and bytes. Calculation never reopens the workspace.
struct StudyAnalysisInput {
    let manifest: ExperimentManifest
    let sourceRunName: String
    let sourceRunExperimentHash: String?
    let epoch: RunEpoch.Check
    let generations: String
    let style: PinnedReasoningStyle?
    var exclusionChecks: [String: AttentionCheck] = [:]
    var declaredTargets: [String: Bool]? = nil
}

struct StudyAnalysisDiagnostic {
    let text: String
    var standardError: Bool = false
}

struct StudyAnalysisResult {
    let entries: [ExperimentTasks.EffectSizeEntry]
    let pooledCount: Int
    let sampledCount: Int
    let ordinalCount: Int
    let exclusions: ExclusionStamp?
    let choiceDeltas: (rows: [[String]], summary: ChoiceDeltas.Summary)
    let margins: [String: ChoiceMarginDiagnostics.Report]
    let diagnostics: [StudyAnalysisDiagnostic]
}

struct StudyStyleResult {
    let rows: [ExperimentTasks.MetricRow]
    let conditions: [String: ExperimentTasks.RescoreStyleReport.ConditionBlock]
}
