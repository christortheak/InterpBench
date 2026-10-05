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
    /// The sampled record's outcome readings beyond the surface measures:
    /// the number the study's numeric parser read (`parsedMonths`, whatever
    /// its unit), the parsed choice, and the reader scores by concept. Each
    /// is `LenientlyDecoded`: a value of an unexpected type reads as absent
    /// instead of costing the record its other measures.
    let parsedMonths: LenientlyDecoded<Double>?
    let parsedChoice: LenientlyDecoded<String>?
    let readerScores: LenientlyDecoded<[String: Double]>?
}

/// A record field the analysis reads when it can and treats as absent when
/// it cannot. A plain optional would fail the WHOLE record on a value of the
/// wrong type, and a record that analyzed before these fields were read must
/// keep analyzing.
struct LenientlyDecoded<Value: Decodable>: Decodable {
    let value: Value?

    init(from decoder: Decoder) throws {
        value = try? Value(from: decoder)
    }
}

/// What `analyze` knows about the study's declared numeric parser. Only its
/// KIND matters here — whether the parsed numbers are months — because the
/// analysis pairs the values the run recorded and re-parses nothing.
enum AnalysisNumericParser: Equatable {
    /// The study names no parser.
    case undeclared
    case resolved(name: String, kind: String)
    /// Declared, but its registry entry cannot be read (or the registry
    /// changed since the study pinned it), in the reader's own words.
    case unreadable(name: String, reason: String)

    var kind: String? {
        if case .resolved(_, let kind) = self { return kind }
        return nil
    }
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
    var numericParser: AnalysisNumericParser = .undeclared
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
    /// Which outcomes reached `entries`, and which could not be produced.
    let outcomes: StudyAnalysisOutcomes.Coverage
    let diagnostics: [StudyAnalysisDiagnostic]
}

struct StudyStyleResult {
    let rows: [ExperimentTasks.MetricRow]
    let conditions: [String: ExperimentTasks.RescoreStyleReport.ConditionBlock]
}
