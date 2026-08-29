import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import SteeringKit
import Tokenizers

/// Headless experiment operations (`steerlab-cli experiment <verb> <name>`).
/// Experiments pin stimuli BY HASH plus extraction options; these tasks
/// re-derive vectors deterministically from the pinned recipe, verify the
/// pins first, and write immutable run directories stamped with the
/// experiment's content hash.
public enum ExperimentTasks {
    public struct ContextBudgetError: Error, CustomStringConvertible, Sendable {
        public let modelID: String
        public let contextWindow: Int
        public let promptTokens: Int
        public let requestedGenerationTokens: Int
        public let reservedTokens: Int

        public var description: String {
            let available = max(0, contextWindow - promptTokens - reservedTokens)
            return "\(modelID) context budget exceeded: prompt is \(promptTokens) tokens, "
                + "requested generation is \(requestedGenerationTokens), and the model window is "
                + "\(contextWindow) tokens. Reduce Max tokens to at most \(available), reset/split "
                + "the chat, or use a model with a larger context window."
        }
    }

    public enum StudyTaskProgress: Sendable {
        case runDirectory(String)
        case generationStarted(condition: String, promptID: String, prompt: String)
        case generationChunk(condition: String, promptID: String, output: String)
        case generationCompleted(StudyGenerationPreview)
        case evaluationDirectory(String)
        case judgmentStarted(condition: String, promptID: String)
        case judgmentCompleted(StudyJudgePreview)
        /// One per-response coding row written (the coding instrument's
        /// sibling of `judgmentCompleted`, 2026-08-04).
        case codingCompleted(StudyCodingPreview)
    }

    public typealias StudyTaskProgressHandler = @Sendable (StudyTaskProgress) async -> Void
    typealias GenerationChunkHandler = @Sendable (String) async -> Void

    private static let contextBudgetReserve = 16


    /// One measured task item. `options` + `target` drive the categorical
    /// instruments; anchor/severity/arm/caseID are science-layer metadata
    /// carried verbatim into records (JSON keys match the server's
    /// `_PROMPT_META_KEYS` + `options` exactly). `transcript` (optional) is a
    /// scripted multi-turn conversation — the metacognition-study instrument:
    /// researcher-authored assistant turns pinned as hashed stimulus data,
    /// with the model generating the reply to the final user turn. When it is
    /// present, `text` is the display text (item `text`/`prompt`, else the
    /// final user turn).
    struct StudyPrompt: Sendable {
        let id: String
        let text: String
        let options: [String]?
        let target: String?
        let anchorMonths: Double?
        let severity: Double?
        let arm: String?
        let caseID: String?
        var transcript: [TranscriptTurn]? = nil
        /// Declared per-item attention check (item key "attentionCheck":
        /// {"expected": …, "grading": <battery grading mode>?}) — graded at
        /// ANALYSIS time against the record's output when the manifest
        /// declares the failedAttentionCheck exclusion rule. nil = no check
        /// (legacy items unaffected).
        var attentionCheck: AttentionCheck? = nil
        /// Factorial-design cell metadata (item key "factors": factor name
        /// → level name, the generator's per-cell object) — carried
        /// verbatim onto every record the item produces so analysis can
        /// stratify by declared factors without rejoining the input file.
        /// nil = key omitted on records (non-factorial items unchanged).
        var factors: [String: String]? = nil
        /// What the prompt asks the model to EMIT (item key
        /// "responseFormat"). Decides whether the answer-token instruments
        /// can read this item at all — see `ResponseFormat`. nil = the key
        /// was omitted (legacy data, deliberately permissive).
        var responseFormat: ResponseFormat? = nil
    }

    struct GenerationRecord: Codable {
        let experiment: String
        let experimentHash: String
        let modelID: String
        let modelRevision: String?
        let taskPromptsFile: String
        let taskPromptsHash: String
        let promptMode: String
        /// The EFFECTIVE system prompt this arm generated under — the agent's
        /// persona composed with the study's frame, persona first
        /// (`SystemPromptComposition`). Never one level alone: replacement
        /// semantics are what the 2026-08-24 ruling ended.
        let systemPrompt: String?
        /// WHICH levels produced `systemPrompt`, as hashes. Always encoded
        /// (both keys explicit, `null` when that level contributed nothing) —
        /// the server stamps `systemPromptComposition` on every record beside
        /// its `systemPromptHash`, and an absent key would read as "this
        /// engine does not stamp composition".
        var systemPromptComposition: SystemPromptCompositionStamp = .none
        let qwenThinkingEnabled: Bool
        let condition: String
        let seed: UInt64
        /// true on local MLX runs: generation is greedy and the MLX sampler
        /// takes no per-run seed, so the recorded seed is provenance only —
        /// never read it as causally meaningful (CLAUDE.md › Sampling &
        /// measurement policy). Server-written records omit this field.
        let seedInert: Bool?
        let promptIndex: Int
        let promptID: String
        let prompt: String
        let output: String
        let wordCount: Int
        let distinct2: Float
        let markerDensity: [String: Float]
        let variantArtifactPath: String?
        let variantArtifactHash: String?
        /// Multi-agent turn identity (nil for every other study kind). These
        /// let the results tab rebuild a readable transcript from
        /// generations.jsonl alone, so transcript.md never has to be a
        /// measurement input. Cross-engine twins of the server's flattened
        /// record keys.
        var speakerName: String? = nil
        var turnTitle: String? = nil
        var routedAgentIDs: [String]? = nil
        var replicateIndex: Int? = nil
        /// The saved agent's stored Playground temperature — PROVENANCE
        /// ONLY (cross-engine key, 2026-07-21 study-owned sampling): the
        /// study manifest owns the measured-run sampling policy for every
        /// condition, and `temperature`/greedy execution is what governed
        /// generation. Stamped on variant-condition records; nil ⇒ key
        /// omitted (ordinary conditions).
        var agentPlaygroundTemperature: Double? = nil
        // Science-layer prompt metadata, carried verbatim from the task item
        // (server `_PROMPT_META_KEYS`); nil ⇒ key omitted, matching the
        // server, which only stamps keys present on the item.
        let target: String?
        let anchorMonths: Double?
        let severity: Double?
        let arm: String?
        let caseID: String?
        /// Built-in outcome-endpoint parses (`Judicial`). The double optionals
        /// mirror the server's
        /// JSON exactly: outer nil ⇒ key absent (case family / options do not
        /// apply to this record); inner nil ⇒ JSON `null` (a parse FAILURE —
        /// the failure rate is a first-class coherence endpoint, so the key
        /// must appear even when parsing fails). Synthesized Codable encodes
        /// `.some(nil)` as `null`; note that *decoding* collapses `null` back
        /// to absent — these records are write-only on this substrate.
        let parsedMonths: Double??
        let parsedChoice: String??
        /// RepE reader scores of the sampled OUTPUT text, one per pinned
        /// reader concept — stamped when `outcomeInstruments` contains
        /// "repeReaderScore" (server `_reader_scores` twin). The capture is
        /// unsteered: interventions are cleared before scoring. nil ⇒ key
        /// omitted (instrument not requested; variant/multi-agent paths).
        var readerScores: [String: Float]? = nil
        /// Which random-control recipe generated the injected direction —
        /// `SteeringVectorMath.randomVectorAlgorithm`, stamped when this
        /// record's condition is a `randomMatchedNorm` control (the server
        /// stamps the identical string inside `interventionState`). nil ⇒
        /// key omitted. An UNSTAMPED random-control record is legacy:
        /// cube-uniform on Swift, Gaussian on the server.
        var randomVectorAlgorithm: String? = nil
        /// true on scripted-transcript items (cross-engine key
        /// "scriptedTranscript"); nil ⇒ key omitted (plain items).
        var scriptedTranscript: Bool? = nil
        /// The scripted transcript itself, carried into the record — records
        /// are the rebuild-without-rerun archive and the transcript is the
        /// stimulus. nil ⇒ key omitted.
        var transcript: [TranscriptTurn]? = nil
        /// Factorial-design cell metadata carried verbatim from the item
        /// (cross-engine key "factors", stamped on sampled AND instrument
        /// records — server `_PROMPT_META_KEYS` twin). nil ⇒ key omitted
        /// (non-factorial items produce byte-identical records).
        var factors: [String: String]? = nil
        /// A multi-agent turn's declared-endpoint parse, carried verbatim
        /// from the turn record (the runner stamped it at write time; nothing
        /// re-parses here — one parse, one place). nil ⇒ key omitted, which
        /// is every non-panel record and every panel turn that declared no
        /// endpoint. Server twin: the `endpoint` key `_panel_records_from`
        /// forwards.
        var endpoint: TurnEndpointStamp? = nil
        /// A multi-agent turn's voice lint (spec §5), carried verbatim from
        /// the turn record — one lint, one place. nil ⇒ key omitted, which is
        /// every non-panel record and every panel turn written before the
        /// lint existed. Server twin: the `voiceLint` key
        /// `_panel_records_from` forwards.
        var voiceLint: VoiceLintStamp? = nil
    }

    /// The declared instrument ids that dispatch the answer-token choice
    /// scoring path (one deterministic readout per condition × prompt).
    /// `ordinalScale` rides the same machinery — it only adds the ordinal
    /// aggregation fields to the record. Server twin:
    /// `tasks.CHOICE_INSTRUMENTS`.
    static let choiceInstruments: Set<String> = [
        "answerTokenLogprob", "choiceProbability", "ordinalScale",
    ]

    /// One answer-token-logprob readout (`instrument: "answerTokenLogprob"`).
    /// Field names MUST match the server's `ChoiceResult.as_record_fields`
    /// (`steerlab_server/experiment/logprob.py`) plus its choice-record
    /// envelope in `tasks.py`, so cross-engine analysis reads one shape.
    /// No `seed`: the instrument is temperature-free by construction.
    struct ChoiceRecord: Codable {
        let experiment: String
        let experimentHash: String
        let modelID: String
        let modelRevision: String?
        let taskPromptsFile: String
        let taskPromptsHash: String
        let promptMode: String
        /// The EFFECTIVE system prompt this readout was taken under, and
        /// which levels produced it — same contract as
        /// `GenerationRecord.systemPrompt`/`systemPromptComposition`.
        let systemPrompt: String?
        var systemPromptComposition: SystemPromptCompositionStamp = .none
        let qwenThinkingEnabled: Bool
        let condition: String
        let promptIndex: Int
        let promptID: String
        let prompt: String
        /// The option the endpoint tracks — the item's DECLARED `target`, and
        /// nil when the item declared none (open-issues #6).
        ///
        /// This used to fall back to `options[0]`, which for an ordinalScale
        /// item is the rating ladder's minimum: every likert record was
        /// stamped `target: "1"`, and the analyze layer then faithfully
        /// reported a `choiceLogOdds` endpoint nobody declared (pole movement
        /// entangled with distribution sharpening). A None target is a fact
        /// about the item; the endpoint is emitted only for declared ones.
        let target: String?
        /// `"declared"` when the item's own `target` supplied the value above,
        /// nil otherwise. Cross-engine contract key (server twin:
        /// `tasks.py`'s choice-record envelope) — it is what lets a consumer
        /// tell a genuinely target-less item from a legacy record whose target
        /// was synthesized.
        let targetSource: String?
        let anchorMonths: Double?
        let severity: Double?
        let arm: String?
        let caseID: String?
        let instrument: String
        let options: [String]
        /// Per-option token counts + max/min ratio (min clamped to 1) — the
        /// option-length guardrail: joint logprobs favor shorter options, so
        /// a ratio well above 1 flags an instrument-design smell.
        let optionTokenCounts: [String: Int]
        let optionLengthRatio: Double
        let optionTokenIDs: [String: [Int]]
        let optionTokenLogprobs: [String: [Float]]
        let optionLogprobs: [String: Double]
        let optionMeanTokenLogprobs: [String: Double]
        let choiceProbability: [String: Double]
        let logOdds: [String: Double]
        let selected: String
        let margin: Double
        /// Ordinal-scale instrument fields (cross-engine contract keys
        /// "ordinalPosition"/"ordinalDistribution"), stamped when the
        /// manifest declares `outcomeInstruments: ["ordinalScale"]`:
        /// the per-option probabilities renormalized over the declared
        /// ladder (in ladder order), and the 1-based ladder position under
        /// the manifest's declared `ordinalAggregation`. nil ⇒ keys omitted
        /// (ordinalScale not declared).
        var ordinalPosition: Double? = nil
        var ordinalDistribution: [Double]? = nil
        /// Random-control recipe stamp — same contract as
        /// `GenerationRecord.randomVectorAlgorithm` (present only on
        /// `randomMatchedNorm` control conditions; unstamped = legacy).
        var randomVectorAlgorithm: String? = nil
        /// Scripted-transcript stamps — same contract as
        /// `GenerationRecord.scriptedTranscript`/`transcript`.
        var scriptedTranscript: Bool? = nil
        var transcript: [TranscriptTurn]? = nil
        /// Factorial cell metadata — same contract as
        /// `GenerationRecord.factors` (instrument readouts carry the item's
        /// factors too, so stratified analysis never rejoins the input).
        var factors: [String: String]? = nil
    }

    struct MetricRow {
        let condition: String
        let seed: UInt64
        let promptIndex: Int
        let promptID: String
        let wordCount: Int
        let distinct2: Float
        let markerDensity: [String: Float]
        /// Reasoning-style feature values (`rs_<featureID>` columns), keyed
        /// by feature id — empty when the manifest pins no taxonomy.
        var reasoningStyle: [String: Double] = [:]
        /// The item's factorial cell (`factor_<name>` metrics.csv columns)
        /// — empty for non-factorial items (no columns appear).
        var factors: [String: String] = [:]
        /// Which transcript (independent play-through) this row came from, for
        /// multi-agent studies; nil everywhere else.
        ///
        /// This is the CLUSTER identity, and it is what makes multi-agent
        /// statistics honest: turns within one transcript are not independent
        /// observations — turn k is conditioned on turns 1..k-1 — so the
        /// analysis aggregates to the transcript before testing anything
        /// (D1). Nil means "this row IS its own unit", which is every other
        /// study kind and leaves their arithmetic untouched.
        var replicate: Int? = nil
    }

    /// Minimal categorical readout retained while assembling report.json.
    /// Full records still stream directly to generations.jsonl; this carries
    /// only what the cross-engine report summaries require.
    struct ReportChoiceReadout: Sendable, Equatable {
        let condition: String
        let promptID: String
        let sampleIndex: UInt64?
        let source: String  // "instrument" or "parsed"
        let selected: String
        let target: String?
        /// The record's ordinal ladder position (instrument readouts of an
        /// ordinalScale study only) — feeds the per-condition
        /// ordinalMean/ordinalSD summary. nil on parsed readouts and
        /// non-ordinal studies.
        var ordinalPosition: Double? = nil
    }

    /// Per-condition capability-battery readout of a STUDY RUN (distinct
    /// from validate-time evidence): the pinned battery scored under this
    /// condition's full intervention. JSON keys are the pinned cross-engine
    /// contract: {"accuracy", "itemCount", "batteryHash"}.
    struct CapabilityBatterySummary: Codable, Equatable {
        let accuracy: Double
        let itemCount: Int
        let batteryHash: String
    }

    /// One battery reading of a study run. These go to `battery.jsonl`,
    /// NEVER `generations.jsonl` — battery items are capability controls,
    /// not study outputs, and must not enter outcome analysis.
    ///
    /// The trailing fields are format-2 only and carry the SERVER's key names
    /// (`battery.score_item` + `BatteryArming.as_record_fields`), so one
    /// reader parses either engine's battery.jsonl. They are Optionals, so a
    /// format-1 record encodes the exact six keys it always did — the server
    /// omits them on legacy rows for the same reason.
    struct BatteryGenerationRecord: Codable, Equatable {
        let condition: String
        let promptIndex: Int
        let prompt: String
        let expected: String
        let output: String
        let correct: Bool
        /// 2 on an isolated battery; absent on legacy rows.
        var batteryFormat: Int? = nil
        var scoring: String? = nil
        var options: [String]? = nil
        var choiceProbability: [String: Double]? = nil
        var selected: String? = nil
        /// What the reading was ARMED with — the arming fields the server
        /// stamps, spelled identically.
        var armingIsolated: Bool? = nil
        var armingPromptMode: String? = nil
        var armingSystemPrompt: Bool? = nil
        /// The effective arming system prompt's hash, and WHICH levels
        /// produced it (2026-08-24 composition ruling). The composition's
        /// second key is `battery`, not `study`: the study frame never enters
        /// a battery generation. Format-2 rows only, like their neighbours.
        ///
        /// The double Optional is this file's established idiom for "absent
        /// and null are different claims" (see `GenerationRecord
        /// .parsedChoice`): outer nil ⇒ key omitted, which is every format-1
        /// row; inner nil ⇒ JSON `null`, a format-2 row armed with no system
        /// text at all. The server stamps exactly that shape.
        var armingSystemPromptHash: String?? = nil
        var armingSystemPromptComposition: BatteryArmingCompositionStamp? = nil
        var armingMaxTokens: Int? = nil
    }

    /// One reasoning-style feature's per-condition aggregate (cross-engine
    /// report keys: {"mean", "n"}).
    struct ReasoningStyleFeatureStat: Codable, Equatable {
        let mean: Double
        let n: Int
    }

    /// The per-condition `reasoningStyle` block of report.json (cross-engine
    /// contract: {"taxonomy", "taxonomyHash", "taxonomyFile",
    /// "diagnosticOnly", "features": {id: {mean, n}}}).
    struct ReasoningStyleConditionReport: Codable, Equatable {
        let taxonomy: String
        let taxonomyHash: String
        /// The pinned taxonomy file, named beside its hash so the report is
        /// self-describing. Optional only for decoding pre-stamp reports.
        var taxonomyFile: String? = nil
        /// Style features are a diagnostic/manipulation check reported
        /// beside outcome endpoints, never an outcome endpoint itself
        /// (docs/METHODS.md). Optional only for decoding pre-stamp reports.
        var diagnosticOnly: Bool? = nil
        let features: [String: ReasoningStyleFeatureStat]
    }

    struct ConditionReport: Codable {
        let generations: Int
        let meanWordCount: Float
        let meanDistinct2: Float
        let meanMarkerDensity: [String: Float]
        /// Number of deterministic answer-token instrument records. Omitted
        /// when the instrument was not run (server report parity).
        var choiceReadouts: Int? = nil
        /// Fraction of parseable sampled outputs that chose the item's target.
        var choiceRate: Double? = nil
        /// Exact readout agreement with the same-item baseline.
        var agreementWithBaseline: ChoiceAgreementSummary? = nil
        /// Ordinal-scale summary over this condition's instrument readouts
        /// (cross-engine contract keys "ordinalMean"/"ordinalSD"; SD is the
        /// population standard deviation, 0 for a single readout). nil when
        /// the ordinalScale instrument produced no readouts.
        var ordinalMean: Double? = nil
        var ordinalSD: Double? = nil
        /// Present when the manifest pins a capability battery (contract key
        /// "capabilityBattery"); nil on legacy reports and unpinned studies.
        let capabilityBattery: CapabilityBatterySummary?
        /// Present when the manifest pins a reasoning-style taxonomy
        /// (contract key "reasoningStyle"); nil otherwise.
        var reasoningStyle: ReasoningStyleConditionReport? = nil
    }

    struct ChoiceAgreementSummary: Codable, Equatable {
        let n: Int
        let agreement: Double
    }

    /// Paired-to-baseline effect size for one (condition, metric): mean of
    /// per-item (condition − same-item-baseline) differences with a
    /// percentile bootstrap CI and a Wilcoxon signed-rank companion
    /// (CLAUDE.md reporting policy). Wilcoxon fields are nil when the test
    /// is undefined (all differences zero).
    struct EffectSizeEntry: Codable, Equatable {
        let condition: String
        let metric: String
        let n: Int
        let meanDiff: Double
        let ciLower: Double
        let ciUpper: Double
        let wilcoxonW: Double?
        let wilcoxonP: Double?
        /// Multiple-comparison-adjusted Wilcoxon p (server contract key
        /// "adjustedP"): BH-FDR for screen/unphased studies, Holm for the
        /// confirm family — filled by `applyCorrection`. nil when the raw
        /// Wilcoxon p is undefined (the correction skips it, mirroring the
        /// server's `apply_correction`).
        var adjustedP: Double? = nil
        /// The correction family applied: "bh" | "holm" (server contract
        /// key "correction"). Stamped on every row of a corrected family,
        /// including rows whose p was undefined.
        var correction: String? = nil
        /// Stratified-analysis provenance (2026-08-06, cross-engine CSV
        /// columns stratifyBy/stratum/unit — server `EffectRow` twin). nil
        /// on pooled rows (keys omitted, so run-report bytes are unchanged;
        /// the CSV writes "pooled" for a nil stratifyBy). Stratified rows
        /// carry the family ("promptID", a factor key, or "×"-joined
        /// crossed keys), the cell label, and what one paired difference
        /// IS: "item" (one pair per item — the pooled semantics restricted
        /// to the stratum) or "sample" (multiple pairs within an item).
        var stratifyBy: String? = nil
        var stratum: String? = nil
        var unit: String? = nil
        /// WHAT this row estimates (cross-engine CSV column "estimand"):
        /// `itemLevel` — one paired difference per item, the same estimand
        /// the pooled rows report, restricted to this stratum; or
        /// `withinItemSamples` — several draws of the SAME item paired
        /// against that item's baseline draws, which is a within-item
        /// variability read, not an item-level effect. nil (column empty) on
        /// pooled rows.
        var estimand: String? = nil
        /// What may be CLAIMED from this row (cross-engine CSV column
        /// "inference"): `corrected` — a member of its family's
        /// multiple-comparison correction, so `adjustedP` is meaningful; or
        /// `diagnostic` — deliberately held OUT of the correction family and
        /// carrying no `adjustedP`, because within-item sample rows are not
        /// independent tests of the pre-registered hypothesis and correcting
        /// across them both inflates the family and licenses a claim the
        /// design cannot support. Raw Wilcoxon and the bootstrap CI are kept
        /// — the row is still readable, just not citable as a test. nil
        /// (column empty) on pooled rows.
        var inference: String? = nil

        /// A row the correction family must exclude (server twin: the
        /// `withinItemSamples`/`diagnostic` pairing).
        var isWithinItemSamples: Bool {
            estimand == EffectSizeEstimand.withinItemSamples
        }
    }

    /// The `estimand` column's closed vocabulary (cross-engine strings).
    enum EffectSizeEstimand {
        static let itemLevel = "itemLevel"
        static let withinItemSamples = "withinItemSamples"
    }

    /// The `inference` column's closed vocabulary (cross-engine strings).
    enum EffectSizeInference {
        static let corrected = "corrected"
        static let diagnostic = "diagnostic"
    }

    struct StudyRunReport: Codable {
        let experiment: String
        let experimentHash: String
        let taskPromptsFile: String
        let taskPromptsHash: String
        let promptMode: String
        let systemPrompt: String?
        let qwenThinkingEnabled: Bool
        let promptCount: Int
        let conditionCount: Int
        let seedCount: Int
        let conditions: [String: ConditionReport]
        /// Paired effect sizes vs the same-item baseline (contract key
        /// "effectSizes"); nil on legacy reports, empty when no non-baseline
        /// condition pairs with a baseline row.
        let effectSizes: [EffectSizeEntry]?
        /// What one row of `effectSizes` is an average OVER (plan D1). Absent
        /// on every ordinary study, where the item is the unit and always was.
        /// "transcript" on multi-agent runs, where turns are dependent within
        /// a play-through and the estimator aggregates to the transcript
        /// before testing — so `n` counts transcripts, not turns. Stamped
        /// rather than assumed, because a reader cannot otherwise tell which
        /// of the two a given `n` means.
        /// Models a panel run actually USED, as a set. The manifest's
        /// `modelID` is a declared default for seats that name none — a
        /// panel's seats may each carry their own, so no scalar describes the
        /// run. `config.json`'s key set is a closed cross-engine contract, so
        /// the multi-model view lives here instead.
        var modelsUsed: [String]? = nil
        var declaredModelID: String? = nil
        var modelBySeat: [String: String]? = nil
        var unitOfAnalysis: String? = nil
        /// Independent play-throughs per condition (multi-agent only). 1 means
        /// no replication: point estimates only, no intervals.
        var transcriptsPerCondition: Int? = nil
        /// Registry-parser provenance (cross-engine contract key
        /// "numericParser": {"name", "kind", "registryFile",
        /// "registryHash"}) — stamped only when a declared parser actually
        /// parsed this run's numeric outcome; nil ⇒ key omitted (legacy
        /// report bytes unchanged).
        var numericParser: ParserRegistry.NumericParserProvenance? = nil
        /// Declared-exclusion stamp (cross-engine contract key
        /// "exclusions"): active rules with plain-language descriptions,
        /// per-condition per-rule exclusion counts, surviving N, and the
        /// pairwise-deletion note. Stamped only when the manifest declares
        /// exclusionRules; nil ⇒ key omitted (legacy report bytes
        /// unchanged).
        var exclusions: ExclusionStamp? = nil
    }

    struct EvaluationGeneration: Decodable {
        let experiment: String?
        let condition: String
        let seed: UInt64
        /// The record's sample cell within its (condition, prompt) —
        /// stamped by server sampled runs (`samplesPerItem`); absent on
        /// local greedy single-sample records and normalizes to 0.
        let sampleIndex: UInt64?
        let promptID: String
        let prompt: String
        let output: String
    }

    struct PairedJudgeRecord: Codable {
        enum CodingKeys: String, CodingKey {
            case experiment
            case experimentHash
            case sourceRunDirectory
            // Server contract: judgment rows stamp the judge's panel name
            // under the key "judge" (`tasks.py` `judgment["judge"]`).
            case judgeName = "judge"
            case judgeKind
            case judgeModel
            case judgeProvider
            case judgeRevision
            case judgePrompt
            case judgeRubricFile
            case judgeRubricHash
            case structuredPrompt
            case condition
            case sampleIndex
            case baselineSeed
            case variantSeed
            case promptID
            case prompt
            case baselineWas
            case conditionWas
            case judgment
            case conditionResult
        }

        let experiment: String
        let experimentHash: String
        let sourceRunDirectory: String
        /// Judge-panel provenance: which judge produced THIS record. Every
        /// new record stamps a name (the panel entry's, or "judge-1" on the
        /// legacy single-judge path); kind is "claude" | "openrouter" |
        /// "local".
        let judgeName: String
        let judgeKind: String
        let judgeModel: String
        /// OpenRouter judges only: the pinned serving provider the client
        /// verified against the response (cross-engine key "judgeProvider").
        var judgeProvider: String? = nil
        /// Local judges: the pinned revision of the judge model that
        /// actually judged (JudgeRef.revision / study-pin fallback,
        /// 2026-07-23) — judgment artifacts name the exact judge bytes.
        var judgeRevision: String? = nil
        let judgePrompt: String
        /// Rubric pin when the study evaluates through a versioned rubric
        /// file; nil = inline draft rubric text (judgePrompt holds it either
        /// way, so records are self-contained).
        let judgeRubricFile: String?
        let judgeRubricHash: String?
        let structuredPrompt: String?
        let condition: String
        /// The pair's sample cell — the cross-engine JOIN key half
        /// (external review 2026-07-22): pairs join on (promptID,
        /// sampleIndex), never the seed, which under the server's
        /// derivedSHA256 policy includes condition identity and therefore
        /// differs between the two sides of a pair by design. Local
        /// records are greedy single-sample, so this is 0 unless the
        /// source run stamped `sampleIndex`.
        let sampleIndex: UInt64
        /// Seed provenance for BOTH sides of the pair (cross-engine keys
        /// "baselineSeed"/"variantSeed" — deliberately no field named
        /// "seed": a pair has two).
        let baselineSeed: UInt64
        let variantSeed: UInt64
        let promptID: String
        let prompt: String
        let baselineWas: String
        let conditionWas: String
        let judgment: PairedJudgeResponse
        let conditionResult: String
    }

    /// One NONCOMPLIANT judgment row in `judgments.jsonl` — a pair whose
    /// judge answered twice and produced no verdict either time (Christian,
    /// 2026-08-09).
    ///
    /// A separate row type on purpose: a `PairedJudgeRecord` cannot exist
    /// without a verdict, and the refusal to invent one stands. What
    /// changed is that the failure is now KEPT — for later examination and
    /// classification — instead of destroying an evaluation that had
    /// already paid for hundreds of good judgments. Because these rows are
    /// not `PairedJudgeRecord`s they are structurally absent from every
    /// tally, agreement statistic, and token sum; nothing has to remember
    /// to filter them.
    ///
    /// JSON keys are the cross-engine contract with the dict
    /// `paired_judge.evaluate` builds and `_judgment_stamp_judge` stamps:
    /// `outcome` and `judgment` are present-and-NULL (a reader must be able
    /// to see the hole, not infer it from an absent key), `noncompliant` /
    /// `noncomplianceReason` carry the classification, and `judge`
    /// (+ `judgeProvider` for OpenRouter) names who failed. Compliant rows
    /// gain no key from any of this.
    struct NoncompliantJudgmentRecord: Encodable {
        enum CodingKeys: String, CodingKey {
            case promptID
            case sampleIndex
            case condition
            case baselineSeed
            case variantSeed
            case baselineWas
            case outcome
            case noncompliant
            case noncomplianceReason
            case judgment
            case judgeName = "judge"
            case judgeProvider
        }

        let promptID: String
        let sampleIndex: UInt64
        let condition: String
        let baselineSeed: UInt64
        let variantSeed: UInt64
        let baselineWas: String
        /// The typed refusal, verbatim and bounded — the raw material a
        /// researcher classifies the failure from.
        let noncomplianceReason: String
        let judgeName: String
        var judgeProvider: String? = nil

        /// Hand-written so the null verdict fields are EMITTED as null.
        /// Synthesized Codable omits nil optionals, and an omitted
        /// `outcome` would read as a legacy row rather than a recorded
        /// hole.
        func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(promptID, forKey: .promptID)
            try container.encode(sampleIndex, forKey: .sampleIndex)
            try container.encode(condition, forKey: .condition)
            try container.encode(baselineSeed, forKey: .baselineSeed)
            try container.encode(variantSeed, forKey: .variantSeed)
            try container.encode(baselineWas, forKey: .baselineWas)
            try container.encodeNil(forKey: .outcome)
            try container.encode(true, forKey: .noncompliant)
            try container.encode(
                noncomplianceReason, forKey: .noncomplianceReason)
            try container.encodeNil(forKey: .judgment)
            try container.encode(judgeName, forKey: .judgeName)
            try container.encodeIfPresent(
                judgeProvider, forKey: .judgeProvider)
        }
    }

    struct PairedJudgeConditionReport: Codable {
        let pairs: Int
        let conditionWins: Int
        let baselineWins: Int
        let ties: Int
        let meanConfidence: Double
        let structuredSummaries: [String: StructuredFieldSummary]
    }

    struct StructuredFieldSummary: Codable {
        let count: Int
        let numericMean: Double?
        let trueCount: Int?
        let falseCount: Int?
        let stringCounts: [String: Int]?

        enum CodingKeys: String, CodingKey {
            case count
            case numericMean = "numeric_mean"
            case trueCount = "true_count"
            case falseCount = "false_count"
            case stringCounts = "string_counts"
        }
    }

    /// Agreement between two judges over the item pairs both judged
    /// (labels: condition | baseline | tie). `kappa` is nil when Cohen's
    /// kappa is undefined (both judges each constant on different labels).
    struct JudgeAgreementReport: Codable {
        let judgeA: String
        let judgeB: String
        let items: Int
        let percentAgreement: Double
        let kappa: Double?
    }

    /// One judge's agreement with the pinned human-validation subset.
    struct HumanAgreementReport: Codable {
        let judge: String
        let items: Int
        let percentAgreement: Double
        let kappa: Double?
    }

    struct PairedJudgeReport: Codable {
        let experiment: String
        let experimentHash: String
        let sourceRunDirectory: String
        /// All resolved judge model ids, comma-joined (legacy readers show a
        /// single string; per-judge detail lives in `judges`).
        let judgeModel: String
        /// Panel names in evaluation order; nil never occurs on new reports
        /// but keeps legacy report decoding intact.
        let judges: [String]?
        let judgeRubricFile: String?
        let judgeRubricHash: String?
        /// Pairwise inter-judge agreement (percent + Cohen's kappa).
        let judgeAgreement: [JudgeAgreementReport]?
        /// Per-judge vs-human agreement over the pinned humanValidation
        /// subset; nil when the manifest pins none.
        let humanAgreement: [HumanAgreementReport]?
        /// true only when an UNSTAMPED source run was accepted via
        /// allow-unverified-epoch (cross-engine key "epochUnverified");
        /// nil (key omitted) on epoch-verified evaluations.
        let epochUnverified: Bool?
        /// The changed-fields description when a hash mismatch was TOLERATED
        /// because every drifted field was measurement-side
        /// (`RunEpoch.measurementFields`); nil (key omitted) otherwise.
        /// Cross-engine key "measurementDrift" — tolerated is never silent.
        var measurementDrift: String? = nil
        /// Where the effective evaluation spec came from (cross-engine key
        /// "evaluationSource", 2026-07-22): "manifest" = an explicit
        /// evaluation block (or caller override); "pinnedRubric" = spec
        /// synthesized from the pinned judges + rubric file; nil (key
        /// omitted) = the legacy no-declaration fallback path.
        let evaluationSource: String?
        let conditions: [String: PairedJudgeConditionReport]
        /// Declared-exclusion stamp (cross-engine shape; also written as
        /// `exclusions.json` in the evaluate run directory). nil ⇒ key
        /// omitted (no rules declared — evaluate unchanged byte-for-byte).
        var exclusions: ExclusionStamp? = nil
        /// Per-judge token totals, keyed by panel name (2026-08-06). The
        /// server stamps the same `completionTokens`/`reasoningTokens` sums
        /// directly on its per-judge report blocks; this engine's `judges`
        /// are bare name strings, so the parallel map is where they live.
        /// Present only for judges whose transport reports usage
        /// (OpenRouter today); nil ⇒ key omitted. REPORTED, NEVER GATED —
        /// no code path reads it to refuse, cap, or select.
        var judgeUsage: [String: PairedJudgeUsage]? = nil
        /// How many pairs this panel ANSWERED without ever producing a
        /// valid verdict (Christian, 2026-08-09). Loud and NONZERO-ONLY
        /// (cross-engine key `noncompliantJudgments`): those pairs carry no
        /// verdict, sit outside every condition tally and agreement
        /// statistic, and survive only as rows in `judgments.jsonl` — a
        /// reader must be able to see that the column is incomplete and by
        /// how much. nil ⇒ key omitted, so a clean report is byte-identical
        /// to before.
        var noncompliantJudgments: Int? = nil
    }

    /// One row of the pinned human-validation subset
    /// (`manifest.humanValidation`). The file format is a cross-engine data
    /// contract (the server's `_load_human_validation` reads the same rows):
    /// `{"condition": …, "promptID": …, "outcome": "baseline"|"variant"|"tie"[, "sampleIndex": …]}`.
    /// "variant" corresponds to this engine's "condition" result label.
    /// Rows key on the pair-cell `sampleIndex` (the pairing join key) —
    /// never a seed, which differs between the two sides of a pair under
    /// derived seeding. A row without a sampleIndex is an explicit
    /// WILDCARD: it matches every sample cell of its (condition, promptID)
    /// that no exact-indexed row claims (exact beats wildcard — the
    /// cross-engine rule since 2026-08-01). Duplicate keys refuse at parse.
    struct HumanValidationRow: Decodable {
        let condition: String
        let promptID: String
        let outcome: String
        let sampleIndex: UInt64?

        /// The outcome mapped onto this engine's conditionResult vocabulary.
        var conditionResult: String { outcome == "variant" ? "condition" : outcome }
    }

    // MARK: - Shared plumbing

    private static func generationPreview(from record: GenerationRecord) -> StudyGenerationPreview {
        let limit = 1_800
        let truncated = record.output.count > limit
        let output = truncated ? String(record.output.prefix(limit)) : record.output
        return StudyGenerationPreview(
            condition: record.condition,
            promptID: record.promptID,
            prompt: record.prompt,
            output: output,
            wordCount: record.wordCount,
            distinct2: record.distinct2,
            markerDensity: record.markerDensity,
            truncated: truncated)
    }

    private static func judgePreview(
        from record: PairedJudgeRecord,
        rawJSON: String
    ) -> StudyJudgePreview {
        StudyJudgePreview(
            condition: record.condition,
            sampleIndex: record.sampleIndex,
            baselineSeed: record.baselineSeed,
            variantSeed: record.variantSeed,
            promptID: record.promptID,
            prompt: record.prompt,
            baselineWas: record.baselineWas,
            conditionWas: record.conditionWas,
            winner: record.judgment.winner,
            conditionResult: record.conditionResult,
            confidence: record.judgment.confidence,
            briefReason: record.judgment.briefReason,
            aScores: record.judgment.aScores,
            bScores: record.judgment.bScores,
            structuredFields: record.judgment.structuredFields,
            rawJSON: rawJSON)
    }

    // MARK: - Cooperative cancellation (App gap A1)

    /// The sweep's cooperative-cancellation pattern, shared by every long
    /// local task (run / validate / extract / paired judging / robustness):
    /// the caller's flag is polled BETWEEN units of work — never mid-
    /// generation — and the first observation is logged with where it was
    /// seen. The flag is one-way for the duration of one operation (the
    /// panel resets it only when the NEXT operation starts), so re-polling
    /// after a phase boundary is safe and cheap.
    ///
    /// A cancelled operation returns normally with honestly-partial
    /// artifacts and a `cancelled.txt` status note — never an error, and
    /// never a fake completion (completion artifacts like `report.json` /
    /// validation evidence are simply not written).
    struct CancelPoller: Sendable {
        let shouldCancel: (@Sendable () async -> Bool)?
        let log: (@Sendable (String) async -> Void)?

        init(
            _ shouldCancel: (@Sendable () async -> Bool)?,
            log: (@Sendable (String) async -> Void)? = nil
        ) {
            self.shouldCancel = shouldCancel
            self.log = log
        }

        /// True when a cancellation request has been observed at this poll
        /// point (logged, sweep-style: `cancellation observed at <where>`).
        func observed(at location: String) async -> Bool {
            guard let shouldCancel, await shouldCancel() else { return false }
            let line = "cancellation observed at \(location)"
            print(line)
            await log?(line)
            return true
        }
    }

    /// Status-note filename a cancelled task leaves in its run directory.
    static let cancellationNoteFileName = "cancelled.txt"

    /// The honest partial-artifact marker: what stopped, and that nothing in
    /// the directory may be read as a completed run. Written where a run
    /// directory already exists; completion artifacts (`report.json`,
    /// validation evidence, `judge-report.json`) are never written for a
    /// cancelled task, so downstream readers (`newestCompletedRun`, freeze's
    /// evidence scan) skip it mechanically, not by convention.
    static func cancellationNote(task: String) -> String {
        "cancelled by user — this \(task) stopped between units of work; "
            + "artifacts in this directory are PARTIAL and this is not a "
            + "completed \(task)"
    }

    /// Best-effort durable stamp (like the run-start advisory): the note
    /// must never sink the cancellation path itself.
    static func writeCancellationNote(task: String, to runDirectory: URL) {
        try? (cancellationNote(task: task) + "\n").write(
            to: runDirectory.appending(component: cancellationNoteFileName),
            atomically: true, encoding: .utf8)
    }

    /// Loads the manifest and hard-fails on any freeze violation.
    static func loadVerified(_ name: String) throws -> ExperimentManifest {
        let manifest = try ExperimentStore.load(name: name)
        let violations = ExperimentStore.verify(manifest)
        guard violations.isEmpty else {
            throw ExperimentError.refusing(
                .pinDrift,
                "experiment '\(name)' failed verification:\n  - "
                    + violations.joined(separator: "\n  - "),
                repair: pinDriftRepair(name: name, violations: violations))
        }
        return manifest
    }

    /// The runnable repair for a `verify()` refusal (WP0 step 7).
    ///
    /// The common case dry run #1 hit — and could not escape — is the
    /// APPEARED-AFTER-ATTACH violation: `attach` pins `validationHash` as
    /// explicitly absent when no `validation.jsonl` exists, so authoring the
    /// file the vacuous-validation repair asks for turns the next `validate`
    /// into a pin violation. The repair is one command (`attach` re-pins), but
    /// nothing on the surface said so, and the freeze gate's own repair omitted
    /// it (§9, P5). It is named here, once, for every verb that verifies.
    static func pinDriftRepair(name: String, violations: [String]) -> String {
        let appeared = violations.compactMap { violation -> String? in
            guard violation.contains("appeared after attach"),
                let start = violation.range(of: "concept '"),
                let end = violation.range(
                    of: "'", range: start.upperBound ..< violation.endIndex)
            else { return nil }
            return String(violation[start.upperBound ..< end.lowerBound])
        }
        if !appeared.isEmpty {
            return "steerlab-cli experiment attach \(name) "
                + "\(appeared.joined(separator: " "))  "
                + "(re-pins the validation.jsonl that appeared after the "
                + "original attach), then steerlab-cli experiment validate \(name)"
        }
        return "steerlab-cli experiment verify \(name) "
            + "(names every drifted pin) ; then restore those files to their "
            + "pinned bytes, or steerlab-cli experiment duplicate \(name) "
            + "\(name)-v2 and re-pin on the copy"
    }

    static func makeRunDirectory(
        experiment: ExperimentManifest, task: String,
        sampling: EvaluateSubsample.Stamp? = nil
    ) throws -> URL {
        let url = try VectorCatalog.makeUniqueRunDirectory(
            slug: "exp-\(experiment.name)-\(task)",
            under: ExperimentStore.runsDirectory)

        // Stamp provenance: the full manifest snapshot + its content hash.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(experiment).write(
            to: url.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(experiment).write(
            to: url.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        // Canonical per-run config.json (uniform across run types).
        // Sampling-policy fields are stamped only for generation-bearing
        // tasks (study runs — incl. multi-agent — and sweeps); extract/
        // validate/evaluate/analyze do not sample by the manifest's policy,
        // so they stamp null rather than inventing one. Defaults mirror the
        // server manifest's (samplesPerItem 1, seedPolicy "manifestSeeds").
        let generates = task == "run" || task == "multi-agent-run" || task == "sweep"
        // Inert carried machinery is stamped, not just logged (2026-08-11):
        // a baseline+agents-only run must be self-describing from its
        // config.json alone — the cross-engine `notes` key
        // `inertConceptMachinery` (server twin stamps the structured form).
        let inertNote = task == "run"
            ? ExperimentStore.inertMachineryNote(experiment) : nil
        try RunMetadata.write(
            runType: task == "multi-agent-run" ? "multi-agent" : task,
            to: url,
            modelID: experiment.modelID,
            revision: experiment.modelRevision,
            experiment: experiment.name,
            experimentHash: ExperimentStore.manifestHash(experiment),
            temperature: generates ? experiment.temperature : nil,
            samplesPerItem: generates ? (experiment.samplesPerItem ?? 1) : nil,
            seedPolicy: generates ? (experiment.seedPolicy ?? "manifestSeeds") : nil,
            notes: inertNote.map { ["inertConceptMachinery": $0] } ?? [:],
            // The seeded evaluate subsample (2026-08-29), stamped so the run
            // is self-describing from its own config.json: a reader who finds
            // an evaluate directory holding a third of a corpus's codings
            // must be able to see WHY from the run itself, not only from the
            // command line that started it. Additive — absent means the full
            // corpus, which is every run written before this existed.
            structuredNotes: sampling.map { ["sampling": $0.jsonObject] } ?? [:])
        // WS7.1 loud, non-blocking study-run-start advisory: when this
        // experiment's scope-matched validate evidence came from the OTHER
        // engine, say so in the run log (stdout for headless runs) and
        // durably in the run directory — never a refusal (freeze already
        // enforces the same-substrate evidence gate; this catches the
        // workspace that moved engines after freezing).
        if task == "run" || task == "multi-agent-run",
            let advisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: experiment)
        {
            emitRunAdvisory(advisory, to: url)
        }
        return url
    }

    /// Log one non-blocking advisory and APPEND it to the run directory's
    /// `advisories.txt`.
    ///
    /// Append, not write: more than one advisory can be true of the same run
    /// (the cross-substrate check and the deprecated-caseFamily selection
    /// both fire at start), and the truncating write this replaced would have
    /// let the second silently erase the first. Best-effort throughout — an
    /// advisory must never sink a run. Server twin:
    /// `tasks._advise_dependency_lock_drift` / `_advise_implicit_case_family`.
    static func emitRunAdvisory(_ advisory: String, to runDirectory: URL) {
        print("ADVISORY: \(advisory)")
        let path = runDirectory.appending(component: "advisories.txt")
        let line = Data((advisory + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: path) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        } else {
            try? line.write(to: path)
        }
    }

    /// Loads task prompts with the drift checks the firewall promises: the
    /// pinned file must match its pinned hash at RUN time, not only at
    /// freeze; an override on a frozen study must be byte-identical to the
    /// pin (frozen means frozen — dev iteration belongs on a duplicate); and
    /// a frozen study cannot run on unpinned prompts at all. Mirrors the
    /// server's `_load_prompts`.
    static func loadTaskPrompts(
        for manifest: ExperimentManifest, override: String? = nil
    ) throws -> (file: String, hash: String, prompts: [StudyPrompt]) {
        let file = override ?? manifest.taskPromptsFile ?? "prompts/dev/dev-prompts.jsonl"
        let url =
            file.hasPrefix("/")
            ? URL(filePath: file)
            : VectorCatalog.projectRoot.appending(path: file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "task prompt file not found: \(url.path)",
                repair: "author \(file) as {\"id\": …, \"prompt\": …} JSONL rows, "
                    + "then steerlab-cli experiment pin-prompts "
                    + "\(manifest.name) \(file)")
        }
        let data = try Data(contentsOf: url)
        // SHA-256 over the raw bytes — identical to `StimulusSet.loadTexts`
        // and the server, so existing pinned hashes stay valid.
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let frozen = manifest.status != .draft
        if override == nil {
            if let pinned = manifest.taskPromptsHash, pinned != hash {
                throw ExperimentError.refusing(
                    .pinDrift,
                    "task prompts '\(file)' drifted from the pinned hash "
                        + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)",
                    repair: "restore \(file) to its pinned bytes ; then "
                        + "steerlab-cli experiment run \(manifest.name)  (on a "
                        + "DRAFT, re-pin instead: steerlab-cli experiment "
                        + "pin-prompts \(manifest.name) \(file))")
            }
            if frozen, manifest.taskPromptsHash == nil {
                // The prose keeps its shape (it already named the three steps);
                // the machine repair is the same three as one runnable line.
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "frozen study has no pinned task prompts — duplicate it "
                        + "('steerlab-cli experiment duplicate \(manifest.name) "
                        + "<new-name>'), pin the prompt set ('steerlab-cli experiment "
                        + "pin-prompts <new-name> prompts/…/file.jsonl'), and re-freeze",
                    repair: "steerlab-cli experiment duplicate \(manifest.name) "
                        + "\(manifest.name)-v2 && steerlab-cli experiment "
                        + "pin-prompts \(manifest.name)-v2 prompts/…/file.jsonl "
                        + "&& steerlab-cli experiment freeze \(manifest.name)-v2 "
                        + "&& steerlab-cli experiment run \(manifest.name)-v2")
            }
        } else if frozen, hash != manifest.taskPromptsHash {
            throw ExperimentError.refusing(
                .pinDrift,
                "prompt override on a FROZEN study must match the pinned prompt "
                    + "set byte-for-byte — duplicate the experiment to iterate",
                repair: "steerlab-cli experiment run \(manifest.name)  (without "
                    + "--prompts: the pin IS the measured task), or steerlab-cli "
                    + "experiment duplicate \(manifest.name) \(manifest.name)-v2 "
                    + "&& steerlab-cli experiment pin-prompts "
                    + "\(manifest.name)-v2 \(file)")
        }
        return (file, hash, try parseTaskPrompts(data))
    }

    /// Cross-engine duplicate-id refusal message (server twin: the same
    /// literal in `tasks._load_prompts`; fixture:
    /// `prompts/fixtures/task-prompts-validation/cases.json`). Duplicate item
    /// ids silently corrupt pairing on BOTH engines — the report's choice
    /// readouts and the paired effect sizes key on `promptID` — so the file
    /// is refused at LOAD, before any generation compute. Item numbers are
    /// 1-based ordinals in the parsed item list (blank lines don't count),
    /// identical on both engines.
    static func duplicateTaskPromptIDMessage(
        id: String, firstItem: Int, duplicateItem: Int
    ) -> String {
        "task prompts: duplicate item id '\(id)' (items \(firstItem) and "
            + "\(duplicateItem)) — ids must be unique for pairing and reporting"
    }

    /// Every duplicated item id with its 1-based item ordinals, in first-
    /// appearance order — the readiness layer's (`data check`) surfacing of
    /// the rule `parseTaskPrompts` refuses on: the parser stops at the FIRST
    /// duplicate; the checklist names them ALL so one edit fixes the file.
    /// Lenient about everything else (undecodable rows are other readiness
    /// rules' findings, and the parser refuses such files outright anyway).
    static func duplicateTaskPromptIDs(_ data: Data) -> [(id: String, items: [Int])] {
        struct IDLine: Decodable { let id: String? }
        let decoder = JSONDecoder()
        var positions: [String: [Int]] = [:]
        var order: [String] = []
        var count = 0
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for raw in lines {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty,
                let line = try? decoder.decode(IDLine.self, from: Data(trimmed.utf8))
            else { continue }
            count += 1
            let id = line.id ?? "prompt-\(count)"
            if positions[id] == nil { order.append(id) }
            positions[id, default: []].append(count)
        }
        return order.compactMap { id in
            guard let items = positions[id], items.count > 1 else { return nil }
            return (id: id, items: items)
        }
    }

    /// Cross-engine refusal for an item whose `factors` is not the flat
    /// string→string object the factorial generator emits (server twin: the
    /// same literal in `tasks._load_prompts`).
    static func taskPromptFactorsMessage(itemID: String) -> String {
        "task prompts: item '\(itemID)' has a 'factors' value that is not "
            + "a flat string-to-string object — factor names and level "
            + "names must both be strings"
    }

    /// JSONL task items: `{"prompt": …}` (server style) or legacy
    /// `{"text": …}`, plus the optional science-layer keys (`options`,
    /// `target`, `anchorMonths`, `severity`, `arm`, `caseID`, `factors` —
    /// exact server names) and the optional scripted `transcript`
    /// (schema-validated here
    /// on BOTH engines with identical messages; `text`/`prompt` becomes
    /// optional when a transcript is present — the display text derives from
    /// the final user turn). Item ids — explicit or auto-numbered — must be
    /// unique (identical refusal on both engines; every task-prompt consumer
    /// — run, validate, sweep, logprob, pin — inherits the gate from this
    /// parser). Split from file IO so the parse is unit-testable.
    static func parseTaskPrompts(_ data: Data) throws -> [StudyPrompt] {
        struct AttentionCheckLine: Decodable {
            let expected: String?
            let grading: String?
        }
        struct Line: Decodable {
            let id: String?
            let prompt: String?
            let text: String?
            let options: [String]?
            let target: String?
            let anchorMonths: Double?
            let severity: Double?
            let arm: String?
            let caseID: String?
            let transcript: [TranscriptTurn]?
            let attentionCheck: AttentionCheckLine?
            let responseFormat: String?
        }
        let decoder = JSONDecoder()
        var prompts: [StudyPrompt] = []
        var seenIDs: [String: Int] = [:]  // id → 1-based item ordinal
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard
                let line = try? decoder.decode(Line.self, from: Data(trimmed.utf8)),
                line.prompt ?? line.text != nil || line.transcript != nil
            else {
                throw ExperimentError(
                    reason: "malformed task prompt JSONL at line \(index + 1)")
            }
            // Explicit ids must be non-empty; null/absent take the shared
            // prompt-<ordinal> fallback (review 2026-08-03, P2 — message
            // string is the cross-engine contract; server twin:
            // _load_prompts).
            let id: String
            if let declared = line.id {
                guard !declared.trimmingCharacters(in: .whitespaces).isEmpty
                else {
                    throw ExperimentError(
                        reason: "task prompts: item \(prompts.count + 1) "
                            + "declares an empty or non-string 'id' — "
                            + "declare a non-empty string, or omit the key "
                            + "for the prompt-<ordinal> fallback")
                }
                id = declared
            } else {
                id = "prompt-\(prompts.count + 1)"
            }
            // Duplicate ids (explicit or auto-collided) refuse BEFORE the
            // per-item transcript checks — cross-engine ordering contract.
            if let firstItem = seenIDs[id] {
                throw ExperimentError(
                    reason: duplicateTaskPromptIDMessage(
                        id: id, firstItem: firstItem,
                        duplicateItem: prompts.count + 1))
            }
            seenIDs[id] = prompts.count + 1
            if let transcript = line.transcript {
                if let violation = transcriptSchemaViolation(transcript, itemID: id) {
                    throw ExperimentError(reason: violation)
                }
            }
            // Per-item attention check (the exclusion instrument's first
            // user), validated at LOAD with plain-language, cross-engine-
            // identical messages; items without a check are untouched.
            var attentionCheck: AttentionCheck?
            if let check = line.attentionCheck {
                if let violation = ExclusionEngine.attentionCheckViolation(
                    expected: check.expected, grading: check.grading, itemID: id)
                {
                    throw ExperimentError(reason: violation)
                }
                attentionCheck = AttentionCheck(
                    expected: check.expected ?? "",
                    grading: check.grading.flatMap(
                        CapabilityBattery.GradingMode.init(rawValue:)))
            }
            // Factorial cell metadata (the generator's `factors` object):
            // validated as a flat string→string map at LOAD via the raw
            // JSON (Codable can't distinguish wrong-shape from absent) —
            // identical message on the server. Empty ⇒ treated as absent.
            var factors: [String: String]?
            if let object = try? JSONSerialization.jsonObject(
                with: Data(trimmed.utf8)) as? [String: Any],
                let rawFactors = object["factors"]
            {
                guard let typed = rawFactors as? [String: String] else {
                    throw ExperimentError(
                        reason: taskPromptFactorsMessage(itemID: id))
                }
                if !typed.isEmpty { factors = typed }
            }
            // Closed vocabulary, validated at LOAD: an unrecognised value
            // must refuse rather than degrade to "unspecified", which would
            // re-open the hole `ResponseFormat` closes (a typo silently
            // restoring permissive behaviour).
            let responseFormat: ResponseFormat?
            do {
                responseFormat = try ResponseFormat.parse(line.responseFormat)
            } catch {
                throw ExperimentError(
                    reason: "task prompt '\(id)': \(error)")
            }
            let text =
                line.prompt ?? line.text
                ?? line.transcript.map(transcriptDisplayText) ?? ""
            prompts.append(
                StudyPrompt(
                    id: id,
                    text: text,
                    options: line.options,
                    target: line.target,
                    anchorMonths: line.anchorMonths,
                    severity: line.severity,
                    arm: line.arm,
                    caseID: line.caseID,
                    transcript: line.transcript,
                    attentionCheck: attentionCheck,
                    factors: factors,
                    responseFormat: responseFormat))
        }
        return prompts
    }

    /// How many of the manifest's pinned items are CHOICE-shaped (WP0 step 7).
    ///
    /// Tolerant by design: nil when the file is absent, unpinned, or
    /// unparseable. Its callers are advisory paths — the sweep's
    /// defaulted-criterion note and `pin-prompts`' missing-instrument note —
    /// and an advisory that guesses is worse than one that stays quiet.
    static func choiceShapedItemCount(
        _ manifest: ExperimentManifest
    ) -> (choice: Int, total: Int)? {
        guard let loaded = try? loadTaskPrompts(for: manifest) else { return nil }
        return choiceShapedItemCount(loaded.prompts)
    }

    static func choiceShapedItemCount(
        _ prompts: [StudyPrompt]
    ) -> (choice: Int, total: Int) {
        (
            choice: prompts.filter { !($0.options ?? []).isEmpty }.count,
            total: prompts.count
        )
    }

    /// The `ResponseFormat` view of loaded prompts — the pure rule's input.
    static func responseFormatItems(_ prompts: [StudyPrompt]) -> [ResponseFormat.Item] {
        prompts.map {
            .init(
                id: $0.id,
                hasOptions: !($0.options ?? []).isEmpty,
                hasTarget: $0.target?.isEmpty == false,
                format: $0.responseFormat)
        }
    }

    /// Run-start gate: the declared option-consuming instruments must be able
    /// to read the items they will be pointed at, and any declared
    /// applicability scope must still select the items it was pinned to.
    /// Server twin: `tasks._check_response_formats`.
    static func checkResponseFormats(
        _ prompts: [StudyPrompt], manifest: ExperimentManifest
    ) throws {
        // The vocabulary gate, FIRST and item-independent: a declaration this
        // engine cannot dispatch is a worse failure than one it can dispatch
        // at the wrong items, because it produces no error at all. Same gate
        // as the rest of this function — `responseFormat` is the one gate
        // whose subject is `outcomeInstruments` and whose repair is
        // `set-instruments`, and its existing family is precisely "a declared
        // instrument that would silently produce zero records" (the zero-item
        // rules, 2026-08-06). Server twin: `tasks._check_response_formats`.
        if let unknown = ExperimentStore.unknownOutcomeInstrumentProblem(manifest) {
            throw ExperimentError.refusing(
                .responseFormat, unknown,
                repair: ExperimentStore.unknownOutcomeInstrumentRepair(
                    manifest.name))
        }
        let items = responseFormatItems(prompts)
        if let scope = manifest.outcomeInstrumentScope,
            let drift = scope.driftRefusal(items: items)
        {
            throw ExperimentError.refusing(
                .responseFormat, drift,
                repair: "steerlab-cli experiment pin-prompts \(manifest.name) "
                    + "<the item file the declared scope was pinned against>, "
                    + "or re-declare the scope on a duplicate: steerlab-cli "
                    + "experiment duplicate \(manifest.name) "
                    + "\(manifest.name)-v2")
        }
        if let refusal = ResponseFormat.refusal(
            items: items,
            declaredInstruments: manifest.outcomeInstruments,
            declaredScope: manifest.outcomeInstrumentScope)
        {
            throw ExperimentError.refusing(
                .responseFormat, refusal,
                repair: "steerlab-cli experiment set-instruments "
                    + "\(manifest.name) sampledText  (score the prose "
                    + "instead), or re-author the items with "
                    + "\"responseFormat\": \"label\" and re-pin: steerlab-cli "
                    + "experiment pin-prompts \(manifest.name) "
                    + (manifest.taskPromptsFile ?? "prompts/…/file.jsonl"))
        }
    }

    /// Submit-time surfacing of the scope-drift refusal — the pure rule,
    /// split from the file loading so it is testable without a workspace.
    /// Run-verb submissions only: the gate is a run-stage rule, and
    /// extract/validate/sweep submissions never read the task prompts this
    /// way (a pipeline's run stage is covered by the server's own
    /// pre-model-load preflight, which this check mirrors).
    static func scopeDriftSubmissionRefusal(
        verb: String,
        studyKind: ExperimentManifest.StudyKind,
        scope: ResponseFormat.Scope?,
        items: [ResponseFormat.Item],
        taskPromptsFile: String
    ) -> String? {
        guard verb == "run", studyKind != .multiAgent, let scope,
            let drift = scope.driftRefusal(items: items)
        else { return nil }
        return drift
            + " — re-declare the instrument scope against the new task file "
            + "('\(taskPromptsFile)') and resubmit"
    }

    /// Filesystem half of the submit-time guard (2026-08-06 field incident:
    /// Duplicate & Adjust plus a task-file swap left a stale
    /// `outcomeInstrumentScope` pin, and a 4-shard Slurm submission spent
    /// ~2.5 minutes per shard staging and loading gemma-3-27b-it before the
    /// server's run-stage gate refused — the refusal was right, the ordering
    /// was not). The server now refuses before its model load; this moves
    /// the same refusal all the way forward to the submit button, before
    /// packaging or upload. A prompt file that cannot be loaded locally
    /// returns nil: the server's own load reports that failure with its
    /// proper message.
    public static func scopeDriftSubmitRefusal(
        for manifest: ExperimentManifest, verb: String
    ) -> String? {
        guard verb == "run", manifest.studyKind != .multiAgent,
            manifest.outcomeInstrumentScope != nil,
            let loaded = try? loadTaskPrompts(for: manifest)
        else { return nil }
        return scopeDriftSubmissionRefusal(
            verb: verb,
            studyKind: manifest.studyKind,
            scope: manifest.outcomeInstrumentScope,
            items: responseFormatItems(loaded.prompts),
            taskPromptsFile: loaded.file)
    }

    /// promptID → declared attention check, from loaded task items (the
    /// exclusion engine's join input; server twin `attention_checks`).
    static func attentionChecks(of prompts: [StudyPrompt]) -> [String: AttentionCheck] {
        var checks: [String: AttentionCheck] = [:]
        for prompt in prompts {
            if let check = prompt.attentionCheck { checks[prompt.id] = check }
        }
        return checks
    }

    /// The exclusion engine's view of one sampled record: pairing identity,
    /// the output (attention-check grading), and the record-level parsed
    /// endpoints with their present-vs-null distinction preserved.
    static func exclusionView(of record: GenerationRecord) -> ExclusionEngine.RecordView {
        var endpoints: [String: Double?] = [:]
        if case .some(let parsed) = record.parsedMonths {
            // `updateValue`, never subscript-assign: the key must appear
            // with a nil VALUE on a parse failure (JSON null), and the
            // subscript would remove it instead.
            endpoints.updateValue(parsed, forKey: "parsedMonths")
        }
        return ExclusionEngine.RecordView(
            condition: record.condition, seed: record.seed,
            promptID: record.promptID, output: record.output,
            endpoints: endpoints)
    }

    /// Record-level parsed endpoints for the analyze path, read from the raw
    /// JSONL line so the present-with-null vs absent distinction survives
    /// (Codable's synthesized decoding collapses `null` to absent). Only the
    /// endpoints the declared rules actually read are extracted; a
    /// non-numeric value is treated as not-applicable, matching the server.
    static func analysisEndpoints(
        jsonLine: Data, names: Set<String>
    ) -> [String: Double?] {
        guard !names.isEmpty,
            let object = try? JSONSerialization.jsonObject(with: jsonLine),
            let record = object as? [String: Any]
        else { return [:] }
        var endpoints: [String: Double?] = [:]
        for name in names {
            guard let value = record[name] else { continue }
            if value is NSNull {
                // `updateValue`, never subscript-assign: the key must appear
                // with a nil VALUE (a parse failure); the subscript would
                // remove it instead.
                endpoints.updateValue(nil, forKey: name)
            } else if let number = value as? NSNumber,
                CFGetTypeID(number) != CFBooleanGetTypeID()
            {
                endpoints.updateValue(number.doubleValue, forKey: name)
            }
        }
        return endpoints
    }

    /// The run-inline exclusion outcome: nil when the manifest declares no
    /// rules (today's behavior exactly); otherwise the stamp + excluded
    /// record keys (sampled rows AND instrument readouts — scope
    /// allRecordTypes) the report's effect sizes apply. Rules were validated
    /// at run start (`ExclusionEngine.preflight`), so evaluation cannot
    /// fail here.
    static func exclusionOutcome(
        manifest: ExperimentManifest, prompts: [StudyPrompt],
        views: [ExclusionEngine.RecordView],
        instrumentViews: [ExclusionEngine.InstrumentRecordView] = []
    ) -> ExclusionEngine.Outcome? {
        guard let rules = manifest.exclusionRules, !rules.isEmpty else { return nil }
        return ExclusionEngine.evaluate(
            rules: rules, checks: attentionChecks(of: prompts), views: views,
            instrumentViews: instrumentViews)
    }

    /// Loads the experiment's model at its pinned revision. A draft with no
    /// pin yet gets the local cache's resolved commit written back into the
    /// manifest, so the revision that actually ran is recorded before any
    /// artifacts exist. (Frozen manifests are immutable: a legacy one with
    /// no pin runs whatever the cache holds, loudly.)
    static func loadContainer(
        pinning manifest: inout ExperimentManifest
    ) async throws -> ModelContainer {
        let container = try await SteeredContainerLoader.load(
            modelID: manifest.modelID, revision: manifest.modelRevision)
        if manifest.modelRevision == nil,
            let resolved = SteeredContainerLoader.cachedRevision(for: manifest.modelID)
        {
            if manifest.status == .draft {
                manifest.modelRevision = resolved
                try ExperimentStore.save(manifest)
                print("pinned model revision \(resolved.prefix(12))… (local HF cache)")
            } else {
                print(
                    "⚠︎ '\(manifest.name)' is \(manifest.status.rawValue) without a "
                        + "pinned model revision — this run used \(resolved.prefix(12))…")
            }
        }
        return container
    }

    /// Loads (and pin-checks) the neutral corpus whenever the manifest pins
    /// one. The corpus is the norm-unit denominator (emotion paper: a fixed
    /// dataset, so α is denominated identically across concepts) and, when
    /// requested, the confound-projection basis. Unpinned ⇒ nil ⇒ legacy
    /// stimulus-based norms (recorded as such in sidecars).
    static func neutralTexts(for manifest: ExperimentManifest) throws -> [String]? {
        guard let pinned = manifest.neutralCorpusHash else {
            guard
                !manifest.concepts.contains(where: { ($0.options.neutralPCCount ?? 0) > 0 })
            else {
                throw ExperimentError(
                    reason: "confound projection requested but no neutral corpus pinned")
            }
            return nil
        }
        let url = VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral", "corpus.jsonl")
        let (texts, hash) = try StimulusSet.loadTexts(url: url)
        guard pinned == hash else {
            throw ExperimentError(reason: "neutral corpus drifted from the pinned hash")
        }
        return texts
    }

    /// One re-derived concept vector set. `stimuli` is present for paired
    /// concepts only; grand-mean concepts extract from the shared pinned
    /// corpus, not a positive/negative directory.
    struct ConceptExtraction {
        let result: ExtractionResult
        let stimuli: StimulusSet?
    }

    /// Re-derives all concept vectors from the pinned recipes, asserting the
    /// stimulus hashes still match (deterministic re-extraction is the
    /// pinning model the project chose over pinned vector bytes).
    ///
    /// `cancel` polls BETWEEN concepts (one concept's extraction is the unit
    /// of work); on observation the partial dictionary is returned and the
    /// CALLER re-polls to detect the cancellation — the flag is one-way for
    /// the operation, so the re-poll is reliable.
    static func extractAll(
        manifest: ExperimentManifest, container: ModelContainer, into runDirectory: URL?,
        cancel: CancelPoller = CancelPoller(nil)
    ) async throws -> [String: ConceptExtraction] {
        var results: [String: ConceptExtraction] = [:]
        // Artifact-pinned concepts (method pinnedArtifact — incl. every
        // OptVec direction) MATERIALIZE from verified bytes instead of
        // re-deriving, and that materialization is implemented on the
        // server engine only today. Refuse up front: silently skipping the
        // concept here would surface later as a baffling "missing vector"
        // at injection time.
        if let pinned = manifest.concepts.first(where: {
            $0.options.method == .pinnedArtifact
        }) {
            throw ExperimentError(
                reason: "concept '\(pinned.name)' is artifact-pinned "
                    + "(method pinnedArtifact) — materialization from pinned "
                    + "bytes runs on the Python server engine only; run this "
                    + "study on the server (bundle submit), not local MLX")
        }
        let neutral = try neutralTexts(for: manifest)
        for ref in manifest.concepts where ref.options.method.isPaired {
            if await cancel.observed(at: "extraction of '\(ref.name)'") {
                return results
            }
            let directory = VectorCatalog.conceptsDirectory.appending(component: ref.name)
            let stimuli = try StimulusSet(directory: directory)
            guard stimuli.hash == ref.stimulusSetHash else {
                throw ExperimentError(
                    reason: "concept '\(ref.name)': stimuli drifted from the pinned hash")
            }
            print("extracting \(ref.name) (\(stimuli.positive.count)+\(stimuli.negative.count) stimuli)…")
            let extraction = try await ConceptExtractor.extract(
                container: container, stimuli: stimuli, options: ref.options,
                neutralTexts: neutral)
            if let runDirectory {
                try saveVectorSidecar(
                    manifest: manifest, ref: ref,
                    stimulusSetHash: stimuli.hash, extraction: extraction,
                    to: runDirectory)
            }
            results[ref.name] = ConceptExtraction(result: extraction, stimuli: stimuli)
        }
        for ref in manifest.concepts
        where ref.options.method == .designatedReference {
            if await cancel.observed(at: "extraction of '\(ref.name)'") {
                return results
            }
            guard let pin = ref.designatedReference else {
                throw ExperimentError(
                    reason: "designated-reference concept '\(ref.name)' has no "
                        + "pinned reference — re-attach with a reference")
            }
            let positive = try ExperimentStore.loadStoriesTexts(for: ref.name)
            let negative = try ExperimentStore.loadStoriesTexts(for: pin.name)
            let stimuli = StimulusSet(
                name: ref.name, positive: positive, negative: negative,
                hash: ref.stimulusSetHash)
            guard ExperimentStore.storiesHash(for: ref.name) == ref.stimulusSetHash
            else {
                throw ExperimentError(
                    reason: "concept '\(ref.name)': stories drifted from the pinned hash")
            }
            print(
                "extracting \(ref.name) (\(positive.count) stories − "
                    + "\(negative.count) reference '\(pin.name)')…")
            let extraction = try await ConceptExtractor.extract(
                container: container, stimuli: stimuli, options: ref.options,
                neutralTexts: neutral)
            if let runDirectory {
                try saveVectorSidecar(
                    manifest: manifest, ref: ref,
                    stimulusSetHash: ref.stimulusSetHash, extraction: extraction,
                    designatedReference: ["name": pin.name, "hash": pin.hash],
                    to: runDirectory)
            }
            results[ref.name] = ConceptExtraction(result: extraction, stimuli: stimuli)
        }
        let grandMean = try await extractGrandMeanConcepts(
            manifest: manifest, container: container, neutralTexts: neutral,
            into: runDirectory, cancel: cancel)
        results.merge(grandMean) { _, new in new }
        return results
    }

    private static func saveVectorSidecar(
        manifest: ExperimentManifest, ref: ExperimentManifest.ConceptRef,
        stimulusSetHash: String, extraction: ExtractionResult,
        grandMeanPopulation: [String: String]? = nil,
        designatedReference: [String: String]? = nil, to runDirectory: URL
    ) throws {
        let normSource =
            extraction.residualNormSource == "neutral-corpus"
            ? "neutral-corpus \(manifest.neutralCorpusHash?.prefix(12) ?? "")"
            : extraction.residualNormSource
        var sidecar = SteeringVectorSidecar(
            modelID: manifest.modelID, revision: manifest.modelRevision,
            concept: ref.name,
            stimulusSetHash: stimulusSetHash, vectors: extraction.vectors,
            options: extraction.options,
            residualNormPerLayer: extraction.residualNormPerLayer,
            residualNormSource: normSource,
            residualNormConvention: extraction.residualNormConvention,
            residualNormRendering: extraction.residualNormRendering,
            readingPositionResolution: extraction.readingPositionResolution,
            // The FULL corpus hash (the identity contract needs it; the
            // legacy 12-char prefix embedded in normSource stays for
            // display compatibility but proves nothing).
            neutralCorpusHash: manifest.neutralCorpusHash)
        sidecar.grandMeanPopulation = grandMeanPopulation
        sidecar.designatedReference = designatedReference
        // Stamp the canonical full-recipe identity from the sidecar's own
        // recorded fields — the stamp always describes THIS artifact, and
        // stamping exercises the same reader promotion uses. An extraction
        // writer that cannot prove its own recipe is a writer bug and must
        // fail loudly, never write an unprovable artifact.
        let candidate = RecipeIdentity.candidate(sidecar: sidecar)
        guard let components = candidate.components else {
            throw ExperimentError(
                reason: "extraction sidecar for '\(ref.name)' is missing recipe "
                    + "fields [\(candidate.missingFields.joined(separator: ", "))] "
                    + "— cannot stamp recipeIdentityHash (writer bug)")
        }
        sidecar.recipeIdentityHash = RecipeIdentity.hash(components)
        try SteeringVectorStore.save(
            vectors: extraction.vectors, sidecar: sidecar,
            to: runDirectory, name: ref.name,
            neutralMeanPerLayer: extraction.neutralMeanPerLayer)
    }

    /// Grand-mean concepts share one pinned population; every target that
    /// shares (reading position, projection) extracts in a single corpus
    /// pass so the denominator is computed once, exactly as the recipe
    /// defines it (server `_extract_grand_mean_bundles` twin).
    private static func extractGrandMeanConcepts(
        manifest: ExperimentManifest, container: ModelContainer,
        neutralTexts: [String]?, into runDirectory: URL?,
        cancel: CancelPoller = CancelPoller(nil)
    ) async throws -> [String: ConceptExtraction] {
        let refs = manifest.concepts.filter { $0.options.method.isGrandMean }
        guard !refs.isEmpty else { return [:] }
        guard let corpus = manifest.grandMeanCorpus else {
            throw ExperimentError(
                reason: "grand-mean concepts attached but no grandMeanCorpus pinned — "
                    + "re-attach with method emotionGrandMean")
        }
        var rows: [StimulusSet.MultiConceptStimulus] = []
        var liveHashes: [String: String] = [:]
        for member in corpus.concepts {
            // Missing/drifted members are verify() violations; extraction
            // reads what exists, like the server's load_corpus.
            guard
                let loaded = try? StimulusSet.loadMultiConceptTexts(
                    url: ExperimentStore.storiesURL(for: member))
            else { continue }
            liveHashes[member] = loaded.hash
            rows.append(contentsOf: loaded.rows)
        }
        guard !rows.isEmpty else {
            throw ExperimentError(reason: "grand-mean corpus is empty on disk")
        }

        struct GroupKey: Hashable {
            let readingPosition: String
            let neutralPCCount: Int?
            let extractionRendering: String
        }
        var groups: [GroupKey: [ExperimentManifest.ConceptRef]] = [:]
        var order: [GroupKey] = []
        for ref in refs {
            // The RENDERING joins the grouping key: two concepts that render
            // differently are two different corpus passes with two different
            // denominators, so pooling them would silently give one of them
            // the other's numbers.
            let key = GroupKey(
                readingPosition: ref.options.readingPosition.label,
                neutralPCCount: ref.options.neutralPCCount,
                extractionRendering: ref.options.resolvedExtractionRendering.label)
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(ref)
        }

        var results: [String: ConceptExtraction] = [:]
        for key in order {
            guard let members = groups[key], let first = members.first else { continue }
            if await cancel.observed(
                at: "grand-mean extraction group (\(members.map(\.name).joined(separator: "+")))")
            {
                return results
            }
            print(
                "extracting grand-mean group "
                    + "(\(members.map(\.name).joined(separator: "+")); "
                    + "\(rows.count) corpus rows)…")
            let extraction = try await ConceptExtractor.extractGrandMean(
                container: container, corpus: rows,
                targetConcepts: Set(members.map(\.name)),
                readingPosition: first.options.readingPosition,
                neutralPCCount: first.options.neutralPCCount,
                neutralTexts: neutralTexts,
                extractionRendering: first.options.resolvedExtractionRendering)
            for ref in members {
                guard let vectors = extraction.vectorsByConcept[ref.name] else {
                    throw ExperimentError(
                        reason: "grand-mean concept '\(ref.name)' has no rows in the "
                            + "pinned corpus")
                }
                let result = ExtractionResult(
                    vectors: vectors,
                    residualNormPerLayer: extraction.residualNormPerLayer,
                    residualNormSource: extraction.residualNormSource,
                    options: ref.options,
                    // Carried, never re-defaulted: a grand-mean denominator may
                    // have come from the token bank's per-position tally, and
                    // relabelling it as the per-text rule here is the exact
                    // stamp dishonesty F1 names.
                    residualNormConvention: extraction.residualNormConvention,
                    residualNormRendering: extraction.residualNormRendering,
                    readingPositionResolution: extraction.readingPositionResolution,
                    neutralMeanPerLayer: extraction.neutralMeanPerLayer)
                if let runDirectory {
                    // Live hashes (like the paired path's stimuli.hash): the
                    // sidecar records what was actually read — including the
                    // FULL population the grand mean was computed over;
                    // verify() reports drift.
                    try saveVectorSidecar(
                        manifest: manifest, ref: ref,
                        stimulusSetHash: liveHashes[ref.name] ?? ref.stimulusSetHash,
                        extraction: result, grandMeanPopulation: liveHashes,
                        to: runDirectory)
                }
                results[ref.name] = ConceptExtraction(result: result, stimuli: nil)
            }
        }
        return results
    }

    // MARK: - extract

    public static func extract(
        experimentName: String,
        shouldCancel: (@Sendable () async -> Bool)? = nil
    ) async throws {
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        var manifest = try loadVerified(experimentName)
        let cancel = CancelPoller(shouldCancel)
        let container = try await loadContainer(pinning: &manifest)
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "extract")
        let results = try await extractAll(
            manifest: manifest, container: container, into: runDirectory,
            cancel: cancel)
        if await cancel.observed(at: "after extraction") {
            // Concepts extracted so far have real sidecar artifacts; the
            // note marks the directory as incomplete — never a fake
            // completion.
            writeCancellationNote(task: "extract run", to: runDirectory)
            print(
                "extract cancelled by user — \(results.count) of "
                    + "\(manifest.concepts.count) concept(s) extracted; partial "
                    + "artifacts kept in \(runDirectory.lastPathComponent)")
            return
        }
        for (name, extraction) in results.sorted(by: { $0.key < $1.key }) {
            let vectors = extraction.result.vectors
            print(
                "\(name): \(vectors.layerCount) layers, hidden \(vectors.hiddenSize), "
                    + "norm @ mid \(vectors.norm(at: vectors.layerCount / 2))")
        }
        print("run artifacts: \(runDirectory.path)")
    }

    // MARK: - validate

    /// The per-concept resolution `validate` performs BEFORE it measures
    /// anything: whether there is a held-out probe to run at all, which
    /// method's semantics score it, and whose DATA it reads.
    ///
    /// Twinned to the head of the server's `_validate_impl` concept loop
    /// (`concept.effective_method`, `concept.data_concept`, the
    /// `has_source_concept` skip, and the exhaustive-semantics refusal). The
    /// three questions travel TOGETHER by construction — resolving the data
    /// concept without also resolving the effective method changes which
    /// file the paired/story lookup reads, which is exactly how the two
    /// engines drifted (open-issues §21).
    ///
    /// Both engines' `verify()` already pin a concept's stimulus and
    /// validation hashes from `dataConcept` under the EFFECTIVE method, so
    /// this is what makes the evidence a `validate` run produces describe
    /// the same bytes the freeze gate pins.
    public enum ValidationProbePlan: Sendable, Equatable {
        /// Nothing to validate. A direction with no source concept was never
        /// read OFF a concept's stimuli: an OptVec vector (evidence: the
        /// eval run's `eval.json`, OptVec plan §6) and an imported Gemma
        /// Scope SAE decoder row (evidence: the pinned candidate roster's
        /// discovery snapshot + qualification artifact) have no stimuli, no
        /// class means and no held-out `validation.jsonl`. Skipped rather
        /// than refused so a MIXED study still validates its ordinary
        /// concepts; the vacuity ledger excludes it for the same reason
        /// (`ExperimentStore.owesHeldOutProbe`), so a skip is never recorded
        /// as vacuous evidence.
        case skipped(reason: String)
        /// A probe to run, with the identities it runs under.
        case probe(Probe)

        public struct Probe: Sendable, Equatable {
            /// The concept whose stimuli and held-out `validation.jsonl` the
            /// probe reads — an artifact-pinned direction keeps the base
            /// concept's data under a new study-side name.
            public let dataConcept: String
            /// The DATA method: an artifact-pinned concept answers with its
            /// recorded SOURCE method, everything else with its declared one.
            public let method: ExtractionMethod

            public init(dataConcept: String, method: ExtractionMethod) {
                self.dataConcept = dataConcept
                self.method = method
            }

            /// Scored against the concept-vs-population midpoint rather than
            /// two class means.
            public var isGrandMean: Bool { method.isGrandMean }
            /// Which recipe root is CANONICAL for the held-out set (server:
            /// `paired=not method.uses_story_corpus`).
            public var validationIsPaired: Bool { !method.usesStoryCorpus }
        }
    }

    /// Resolves one pinned concept's `ValidationProbePlan`. Pure — no model,
    /// no filesystem — so both engines' agreement is unit-testable against a
    /// committed manifest fixture.
    public static func validationProbePlan(
        for ref: ExperimentManifest.ConceptRef
    ) throws -> ValidationProbePlan {
        // An unknown recorded source method is a violation, never a
        // fallback: this engine cannot say where the concept's held-out data
        // lives, so it must not guess. (`verify()` refuses the same manifest
        // under the same reasoning — validate only ever sees one because a
        // caller skipped verification.)
        guard let method = ref.effectiveMethod else {
            throw ExperimentError(
                reason: "concept '\(ref.name)' pins an artifact whose "
                    + "sourceMethod "
                    + "'\(ref.vectorArtifact?.sourceMethod ?? "")' this "
                    + "engine does not know — validate cannot resolve what "
                    + "its held-out probe would mean")
        }
        guard method.hasSourceConcept else {
            return .skipped(
                reason: "concept '\(ref.name)': method '\(method.rawValue)' "
                    + "has no source concept — no stimuli, no class means and "
                    + "no held-out validation.jsonl, so there is no probe to "
                    + "run. Its evidence lives in its own artifacts (an "
                    + "OptVec direction's eval run; an imported SAE feature's "
                    + "discovery snapshot + qualification), not in a "
                    + "concept's held-out set")
        }
        // Branch on what validation MEANS: contrastive (two class means) vs
        // population (grand mean). A method that is neither refuses loudly
        // instead of falling into whichever branch is syntactically last
        // (review 2026-07-31 round 2, finding 2). Asked of the EFFECTIVE
        // method — `pinnedArtifact` itself declares no semantics, and the
        // total `validationSemantics` property would answer `.population`
        // for it, which is a silent wrong answer.
        guard method.usesContrastiveValidation || method.isGrandMean else {
            throw ExperimentError(
                reason: "concept '\(ref.name)': method '\(method.rawValue)' "
                    + "declares no validation semantics (neither contrastive "
                    + "nor grand-mean)")
        }
        return .probe(
            ValidationProbePlan.Probe(
                dataConcept: ref.dataConcept, method: method))
    }

    /// `log` mirrors the CLI's printed progress line-for-line so panel
    /// clients can stream the same evidence trail into a live display; nil
    /// keeps the print-only CLI behavior.
    @discardableResult
    public static func validate(
        experimentName: String,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        log: (@Sendable (String) async -> Void)? = nil
    ) async throws -> URL {
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        var manifest = try loadVerified(experimentName)
        let cancel = CancelPoller(shouldCancel, log: log)
        /// A cancelled validation writes the status note and returns the run
        /// directory WITHOUT a validation report or evidence file — it can
        /// never satisfy the freeze gate, mechanically.
        @Sendable func cancelledValidation(_ runDirectory: URL) async -> URL {
            writeCancellationNote(task: "validation run", to: runDirectory)
            let line = "validation cancelled by user — partial artifacts kept in "
                + "\(runDirectory.lastPathComponent); NO validation evidence was written"
            print(line)
            await log?(line)
            return runDirectory
        }
        await log?("verified pins for '\(manifest.name)' — loading \(manifest.modelID)…")
        // Variant studies validate the capability battery per condition; the
        // battery is pinned into a draft manifest BEFORE the run directory is
        // made so the evidence's manifest snapshot carries the pin (the
        // validation scope hash includes it when variants are present).
        if !manifest.variantConditions.isEmpty, manifest.capabilityBatteryFile == nil,
            manifest.status == .draft
        {
            if let battery = ExperimentStore.pinCapabilityBattery(into: &manifest) {
                try ExperimentStore.save(manifest)
                let line = "pinned capability battery \(battery.file) "
                    + "@ \(battery.hash.prefix(12))…"
                print(line)
                await log?(line)
            }
        }
        let container = try await loadContainer(pinning: &manifest)
        // The run dir is made AFTER revision pinning: its manifest snapshot
        // is what freeze later accepts as validation evidence, so it must
        // record the exact revision this validation ran against.
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "validate")
        await log?("validation artifacts → \(runDirectory.lastPathComponent)")
        await log?("re-deriving vectors for \(manifest.concepts.count) pinned concept"
            + (manifest.concepts.count == 1 ? "" : "s") + "…")
        let extractions = try await extractAll(
            manifest: manifest, container: container, into: runDirectory,
            cancel: cancel)
        if await cancel.observed(at: "after extraction") {
            return await cancelledValidation(runDirectory)
        }

        var report: [String: Any] = ["experiment": manifest.name]

        // 1. Never-named validation scenarios (convergent validity).
        // Grand-mean corpus activations are computed once per reading
        // position and shared across that position's concepts (server
        // `_corpus_activations` twin).
        var corpusActivationCache: [String: (concepts: [String], activations: StimulusActivations)] =
            [:]
        func corpusActivations(
            _ reading: ReadingPosition, _ rendering: ExtractionRendering
        ) async throws -> (concepts: [String], activations: StimulusActivations) {
            // The rendering joins the cache key because it changes the token
            // sequence, hence the activations.
            let cacheKey = "\(reading.label) | \(rendering.label)"
            if let cached = corpusActivationCache[cacheKey] { return cached }
            guard let corpus = manifest.grandMeanCorpus else {
                throw ExperimentError(
                    reason: "grand-mean concepts attached but no grandMeanCorpus pinned — "
                        + "re-attach with method emotionGrandMean")
            }
            var rows: [StimulusSet.MultiConceptStimulus] = []
            for member in corpus.concepts {
                guard
                    let loaded = try? StimulusSet.loadMultiConceptTexts(
                        url: ExperimentStore.storiesURL(for: member))
                else { continue }
                rows.append(contentsOf: loaded.rows)
            }
            let computed = try await ConceptExtractor.multiConceptActivations(
                container: container, corpus: rows, readingPosition: reading,
                rendering: rendering)
            corpusActivationCache[cacheKey] = computed
            return computed
        }

        // Range refusal BEFORE any measurement. Depth is known the moment
        // extraction returns; checking later meant scenario accuracy was
        // already computed at a CLAMPED layer the researcher never
        // declared, and on the server a cosine matrix was persisted at it.
        try requireValidationLayerInRange(
            manifest,
            layerCount: try requireUniformDepth(extractions))
        var validation: [String: Any] = [:]
        // Concepts that OWED a held-out probe and got none — vacuous
        // evidence (2026-08-17 firewall repair). Stamped into
        // validation-evidence.json, where freeze's validateEvidence gate
        // reads it; a validate run with nothing to probe used to satisfy
        // that gate silently, on the default path (a seeded workspace has
        // no validation.jsonl for any concept).
        var vacuousConcepts: [String] = []
        func recordVacuous(_ ref: ExperimentManifest.ConceptRef) {
            guard ExperimentStore.owesHeldOutProbe(ref) else { return }
            vacuousConcepts.append(ref.name)
        }
        for ref in manifest.concepts {
            if await cancel.observed(at: "validation scenarios for '\(ref.name)'") {
                return await cancelledValidation(runDirectory)
            }
            // Method, data identity and the skip rule resolve TOGETHER
            // (open-issues §21): the DATA method (an artifact-pinned concept
            // answers with its source method), the DATA concept whose files
            // the probe reads, and "is there a probe at all". Exhaustive
            // semantics live in the resolver: a method that is neither
            // contrastive nor grand-mean refuses loudly rather than falling
            // into whichever branch is syntactically last (recording a note
            // and continuing was fail-open — review round 3, finding 1).
            let probe: ValidationProbePlan.Probe
            switch try validationProbePlan(for: ref) {
            case .skipped(let reason):
                // Skipped, not refused, so a MIXED study still validates its
                // ordinary concepts — and skipped WITHOUT a report entry and
                // without a vacuity record, exactly as the server does: the
                // concept owes no held-out probe, so counting it as
                // unmeasured would invent an obligation it can never meet.
                print("skipping validation — \(reason)")
                await log?("skipping validation — \(reason)")
                continue
            case .probe(let resolved):
                probe = resolved
            }
            let grandMean = probe.isGrandMean
            // Dual-root lookup (2026-08-19): the recipe's canonical home
            // first, the OTHER recipe's home as a fallback. A set filed
            // under the wrong root used to be read as absent — no probe
            // scored, no pin, no error — so the fallback is LOUD: the
            // advisory names where it was found and where it belongs, and
            // the probe still runs. The lookup is under the DATA concept
            // (what `verify()` pins on both engines); the advisory is
            // labelled with the STUDY-side name, which is what the
            // researcher declared.
            let location = ExperimentStore.resolveConceptValidation(
                name: probe.dataConcept, isPaired: probe.validationIsPaired)
            if let advisory = ExperimentStore.validationLookupAdvisory(
                concept: ref.name, location: location)
            {
                print("advisory: \(advisory)")
                await log?("advisory: \(advisory)")
            }
            let conceptDirectory =
                location?.url.deletingLastPathComponent()
                ?? (probe.method.usesStoryCorpus
                    ? ExperimentStore.emotionsDirectory
                        .appending(component: probe.dataConcept)
                    : VectorCatalog.conceptsDirectory
                        .appending(component: probe.dataConcept))
            guard let scenarios = try StimulusSet.loadValidation(directory: conceptDirectory),
                !scenarios.isEmpty,
                let extraction = extractions[ref.name]
            else {
                validation[ref.name] = "no validation.jsonl — convergent gate NOT run"
                recordVacuous(ref)
                continue
            }
            let resolutions = try validationLayerResolutions(
                for: ref.name, manifest: manifest,
                layerCount: extraction.result.vectors.layerCount)
            var depthEntries: [[String: Any]] = []
            var failure: String?
            if grandMean {
                // Convergent validity against the concept-vs-population
                // midpoint — the population plays the negative-class role.
                // Activations are captured once for all layers; each
                // declared depth is per-layer arithmetic on the same pass.
                // Held-out activations must be read where AND rendered how
                // the vector was: a probe score is a projection onto that
                // direction, and a raw-tokenized scenario is a sample from a
                // different distribution than a templated one.
                //
                // VALIDATION IS FRAME-FREE, deliberately: the study's
                // `manifest.systemPrompt` — and, since the 2026-08-24
                // composition ruling, any agent persona composed with it —
                // governs GENERATION arming and nothing else. It must never
                // reach a held-out read, or the probe would score a
                // distribution the vector was not extracted from and the
                // accuracy would move with a run-time deployment choice. The
                // ONE sanctioned channel for persona- or template-conditioned
                // validation is the recipe's own pinned
                // `extractionRendering.systemPrompt`, resolved here — it is
                // part of recipe identity, so extraction and validation
                // cannot silently disagree about it. (Asserted by
                // `SystemPromptCompositionTests`; server twin: the rendering
                // resolution in `tasks._task_validate`.)
                let (labels, corpusActs) = try await corpusActivations(
                    ref.options.readingPosition,
                    ref.options.resolvedExtractionRendering)
                let scenarioActs = try await ConceptExtractor.activations(
                    container: container, texts: scenarios.map(\.text),
                    position: ref.options.readingPosition,
                    rendering: ref.options.resolvedExtractionRendering)
                for resolution in resolutions {
                    let layer = resolution.layer
                    let direction = extraction.result.vectors.perLayer[layer]
                    // Rows are selected by the DATA concept's label in the
                    // pinned corpus — an artifact-pinned "crit-gm" reads
                    // "crit"'s rows (server: `c == data_concept`).
                    let conceptRows = zip(labels, corpusActs.values)
                        .filter { $0.0 == probe.dataConcept }
                        .map { $0.1[layer] }
                    let populationRows = corpusActs.values.map { $0[layer] }
                    guard
                        let computed = ConceptStats.scenarioAccuracyGrandMean(
                            direction: direction,
                            concept: conceptRows, population: populationRows,
                            scenarios: scenarioActs.values.map { $0[layer] },
                            labels: scenarios.map(\.expresses))
                    else {
                        failure = "grand-mean corpus produced no scorable rows "
                            + "— convergent gate NOT run"
                        break
                    }
                    let conceptMean = try SteeringVectorMath.mean(conceptRows)
                    let populationMean = try SteeringVectorMath.mean(
                        populationRows)
                    let conceptProjection = SteeringVectorMath.dot(
                        direction, conceptMean)
                    let populationProjection = SteeringVectorMath.dot(
                        direction, populationMean)
                    let diagnostics = try ScenarioDiagnostics.report(
                        scenarioIDs: scenarios.map { _ in nil },
                        scenarioTexts: scenarios.map(\.text),
                        projections: scenarioActs.values.map {
                            Double(SteeringVectorMath.dot(direction, $0[layer]))
                        },
                        labels: scenarios.map(\.expresses),
                        threshold: Double(
                            (conceptProjection + populationProjection) / 2),
                        classMeans: [
                            "concept": Double(conceptProjection),
                            "population": Double(populationProjection),
                        ],
                        layer: layer,
                        directionNorm: Double(
                            SteeringVectorMath.l2Norm(direction)))
                    depthEntries.append(depthEntry(
                        layer: layer, accuracy: computed,
                        diagnostics: diagnostics, resolution: resolution))
                }
            } else {
                // The class means come from the stimuli EXTRACTION read, which
                // on this engine are already the data concept's: an
                // artifact-pinned concept (the only kind whose data concept
                // differs from its name) is refused up front by `extractAll`
                // — materialization from pinned bytes is server-only — so
                // `extraction.stimuli` and `probe.dataConcept` cannot
                // disagree here. The server re-loads them under
                // `data_concept` for the same reason.
                guard let stimuli = extraction.stimuli else {
                    validation[ref.name] = "no stimulus set — convergent gate NOT run"
                    recordVacuous(ref)
                    continue
                }
                let profile = try await scenarioAccuracyProfile(
                    scenarios: scenarios, extraction: extraction.result,
                    stimuli: stimuli, layers: resolutions.map(\.layer),
                    options: ref.options, container: container)
                for (scored, resolution) in zip(profile, resolutions) {
                    depthEntries.append(depthEntry(
                        layer: scored.layer, accuracy: scored.accuracy,
                        diagnostics: scored.diagnostics, resolution: resolution))
                }
            }
            if let failure {
                validation[ref.name] = failure
                recordVacuous(ref)
                continue
            }
            // `depths` is the canonical shape; the flat single-depth mirror
            // is kept EXACTLY when one depth resolves, so every pre-list
            // consumer reads unchanged. With several depths there is no flat
            // mirror — nothing may silently read depth[0] as "the" accuracy.
            var entry: [String: Any] = ["scenarios": scenarios.count]
            entry["depths"] = depthEntries
            if depthEntries.count == 1, let only = depthEntries.first {
                entry.merge(only) { current, _ in current }
            }
            validation[ref.name] = entry
            for (sub, resolution) in zip(depthEntries, resolutions) {
                let accuracy = sub["accuracy"] as? Float ?? 0
                let line =
                    "\(ref.name): validation accuracy "
                    + "\(Int((accuracy * 100).rounded()))% over \(scenarios.count) "
                    + "never-named scenarios @ \(resolution.summary)"
                print(line)
                await log?(line)
            }
        }
        report["validation"] = validation
        // The vacuity verdict rides the REPORT as well as the evidence file,
        // so a run directory says on its own face what it did not measure.
        report["vacuousConcepts"] = vacuousConcepts.sorted()
        if !vacuousConcepts.isEmpty {
            let byName = Dictionary(
                manifest.concepts.map { ($0.name, $0) },
                uniquingKeysWith: { first, _ in first })
            let paths = vacuousConcepts.sorted()
                .compactMap { byName[$0].map(ExperimentStore.heldOutProbePath) }
            let line =
                "WARNING: VACUOUS validation — no held-out probe was scored for "
                + "concept(s) \(vacuousConcepts.sorted().joined(separator: ", ")). "
                + "Author the never-named scenarios "
                + "(\(paths.joined(separator: ", "))) as "
                + "{\"text\": …, \"expresses\": true|false} rows and re-run "
                + "validate; this run is stamped vacuous and will NOT satisfy "
                + "freeze's validateEvidence gate"
            print(line)
            await log?(line)
        }

        // 1b. Logit-lens sanity check: read each vector through the model's
        // unembedding/output head. This is not a gate, but it catches dead or
        // obviously confounded vectors before expensive steering runs.
        var logitLens: [String: Any] = [:]
        for ref in manifest.concepts {
            guard let extraction = extractions[ref.name]?.result else { continue }
            var perDepth: [Any] = []
            for resolution in try validationLayerResolutions(
                for: ref.name, manifest: manifest,
                layerCount: extraction.vectors.layerCount)
            {
                let layer = resolution.layer
                do {
                    let lens = try await ConceptExtractor.logitLens(
                        container: container,
                        vectors: extraction.vectors,
                        layer: layer,
                        topK: 10)
                    perDepth.append([
                        "layer": lens.layer,
                        "topPositive": lens.topPositive.map {
                            ["tokenID": $0.tokenID, "token": $0.token, "logit": $0.logit]
                        },
                        "topNegative": lens.topNegative.map {
                            ["tokenID": $0.tokenID, "token": $0.token, "logit": $0.logit]
                        },
                    ] as [String: Any])
                    let top = lens.topPositive.prefix(5).map(\.token).joined(separator: ", ")
                    let line = "\(ref.name): logit-lens top tokens @ L\(layer): \(top)"
                    print(line)
                    await log?(line)
                } catch {
                    perDepth.append("logit-lens skipped: \(error)")
                }
            }
            // Single depth keeps the historical flat shape; a list of depths
            // is a list of the same blocks (server twin).
            logitLens[ref.name] = perDepth.count == 1 ? perDepth[0] : perDepth
        }
        report["logitLens"] = logitLens

        // 2. Cross-concept cosine matrix (discriminant validity) over the
        // study's concepts plus its DECLARED controls (C2).
        //
        // This used to sweep in "every other concept on disk", extracted with
        // the first pinned paired concept's options. The control SET was
        // ambient — it changed whenever unrelated work landed in the
        // workspace, so `worstCosinePair` was not a property of the study and
        // the same manifest gave different evidence on two machines. The
        // control RECIPE was borrowed, so a control authored for grand-mean
        // extraction was read at the wrong position by the wrong method. And
        // the server had no controls at all, so the engines disagreed about
        // what validate measures.
        var allExtractions = extractions
        let neutral = try neutralTexts(for: manifest)
        for control in manifest.validationControls ?? [] {
            if let declared = control.modelRevision,
                let pinned = manifest.modelRevision, declared != pinned
            {
                throw ExperimentError(
                    reason: "validation control '\(control.concept)' pins model "
                        + "revision \(declared), but the study pins \(pinned) — a "
                        + "control extracted from a different revision is not "
                        + "comparable to the study's directions")
            }
            let directory = VectorCatalog.conceptsDirectory
                .appending(component: control.concept)
            guard let stimuli = try? StimulusSet(directory: directory) else {
                throw ExperimentError(
                    reason: "validation control '\(control.concept)' has no "
                        + "readable stimulus set at \(directory.path) — declared "
                        + "controls are pinned inputs, not best-effort extras")
            }
            guard stimuli.hash == control.stimulusSetHash else {
                throw ExperimentError(
                    reason: "validation control '\(control.concept)' stimulus set "
                        + "drifted from its pin (\(control.stimulusSetHash.prefix(12))… "
                        + "→ \(stimuli.hash.prefix(12))…) — re-pin the control or "
                        + "restore the file")
            }
            if await cancel.observed(at: "control extraction '\(control.concept)'") {
                return await cancelledValidation(runDirectory)
            }
            print("extracting control '\(control.concept)'…")
            await log?("extracting control '\(control.concept)' with its own recipe…")
            let extraction = try await ConceptExtractor.extract(
                container: container, stimuli: stimuli,
                // The control's OWN options — never a study concept's.
                options: control.options,
                neutralTexts: neutral)
            allExtractions[control.concept] = ConceptExtraction(
                result: extraction, stimuli: stimuli)
        }
        // Nothing disappears silently: name the concepts the old ambient
        // rule would have folded in, and say how to keep them.
        for advisory in ExperimentStore.undeclaredControlAdvisories(
            manifest, availableConcepts: VectorCatalog.conceptNames())
        {
            print(advisory)
            await log?(advisory)
        }

        let names = allExtractions.keys.sorted()
        // ONE layer for the whole matrix. Resolving it per ROW made the
        // matrix asymmetric — (A,B) and (B,A) measured at different depths —
        // and an asymmetric matrix has no defined reading for the
        // maxCrossConceptCosine gate. The residual stream drifts with depth
        // (the same concept a few layers apart can be near-orthogonal to
        // itself), so a cosine spanning two depths conflates "different
        // concepts" with "different depths".
        let matrixLayerList = try matrixLayers(
            manifest: manifest, extractions: allExtractions)
        // One complete matrix per declared depth; the worst pair is tracked
        // across ALL of them (the gate reads the worst cosine anywhere the
        // study declared it would look).
        // Optional, not a ("", "", 0) sentinel: with a single direction in
        // the matrix no PAIR exists, and the sentinel printed (and stored)
        // "worst cross-concept |cosine|:  ×  = 0.0" — two empty
        // interpolation slots reading as a perfect discriminant-validity
        // result nobody measured (WP0 dry run #0, P0-1). Nil now means "no
        // pair was compared" and is reported as such.
        var worst: (a: String, b: String, cosine: Float)?
        for (index, oneLayer) in matrixLayerList.enumerated() {
            var csv = "concept,layer," + names.joined(separator: ",") + "\n"
            for a in names {
                // No per-artifact clamp: depth is uniform (checked above), so
                // every cell in this matrix is read at exactly `oneLayer`.
                var row = [a, String(oneLayer)]
                for b in names {
                    let va = allExtractions[a]!.result.vectors.perLayer[oneLayer]
                    let vb = allExtractions[b]!.result.vectors.perLayer[oneLayer]
                    let cosine = (try? SteeringVectorMath.cosineSimilarity(va, vb)) ?? 0
                    row.append(String(format: "%.4f", cosine))
                    if a < b, worst.map({ abs(cosine) > abs($0.cosine) }) ?? true {
                        worst = (a, b, cosine)
                    }
                }
                csv += row.joined(separator: ",") + "\n"
            }
            // The first matrix keeps the historical filename so every
            // existing consumer still finds it; additional depths are
            // suffixed with the layer they were read at (server twin).
            let filename = index == 0
                ? "cosine-matrix.csv" : "cosine-matrix-L\(oneLayer).csv"
            try csv.write(
                to: runDirectory.appending(component: filename),
                atomically: true, encoding: .utf8)
        }
        report["cosineMatrixLayer"] = matrixLayerList[0]
        report["cosineMatrixLayers"] = matrixLayerList
        let worstLine: String
        if let worst {
            report["worstCosinePair"] = "\(worst.a) × \(worst.b) = \(worst.cosine)"
            worstLine =
                "worst cross-concept |cosine|: \(worst.a) × \(worst.b) = \(worst.cosine)"
        } else {
            // No key at all rather than a fabricated one: every reader of
            // `worstCosinePair` (the app's caption, the results explorer)
            // treats absence as "not measured", which is the truth.
            worstLine =
                "worst cross-concept |cosine|: not measured — the matrix holds "
                + "\(names.count) direction\(names.count == 1 ? "" : "s"), so "
                + "there is no cross-concept PAIR. Declare validation controls "
                + "(validationControls) or attach a second concept to measure "
                + "discriminant validity"
        }
        print(worstLine)
        await log?(worstLine)

        // 3. Capability battery as evidence: greedy battery pass through
        // baseline + every variant condition (the freeze variant gate
        // requires these results per condition in the matching evidence).
        var batteryEvidence: [CapabilityBatteryConditionResult]?
        if !manifest.variantConditions.isEmpty {
            await log?(
                "running capability battery through baseline + "
                    + "\(manifest.variantConditions.count) variant condition"
                    + (manifest.variantConditions.count == 1 ? "" : "s") + "…")
            guard let battery = ExperimentStore.effectiveCapabilityBattery(for: manifest)
            else {
                throw ExperimentError(
                    reason: "capability battery missing or drifted "
                        + "(\(manifest.capabilityBatteryFile ?? VariantRobustness.defaultPreset.batteryFile)) "
                        + "— cannot validate variant conditions")
            }
            guard
                let results = try await capabilityBatteryEvidence(
                    manifest: manifest, container: container,
                    batteryFile: battery.file, batteryHash: battery.hash,
                    cancel: cancel)
            else {
                return await cancelledValidation(runDirectory)
            }
            batteryEvidence = results
            report["capabilityBattery"] = results.map { result in
                var row: [String: Any] = [
                    "condition": result.condition,
                    "batteryHash": result.batteryHash,
                    "total": result.total,
                    "correct": result.correct,
                    "accuracy": result.accuracy,
                ]
                // The server's evidence row stamps how the reading was armed
                // (`_row`); absent on legacy evidence, never null.
                if let format = result.batteryFormat {
                    row["batteryFormat"] = format
                }
                if let isolated = result.armingIsolated {
                    row["armingIsolated"] = isolated
                }
                return row
            }
        }

        let reportData = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        // Canonical filename matches the server's ("validation-report.json",
        // named by the evidence's reportFile); the byte-identical legacy
        // "report.json" copy keeps existing readers/UI working.
        try reportData.write(
            to: runDirectory.appending(component: "validation-report.json"))
        try reportData.write(to: runDirectory.appending(component: "report.json"))
        try ExperimentStore.writeValidationEvidence(
            for: manifest, runDirectory: runDirectory,
            capabilityBattery: batteryEvidence,
            vacuousConcepts: vacuousConcepts)
        print("run artifacts: \(runDirectory.path)")
        await log?("run artifacts: \(runDirectory.path)")
        return runDirectory
    }

    /// Runs the pinned capability battery through baseline and every variant
    /// condition and scores it with the `CapabilityBattery` scorer. One
    /// result row per condition; baseline is always included as the reference
    /// cell.
    ///
    /// Arming follows the battery's FORMAT (server
    /// `_capability_battery_evidence` twin): a format-2 battery declares its
    /// own rendering context and every condition — baseline included — is
    /// scored under it, and its choice items are read by the answer-token
    /// logprob instrument; a legacy battery keeps the historical behaviour
    /// (the manifest's context for baseline, the variant artifact's for each
    /// variant) so its pinned hash keeps its meaning, and earns a loud
    /// contamination advisory when a study system prompt is in play.
    /// Returns nil when `cancel` observes a cancellation between battery
    /// items or conditions — evidence from a partially scored battery is
    /// never written (the caller treats nil as "validation cancelled").
    static func capabilityBatteryEvidence(
        manifest: ExperimentManifest,
        container: ModelContainer,
        batteryFile: String,
        batteryHash: String,
        cancel: CancelPoller = CancelPoller(nil)
    ) async throws -> [CapabilityBatteryConditionResult]? {
        let battery = try CapabilityBattery(
            url: ExperimentStore.resolveProjectPath(batteryFile))
        guard !battery.items.isEmpty else {
            throw ExperimentError(reason: "capability battery '\(batteryFile)' is empty")
        }

        struct Runtime {
            let name: String
            let variant: ModelVariantArtifact?
        }
        let runtimes =
            [Runtime(name: "baseline", variant: nil)]
            + manifest.variantConditions.map { Runtime(name: $0.name, variant: $0.artifact) }

        var results: [CapabilityBatteryConditionResult] = []
        for runtime in runtimes {
            let activeAdapter: LoRAContainer?
            if let variant = runtime.variant {
                activeAdapter = try await loadAdapter(variant, into: container)
            } else {
                activeAdapter = nil
            }
            let conditionInjections = try runtime.variant.map(injections(for:)) ?? []
            let promptMode = runtime.variant
                .flatMap { ExperimentManifest.PromptMode(rawValue: $0.promptMode) }
                ?? (manifest.promptMode ?? .chatAssistant)
            let systemPrompt = runtime.variant?.systemPrompt ?? manifest.systemPrompt
            let qwenThinking = runtime.variant?.qwenThinkingEnabled
                ?? manifest.qwenThinkingEnabled ?? false
            // `systemPrompt:` is the FORMAT-1 caller context and keeps its
            // historical replacement shape, so a legacy battery's pinned hash
            // keeps its meaning. `agentSystemPrompt:` is the format-2 channel
            // (2026-08-24 battery-isolation ruling): the arm's persona
            // composes ahead of the battery's own declared arming, and the
            // study frame enters neither — baseline reads the battery bare.
            let arming = battery.resolveArming(
                promptMode: promptMode, systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinking,
                agentSystemPrompt: runtime.variant?.systemPrompt)
            if let advisory = battery.contaminationAdvisory(arming) {
                print("WARNING: \(advisory)")
            }
            let scored: (records: [BatteryGenerationRecord],
                summary: CapabilityBatterySummary)?
            do {
                scored = try await ExperimentTasks.runBattery(
                    container, battery: battery, batteryHash: batteryHash,
                    condition: runtime.name, modelID: manifest.modelID,
                    injections: conditionInjections, arming: arming,
                    cancel: { _ in
                        await cancel.observed(
                            at: "battery evidence '\(runtime.name)'")
                    })
            } catch {
                try? await setInterventions(container, [])
                if let activeAdapter {
                    await unloadAdapter(activeAdapter, from: container)
                }
                throw error
            }
            try await setInterventions(container, [])
            if let activeAdapter {
                await unloadAdapter(activeAdapter, from: container)
            }
            guard let scored else { return nil }
            let correct = scored.records.count { $0.correct }
            let accuracy = Double(correct) / Double(battery.items.count)
            results.append(
                CapabilityBatteryConditionResult(
                    condition: runtime.name,
                    batteryHash: batteryHash,
                    total: battery.items.count,
                    correct: correct,
                    accuracy: accuracy,
                    batteryFormat: battery.formatVersion,
                    armingIsolated: battery.isolated))
            print(
                "capability battery \(runtime.name): \(correct)/\(battery.items.count) "
                    + String(format: "(%.0f%%)", accuracy * 100))
        }
        return results
    }

    // MARK: - sweep

    static func setInterventions(
        _ container: ModelContainer, _ interventions: [any LayerIntervention]
    ) async throws {
        try await container.perform { context in
            guard let model = context.model as? InterventionHookable else {
                throw ExperimentError(reason: "loaded model has no intervention hooks")
            }
            model.interventions = interventions
        }
    }

    /// One steering injection for a generation, in raw-alpha units.
    struct CellInjection {
        let layer: Int
        let vector: [Float]
        /// α when `mode == .add`, λ when `.ablate`.
        let alpha: Float
        /// Defaulted so every existing construction is unchanged: a study
        /// that declares no ablation builds exactly the chain it always did.
        var mode: InterventionPlan.Mode = .add
        /// Orders the ablation basis (Gram-Schmidt is order-dependent).
        /// Irrelevant for `.add`, which commutes.
        var concept: String = ""

        var planEdit: InterventionPlan.Edit {
            .init(
                layer: layer, vector: vector, strength: alpha, mode: mode,
                concept: concept)
        }
    }

    /// Generates with the given injections installed for exactly this
    /// prompt. The prompt's token count is threaded into each injector so
    /// chunked prefill (prefillStepSize 512) cannot steer mid-prompt chunk
    /// tails — prompt lengths differ per generation, so injectors must be
    /// rebuilt here, not installed once per condition.
    ///
    /// `transcript` (scripted-transcript study items) renders the item's
    /// whole scripted conversation through the chat template
    /// (`studyUserInput`): the transcript IS the prompt once rendered, so the
    /// injector gate and the context budget see the full rendered token
    /// count, and the transcript's own system turn replaces `systemPrompt`.
    static func generate(
        _ container: ModelContainer, prompt: String, modelID: String, maxTokens: Int,
        temperature: Double = 0,
        injections: [CellInjection] = [],
        promptMode: ExperimentManifest.PromptMode = .chatAssistant,
        systemPrompt: String? = nil,
        qwenThinkingEnabled: Bool = false,
        transcript: [TranscriptTurn]? = nil,
        onChunk: GenerationChunkHandler? = nil
    ) async throws -> String {
        try await generateMeasured(
            container, prompt: prompt, modelID: modelID, maxTokens: maxTokens,
            temperature: temperature, injections: injections,
            promptMode: promptMode, systemPrompt: systemPrompt,
            qwenThinkingEnabled: qwenThinkingEnabled, transcript: transcript,
            onChunk: onChunk
        ).text
    }

    /// One sampled generation plus the stream's own account of why it
    /// stopped. `hitTokenCap` is the honest truncation signal the choice
    /// parser needs — a decode that ended by exhausting `maxTokens` was cut
    /// off, not finished, and must parse as a failure rather than as its
    /// first-enumerated option (`Judicial.parseChoice`). Server twin: the
    /// sampled loop's `token_ids_out` count against the manifest's budget.
    struct MeasuredGeneration {
        let text: String
        let hitTokenCap: Bool
    }

    static func generateMeasured(
        _ container: ModelContainer, prompt: String, modelID: String, maxTokens: Int,
        temperature: Double = 0,
        injections: [CellInjection] = [],
        promptMode: ExperimentManifest.PromptMode = .chatAssistant,
        systemPrompt: String? = nil,
        qwenThinkingEnabled: Bool = false,
        transcript: [TranscriptTurn]? = nil,
        onChunk: GenerationChunkHandler? = nil
    ) async throws -> MeasuredGeneration {
        let input = try await container.prepare(
            input: studyUserInput(
                text: prompt,
                transcript: transcript,
                modelID: modelID,
                promptMode: promptMode,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled))
        let promptTokenCount = input.text.tokens.size
        try await validateContextBudget(
            container,
            modelID: modelID,
            promptTokens: promptTokenCount,
            requestedGenerationTokens: maxTokens)
        try await setInterventions(
            container,
            try InterventionPlan.interventions(
                injections.map(\.planEdit),
                promptTokenCount: promptTokenCount))
        let stream = try await container.generate(
            input: input,
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: Float(temperature),
                prefillStepSize: 512))
        var text = ""
        var lastProgressCount = 0
        var hitTokenCap = false
        for await event in stream {
            try Task.checkCancellation()
            switch event {
            case .chunk(let chunk):
                text += chunk
                if let onChunk,
                    text.count - lastProgressCount >= 120 || chunk.contains("\n")
                {
                    lastProgressCount = text.count
                    await onChunk(text)
                }
            case .info(let info):
                hitTokenCap = info.stopReason == .length
            default:
                break
            }
        }
        if let onChunk, text.count != lastProgressCount {
            await onChunk(text)
        }
        return MeasuredGeneration(text: text, hitTokenCap: hitTokenCap)
    }

    public static func preparedPromptTokenCount(
        _ container: ModelContainer,
        prompt: String,
        modelID: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String? = nil,
        qwenThinkingEnabled: Bool = false
    ) async throws -> Int {
        let input = try await container.prepare(
            input: userInput(
                prompt: prompt,
                modelID: modelID,
                promptMode: promptMode,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled))
        return input.text.tokens.size
    }

    public static func validateContextBudget(
        _ container: ModelContainer,
        modelID: String,
        promptTokens: Int,
        requestedGenerationTokens: Int
    ) async throws {
        let contextWindow = await container.perform { context -> Int? in
            (context.model as? ContextWindowProviding)?.contextWindow
        }
        guard let contextWindow else { return }
        let required = promptTokens + requestedGenerationTokens + contextBudgetReserve
        guard required <= contextWindow else {
            throw ContextBudgetError(
                modelID: modelID,
                contextWindow: contextWindow,
                promptTokens: promptTokens,
                requestedGenerationTokens: requestedGenerationTokens,
                reservedTokens: contextBudgetReserve)
        }
    }

    /// Study-path guard (exact port of the server's `_check_option_lengths`):
    /// joint logprobs favor shorter options, so unequal scored-option token
    /// counts silently bias `selected`. Refuse unless the manifest explicitly
    /// acknowledges the imbalance. Best practice is short canonical labels
    /// (A/B) with descriptions outside the scored tokens.
    static func checkOptionLengths(
        _ choice: ChoiceResult, manifest: ExperimentManifest, promptID: String
    ) throws {
        let counts = Set(choice.options.map { $0.tokenIDs.count })
        guard counts.count > 1, manifest.acknowledgeUnequalOptionLengths != true else {
            return
        }
        let detail = choice.options
            .map { "'\($0.option)'=\($0.tokenIDs.count)" }
            .joined(separator: ", ")
        throw ExperimentError(
            reason: "item '\(promptID)': scored options have unequal token counts "
                + "(\(detail)) — joint logprobs favor shorter options. Use canonical "
                + "labels of equal length, or set acknowledgeUnequalOptionLengths "
                + "in the manifest to accept the bias knowingly")
    }

    /// Loads the manifest's pinned RepE reader artifacts for the
    /// `repeReaderScore` instrument (server `_reader_scorers` twin). Hash
    /// drift is `verify()`'s job (loadVerified already ran); the FULL
    /// binding — substrate, model, revision, concept — is re-checked here
    /// through the SAME helper verify uses (review 2026-08-02: the runtime
    /// checked only substrate, so a forced freeze could score one reader
    /// while calling it another concept).
    static func loadReaderScorers(
        _ manifest: ExperimentManifest
    ) throws -> [(concept: String, reader: RepEReader.Artifact)] {
        var scorers: [(concept: String, reader: RepEReader.Artifact)] = []
        for ref in manifest.readerRefs ?? [] {
            let url = ExperimentStore.resolveProjectPath(ref.path)
            let reader: RepEReader.Artifact
            do {
                reader = try RepEReader.loadArtifact(url: url)
            } catch {
                throw ExperimentError(
                    reason: "reader '\(ref.concept)' (\(ref.path)): \(error)")
            }
            let problems = ExperimentStore.readerBindingProblems(
                reader, refConcept: ref.concept, manifest: manifest)
            guard problems.isEmpty else {
                throw ExperimentError(
                    reason: "(\(ref.path)) "
                        + problems.joined(separator: "; "))
            }
            scorers.append((ref.concept, reader))
        }
        return scorers
    }

    /// The measured-generation render for one prompt.
    ///
    /// The family rules themselves — Gemma 3 has no system role; Qwen3
    /// studies disable thinking mode — live in `SteeringKit.PromptRendering`,
    /// which EXTRACTION also calls when a concept declares a chat-template
    /// `extractionRendering`. One definition, two callers: that is what makes
    /// "extraction rendered the way generation does" a checkable claim rather
    /// than two copies drifting apart (the drift that made a concept's
    /// direction depend on an undeclared detail).
    static func userInput(
        prompt: String,
        modelID: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String?,
        qwenThinkingEnabled: Bool
    ) -> UserInput {
        switch promptMode {
        case .rawCompletion:
            return UserInput(
                prompt: .text(
                    PromptRendering.rawCompletionText(
                        prompt: prompt, modelID: modelID,
                        systemPrompt: systemPrompt,
                        qwenThinkingEnabled: qwenThinkingEnabled)))

        case .chatAssistant:
            return UserInput(
                chat: PromptRendering.chatMessages(
                    prompt: prompt, modelID: modelID,
                    systemPrompt: systemPrompt),
                additionalContext: qwenContext(
                    modelID: modelID, qwenThinkingEnabled: qwenThinkingEnabled))
        }
    }

    /// Forwarder to the single definition (`SteeringKit.PromptRendering`),
    /// kept because several call sites in this file already name it.
    static func qwenContext(
        modelID: String, qwenThinkingEnabled: Bool
    ) -> [String: any Sendable]? {
        PromptRendering.qwenContext(
            modelID: modelID, qwenThinkingEnabled: qwenThinkingEnabled)
    }

    // MARK: - Transcript-with-roles rendering (send-as-assistant / prefill)

    /// One transcript turn for multi-turn rendering. `seeded` marks a
    /// researcher-authored assistant turn (the metacognition instrument:
    /// "words in the model's mouth"); it is PROVENANCE ONLY — rendering must
    /// treat a seeded assistant turn byte-identically to a real one, so the
    /// flag never enters this adapter's output.
    ///
    /// The scripted-transcript STUDY instrument is the built-out twin of this
    /// path: task-prompt items pin a `transcript` as hashed stimulus data and
    /// render through `transcriptMessages` / `studyUserInput` — the same
    /// role-tagged template path the interactive Playground uses — so pinned
    /// bytes and the interactive instrument can never drift. (Server twin:
    /// `prompt_render.render_messages` / `render_transcript`.)
    public struct ChatTurn: Sendable {
        public enum Role: Sendable { case user, assistant }
        public var role: Role
        public var text: String
        public var seeded: Bool

        public init(role: Role, text: String, seeded: Bool = false) {
            self.role = role
            self.text = text
            self.seeded = seeded
        }
    }

    // MARK: Conversation-structure constraints (per vendored family)

    /// What a family's chat template refuses at render time. Determined
    /// EMPIRICALLY against the pinned tokenizers (2026-07-13; all eight
    /// cached models — the live-template fixture test in
    /// `GoldenRenderFixtureTests` fails loudly if a re-vendored template
    /// drifts from this table):
    ///
    /// - **Gemma 3** (google + all mlx-community 4b/12b/27b): the template's
    ///   role loop raises `TemplateError("Conversation roles must alternate
    ///   user/assistant/user/assistant/...")` for an assistant-first history
    ///   AND for any consecutive same-role pair.
    /// - **Qwen3** (0.6B/4B/14B/32B): fully permissive — assistant-first and
    ///   consecutive same-role histories all render.
    ///
    /// Python twin: `prompt_render.conversation_constraints`.
    public struct ConversationConstraints: Sendable, Equatable {
        public var requiresLeadingUserTurn: Bool
        public var forbidsConsecutiveSameRole: Bool

        public init(requiresLeadingUserTurn: Bool, forbidsConsecutiveSameRole: Bool) {
            self.requiresLeadingUserTurn = requiresLeadingUserTurn
            self.forbidsConsecutiveSameRole = forbidsConsecutiveSameRole
        }
    }

    /// Static family table — legitimate because the vendored families are
    /// pinned; re-vendoring is caught by the live-template fixture test.
    public static func conversationConstraints(modelID: String) -> ConversationConstraints {
        if modelID.lowercased().contains("gemma") {
            return ConversationConstraints(
                requiresLeadingUserTurn: true, forbidsConsecutiveSameRole: true)
        }
        return ConversationConstraints(
            requiresLeadingUserTurn: false, forbidsConsecutiveSameRole: false)
    }

    /// Short family label for constraint messages ("gemma-3", "qwen3", or the
    /// model id itself when the family is unrecognized).
    static func familyLabel(modelID: String) -> String {
        let lowered = modelID.lowercased()
        if lowered.contains("gemma") { return "gemma-3" }
        if lowered.contains("qwen") { return "qwen3" }
        return modelID
    }

    /// Pre-flight: would this role sequence violate the model's chat-template
    /// constraints? Returns a self-naming, actionable reason (nil = renders).
    /// Callers pass the roles of NON-EMPTY turns only, matching
    /// `chatHistoryMessages` / the server's `render_messages` cleaning.
    public static func conversationConstraintViolation(
        roles: [ChatTurn.Role], modelID: String
    ) -> String? {
        let constraints = conversationConstraints(modelID: modelID)
        let family = familyLabel(modelID: modelID)
        if constraints.requiresLeadingUserTurn, roles.first == .assistant {
            return "\(family)'s chat template requires the conversation to "
                + "start with a user turn — seed or send a user message first"
        }
        if constraints.forbidsConsecutiveSameRole {
            for (previous, next) in zip(roles, roles.dropFirst()) where previous == next {
                let role = previous == .user ? "user" : "assistant"
                return "\(family)'s chat template requires strict "
                    + "user/assistant alternation — two consecutive \(role) "
                    + "turns cannot be rendered"
            }
        }
        return nil
    }

    /// Pre-flight for APPENDING one turn to an existing transcript (the
    /// composer gate): the reason the template would refuse, or nil.
    public static func conversationConstraintViolation(
        appending role: ChatTurn.Role,
        toTranscriptRoles roles: [ChatTurn.Role],
        modelID: String
    ) -> String? {
        conversationConstraintViolation(roles: roles + [role], modelID: modelID)
    }

    // MARK: - Scripted study transcripts (the metacognition-study instrument)

    /// One turn of a task-prompt item's scripted `transcript` — a pinned
    /// multi-turn conversation (researcher-authored assistant turns included:
    /// "words in the model's mouth") that rides in the task-prompts file and
    /// is covered byte-exactly by the existing `taskPromptsHash` pin. Role is
    /// a plain string so schema violations produce the cross-engine message,
    /// not a decode failure; unknown per-turn keys (e.g. a Playground
    /// export's `seeded` flag) are dropped on decode — rendering must ignore
    /// them, and the RECORD copy of the transcript is identical across
    /// engines (the server normalizes to `{role, content}` the same way).
    public struct TranscriptTurn: Codable, Sendable, Equatable {
        public var role: String
        public var content: String

        enum CodingKeys: String, CodingKey {
            case role
            case content
        }

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }

        // `Swift.Decoder`: the `Tokenizers` import shadows the stdlib name.
        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            role = try container.decodeIfPresent(String.self, forKey: .role) ?? ""
            content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        }
    }

    static let transcriptRoles: Set<String> = ["system", "user", "assistant"]

    /// Cross-engine message for the rawCompletion incompatibility (server
    /// twin: `prompt_render.TRANSCRIPT_RAW_COMPLETION_MESSAGE`).
    static let transcriptRawCompletionMessage =
        "task prompts include scripted transcripts but promptMode is "
        + "rawCompletion — transcript items render through the chat template "
        + "by definition; use chatAssistant"

    /// First schema violation of one item's scripted transcript, or nil.
    ///
    /// The rules (validated identically at load on BOTH engines — message
    /// strings are the cross-engine contract, pinned by the committed fixture
    /// `prompts/fixtures/transcript-validation/cases.json`): non-empty; roles
    /// from {system, user, assistant}; non-empty content per turn; at most
    /// one system turn, and only first; the FINAL turn must be `user` —
    /// generation produces the assistant's reply (a trailing assistant turn
    /// would be assistant-prefix continuation, out of scope v1). Server twin:
    /// `prompt_render.transcript_schema_violation`.
    static func transcriptSchemaViolation(
        _ turns: [TranscriptTurn], itemID: String
    ) -> String? {
        guard !turns.isEmpty else {
            return "item '\(itemID)': transcript is empty — a scripted "
                + "transcript needs at least a final user turn"
        }
        for (index, turn) in turns.enumerated() {
            guard transcriptRoles.contains(turn.role) else {
                return "item '\(itemID)': transcript turn \(index + 1) has role "
                    + "'\(turn.role)' — allowed roles are system, user, assistant"
            }
            guard !turn.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return "item '\(itemID)': transcript turn \(index + 1) has "
                    + "empty content"
            }
        }
        guard !turns.dropFirst().contains(where: { $0.role == "system" }) else {
            return "item '\(itemID)': transcript may carry at most one system "
                + "turn, and it must be first"
        }
        switch turns.last?.role {
        case "user":
            return nil
        case "assistant":
            return "item '\(itemID)': transcript ends with an assistant turn — "
                + "generation produces the assistant's reply to a final user "
                + "turn; assistant-prefix continuation is out of scope for "
                + "scripted-transcript studies (v1)"
        default:
            return "item '\(itemID)': transcript must end with a user turn "
                + "(generation produces the assistant's reply to it)"
        }
    }

    /// First family chat-template constraint the transcript violates, or nil
    /// — checked over the NON-SYSTEM turns (a leading system turn composes
    /// with the study system prompt through the renderer's own family
    /// convention). Positions are 1-based indices into the FULL transcript.
    /// Server twin: `prompt_render.transcript_family_violation`.
    static func transcriptFamilyViolation(
        _ turns: [TranscriptTurn], itemID: String, modelID: String
    ) -> String? {
        let constraints = conversationConstraints(modelID: modelID)
        let family = familyLabel(modelID: modelID)
        let indexed = turns.enumerated().filter { $0.element.role != "system" }
        if constraints.requiresLeadingUserTurn,
            indexed.first?.element.role == "assistant"
        {
            return "item '\(itemID)': \(family)'s chat template requires the "
                + "conversation to start with a user turn — this transcript "
                + "starts with an assistant turn"
        }
        if constraints.forbidsConsecutiveSameRole {
            for (a, b) in zip(indexed, indexed.dropFirst())
            where a.element.role == b.element.role {
                return "item '\(itemID)': \(family)'s chat template requires "
                    + "strict user/assistant alternation — transcript turns "
                    + "\(a.offset + 1) and \(b.offset + 1) are consecutive "
                    + "\(a.element.role) turns"
            }
        }
        return nil
    }

    /// The record's display text for a transcript item without its own
    /// `text`/`prompt`: the final user turn (schema-validated to exist).
    static func transcriptDisplayText(_ turns: [TranscriptTurn]) -> String {
        turns.last?.content ?? ""
    }

    /// Run-START refusal for scripted-transcript items (never a mid-run
    /// template error): rawCompletion cannot render a transcript, and every
    /// transcript must satisfy the study model family's chat-template
    /// constraints (Gemma's user-first strict alternation) — a
    /// Gemma-incompatible transcript fails loudly here with the item id and
    /// rule named, BEFORE the model loads or any generation starts. Message
    /// strings are the cross-engine contract (server twin:
    /// `tasks._check_transcript_prompts`).
    static func checkTranscriptPrompts(
        _ prompts: [StudyPrompt], manifest: ExperimentManifest
    ) throws {
        let transcripted = prompts.filter { $0.transcript?.isEmpty == false }
        guard !transcripted.isEmpty else { return }
        guard (manifest.promptMode ?? .chatAssistant) != .rawCompletion else {
            throw ExperimentError(reason: transcriptRawCompletionMessage)
        }
        let violations = transcripted.compactMap { prompt in
            prompt.transcript.flatMap {
                transcriptFamilyViolation(
                    $0, itemID: prompt.id, modelID: manifest.modelID)
            }
        }
        guard violations.isEmpty else {
            throw ExperimentError(
                reason: "scripted transcripts are incompatible with "
                    + "\(manifest.modelID)'s chat template: "
                    + violations.joined(separator: "; "))
        }
    }

    /// Verify/pin-time transcript violations for a parsed task-prompts file
    /// (the model family is known from the manifest): per-item family
    /// constraints plus the rawCompletion incompatibility, as verify()
    /// violation strings. Schema violations surface as parse errors before
    /// this runs. Server twin: `Manifest._transcript_violations`.
    static func transcriptPinViolations(
        _ prompts: [StudyPrompt], manifest: ExperimentManifest
    ) -> [String] {
        let transcripted = prompts.filter { $0.transcript?.isEmpty == false }
        guard !transcripted.isEmpty else { return [] }
        var violations = transcripted.compactMap { prompt in
            prompt.transcript.flatMap {
                transcriptFamilyViolation(
                    $0, itemID: prompt.id, modelID: manifest.modelID)
            }
        }
        if (manifest.promptMode ?? .chatAssistant) == .rawCompletion {
            violations.append(transcriptRawCompletionMessage)
        }
        return violations
    }

    /// Family-correct chat messages for one scripted-transcript study item —
    /// the STUDY twin of the server's `prompt_render.render_transcript` →
    /// `render_messages` path, byte-parity-pinned by the `study_transcript`
    /// golden fixtures.
    ///
    /// System-prompt composition RULE (cross-engine): a transcript's own
    /// system turn REPLACES the study-level system prompt for that item — the
    /// transcript is the more specific declaration. Family handling then
    /// follows the SERVER transcript convention exactly (`render_messages`):
    /// Gemma folds the effective system text into the FIRST user turn (not
    /// every turn — that is the local interactive `ChatService` convention);
    /// other families get a leading `.system` message.
    static func transcriptMessages(
        _ transcript: [TranscriptTurn],
        modelID: String,
        studySystemPrompt: String?
    ) -> [Chat.Message] {
        var turns = transcript[...]
        var system = studySystemPrompt?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let first = turns.first, first.role == "system" {
            system = first.content.trimmingCharacters(in: .whitespacesAndNewlines)
            turns = turns.dropFirst()
        }
        // Same cleaning as the server's render_messages: empty turns drop
        // (schema validation refuses them for pinned studies anyway).
        var messages: [Chat.Message] = turns.compactMap { turn in
            let text = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return turn.role == "assistant" ? .assistant(text) : .user(text)
        }
        if !system.isEmpty {
            if modelID.lowercased().contains("gemma") {
                if let index = messages.firstIndex(where: { $0.role == .user }) {
                    messages[index] = .user(system + "\n\n" + messages[index].content)
                } else {
                    messages.insert(.user(system), at: 0)
                }
            } else if messages.first?.role != .system {
                messages.insert(.system(system), at: 0)
            }
        }
        return messages
    }

    /// The `UserInput` for one study item — the single dispatch point the
    /// unified run loop (ordinary, steered, AND variant conditions) and the
    /// logprob instrument all render through, so every condition type renders
    /// a transcript identically. Plain items go through `userInput`
    /// unchanged; transcript items render their whole scripted conversation
    /// through the model family's real chat template.
    static func studyUserInput(
        text: String,
        transcript: [TranscriptTurn]?,
        modelID: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String?,
        qwenThinkingEnabled: Bool
    ) throws -> UserInput {
        guard let transcript, !transcript.isEmpty else {
            return userInput(
                prompt: text, modelID: modelID, promptMode: promptMode,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled)
        }
        // Defense in depth — the manifest/run gates refuse this combination
        // long before generation.
        guard promptMode == .chatAssistant else {
            throw ExperimentError(
                reason: "scripted transcripts require chatAssistant prompt mode")
        }
        return UserInput(
            chat: transcriptMessages(
                transcript, modelID: modelID, studySystemPrompt: systemPrompt),
            additionalContext: qwenContext(
                modelID: modelID, qwenThinkingEnabled: qwenThinkingEnabled))
    }

    /// Family-correct multi-turn history for the local chat path — the
    /// prompt-render adapter for transcripts, so no family branching leaks
    /// into ChatService or views. Empty turns are dropped (matching the
    /// server's `render_messages` cleaning).
    ///
    /// - Gemma 3 has no system role: the system text folds into EVERY user
    ///   turn — the local interactive convention (`ChatService.chatPrompt`
    ///   prepends per turn, so this is what the model actually saw on the
    ///   turns being replayed). The server transcript convention folds into
    ///   the first turn only; the divergence is pinned in the fixture README.
    /// - Non-Gemma: NO system message is added here — the caller supplies
    ///   instructions to `ChatSession` (or adds `.system` itself for offline
    ///   renders).
    public static func chatHistoryMessages(
        turns: [ChatTurn],
        modelID: String,
        systemPrompt: String?
    ) -> [Chat.Message] {
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let foldSystem = modelID.lowercased().contains("gemma") && !system.isEmpty
        return turns.compactMap { turn in
            let text = turn.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            switch turn.role {
            case .user:
                return .user(foldSystem ? system + "\n\n" + text : text)
            case .assistant:
                return .assistant(text)
            }
        }
    }

    /// The rendered prompt for an assistant-prefix continuation ("prefill").
    public struct ContinuationRender: Sendable {
        /// Token ids fed to the model (ends mid-assistant-turn).
        public let tokenIDs: [Int]
        /// The rendered string those ids encode (for records/inspection).
        public let text: String
    }

    /// transformers' sentinel for `continue_final_message` (5.12.x,
    /// `utils/chat_template_utils.py`) — mirrored verbatim so both engines cut
    /// the render at the same place.
    static let continueFinalMessageSentinel = "CONTINUE_FINAL_MESSAGE_TAG "

    /// Renders a transcript whose FINAL turn is an INCOMPLETE assistant turn
    /// so generation continues it mid-turn: no end-of-turn marker, no new
    /// generation prompt. Swift twin of HF transformers'
    /// `apply_chat_template(..., continue_final_message=True)` — the
    /// identical sentinel algorithm, because swift-transformers'
    /// `applyChatTemplate` has no native equivalent (verified 1.3.x,
    /// `Sources/Tokenizers/Tokenizer.swift`: only `addGenerationPrompt`).
    ///
    /// Honesty guards (each throws instead of degrading):
    /// - the render is recovered from the template engine's own token output
    ///   via decode, and re-encoding must reproduce the ids byte-for-byte
    ///   (so the string surgery operates on the true render);
    /// - the final content must survive the template verbatim (a template
    ///   that transforms it — e.g. stripping think blocks — cannot honestly
    ///   be continued);
    /// - byte-parity with the Python engine is pinned by the
    ///   `chat_prefill_continue` golden fixtures for both families.
    public static func continuationRender(
        tokenizer: any Tokenizers.Tokenizer,
        turns: [ChatTurn],
        modelID: String,
        systemPrompt: String?,
        qwenThinkingEnabled: Bool
    ) throws -> ContinuationRender {
        var messages = chatHistoryMessages(
            turns: turns, modelID: modelID, systemPrompt: systemPrompt)
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !system.isEmpty, !modelID.lowercased().contains("gemma") {
            messages.insert(.system(system), at: 0)
        }
        guard let last = messages.last, last.role == .assistant,
            !last.content.isEmpty
        else {
            throw ExperimentError(
                reason: "assistant-prefix continuation requires the final turn "
                    + "to be a non-empty assistant prefix")
        }
        let finalContent = last.content
        messages[messages.count - 1] = .assistant(
            finalContent + Self.continueFinalMessageSentinel)

        let dictionaries = DefaultMessageGenerator().generate(messages: messages)
        let taggedIDs = try tokenizer.applyChatTemplate(
            messages: dictionaries,
            chatTemplate: nil,
            addGenerationPrompt: false,
            truncation: false,
            maxLength: nil,
            tools: nil,
            additionalContext: qwenContext(
                modelID: modelID, qwenThinkingEnabled: qwenThinkingEnabled))
        let rendered = tokenizer.decode(tokens: taggedIDs, skipSpecialTokens: false)
        guard tokenizer.encode(text: rendered, addSpecialTokens: false) == taggedIDs else {
            throw ExperimentError(
                reason: "this tokenizer's decode/encode round trip is not "
                    + "byte-stable, so an assistant-prefix continuation cannot "
                    + "be rendered honestly on this engine for \(modelID)")
        }
        let cut = try continuationCut(rendered: rendered, finalContent: finalContent)
        return ContinuationRender(
            tokenIDs: tokenizer.encode(text: cut, addSpecialTokens: false),
            text: cut)
    }

    /// The pure sentinel cut — transformers' algorithm verbatim: find the
    /// LAST occurrence of the trimmed sentinel; if the template preserved the
    /// sentinel's trailing spacing, cut exactly there, otherwise cut and trim
    /// trailing whitespace (the template trimmed the message tail, so the
    /// prefix's own trailing spacing cannot survive either).
    static func continuationCut(
        rendered: String, finalContent: String
    ) throws -> String {
        let sentinel = continueFinalMessageSentinel
        let trimmedSentinel = sentinel.trimmingCharacters(in: .whitespaces)
        let trimmedContent = finalContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard rendered.contains(trimmedContent),
            let tagRange = rendered.range(of: trimmedSentinel, options: .backwards)
        else {
            throw ExperimentError(
                reason: "the chat template transformed the final assistant "
                    + "message, so it cannot be continued honestly (the seeded "
                    + "prefix does not appear verbatim in the rendered prompt)")
        }
        let head = String(rendered[..<tagRange.lowerBound])
        let tagWithSpacing = rendered[tagRange.lowerBound...]
        if tagWithSpacing.hasPrefix(sentinel) {
            return head
        }
        // Trailing spacing was trimmed by the template — mirror that trim.
        var trimmed = Substring(head)
        while let lastCharacter = trimmed.last, lastCharacter.isWhitespace {
            trimmed = trimmed.dropLast()
        }
        return String(trimmed)
    }

    /// Generation from pre-rendered token ids (the continuation path): the
    /// prompt bytes were produced by `continuationRender`, so no further
    /// template application happens here. Injection gates on the true prompt
    /// token count, exactly like the text path.
    static func generate(
        _ container: ModelContainer, promptTokenIDs: [Int], modelID: String,
        maxTokens: Int,
        temperature: Double = 0,
        injections: [CellInjection] = [],
        onChunk: GenerationChunkHandler? = nil
    ) async throws -> String {
        try await validateContextBudget(
            container,
            modelID: modelID,
            promptTokens: promptTokenIDs.count,
            requestedGenerationTokens: maxTokens)
        try await setInterventions(
            container,
            try InterventionPlan.interventions(
                injections.map(\.planEdit),
                promptTokenCount: promptTokenIDs.count))
        let stream = try await container.generate(
            input: LMInput(tokens: MLXArray(promptTokenIDs)),
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: Float(temperature),
                prefillStepSize: 512))
        var text = ""
        var lastProgressCount = 0
        for await event in stream {
            try Task.checkCancellation()
            if case .chunk(let chunk) = event {
                text += chunk
                if let onChunk,
                    text.count - lastProgressCount >= 120 || chunk.contains("\n")
                {
                    lastProgressCount = text.count
                    await onChunk(text)
                }
            }
        }
        if let onChunk, text.count != lastProgressCount {
            await onChunk(text)
        }
        return text
    }

    /// The sweep's capability constraint under one cell's injections, armed
    /// by the BATTERY (format 2) or by the study manifest (legacy) — server
    /// `_battery_accuracy` twin.
    ///
    /// Returns nil when `shouldCancel` observes a cancellation between
    /// battery items — a partially scored battery is never reported as an
    /// accuracy (the caller drops the whole cell instead).
    static func batteryAccuracy(
        _ container: ModelContainer, battery: CapabilityBattery, modelID: String,
        arming: BatteryArming,
        injections: [CellInjection] = [],
        shouldCancel: (@Sendable () async -> Bool)? = nil
    ) async throws -> Float? {
        let backends = batteryBackends(
            container, modelID: modelID, injections: injections)
        var correct = 0
        for item in battery.items {
            if let shouldCancel, await shouldCancel() { return nil }
            let score = try await battery.scoreItem(
                item, arming: arming,
                generate: backends.generate, choice: backends.choice)
            if score.correct { correct += 1 }
        }
        return Float(correct) / Float(battery.items.count)
    }

    /// Loud, non-fatal ablation preflight (server twins:
    /// `tasks._warn_on_mean_aligned_ablation` /
    /// `model_variant._preflight_mean_alignment`; same threshold constant).
    ///
    /// With a mean available, names the worst-aligned layer when it crosses
    /// `SteeringVectorMath.ablationMeanAlignmentWarnThreshold`; with no mean,
    /// says the check is impossible — unknown alignment is never reported as
    /// safe. Returns the warning it printed so UI callers can surface it.
    @discardableResult
    static func reportAblationMeanAlignment(
        concept: String, vectors: [[Float]], firstLayer: Int,
        neutralMean: [[Float]]?, where context: String, remedy: String
    ) -> String? {
        guard let neutralMean else {
            let warning = "\(context): ablating '\(concept)' with no stored "
                + "neutral mean — mean-alignment preflight impossible (artifact "
                + "predates the neutral-mean stamp or was extracted without a "
                + "neutral corpus). λ=1 ablation of a mean-aligned direction "
                + "collapses generation; re-extract to enable the check and "
                + "neutral-mean centering"
            print("warning: \(warning)")
            return warning
        }
        var worstLayer = -1
        var worst: Float = 0
        for (offset, pair) in zip(vectors, neutralMean).enumerated() {
            let alignment = SteeringVectorMath.meanAlignment(pair.0, with: pair.1)
            if alignment > worst {
                worst = alignment
                worstLayer = firstLayer + offset
            }
        }
        guard worst > SteeringVectorMath.ablationMeanAlignmentWarnThreshold else {
            return nil
        }
        let warning = "\(context): ablation direction for '\(concept)' is "
            + "strongly aligned with the neutral residual mean (|cos| "
            + String(format: "%.2f", worst) + " at layer \(worstLayer); warn "
            + "threshold \(SteeringVectorMath.ablationMeanAlignmentWarnThreshold)). "
            + "Full ablation of mean-aligned directions collapses generation "
            + "into single-token repetition — \(remedy), or expect incoherent "
            + "output"
        print("warning: \(warning)")
        return warning
    }

    static func injections(
        for condition: ExperimentManifest.Condition,
        extractions: [String: ConceptExtraction]
    ) throws -> [CellInjection] {
        var cells: [CellInjection] = []
        let neutralBasis = try condition.neutralPCBasisPath.map { try NeutralPCStore.load(path: $0).basis }
        let width = max(1, condition.bandWidth)
        let half = width / 2
        for slot in condition.slots {
            guard let extraction = extractions[slot.concept]?.result else {
                throw ExperimentError(
                    reason: "condition '\(condition.name)' references unextracted concept "
                        + "'\(slot.concept)'")
            }
            let layerCount = extraction.vectors.layerCount
            let isAblation = slot.effectiveMode == .ablate
            // Ablation covers the whole network: removing a direction at one
            // layer is usually undone by the layers above it. Steering keeps
            // its declared layer widened by the condition's band.
            let start = isAblation ? 0 : max(0, slot.layer - half)
            let end = isAblation ? layerCount - 1 : min(layerCount - 1, slot.layer + half)
            if isAblation, condition.controlType != "randomDirectionAblation" {
                // Ablation mean-alignment preflight (2026-08-06 collapse
                // study). Diagnostic only — a frozen manifest's semantics
                // are never changed under it (post-submit drift policy:
                // continue loudly). The random-direction control is exempt:
                // an arbitrary direction is its whole point.
                reportAblationMeanAlignment(
                    concept: slot.concept,
                    vectors: (start ... end).map { extraction.vectors.perLayer[$0] },
                    firstLayer: start,
                    neutralMean: extraction.neutralMeanPerLayer.map { mean in
                        (start ... end).map { mean[$0] }
                    },
                    where: "condition '\(condition.name)'",
                    remedy: "center the direction against the neutral mean")
            }
            for layer in start ... end {
                var vector: [Float]
                if let neutralBasis {
                    vector = SteeringVectorMath.projectingOut(
                        extraction.vectors.perLayer[layer],
                        components: neutralBasis.components(for: layer))
                } else {
                    vector = extraction.vectors.perLayer[layer]
                }
                let vectorNorm = SteeringVectorMath.l2Norm(vector)
                guard vectorNorm > 0 else { continue }
                if isAblation, condition.controlType == "randomDirectionAblation" {
                    // The ablation analogue of the matched-norm control.
                    // Norm-matching is meaningless for a projection — the
                    // removal is scaled by what the residual stream contains,
                    // not by the direction's length — so the control removes a
                    // random DIRECTION instead, answering the question the
                    // concept ablation raises: is the effect specific to this
                    // direction, or does removing any rank-1 subspace do it?
                    // Same seeding convention, so the cell is reproducible
                    // across re-runs of a frozen study.
                    vector = try SweepSelectionRule.controlVector(
                        seedText: "\(condition.name)|\(slot.concept)|\(layer)",
                        dimension: vector.count, norm: vectorNorm)
                } else if condition.controlType == "randomMatchedNorm" {
                    // Magnitude/noise control: a deterministic random
                    // direction with the SAME L2 norm as the concept vector
                    // at this layer, seeded from stable identifiers only —
                    // reproducible across re-runs of a frozen study (same
                    // seed-text convention as the server's
                    // `_matched_norm_random`). Without this substitution a
                    // control condition would silently inject the CONCEPT
                    // vector — the exact confound the control exists to rule
                    // out.
                    vector = try SweepSelectionRule.controlVector(
                        seedText: "\(condition.name)|\(slot.concept)|\(layer)",
                        dimension: vector.count, norm: vectorNorm)
                }
                let alpha: Float
                if isAblation {
                    // λ is never scaled by the residual norm. That denominator
                    // makes α comparable across concepts and layers; ablation
                    // removes exactly what is present, so it already scales
                    // itself and the conversion would silently change how much
                    // comes out.
                    alpha = Float(slot.alpha)
                } else if condition.alphaInNormUnits {
                    // ONE out-of-range rule, every verb, both engines
                    // (2026-08-28 audit, F7/F13). This site used to clamp to
                    // the last entry — dosing the deepest layers with a
                    // shallower layer's number while the server's condition
                    // path refused on the same artifact — and an EMPTY table
                    // made the clamp index [-1] and crash.
                    if let problem = ResidualNormConvention.residualNormProblem(
                        extraction.residualNormPerLayer, layer: layer,
                        artifact: slot.concept)
                    {
                        throw ExperimentError(
                            reason: "condition '\(condition.name)': \(problem)")
                    }
                    let norm = extraction.residualNormPerLayer[layer]
                    alpha = try SteeringVectorMath.normUnitScale(
                        alpha: Float(slot.alpha), residualNorm: norm,
                        vectorNorm: vectorNorm)
                } else {
                    alpha = Float(slot.alpha)
                }
                cells.append(
                    CellInjection(
                        layer: layer, vector: vector, alpha: alpha,
                        mode: slot.effectiveMode, concept: slot.concept))
            }
        }
        return cells
    }

    static func injections(for variant: ModelVariantArtifact) throws -> [CellInjection] {
        var cells: [CellInjection] = []
        let neutralBasis = try variant.neutralPCBasisPath.map { try NeutralPCStore.load(path: $0).basis }
        let width = max(1, variant.bandWidth)
        let half = width / 2
        for injection in variant.injections {
            // Resolved, not taken literally: a server-promoted agent stores
            // this workspace-relative, and `URL(filePath:)` would join it
            // against the process working directory (see ArtifactIdentity).
            let vectorURL = ArtifactIdentity.resolve(injection.vectorArtifactID)
            let name = vectorURL.lastPathComponent
            let directory = vectorURL.deletingLastPathComponent()
            let (vectors, sidecar) = try SteeringVectorStore.load(from: directory, name: name)
            let isAblation = injection.effectiveMode == .ablate
            let declared = injection.resolvedLayers(layerCount: vectors.layerCount)
            let center = min(max(0, injection.layer), vectors.layerCount - 1)
            let start = isAblation ? (declared.first ?? 0) : max(0, center - half)
            let end = isAblation
                ? (declared.last ?? vectors.layerCount - 1)
                : min(vectors.layerCount - 1, center + half)
            // Ablation-direction centering — declared per injection, never
            // silent (server twin: `model_variant.variant_injections`; see
            // `InjectionRef.centering`). Undeclared runs the mean-alignment
            // preflight instead: diagnostic only, semantics unchanged.
            let centering = injection.effectiveCentering
            var neutralMean: [[Float]]?
            if isAblation {
                guard centering == "none" || centering == "neutralMean" else {
                    throw ExperimentError(
                        reason: "variant '\(variant.name)' injection "
                            + "'\(injection.concept)' declares unknown centering "
                            + "'\(centering)' — this engine implements 'none' "
                            + "and 'neutralMean'")
                }
                neutralMean = try SteeringVectorStore.loadNeutralMean(
                    from: directory, name: name)
                if centering == "neutralMean", neutralMean == nil {
                    throw ExperimentError(
                        reason: "variant '\(variant.name)' declares neutral-mean "
                            + "centering for '\(injection.concept)' but the vector "
                            + "artifact carries no stored neutral mean — re-extract "
                            + "the concept with a neutral corpus (artifacts stamp "
                            + "neutralMeanSource since 2026-08-06)")
                }
                if centering == "none" {
                    reportAblationMeanAlignment(
                        concept: injection.concept,
                        vectors: (start ... end).map { vectors.perLayer[$0] },
                        firstLayer: start,
                        neutralMean: neutralMean.map { mean in
                            (start ... end).map { mean[$0] }
                        },
                        where: "variant '\(variant.name)'",
                        remedy: "declare centering \"neutralMean\" on the injection")
                }
            } else if centering != "none" {
                throw ExperimentError(
                    reason: "variant '\(variant.name)' declares centering on a "
                        + "STEERING injection ('\(injection.concept)') — centering "
                        + "is an ablation-direction transform; remove it or set "
                        + "mode ablate")
            }
            for layer in start ... end {
                var vector = vectors.perLayer[layer]
                // Centering FIRST, then any neutral-PC basis projection —
                // the server applies the same order (variant_injections),
                // and sequential projections do not commute in general.
                if isAblation, centering == "neutralMean", let neutralMean {
                    vector = SteeringVectorMath.meanCentered(
                        vector, against: neutralMean[layer])
                }
                if let neutralBasis {
                    vector = SteeringVectorMath.projectingOut(
                        vector, components: neutralBasis.components(for: layer))
                }
                let vectorNorm = SteeringVectorMath.l2Norm(vector)
                guard vectorNorm > 0 else { continue }
                let alpha: Float
                if isAblation {
                    // λ bypasses the norm-unit denominator entirely — which
                    // also means an ablating agent never hits the
                    // missing-residual-norms refusal below for a conversion it
                    // does not perform.
                    alpha = Float(injection.alpha)
                } else if variant.alphaInNormUnits {
                    guard let norms = sidecar.residualNormPerLayer else {
                        throw ExperimentError(
                            reason: "variant '\(variant.name)' uses residual-norm alpha, "
                                + "but vector \(injection.concept) has no residual norms "
                                + "— backfill norms in "
                                + "\(SteeringGuidance.normBackfillLocation), or switch "
                                + "the variant to raw alpha")
                    }
                    // Same rule as the condition and sweep paths (2026-08-28
                    // audit, F7/F13): a layer the denominator table does not
                    // reach refuses, where this site used to clamp.
                    if let problem = ResidualNormConvention.residualNormProblem(
                        norms, layer: layer, artifact: injection.concept)
                    {
                        throw ExperimentError(
                            reason: "variant '\(variant.name)': \(problem)")
                    }
                    let norm = norms[layer]
                    alpha = try SteeringVectorMath.normUnitScale(
                        alpha: Float(injection.alpha),
                        residualNorm: norm,
                        vectorNorm: vectorNorm)
                } else {
                    alpha = Float(injection.alpha)
                }
                cells.append(
                    CellInjection(
                        layer: layer, vector: vector, alpha: alpha,
                        mode: injection.effectiveMode,
                        concept: injection.concept))
            }
        }
        return cells
    }

    /// Substrate gate at adapter application: refuse an EXPLICITLY
    /// foreign-stamped adapter sidecar (e.g. trained as "hf-peft-lora" on
    /// "python-hf-transformers" — those weights do not load in MLX). The ref
    /// only carries paths, so the stamp is read from the sidecar at
    /// `artifactPath`; an unreadable/absent sidecar carries no stamp and
    /// behaves as before (legacy tolerance).
    static func requireLocallyLoadable(_ adapterRef: ModelVariantArtifact.AdapterRef) throws {
        let sidecarURL = FineTuneStore.absoluteURL(adapterRef.artifactPath)
        guard
            let data = try? Data(contentsOf: sidecarURL),
            let artifact = try? JSONDecoder().decode(FineTuneArtifact.self, from: data)
        else { return }
        if let refusal = AdapterSubstrateGate.refusalMessage(
            name: artifact.name,
            substrate: artifact.substrate,
            adapterFormat: artifact.adapterFormat)
        {
            throw ExperimentError(reason: refusal)
        }
    }

    static func loadAdapter(_ variant: ModelVariantArtifact, into container: ModelContainer) async throws -> LoRAContainer? {
        guard let adapterRef = variant.adapters.first else { return nil }
        try requireLocallyLoadable(adapterRef)
        let adapter = try LoRAContainer.from(
            directory: FineTuneStore.absoluteURL(adapterRef.adapterDirectory))
        try await container.perform { context in
            try adapter.load(into: context.model)
        }
        return adapter
    }

    static func unloadAdapter(_ adapter: LoRAContainer, from container: ModelContainer) async {
        await container.perform { context in
            adapter.unload(from: context.model)
        }
    }

    // MARK: - Shared per-record contract (ordinary + variant conditions)

    /// Built-in outcome-endpoint parses for one sampled record — declared
    /// parser or case-family driven,
    /// data not code. `.some(nil)` is a parse FAILURE (encoded as JSON null);
    /// `.none` means the key does not apply. SHARED by the ordinary and
    /// variant condition paths so the per-record contract cannot fork.
    /// A DECLARED registry parser (`numericParser`, resolved once at run
    /// start) wins over the historical caseFamily rule — same dispatch as
    /// the server's `_execute_condition`.
    static func judicialParses(
        output: String, options: [String]?, caseFamily: String?,
        numericParser: ParserRegistry.ResolvedNumericParser? = nil,
        hitTokenCap: Bool = false
    ) -> (parsedMonths: Double??, parsedChoice: String??) {
        let parsedMonths: Double?? =
            if let numericParser {
                .some(numericParser.parse(output))
            } else if caseFamily == "sentencing" {
                .some(Judicial.parseMonths(output))
            } else {
                .none
            }
        let parsedChoice: String?? =
            if let options, !options.isEmpty {
                // A capped generation was cut off, not finished — the parser
                // turns it into a counted failure so a declared
                // unparseableEndpoint exclusion sees it (`parseChoice`).
                .some(
                    Judicial.parseChoice(
                        output, options: options, truncated: hitTokenCap))
            } else {
                .none
            }
        return (parsedMonths, parsedChoice)
    }

    /// THE sampled-generation record factory, used by BOTH the ordinary
    /// condition loop and the variant-comparison loop (2026-07-13 parity
    /// repair: the variant path used to stamp nil for every science-layer
    /// field and skip the endpoint parses — variant records now carry the
    /// identical per-record contract, plus their two extra
    /// `variantArtifact*` provenance keys). Engine-pure and unit-tested for
    /// baseline-vs-variant field parity.
    static func sampledGenerationRecord(
        manifest: ExperimentManifest,
        experimentHash: String,
        taskPromptsFile: String,
        taskPromptsHash: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String?,
        // WHICH levels produced `systemPrompt` (2026-08-24 composition
        // ruling). Defaulted so the engine-pure unit tests that predate the
        // ruling keep compiling; both run loops pass the stamp built
        // alongside the effective text (`ArmSystemPrompt`), so a record can
        // never describe a composition its generation did not run under.
        systemPromptComposition: SystemPromptCompositionStamp = .none,
        qwenThinkingEnabled: Bool,
        condition: String,
        seed: UInt64,
        promptIndex: Int,
        prompt: StudyPrompt,
        output: String,
        row: MetricRow,
        variantArtifactPath: String? = nil,
        variantArtifactHash: String? = nil,
        agentPlaygroundTemperature: Double? = nil,
        readerScores: [String: Float]? = nil,
        randomVectorAlgorithm: String? = nil,
        numericParser: ParserRegistry.ResolvedNumericParser? = nil,
        // Defaulted (like `systemPromptComposition`) so engine-pure unit
        // tests that predate the truncation rule keep compiling; both run
        // loops pass the stream's own stop reason (`MeasuredGeneration`).
        hitTokenCap: Bool = false
    ) -> GenerationRecord {
        let parses = judicialParses(
            output: output, options: prompt.options,
            caseFamily: manifest.caseFamily, numericParser: numericParser,
            hitTokenCap: hitTokenCap)
        return GenerationRecord(
            experiment: manifest.name,
            experimentHash: experimentHash,
            modelID: manifest.modelID,
            modelRevision: manifest.modelRevision,
            taskPromptsFile: taskPromptsFile,
            taskPromptsHash: taskPromptsHash,
            promptMode: promptMode.rawValue,
            systemPrompt: systemPrompt,
            systemPromptComposition: systemPromptComposition,
            qwenThinkingEnabled: qwenThinkingEnabled,
            condition: condition,
            seed: seed,
            seedInert: true,
            promptIndex: promptIndex,
            promptID: prompt.id,
            prompt: prompt.text,
            output: output,
            wordCount: row.wordCount,
            distinct2: row.distinct2,
            markerDensity: row.markerDensity,
            variantArtifactPath: variantArtifactPath,
            variantArtifactHash: variantArtifactHash,
            agentPlaygroundTemperature: agentPlaygroundTemperature,
            target: prompt.target,
            anchorMonths: prompt.anchorMonths,
            severity: prompt.severity,
            arm: prompt.arm,
            caseID: prompt.caseID,
            parsedMonths: parses.parsedMonths,
            parsedChoice: parses.parsedChoice,
            readerScores: readerScores,
            randomVectorAlgorithm: randomVectorAlgorithm,
            scriptedTranscript: prompt.transcript != nil ? true : nil,
            transcript: prompt.transcript,
            factors: prompt.factors)
    }

    /// The answer-token-logprob record factory — shared by the ordinary and
    /// variant condition loops for the same parity reason as
    /// `sampledGenerationRecord`.
    static func choiceRecord(
        manifest: ExperimentManifest,
        experimentHash: String,
        taskPromptsFile: String,
        taskPromptsHash: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String?,
        /// Same contract as `sampledGenerationRecord`'s.
        systemPromptComposition: SystemPromptCompositionStamp = .none,
        qwenThinkingEnabled: Bool,
        condition: String,
        promptIndex: Int,
        prompt: StudyPrompt,
        choice: ChoiceResult,
        randomVectorAlgorithm: String? = nil
    ) -> ChoiceRecord {
        // DECLARED target only (open-issues #6, server twin `tasks.py`'s
        // choice-record envelope). Defaulting to `options[0]` stamped every
        // ordinalScale record with the rating ladder's minimum as its
        // "target", so a likert study's citable per-item artifacts described
        // the log-odds of answering "1" as if the study had declared it.
        // Nothing is synthesized here; a genuine choice item that declares no
        // target is refused at run start by `checkResponseFormats`.
        let target = prompt.target?.isEmpty == false ? prompt.target : nil
        // Ordinal-scale aggregation over the declared ladder (the item's
        // options, in order). Stamped only when the manifest declares the
        // instrument AND a known aggregation — verify() enforces the
        // declaration before any measured run (loadVerified), so an absent
        // stamp here can only mean ordinalScale was not declared.
        var ordinalPosition: Double? = nil
        var ordinalDistribution: [Double]? = nil
        if (manifest.outcomeInstruments ?? []).contains("ordinalScale"),
            let aggregation = OrdinalAggregation(
                rawValue: manifest.ordinalAggregation ?? "")
        {
            let distribution = LogprobInstrument.ordinalDistribution(
                choice.orderedProbabilities)
            ordinalDistribution = distribution
            ordinalPosition = LogprobInstrument.ordinalPosition(
                distribution: distribution, aggregation: aggregation)
        }
        return ChoiceRecord(
            experiment: manifest.name,
            experimentHash: experimentHash,
            modelID: manifest.modelID,
            modelRevision: manifest.modelRevision,
            taskPromptsFile: taskPromptsFile,
            taskPromptsHash: taskPromptsHash,
            promptMode: promptMode.rawValue,
            systemPrompt: systemPrompt,
            systemPromptComposition: systemPromptComposition,
            qwenThinkingEnabled: qwenThinkingEnabled,
            condition: condition,
            promptIndex: promptIndex,
            promptID: prompt.id,
            prompt: prompt.text,
            target: target,
            targetSource: target != nil ? "declared" : nil,
            anchorMonths: prompt.anchorMonths,
            severity: prompt.severity,
            arm: prompt.arm,
            caseID: prompt.caseID,
            instrument: "answerTokenLogprob",
            options: choice.optionNames,
            optionTokenCounts: choice.tokenCountsByOption,
            optionLengthRatio: choice.optionLengthRatio,
            optionTokenIDs: choice.tokenIDsByOption,
            optionTokenLogprobs: choice.tokenLogprobsByOption,
            optionLogprobs: choice.logprobByOption,
            optionMeanTokenLogprobs: choice.meanTokenLogprobByOption,
            choiceProbability: choice.probability,
            logOdds: choice.logOdds,
            selected: choice.selected,
            margin: choice.margin,
            ordinalPosition: ordinalPosition,
            ordinalDistribution: ordinalDistribution,
            randomVectorAlgorithm: randomVectorAlgorithm,
            scriptedTranscript: prompt.transcript != nil ? true : nil,
            transcript: prompt.transcript,
            factors: prompt.factors)
    }

    // MARK: - run

    /// The ordinary (non-variant) condition list a run executes — baseline
    /// plus the manifest's injection conditions WHEN the concept machinery
    /// is operative. Engineer finding 2026-07-19: a compare-agents study
    /// with no agents said "Baseline only" in the UI while this path
    /// silently executed CARRIED concept conditions; inert configuration
    /// must be inert at run time too. Pure — unit-tested engine-side
    /// without a model.
    static func ordinaryRunConditions(
        for manifest: ExperimentManifest
    ) -> [ExperimentManifest.Condition] {
        let operative = ExperimentStore.conceptMachineryOperative(manifest)
            ? manifest.conditions : []
        if operative.isEmpty {
            return [ExperimentManifest.Condition(name: "baseline", slots: [])]
        }
        return operative.contains { $0.name == "baseline" }
            ? operative
            : [ExperimentManifest.Condition(name: "baseline", slots: [])]
                + operative
    }

    /// Every condition name that will key a record, in execution order.
    ///
    /// `resume.record_key` (both engines) LEADS with the condition name, so
    /// this list IS a run's record-identity vocabulary: shard membership,
    /// resume skipping, and merge completeness all speak it. Two executed
    /// conditions sharing a name alias into one key and cannot be told apart
    /// in the outputs.
    ///
    /// Derived from `ordinaryRunConditions` rather than the declared
    /// collections, because the two differ in both directions: the baseline
    /// is IMPLICIT (in no declared collection, yet always executed), and
    /// carried-but-inert configuration is declared yet never executed
    /// (external review round 11). The Python twin is
    /// `manifest.effective_condition_names`.
    static func effectiveConditionNames(
        for manifest: ExperimentManifest
    ) -> [String] {
        // A MULTI-AGENT study executes neither collection: it runs a
        // scenario, and its record identity is the panel's own fixed
        // vocabulary. It may legally CARRY model-output configuration under
        // the never-delete rule, so evaluating it against the model-output
        // matrix invented conditions it never runs — a carried variant named
        // "baseline" resolved to ["baseline", "baseline"] and refused a legal
        // manifest (external review round 12).
        guard manifest.studyKind == .modelOutput else {
            return manifest.multiAgentIncludeBaseline
                ? ["baseline", "configured"] : ["configured"]
        }
        // A variant comparison runs baseline + variants only; carried
        // injection conditions are inert (the runVariantComparison rule).
        let ordinary = manifest.variantConditions.isEmpty
            ? ordinaryRunConditions(for: manifest).map(\.name)
            : ["baseline"]
        return ordinary + manifest.variantConditions.map(\.name)
    }

    /// Run-status tracking shell around `runImpl` (2026-07-27): the local
    /// engine writes the same cross-engine `run-status.json` + `FAILED.md`
    /// contract the server does, at the same lifecycle points — in-progress
    /// once the run directory exists, terminal status on the way out, the
    /// failure annotated before the error is rethrown. Without it a failed
    /// local run left real artifacts in a directory with NO status, which
    /// `RunStatusFile.isPartial` read as legacy — i.e. trusted. The tracker
    /// learns the directory from the task's own `.runDirectory` progress
    /// event, which every branch (ordinary, variant comparison, multi-agent)
    /// reports.
    @discardableResult
    public static func run(
        experimentName: String,
        promptsFile: String? = nil,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        progress: StudyTaskProgressHandler? = nil
    ) async throws -> URL {
        let status = RunStatusFile.Tracker(
            stage: "run", experiment: experimentName,
            itemLabel: "record", itemsFile: "generations.jsonl")
        let tracked: StudyTaskProgressHandler = { event in
            switch event {
            case .runDirectory(let path):
                await status.begin(directoryPath: path)
            case .generationCompleted:
                await status.noteItem()
            default:
                break
            }
            await progress?(event)
        }
        do {
            let url = try await runImpl(
                experimentName: experimentName, promptsFile: promptsFile,
                shouldCancel: shouldCancel, progress: tracked)
            await status.finish()
            return url
        } catch {
            await status.fail(error)
            throw error
        }
    }

    private static func runImpl(
        experimentName: String,
        promptsFile: String?,
        shouldCancel: (@Sendable () async -> Bool)?,
        progress: StudyTaskProgressHandler?
    ) async throws -> URL {
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        var manifest = try loadVerified(experimentName)
        let cancel = CancelPoller(shouldCancel)
        if manifest.studyKind == .multiAgent {
            return try await runMultiAgentStudy(
                manifest: manifest, cancel: cancel, progress: progress)
        }
        let taskPrompts = try loadTaskPrompts(for: manifest, override: promptsFile)
        guard !taskPrompts.prompts.isEmpty else {
            throw ExperimentError(reason: "task prompt file has no prompts")
        }
        // Run-START transcript gates (rawCompletion + family constraints),
        // BEFORE the model loads — covers the ordinary AND variant branches.
        try checkTranscriptPrompts(taskPrompts.prompts, manifest: manifest)
        // Exclusion-rule preflight at run START (same rule as transcripts):
        // a malformed rule declaration, or failedAttentionCheck with no
        // checked items, refuses before any generation compute — the rules
        // join the paired statistics, but a run whose analysis is doomed
        // should not spend GPU time. Covers both branches.
        try ExclusionEngine.preflight(
            rules: manifest.exclusionRules,
            checks: attentionChecks(of: taskPrompts.prompts))
        // Ladder-window advisory (2026-08-06): an outOfRange keep-window
        // whose bounds cannot bind the scale the items' options imply
        // (min 0 / max 100 on a 1–7 ladder) is legal but inert — worth a
        // loud line before compute, never a refusal (the endpoint may
        // lawfully take non-ladder values).
        for warning in ExclusionEngine.ladderWarnings(
            rules: manifest.exclusionRules,
            optionLadders: taskPrompts.prompts.compactMap(\.options))
        {
            print("warning: \(warning)")
        }
        // Response-format gate, also BEFORE the model loads and covering both
        // branches: an answer-token instrument pointed at rows that ask for a
        // JSON object scores the opening brace's position, not the choice.
        // That is a silently wrong measurement rather than a failure, which
        // is why it refuses rather than warns.
        try checkResponseFormats(taskPrompts.prompts, manifest: manifest)
        // A declared-but-empty agent comparison carrying injection
        // conditions would measure baseline only while every declared arm
        // silently vanished (observed live 2026-08-11 on the server engine;
        // same inert-machinery rule here). verify() reports the same
        // violation; drafts only warn there, so the run refuses here —
        // before the model loads.
        if let inertProblem = ExperimentStore.inertConditionsProblem(manifest) {
            throw ExperimentError.refusing(
                .inertConditions, inertProblem,
                repair: "steerlab-cli experiment declare-condition "
                    + "\(manifest.name) <arm> --slots <concept>:<layer>:<alpha>  "
                    + "(or change the declared studyType, which is what is "
                    + "inerting the arms you already declared)")
        }
        // The other road to a silent baseline-only run (WP0 dry run #0,
        // P0-2): a concept study whose arms were never declared at all.
        // Same place, same reason — before the model loads.
        if let nothingToMeasure = ExperimentStore.noMeasuredConditionsProblem(manifest) {
            throw ExperimentError.refusing(
                .inertConditions, nothingToMeasure,
                repair: "steerlab-cli experiment declare-condition "
                    + "\(manifest.name) <arm> --slots <concept>:<layer>:<alpha>  "
                    + "(a concept study with no injection arm runs the "
                    + "implicit baseline alone and measures nothing)")
        }
        // The third road to a silently incomplete run (2026-08-28 audit,
        // F12): SAE latent arms, which this engine carries but cannot
        // execute. Same place, same reason — before the model loads. Only
        // the run path refuses; verify and freeze stay open, because a
        // latent study authored here and submitted to a server is legal.
        if let latentProblem = ExperimentStore.latentArmsNotExecutableProblem(manifest) {
            throw ExperimentError.refusing(
                .inertConditions, latentProblem,
                repair: "steerlab-cli remote package \(manifest.name) && "
                    + "steerlab-cli remote submit-bundle <bundle> --verb run "
                    + "(--site <id> | --url <server>)  — the Python server "
                    + "executes SAE latent arms; this engine only carries "
                    + "them")
        }
        // The shape that legally proceeds (agent arms exist) while carrying
        // inert concepts/conditions must be LOUD at start — a baseline-only
        // result otherwise looks completed and ordinary (2026-08-11).
        // makeRunDirectory stamps the same fact into config.json's notes.
        if let inertNote = ExperimentStore.inertMachineryNote(manifest) {
            print("WARNING: \(inertNote)")
        }
        let conditions = ordinaryRunConditions(for: manifest)
        guard !conditions.isEmpty else {
            throw ExperimentError.refusing(
                .inertConditions, "study has no conditions",
                repair: "steerlab-cli experiment declare-condition "
                    + "\(manifest.name) <arm> --slots <concept>:<layer>:<alpha>")
        }
        if !manifest.variantConditions.isEmpty {
            return try await runVariantComparison(
                manifest: manifest,
                taskPrompts: taskPrompts,
                conditions: manifest.variantConditions,
                cancel: cancel,
                progress: progress)
        }
        try requireGreedyLocalDesign(manifest)

        let container = try await loadContainer(pinning: &manifest)
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "run")
        await progress?(.runDirectory(runDirectory.path))
        // Inert concept machinery is inert here too: a compare-agents
        // study never re-derives carried concepts' vectors (whose stimuli
        // may not even exist any more).
        let extractions =
            ExperimentStore.conceptMachineryOperative(manifest)
            ? try await extractAll(
                manifest: manifest, container: container, into: runDirectory,
                cancel: cancel)
            : [:]
        if await cancel.observed(at: "after extraction") {
            writeCancellationNote(task: "study run", to: runDirectory)
            print("study run cancelled by user — no generations were produced")
            return runDirectory
        }
        let rubrics = Dictionary(
            uniqueKeysWithValues: manifest.concepts.compactMap { ref in
                let directory = VectorCatalog.conceptsDirectory.appending(component: ref.name)
                return MarkerRubric(directory: directory).map { (ref.name, $0) }
            })
        let conceptNames = manifest.concepts.map(\.name)
        // Reasoning-style scoring rides on sampled text (no model access):
        // loaded up front so a drifted/broken taxonomy fails BEFORE
        // generation, scored per generation into metrics/report.
        let style = try ExperimentStore.loadPinnedReasoningStyle(manifest)
        // Declared numeric-answer parser (registry data, not code): resolved
        // once here — a missing/malformed registry or a drifted pin refuses
        // the run at START, never mid-generation. nil = the historical
        // caseFamily path (server `_run_impl` twin).
        let numericParser = try ParserRegistry.resolveNumericParser(manifest)
        // …and when nothing was declared, the DEPRECATED caseFamily trigger is
        // what chose this run's numeric endpoint. Said at START, where the
        // rest of the run's configuration is reported — never once per record
        // (server `_advise_implicit_case_family` twin).
        if numericParser == nil, manifest.usesImplicitCaseFamilyEndpoint {
            emitRunAdvisory(
                ExperimentManifest.implicitCaseFamilyAdvisory,
                to: runDirectory)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let generationsURL = runDirectory.appending(component: "generations.jsonl")
        FileManager.default.createFile(atPath: generationsURL.path, contents: nil)
        let generationsHandle = try FileHandle(forWritingTo: generationsURL)
        defer { try? generationsHandle.close() }

        var rows: [MetricRow] = []
        var reportChoiceReadouts: [ReportChoiceReadout] = []
        // Sampled-record + instrument-readout views for the declared
        // exclusion rules (empty and unused when the manifest declares
        // none). Scope allRecordTypes: endpoint rules read endpoints the
        // record itself carries, so instrument views extract the same
        // declared endpoint names from the encoded record bytes.
        var exclusionViews: [ExclusionEngine.RecordView] = []
        var instrumentExclusionViews: [ExclusionEngine.InstrumentRecordView] = []
        let exclusionEndpointNames = Set(
            (manifest.exclusionRules ?? [])
                .filter { $0.rule != ExclusionEngine.ruleFailedAttentionCheck }
                .map(ExclusionEngine.resolvedEndpoint))
        let experimentHash = ExperimentStore.manifestHash(manifest)
        // Outcome-instrument dispatch (server `_run_impl` twin): the choice
        // instrument runs once per (condition, prompt); sampled text runs by
        // default and can be switched off for choice-only studies.
        let instruments = Set(manifest.outcomeInstruments ?? [])
        let wantsChoice = !instruments.isDisjoint(with: choiceInstruments)
        // RepE reader scoring rides on sampled text: each output is re-read
        // through every pinned reader and stamped as readerScores (server
        // `_run_impl` twin, incl. repeReaderScore implying sampled output).
        let wantsSampled = instruments.isEmpty || instruments.contains("sampledText")
            || instruments.contains("repeReaderScore")
        var readerScorers: [(concept: String, reader: RepEReader.Artifact)] = []
        if instruments.contains("repeReaderScore") {
            readerScorers = try loadReaderScorers(manifest)
        }
        // Battery-in-run: a pinned capability battery is scored under EVERY
        // condition of the measured run (verify() already checked the pin;
        // loadVerified hard-fails on drift). Battery readings go to a
        // separate battery.jsonl — they are capability controls, never study
        // outputs.
        //
        // Arming follows the battery's FORMAT (server `_run_capability_
        // battery` twin): a format-2 battery declares its own rendering
        // context and token cap and EVERY condition is scored under it, so
        // the intervention is the only thing that varies; a legacy battery
        // keeps the manifest's context — and says so once, loudly, when a
        // study system prompt is in play.
        var runBattery: (battery: CapabilityBattery, hash: String)?
        var batteryArming: BatteryArming?
        var batterySummaries: [String: CapabilityBatterySummary] = [:]
        var batteryHandle: FileHandle?
        defer { try? batteryHandle?.close() }
        // Every arm of THIS loop is agent-free — baseline and steering
        // conditions carry no persona — so the 2026-08-24 composition
        // degrades to the study frame itself, byte for byte what this loop
        // has always rendered and stamped. It goes through `ArmSystemPrompt`
        // anyway so there is ONE place per loop where an arm's arming is
        // resolved, and so the record stamp is built from the same two levels
        // the generation used (server `_effective_ordinary_condition` twin).
        let armSystemPrompt = ArmSystemPrompt(
            agent: nil, study: manifest.systemPrompt)
        // No comparability advisory here, by construction rather than by
        // omission: every arm of this loop resolves the SAME
        // `armSystemPrompt`, so `SystemPromptComposition.divergenceAdvisory`
        // could only ever return nil. `runVariantComparison` — the loop that
        // can mix agent arms with a bare baseline — emits it.

        if let file = manifest.capabilityBatteryFile,
            let pinned = manifest.capabilityBatteryHash
        {
            let battery = try CapabilityBattery(
                url: ExperimentStore.resolveProjectPath(file))
            guard !battery.items.isEmpty else {
                throw ExperimentError(reason: "capability battery '\(file)' is empty")
            }
            runBattery = (battery, pinned)
            // `systemPrompt:` is the FORMAT-1 caller context, unchanged so a
            // legacy battery's pinned hash keeps its historical meaning. No
            // `agentSystemPrompt:` — these arms have no persona, so a
            // format-2 reading here is armed by the battery file alone.
            let arming = battery.resolveArming(
                promptMode: manifest.promptMode ?? .chatAssistant,
                systemPrompt: manifest.systemPrompt,
                qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false)
            batteryArming = arming
            if let advisory = battery.contaminationAdvisory(arming) {
                print("WARNING: \(advisory)")
            }
            let batteryURL = runDirectory.appending(component: "battery.jsonl")
            FileManager.default.createFile(atPath: batteryURL.path, contents: nil)
            batteryHandle = try FileHandle(forWritingTo: batteryURL)
        }
        var cancelled = false
        conditionLoop: for condition in conditions {
            let conditionInjections = try injections(for: condition, extractions: extractions)
            // Recipe stamp for random-control records: which algorithm
            // generated the injected direction (nil ⇒ key omitted; an
            // unstamped random-control record is a legacy cube-uniform run).
            let randomAlgorithm: String? =
                condition.controlType == "randomMatchedNorm"
                ? SteeringVectorMath.randomVectorAlgorithm : nil

            // Answer-token logprob instrument: one deterministic,
            // temperature-free readout per (condition, prompt) — the primary
            // categorical endpoint. Choice records carry no seed and never
            // enter metrics.csv (server behavior).
            if wantsChoice {
                for (index, prompt) in taskPrompts.prompts.enumerated() {
                    guard let options = prompt.options, !options.isEmpty,
                        // A declared applicability scope is honored HERE, not
                        // merely validated at run start: declining to measure
                        // out-of-scope rows is the whole point of declaring
                        // one (ResponseFormat.Scope).
                        manifest.outcomeInstrumentScope?.includes(
                            .init(
                                id: prompt.id, hasOptions: true,
                                format: prompt.responseFormat)) ?? true
                    else { continue }
                    if await cancel.observed(
                        at: "\(condition.name) choice \(index + 1)/\(taskPrompts.prompts.count)")
                    {
                        cancelled = true
                        break conditionLoop
                    }
                    await progress?(
                        .generationStarted(
                            condition: condition.name,
                            promptID: prompt.id,
                            prompt: prompt.text))
                    let choice = try await LogprobInstrument.scoreOptions(
                        container, prompt: prompt.text, options: options,
                        modelID: manifest.modelID,
                        injections: conditionInjections,
                        promptMode: manifest.promptMode ?? .chatAssistant,
                        systemPrompt: armSystemPrompt.effective,
                        qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false,
                        transcript: prompt.transcript)
                    try checkOptionLengths(
                        choice, manifest: manifest, promptID: prompt.id)
                    let record = choiceRecord(
                        manifest: manifest,
                        experimentHash: experimentHash,
                        taskPromptsFile: taskPrompts.file,
                        taskPromptsHash: taskPrompts.hash,
                        promptMode: manifest.promptMode ?? .chatAssistant,
                        systemPrompt: armSystemPrompt.effective,
                        systemPromptComposition: armSystemPrompt.stamp,
                        qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false,
                        condition: condition.name,
                        promptIndex: index + 1,
                        prompt: prompt,
                        choice: choice,
                        randomVectorAlgorithm: randomAlgorithm)
                    reportChoiceReadouts.append(
                        ReportChoiceReadout(
                            condition: condition.name,
                            promptID: prompt.id,
                            sampleIndex: nil,
                            source: "instrument",
                            selected: record.selected,
                            target: record.target,
                            ordinalPosition: record.ordinalPosition))
                    let recordData = try encoder.encode(record)
                    if manifest.exclusionRules?.isEmpty == false {
                        instrumentExclusionViews.append(
                            ExclusionEngine.InstrumentRecordView(
                                condition: condition.name,
                                promptID: prompt.id,
                                endpoints: analysisEndpoints(
                                    jsonLine: recordData,
                                    names: exclusionEndpointNames)))
                    }
                    try generationsHandle.write(contentsOf: recordData)
                    try generationsHandle.write(contentsOf: Data("\n".utf8))
                    let probability = choice.probability[choice.selected] ?? 0
                    let summary = String(
                        format: "answerTokenLogprob: selected=%@ p=%.4f margin=%.4f",
                        choice.selected, probability, choice.margin)
                    await progress?(
                        .generationCompleted(
                            StudyGenerationPreview(
                                condition: condition.name,
                                promptID: prompt.id,
                                prompt: prompt.text,
                                output: summary,
                                wordCount: 0,
                                distinct2: 0,
                                markerDensity: [:],
                                truncated: false)))
                    print(
                        "\(condition.name) choice \(prompt.id) "
                            + "(\(index + 1)/\(taskPrompts.prompts.count)): \(summary)")
                }
            }

            // Battery under this condition's interventions, armed by the
            // battery's own format (`runBattery`: generation + text match for
            // a legacy file, the answer-token logprob instrument for a
            // format-2 choice item). A cancelled battery drops the whole
            // condition's battery cell — a partially scored battery is never
            // reported as an accuracy.
            if let (battery, batteryHash) = runBattery, let batteryHandle,
                let arming = batteryArming
            {
                guard
                    let scored = try await ExperimentTasks.runBattery(
                        container, battery: battery, batteryHash: batteryHash,
                        condition: condition.name, modelID: manifest.modelID,
                        injections: conditionInjections, arming: arming,
                        cancel: { _ in
                            await cancel.observed(at: "\(condition.name) battery")
                        })
                else {
                    cancelled = true
                    break conditionLoop
                }
                for record in scored.records {
                    try batteryHandle.write(contentsOf: encoder.encode(record))
                    try batteryHandle.write(contentsOf: Data("\n".utf8))
                }
                batterySummaries[condition.name] = scored.summary
                print(
                    "capability battery \(condition.name): "
                        + "\(Int((scored.summary.accuracy * Double(scored.summary.itemCount)).rounded()))"
                        + "/\(scored.summary.itemCount) "
                        + String(format: "(%.0f%%)", scored.summary.accuracy * 100))
            }

            guard wantsSampled else { continue }
            for seed in manifest.seeds {
                for (index, prompt) in taskPrompts.prompts.enumerated() {
                    if await cancel.observed(
                        at: "\(condition.name) seed \(seed) prompt "
                            + "\(index + 1)/\(taskPrompts.prompts.count)")
                    {
                        cancelled = true
                        break conditionLoop
                    }
                    await progress?(
                        .generationStarted(
                            condition: condition.name,
                            promptID: prompt.id,
                            prompt: prompt.text))
                    let generation = try await generateMeasured(
                        container, prompt: prompt.text, modelID: manifest.modelID,
                        maxTokens: manifest.maxTokens, temperature: manifest.temperature,
                        injections: conditionInjections,
                        promptMode: manifest.promptMode ?? .chatAssistant,
                        systemPrompt: armSystemPrompt.effective,
                        qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false,
                        transcript: prompt.transcript
                    ) { output in
                        await progress?(
                            .generationChunk(
                                condition: condition.name,
                                promptID: prompt.id,
                                output: output))
                    }
                    let output = generation.text
                    let markerDensity = Dictionary(
                        uniqueKeysWithValues: conceptNames.map { concept in
                            (concept, rubrics[concept]?.density(in: output) ?? 0)
                        })
                    let row = MetricRow(
                        condition: condition.name, seed: seed, promptIndex: index + 1,
                        promptID: prompt.id, wordCount: wordCount(output),
                        distinct2: distinctBigramRatio(output),
                        markerDensity: markerDensity,
                        reasoningStyle: style.map { $0.taxonomy.score(output) } ?? [:],
                        factors: prompt.factors ?? [:])
                    rows.append(row)
                    // Reader readout of the sampled output. The scaffold
                    // capture must be UNSTEERED (the reader measures what the
                    // text expresses, not what the injection pushes) — clear
                    // this condition's injectors first; the next `generate`
                    // reinstalls them per prompt. Mirrors the server, whose
                    // scoring path installs no injection hooks.
                    var readerScores: [String: Float]?
                    if !readerScorers.isEmpty {
                        try await setInterventions(container, [])
                        var scores: [String: Float] = [:]
                        for scorer in readerScorers {
                            scores[scorer.concept] = try await RepEReader.scoreText(
                                container: container, modelID: manifest.modelID,
                                reader: scorer.reader, text: output)
                        }
                        readerScores = scores
                    }
                    let record = sampledGenerationRecord(
                        manifest: manifest,
                        experimentHash: experimentHash,
                        taskPromptsFile: taskPrompts.file,
                        taskPromptsHash: taskPrompts.hash,
                        promptMode: manifest.promptMode ?? .chatAssistant,
                        systemPrompt: armSystemPrompt.effective,
                        systemPromptComposition: armSystemPrompt.stamp,
                        qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false,
                        condition: condition.name,
                        seed: seed,
                        promptIndex: index + 1,
                        prompt: prompt,
                        output: output,
                        row: row,
                        readerScores: readerScores,
                        randomVectorAlgorithm: randomAlgorithm,
                        numericParser: numericParser,
                        hitTokenCap: generation.hitTokenCap)
                    if manifest.exclusionRules?.isEmpty == false {
                        exclusionViews.append(exclusionView(of: record))
                    }
                    // Single-source rule: the readout reads the RECORD's own
                    // stamped parse (`parsedChoice`, written by
                    // `judicialParses` inside the record factory) — never a
                    // re-parse of the output — so report.json and
                    // generations.jsonl cannot diverge by construction (the
                    // server derives its readouts from records the same way).
                    // `.some(.some(…))` = options applied AND parse succeeded;
                    // a parse FAILURE (`.some(nil)`) can neither agree nor
                    // disagree and stays out of the readouts (server rule).
                    if case .some(.some(let selected)) = record.parsedChoice {
                        reportChoiceReadouts.append(
                            ReportChoiceReadout(
                                condition: condition.name,
                                promptID: record.promptID,
                                sampleIndex: seed,
                                source: "parsed",
                                selected: selected,
                                target: record.target))
                    }
                    try generationsHandle.write(contentsOf: encoder.encode(record))
                    try generationsHandle.write(contentsOf: Data("\n".utf8))
                    await progress?(.generationCompleted(generationPreview(from: record)))
                    print(
                        "\(condition.name) seed \(seed) prompt \(index + 1)/"
                            + "\(taskPrompts.prompts.count): \(row.wordCount) words")
                }
            }
        }
        // A cancel-break can leave the last condition's injectors armed —
        // disarm before anything else touches the container.
        try await setInterventions(container, [])

        try metricsCSV(
            rows: rows, concepts: conceptNames,
            styleFeatureIDs: style?.taxonomy.featureIDs ?? []
        ).write(
            to: runDirectory.appending(component: "metrics.csv"),
            atomically: true, encoding: .utf8)
        if cancelled {
            // Honest partial: generations.jsonl / battery.jsonl hold what
            // completed, metrics.csv covers the completed rows, and the
            // status note marks the directory. NO report.json — the run is
            // mechanically invisible to newestCompletedRun / analyze /
            // evaluate, exactly like the server's cancelled runs.
            writeCancellationNote(task: "study run", to: runDirectory)
            print(
                "study run cancelled by user — partial artifacts kept in "
                    + runDirectory.lastPathComponent)
            return runDirectory
        }
        let report = report(
            experiment: manifest, experimentHash: experimentHash,
            taskPrompts: taskPrompts, rows: rows, conditionCount: conditions.count,
            concepts: conceptNames, batterySummaries: batterySummaries,
            choiceReadouts: reportChoiceReadouts, style: style,
            numericParser: numericParser?.provenance,
            exclusions: exclusionOutcome(
                manifest: manifest, prompts: taskPrompts.prompts,
                views: exclusionViews,
                instrumentViews: instrumentExclusionViews))
        if let entries = report.effectSizes, !entries.isEmpty {
            try effectSizesCSV(entries).write(
                to: runDirectory.appending(component: "effect-sizes.csv"),
                atomically: true, encoding: .utf8)
        }
        try encoder.encode(report).write(to: runDirectory.appending(component: "report.json"))
        print("run artifacts: \(runDirectory.path)")
        return runDirectory
    }

    /// Everything a multi-agent run does BEFORE it generates: resolve the
    /// pinned scenario, refuse drift, mint the run directory, and snapshot
    /// the scenario there verbatim.
    ///
    /// A seam rather than inline code so the artifact contract is testable
    /// without loading a model — the run path calls exactly this, in this
    /// order, and the order is the point: a drifted scenario must leave no
    /// run directory behind, and the snapshot must exist before the first
    /// turn is generated.
    ///
    /// Server twin: `tasks._run_multi_agent_study`'s prologue.
    static func prepareMultiAgentRun(
        manifest: ExperimentManifest
    ) throws -> (
        scenario: MultiAgentScenario, scenarioPath: String, scenarioURL: URL,
        scenarioHash: String, runDirectory: URL
    ) {
        guard let scenarioPath = manifest.multiAgentScenarioPath,
            !scenarioPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw ExperimentError(reason: "multi-agent study needs a selected scenario")
        }
        let scenarioURL = scenarioURL(from: scenarioPath)
        // One read serves the parse, the hash and the snapshot below, so the
        // three cannot disagree about what ran.
        let data = try Data(contentsOf: scenarioURL)
        let scenarioHash = MultiAgentScenarioStore.hash(data)
        // Drift in a pinned input is a violation, never a silent copy: the
        // snapshot is only evidence if the bytes it preserves are the bytes
        // the study pinned. Checked before the run directory exists, so a
        // drifted scenario produces no half-run to explain away.
        if let pinned = manifest.multiAgentScenarioHash, pinned != scenarioHash {
            throw ExperimentError(
                reason: "multi-agent scenario '\(scenarioPath)' changed since pinning "
                    + "(have \(scenarioHash.prefix(12))…, pinned \(pinned.prefix(12))…) — "
                    + "refusing to run '\(manifest.name)' against an input its manifest "
                    + "does not describe")
        }
        let scenario = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "multi-agent-run")
        // Snapshot the scenario VERBATIM beside experiment.json, before a
        // single turn is generated. experiment.json only POINTS at the
        // scenario, but the seat→variant attribution
        // (agents[].variantArtifactPath/Hash) lives in the scenario itself —
        // so without this a finished run cannot answer "which seat carried
        // which agent variant" without the live workspace file, and a reader
        // that sees only the run directory (the Results Explorer's bridge
        // serves runs/) cannot answer it at all. Raw bytes, not a re-encode:
        // a re-serialised copy drops unknown keys, re-orders known ones, and
        // no longer hashes to the pin it was just checked against.
        try data.write(to: runDirectory.appending(component: "scenario.json"))
        return (scenario, scenarioPath, scenarioURL, scenarioHash, runDirectory)
    }

    private static func runMultiAgentStudy(
        manifest: ExperimentManifest,
        cancel: CancelPoller = CancelPoller(nil),
        progress: StudyTaskProgressHandler?
    ) async throws -> URL {
        // Deliberately NOT gated by `requireGreedyLocalDesign` (which the
        // ordinary and variant study paths call): a warm multi-agent run is
        // supported locally as EXPLORATION. MLX samples warm fine; what it
        // cannot do is seed, so every record here stays `seedInert: true` and
        // the transcripts are varied but not re-runnable. Stochastic EVIDENCE
        // comes from the server, which seeds per turn — see
        // MULTI-AGENT-SUBSTRATE-PARITY-PLAN § A5.
        guard manifest.temperature >= 0 else {
            throw ExperimentError(
                reason: "temperature must be >= 0, got \(manifest.temperature)")
        }
        let prepared = try prepareMultiAgentRun(manifest: manifest)
        let scenarioPath = prepared.scenarioPath
        let scenarioURL = prepared.scenarioURL
        let scenarioHash = prepared.scenarioHash
        let scenario = prepared.scenario
        let runDirectory = prepared.runDirectory
        let experimentHash = ExperimentStore.manifestHash(manifest)
        await progress?(.runDirectory(runDirectory.path))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let generationsURL = runDirectory.appending(component: "generations.jsonl")
        FileManager.default.createFile(atPath: generationsURL.path, contents: nil)
        let generationsHandle = try FileHandle(forWritingTo: generationsURL)
        defer { try? generationsHandle.close() }

        let prompt = multiAgentEvaluationPrompt(scenario: scenario)
        // No system-prompt divergence advisory here, deliberately (2026-08-24
        // casting ruling), and the server's panel path says the same. The
        // advisory's unit is the ARM, and a panel's two arms cannot diverge in
        // effective system content: the split is `stripInterventions`, which
        // drops injections and the adapter and never touches a seat's persona
        // or role text (`MultiAgentRunner.runtimeSettings`). Seats WITHIN a
        // transcript are armed differently on purpose — that is what casting
        // is — so per-seat divergence is the design, not a finding; the
        // per-turn `systemPromptComposition` stamp is what records it.
        let conditionSpecs: [(name: String, strip: Bool)] =
            manifest.multiAgentIncludeBaseline
            ? [("baseline", true), ("configured", false)]
            : [("configured", false)]
        // Reasoning-style scoring is study-kind-agnostic: a pinned taxonomy
        // scores the whole transcript per condition (drift fails up front).
        let style = try ExperimentStore.loadPinnedReasoningStyle(manifest)
        var rows: [MetricRow] = []

        // Replicates: the STUDY manifest owns sampling policy, so samplesPerItem
        // is the replicate count and its temperature overrides the scenario's
        // authoring value. Each replicate is an independent play-through.
        // samplesPerItem is optional here (the server defaults it to 1);
        // absent means one play-through, the historical behaviour.
        let replicates = max(1, manifest.samplesPerItem ?? 1)
        var cancelled = false
        conditionLoop: for condition in conditionSpecs {
          for replicate in 0 ..< replicates {
            // Cancellation unit = one whole condition transcript (turn-level
            // cancellation would leave a mid-scenario transcript that is not
            // a measurement of anything).
            if await cancel.observed(at: "multi-agent condition '\(condition.name)'") {
                cancelled = true
                break conditionLoop
            }
            await progress?(
                .generationStarted(
                    condition: condition.name,
                    promptID: "scenario",
                    prompt: prompt))
            // Single-replicate runs keep the historical <run>/<condition>/
            // layout so existing consumers are untouched; replicates nest one
            // level deeper (same rule as the server).
            let conditionDirectory =
                replicates == 1
                ? runDirectory.appending(component: condition.name)
                : runDirectory.appending(component: condition.name)
                    .appending(component: "replicate-\(replicate)")
            let turnRunDirectory = try await MultiAgentRunner.run(
                scenario: scenario,
                scenarioURL: scenarioURL,
                runDirectory: conditionDirectory,
                conditionName: condition.name,
                stripInterventions: condition.strip,
                defaultRevision: manifest.modelRevision,
                temperature: manifest.temperature,
                replicateIndex: replicate) { event in
                    switch event {
                    case .turnChunk(_, _, _, let output):
                        await progress?(
                            .generationChunk(
                                condition: condition.name,
                                promptID: "scenario",
                                output: output))
                    default:
                        break
                    }
                }
            // C1: one record per TURN, matching the server's shape. The
            // whole-transcript record this replaced made a run n=1 per
            // condition — no statistics were possible — and scored
            // transcript.md's own chrome (section headers, "Routed to:" agent
            // UUIDs) as if it were model output. turns.jsonl is the source of
            // truth; transcript.md is now a human convenience only.
            let turnsURL = turnRunDirectory.appending(component: "turns.jsonl")
            let turnLines = try String(contentsOf: turnsURL, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: true)
            let decoder = JSONDecoder()
            for line in turnLines {
                let turn = try decoder.decode(
                    MultiAgentTurnResult.self, from: Data(line.utf8))
                let row = MetricRow(
                    condition: condition.name,
                    seed: manifest.seeds.first ?? 20260610,
                    promptIndex: turn.turnIndex - 1,
                    promptID: turn.turnID,
                    wordCount: wordCount(turn.output),
                    distinct2: distinctBigramRatio(turn.output),
                    markerDensity: [:],
                    reasoningStyle: style.map { $0.taxonomy.score(turn.output) } ?? [:],
                    replicate: replicate)
                rows.append(row)
                let record = GenerationRecord(
                    experiment: manifest.name,
                    experimentHash: experimentHash,
                    modelID: scenario.baseModelID,
                    // The revision that actually generated this turn.
                    modelRevision: turn.modelRevision
                        ?? manifest.modelRevision
                        ?? SteeredContainerLoader.cachedRevision(
                            for: scenario.baseModelID),
                    taskPromptsFile: scenarioPath,
                    taskPromptsHash: scenarioHash,
                    promptMode: "multi-agent",
                    systemPrompt: nil,
                    qwenThinkingEnabled: false,
                    condition: condition.name,
                    seed: row.seed,
                    // Local MLX cannot pin a sampling seed, warm or not.
                    seedInert: true,
                    promptIndex: row.promptIndex,
                    promptID: row.promptID,
                    prompt: turn.prompt,
                    output: turn.output,
                    wordCount: row.wordCount,
                    distinct2: row.distinct2,
                    markerDensity: row.markerDensity,
                    variantArtifactPath: nil,
                    variantArtifactHash: nil,
                    speakerName: turn.speakerName,
                    turnTitle: turn.title,
                    routedAgentIDs: turn.routedAgentIDs,
                    replicateIndex: replicate,
                    target: nil,
                    anchorMonths: nil,
                    severity: nil,
                    arm: nil,
                    caseID: nil,
                    parsedMonths: nil,
                    parsedChoice: nil,
                    endpoint: turn.endpoint,
                    voiceLint: turn.voiceLint)
                try generationsHandle.write(contentsOf: encoder.encode(record))
                try generationsHandle.write(contentsOf: Data("\n".utf8))
                await progress?(.generationCompleted(generationPreview(from: record)))
            }
          }
        }

        try metricsCSV(
            rows: rows, concepts: [],
            styleFeatureIDs: style?.taxonomy.featureIDs ?? []
        ).write(
            to: runDirectory.appending(component: "metrics.csv"),
            atomically: true,
            encoding: .utf8)
        if cancelled {
            writeCancellationNote(task: "multi-agent study run", to: runDirectory)
            print(
                "multi-agent study run cancelled by user — partial artifacts "
                    + "kept in \(runDirectory.lastPathComponent)")
            return runDirectory
        }
        // C3: panel-effect decomposition, the Swift twin of the server's
        // panel_effects. Needs BOTH arms of the pair, and — like the server —
        // only runs at one replicate: it pairs ONE configured transcript
        // against ONE baseline transcript, and pooling across replicates is
        // the D1 clustering question, not a mean of means.
        if manifest.multiAgentIncludeBaseline, replicates == 1 {
            writePanelEffects(
                scenario: scenario, runDirectory: runDirectory,
                conditionNames: conditionSpecs.map(\.name))
        } else if manifest.multiAgentIncludeBaseline {
            print(
                "panel effects: skipped — \(replicates) replicates per condition, "
                    + "and pooling across replicates needs the clustered estimator. "
                    + "Re-run with samplesPerItem 1 for the paired "
                    + "single-transcript decomposition.")
        }

        // D1: transcript-level effect sizes. Nil when there is only one
        // transcript per arm — see `transcriptEffectSizes`.
        let entries = transcriptEffectSizes(
            rows: rows, styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
            phase: manifest.phase)
        if let entries, !entries.isEmpty {
            try effectSizesCSV(entries).write(
                to: runDirectory.appending(component: "effect-sizes.csv"),
                atomically: true, encoding: .utf8)
        } else if replicates < 2 {
            print(
                "effect sizes: skipped — 1 transcript per condition. Turns are "
                    + "not independent observations (turn k is conditioned on "
                    + "1..k-1), so the unit of analysis is the transcript and a "
                    + "single one per arm supports no interval. Re-run with "
                    + "samplesPerItem > 1.")
        }
        var byCondition: [String: [MetricRow]] = [:]
        for row in rows { byCondition[row.condition, default: []].append(row) }
        let report = StudyRunReport(
            experiment: manifest.name,
            experimentHash: experimentHash,
            taskPromptsFile: scenarioPath,
            taskPromptsHash: scenarioHash,
            promptMode: "multi-agent",
            systemPrompt: nil,
            qwenThinkingEnabled: false,
            promptCount: scenario.turns.count,
            conditionCount: conditionSpecs.count,
            seedCount: 1,
            conditions: byCondition.mapValues { conditionRows in
                ConditionReport(
                    generations: conditionRows.count,
                    meanWordCount: conditionRows.isEmpty
                        ? 0
                        : Float(conditionRows.reduce(0) { $0 + $1.wordCount })
                            / Float(conditionRows.count),
                    meanDistinct2: conditionRows.isEmpty
                        ? 0
                        : conditionRows.reduce(0) { $0 + $1.distinct2 }
                            / Float(conditionRows.count),
                    meanMarkerDensity: [:],
                    capabilityBattery: nil,
                    reasoningStyle: reasoningStyleReport(rows: conditionRows, style: style))
            },
            effectSizes: entries,
            modelsUsed: [scenario.baseModelID].filter { !$0.isEmpty },
            declaredModelID: manifest.modelID,
            modelBySeat: Dictionary(
                scenario.agents.map { ($0.name, $0.baseModelID) },
                uniquingKeysWith: { first, _ in first }),
            unitOfAnalysis: "transcript",
            transcriptsPerCondition: replicates)
        try encoder.encode(report).write(to: runDirectory.appending(component: "report.json"))
        return runDirectory
    }

    /// Write panel-effects.csv from a completed multi-agent run's two arms.
    /// Best-effort: a decomposition that cannot be built must never sink a run
    /// whose generations already landed.
    private static func writePanelEffects(
        scenario: MultiAgentScenario, runDirectory: URL, conditionNames: [String]
    ) {
        func turns(_ condition: String) -> [MultiAgentTurnResult] {
            let url = runDirectory.appending(components: condition, "turns.jsonl")
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
            let decoder = JSONDecoder()
            return text.split(separator: "\n", omittingEmptySubsequences: true)
                .compactMap { try? decoder.decode(MultiAgentTurnResult.self, from: Data($0.utf8)) }
        }
        let configured = turns("configured")
        let baseline = turns("baseline")
        guard !configured.isEmpty, !baseline.isEmpty else { return }

        let treated = PanelEffects.treatedAgentIDs(in: scenario)
        // wordCount is the always-available endpoint; the server adds parsed
        // months for the sentencing family, which rides the same seam when a
        // Swift-side parser is pinned.
        //
        // The voice-lint roll-up (spec §5) is deliberately NOT here. It is a
        // different grain — one row per speaker × condition, in its own
        // `panel-voice-lint.csv` — and, like the months parser, it is
        // server-side today: Swift stamps every turn identically (see
        // `MultiAgentRunner.voiceLintStamp`) and the server aggregates.
        let exposed = PanelEffects.exposureByTurn(scenario: scenario, treated: treated)
        let rows = [
            PanelEffects.compute(
                configured: configured, baseline: baseline, treated: treated,
                endpoint: "wordCount",
                parse: { Double(wordCount($0)) },
                exposedTurns: exposed)
        ]
        try? PanelEffects.csv(rows).write(
            to: runDirectory.appending(component: "panel-effects.csv"),
            atomically: true, encoding: .utf8)
    }

    /// Multi-agent effect sizes at the TRANSCRIPT level (plan D1).
    ///
    /// Turns are not independent observations: turn k is conditioned on turns
    /// 1..k-1, and after turn 1 the baseline and configured arms diverge, so
    /// what is paired across conditions is SCRIPT POSITION, not matched input.
    /// That divergence is the propagation effect the multi-agent layer exists
    /// to measure — but it has to be modelled, not ignored. Running the
    /// ordinary per-item bootstrap over turns would treat N dependent turns as
    /// N independent draws and report intervals far tighter than the design
    /// earns.
    ///
    /// So each transcript is reduced to its MEAN paired difference first, and
    /// the existing paired bootstrap + Wilcoxon then run over those
    /// transcript-level values. Clusters are balanced by construction (every
    /// transcript plays the same turn script), which is exactly when
    /// cluster-level aggregation is the right estimator — and it needs no new
    /// statistics. `n` in effect-sizes.csv is therefore the number of
    /// TRANSCRIPTS, which is the unit of analysis.
    ///
    /// Returns nil when either arm has fewer than 2 transcripts: a single
    /// transcript supports a point estimate but no interval, and a zero-width
    /// CI would read as certainty rather than as absent replication.
    static func transcriptEffectSizes(
        rows: [MetricRow], styleFeatureIDs: [String], phase: String?
    ) -> [EffectSizeEntry]? {
        var metrics: [(name: String, value: (MetricRow) -> Double)] = [
            ("wordCount", { Double($0.wordCount) }),
            ("distinct2", { Double($0.distinct2) }),
        ]
        for id in styleFeatureIDs {
            metrics.append(("rs_\(id)", { $0.reasoningStyle[id] ?? 0 }))
        }

        // Pair on (replicate, turn id): the same script position in the same
        // play-through index. Never on the seed — the server derives per-turn
        // seeds that include the CONDITION, so seeds differ across arms by
        // design.
        var baseline: [String: MetricRow] = [:]
        for row in rows where row.condition == "baseline" {
            baseline["\(row.replicate ?? 0)::\(row.promptID)"] = row
        }
        guard !baseline.isEmpty else { return nil }

        var conditionOrder: [String] = []
        var seen = Set<String>()
        for row in rows where row.condition != "baseline" {
            if seen.insert(row.condition).inserted { conditionOrder.append(row.condition) }
        }

        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            for metric in metrics {
                // Turn-level differences, grouped by transcript.
                var byTranscript: [Int: [Double]] = [:]
                for row in rows where row.condition == condition {
                    let key = "\(row.replicate ?? 0)::\(row.promptID)"
                    guard let base = baseline[key] else { continue }
                    byTranscript[row.replicate ?? 0, default: []]
                        .append(metric.value(row) - metric.value(base))
                }
                // One value per transcript, in replicate order.
                let perTranscript = byTranscript.keys.sorted().compactMap { key -> Double? in
                    guard let values = byTranscript[key], !values.isEmpty else { return nil }
                    return values.reduce(0, +) / Double(values.count)
                }
                guard perTranscript.count >= 2 else { return nil }
                let ci = StudyStatistics.pairedBootstrapCI(
                    perTranscript, replicates: 10_000, seed: 0)
                let wilcoxon = StudyStatistics.wilcoxonSignedRank(perTranscript)
                entries.append(
                    EffectSizeEntry(
                        condition: condition,
                        metric: metric.name,
                        n: ci.n,
                        meanDiff: ci.mean,
                        ciLower: ci.ciLower,
                        ciUpper: ci.ciUpper,
                        wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                        wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p))
            }
        }
        return applyCorrection(entries, phase: phase)
    }

    private static func scenarioURL(from path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(filePath: path)
        }
        return VectorCatalog.projectRoot.appending(path: path)
    }

    private static func multiAgentEvaluationPrompt(scenario: MultiAgentScenario) -> String {
        var parts = [
            "Scenario: \(scenario.name)",
            scenario.description,
        ].filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        if !scenario.sharedMaterials.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parts.append("Shared materials:\n\(scenario.sharedMaterials)")
        }
        return parts.joined(separator: "\n\n")
    }

    /// The local measured-run sampling gate (CLAUDE.md › Sampling &
    /// measurement policy): local MLX runs are GREEDY-ONLY — the MLX
    /// generator does not pin a per-run sampling seed, so a positive
    /// temperature would produce unreproducible records. Shared by the
    /// ordinary and variant (saved-agent) study paths; stochastic designs
    /// route to the Python server (`SubstrateRouting`), which seeds
    /// PyTorch per record. Extracted so the refusal is unit-testable.
    /// The repair for the local greedy-only policy (WP0 step 7): the study
    /// either becomes deterministic here, or it moves to the substrate that
    /// seeds per record. Both are commands — the stay-local arm became one
    /// when `set-sampling` landed (before it, no CLI verb set the
    /// temperature, so the repair could only say "edit the manifest").
    static func samplingPolicyRepair(_ name: String) -> String {
        "steerlab-cli remote package \(name) && steerlab-cli remote "
            + "submit-bundle <bundle> --verb run (--site <id> | --url <server>)"
            + "  — the Python server seeds PyTorch per record; the local MLX "
            + "generator pins no per-run seed, so a stochastic study runs "
            + "there. To stay local instead: steerlab-cli experiment "
            + "duplicate \(name) \(name)-v2 && steerlab-cli experiment "
            + "set-sampling \(name)-v2 --temperature 0, set one seed, and "
            + "re-freeze."
    }

    static func requireGreedyLocalDesign(_ manifest: ExperimentManifest) throws {
        // E1: the greedy requirement exists because the MLX generator has no
        // per-run sampling seed. A study that never SAMPLES is unaffected by
        // that limitation, so a deterministic-instrument study must not be
        // refused over a temperature nothing reads. The advisory below still
        // says the declared value is inert.
        guard ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
            .samplingIsOperative
        else { return }
        if manifest.temperature == 0, manifest.seeds.count > 1 {
            throw ExperimentError.refusing(
                .samplingPolicy,
                "temperature 0 is greedy and ignores seeds; use exactly one seed "
                    + "or raise the temperature after stochastic sampling is implemented",
                repair: Self.samplingPolicyRepair(manifest.name))
        }
        guard manifest.temperature == 0 else {
            throw ExperimentError.refusing(
                .samplingPolicy,
                "study execution currently requires temperature 0 because "
                    + "mlx-swift-lm GenerateParameters does not expose a per-run seed; "
                    + "set the study temperature to 0 for reproducible greedy runs",
                repair: Self.samplingPolicyRepair(manifest.name))
        }
    }

    private static func runVariantComparison(
        manifest: ExperimentManifest,
        taskPrompts: (file: String, hash: String, prompts: [StudyPrompt]),
        conditions: [ExperimentManifest.VariantCondition],
        cancel: CancelPoller = CancelPoller(nil),
        progress: StudyTaskProgressHandler?
    ) async throws -> URL {
        try requireGreedyLocalDesign(manifest)
        // Study-owned sampling (2026-07-21): the STUDY manifest owns the
        // measured-run sampling policy for every condition; a saved agent's
        // stored temperature is a Playground convenience, non-operative in
        // measured runs (this engine generates greedy at the manifest's
        // required temperature 0 for every condition, and stamps the
        // artifact value as `agentPlaygroundTemperature` provenance). The
        // historical artifact-temperature refusal policed a dead field —
        // and would wrongly refuse agents saved from a warm Playground.
        for condition in conditions {
            guard condition.artifact.baseModelID == manifest.modelID else {
                throw ExperimentError(
                    reason: "variant '\(condition.name)' uses \(condition.artifact.baseModelID), "
                        + "not study base model \(manifest.modelID)")
            }
        }

        var pinnedManifest = manifest
        let container = try await loadContainer(pinning: &pinnedManifest)
        let runDirectory = try makeRunDirectory(experiment: pinnedManifest, task: "run")
        await progress?(.runDirectory(runDirectory.path))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let generationsURL = runDirectory.appending(component: "generations.jsonl")
        FileManager.default.createFile(atPath: generationsURL.path, contents: nil)
        let generationsHandle = try FileHandle(forWritingTo: generationsURL)
        defer { try? generationsHandle.close() }

        var rows: [MetricRow] = []
        var reportChoiceReadouts: [ReportChoiceReadout] = []
        // Sampled-record + instrument-readout views for the declared
        // exclusion rules (empty and unused when the manifest declares
        // none) — same scope-allRecordTypes collection as the ordinary
        // condition loop.
        var exclusionViews: [ExclusionEngine.RecordView] = []
        var instrumentExclusionViews: [ExclusionEngine.InstrumentRecordView] = []
        let exclusionEndpointNames = Set(
            (pinnedManifest.exclusionRules ?? [])
                .filter { $0.rule != ExclusionEngine.ruleFailedAttentionCheck }
                .map(ExclusionEngine.resolvedEndpoint))
        let experimentHash = ExperimentStore.manifestHash(pinnedManifest)
        // Reasoning-style scoring (pinned taxonomy) applies to variant
        // studies exactly as to steering conditions.
        let style = try ExperimentStore.loadPinnedReasoningStyle(pinnedManifest)
        // Declared numeric-answer parser — identical resolution + dispatch
        // to the ordinary condition path (per-record contract parity).
        let numericParser = try ParserRegistry.resolveNumericParser(pinnedManifest)
        // Per-record contract parity with the ordinary condition path
        // (2026-07-13 repair — variant conditions used to bypass the
        // instruments): the same outcome-instrument dispatch, the same
        // marker rubrics, the same reader scorers.
        let rubrics = Dictionary(
            uniqueKeysWithValues: pinnedManifest.concepts.compactMap { ref in
                let directory = VectorCatalog.conceptsDirectory.appending(component: ref.name)
                return MarkerRubric(directory: directory).map { (ref.name, $0) }
            })
        let conceptNames = pinnedManifest.concepts.map(\.name)
        let instruments = Set(pinnedManifest.outcomeInstruments ?? [])
        let wantsChoice = !instruments.isDisjoint(with: choiceInstruments)
        let wantsSampled = instruments.isEmpty || instruments.contains("sampledText")
            || instruments.contains("repeReaderScore")
        var readerScorers: [(concept: String, reader: RepEReader.Artifact)] = []
        if instruments.contains("repeReaderScore") {
            readerScorers = try loadReaderScorers(pinnedManifest)
        }

        // Battery-in-run (variant studies always have a pinned battery by
        // freeze time; drafts may not — pinned only). A format-2 battery's
        // arming is resolved ONCE here and reused for every condition,
        // baseline and variants alike: that identity is what makes the
        // capability control comparable (server `_run_capability_battery`).
        // A legacy battery still resolves per condition, from that
        // condition's own rendering context.
        var runBattery: (battery: CapabilityBattery, hash: String)?
        var batteryAdvised = false
        var batterySummaries: [String: CapabilityBatterySummary] = [:]
        var batteryHandle: FileHandle?
        defer { try? batteryHandle?.close() }
        if let file = pinnedManifest.capabilityBatteryFile,
            let pinned = pinnedManifest.capabilityBatteryHash
        {
            let battery = try CapabilityBattery(
                url: ExperimentStore.resolveProjectPath(file))
            guard !battery.items.isEmpty else {
                throw ExperimentError(reason: "capability battery '\(file)' is empty")
            }
            runBattery = (battery, pinned)
            let batteryURL = runDirectory.appending(component: "battery.jsonl")
            FileManager.default.createFile(atPath: batteryURL.path, contents: nil)
            batteryHandle = try FileHandle(forWritingTo: batteryURL)
        }

        struct RuntimeCondition {
            let name: String
            let variant: ModelVariantArtifact?
            let artifactPath: String?
            let artifactHash: String?
        }
        let runtimeConditions =
            [RuntimeCondition(name: "baseline", variant: nil, artifactPath: nil, artifactHash: nil)]
            + conditions.map {
                RuntimeCondition(
                    name: $0.name,
                    variant: $0.artifact,
                    artifactPath: $0.artifactPath,
                    artifactHash: $0.artifactHash)
            }

        // Comparability advisory (2026-08-24 ruling): are all the arms of
        // this run armed with the same effective system content? Loud,
        // non-blocking, and silent unless they diverge — which, before a
        // researcher gives an agent a persona, they never do. A
        // persona-varying design IS legitimate, so this cannot gate; what it
        // must not do is let the difference pass unremarked, because a
        // contrast between two differently-armed arms mixes identity and
        // framing into whatever the intervention did. Server twin:
        // `tasks._advise_system_prompt_divergence`.
        if let advisory = SystemPromptComposition.divergenceAdvisory(
            arms: runtimeConditions.map {
                ($0.name,
                 SystemPromptComposition.compose(
                    agent: $0.variant?.systemPrompt,
                    frame: pinnedManifest.systemPrompt))
            })
        {
            emitRunAdvisory(advisory, to: runDirectory)
        }

        var cancelled = false
        for condition in runtimeConditions {
            let activeAdapter: LoRAContainer?
            if let variant = condition.variant {
                activeAdapter = try await loadAdapter(variant, into: container)
            } else {
                activeAdapter = nil
            }
            let conditionInjections = try condition.variant.map(injections(for:)) ?? []
            let promptMode = condition.variant
                .flatMap { ExperimentManifest.PromptMode(rawValue: $0.promptMode) }
                ?? (pinnedManifest.promptMode ?? .chatAssistant)
            // System-prompt COMPOSITION (maintainer ruling, 2026-08-24).
            // This resolution used to be a REPLACEMENT: an agent arm ran
            // under its persona with the study's frame silently not applied,
            // while the baseline arm ran under the frame — so the frame was
            // part of the contrast instead of held constant, and the two arms
            // were not comparable. Now an agent arm runs under BOTH, persona
            // first. An agent with no persona (every agent artifact in the
            // workspace today, and every newborn agent since promotion
            // stopped inheriting the frame) composes to the frame alone —
            // byte-identical to the historical behaviour, baseline included.
            // Server twin: `_effective_variant_condition`.
            let armSystemPrompt = ArmSystemPrompt(
                agent: condition.variant?.systemPrompt,
                study: pinnedManifest.systemPrompt)
            let systemPrompt = armSystemPrompt.effective
            let qwenThinking = condition.variant?.qwenThinkingEnabled
                ?? pinnedManifest.qwenThinkingEnabled
                ?? false
            // Condition-scoped transcript gate: variant conditions carry
            // their OWN prompt mode, so a rawCompletion variant over
            // transcript items must refuse at condition start — never as a
            // mid-run template error (server `_execute_condition` twin).
            if promptMode == .rawCompletion,
                taskPrompts.prompts.contains(where: { $0.transcript?.isEmpty == false })
            {
                throw ExperimentError(
                    reason: "condition '\(condition.name)': task prompts include "
                        + "scripted transcripts but the condition's promptMode is "
                        + "rawCompletion — transcript items render through the "
                        + "chat template by definition; use chatAssistant")
            }
            // The whole per-condition body may throw or observe cancel; the
            // adapter/injector cleanup below must run either way.
            do {
                // Answer-token logprob instrument, identical dispatch to the
                // ordinary condition path (adapter loaded, condition
                // injections installed by the instrument per prompt).
                if wantsChoice {
                    choiceLoop: for (index, prompt) in taskPrompts.prompts.enumerated() {
                        guard let options = prompt.options, !options.isEmpty,
                            pinnedManifest.outcomeInstrumentScope?.includes(
                                .init(
                                    id: prompt.id, hasOptions: true,
                                    format: prompt.responseFormat)) ?? true
                        else { continue }
                        if await cancel.observed(
                            at: "\(condition.name) choice \(index + 1)/\(taskPrompts.prompts.count)")
                        {
                            cancelled = true
                            break choiceLoop
                        }
                        await progress?(
                            .generationStarted(
                                condition: condition.name,
                                promptID: prompt.id,
                                prompt: prompt.text))
                        let choice = try await LogprobInstrument.scoreOptions(
                            container, prompt: prompt.text, options: options,
                            modelID: pinnedManifest.modelID,
                            injections: conditionInjections,
                            promptMode: promptMode,
                            systemPrompt: systemPrompt,
                            qwenThinkingEnabled: qwenThinking,
                            transcript: prompt.transcript)
                        try checkOptionLengths(
                            choice, manifest: pinnedManifest, promptID: prompt.id)
                        let record = choiceRecord(
                            manifest: pinnedManifest,
                            experimentHash: experimentHash,
                            taskPromptsFile: taskPrompts.file,
                            taskPromptsHash: taskPrompts.hash,
                            promptMode: promptMode,
                            systemPrompt: systemPrompt,
                            systemPromptComposition: armSystemPrompt.stamp,
                            qwenThinkingEnabled: qwenThinking,
                            condition: condition.name,
                            promptIndex: index + 1,
                            prompt: prompt,
                            choice: choice)
                        reportChoiceReadouts.append(
                            ReportChoiceReadout(
                                condition: condition.name,
                                promptID: prompt.id,
                                sampleIndex: nil,
                                source: "instrument",
                                selected: record.selected,
                                target: record.target,
                                ordinalPosition: record.ordinalPosition))
                        let recordData = try encoder.encode(record)
                        if pinnedManifest.exclusionRules?.isEmpty == false {
                            instrumentExclusionViews.append(
                                ExclusionEngine.InstrumentRecordView(
                                    condition: condition.name,
                                    promptID: prompt.id,
                                    endpoints: analysisEndpoints(
                                        jsonLine: recordData,
                                        names: exclusionEndpointNames)))
                        }
                        try generationsHandle.write(contentsOf: recordData)
                        try generationsHandle.write(contentsOf: Data("\n".utf8))
                        let probability = choice.probability[choice.selected] ?? 0
                        let summary = String(
                            format: "answerTokenLogprob: selected=%@ p=%.4f margin=%.4f",
                            choice.selected, probability, choice.margin)
                        await progress?(
                            .generationCompleted(
                                StudyGenerationPreview(
                                    condition: condition.name,
                                    promptID: prompt.id,
                                    prompt: prompt.text,
                                    output: summary,
                                    wordCount: 0,
                                    distinct2: 0,
                                    markerDensity: [:],
                                    truncated: false)))
                        print(
                            "\(condition.name) choice \(prompt.id) "
                                + "(\(index + 1)/\(taskPrompts.prompts.count)): \(summary)")
                    }
                }
                if wantsSampled, !cancelled {
                    seedLoop: for seed in pinnedManifest.seeds {
                        for (index, prompt) in taskPrompts.prompts.enumerated() {
                            if await cancel.observed(
                                at: "\(condition.name) seed \(seed) prompt "
                                    + "\(index + 1)/\(taskPrompts.prompts.count)")
                            {
                                cancelled = true
                                break seedLoop
                            }
                            await progress?(
                                .generationStarted(
                                    condition: condition.name,
                                    promptID: prompt.id,
                                    prompt: prompt.text))
                            let generation = try await generateMeasured(
                                container,
                                prompt: prompt.text,
                                modelID: pinnedManifest.modelID,
                                maxTokens: pinnedManifest.maxTokens,
                                temperature: 0,
                                injections: conditionInjections,
                                promptMode: promptMode,
                                systemPrompt: systemPrompt,
                                qwenThinkingEnabled: qwenThinking,
                                transcript: prompt.transcript
                            ) { output in
                                await progress?(
                                    .generationChunk(
                                        condition: condition.name,
                                        promptID: prompt.id,
                                        output: output))
                            }
                            let output = generation.text
                            // Marker densities from the same attached-concept
                            // rubrics the ordinary path scores (a variant
                            // study with no attached concepts scores none —
                            // identical rule, not a fork).
                            let markerDensity = Dictionary(
                                uniqueKeysWithValues: conceptNames.map { concept in
                                    (concept, rubrics[concept]?.density(in: output) ?? 0)
                                })
                            let row = MetricRow(
                                condition: condition.name,
                                seed: seed,
                                promptIndex: index + 1,
                                promptID: prompt.id,
                                wordCount: wordCount(output),
                                distinct2: distinctBigramRatio(output),
                                markerDensity: markerDensity,
                                reasoningStyle: style.map { $0.taxonomy.score(output) } ?? [:],
                                factors: prompt.factors ?? [:])
                            rows.append(row)
                            // Reader readout, same clear-injections rule as
                            // the ordinary path. The condition's ADAPTER (if
                            // any) stays applied — it is the condition's
                            // model, not an intervention the reader can
                            // strip.
                            var readerScores: [String: Float]?
                            if !readerScorers.isEmpty {
                                try await setInterventions(container, [])
                                var scores: [String: Float] = [:]
                                for scorer in readerScorers {
                                    scores[scorer.concept] = try await RepEReader.scoreText(
                                        container: container,
                                        modelID: pinnedManifest.modelID,
                                        reader: scorer.reader, text: output)
                                }
                                readerScores = scores
                            }
                            let record = sampledGenerationRecord(
                                manifest: pinnedManifest,
                                experimentHash: experimentHash,
                                taskPromptsFile: taskPrompts.file,
                                taskPromptsHash: taskPrompts.hash,
                                promptMode: promptMode,
                                systemPrompt: systemPrompt,
                                systemPromptComposition: armSystemPrompt.stamp,
                                qwenThinkingEnabled: qwenThinking,
                                condition: condition.name,
                                seed: seed,
                                promptIndex: index + 1,
                                prompt: prompt,
                                output: output,
                                row: row,
                                variantArtifactPath: condition.artifactPath,
                                variantArtifactHash: condition.artifactHash,
                                agentPlaygroundTemperature: condition.variant?.temperature,
                                readerScores: readerScores,
                                numericParser: numericParser,
                                hitTokenCap: generation.hitTokenCap)
                            if pinnedManifest.exclusionRules?.isEmpty == false {
                                exclusionViews.append(exclusionView(of: record))
                            }
                            // Single-source rule — read the record's stamped
                            // `parsedChoice`, never a re-parse (see the
                            // ordinary condition loop's twin block).
                            if case .some(.some(let selected)) = record.parsedChoice {
                                reportChoiceReadouts.append(
                                    ReportChoiceReadout(
                                        condition: condition.name,
                                        promptID: record.promptID,
                                        sampleIndex: seed,
                                        source: "parsed",
                                        selected: selected,
                                        target: record.target))
                            }
                            try generationsHandle.write(contentsOf: encoder.encode(record))
                            try generationsHandle.write(contentsOf: Data("\n".utf8))
                            await progress?(.generationCompleted(generationPreview(from: record)))
                            print(
                                "\(condition.name) seed \(seed) prompt \(index + 1)/"
                                    + "\(taskPrompts.prompts.count): \(row.wordCount) words")
                        }
                    }
                }
                // Battery under this condition (adapter still loaded, same
                // injections the study outputs used). A cancelled battery
                // drops the whole cell — never a partial accuracy.
                if let (battery, batteryHash) = runBattery, let batteryHandle, !cancelled {
                    // `systemPrompt:` — the format-1 caller context. It is
                    // the arm's EFFECTIVE prompt, which for a legacy battery
                    // is what the surrounding instrument armed with, exactly
                    // as before. `agentSystemPrompt:` is the format-2 channel:
                    // the persona composes ahead of the battery's own declared
                    // arming, and the STUDY FRAME reaches a format-2 reading
                    // through neither — a baseline arm reads the battery bare,
                    // which is what makes it the control (2026-08-24 ruling).
                    let arming = battery.resolveArming(
                        promptMode: promptMode, systemPrompt: systemPrompt,
                        qwenThinkingEnabled: qwenThinking,
                        agentSystemPrompt: condition.variant?.systemPrompt)
                    if !batteryAdvised,
                        let advisory = battery.contaminationAdvisory(arming)
                    {
                        batteryAdvised = true
                        print("WARNING: \(advisory)")
                    }
                    let scored = try await ExperimentTasks.runBattery(
                        container, battery: battery, batteryHash: batteryHash,
                        condition: condition.name,
                        modelID: pinnedManifest.modelID,
                        injections: conditionInjections, arming: arming,
                        cancel: { _ in
                            await cancel.observed(at: "\(condition.name) battery")
                        })
                    if let scored {
                        for record in scored.records {
                            try batteryHandle.write(contentsOf: encoder.encode(record))
                            try batteryHandle.write(contentsOf: Data("\n".utf8))
                        }
                        batterySummaries[condition.name] = scored.summary
                        print(
                            "capability battery \(condition.name): "
                                + String(format: "%.0f%%", scored.summary.accuracy * 100)
                                + " over \(scored.summary.itemCount) items")
                    } else {
                        cancelled = true
                    }
                }
            } catch {
                try? await setInterventions(container, [])
                if let activeAdapter {
                    await unloadAdapter(activeAdapter, from: container)
                }
                throw error
            }
            try await setInterventions(container, [])
            if let activeAdapter {
                await unloadAdapter(activeAdapter, from: container)
            }
            if cancelled { break }
        }

        try metricsCSV(
            rows: rows, concepts: conceptNames,
            styleFeatureIDs: style?.taxonomy.featureIDs ?? []
        ).write(
            to: runDirectory.appending(component: "metrics.csv"),
            atomically: true,
            encoding: .utf8)
        if cancelled {
            writeCancellationNote(task: "study run", to: runDirectory)
            print(
                "study run cancelled by user — partial artifacts kept in "
                    + runDirectory.lastPathComponent)
            return runDirectory
        }
        let report = report(
            experiment: pinnedManifest,
            experimentHash: experimentHash,
            taskPrompts: taskPrompts,
            rows: rows,
            conditionCount: runtimeConditions.count,
            concepts: conceptNames,
            batterySummaries: batterySummaries,
            choiceReadouts: reportChoiceReadouts,
            style: style,
            numericParser: numericParser?.provenance,
            exclusions: exclusionOutcome(
                manifest: pinnedManifest, prompts: taskPrompts.prompts,
                views: exclusionViews,
                instrumentViews: instrumentExclusionViews))
        if let entries = report.effectSizes, !entries.isEmpty {
            try effectSizesCSV(entries).write(
                to: runDirectory.appending(component: "effect-sizes.csv"),
                atomically: true, encoding: .utf8)
        }
        try encoder.encode(report).write(to: runDirectory.appending(component: "report.json"))
        let robustnessJudge: String? =
            if let evaluation = pinnedManifest.evaluation,
                evaluation.kind == .pairedJudge,
                !evaluation.judgeModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                evaluation.judgeModel
            } else {
                nil
            }
        var robustnessReports: [String: VariantRobustnessReport] = [:]
        let robustnessPreset =
            VariantRobustness.preset(id: "study") ?? VariantRobustness.defaultPreset
        robustnessLoop: for condition in conditions {
            // Cancellation observed inside a robustness pass drops THAT
            // report (nil — a partial battery is never scored) and stops the
            // remaining passes; the study report above is already complete
            // and stays.
            guard
                let report = try await VariantRobustness.evaluate(
                    variant: condition.artifact,
                    variantPath: condition.artifactPath,
                    variantHash: condition.artifactHash,
                    container: container,
                    batteryFile: robustnessPreset.batteryFile,
                    coherencePromptsFile: robustnessPreset.coherencePromptsFile,
                    maxCoherencePrompts: robustnessPreset.maxCoherencePrompts,
                    maxTokens: robustnessPreset.maxTokens,
                    presetID: robustnessPreset.id,
                    judgeModel: robustnessJudge,
                    shouldCancel: cancel.shouldCancel)
            else {
                print(
                    "robustness pass for '\(condition.name)' cancelled by user — "
                        + "no robustness report for it (a partial battery is never scored)")
                break robustnessLoop
            }
            robustnessReports[condition.name] = report
        }
        if !robustnessReports.isEmpty {
            try encoder.encode(robustnessReports).write(
                to: runDirectory.appending(component: "robustness-report.json"))
        }
        print("run artifacts: \(runDirectory.path)")
        return runDirectory
    }

    // MARK: - Epoch guard (cross-engine contract)

    /// The experiment content hash a run directory was stamped with, if any:
    /// `experiment-hash.txt` first, then the canonical `config.json`'s
    /// "experimentHash". nil = unstamped legacy run.
    /// Delegates to `RunEpoch` — this was a byte-for-byte second copy of
    /// the same reader, and the two drifted: `RunEpoch` learned that a
    /// foreign-substrate stamp is not comparable, while this side kept
    /// reporting such runs as "a different manifest epoch".
    static func runExperimentHashStamp(at runDirectory: URL) -> String? {
        RunEpoch.stampedExperimentHash(runDirectory)
    }

    /// The epoch guard for MEASUREMENT verbs (evaluate/analyze/rescore): a
    /// run may only be judged/analyzed under the exact manifest that
    /// produced it — its experiment-hash stamp must equal the LIVE
    /// manifest's content hash (matching by experiment NAME alone let a
    /// pre-edit draft run be judged under a frozen manifest, the firewall
    /// gap this closes).
    ///
    /// One rule, one implementation: this delegates to `RunEpoch.check`, the
    /// cross-engine twin of `run_epoch.epoch_refusal`, exactly as the
    /// server's `_require_source_epoch` does. It was a SECOND, stricter
    /// comparison here, and the two diverged: `RunEpoch` learned the
    /// foreign-substrate, self-projection, revision-repair and
    /// measurement-drift rules while this side kept refusing runs the server
    /// accepted — the same run behaved differently per engine.
    ///
    /// Every caller here is a measurement verb, so drift confined to
    /// `RunEpoch.measurementFields` is TOLERATED rather than refused: those
    /// fields cannot have affected a byte of the source run's generations,
    /// and refusing them forced a full GPU re-run to swap a judge whose
    /// model had died at its provider (2026-08-05). The returned check's
    /// `unverified` / `measurementDrift` are obligations: log them loudly
    /// and stamp them (`epochUnverified`, `measurementDrift`) into the
    /// verb's own output. `promote` stays STRICT (`RunEpoch.refusal`).
    ///
    /// The returned `refusal` is always nil — a refusal throws.
    @discardableResult
    static func verifyRunEpoch(
        verb: String,
        runDirectory: URL,
        manifest: ExperimentManifest,
        allowUnverified: Bool = false
    ) throws -> RunEpoch.Check {
        let check = RunEpoch.check(
            verb: verb, experiment: manifest.name,
            liveHash: ExperimentStore.manifestHash(manifest),
            runDirectory: runDirectory, liveManifest: manifest,
            allowUnverified: allowUnverified,
            tolerateMeasurementDrift: true,
            // The whole family reads the source run's RECORDS, so a run from
            // the other engine is refused here rather than measured into an
            // empty result that exits 0 (WP0 dry run #2, P0).
            refuseForeignSubstrate: true)
        if let refusal = check.refusal {
            // A foreign run's repair is not "re-run" and is certainly not
            // `--allow-unverified-epoch` (which forgives a missing stamp, and
            // would leave this run just as unreadable) — it is the same verb
            // on the engine that wrote the records.
            let repair =
                RunEpoch.foreignSubstrate(runDirectory) != nil
                ? "steerlab-server experiment \(verb) \(manifest.name)  "
                    + "(on the engine that produced the run; the Mac reads "
                    + "its results, it does not re-measure them)"
                : "steerlab-cli experiment run \(manifest.name)  "
                    + "(a run of the CURRENT manifest), or read the older run "
                    + "under its own epoch with steerlab-cli experiment "
                    + "\(verb) \(manifest.name) --allow-unverified-epoch  "
                    + "(which only bypasses an UNSTAMPED run, never a "
                    + "mismatched one)"
            throw ExperimentError.refusing(.manifestEpoch, refusal, repair: repair)
        }
        // Tolerated is never silent (server twin: the `_log` warnings in
        // `tasks.evaluate`/`analyze`/`rescore_style`).
        if let drift = check.measurementDrift {
            print(
                "WARNING: '\(manifest.name)' drifted from source run "
                    + "'\(runDirectory.lastPathComponent)' in MEASUREMENT-side "
                    + "fields only (\(drift)) — the generations are "
                    + "unaffected; \(verb) proceeds under the LIVE settings "
                    + "and the output is stamped measurementDrift")
        }
        if check.unverified {
            print(
                "WARNING: source run '\(runDirectory.lastPathComponent)' "
                    + "carries no experiment-hash stamp — \(verb) under "
                    + "allowUnverifiedEpoch; the output is stamped "
                    + "epochUnverified")
        }
        return check
    }

    // MARK: - analyze (headless statistics)

    /// The newest COMPLETED study run of this experiment (directory name
    /// `…-exp-<name>-run[-N]`, with generations.jsonl + report.json present
    /// and a manifest snapshot naming this experiment).
    public static func newestCompletedRun(experimentName: String) -> URL? {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: ExperimentStore.runsDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        for entry in entries.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                entry.lastPathComponent.range(
                    of: "-exp-\(experimentName)-run(-\\d+)?$",
                    options: .regularExpression) != nil,
                fm.fileExists(
                    atPath: entry.appending(component: "generations.jsonl").path),
                fm.fileExists(atPath: entry.appending(component: "report.json").path),
                let data = try? Data(
                    contentsOf: entry.appending(component: "experiment.json")),
                let snapshot = try? JSONDecoder().decode(
                    ExperimentManifest.self, from: data),
                snapshot.name == experimentName
            else { continue }
            return entry
        }
        return nil
    }

    /// Minimal per-record view of generations.jsonl for analysis: sampled
    /// records map onto MetricRow; choice-instrument records contribute
    /// their ordinalPosition (when the run declared ordinalScale) and are
    /// otherwise skipped.
    private struct AnalysisGeneration: Decodable {
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

    /// Headless `experiment analyze <name>`: recomputes paired-to-baseline
    /// effect sizes (bootstrap CI + Wilcoxon via StudyStatistics, plus the
    /// phase's multiple-comparison correction — BH-FDR for screens, Holm
    /// for confirms, the server's exact rule) from the newest completed
    /// run's generations, under the epoch guard. Pure CPU —
    /// no model load; writes `analysis.json` + `effect-sizes.csv` (plus
    /// `choice-deltas.csv`/`.json` when the run carries answer-token
    /// instrument readouts) into a fresh immutable analyze run directory
    /// (runs are never mutated).
    @discardableResult
    /// Warning text for an analysis whose source run holds nothing but
    /// baseline records — the paired statistics have no contrast to compute,
    /// so the analysis is structurally empty however many generations it
    /// read (WP0 dry run #0, P0-2). nil when a non-baseline condition
    /// produced at least one record, or when the run held no records at all
    /// (a different refusal covers that). Pure, so the wording is testable
    /// without a run directory; the Python twin prints the same sentence in
    /// `tasks.analyze`.
    static func baselineOnlyAnalysisWarning(
        runName: String, conditions: Set<String>
    ) -> String? {
        guard !conditions.isEmpty,
              !conditions.contains(where: { $0 != "baseline" })
        else { return nil }
        return "WARNING: run '\(runName)' contains only BASELINE records — "
            + "there is no non-baseline condition to pair against, so this "
            + "analysis will produce no effect sizes. Check the study's "
            + "conditions before citing it."
    }

    /// The historical `emptyAnalysis` detail: the ONE cause the advisory used
    /// to claim unconditionally. Still exactly right when it is what happened.
    /// Server twin: `cli_payloads.EMPTY_ANALYSIS_NO_CONTRAST`.
    static let emptyAnalysisNoContrast =
        "0 effect-size entries — the source run has no non-baseline condition "
        + "to pair against"

    /// The `emptyAnalysis` advisory's DETAIL: what an analysis with zero
    /// effect-size entries actually observed.
    ///
    /// Until WP0 dry run #2 this said one thing unconditionally, and on the
    /// run that produced the finding it was simply false — the run carried
    /// two conditions and 24 records, and what failed was READING and PAIRING
    /// them. An advisory that names the wrong cause is worse than a vague
    /// one: it sends the reader to audit a study design that is fine while
    /// the real fault (foreign-engine artifacts, or a record-schema
    /// mismatch) goes unlooked-at.
    ///
    /// Pure, so every branch is testable without a run directory. Server
    /// twin: `cli_payloads.empty_analysis_detail`.
    static func emptyAnalysisDetail(
        runName: String, recordCount: Int?, conditions: Set<String>
    ) -> String {
        guard let recordCount else {
            return "0 effect-size entries — the records of source run "
                + "'\(runName)' could not be read here, so this analysis "
                + "cannot say what it measured nothing over"
        }
        if recordCount == 0 {
            return "0 effect-size entries — source run '\(runName)' holds no "
                + "records at all"
        }
        guard conditions.contains(where: { $0 != "baseline" }) else {
            return emptyAnalysisNoContrast
        }
        let named = conditions.sorted().joined(separator: ", ")
        return "0 effect-size entries — source run '\(runName)' holds "
            + "\(recordCount) record\(recordCount == 1 ? "" : "s") across "
            + "condition\(conditions.count == 1 ? "" : "s") \(named), so it "
            + "HAS a contrast: analyze could not read or pair those records. "
            + "Likeliest cause: artifacts produced on the other engine "
            + "(record schemas and pairing keys are per-engine), or a "
            + "record-schema mismatch"
    }

    /// `(recordCount, conditions)` read straight off a source run's
    /// generations, for `emptyAnalysisDetail`. A nil count means the file
    /// could not be read at all — a different fact from "it was empty".
    static func analysisSourceRecords(
        at runDirectory: URL
    ) -> (recordCount: Int?, conditions: Set<String>) {
        guard
            let text = try? String(
                contentsOf: runDirectory.appending(component: "generations.jsonl"),
                encoding: .utf8)
        else { return (nil, []) }
        struct ConditionOnly: Decodable { let condition: String }
        var conditions = Set<String>()
        var count = 0
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            count += 1
            if let record = try? decoder.decode(
                ConditionOnly.self, from: Data(line.utf8))
            {
                conditions.insert(record.condition)
            }
        }
        return (count, conditions)
    }

    /// The detail for a named source run under the workspace's `runs/`.
    public static func emptyAnalysisDetail(sourceRunNamed runName: String) -> String {
        let (count, conditions) = analysisSourceRecords(
            at: ExperimentStore.runsDirectory.appending(path: runName))
        return emptyAnalysisDetail(
            runName: runName, recordCount: count, conditions: conditions)
    }

    public static func analyze(
        experimentName: String,
        allowUnverifiedEpoch: Bool = false
    ) throws -> URL {
        let manifest = try loadVerified(experimentName)
        guard let sourceRun = newestCompletedRun(experimentName: experimentName) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "no completed study run found for '\(experimentName)' "
                    + "(need generations.jsonl + report.json under runs/)",
                repair: "steerlab-cli experiment run \(experimentName) && "
                    + "steerlab-cli experiment analyze \(experimentName)")
        }
        let epoch = try verifyRunEpoch(
            verb: "analyze", runDirectory: sourceRun, manifest: manifest,
            allowUnverified: allowUnverifiedEpoch)

        // Reasoning-style values are derived, not stored: recompute them from
        // each record's output through the pinned (hash-checked) taxonomy so
        // rs_<featureID> joins the same paired effect-size machinery.
        let style = try ExperimentStore.loadPinnedReasoningStyle(manifest)

        // Declared exclusion rules join HERE — records drop from the paired
        // statistics only (pairwise deletion falls out of the (seed,
        // promptID) baseline join), never from generations.jsonl, and the
        // stamp lands in analysis.json + exclusions.json. Scope is
        // allRecordTypes (the engine default): instrument readouts are
        // considered too — endpoint rules read endpoints the record itself
        // carries (e.g. ordinalPosition), and a cell whose every sampled
        // record failed its attention check drops its instrument readout
        // from the ordinal pairing with it. No rules declared = today's
        // behavior byte-for-byte. Server twin: `tasks.analyze`.
        let exclusionRules = manifest.exclusionRules ?? []
        let ruleProblems = ExclusionEngine.violations(exclusionRules)
        guard ruleProblems.isEmpty else {
            throw ExperimentError(reason: ruleProblems.joined(separator: "; "))
        }
        var exclusionChecks: [String: AttentionCheck] = [:]
        if ExclusionEngine.needsChecks(exclusionRules) {
            guard manifest.taskPromptsHash != nil else {
                // WP0 step 8: the deferred cross-engine-twinned message gets
                // its id on BOTH engines. The STRING is unchanged and stays
                // byte-identical to the server's `PIN_REQUIRED_MESSAGE`
                // (asserted on both sides); only the gate id and the runnable
                // repair are new.
                throw ExperimentError.refusing(
                    .missingPrerequisite, ExclusionEngine.pinRequiredMessage,
                    repair: ExclusionEngine.pinRequiredRepair)
            }
            exclusionChecks = attentionChecks(
                of: try loadTaskPrompts(for: manifest).prompts)
            guard !exclusionChecks.isEmpty else {
                throw ExperimentError(reason: ExclusionEngine.noChecksMessage)
            }
        }
        let exclusionEndpoints = Set(
            exclusionRules
                .filter { $0.rule != ExclusionEngine.ruleFailedAttentionCheck }
                .map(ExclusionEngine.resolvedEndpoint))
        var exclusionViews: [ExclusionEngine.RecordView] = []
        var instrumentExclusionViews: [ExclusionEngine.InstrumentRecordView] = []

        let text = try String(
            contentsOf: sourceRun.appending(component: "generations.jsonl"),
            encoding: .utf8)
        var rows: [MetricRow] = []
        var ordinalReadouts: [ReportChoiceReadout] = []
        var conceptSet = Set<String>()
        // D3: per-condition option logprobs, for the distance-from-boundary
        // diagnostics written alongside the effect sizes.
        var optionLogprobsByCondition: [String: [[String: Double]]] = [:]
        // Phase 3: per-item choice deltas, paired to the same item's baseline
        // readout. Collected in run order; the pairing and the sort happen in
        // ChoiceDeltas (server twin: choice_deltas.rows).
        var choiceReadouts: [ChoiceDeltas.Readout] = []
        // Declared-target map from the PINNED task file (open-issues #6, the
        // exact authority — server twin: `tasks.analyze`'s `declared_targets`).
        // It keeps a mixed instrument's legitimate endpoint (an item with both
        // a declared A/B target and an ordinal readout on one record) while
        // dropping the ordinalScale items whose "target" was synthesized. An
        // unloadable prompts file falls back to the per-record ladder inside
        // `ChoiceDeltas.targetIsDeclared`.
        var declaredTargets: [String: Bool]? = nil
        if manifest.taskPromptsHash != nil,
            let loaded = try? loadTaskPrompts(for: manifest)
        {
            declaredTargets = Dictionary(
                loaded.prompts.map { ($0.id, $0.target?.isEmpty == false) },
                uniquingKeysWith: { first, _ in first })
        }
        // Item → declared factor levels (arm/caseID + the factorial
        // `factors` object), for the stratified effect rows. Records carry
        // the item metadata verbatim, so no rejoin of the task-prompts file;
        // first record per item wins (items stamp identically).
        var factorsByItem: [String: [String: String]] = [:]
        // Every condition name that actually produced a record. A run with
        // no NON-baseline condition has nothing to pair against — analyze
        // still writes its (empty) artifacts and exits 0, so the fact has to
        // be said out loud on stderr (WP0 dry run #0, P0-2).
        var conditionsSeen = Set<String>()
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            guard
                let record = try? decoder.decode(
                    AnalysisGeneration.self, from: Data(line.utf8))
            else { continue }
            conditionsSeen.insert(record.condition)
            if factorsByItem[record.promptID] == nil {
                var levels: [String: String] = [:]
                if let arm = record.arm, !arm.isEmpty { levels["arm"] = arm }
                if let caseID = record.caseID, !caseID.isEmpty {
                    levels["caseID"] = caseID
                }
                for (key, value) in record.factors ?? [:] where !value.isEmpty {
                    levels[key] = value
                }
                factorsByItem[record.promptID] = levels
            }
            if let logprobs = record.optionLogprobs, !logprobs.isEmpty {
                optionLogprobsByCondition[record.condition, default: []]
                    .append(logprobs)
            }
            if record.instrument != nil {
                // Instrument records carry no sampled metrics, but an
                // ordinalScale run's ladder positions are per-item numeric
                // data for the SAME paired effect-size machinery — and
                // under scope allRecordTypes the declared rules consider
                // the readout itself (its own endpoints; its cell's
                // attention evidence).
                if !exclusionRules.isEmpty {
                    instrumentExclusionViews.append(
                        ExclusionEngine.InstrumentRecordView(
                            condition: record.condition,
                            promptID: record.promptID,
                            endpoints: analysisEndpoints(
                                jsonLine: Data(line.utf8),
                                names: exclusionEndpoints)))
                }
                // Every answer-token readout with a DECLARED target is
                // collected, including one whose logOdds is missing or has no
                // entry for that target: that is an unreadable measurement,
                // counted as such downstream, never quietly absent from the
                // coverage numbers. A readout whose target was never declared
                // (open-issues #6) is not an unreadable choice — it is not a
                // choice measurement at all, so it stays out of the table
                // rather than inflating its skip counts.
                if record.instrument == ChoiceDeltas.instrument,
                    ChoiceDeltas.targetIsDeclared(
                        promptID: record.promptID,
                        targetSource: record.targetSource,
                        ordinalPosition: record.ordinalPosition,
                        declaredTargets: declaredTargets)
                {
                    choiceReadouts.append(
                        ChoiceDeltas.Readout(
                            condition: record.condition,
                            promptID: record.promptID,
                            sampleIndex: record.sampleIndex.map { String($0) } ?? "",
                            target: record.target ?? "",
                            logOdds: record.logOdds ?? [:],
                            choiceProbability: record.choiceProbability ?? [:],
                            selected: record.selected ?? ""))
                }
                if let position = record.ordinalPosition {
                    ordinalReadouts.append(
                        ReportChoiceReadout(
                            condition: record.condition,
                            promptID: record.promptID,
                            sampleIndex: nil,
                            source: "instrument",
                            selected: "",
                            target: nil,
                            ordinalPosition: position))
                }
                continue
            }
            guard let wordCount = record.wordCount else { continue }
            if !exclusionRules.isEmpty {
                exclusionViews.append(
                    ExclusionEngine.RecordView(
                        condition: record.condition,
                        seed: record.seed ?? 0,
                        promptID: record.promptID,
                        output: record.output ?? "",
                        endpoints: analysisEndpoints(
                            jsonLine: Data(line.utf8),
                            names: exclusionEndpoints)))
            }
            let markerDensity = record.markerDensity ?? [:]
            conceptSet.formUnion(markerDensity.keys)
            let reasoningStyle: [String: Double] =
                if let style, let output = record.output {
                    style.taxonomy.score(output)
                } else {
                    [:]
                }
            rows.append(
                MetricRow(
                    condition: record.condition,
                    seed: record.seed ?? 0,
                    promptIndex: record.promptIndex ?? 0,
                    promptID: record.promptID,
                    wordCount: wordCount,
                    distinct2: record.distinct2 ?? 0,
                    markerDensity: markerDensity,
                    reasoningStyle: reasoningStyle))
        }
        // Choice readouts count as analyzable material alongside sampled
        // generations and ordinal readouts: a study whose whole instrument is
        // the answer-token logprob (no prose arm at all) has per-item deltas
        // to report, and refusing it here would make choice-deltas.csv
        // unreachable on this engine. The server has never had this guard.
        guard !rows.isEmpty || !ordinalReadouts.isEmpty || !choiceReadouts.isEmpty
        else {
            throw ExperimentError(
                reason: "run '\(sourceRun.lastPathComponent)' has no sampled "
                    + "generations or instrument readouts to analyze")
        }
        // Records exist, but every one of them is the baseline: the paired
        // statistics have no contrast to compute, so this analysis is
        // structurally empty however many generations it read. A warning,
        // not a refusal — the run's own artifacts are still legitimate
        // material — but never silent (WP0 dry run #0, P0-2). Server twin:
        // the same line in `tasks.analyze`.
        if let warning = baselineOnlyAnalysisWarning(
            runName: sourceRun.lastPathComponent, conditions: conditionsSeen)
        {
            FileHandle.standardError.write(Data((warning + "\n").utf8))
        }
        var exclusionStamp: ExclusionStamp?
        if !exclusionRules.isEmpty {
            let outcome = ExclusionEngine.evaluate(
                rules: exclusionRules, checks: exclusionChecks,
                views: exclusionViews,
                instrumentViews: instrumentExclusionViews)
            exclusionStamp = outcome.stamp
            rows = rows.filter {
                !outcome.excludedKeys.contains(
                    ExclusionEngine.rowKey(
                        condition: $0.condition, seed: $0.seed,
                        promptID: $0.promptID))
            }
            ordinalReadouts = ordinalReadouts.filter {
                !outcome.excludedInstrumentKeys.contains(
                    ExclusionEngine.instrumentKey(
                        condition: $0.condition, promptID: $0.promptID))
            }
            // Same drop for the choice-delta table: an excluded readout must
            // not reappear as a citable per-item delta.
            choiceReadouts = choiceReadouts.filter {
                !outcome.excludedInstrumentKeys.contains(
                    ExclusionEngine.instrumentKey(
                        condition: $0.condition, promptID: $0.promptID))
            }
            print(
                "exclusions: \(outcome.stamp.excludedRecords) record(s) "
                    + "excluded by \(exclusionRules.count) declared rule(s); "
                    + "surviving N per condition: "
                    + outcome.stamp.survivingN.sorted { $0.key < $1.key }
                        .map { "\($0.key)=\($0.value)" }
                        .joined(separator: ", "))
        }
        let pooledEntries = effectSizes(
            rows: rows, concepts: conceptSet.sorted(),
            styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
            choiceReadouts: ordinalReadouts,
            phase: manifest.phase)
        // Per-cell strata beside the pooled rows (same file, extra rows):
        // pooling across items has both hidden a real single-cell effect
        // behind saturated cells and manufactured pooled effects from one
        // cell's parse garbage. Pooled entries keep their exact semantics
        // and correction family; each stratified family is corrected
        // independently. Server twin: tasks.analyze.
        let entries = pooledEntries
            + stratifiedEffectSizes(
                rows: rows, concepts: conceptSet.sorted(),
                styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
                choiceReadouts: ordinalReadouts,
                factorsByItem: factorsByItem,
                phase: manifest.phase)

        let runDirectory = try makeRunDirectory(experiment: manifest, task: "analyze")
        let report = AnalyzeReport(
            experiment: manifest.name,
            experimentHash: ExperimentStore.manifestHash(manifest),
            sourceRun: sourceRun.lastPathComponent,
            sourceRunExperimentHash: runExperimentHashStamp(at: sourceRun),
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            effectSizes: entries,
            exclusions: exclusionStamp)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: runDirectory.appending(component: "analysis.json"))
        if let exclusionStamp {
            // The stamp file the server also writes — one artifact name to
            // look for on either engine.
            try encoder.encode(exclusionStamp).write(
                to: runDirectory.appending(component: "exclusions.json"))
        }
        try effectSizesCSV(entries).write(
            to: runDirectory.appending(component: "effect-sizes.csv"),
            atomically: true, encoding: .utf8)
        // Per-item choice deltas (server twin: choice-deltas.csv +
        // choice-deltas.json). Absence over empty artifacts: a run with no
        // non-baseline choice readouts grows no table implying it had some.
        // When there ARE readouts the file is written even if every one of
        // them was skipped — the skip counts are the finding in that case.
        let choiceDeltas = ChoiceDeltas.table(choiceReadouts)
        if !choiceDeltas.summary.conditions.isEmpty {
            try ChoiceDeltas.csv(choiceDeltas.rows).write(
                to: runDirectory.appending(component: "choice-deltas.csv"),
                atomically: true, encoding: .utf8)
            try encoder.encode(choiceDeltas.summary).write(
                to: runDirectory.appending(component: "choice-deltas.json"))
            let flips = choiceDeltas.summary.conditions.values
                .reduce(0) { $0 + $1.flipped }
            print(
                "choice deltas: \(choiceDeltas.rows.count) paired item(s) "
                    + "across \(choiceDeltas.summary.conditions.count) "
                    + "condition(s), \(flips) flip(s), "
                    + "\(choiceDeltas.summary.skippedNoBaseline) skipped "
                    + "(no baseline partner) → choice-deltas.csv")
            if choiceDeltas.summary.skippedNoTargetValue > 0 {
                print(
                    "choice deltas: \(choiceDeltas.summary.skippedNoTargetValue) "
                        + "readout(s) skipped — no log-odds entry for the "
                        + "item's own target option")
            }
        }
        // D3: a large joint-logprob margin means the FLIP RATE has poor
        // sensitivity — an intervention can move the log-odds a long way
        // without flipping any item — while the log-odds itself keeps moving
        // continuously. Calling that "saturation" invites the wrong
        // conclusion; true numerical saturation is the separately counted
        // clamp incidence.
        var marginReports: [String: ChoiceMarginDiagnostics.Report] = [:]
        for (condition, logprobs) in optionLogprobsByCondition {
            let block = ChoiceMarginDiagnostics.report(
                optionLogprobsPerItem: logprobs)
            if block.scoredItems > 0 { marginReports[condition] = block }
        }
        if !marginReports.isEmpty {
            try encoder.encode(marginReports).write(
                to: runDirectory.appending(component: "choice-margins.json"))
            for (condition, block) in marginReports.sorted(by: { $0.key < $1.key }) {
                print("\(condition): \(block.interpretation ?? "")")
            }
        }
        let ordinalNote =
            ordinalReadouts.isEmpty
            ? "" : " + \(ordinalReadouts.count) ordinal readouts"
        let stratifiedCount = entries.count - pooledEntries.count
        print(
            "analyzed \(rows.count) generations\(ordinalNote) from "
                + "\(sourceRun.lastPathComponent): "
                + "\(pooledEntries.count) effect-size "
                + "entr\(pooledEntries.count == 1 ? "y" : "ies")"
                + (stratifiedCount > 0 ? " + \(stratifiedCount) stratified" : ""))
        print("analysis artifacts: \(runDirectory.path)")
        return runDirectory
    }

    // MARK: - rescore-style (post-hoc reasoning-style scoring)

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

    /// Headless `experiment rescore-style <name> [--run DIR]`: recomputes
    /// reasoning-style feature values for an EXISTING completed run's sampled
    /// generations from the manifest's pinned taxonomy — pure CPU, no model —
    /// and writes `reasoning-style.csv` + `reasoning-style.json` into a fresh
    /// immutable rescore run directory. The source run is NEVER mutated (run
    /// immutability), and the epoch guard applies exactly as for analyze:
    /// the run's experiment-hash stamp must equal the live manifest's hash.
    @discardableResult
    public static func rescoreStyle(
        experimentName: String,
        runDirectoryName: String? = nil,
        allowUnverifiedEpoch: Bool = false
    ) throws -> URL {
        let manifest = try loadVerified(experimentName)
        guard let style = try ExperimentStore.loadPinnedReasoningStyle(manifest) else {
            throw ExperimentError(
                reason: "experiment '\(experimentName)' pins no reasoning-style "
                    + "taxonomy — pin one first: steerlab-cli experiment "
                    + "set-style-taxonomy \(experimentName) "
                    + "prompts/taxonomies/<name>.json")
        }
        let sourceRun: URL
        if let runDirectoryName {
            sourceRun =
                runDirectoryName.hasPrefix("/")
                ? URL(filePath: runDirectoryName)
                : ExperimentStore.runsDirectory.appending(path: runDirectoryName)
        } else if let newest = newestCompletedRun(experimentName: experimentName) {
            sourceRun = newest
        } else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "no completed study run found for '\(experimentName)' "
                    + "(need generations.jsonl + report.json under runs/) — "
                    + "run it first, or pass --run",
                repair: "steerlab-cli experiment run \(experimentName) && "
                    + "steerlab-cli experiment rescore-style \(experimentName)")
        }
        let epoch = try verifyRunEpoch(
            verb: "rescore-style", runDirectory: sourceRun, manifest: manifest,
            allowUnverified: allowUnverifiedEpoch)

        let text = try String(
            contentsOf: sourceRun.appending(component: "generations.jsonl"),
            encoding: .utf8)
        var rows: [MetricRow] = []
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            guard
                let record = try? decoder.decode(
                    AnalysisGeneration.self, from: Data(line.utf8)),
                record.instrument == nil,
                let output = record.output
            else { continue }
            rows.append(
                MetricRow(
                    condition: record.condition,
                    seed: record.seed ?? 0,
                    promptIndex: record.promptIndex ?? 0,
                    promptID: record.promptID,
                    wordCount: record.wordCount ?? 0,
                    distinct2: record.distinct2 ?? 0,
                    markerDensity: [:],
                    reasoningStyle: style.taxonomy.score(output)))
        }
        guard !rows.isEmpty else {
            throw ExperimentError(
                reason: "run '\(sourceRun.lastPathComponent)' has no sampled "
                    + "generations to rescore")
        }

        // NEW immutable artifacts only — never a byte into the source run.
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "rescore-style")
        let header = ["condition", "seed", "promptIndex", "promptID"]
            + style.taxonomy.featureIDs.map { "rs_\($0)" }
        var lines = [header.joined(separator: ",")]
        for row in rows {
            let cells = [
                csvEscape(row.condition),
                String(row.seed),
                String(row.promptIndex),
                csvEscape(row.promptID),
            ] + style.taxonomy.featureIDs.map { String(row.reasoningStyle[$0] ?? 0) }
            lines.append(cells.joined(separator: ","))
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: runDirectory.appending(component: "reasoning-style.csv"),
            atomically: true, encoding: .utf8)

        let grouped = Dictionary(grouping: rows, by: \.condition)
        let report = RescoreStyleReport(
            experiment: manifest.name,
            experimentHash: ExperimentStore.manifestHash(manifest),
            sourceRun: sourceRun.lastPathComponent,
            sourceRunExperimentHash: runExperimentHashStamp(at: sourceRun),
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            taxonomy: style.taxonomy.name,
            taxonomyHash: style.hash,
            taxonomyFile: style.path,
            diagnosticOnly: true,
            conditions: grouped.compactMapValues { conditionRows in
                reasoningStyleReport(rows: conditionRows, style: style)
                    .map { RescoreStyleReport.ConditionBlock(features: $0.features) }
            })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: runDirectory.appending(component: "reasoning-style.json"))
        print(
            "rescored \(rows.count) generations from \(sourceRun.lastPathComponent): "
                + "\(style.taxonomy.featureIDs.count) feature(s) × "
                + "\(grouped.count) condition(s)")
        print("rescore artifacts: \(runDirectory.path)")
        return runDirectory
    }

    // MARK: - paired judge evaluation

    /// One resolved judge pass: panel entry name, kind, and the model id
    /// that will actually run.
    struct ResolvedJudge: Sendable {
        let name: String
        let kind: String
        let model: String
        /// True when a LOCAL judge declared no model and resolved to the
        /// study model (`manifest.modelID`) — logged at sweep start.
        var modelDefaulted: Bool = false
        /// OpenRouter judges only: the pinned serving provider (required —
        /// resolution refuses an openrouter judge without one).
        var provider: String? = nil
        /// Local judges: the pinned model revision (JudgeRef.revision,
        /// 2026-07-23); a study-model judge falls back to the study pin at
        /// load time when nil.
        var revision: String? = nil
        /// Local judges: the pinned load dtype (`JudgeRef.dtype`, declarable
        /// through `--judge-pin <name>=<revision>[:<dtype>]`); a study-model
        /// judge that declares none INHERITS the study's, so nil here means
        /// "whatever the study loads at" and never diverges. Declaration
        /// only — this engine's loader takes no dtype, which is precisely
        /// why a divergent one has to be refused rather than honoured.
        var dtype: String? = nil
    }

    /// WHICH loaded weights a local judge judges through: the model id AND
    /// the revision pinned beside it — never the id alone.
    ///
    /// The id alone was the key of the per-run judge cache, and a panel is
    /// allowed to name the same model twice at two revisions (a stability
    /// check across checkpoints is exactly that panel). The second judge then
    /// found the FIRST judge's container under its model id, judged every
    /// pair with revision A's weights, and the judgment rows stamped
    /// `judgeRevision: B` — the evidence naming bytes that never ran. Silent
    /// wrong-model execution with a provenance stamp on top, which is the
    /// worst shape a bug in this codebase can take (review round 9,
    /// finding 1).
    ///
    /// A nil revision is its OWN key, never a wildcard that matches a pinned
    /// one: an unpinned judge loads through `SteeredContainerLoader.load`'s
    /// own resolution (whatever `refs/main` points at in this cache) and
    /// stamps nil, so it is a different declaration from a judge that named
    /// that same commit — and the two get different containers, which is the
    /// only reading under which the stamp and the weights agree.
    ///
    /// Python needs no twin of this type: `_local_judge_generation` builds
    /// one `held` slot per judge closure, so a revision can never leak
    /// sideways there.
    struct LoadedModelKey: Hashable, Sendable {
        let model: String
        let revision: String?

        init(model: String, revision: String?) {
            self.model = model
            self.revision = revision
        }

        init(_ judge: ResolvedJudge) {
            self.init(model: judge.model, revision: judge.revision)
        }
    }

    /// One loaded container per DISTINCT (model, revision) a local panel
    /// names — the shared core of the paired-judging and response-coding
    /// loops, which held two copies of this cache and the same hole twice.
    ///
    /// Generic over what a load returns so a test can count loads and
    /// inspect the mapping without weights on the machine: the study path
    /// instantiates it with `ModelContainer`, `JudgeContainerCacheTests`
    /// with a marker string.
    static func loadLocalJudgeContainers<Container>(
        for judges: [ResolvedJudge],
        load: (ResolvedJudge) async throws -> Container
    ) async rethrows -> [LoadedModelKey: Container] {
        var containers: [LoadedModelKey: Container] = [:]
        for judge in judges where judge.kind == "local" {
            let key = LoadedModelKey(judge)
            if containers[key] == nil {
                containers[key] = try await load(judge)
            }
        }
        return containers
    }

    /// The load line, naming the revision when one is pinned — a log that
    /// said only the model id could not distinguish the two loads the key
    /// above now keeps apart.
    static func localJudgeLoadLogLine(_ judge: ResolvedJudge) -> String {
        let pin = (judge.revision?.trimmingCharacters(in: .whitespacesAndNewlines))
            .flatMap { $0.isEmpty ? nil : " at revision \($0.prefix(12))…" }
        return "loading local judge model \(judge.model)\(pin ?? "")"
    }

    /// The judge panel an evaluation runs: `manifest.judges` when pinned
    /// (>=2 enforced at freeze), else a single legacy judge synthesized from
    /// the evaluation spec's judgeModel so pre-panel studies keep working.
    /// The provider stamped onto one judgment record: the VERIFIED serving
    /// provider the client read off the response (and refused unless it
    /// matched the pin), canonicalized — falling back to the canonical pin
    /// only for a verdict from an older client that carried none.
    ///
    /// Fixed 2026-07-24: this used to stamp `judge.provider` verbatim, so
    /// the same OpenRouter judge recorded `"Google AI Studio"` from a Mac
    /// inline evaluate and `"google-ai-studio"` from the server, and the
    /// recorded string was the REQUESTED provider rather than the verified
    /// one. Cross-engine twin: `tasks._judgment_stamp_judge`.
    static func verifiedJudgeProvider(
        judge: ResolvedJudge, judgment: PairedJudgeResponse
    ) -> String? {
        guard judge.kind == "openrouter" else { return nil }
        let served = judgment.provider ?? judge.provider ?? ""
        let canonical = OpenRouterProviderIdentity.canonical(served)
        return canonical.isEmpty ? nil : canonical
    }

    /// Check every OpenRouter judge's provider pin before any judging.
    ///
    /// Twin of the server's `_preflight_openrouter_judges`, including its
    /// deliberate asymmetry: refuse only on POSITIVE evidence that the pin
    /// is wrong (the catalogue answered and the provider does not serve
    /// this model). An unreachable catalogue warns and proceeds — a laptop
    /// offline, or a compute node with no outbound network, must not make
    /// a study unrunnable, and the call-time off-pin refusal still
    /// guarantees correctness.
    static func preflightOpenRouterJudges(
        _ judges: [ResolvedJudge],
        fetch: OpenRouterCatalog.Fetcher? = nil
    ) async throws {
        for judge in judges where judge.kind == "openrouter" {
            let result = await OpenRouterCatalog.preflight(
                model: judge.model, provider: judge.provider ?? "",
                fetch: fetch)
            for warning in result.warnings {
                print("WARNING: judge '\(judge.name)': \(warning)")
            }
            if let problem = result.problem {
                throw ExperimentError(
                    reason: "judge '\(judge.name)': \(problem)")
            }
            if result.checked {
                print(
                    "judge '\(judge.name)': provider pin "
                        + "'\(OpenRouterProviderIdentity.canonical(judge.provider ?? ""))' "
                        + "verified against OpenRouter's catalogue for "
                        + "'\(judge.model)'")
            }
        }
    }

    static func resolvedJudges(
        manifest: ExperimentManifest,
        evaluation: ExperimentManifest.EvaluationSpec?
    ) throws -> [ResolvedJudge] {
        if let judges = manifest.judges, !judges.isEmpty {
            return try judges.map { judge in
                let model = (judge.model ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                switch judge.kind {
                case "claude":
                    return ResolvedJudge(
                        name: judge.name, kind: "claude",
                        model: model.isEmpty ? ClaudePairedJudge.defaultModel : model)
                case "openrouter":
                    // No defaults to fill (server rule, 2026-07-19): an
                    // explicit model slug AND a pinned provider are
                    // required — resolution refuses rather than inventing
                    // either.
                    let provider = (judge.provider ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !model.isEmpty else {
                        throw ExperimentError(
                            reason: "openrouter judge '\(judge.name)' has "
                                + "no model slug — there is no default to "
                                + "resolve")
                    }
                    guard !provider.isEmpty else {
                        throw ExperimentError(
                            reason: "openrouter judge '\(judge.name)' has "
                                + "no pinned provider — an unpinned "
                                + "provider is not a pinned judge")
                    }
                    return ResolvedJudge(
                        name: judge.name, kind: "openrouter", model: model,
                        provider: provider)
                case "local":
                    // Cross-engine rule (2026-07-08): a local judge with an
                    // empty/absent model resolves to the STUDY model — it
                    // judges with the same model that generated the outputs.
                    // Never judge-NAME-as-model-id (the server bug this rule
                    // replaced).
                    guard !model.isEmpty else {
                        return ResolvedJudge(
                            name: judge.name, kind: "local",
                            model: manifest.modelID, modelDefaulted: true,
                            revision: judge.revision ?? manifest.modelRevision,
                            dtype: judge.dtype ?? manifest.dtype)
                    }
                    return ResolvedJudge(
                        name: judge.name, kind: "local", model: model,
                        revision: judge.revision
                            ?? (model == manifest.modelID
                                ? manifest.modelRevision : nil),
                        dtype: judge.dtype
                            ?? (model == manifest.modelID
                                ? manifest.dtype : nil))
                default:
                    throw ExperimentError(
                        reason: "judge '\(judge.name)' has unknown kind "
                            + "'\(judge.kind)' (expected claude|openrouter|local)")
                }
            }
        }
        let model = (evaluation?.judgeModel ?? ClaudePairedJudge.defaultModel)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // The ad-hoc selector is a single string, read through the SAME
        // spelling the Robustness Check's judge uses — so an `openrouter:…`
        // pick resolves to an openrouter judge with its provider pin intact
        // rather than being mistaken for a local repo id. Blank still means
        // the default Claude judge.
        guard let selection = JudgeModelSpelling.parse(model) else {
            return [
                ResolvedJudge(
                    name: "judge-1", kind: "claude",
                    model: ClaudePairedJudge.defaultModel)
            ]
        }
        if case .openRouterUnpinned(let slug) = selection {
            throw ExperimentError(
                reason: JudgeModelSpelling.unpinnedProviderRefusal(model: slug))
        }
        return [
            ResolvedJudge(
                name: "judge-1", kind: selection.kind, model: selection.model,
                provider: selection.provider)
        ]
    }

    /// The evaluate-start log line for one LOCAL judge (cross-engine wording
    /// with the server's `tasks.evaluate`): a judge resolved to the study
    /// model says so — the judge's name is a label, never a model id — and a
    /// judge naming a different local model names it. Pure, so the wording
    /// is unit-testable.
    static func localJudgeResolutionLogLine(
        _ judge: ResolvedJudge, studyModelID: String
    ) -> String {
        judge.model == studyModelID
            ? "local judge '\(judge.name)' resolves to the study model "
                + studyModelID
            : "local judge '\(judge.name)' judges with local model "
                + "'\(judge.model)'"
    }

    /// Plain-language load failure for a local judge that genuinely declares
    /// a different model (cross-engine wording family with the server): name
    /// the judge, the model, and the remedies — never surface the raw
    /// hub/network dump, which is misleading when the machine is offline.
    static func localJudgeLoadFailureMessage(
        judgeName: String, model: String
    ) -> String {
        "local judge '\(judgeName)' declares model '\(model)', which could "
            + "not be loaded — install the model on this Mac, or leave the "
            + "judge's model empty to judge with the study model"
    }

    /// The same failure, decided BEFORE the loader is asked (review round 7,
    /// finding 1). `SteeredContainerLoader.load` downloads what it cannot
    /// find, so "load it and see" turns a judge model this Mac does not hold
    /// into an invisible multi-gigabyte fetch at the start of an evaluate.
    /// Refusing on the is-installed test — the same membership test every
    /// installed badge reads — costs a directory listing.
    ///
    /// `revision` names the PIN when the judge declares one (round 8, finding
    /// 3): the cache can hold this model at a different revision, in which
    /// case "not installed" on its own sends the reader to look at a Models
    /// list that shows the model present.
    static func localJudgeNotInstalledMessage(
        judgeName: String, model: String, revision: String? = nil
    ) -> String {
        let pin = (revision?.trimmingCharacters(in: .whitespacesAndNewlines))
            .flatMap { $0.isEmpty ? nil : " at the pinned revision \($0.prefix(12))…" }
            ?? ""
        return "local judge '\(judgeName)' declares model '\(model)', which is "
            + "not installed on this Mac\(pin) — install it first; a judging "
            + "run never downloads weights on your behalf"
    }

    /// TEST SEAM (counting fake judge): when set, every judge call routes
    /// here instead of the Claude/OpenRouter/local clients, so tests can
    /// assert exactly which pairs were judged (e.g. that an excluded
    /// record's judge call never happens) without a model or network.
    /// nonisolated(unsafe) is justified as with
    /// `ExperimentStore.rootOverride`: mutated only by tests holding the
    /// shared override lock.
    nonisolated(unsafe) static var judgeOverrideForTesting:
        (@Sendable (
            _ judgeName: String, _ prompt: String,
            _ responseA: String, _ responseB: String
        ) async throws -> PairedJudgeResponse)?

    /// Run-status tracking shell around `evaluatePairedJudgeImpl`
    /// (2026-07-27): same contract and rationale as `run` — a failed local
    /// evaluate must leave `run-status.json` + `FAILED.md` beside whatever
    /// judgments it produced, never an unannotated directory that reads as
    /// legacy/trusted. The tracker learns the evaluate directory from the
    /// `.evaluationDirectory` progress event.
    @discardableResult
    public static func evaluatePairedJudge(
        experimentName: String,
        sourceRunDirectory: URL,
        evaluation override: ExperimentManifest.EvaluationSpec? = nil,
        allowUnverifiedEpoch: Bool = false,
        /// A seeded, stratified per-condition draw over the source run's
        /// records instead of coding all of them (2026-08-29). Per-response
        /// coding only — a paired evaluate refuses, because a pair is not a
        /// record (`EvaluateSubsample.pairedRefusal`).
        subsample: EvaluateSubsample.Request? = nil,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        progress: StudyTaskProgressHandler? = nil
    ) async throws -> URL {
        // A coding-mode rubric (2026-08-04) writes codings, not judgments —
        // the status file must name the artifact that actually exists. The
        // peek is best-effort: any load/parse failure here recurs loudly in
        // the impl below.
        let isCoding: Bool = {
            guard
                let manifest = try? ExperimentStore.load(name: experimentName),
                let rubric = try? JudgeRubricStore.resolveRubric(
                    for: manifest,
                    inlineRubric: override?.judgePrompt
                        ?? manifest.evaluation?.judgePrompt)
            else { return false }
            // `try?` flattens the optional: a paired rubric (parse returns
            // nil) and a malformed declaration (parse throws) both land as
            // nil here — only a well-formed coding declaration is true.
            return (try? ResponseCoding.parseRubric(rubric.text)) != nil
        }()
        let status = RunStatusFile.Tracker(
            stage: "evaluate", experiment: experimentName,
            sourceRun: sourceRunDirectory.lastPathComponent,
            itemLabel: isCoding ? "coding" : "judgment",
            itemsFile: isCoding ? "codings.jsonl" : "judgments.jsonl")
        let tracked: StudyTaskProgressHandler = { event in
            switch event {
            case .evaluationDirectory(let path):
                await status.begin(directoryPath: path)
            case .judgmentCompleted, .codingCompleted:
                // Keep `itemsWritten` current ON DISK, as the Python writer
                // does per item: a hard stop must not leave a directory of
                // real judgments described as holding zero.
                await status.noteItem()
            default:
                break
            }
            await progress?(event)
        }
        do {
            let url = try await evaluatePairedJudgeImpl(
                experimentName: experimentName,
                sourceRunDirectory: sourceRunDirectory,
                evaluation: override,
                allowUnverifiedEpoch: allowUnverifiedEpoch,
                subsample: subsample,
                shouldCancel: shouldCancel, progress: tracked,
                onInvalidVerdict: { await status.noteInvalidResponse($0) })
            await status.finish()
            return url
        } catch {
            await status.fail(error)
            throw error
        }
    }

    private static func evaluatePairedJudgeImpl(
        experimentName: String,
        sourceRunDirectory: URL,
        evaluation override: ExperimentManifest.EvaluationSpec?,
        allowUnverifiedEpoch: Bool,
        subsample: EvaluateSubsample.Request? = nil,
        shouldCancel: (@Sendable () async -> Bool)?,
        progress: StudyTaskProgressHandler?,
        onInvalidVerdict: (@Sendable ([String: String]) async -> Void)? = nil
    ) async throws -> URL {
        let manifest = try loadVerified(experimentName)
        // The study's DECLARED sampling design outranks the flags, and the
        // flags are checked against it (review round 12). Reconciled HERE
        // rather than at the CLI edge deliberately: this is the first point
        // that holds the manifest, so the CLI, the panel and any later
        // caller all get the same cross-check on the same bytes.
        let subsample = try EvaluateSubsample.reconcile(
            flags: subsample,
            declaration: try EvaluateSubsample.declaredRequest(
                manifest.evaluationSampling, experiment: experimentName,
                program: "steerlab-cli"),
            program: "steerlab-cli")
        if let subsample, subsample.declared {
            print(
                "evaluate: the study declares a sampling design — "
                    + "\(subsample.samplePerCondition) record(s) per "
                    + "condition at seed \(subsample.seedText) "
                    + "(evaluationSampling)")
        }
        let cancel = CancelPoller(shouldCancel)
        // Epoch guard: the source run must have been produced under THIS
        // manifest (name matching alone let a pre-edit draft run be judged
        // under a frozen manifest).
        let epoch = try verifyRunEpoch(
            verb: "evaluate", runDirectory: sourceRunDirectory,
            manifest: manifest, allowUnverified: allowUnverifiedEpoch)
        // Effective evaluation (2026-07-22 incident): an explicit spec (the
        // caller's override, else the manifest block) wins; with neither,
        // pinned judges + a pinned rubric file ARE the paired-judge
        // declaration — the spec is synthesized from those pins instead of
        // judging on legacy fallbacks. The judge report stamps where the
        // spec came from (cross-engine key "evaluationSource": "manifest" |
        // "pinnedRubric"; nil = the legacy no-declaration fallback path).
        var evaluation = override ?? manifest.evaluation
        var evaluationSource: String? = evaluation == nil ? nil : "manifest"
        if evaluation == nil,
            let effective = ExperimentStore.effectiveEvaluation(manifest)
        {
            evaluation = effective.spec
            evaluationSource = effective.source
        }
        if let evaluation, evaluation.kind != .pairedJudge {
            throw ExperimentError(reason: "study has no paired-judge evaluation")
        }
        // Rubric: the pinned rubric file wins whenever present (hash-checked
        // — drift throws); inline draft text is the fallback.
        let rubric = try JudgeRubricStore.resolveRubric(
            for: manifest, inlineRubric: evaluation?.judgePrompt)
        let structuredPrompt = evaluation?.structuredPrompt
        let judges = try resolvedJudges(manifest: manifest, evaluation: evaluation)
        // Local-judge resolution, logged at evaluate START (cross-engine
        // rule, unified with the sweep 2026-07-22 — the judge's NAME is a
        // label, never a model id; the server's dead `model or name`
        // fallback sent 'judge-1' to HuggingFace as a model id on an
        // offline compute node). Server twin: `tasks.evaluate`.
        for judge in judges where judge.kind == "local" {
            print(localJudgeResolutionLogLine(
                judge, studyModelID: manifest.modelID))
        }
        // Provider pins verified against OpenRouter's public catalogue
        // BEFORE the first paid call (2026-07-24). The check existed on
        // this engine but was reachable only from the manual Discover
        // control — so Swift-side judging still learned about a wrong pin
        // from an off-pin refusal mid-run, which the server had already
        // stopped doing. The call-time refusal remains either way.
        try await preflightOpenRouterJudges(judges)

        let loaded = try loadEvaluationGenerations(
            sourceRunDirectory.appending(component: "generations.jsonl"))
        try verifyEvaluationSourceRun(
            sourceRunDirectory,
            manifest: manifest,
            generations: loaded.map(\.record))

        // Declared exclusion rules join HERE — BEFORE judging, so no judge
        // call is ever spent on an excluded record. Pairwise deletion
        // matches analyze: filtering the baseline record removes its item
        // from every condition's pairs (the baseline join below simply
        // finds no partner). The stamp lands in judge-report.json +
        // exclusions.json; excluded records stay in the source run's
        // generations.jsonl (runs are immutable). No rules declared =
        // today's behavior byte-for-byte. Server twin: `tasks.evaluate`.
        let exclusionRules = manifest.exclusionRules ?? []
        let ruleProblems = ExclusionEngine.violations(exclusionRules)
        guard ruleProblems.isEmpty else {
            throw ExperimentError(reason: ruleProblems.joined(separator: "; "))
        }
        var exclusionStamp: ExclusionStamp?
        var judgeable = loaded
        if !exclusionRules.isEmpty {
            var checks: [String: AttentionCheck] = [:]
            if ExclusionEngine.needsChecks(exclusionRules) {
                guard manifest.taskPromptsHash != nil else {
                    // WP0 step 8: the deferred cross-engine-twinned message gets
                // its id on BOTH engines. The STRING is unchanged and stays
                // byte-identical to the server's `PIN_REQUIRED_MESSAGE`
                // (asserted on both sides); only the gate id and the runnable
                // repair are new.
                throw ExperimentError.refusing(
                    .missingPrerequisite, ExclusionEngine.pinRequiredMessage,
                    repair: ExclusionEngine.pinRequiredRepair)
                }
                checks = attentionChecks(
                    of: try loadTaskPrompts(for: manifest).prompts)
                guard !checks.isEmpty else {
                    throw ExperimentError(reason: ExclusionEngine.noChecksMessage)
                }
            }
            let endpointNames = Set(
                exclusionRules
                    .filter { $0.rule != ExclusionEngine.ruleFailedAttentionCheck }
                    .map(ExclusionEngine.resolvedEndpoint))
            let outcome = ExclusionEngine.evaluate(
                rules: exclusionRules, checks: checks,
                views: loaded.map { entry in
                    ExclusionEngine.RecordView(
                        condition: entry.record.condition,
                        seed: entry.record.seed,
                        promptID: entry.record.promptID,
                        output: entry.record.output,
                        endpoints: analysisEndpoints(
                            jsonLine: entry.line, names: endpointNames))
                },
                note: ExclusionEngine.evaluateNote,
                scope: ExclusionEngine.scopeSampledRecords)
            exclusionStamp = outcome.stamp
            judgeable = loaded.filter { entry in
                !outcome.excludedKeys.contains(
                    ExclusionEngine.rowKey(
                        condition: entry.record.condition,
                        seed: entry.record.seed,
                        promptID: entry.record.promptID))
            }
            print(
                "exclusions: \(outcome.stamp.excludedRecords) record(s) "
                    + "excluded before judging by \(exclusionRules.count) "
                    + "declared rule(s); surviving N per condition: "
                    + outcome.stamp.survivingN.sorted { $0.key < $1.key }
                        .map { "\($0.key)=\($0.value)" }
                        .joined(separator: ", "))
        }

        let generations = judgeable.map(\.record)
        // Per-response coding fork (2026-08-04, server twin:
        // `tasks._evaluate_response_coding`): a rubric whose frontmatter
        // declares `mode: perResponseCoding` runs the coding instrument —
        // every sampled-text record coded individually, blinded, no
        // pairing and no winner. Everything above (epoch guard, exclusions,
        // roster resolution, provider preflight) is shared; everything
        // below is paired-only.
        if let codingSchema = try ResponseCoding.parseRubric(rubric.text) {
            if let structured = structuredPrompt,
                !structured.trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
            {
                throw ExperimentError(
                    reason: "the study declares a paired "
                        + "structured-comparison prompt but the pinned "
                        + "rubric is perResponseCoding — the two contracts "
                        + "cannot combine; clear the structured prompt or "
                        + "pin a paired rubric")
            }
            guard manifest.humanValidation == nil else {
                throw ExperimentError(
                    reason: "judge-vs-human agreement for per-response "
                        + "coding is not implemented yet — unpin "
                        + "humanValidation (the paired-shape "
                        + "baseline|variant|tie labels do not describe "
                        + "per-response codes)")
            }
            return try await runResponseCoding(
                manifest: manifest,
                schema: codingSchema,
                rubric: rubric,
                judges: judges,
                generations: generations,
                sourceRunDirectory: sourceRunDirectory,
                exclusionStamp: exclusionStamp,
                epoch: epoch,
                evaluationSource: evaluationSource,
                cancel: cancel,
                progress: progress,
                subsample: subsample,
                onInvalidVerdict: onInvalidVerdict)
        }
        if subsample != nil {
            // Reached only on the paired path: the sample flags name a design
            // the paired judge's unit of analysis cannot express. Refusing
            // beats half-executing a correct-looking command line.
            throw EvaluateSubsample.pairedRefusal(program: "steerlab-cli")
        }
        // Join rule (cross-engine, external review 2026-07-22): pairs join
        // on (promptID, sampleIndex) — never the seed, which under derived
        // seeding differs between baseline and variant by design. Server
        // twin: `paired_judge._pair_generations`.
        let baseline = generations.filter { $0.condition == "baseline" }.reduce(
            into: [String: EvaluationGeneration]()
        ) { pairs, generation in
            pairs[
                evaluationKey(
                    sampleIndex: generation.sampleIndex ?? 0,
                    promptID: generation.promptID)] = generation
        }
        let candidates = generations.filter { $0.condition != "baseline" }
        guard !baseline.isEmpty, !candidates.isEmpty else {
            let excludedAll = (exclusionStamp?.excludedRecords ?? 0) > 0
            throw ExperimentError(
                reason: "paired judge requires baseline and non-baseline generations"
                    + (excludedAll
                        ? " — the declared exclusion rules left none to judge"
                        : ""))
        }
        // Zero-pairs refusal (P0 message family shared verbatim with the
        // server): both sides present but no candidate shares a cell with
        // a baseline record — a quiet empty judged report is never written.
        guard
            candidates.contains(where: { generation in
                baseline[
                    evaluationKey(
                        sampleIndex: generation.sampleIndex ?? 0,
                        promptID: generation.promptID)] != nil
            })
        else {
            throw ExperimentError(reason: Self.noPairsMessage)
        }

        let runDirectory = try makeRunDirectory(experiment: manifest, task: "evaluate")
        if let exclusionStamp {
            // The stamp file the server also writes — one artifact name to
            // look for on either engine, written BEFORE judging starts so
            // even a cancelled evaluation records what was excluded.
            let stampEncoder = JSONEncoder()
            stampEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try stampEncoder.encode(exclusionStamp).write(
                to: runDirectory.appending(component: "exclusions.json"))
        }
        await progress?(.evaluationDirectory(runDirectory.path))
        let experimentHash = ExperimentStore.manifestHash(manifest)
        // Local judge models load once each, even when the panel repeats a
        // model under two names. A judge resolved to the study model loads
        // at the manifest's PINNED revision — the same weights the study
        // ran; a load failure for a genuinely-declared different model is
        // wrapped in plain language (never the raw hub/network dump) with
        // the underlying error kept in the run log.
        var localContainers: [LoadedModelKey: ModelContainer] = [:]
        if judgeOverrideForTesting == nil {
            localContainers = try await loadLocalJudgeContainers(for: judges) {
                judge in
                // The guard asks about the EXACT revision the next line
                // loads (review round 8, finding 3) — a cache holding
                // revision A used to satisfy a judge pinned to B, and the
                // load then fetched B over the network, which is the one
                // thing this guard exists to prevent.
                guard
                    SteeredContainerLoader.isCached(
                        modelID: judge.model, revision: judge.revision)
                else {
                    throw ExperimentError(
                        reason: localJudgeNotInstalledMessage(
                            judgeName: judge.name, model: judge.model,
                            revision: judge.revision))
                }
                print(localJudgeLoadLogLine(judge))
                do {
                    // The judge's own pinned revision wins
                    // (JudgeRef.revision, 2026-07-23); a study-model
                    // judge already carries the study pin from
                    // resolution.
                    return try await SteeredContainerLoader.load(
                        modelID: judge.model, revision: judge.revision)
                } catch {
                    guard judge.model != manifest.modelID else { throw error }
                    print("judge model load failed: \(error)")
                    throw ExperimentError(
                        reason: localJudgeLoadFailureMessage(
                            judgeName: judge.name, model: judge.model))
                }
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let judgmentsURL = runDirectory.appending(component: "judgments.jsonl")
        FileManager.default.createFile(atPath: judgmentsURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: judgmentsURL)
        defer { try? handle.close() }

        var records: [PairedJudgeRecord] = []
        /// Pairs this panel answered without ever producing a verdict —
        /// recorded rows, never verdicts (see the catch below).
        ///
        /// This engine has no resume/reuse path (targeted retry of a failed
        /// evaluation is a server verb the app drives remotely), so the
        /// server's companion rule has no local twin yet: when reuse lands
        /// here, a noncompliant row from an earlier session is a RECORDED
        /// FAILURE, not a verdict, and the cell must be re-judged rather
        /// than reused or refused over.
        var noncompliantJudgments = 0
        var cancelled = false
        judgeLoop: for judge in judges {
            var judgePairs = 0
            var judgeNoncompliant = 0
            for generation in candidates {
                guard
                    let base = baseline[
                        evaluationKey(
                            sampleIndex: generation.sampleIndex ?? 0,
                            promptID: generation.promptID)]
                else { continue }
                if await cancel.observed(
                    at: "[\(judge.name)] \(generation.condition) \(generation.promptID)")
                {
                    cancelled = true
                    break judgeLoop
                }
                judgePairs += 1
                await progress?(
                    .judgmentStarted(
                        condition: generation.condition,
                        promptID: generation.promptID))
                // The A/B flip is derived from the ITEM, not the judge, so
                // every judge sees the same blinded assignment and agreement
                // is computed on the unblinded conditionResult. The seed
                // input is the VARIANT record's (recorded as variantSeed) —
                // kept so re-judging an existing greedy run reproduces its
                // historical blinding orientation.
                let flip = shouldFlip(
                    experiment: manifest.name,
                    condition: generation.condition,
                    seed: generation.seed,
                    promptID: generation.promptID)
                let responseA = flip ? generation.output : base.output
                let responseB = flip ? base.output : generation.output
                func obtainVerdict() async throws -> PairedJudgeResponse {
                    if let fake = judgeOverrideForTesting {
                        try await fake(
                            judge.name, generation.prompt, responseA, responseB)
                    } else if let localJudgeContainer =
                        localContainers[LoadedModelKey(judge)]
                    {
                        try await LocalPairedJudge.judge(
                            container: localJudgeContainer,
                            modelID: judge.model,
                            rubric: rubric.text,
                            structuredPrompt: structuredPrompt,
                            prompt: generation.prompt,
                            responseA: responseA,
                            responseB: responseB)
                    } else if judge.kind == "openrouter" {
                        // Provider-pinned by construction: resolvedJudges
                        // refused an openrouter judge without one, and the
                        // client refuses a response served off-pin (or
                        // unattributed).
                        try await OpenRouterPairedJudge.judge(
                            model: judge.model,
                            provider: judge.provider ?? "",
                            rubric: rubric.text,
                            structuredPrompt: structuredPrompt,
                            prompt: generation.prompt,
                            responseA: responseA,
                            responseB: responseB)
                    } else {
                        try await ClaudePairedJudge.judge(
                            model: judge.model,
                            rubric: rubric.text,
                            structuredPrompt: structuredPrompt,
                            prompt: generation.prompt,
                            responseA: responseA,
                            responseB: responseB)
                    }
                }
                // Invalid-verdict closure (2026-07-20): an out-of-vocabulary
                // winner is retried once, then refuses — never recorded as
                // an invented tie (a tie is substantive data; inventing it
                // corrupts the preference tallies).
                let judgment: PairedJudgeResponse
                do {
                    judgment = try await judgmentWithValidWinner(
                        judgeName: judge.name,
                        item: "pair \(generation.condition)/"
                            + "\(generation.promptID)",
                        onInvalid: onInvalidVerdict,
                        obtainVerdict)
                } catch let error as JudgeNoncompliantError {
                    // A judge that ANSWERS but will not produce a verdict
                    // for THIS pair no longer kills the whole evaluation
                    // (Christian, 2026-08-09): after hours of generation
                    // and hundreds of good judgments, one flaky refusal
                    // was aborting entire runs. The refusal-to-invent
                    // stands — no winner is recorded — but the failure
                    // becomes a per-pair ROW, kept for later examination
                    // and classification, excluded from every tally and
                    // agreement statistic (it is not a PairedJudgeRecord,
                    // so no aggregate can see it). Systemic failure is
                    // different (the cap check after this judge's column),
                    // and TRANSPORT failure is different too — every other
                    // error type propagates from here and still fails the
                    // session, which resuming completes without re-paying
                    // finished judgments.
                    judgeNoncompliant += 1
                    noncompliantJudgments += 1
                    let reason = JudgeNoncompliance.recordedReason(error)
                    let row = NoncompliantJudgmentRecord(
                        promptID: generation.promptID,
                        sampleIndex: generation.sampleIndex ?? 0,
                        condition: generation.condition,
                        baselineSeed: base.seed,
                        variantSeed: generation.seed,
                        baselineWas: flip ? "B" : "A",
                        noncomplianceReason: reason,
                        judgeName: judge.name,
                        judgeProvider: judge.kind == "openrouter"
                            ? judge.provider : nil)
                    let rowData = try encoder.encode(row)
                    try handle.write(contentsOf: rowData)
                    try handle.write(contentsOf: Data("\n".utf8))
                    // Counted like any other written row: a status file
                    // that under-reports what is on disk is its own bug.
                    await progress?(
                        .judgmentCompleted(
                            StudyJudgePreview(
                                condition: generation.condition,
                                sampleIndex: generation.sampleIndex ?? 0,
                                baselineSeed: base.seed,
                                variantSeed: generation.seed,
                                promptID: generation.promptID,
                                prompt: generation.prompt,
                                baselineWas: flip ? "B" : "A",
                                conditionWas: flip ? "A" : "B",
                                winner: "",
                                conditionResult: "noncompliant",
                                confidence: 0,
                                briefReason: reason,
                                aScores: nil, bScores: nil,
                                structuredFields: nil,
                                rawJSON: String(
                                    decoding: rowData, as: UTF8.self))))
                    print(
                        "NONCOMPLIANT [\(judge.name)] \(generation.condition) "
                            + "\(generation.promptID): no verdict recorded — "
                            + "row kept for review")
                    continue
                }
                let result: String
                switch judgment.winner.lowercased() {
                case "a":
                    result = flip ? "condition" : "baseline"
                case "b":
                    result = flip ? "baseline" : "condition"
                default:
                    result = "tie"  // "tie" — validated above
                }
                let record = PairedJudgeRecord(
                    experiment: manifest.name,
                    experimentHash: experimentHash,
                    sourceRunDirectory: sourceRunDirectory.path,
                    judgeName: judge.name,
                    judgeKind: judge.kind,
                    judgeModel: judge.model,
                    judgeProvider: Self.verifiedJudgeProvider(
                        judge: judge, judgment: judgment),
                    judgeRevision: judge.kind == "local" ? judge.revision : nil,
                    judgePrompt: rubric.text,
                    judgeRubricFile: rubric.file,
                    judgeRubricHash: rubric.hash,
                    structuredPrompt: structuredPrompt,
                    condition: generation.condition,
                    sampleIndex: generation.sampleIndex ?? 0,
                    baselineSeed: base.seed,
                    variantSeed: generation.seed,
                    promptID: generation.promptID,
                    prompt: generation.prompt,
                    baselineWas: flip ? "B" : "A",
                    conditionWas: flip ? "A" : "B",
                    judgment: judgment,
                    conditionResult: result)
                records.append(record)
                let recordData = try encoder.encode(record)
                try handle.write(contentsOf: recordData)
                try handle.write(contentsOf: Data("\n".utf8))
                await progress?(
                    .judgmentCompleted(
                        judgePreview(
                            from: record,
                            rawJSON: String(decoding: recordData, as: UTF8.self))))
                print(
                    "judged [\(judge.name)] \(generation.condition) "
                        + "\(generation.promptID): \(result) (\(judgment.confidence))")
            }
            if judgeNoncompliant > 0 {
                print(
                    "judge '\(judge.name)' judged "
                        + "\(judgePairs - judgeNoncompliant) pair(s) "
                        + "(\(judgeNoncompliant) noncompliant, kept as rows "
                        + "for review)")
            }
            // A few flaky refusals are survivable; a judge that fails a
            // quarter of its column is broken, and a "completed"
            // evaluation built on it would be worse than a failed one.
            // Every row — compliant and not — is already on disk, so the
            // refusal loses nothing.
            if judgeNoncompliant > 0, judgePairs > 0,
                Double(judgeNoncompliant) / Double(judgePairs)
                    > JudgeNoncompliance.cap
            {
                throw ExperimentError(
                    reason: "judge '\(judge.name)' was noncompliant on "
                        + "\(judgeNoncompliant) of \(judgePairs) pairs "
                        + "(> \(JudgeNoncompliance.capPercentText) cap) — "
                        + "this is systemic judge failure, not flakiness. "
                        + "All rows (including the noncompliant ones, with "
                        + "raw reasons) were persisted; fix or swap the "
                        + "judge, then re-run evaluate")
            }
        }

        if cancelled {
            // Judgments completed so far stay in judgments.jsonl; the status
            // note marks the directory and NO judge-report.json is written —
            // a partial panel is never summarized as a report.
            writeCancellationNote(task: "paired-judge evaluation", to: runDirectory)
            print(
                "paired judge cancelled by user — \(records.count) judgment(s) "
                    + "kept in \(runDirectory.lastPathComponent); no judge report written")
            return runDirectory
        }

        // Human-validation subset, when pinned: hash-checked at read time
        // (verify() already gates it, but this run must not read a drifted
        // file either).
        var humanRows: [HumanValidationRow] = []
        if let human = manifest.humanValidation {
            let url = ExperimentStore.resolveProjectPath(human.path)
            let data = try Data(contentsOf: url)
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard hash == human.hash else {
                throw ExperimentError(
                    reason: "human validation set '\(human.path)' drifted from the "
                        + "pinned hash")
            }
            humanRows = try parseHumanValidation(data)
        }

        // Conditions aggregate across the whole judge panel (pairs therefore
        // count judges × items); per-judge splits live in the records.
        let grouped = Dictionary(grouping: records, by: \.condition)
        let report = PairedJudgeReport(
            experiment: manifest.name,
            experimentHash: experimentHash,
            sourceRunDirectory: sourceRunDirectory.path,
            judgeModel: judges.map(\.model).joined(separator: ", "),
            judges: judges.map(\.name),
            judgeRubricFile: rubric.file,
            judgeRubricHash: rubric.hash,
            judgeAgreement: judgeAgreement(records: records, judges: judges.map(\.name)),
            humanAgreement: humanRows.isEmpty
                ? nil
                : humanAgreement(
                    records: records, judges: judges.map(\.name), human: humanRows),
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            evaluationSource: evaluationSource,
            conditions: grouped.mapValues { rows in
                PairedJudgeConditionReport(
                    pairs: rows.count,
                    conditionWins: rows.filter { $0.conditionResult == "condition" }.count,
                    baselineWins: rows.filter { $0.conditionResult == "baseline" }.count,
                    ties: rows.filter { $0.conditionResult == "tie" }.count,
                    meanConfidence: rows.isEmpty
                        ? 0
                        : rows.map(\.judgment.confidence).reduce(0, +) / Double(rows.count),
                    structuredSummaries: structuredSummaries(rows))
            },
            exclusions: exclusionStamp,
            judgeUsage: judgeUsageTotals(records: records),
            // Nonzero-only: a clean evaluation's report is unchanged.
            noncompliantJudgments: noncompliantJudgments > 0
                ? noncompliantJudgments : nil)
        try encoder.encode(report).write(to: runDirectory.appending(component: "judge-report.json"))
        print("evaluation artifacts: \(runDirectory.path)")
        return runDirectory
    }

    /// Per-judge token sums over a run's judgment records (2026-08-06).
    ///
    /// REPORT, NEVER GATE (researcher directive): these totals land in the
    /// judge report so a researcher can SEE that a judge is spending far
    /// more on hidden reasoning than the answer needs — no code reads them
    /// to refuse, cap, or select. Judges whose transport reports no usage
    /// (local and Claude judges today) contribute no entry, and a run with
    /// no reported usage at all returns nil so the key is simply omitted.
    /// Server twin: `paired_judge.sum_judgment_usage`, whose sums are
    /// stamped on each per-judge report block under the same key names.
    static func judgeUsageTotals(
        records: [PairedJudgeRecord]
    ) -> [String: PairedJudgeUsage]? {
        var totals: [String: PairedJudgeUsage] = [:]
        for record in records {
            guard let usage = record.judgment.usage, !usage.isEmpty else { continue }
            var running = totals[record.judgeName] ?? PairedJudgeUsage()
            if let completion = usage.completionTokens {
                running.completionTokens = (running.completionTokens ?? 0) + completion
            }
            if let reasoning = usage.reasoningTokens {
                running.reasoningTokens = (running.reasoningTokens ?? 0) + reasoning
            }
            totals[record.judgeName] = running
        }
        return totals.isEmpty ? nil : totals
    }

    // MARK: - Per-response coding instrument (2026-08-04)

    /// Test seam for the coding loop: (judgeName, codingPrompt) → the
    /// judge's raw text response (sibling of `judgeOverrideForTesting`).
    nonisolated(unsafe) static var codingOverrideForTesting:
        (@Sendable (String, String) async throws -> String)?

    /// One coding row in `codings.jsonl` (cross-engine keys — server twin:
    /// the row dict in `tasks._evaluate_response_coding`).
    struct CodingRecord: Codable {
        enum CodingKeys: String, CodingKey {
            case experiment, condition, promptID, sampleIndex, seed
            case wordCount, codes, undeclaredCodes, briefReason
            // Server contract: coding rows stamp the judge's panel name
            // under the key "judge".
            case judgeName = "judge"
            case judgeKind, judgeModel, judgeProvider, judgeRevision
        }
        let experiment: String
        let condition: String
        let promptID: String
        let sampleIndex: UInt64
        let seed: UInt64
        let wordCount: Int
        /// The DECLARED fields — the measurement, and the only thing the
        /// aggregates and agreement statistics read.
        let codes: [String: JSONValue]
        /// Keys the coder invented that the rubric never declared: kept
        /// verbatim, kept out of `codes`, and absent from the row entirely
        /// when the coder invented nothing (the ordinary case). NOT
        /// evidence — nothing aggregates it.
        let undeclaredCodes: [String: JSONValue]?
        let briefReason: String
        let judgeName: String
        let judgeKind: String
        let judgeModel: String
        let judgeProvider: String?
        let judgeRevision: String?
    }

    /// One NONCOMPLIANT coding row in `codings.jsonl` — a record whose
    /// coder answered twice and produced no valid code set either time
    /// (Christian, 2026-08-09; server twin: the `codes: None` row dict in
    /// `tasks._evaluate_response_coding`).
    ///
    /// The paired judge's policy, applied to coding: the refusal to invent
    /// codes stands, but the failure is KEPT as a classifiable row instead
    /// of aborting a column that had already been paid for. Identity keys
    /// match a normal coding row exactly; `codes` is present-and-NULL, so
    /// the hole is visible rather than inferred. Not a `CodingRecord`,
    /// which is why no aggregate or agreement statistic can reach it.
    struct NoncompliantCodingRecord: Encodable {
        enum CodingKeys: String, CodingKey {
            case experiment, condition, promptID, sampleIndex, seed
            case codes, noncompliant, noncomplianceReason
            case judgeName = "judge"
            case judgeKind, judgeModel
        }

        let experiment: String
        let condition: String
        let promptID: String
        let sampleIndex: UInt64
        let seed: UInt64
        let noncomplianceReason: String
        let judgeName: String
        let judgeKind: String
        let judgeModel: String

        /// Hand-written for the same reason as the judgment twin: a nil
        /// optional would be OMITTED, and an absent `codes` reads as a
        /// malformed row rather than a recorded refusal.
        func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(experiment, forKey: .experiment)
            try container.encode(condition, forKey: .condition)
            try container.encode(promptID, forKey: .promptID)
            try container.encode(sampleIndex, forKey: .sampleIndex)
            try container.encode(seed, forKey: .seed)
            try container.encodeNil(forKey: .codes)
            try container.encode(true, forKey: .noncompliant)
            try container.encode(
                noncomplianceReason, forKey: .noncomplianceReason)
            try container.encode(judgeName, forKey: .judgeName)
            try container.encode(judgeKind, forKey: .judgeKind)
            try container.encode(judgeModel, forKey: .judgeModel)
        }
    }

    struct CodingFieldAggregate: Codable, Equatable {
        let n: Int
        let nulls: Int
        var trueCount: Int? = nil
        var trueShare: Double? = nil
        var mean: Double? = nil
        var counts: [String: Int]? = nil
    }

    struct CodingConditionReport: Codable, Equatable {
        let codedResponses: Int
        let codings: Int
        let meanWordCount: Double?
        let fields: [String: CodingFieldAggregate]
    }

    struct CodingAgreementEntry: Codable, Equatable {
        let field: String
        let judgeA: String
        let judgeB: String
        let n: Int
        var percentAgreement: Double? = nil
        var kappa: Double? = nil
        var meanAbsoluteDifference: Double? = nil
        /// The confusion counts beside the statistic they explain, from the
        /// very label pairs kappa was computed over: `confusion[a]?[b]` is
        /// how many shared cells judgeA coded `a` while judgeB coded `b`,
        /// and the counts sum to `n`. Categorical fields only (cross-engine
        /// key "confusion") — an analysis layer must never have to
        /// re-derive the cell key, the intersection, or the label
        /// normalization to see WHERE two coders part ways.
        var confusion: [String: [String: Int]]? = nil
    }

    struct CodingJudgeDetail: Codable, Equatable {
        let name: String
        let kind: String
        let requestedModel: String
        let actualModel: String
        var revision: String? = nil
        /// Records THIS judge answered without ever producing valid codes
        /// (cross-engine key `noncompliantCodings`, nonzero-only): its
        /// column has that many holes, and per-judge is where a reader can
        /// tell one bad coder from a bad rubric.
        var noncompliantCodings: Int? = nil
    }

    /// `coding-report.json` (cross-engine keys with the server's report).
    struct CodingReport: Codable {
        let mode: String
        let experiment: String
        let experimentHash: String
        let sourceRun: String
        let judges: [String]
        let judgeModel: String
        let judgeDetails: [CodingJudgeDetail]
        let judgeRubricFile: String?
        let judgeRubricHash: String?
        let fields: [ResponseCoding.Field]
        let codings: Int
        let conditions: [String: CodingConditionReport]
        /// Pairwise inter-rater agreement, one entry per (judge pair, field)
        /// — OMITTED entirely for a single-coder run, where
        /// `fieldAgreementAbsentReason` says why instead. An EMPTY list
        /// reads as "agreement was measured and there was none", which is a
        /// different and false claim; a one-judge design has no pair to
        /// measure. (The gate stopped requiring two judges on 2026-08-28;
        /// this is the report's half of that ruling.)
        let fieldAgreement: [CodingAgreementEntry]?
        /// Present exactly when `fieldAgreement` is absent. Twin literal:
        /// `response_coding.SINGLE_CODER_AGREEMENT_ABSENT_REASON`.
        let fieldAgreementAbsentReason: String?
        let evaluationSource: String?
        let epochUnverified: Bool?
        /// Tolerated measurement-side drift, verbatim (cross-engine key
        /// "measurementDrift"); nil ⇒ key omitted.
        var measurementDrift: String? = nil
        let exclusions: ExclusionStamp?
        /// Records the whole panel answered without ever producing valid
        /// codes (Christian, 2026-08-09). Nonzero-only, like the paired
        /// judge's `noncompliantJudgments`: these rows carry no codes and
        /// sit outside every aggregate and agreement statistic — the report
        /// must say the columns are incomplete and by how much. Note
        /// `codings` above counts every row WRITTEN (the server's
        /// `len(rows)`), so the two together say how much of the file is
        /// measurement.
        var noncompliantCodings: Int? = nil
        /// The seeded, stratified subsample this report covers (2026-08-29,
        /// cross-engine key `sampling`); nil ⇒ key omitted ⇒ the FULL source
        /// corpus was coded, which is what every report written before this
        /// existed means.
        var sampling: EvaluateSubsample.Stamp? = nil
    }

    /// The per-response coding evaluate body (server twin:
    /// `tasks._evaluate_response_coding`). Every sampled-text record —
    /// baseline INCLUDED — goes to every judge individually and blinded
    /// (the coder sees the task prompt and one response, never the
    /// condition, never a second response). Codes are validated against
    /// the rubric's declared schema (retry once, then refuse — invented
    /// data is never recorded), streamed row-by-row to `codings.jsonl`,
    /// and summarized in `coding-report.json` with per-condition per-field
    /// aggregates, engine-computed word counts, and per-field inter-judge
    /// agreement. There is no pairing and no winner anywhere on this path.
    private static func runResponseCoding(
        manifest: ExperimentManifest,
        schema: ResponseCoding.Schema,
        rubric: (text: String, file: String?, hash: String?),
        judges: [ResolvedJudge],
        generations: [EvaluationGeneration],
        sourceRunDirectory: URL,
        exclusionStamp: ExclusionStamp?,
        epoch: RunEpoch.Check,
        evaluationSource: String?,
        cancel: CancelPoller,
        progress: StudyTaskProgressHandler?,
        subsample: EvaluateSubsample.Request? = nil,
        onInvalidVerdict: (@Sendable ([String: String]) async -> Void)?
    ) async throws -> URL {
        guard !generations.isEmpty else {
            throw ExperimentError(reason: ResponseCoding.noCodeableMessage)
        }
        // The seeded draw runs BEFORE the run directory is minted (2026-08-29,
        // server twin: `tasks._evaluate_response_coding`), deliberately: an
        // over-ask against a condition's population is a power-computation
        // error the caller must see, and a refusal that leaves a run
        // directory behind has already broken the rule that refusals never
        // write.
        let sourceTotal = generations.count
        var generations = generations
        var sampling: EvaluateSubsample.Stamp? = nil
        if let subsample {
            let kept = try EvaluateSubsample.selectedPositions(
                generations.map {
                    EvaluateSubsample.Coordinate(
                        condition: $0.condition, promptID: $0.promptID,
                        sampleIndex: $0.sampleIndex ?? 0)
                },
                request: subsample, program: "steerlab-cli")
            generations = kept.map { generations[$0] }
            sampling = EvaluateSubsample.stamp(
                subsample, sampled: generations.count, source: sourceTotal)
        }
        let codedPhrase = EvaluateSubsample.codedPhrase(
            sampling, total: sourceTotal)
        let runDirectory = try makeRunDirectory(
            experiment: manifest, task: "evaluate", sampling: sampling)
        if let exclusionStamp {
            let stampEncoder = JSONEncoder()
            stampEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try stampEncoder.encode(exclusionStamp).write(
                to: runDirectory.appending(component: "exclusions.json"))
        }
        await progress?(.evaluationDirectory(runDirectory.path))
        let experimentHash = ExperimentStore.manifestHash(manifest)
        print(
            "coding \(codedPhrase) × \(judges.count) "
                + "judge(s) under perResponseCoding rubric "
                + "'\(rubric.file ?? "(inline draft)")'")
        if let sampling {
            print(
                "seeded subsample: \(sampling.samplePerCondition) record(s) "
                    + "per condition at seed \(sampling.sampleSeed) — "
                    + "\(sampling.rule)")
        }
        // Local judge models load once each — same rules as the paired
        // loop (pinned revision wins; a study-model judge loads the
        // study's exact weights; load failures wrap in plain language).
        var localContainers: [LoadedModelKey: ModelContainer] = [:]
        if codingOverrideForTesting == nil {
            localContainers = try await loadLocalJudgeContainers(for: judges) {
                judge in
                // Same exact-revision guard as the paired loop above, and the
                // same (model, revision) cache identity.
                guard
                    SteeredContainerLoader.isCached(
                        modelID: judge.model, revision: judge.revision)
                else {
                    throw ExperimentError(
                        reason: localJudgeNotInstalledMessage(
                            judgeName: judge.name, model: judge.model,
                            revision: judge.revision))
                }
                print(localJudgeLoadLogLine(judge))
                do {
                    return try await SteeredContainerLoader.load(
                        modelID: judge.model, revision: judge.revision)
                } catch {
                    guard judge.model != manifest.modelID else {
                        throw error
                    }
                    print("judge model load failed: \(error)")
                    throw ExperimentError(
                        reason: localJudgeLoadFailureMessage(
                            judgeName: judge.name, model: judge.model))
                }
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let codingsURL = runDirectory.appending(component: "codings.jsonl")
        FileManager.default.createFile(atPath: codingsURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: codingsURL)
        defer { try? handle.close() }

        var records: [CodingRecord] = []
        var judgeDetails: [CodingJudgeDetail] = []
        /// Records the panel answered without ever producing valid codes —
        /// recorded rows, never measurements (see the catch below).
        var noncompliantCodings = 0
        var cancelled = false
        judgeLoop: for judge in judges {
            var judgeCodeable = 0
            var judgeNoncompliant = 0
            for generation in generations {
                if await cancel.observed(
                    at: "[\(judge.name)] \(generation.condition) "
                        + "\(generation.promptID)")
                {
                    cancelled = true
                    break judgeLoop
                }
                judgeCodeable += 1
                await progress?(
                    .judgmentStarted(
                        condition: generation.condition,
                        promptID: generation.promptID))
                let sampleIndex = generation.sampleIndex ?? 0
                let codingPrompt = ResponseCoding.buildPrompt(
                    schema: schema, response: generation.output,
                    taskPrompt: generation.prompt.isEmpty
                        ? nil : generation.prompt)
                let verdict: ResponseCoding.Verdict
                do {
                    verdict = try await ResponseCoding.codesWithValidSchema(
                        judgeName: judge.name,
                        item: "record \(generation.condition)/"
                            + "\(generation.promptID)[\(sampleIndex)]",
                        schema: schema,
                        onInvalid: onInvalidVerdict
                    ) { () async throws -> (text: String, provider: String?) in
                        if let fake = codingOverrideForTesting {
                            return (try await fake(judge.name, codingPrompt), nil)
                        } else if let container =
                            localContainers[LoadedModelKey(judge)]
                        {
                            return (
                                try await generate(
                                    container, prompt: codingPrompt,
                                    modelID: judge.model,
                                    maxTokens: PairedJudgeBudget.maxTokens,
                                    temperature: 0),
                                nil
                            )
                        } else if judge.kind == "openrouter" {
                            let result = try await OpenRouterPairedJudge.complete(
                                model: judge.model,
                                provider: judge.provider ?? "",
                                prompt: codingPrompt)
                            return (result.text, result.provider)
                        } else {
                            return (
                                try await ClaudePairedJudge.complete(
                                    model: judge.model, prompt: codingPrompt,
                                    responseSchema:
                                        ClaudePairedJudge.codingResponseSchema),
                                nil
                            )
                        }
                    }
                } catch let error as JudgeNoncompliantError {
                    // Same policy as the paired judge (Christian,
                    // 2026-08-09): one record the coder ANSWERS but will
                    // not code becomes a recorded, classifiable ROW —
                    // codes: null, outside every aggregate and agreement
                    // statistic — instead of aborting hours of completed
                    // work. Transport errors still fail the session (they
                    // are a different type and propagate from here);
                    // systemic noncompliance still fails too, at the cap
                    // check after this judge's column.
                    judgeNoncompliant += 1
                    noncompliantCodings += 1
                    let reason = JudgeNoncompliance.recordedReason(error)
                    let row = NoncompliantCodingRecord(
                        experiment: manifest.name,
                        condition: generation.condition,
                        promptID: generation.promptID,
                        sampleIndex: sampleIndex,
                        seed: generation.seed,
                        noncomplianceReason: reason,
                        judgeName: judge.name,
                        judgeKind: judge.kind,
                        judgeModel: judge.model)
                    let rowData = try encoder.encode(row)
                    try handle.write(contentsOf: rowData)
                    try handle.write(contentsOf: Data("\n".utf8))
                    await progress?(
                        .codingCompleted(
                            StudyCodingPreview(
                                judge: judge.name,
                                condition: generation.condition,
                                sampleIndex: sampleIndex,
                                promptID: generation.promptID,
                                wordCount: ResponseCoding.wordCount(
                                    generation.output),
                                codes: [:],
                                briefReason: reason)))
                    print(
                        "NONCOMPLIANT [\(judge.name)] \(generation.condition) "
                            + "\(generation.promptID): no codes recorded — "
                            + "row kept for review")
                    continue
                }
                let record = CodingRecord(
                    experiment: manifest.name,
                    condition: generation.condition,
                    promptID: generation.promptID,
                    sampleIndex: sampleIndex,
                    seed: generation.seed,
                    wordCount: ResponseCoding.wordCount(generation.output),
                    codes: verdict.codes,
                    undeclaredCodes: verdict.undeclaredCodes.isEmpty
                        ? nil : verdict.undeclaredCodes,
                    briefReason: verdict.briefReason,
                    judgeName: judge.name,
                    judgeKind: judge.kind,
                    judgeModel: judge.model,
                    judgeProvider: verdict.provider,
                    judgeRevision: judge.kind == "local"
                        ? judge.revision : nil)
                records.append(record)
                // Flushed per row: a hard stop still leaves every coding
                // it finished.
                let recordData = try encoder.encode(record)
                try handle.write(contentsOf: recordData)
                try handle.write(contentsOf: Data("\n".utf8))
                await progress?(
                    .codingCompleted(
                        StudyCodingPreview(
                            judge: judge.name,
                            condition: generation.condition,
                            sampleIndex: sampleIndex,
                            promptID: generation.promptID,
                            wordCount: record.wordCount,
                            codes: verdict.codes,
                            briefReason: verdict.briefReason)))
                print(
                    "coded [\(judge.name)] \(generation.condition) "
                        + "\(generation.promptID): \(verdict.codes.count) "
                        + "field(s)")
            }
            judgeDetails.append(
                CodingJudgeDetail(
                    name: judge.name, kind: judge.kind,
                    requestedModel: judge.model, actualModel: judge.model,
                    revision: judge.kind == "local" ? judge.revision : nil,
                    noncompliantCodings: judgeNoncompliant > 0
                        ? judgeNoncompliant : nil))
            print(
                "judge '\(judge.name)' coded "
                    + (sampling != nil
                        ? "\(judgeCodeable - judgeNoncompliant) of "
                            + "\(sourceTotal) record(s) (seeded subsample)"
                        : "\(judgeCodeable - judgeNoncompliant) record(s)")
                    + (judgeNoncompliant > 0
                        ? " (\(judgeNoncompliant) noncompliant, kept as rows "
                            + "for review)"
                        : ""))
            // A handful of refused records is survivable; a coder that
            // fails a quarter of its column is broken, and a report built
            // mostly on holes is not a result. Every row is already in
            // codings.jsonl, so the refusal loses nothing.
            if judgeNoncompliant > 0, judgeCodeable > 0,
                Double(judgeNoncompliant) / Double(judgeCodeable)
                    > JudgeNoncompliance.cap
            {
                throw ExperimentError(
                    reason: "judge '\(judge.name)' was noncompliant on "
                        + "\(judgeNoncompliant) of \(judgeCodeable) "
                        + "record(s) (> \(JudgeNoncompliance.capPercentText) "
                        + "cap) — systemic coder failure, not flakiness. "
                        + "Every row (including the noncompliant ones, with "
                        + "raw reasons) is in codings.jsonl; fix or swap the "
                        + "judge, then re-run evaluate")
            }
        }

        if cancelled {
            // Codings completed so far stay in codings.jsonl; no
            // coding-report.json is written — a partial panel is never
            // summarized as a report.
            writeCancellationNote(
                task: "per-response coding evaluation", to: runDirectory)
            print(
                "per-response coding cancelled by user — \(records.count) "
                    + "coding(s) kept in \(runDirectory.lastPathComponent); "
                    + "no coding report written")
            return runDirectory
        }

        let report = CodingReport(
            mode: "perResponseCoding",
            experiment: manifest.name,
            experimentHash: experimentHash,
            sourceRun: sourceRunDirectory.lastPathComponent,
            judges: judges.map(\.name),
            judgeModel: judges.map(\.model).joined(separator: ", "),
            judgeDetails: judgeDetails,
            judgeRubricFile: rubric.file,
            judgeRubricHash: rubric.hash,
            fields: schema.fields,
            // Every row WRITTEN, holes included (the server's `len(rows)`);
            // the per-condition `codings` below count only measurements.
            codings: records.count + noncompliantCodings,
            conditions: codingConditionAggregates(
                records: records, schema: schema),
            // Absent-with-reason rather than empty when one coder coded the
            // run: there is no pair to compare, which is not the same fact
            // as a pair that agreed about nothing.
            fieldAgreement: judges.count >= 2
                ? codingFieldAgreement(
                    records: records, schema: schema,
                    judges: judges.map(\.name))
                : nil,
            fieldAgreementAbsentReason: judges.count >= 2
                ? nil : ExperimentStore.singleCoderAgreementAbsentReason,
            evaluationSource: evaluationSource,
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            exclusions: exclusionStamp,
            // Nonzero-only: a clean coding report is unchanged.
            noncompliantCodings: noncompliantCodings > 0
                ? noncompliantCodings : nil,
            // Additive and LOUD: absent means the full corpus was coded, so
            // every report written before this existed reads back unchanged,
            // and a report that CARRIES the block cannot be mistaken for a
            // full-corpus coding by a reader who only looks at `codings`.
            sampling: sampling)
        try encoder.encode(report).write(
            to: runDirectory.appending(component: "coding-report.json"))
        if let sampling {
            print(
                "coded \(codedPhrase) at seed \(sampling.sampleSeed) — this "
                    + "report covers a SUBSAMPLE, not the full corpus")
        }
        print("coding evaluation artifacts: \(runDirectory.path)")
        return runDirectory
    }

    /// Per-condition, per-field aggregates across the whole judge panel
    /// (per-judge splits live in the rows; server twin:
    /// `response_coding.aggregate_conditions`). Booleans → trueShare over
    /// non-null codes; numbers → mean; strings/enums → counts. Nulls are
    /// reported, never imputed.
    ///
    /// Noncompliant rows (the coder refused/failed that record, kept for
    /// review) sit outside every aggregate — structurally here, since they
    /// are `NoncompliantCodingRecord`s and never enter this array. The
    /// server's twin filters `noncompliant`/`codes is None` explicitly
    /// because Python has one row type; the guarantee is the same.
    static func codingConditionAggregates(
        records: [CodingRecord], schema: ResponseCoding.Schema
    ) -> [String: CodingConditionReport] {
        Dictionary(grouping: records, by: \.condition).mapValues { rows in
            var fields: [String: CodingFieldAggregate] = [:]
            for field in schema.fields {
                let values = rows.map { $0.codes[field.name] ?? JSONValue.null }
                let coded = values.filter { value in
                    if case .null = value { return false }
                    return true
                }
                var aggregate = CodingFieldAggregate(
                    n: coded.count, nulls: values.count - coded.count)
                switch field.type {
                case "boolean":
                    let trues = coded.filter { value in
                        if case .bool(true) = value { return true }
                        return false
                    }.count
                    aggregate.trueCount = trues
                    aggregate.trueShare = coded.isEmpty
                        ? nil : Double(trues) / Double(coded.count)
                case "integer", "number":
                    let numbers = coded.compactMap { value -> Double? in
                        if case .number(let number) = value { return number }
                        return nil
                    }
                    aggregate.mean = numbers.isEmpty
                        ? nil : numbers.reduce(0, +) / Double(numbers.count)
                default:
                    var counts: [String: Int] = [:]
                    for value in coded {
                        counts[
                            ResponseCoding.categoricalLabel(value),
                            default: 0] += 1
                    }
                    aggregate.counts = counts
                }
                fields[field.name] = aggregate
            }
            let uniqueResponses = Set(
                rows.map { "\($0.promptID)\u{1}\($0.sampleIndex)" })
            return CodingConditionReport(
                codedResponses: uniqueResponses.count,
                codings: rows.count,
                meanWordCount: rows.isEmpty
                    ? nil
                    : Double(rows.map(\.wordCount).reduce(0, +))
                        / Double(rows.count),
                fields: fields)
        }
    }

    /// Per-FIELD inter-judge agreement, every judge pair, over the
    /// intersection of coded cells (server twin:
    /// `response_coding.field_agreement`). Categorical fields report
    /// percent agreement + Cohen's kappa — the statistic the K&Z paper
    /// used to validate its coders; numeric fields report mean absolute
    /// difference. Noncompliant rows never reach here (they are a distinct
    /// row type): agreement over a refused record would compare a judgment
    /// to a hole.
    static func codingFieldAgreement(
        records: [CodingRecord], schema: ResponseCoding.Schema,
        judges: [String]
    ) -> [CodingAgreementEntry] {
        var byJudge: [String: [String: [String: JSONValue]]] = [:]
        for record in records {
            let key = "\(record.condition)\u{1}\(record.sampleIndex)"
                + "\u{1}\(record.promptID)"
            byJudge[record.judgeName, default: [:]][key] = record.codes
        }
        var entries: [CodingAgreementEntry] = []
        for (index, judgeA) in judges.enumerated() {
            for judgeB in judges.dropFirst(index + 1) {
                let cellsA = byJudge[judgeA] ?? [:]
                let cellsB = byJudge[judgeB] ?? [:]
                let shared = Set(cellsA.keys)
                    .intersection(cellsB.keys).sorted()
                guard !shared.isEmpty else { continue }
                for field in schema.fields {
                    if field.type == "integer" || field.type == "number" {
                        var differences: [Double] = []
                        for key in shared {
                            guard
                                case .number(let a)? = cellsA[key]?[field.name],
                                case .number(let b)? = cellsB[key]?[field.name]
                            else { continue }
                            differences.append(abs(a - b))
                        }
                        guard !differences.isEmpty else { continue }
                        entries.append(
                            CodingAgreementEntry(
                                field: field.name, judgeA: judgeA,
                                judgeB: judgeB, n: differences.count,
                                meanAbsoluteDifference:
                                    differences.reduce(0, +)
                                    / Double(differences.count)))
                        continue
                    }
                    let labelsA = shared.map {
                        ResponseCoding.categoricalLabel(
                            cellsA[$0]?[field.name])
                    }
                    let labelsB = shared.map {
                        ResponseCoding.categoricalLabel(
                            cellsB[$0]?[field.name])
                    }
                    let kappa = StudyStatistics.cohensKappa(labelsA, labelsB)
                    let agreed = zip(labelsA, labelsB)
                        .filter { $0.0 == $0.1 }.count
                    var confusion: [String: [String: Int]] = [:]
                    for (a, b) in zip(labelsA, labelsB) {
                        confusion[a, default: [:]][b, default: 0] += 1
                    }
                    entries.append(
                        CodingAgreementEntry(
                            field: field.name, judgeA: judgeA,
                            judgeB: judgeB, n: shared.count,
                            percentAgreement: Double(agreed)
                                / Double(shared.count),
                            kappa: kappa.isNaN ? nil : kappa,
                            confusion: confusion))
                }
            }
        }
        return entries
    }

    /// The ONE human-validation parser (review 2026-08-01, P1: pin-time
    /// validation and evaluation must read the same rules). The
    /// cross-engine row semantics, shared verbatim with the server's
    /// `_load_human_validation`:
    /// - `sampleIndex` present = that exact pair cell; ABSENT = an explicit
    ///   wildcard matching every judged cell of its (condition, promptID) —
    ///   with an exact row beating the wildcard for its own cell.
    /// - duplicate keys REFUSE (the engines used to keep first here and
    ///   last there — same file, different numbers).
    /// - `sampleIndex`, when present, must be a non-negative integer (the
    ///   decoder enforces it — strings, bools, fractions all refuse).
    /// - an empty file refuses: a pinned subset with no rows is a mistake,
    ///   not a smaller study.
    static func parseHumanValidation(_ data: Data) throws -> [HumanValidationRow] {
        let decoder = JSONDecoder()
        var rows: [HumanValidationRow] = []
        var seen = Set<String>()
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard
                let row = try? decoder.decode(
                    HumanValidationRow.self, from: Data(trimmed.utf8)),
                ["baseline", "variant", "tie"].contains(row.outcome),
                !row.condition.isEmpty, !row.promptID.isEmpty
            else {
                throw ExperimentError(
                    reason: "malformed human validation JSONL at line \(index + 1) "
                        + "(need non-empty condition/promptID, outcome in "
                        + "baseline|variant|tie, and an optional non-negative "
                        + "integer sampleIndex)")
            }
            let key = [
                row.condition, row.promptID,
                row.sampleIndex.map(String.init) ?? "*",
            ].joined(separator: "\u{1}")
            guard seen.insert(key).inserted else {
                throw ExperimentError(
                    reason: "duplicate human validation row at line \(index + 1) "
                        + "for (\(row.condition), \(row.promptID), "
                        + "\(row.sampleIndex.map(String.init) ?? "wildcard")) — "
                        + "one human judgment per pair cell")
            }
            rows.append(row)
        }
        guard !rows.isEmpty else {
            throw ExperimentError(
                reason: "human validation file has no labeled rows")
        }
        return rows
    }

    /// Pairwise inter-judge agreement over the intersection of judged items
    /// (key: condition::sampleIndex::promptID — the pair-cell identity, the
    /// server's `_judgment_key` twin; label: conditionResult). Pure —
    /// unit-tested against hand-built records.
    static func judgeAgreement(
        records: [PairedJudgeRecord], judges: [String]
    ) -> [JudgeAgreementReport]? {
        guard judges.count >= 2 else { return nil }
        func labels(_ judge: String) -> [String: String] {
            records.filter { $0.judgeName == judge }.reduce(into: [:]) { table, record in
                table[
                    "\(record.condition)::\(record.sampleIndex)::\(record.promptID)"
                ] = record.conditionResult
            }
        }
        let labelsByJudge = Dictionary(
            uniqueKeysWithValues: judges.map { ($0, labels($0)) })
        var reports: [JudgeAgreementReport] = []
        for i in judges.indices {
            for j in judges.indices where j > i {
                let a = labelsByJudge[judges[i]] ?? [:]
                let b = labelsByJudge[judges[j]] ?? [:]
                let shared = Set(a.keys).intersection(b.keys).sorted()
                let labelsA = shared.compactMap { a[$0] }
                let labelsB = shared.compactMap { b[$0] }
                let kappa = StudyStatistics.cohensKappa(labelsA, labelsB)
                reports.append(
                    JudgeAgreementReport(
                        judgeA: judges[i],
                        judgeB: judges[j],
                        items: shared.count,
                        percentAgreement: shared.isEmpty
                            ? 0
                            : StudyStatistics.percentAgreement(labelsA, labelsB),
                        kappa: kappa.isNaN ? nil : kappa))
            }
        }
        return reports
    }

    /// Per-judge agreement against the pinned human subset. A human row
    /// with a sampleIndex matches its pair cell exactly; without one it
    /// matches every sample cell of its (condition, promptID).
    static func humanAgreement(
        records: [PairedJudgeRecord], judges: [String], human: [HumanValidationRow]
    ) -> [HumanAgreementReport] {
        judges.map { judge in
            var judgeLabels: [String] = []
            var humanLabels: [String] = []
            for record in records where record.judgeName == judge {
                // Exact-cell row first; the wildcard (absent sampleIndex)
                // only covers cells no exact row claims — the cross-engine
                // rule (2026-08-01), shared with the server's
                // `_materialize_human_validation`.
                let exact = human.first {
                    $0.condition == record.condition && $0.promptID == record.promptID
                        && $0.sampleIndex == record.sampleIndex
                }
                let match = exact ?? human.first {
                    $0.condition == record.condition && $0.promptID == record.promptID
                        && $0.sampleIndex == nil
                }
                guard let match else { continue }
                judgeLabels.append(record.conditionResult)
                humanLabels.append(match.conditionResult)
            }
            let kappa = StudyStatistics.cohensKappa(judgeLabels, humanLabels)
            return HumanAgreementReport(
                judge: judge,
                items: judgeLabels.count,
                percentAgreement: judgeLabels.isEmpty
                    ? 0
                    : StudyStatistics.percentAgreement(judgeLabels, humanLabels),
                kappa: kappa.isNaN ? nil : kappa)
        }
    }

    private static func structuredSummaries(_ rows: [PairedJudgeRecord]) -> [String: StructuredFieldSummary] {
        var valuesByField = rows.reduce(into: [String: [JSONValue]]()) { partial, row in
            for (key, value) in row.judgment.structuredFields ?? [:] {
                if case .null = value { continue }
                partial[key, default: []].append(value)
            }
        }
        addUnblindedPairSummaries(rows, into: &valuesByField)
        return valuesByField.mapValues { values in
            let numbers = values.compactMap { value -> Double? in
                if case .number(let number) = value { number } else { nil }
            }
            let bools = values.compactMap { value -> Bool? in
                if case .bool(let bool) = value { bool } else { nil }
            }
            let strings = values.compactMap { value -> String? in
                if case .string(let string) = value { string } else { nil }
            }
            let stringCounts = strings.reduce(into: [String: Int]()) { counts, value in
                counts[value, default: 0] += 1
            }
            return StructuredFieldSummary(
                count: values.count,
                numericMean: numbers.isEmpty ? nil : numbers.reduce(0, +) / Double(numbers.count),
                trueCount: bools.isEmpty ? nil : bools.filter { $0 }.count,
                falseCount: bools.isEmpty ? nil : bools.filter { !$0 }.count,
                stringCounts: stringCounts.isEmpty ? nil : stringCounts)
        }
    }

    private static func addUnblindedPairSummaries(
        _ rows: [PairedJudgeRecord],
        into valuesByField: inout [String: [JSONValue]]
    ) {
        for row in rows {
            guard let fields = row.judgment.structuredFields else { continue }
            for (key, aValue) in fields where key.hasPrefix("a_") {
                let baseName = String(key.dropFirst(2))
                guard let bValue = fields["b_\(baseName)"] else { continue }
                let baselineValue = row.baselineWas == "A" ? aValue : bValue
                let conditionValue = row.conditionWas == "A" ? aValue : bValue
                if case .number(let baselineNumber) = baselineValue,
                    case .number(let conditionNumber) = conditionValue
                {
                    valuesByField[
                        "condition_minus_baseline_\(baseName)",
                        default: []
                    ].append(.number(conditionNumber - baselineNumber))
                } else {
                    valuesByField[
                        "baseline_to_condition_\(baseName)",
                        default: []
                    ].append(
                        .string(
                            baselineValue.displayString + " -> "
                                + conditionValue.displayString))
                }
            }
        }
    }

    /// Each decoded generation rides with its RAW JSONL line so the
    /// exclusion rules can read record-level parsed endpoints (the
    /// present-with-null vs absent distinction survives only in the raw
    /// bytes — the analyze path's rule).
    private static func loadEvaluationGenerations(
        _ url: URL
    ) throws -> [(record: EvaluationGeneration, line: Data)] {
        /// Choice-instrument records carry no seed/output and are not
        /// judgeable text — skip them explicitly (the sampled records in the
        /// same run still decode strictly).
        struct InstrumentProbe: Decodable {
            let instrument: String?
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        return try text.split(separator: "\n").compactMap { line in
            let data = Data(line.utf8)
            if let probe = try? JSONDecoder().decode(InstrumentProbe.self, from: data),
                probe.instrument == "answerTokenLogprob"
            {
                return nil
            }
            return (
                record: try JSONDecoder().decode(EvaluationGeneration.self, from: data),
                line: data)
        }
    }

    private static func verifyEvaluationSourceRun(
        _ sourceRunDirectory: URL,
        manifest: ExperimentManifest,
        generations: [EvaluationGeneration]
    ) throws {
        let manifestURL = sourceRunDirectory.appending(component: "experiment.json")
        if let data = try? Data(contentsOf: manifestURL),
            let sourceManifest = try? JSONDecoder().decode(ExperimentManifest.self, from: data),
            sourceManifest.name != manifest.name
        {
            throw ExperimentError(
                reason: "paired judge source run belongs to '\(sourceManifest.name)', "
                    + "not selected study '\(manifest.name)'")
        }
        if let wrong = generations.first(where: { $0.experiment.map { $0 != manifest.name } ?? false }) {
            throw ExperimentError(
                reason: "paired judge source generation \(wrong.promptID) belongs to "
                    + "'\(wrong.experiment ?? "unknown")', not selected study '\(manifest.name)'")
        }
    }

    /// Cross-engine zero-pairs refusal (external review 2026-07-22, P0) —
    /// keep byte-identical with the server's
    /// `paired_judge.NO_PAIRS_MESSAGE`.
    static let noPairsMessage =
        "paired judging produced zero pairs: records join on (promptID, "
        + "sampleIndex) and every non-baseline record needs a baseline record "
        + "in the same cell — this source run has none. Likely causes: the "
        + "run has no baseline condition, or no sampled text records "
        + "(instrument readouts and error records are never judged). Refusing "
        + "to write an empty judged report."

    private static func evaluationKey(sampleIndex: UInt64, promptID: String) -> String {
        "\(sampleIndex)::\(promptID)"
    }

    static func shouldFlip(
        experiment: String, condition: String, seed: UInt64, promptID: String
    ) -> Bool {
        let data = Data("\(experiment)::\(condition)::\(seed)::\(promptID)".utf8)
        return Array(SHA256.hash(data: data)).first.map { $0 % 2 == 0 } ?? false
    }

    /// Layer × alpha dose-response on dev prompts. Per cell: concept
    /// expression (marker density), degeneration (distinct-bigram ratio),
    /// and the capability battery. Greedy decoding throughout, so cells are
    /// deterministic. The winning cell is chosen by the manifest's DECLARED
    /// selection criterion (`sweep.selection`, resolved by
    /// `SweepSelectionRule` — absent resolves to the historical rule:
    /// highest marker density with battery within 0.15 of baseline and
    /// distinct-2 ≥ 0.45) and written into a draft manifest as
    /// `<concept>-recommended` WITH full selection provenance (sweep run,
    /// resolved criterion, dev-split hash, metrics, control) — the block
    /// `promote` later copies into an agent's birth certificate.
    ///
    /// `shouldCancel` is polled between concepts, layers, alphas, dev-prompt
    /// generations, and battery items (server parity: `should_cancel` in
    /// `tasks.py`): the sweep stops after the current generation, keeps the
    /// grid rows computed so far in `sweep.csv`, and never selects a winner
    /// from an incomplete grid.
    @discardableResult
    public static func sweep(
        experimentName: String,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        log: (@Sendable (String) async -> Void)? = nil
    ) async throws -> SweepOutcome {
        // Progress hook (same shape as `validate`): every line still prints
        // for the CLI; a panel-initiated sweep additionally mirrors into the
        // app's display pane.
        @Sendable func emit(_ line: String) async {
            print(line)
            await log?(line)
        }
        // True (logged once, with where) when a cancellation request has
        // been observed — the sweep's own twin of the server's
        // `_observe_cancel`. @Sendable: it is also polled from inside
        // `batteryAccuracy`'s cancel hook.
        @Sendable func cancellationObserved(_ location: String) async -> Bool {
            guard let shouldCancel, await shouldCancel() else { return false }
            await emit("cancellation observed at \(location)")
            return true
        }
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        var manifest = try loadVerified(experimentName)
        let spec = manifest.sweep ?? ExperimentManifest.SweepSpec()
        // Resolve the criterion AND its objective's instrument config BEFORE
        // loading the model: an unknown metric, a missing/empty choice file,
        // missing judge pins, or a credential-less Claude judge fails at
        // sweep start, never mid-run.
        // WP0 step 7: the ~24 selection-rule refusals and the objective's
        // instrument refusals all arrive as bare `ExperimentError`s from the
        // shared pure rule, which has no way to know which verb is asking.
        // Tagging them HERE — at the one call site on the agent path — gives
        // every one of them a gate id and a runnable repair without touching
        // a single refusal string (they are the cross-engine contract).
        let criterion = try taggingSweepRefusals(
            .sweepSelectionRule, experiment: manifest.name
        ) { try SweepSelectionRule.resolve(spec.selection) }
        let objective = try taggingSweepRefusals(
            .sweepSelectionRule, experiment: manifest.name
        ) {
            try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: spec.selection, manifest: manifest)
        }
        // The declared-criterion advisory, at sweep START and before the
        // model loads (WP0 step 7, P2/P3): a sweep that silently defaults to
        // marker density on a choice task is choosing the one objective the
        // methods note forbids for decision studies.
        if let counts = choiceShapedItemCount(manifest),
            let advisory = SweepSelectionRule.defaultedSelectionAdvisory(
                spec: spec.selection, choiceItemCount: counts.choice,
                totalItemCount: counts.total)
        {
            await emit("⚠︎ " + advisory)
        }
        // Logged AT SWEEP START (cross-engine contract): a local judge with
        // no declared model judges with the study model.
        for judge in objective.judgePanel where judge.modelDefaulted {
            await emit(
                "judge '\(judge.name)': no model set — using the study model "
                    + manifest.modelID)
        }
        // Provider pins verified at SWEEP START too (2026-07-24): the sweep
        // is where a wrong pin costs the most, surfacing mid-grid after the
        // study model has loaded and cells have been generated.
        try await taggingSweepRefusals(
            .sweepJudgeCapacity, experiment: manifest.name
        ) { try await preflightOpenRouterJudges(objective.judgePanel) }
        let root = VectorCatalog.projectRoot
        let (devPrompts, devPromptsHash) = try StimulusSet.loadTexts(
            url: root.appending(path: spec.devPromptsFile))
        let batteryData = try Data(
            contentsOf: root.appending(path: spec.batteryFile))
        // Sweep-input pin enforcement at the moment of use (firewall
        // closure 2026-07-20, server twin: `_sweep_with_spec`): a pinned
        // dev-prompts / battery hash must match the bytes the sweep is
        // ABOUT to select on. The refusal is what keeps the ex-post
        // provenance stamp (selection.devPromptsHash) and the manifest pin
        // in agreement — a mismatch refuses, never silently overwrites
        // either.
        // Choice-instrument pins are enforced when the objective READS
        // them (review 2026-08-02, P1) — the resolved sets carry the live
        // hashes, so a drifted file refuses here exactly like dev/battery.
        var choicePinChecks: [String?] = []
        if objective.metric == "logprobShift" {
            for entry in ExperimentStore.sweepChoicePinEntries(spec) {
                let live: String
                if let concept = entry.concept {
                    live = try objective.choiceSet(for: concept).hash
                } else {
                    live = objective.choicePromptsHash ?? ""
                }
                choicePinChecks.append(sweepInputPinViolation(
                    label: entry.label, file: entry.file,
                    liveHash: live, pinned: entry.pinned))
            }
        }
        for problem in ([
            sweepInputPinViolation(
                label: "sweep dev prompts", file: spec.devPromptsFile,
                liveHash: devPromptsHash, pinned: spec.devPromptsHash),
            sweepInputPinViolation(
                label: "sweep capability battery", file: spec.batteryFile,
                liveHash: ExperimentStore.sha256Hex(batteryData),
                pinned: spec.batteryHash),
        ] + choicePinChecks).compactMap({ $0 }) {
            throw ExperimentError.refusing(
                .sweepInputDrift, problem,
                repair: "restore the named sweep input to its pinned bytes ; "
                    + "then steerlab-cli experiment sweep \(manifest.name)  "
                    + "(a frozen pin is never re-pinned — steerlab-cli "
                    + "experiment duplicate \(manifest.name) "
                    + "\(manifest.name)-v2 to change the input)")
        }
        let battery = try CapabilityBattery(url: root.appending(path: spec.batteryFile))
        // The sweep's capability constraint is armed by the battery's FORMAT,
        // not by the study's prompt config (server `_battery_accuracy`). A
        // legacy battery is still armed from the manifest — which is exactly
        // how a cell can win by writing in the study's format rather than by
        // keeping capability — so it says so out loud.
        let sweepBatteryArming = battery.resolveArming(
            promptMode: manifest.promptMode ?? .chatAssistant,
            systemPrompt: manifest.systemPrompt,
            qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false)
        if let advisory = battery.contaminationAdvisory(sweepBatteryArming) {
            await emit("WARNING: \(advisory)")
        }
        // C4: say what the declared tolerance can actually gate on, BEFORE
        // the grid runs. Battery accuracy moves in steps of 1/N, so a
        // tolerance between steps is not the tolerance that operates — and
        // a sweep that reports "tolerance 0.15" while gating at 0.2 is
        // reporting a number that did not decide anything.
        if let resolution = SweepSelectionRule.batteryResolution(
            itemCount: battery.items.count,
            capabilityTolerance: criterion.capabilityTolerance)
        {
            await emit(
                (resolution.isCoarse ? "⚠︎ " : "") + resolution.summary)
        }
        let container = try await loadContainer(pinning: &manifest)
        let runDirectory = try makeRunDirectory(experiment: manifest, task: "sweep")
        await emit("sweep run directory: \(runDirectory.lastPathComponent)")

        // Objective instruments arm BEFORE any grid work: the choice
        // baseline pass (and its option-length guard) and the judge panel
        // fail here, at sweep start — never mid-grid.
        // Per-concept instruments (choicePromptsFiles, 2026-08-02): each
        // concept's cells are scored on ITS OWN rows, so the baseline pass
        // runs once per DISTINCT file (keyed by content hash — the singular
        // declaration therefore keeps its single shared baseline pass).
        var choiceBaselineByHash: [String: [String: Double]] = [:]
        if objective.metric == "logprobShift" {
            for ref in manifest.concepts {
                let chosen = try objective.choiceSet(for: ref.name)
                guard choiceBaselineByHash[chosen.hash] == nil else { continue }
                await emit(
                    "choice baseline ('\(chosen.file)', \(chosen.rows.count) rows)…")
                choiceBaselineByHash[chosen.hash] = try await
                    sweepChoiceTargetLogprobs(
                        container, manifest: manifest, rows: chosen.rows,
                        injections: [])
            }
        }
        var judgePanel: [SweepJudge] = []
        if objective.metric == "judgeScore" {
            // Local judges generate through the sweep's already-loaded study
            // container — never a second load (non-study models were refused
            // at resolve, above).
            judgePanel = try taggingSweepRefusals(
                .sweepJudgeCapacity, experiment: manifest.name
            ) {
                try sweepJudgePanel(
                    objective: objective, studyModelID: manifest.modelID,
                    studyRevision: manifest.modelRevision,
                    studyDtype: manifest.dtype,
                    container: container)
            }
        }

        // Persist the sweep's re-derived vectors into the sweep run directory
        // (the same sidecar writer extract/validate use — never a parallel
        // one): the sweep run itself then carries recipe-matching artifacts,
        // so "sweep then promote" needs no separate extract run for
        // `AgentPromotion.matchingVectorArtifact` to find them.
        let extractions = try await extractAll(
            manifest: manifest, container: container, into: runDirectory)

        let extraMetric = objective.metric != "markerDensity"
        var csv = SweepRunCatalog.csvHeader
        csv += extraMetric ? ",objective\n" : "\n"
        var recommendations: [String: Any] = [:]

        var cancelled = false
        // (baseline texts, baseline battery accuracy) — generated once,
        // shared by every concept (see the baseline comment in the loop).
        var sharedBaseline: ([String], Float)?
        conceptLoop: for ref in manifest.concepts {
            if await cancellationObserved("concept=\(ref.name)") {
                cancelled = true
                break
            }
            guard let extraction = extractions[ref.name]?.result else { continue }
            let conceptDirectory = VectorCatalog.conceptsDirectory.appending(
                component: ref.name)
            let rubric = MarkerRubric(directory: conceptDirectory)
            if rubric == nil {
                await emit("⚠︎ \(ref.name): no markers.json — expression scores will be 0")
            }
            let layerCount = extraction.vectors.layerCount
            let layers = spec.resolvedLayers(layerCount: layerCount)

            // Baseline cell: no injection. Baseline texts are kept —
            // judgeScore pairs every cell against them. The GENERATIONS
            // are concept-independent (no injection, same dev prompts,
            // same battery) and are generated ONCE for the whole sweep
            // (review 2026-08-02, P2 — a multi-concept sweep regenerated
            // them per concept); only the marker DENSITY is per-concept,
            // rescored from the cached texts with this concept's rubric.
            if sharedBaseline == nil {
                var texts: [String] = []
                for (index, prompt) in devPrompts.enumerated() {
                    if await cancellationObserved(
                        "baseline dev \(index + 1)/\(devPrompts.count)")
                    {
                        cancelled = true
                        break conceptLoop
                    }
                    let text = try await generate(
                        container, prompt: prompt, modelID: manifest.modelID,
                        maxTokens: spec.maxTokens)
                    texts.append(text)
                    // Persisted AS GENERATED (server `_dev_texts` twin): the
                    // texts are the sweep's qualitative evidence, and a
                    // mid-grid kill must not reduce them to log previews.
                    try SweepRunCatalog.appendDevGeneration(
                        runDirectory: runDirectory, kind: "baseline",
                        concept: nil, layer: -1, alpha: 0,
                        promptIndex: index, text: text)
                    await emit(sweepDevPreviewLine(
                        label: "baseline", index: index + 1,
                        total: devPrompts.count, text: text))
                }
                guard
                    let accuracy = try await batteryAccuracy(
                        container, battery: battery, modelID: manifest.modelID,
                        arming: sweepBatteryArming,
                        shouldCancel: {
                            await cancellationObserved("baseline battery")
                        })
                else {
                    cancelled = true
                    break
                }
                sharedBaseline = (texts, accuracy)
            }
            guard let (baselineTexts, baselineAccuracy) = sharedBaseline else {
                break
            }
            let baselineDensity = mean(baselineTexts.map { rubric?.density(in: $0) ?? 0 })
            let baselineDistinct = mean(baselineTexts.map(distinctBigramRatio))
            // Mean output length, measured for the first time on this engine
            // (the server has always written it as `words`). It is the input
            // to the length-inflation flag: the degenerate cell that motivated
            // the baseline-relative floor ran 65% long, and a reader looking
            // at a logprobShift is owed that fact.
            let baselineWords = mean(baselineTexts.map { Float(wordCount($0)) })
            let baselineMetric = SweepSelectionRule.baselineMetric(
                objective.metric, baselineDensity: Double(baselineDensity))
            // The baseline is its own reference, so its ratio is 1 and it is
            // never length-inflated.
            csv += "\(ref.name),-1,0,\(baselineDensity),\(baselineDistinct),1.0,"
            csv += "\(baselineWords),false,\(baselineAccuracy)"
            csv += extraMetric ? ",\(baselineMetric)\n" : "\n"
            await emit(
                "\(ref.name) baseline: density \(baselineDensity), "
                    + "distinct2 \(baselineDistinct), battery \(baselineAccuracy)")

            var cells: [SweepSelectionRule.Cell] = []
            for layer in layers {
                if await cancellationObserved("concept=\(ref.name) layer=\(layer)") {
                    cancelled = true
                    break conceptLoop
                }
                // Same rule as the condition and variant paths (2026-08-28
                // audit, F7/F13): a layer the denominator table does not
                // reach refuses, where this site used to clamp to the last
                // entry and dose the deepest sweep cells with a shallower
                // layer's number.
                if let problem = ResidualNormConvention.residualNormProblem(
                    extraction.residualNormPerLayer, layer: layer,
                    artifact: ref.name)
                {
                    throw ExperimentError(reason: "concept '\(ref.name)': \(problem)")
                }
                let norm = extraction.residualNormPerLayer[layer]
                for alpha in spec.alphas {
                    let cellLabel = sweepCellLabel(layer: layer, alpha: alpha)
                    if await cancellationObserved(
                        "concept=\(ref.name) layer=\(layer) "
                            + "alpha=\(String(format: "%g", alpha))")
                    {
                        cancelled = true
                        break conceptLoop
                    }
                    let cell = [
                        CellInjection(
                            layer: layer,
                            vector: extraction.vectors.perLayer[layer],
                            alpha: try SteeringVectorMath.normUnitScale(
                                alpha: Float(alpha), residualNorm: norm,
                                vectorNorm: SteeringVectorMath.l2Norm(
                                    extraction.vectors.perLayer[layer])))
                    ]
                    var texts: [String] = []
                    for (index, prompt) in devPrompts.enumerated() {
                        if await cancellationObserved(
                            "\(cellLabel) dev \(index + 1)/\(devPrompts.count)")
                        {
                            cancelled = true
                            break conceptLoop
                        }
                        let text = try await generate(
                            container, prompt: prompt, modelID: manifest.modelID,
                            maxTokens: spec.maxTokens, injections: cell)
                        texts.append(text)
                        try SweepRunCatalog.appendDevGeneration(
                            runDirectory: runDirectory, kind: "cell",
                            concept: ref.name, layer: layer, alpha: alpha,
                            promptIndex: index, text: text)
                        await emit(sweepDevPreviewLine(
                            label: cellLabel, index: index + 1,
                            total: devPrompts.count, text: text))
                    }
                    let density = mean(texts.map { rubric?.density(in: $0) ?? 0 })
                    let distinct = mean(texts.map(distinctBigramRatio))
                    let words = mean(texts.map { Float(wordCount($0)) })
                    guard
                        let accuracy = try await batteryAccuracy(
                            container, battery: battery, modelID: manifest.modelID,
                            arming: sweepBatteryArming,
                            injections: cell,
                            shouldCancel: {
                                await cancellationObserved("\(cellLabel) battery")
                            })
                    else {
                        cancelled = true
                        break conceptLoop
                    }
                    // The declared objective's value for this cell — the dev
                    // texts are the SAME ones the constraints use.
                    let metricValue: Double
                    switch objective.metric {
                    case "logprobShift":
                        try await setInterventions(container, [])
                        let chosen = try objective.choiceSet(for: ref.name)
                        let cellLogprobs = try await sweepChoiceTargetLogprobs(
                            container, manifest: manifest, rows: chosen.rows,
                            injections: cell)
                        metricValue = meanLogprobShift(
                            cell: cellLogprobs,
                            baseline: choiceBaselineByHash[chosen.hash] ?? [:])
                    case "judgeScore":
                        metricValue = try await sweepJudgePreference(
                            experiment: manifest.name,
                            conditionTag: "sweep:\(ref.name):L\(layer):a\(alpha)",
                            prompts: devPrompts, cellTexts: texts,
                            baselineTexts: baselineTexts, judges: judgePanel)
                    default:
                        metricValue = Double(density)
                    }
                    // Reported for EVERY cell whichever coherence rule is in
                    // force: the number the baseline-relative floor gates on,
                    // and the length flag beside it.
                    let ratio = SweepSelectionRule.distinct2Ratio(
                        distinct2: Double(distinct),
                        baselineDistinct2: Double(baselineDistinct))
                    let inflated = SweepSelectionRule.lengthInflated(
                        meanWords: Double(words),
                        baselineMeanWords: Double(baselineWords))
                    let ratioField: String = ratio.map { "\($0)" } ?? ""
                    csv += "\(ref.name),\(layer),\(alpha),\(density),\(distinct),"
                    csv += ratioField
                    csv += ",\(words),\(inflated),\(accuracy)"
                    csv += extraMetric ? ",\(metricValue)\n" : "\n"
                    var line = "\(ref.name) L\(layer) α\(alpha): density \(density), "
                    line += "distinct2 \(distinct)"
                    if let ratio {
                        line += " (\(ratio)× baseline)"
                    }
                    line += ", battery \(accuracy)"
                    if inflated {
                        line += ", ⚠︎ output \(words) words vs baseline "
                            + "\(baselineWords)"
                    }
                    if extraMetric {
                        line += ", \(objective.metric) \(metricValue)"
                    }
                    await emit(line)
                    cells.append(
                        SweepSelectionRule.Cell(
                            layer: layer, alpha: alpha, metric: metricValue,
                            distinct2: Double(distinct),
                            batteryAccuracy: Double(accuracy),
                            words: Double(words)))
                }
            }
            try await setInterventions(container, [])

            let baseline = SweepSelectionRule.Baseline(
                metric: baselineMetric,
                distinct2: Double(baselineDistinct),
                batteryAccuracy: Double(baselineAccuracy))
            guard
                var best = SweepSelectionRule.select(
                    cells: cells, baseline: baseline, criterion: criterion)
            else {
                // WHICH of the two possible reasons (review round 9, finding
                // 6; server twin since 2026-07-26): gates that refused
                // everything, or an eligible grid that never beat the
                // baseline. The old sentence always claimed the first, which
                // sends a researcher to loosen a tolerance that was never
                // binding.
                recommendations[ref.name] = SweepSelectionRule.noSelectionReason(
                    cells: cells, baseline: baseline, criterion: criterion)
                continue
            }

            // Matched-norm random control, when declared: a deterministic
            // random direction norm-matched to the concept vector at the
            // candidate's layer, same alpha, same dev prompts. The control
            // evaluates the SAME declared objective as the grid. applyTo
            // "winner" (historical) controls only the argmax; "topK"
            // (2026-08-03) walks the top K promotable cells and promotes
            // the FIRST that beats its own control (server twin).
            var control: ExperimentManifest.SelectionProvenance.Control?
            var controlsEvaluated: [SweepSelectionRule.EvaluatedControl] = []
            if let margin = criterion.matchedNormRandomMargin {
                let candidates =
                    criterion.controlApplyTo == "topK"
                    ? SweepSelectionRule.rankedCandidates(
                        cells: cells, baseline: baseline, criterion: criterion,
                        k: criterion.controlTopK ?? 1)
                    : [best]
                var promoted: SweepSelectionRule.Cell?
                for candidate in candidates {
                    let conceptVector = extraction.vectors.perLayer[candidate.layer]
                    let vectorNorm = SteeringVectorMath.l2Norm(conceptVector)
                    // The control cell reads the denominator under the same
                    // rule as the grid cell it controls (2026-08-28 audit,
                    // F7/F13) — a clamp here would have matched the control's
                    // dose to a layer the concept cell never used.
                    if let problem = ResidualNormConvention.residualNormProblem(
                        extraction.residualNormPerLayer, layer: candidate.layer,
                        artifact: ref.name)
                    {
                        throw ExperimentError(reason: "concept '\(ref.name)': \(problem)")
                    }
                    let residual = extraction.residualNormPerLayer[candidate.layer]
                    let randomVector = try SweepSelectionRule.controlVector(
                        seedText: "\(ref.name)-recommended|\(ref.name)|\(candidate.layer)",
                        dimension: conceptVector.count, norm: vectorNorm)
                    let controlCell = [
                        CellInjection(
                            layer: candidate.layer, vector: randomVector,
                            alpha: try SteeringVectorMath.normUnitScale(
                                alpha: Float(candidate.alpha), residualNorm: residual,
                                vectorNorm: vectorNorm))
                    ]
                    let controlMetric: Double
                    if objective.metric == "logprobShift" {
                        let chosen = try objective.choiceSet(for: ref.name)
                        let controlLogprobs = try await sweepChoiceTargetLogprobs(
                            container, manifest: manifest, rows: chosen.rows,
                            injections: controlCell)
                        controlMetric = meanLogprobShift(
                            cell: controlLogprobs,
                            baseline: choiceBaselineByHash[chosen.hash] ?? [:])
                    } else {
                        let controlLabel =
                            "control " + sweepCellLabel(
                                layer: candidate.layer, alpha: candidate.alpha)
                        var controlTexts: [String] = []
                        for (index, prompt) in devPrompts.enumerated() {
                            // A concept whose control pass was cut short gets
                            // no recommendation — the control gate never ran.
                            if await cancellationObserved(
                                "\(controlLabel) dev \(index + 1)/\(devPrompts.count)")
                            {
                                cancelled = true
                                break conceptLoop
                            }
                            let text = try await generate(
                                container, prompt: prompt, modelID: manifest.modelID,
                                maxTokens: spec.maxTokens, injections: controlCell)
                            controlTexts.append(text)
                            try SweepRunCatalog.appendDevGeneration(
                                runDirectory: runDirectory, kind: "control",
                                concept: ref.name, layer: candidate.layer,
                                alpha: candidate.alpha,
                                promptIndex: index, text: text)
                            await emit(sweepDevPreviewLine(
                                label: controlLabel, index: index + 1,
                                total: devPrompts.count, text: text))
                        }
                        try await setInterventions(container, [])
                        if objective.metric == "judgeScore" {
                            controlMetric = try await sweepJudgePreference(
                                experiment: manifest.name,
                                conditionTag:
                                    "sweep-control:\(ref.name):L\(candidate.layer):a\(candidate.alpha)",
                                prompts: devPrompts, cellTexts: controlTexts,
                                baselineTexts: baselineTexts, judges: judgePanel)
                        } else {
                            controlMetric = Double(
                                mean(controlTexts.map { rubric?.density(in: $0) ?? 0 }))
                        }
                    }
                    let passed = SweepSelectionRule.controlPasses(
                        bestMetric: candidate.metric,
                        controlMetric: controlMetric, margin: margin)
                    controlsEvaluated.append(
                        .init(
                            layer: candidate.layer, alpha: candidate.alpha,
                            metricValue: candidate.metric,
                            controlMetricValue: controlMetric, passed: passed))
                    await emit(
                        "\(ref.name) control L\(candidate.layer) "
                            + "α\(candidate.alpha): cell \(candidate.metric) vs "
                            + "control \(controlMetric) → "
                            + (passed ? "passes" : "fails"))
                    if passed {
                        promoted = candidate
                        control = .init(
                            type: "randomMatchedNorm", metricValue: controlMetric,
                            margin: margin,
                            // Recipe stamp (cross-engine contract string); an
                            // unstamped control block is a legacy cube-uniform
                            // run.
                            randomVectorAlgorithm:
                                SteeringVectorMath.randomVectorAlgorithm)
                        break
                    }
                }
                guard let promoted else {
                    let message =
                        criterion.controlApplyTo == "topK"
                        ? SweepSelectionRule.topKControlFailureMessage(
                            controlsEvaluated, margin: margin)
                        : SweepSelectionRule.controlFailureMessage(
                            bestMetric: best.metric,
                            controlMetric: controlsEvaluated.first?
                                .controlMetricValue ?? 0,
                            margin: margin)
                    recommendations[ref.name] = message
                    await emit("\(ref.name): \(message)")
                    continue
                }
                // The PROMOTED cell may not be the argmax under topK — every
                // downstream stamp describes the cell that actually passed
                // its control.
                best = promoted
            }

            let provenance = ExperimentManifest.SelectionProvenance(
                sweepRun: runDirectory.lastPathComponent,
                // Per-concept instruments: the provenance block pins the
                // choice file THIS concept's cells were scored on.
                criterion: criterion.asCriterion(
                    objective: objective, concept: ref.name),
                devPromptsHash: devPromptsHash,
                // The length the coherence floor was measured at (c18):
                // stamped so a reader can compare it against the study's
                // maxTokens instead of trusting a since-edited sweep spec.
                devMaxTokens: spec.maxTokens,
                winningCell: .init(layer: best.layer, alpha: best.alpha),
                metrics: sweepMetricsBlock(
                    metric: objective.metric,
                    best: best,
                    baselineMetric: baselineMetric,
                    baselineDensity: Double(baselineDensity),
                    baselineDistinct2: Double(baselineDistinct),
                    baselineWords: Double(baselineWords),
                    baselineAccuracy: Double(baselineAccuracy)),
                control: control)
            recommendations[ref.name] = try jsonObject(of: provenance)
            if manifest.status == .draft {
                let name = "\(ref.name)-recommended"
                manifest.conditions.removeAll { $0.name == name }
                manifest.conditions.append(
                    .init(
                        name: name,
                        slots: [
                            .init(concept: ref.name, layer: best.layer, alpha: best.alpha)
                        ],
                        bandWidth: 1, alphaInNormUnits: true,
                        selection: provenance))
            }
        }

        if cancelled {
            // A break can leave the last cell's injectors armed on the
            // container — disarm before anything else touches it. Rows
            // computed so far stay (server parity: partial sweep.csv +
            // recommendations for concepts that completed).
            try await setInterventions(container, [])
            await emit(
                "sweep cancelled early — partial grid rows kept; "
                    + "no selection from an incomplete grid")
        }
        try csv.write(
            to: runDirectory.appending(component: "sweep.csv"),
            atomically: true, encoding: .utf8)
        let recommendationData = try JSONSerialization.data(
            withJSONObject: recommendations, options: [.prettyPrinted, .sortedKeys])
        try recommendationData.write(
            to: runDirectory.appending(component: "recommendations.json"))

        let recommendationsOnly = manifest.status != .draft
        if manifest.status == .draft {
            try ExperimentStore.save(manifest)
            await emit("recommended conditions written into draft manifest")
        } else {
            await emit("manifest is \(manifest.status.rawValue) — recommendations reported only")
        }
        await emit("run artifacts: \(runDirectory.path)")
        // WP0 step 7 (P2): what the sweep DECIDED, returned rather than left
        // on disk for the caller to go find. `sweep` returned Void, so its
        // `--json` envelope could say only "swept 'x'" — no run directory, no
        // winning cell, no criterion, no metrics, and the
        // recommendations-only caveat as stderr prose. The typed outcome is
        // read back through the SHARED recommendations parser, so the
        // envelope can never describe a different selection than the one
        // `recommendations.json` records.
        let parsed =
            (try? SweepRunCatalog.parseRecommendations(recommendationData)) ?? [:]
        return SweepOutcome(
            runDirectory: runDirectory,
            experiment: manifest.name,
            manifestStatus: manifest.status.rawValue,
            recommendationsOnly: recommendationsOnly,
            cancelled: cancelled,
            criterion: criterion.metric,
            devPromptsHash: devPromptsHash,
            recommendations: parsed.keys.sorted().map { concept in
                switch parsed[concept] {
                case .selected(let provenance):
                    return SweepOutcome.ConceptRecommendation(
                        concept: concept,
                        layer: provenance.winningCell.layer,
                        alpha: provenance.winningCell.alpha,
                        criterion: provenance.criterion.objective?.metric
                            ?? criterion.metric,
                        metrics: provenance.metrics,
                        control: provenance.control.map(\.type))
                case .failure(let message):
                    return SweepOutcome.ConceptRecommendation(
                        concept: concept, failure: message)
                case nil:
                    return SweepOutcome.ConceptRecommendation(concept: concept)
                }
            })
    }

    /// What one sweep decided (WP0 step 7, punch list #1 P2).
    ///
    /// Every field is already true of the run directory on disk; the point is
    /// that a headless caller no longer has to open `recommendations.json` and
    /// re-implement its shape to learn the winning cell. The frozen-manifest
    /// caveat rides here as DATA (`recommendationsOnly`) rather than as a line
    /// on stderr — an agent that misses it promotes from a manifest that
    /// carries no `<concept>-recommended` condition and cannot say why.
    public struct SweepOutcome: Sendable {

        public struct ConceptRecommendation: Sendable {
            public var concept: String
            /// nil when the sweep selected NO cell for this concept.
            public var layer: Int?
            public var alpha: Double?
            /// The objective metric that decided it.
            public var criterion: String?
            /// The winning cell's metric readouts, as the provenance stamps
            /// them.
            public var metrics: [String: Double]
            /// The control recipe the winner had to beat, when one was
            /// declared.
            public var control: String?
            /// Why no cell was selected — the sweep's own message, which the
            /// run's `recommendations.json` records verbatim and `promote`
            /// stamps as `selectionOutcome`.
            public var failure: String?

            public init(
                concept: String, layer: Int? = nil, alpha: Double? = nil,
                criterion: String? = nil, metrics: [String: Double] = [:],
                control: String? = nil, failure: String? = nil
            ) {
                self.concept = concept
                self.layer = layer
                self.alpha = alpha
                self.criterion = criterion
                self.metrics = metrics
                self.control = control
                self.failure = failure
            }

            public var selected: Bool { layer != nil && alpha != nil }
        }

        public var runDirectory: URL
        public var experiment: String
        public var manifestStatus: String
        /// True when the manifest was not a draft, so the recommendations were
        /// reported into the run directory ONLY: no `<concept>-recommended`
        /// condition was written, and `promote` will read the run entry.
        public var recommendationsOnly: Bool
        /// True when the grid stopped early; no selection is made from an
        /// incomplete grid.
        public var cancelled: Bool
        /// The RESOLVED objective metric — what actually decided, which is not
        /// the same as what was declared when nothing was declared.
        public var criterion: String
        public var devPromptsHash: String
        public var recommendations: [ConceptRecommendation]

        public init(
            runDirectory: URL, experiment: String, manifestStatus: String,
            recommendationsOnly: Bool, cancelled: Bool, criterion: String,
            devPromptsHash: String,
            recommendations: [ConceptRecommendation]
        ) {
            self.runDirectory = runDirectory
            self.experiment = experiment
            self.manifestStatus = manifestStatus
            self.recommendationsOnly = recommendationsOnly
            self.cancelled = cancelled
            self.criterion = criterion
            self.devPromptsHash = devPromptsHash
            self.recommendations = recommendations
        }
    }

    private static func mean(_ values: [Float]) -> Float {
        values.isEmpty ? 0 : values.reduce(0, +) / Float(values.count)
    }

    /// One live-log line per dev-prompt generation during a sweep — the
    /// cross-engine format (the server's `_dev_texts` emits the identical
    /// shape): `<label> dev <i>/<n>: "<preview>"`, where the label is
    /// `baseline`, `L<layer> α<alpha>`, or `control L<layer> α<alpha>`.
    /// Battery generations get no previews (volume would drown the dev
    /// previews that matter).
    static func sweepDevPreviewLine(
        label: String, index: Int, total: Int, text: String
    ) -> String {
        "\(label) dev \(index)/\(total): \"\(generationPreview(text))\""
    }

    /// The grid cell's log label, alpha in `%g` (server parity: `α{alpha:g}`).
    static func sweepCellLabel(layer: Int, alpha: Double) -> String {
        "L\(layer) α\(String(format: "%g", alpha))"
    }

    /// The sweep-start refusal for a pinned-but-drifted sweep input (pure,
    /// unit-testable; server twin: the pin check in `_sweep_with_spec` —
    /// keep the message identical). nil = unpinned (legacy) or matching.
    /// Attach a gate id and a runnable repair to a refusal thrown by the
    /// SHARED sweep rules (WP0 step 7).
    ///
    /// The rules are pure and cross-engine — their strings are the contract —
    /// so they cannot name a CLI command themselves. This is the seam where
    /// the verb that asked is known. An error that already carries a typed
    /// refusal passes through untouched, so a nested freeze/lifecycle refusal
    /// keeps its own, more specific, id.
    static func taggingSweepRefusals<T>(
        _ gate: LifecycleGate, experiment: String,
        _ body: () async throws -> T
    ) async rethrows -> T {
        do {
            return try await body()
        } catch let error as ExperimentError
            where error.freezeRefusal == nil && error.lifecycleRefusal == nil
        {
            throw ExperimentError.refusing(
                gate, error.reason,
                repair: sweepRefusalRepair(gate, experiment: experiment))
        }
    }

    /// The synchronous twin — `resolve` and `sweepJudgePanel` do not await.
    static func taggingSweepRefusals<T>(
        _ gate: LifecycleGate, experiment: String, _ body: () throws -> T
    ) rethrows -> T {
        do {
            return try body()
        } catch let error as ExperimentError
            where error.freezeRefusal == nil && error.lifecycleRefusal == nil
        {
            throw ExperimentError.refusing(
                gate, error.reason,
                repair: sweepRefusalRepair(gate, experiment: experiment))
        }
    }

    static func sweepRefusalRepair(
        _ gate: LifecycleGate, experiment: String
    ) -> String {
        switch gate {
        case .sweepJudgeCapacity:
            return "steerlab-cli experiment pin-rubric \(experiment) "
                + "<prompts/rubrics/file.md> --judges <name>:local  — a LOCAL "
                + "judge with no model resolves to the study model at its "
                + "pinned revision, which is the only judge a local sweep can "
                + "run (a second resident model is impossible here)"
        default:
            return "steerlab-cli experiment set-sweep-selection \(experiment) "
                + "--objective markerDensity|judgeScore|logprobShift "
                + "[--choice-prompts <file>] [--capability-tolerance X] "
                + "[--coherence-floor Y] [--control-margin M]"
        }
    }

    static func sweepInputPinViolation(
        label: String, file: String, liveHash: String, pinned: String?
    ) -> String? {
        guard let pinned, pinned != liveHash else { return nil }
        return "\(label) '\(file)' do not match the manifest's pinned hash "
            + "(have \(liveHash.prefix(12))…, pinned \(pinned.prefix(12))…) "
            + "— the sweep would select on data the study did not pin; "
            + "restore the pinned file, or duplicate the study and "
            + "re-declare the sweep"
    }

    /// First ~`limit` characters of a generation for a single live-log line:
    /// whitespace runs (newlines included) collapse to single spaces, one
    /// ellipsis when truncated — the Swift twin of the server's
    /// `_preview_line`.
    static func generationPreview(_ text: String, limit: Int = 160) -> String {
        let collapsed = text.split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        guard collapsed.count > limit else { return collapsed }
        let cut = String(collapsed.prefix(limit))
        return cut.trimmingCharacters(in: .whitespaces) + "…"
    }

    /// Codable → JSON object, for embedding typed provenance inside the
    /// `[String: Any]` recommendations payload.
    private static func jsonObject(of value: some Encodable) throws -> Any {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try JSONSerialization.jsonObject(with: encoder.encode(value))
    }

    private static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    static func metricsCSV(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = []
    ) -> String {
        var header = ["condition", "seed", "promptIndex", "promptID", "wordCount", "distinct2"]
        header.append(contentsOf: concepts.map { "\($0)MarkerDensity" })
        // Reasoning-style columns in declared taxonomy order (cross-engine
        // contract: one `rs_<featureID>` column per feature).
        header.append(contentsOf: styleFeatureIDs.map { "rs_\($0)" })
        // Factorial cell columns (`factor_<name>`, sorted union, appended
        // last — cross-engine contract): present only when at least one
        // row's item declared factors, so factor-less runs keep their
        // exact header and bytes.
        let factorNames = Set(rows.flatMap(\.factors.keys)).sorted()
        header.append(contentsOf: factorNames.map { "factor_\($0)" })
        var lines = [header.joined(separator: ",")]
        for row in rows {
            var cells = [
                csvEscape(row.condition),
                String(row.seed),
                String(row.promptIndex),
                csvEscape(row.promptID),
                String(row.wordCount),
                String(row.distinct2),
            ]
            cells.append(contentsOf: concepts.map { String(row.markerDensity[$0] ?? 0) })
            cells.append(contentsOf: styleFeatureIDs.map { String(row.reasoningStyle[$0] ?? 0) })
            cells.append(contentsOf: factorNames.map { csvEscape(row.factors[$0] ?? "") })
            lines.append(cells.joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Per-condition `reasoningStyle` report block: the mean of each
    /// feature's per-generation values over this condition's rows. nil when
    /// no taxonomy is pinned (key omitted — legacy reports unchanged).
    static func reasoningStyleReport(
        rows: [MetricRow], style: PinnedReasoningStyle?
    ) -> ReasoningStyleConditionReport? {
        guard let style, !rows.isEmpty else { return nil }
        let features = Dictionary(
            uniqueKeysWithValues: style.taxonomy.featureIDs.map { id in
                (
                    id,
                    ReasoningStyleFeatureStat(
                        mean: rows.map { $0.reasoningStyle[id] ?? 0 }.reduce(0, +)
                            / Double(rows.count),
                        n: rows.count)
                )
            })
        return ReasoningStyleConditionReport(
            taxonomy: style.taxonomy.name, taxonomyHash: style.hash,
            taxonomyFile: style.path, diagnosticOnly: true,
            features: features)
    }

    static func report(
        experiment: ExperimentManifest, experimentHash: String,
        taskPrompts: (file: String, hash: String, prompts: [StudyPrompt]),
        rows: [MetricRow], conditionCount: Int, concepts: [String],
        batterySummaries: [String: CapabilityBatterySummary] = [:],
        choiceReadouts: [ReportChoiceReadout] = [],
        style: PinnedReasoningStyle? = nil,
        numericParser: ParserRegistry.NumericParserProvenance? = nil,
        exclusions: ExclusionEngine.Outcome? = nil
    ) -> StudyRunReport {
        struct ChoiceKey: Hashable {
            let promptID: String
            let sampleIndex: UInt64?
            let source: String
        }

        let rowGroups = Dictionary(grouping: rows, by: \.condition)
        let choiceGroups = Dictionary(grouping: choiceReadouts, by: \.condition)
        let conditionNames = Set(rowGroups.keys)
            .union(choiceGroups.keys)
            .union(batterySummaries.keys)
        // Last-wins merge, NEVER `uniqueKeysWithValues:`: duplicate item ids
        // are refused at prompt load, but report assembly runs after all
        // generation compute is spent and must not trap on a bypassed
        // duplicate key — last-wins matches the server's dict semantics
        // (`_choice_readouts` keys through a plain dict).
        let baselineChoices = Dictionary(
            (choiceGroups["baseline"] ?? []).map {
                (
                    ChoiceKey(
                        promptID: $0.promptID,
                        sampleIndex: $0.sampleIndex,
                        source: $0.source),
                    $0.selected
                )
            },
            uniquingKeysWith: { _, new in new })

        var conditionReports: [String: ConditionReport] = [:]
        for condition in conditionNames {
            let conditionRows = rowGroups[condition] ?? []
            let conditionChoices = choiceGroups[condition] ?? []
            let markerMeans = Dictionary(
                uniqueKeysWithValues: concepts.map { concept in
                    (concept, mean(conditionRows.map { $0.markerDensity[concept] ?? 0 }))
                })
            let instrumentReadouts = conditionChoices.filter {
                $0.source == "instrument"
            }.count
            let targetReadouts = conditionChoices.filter {
                $0.source == "parsed" && $0.target != nil
            }
            let choiceRate: Double? =
                targetReadouts.isEmpty
                ? nil
                : Double(targetReadouts.filter { $0.selected == $0.target }.count)
                    / Double(targetReadouts.count)
            // Ordinal-scale summary (contract keys ordinalMean/ordinalSD;
            // population SD — defined for a single readout). Server twin:
            // `_write_report` in steerlab_server/experiment/tasks.py.
            let ordinalPositions = conditionChoices.compactMap {
                $0.source == "instrument" ? $0.ordinalPosition : nil
            }
            var ordinalMean: Double?
            var ordinalSD: Double?
            if !ordinalPositions.isEmpty {
                let n = Double(ordinalPositions.count)
                let m = ordinalPositions.reduce(0, +) / n
                ordinalMean = m
                ordinalSD =
                    (ordinalPositions.reduce(0) { $0 + ($1 - m) * ($1 - m) } / n)
                    .squareRoot()
            }

            var agreement: ChoiceAgreementSummary?
            if condition != "baseline", !baselineChoices.isEmpty {
                // Same defensive last-wins merge as `baselineChoices` above.
                let mine = Dictionary(
                    conditionChoices.map {
                        (
                            ChoiceKey(
                                promptID: $0.promptID,
                                sampleIndex: $0.sampleIndex,
                                source: $0.source),
                            $0.selected
                        )
                    },
                    uniquingKeysWith: { _, new in new })
                let shared = Set(mine.keys).intersection(baselineChoices.keys)
                if !shared.isEmpty {
                    let agreed = shared.filter { mine[$0] == baselineChoices[$0] }.count
                    agreement = ChoiceAgreementSummary(
                        n: shared.count,
                        agreement: Double(agreed) / Double(shared.count))
                }
            }

            conditionReports[condition] = ConditionReport(
                generations: conditionRows.count,
                meanWordCount: mean(conditionRows.map { Float($0.wordCount) }),
                meanDistinct2: mean(conditionRows.map(\.distinct2)),
                meanMarkerDensity: markerMeans,
                choiceReadouts: instrumentReadouts > 0 ? instrumentReadouts : nil,
                choiceRate: choiceRate,
                agreementWithBaseline: agreement,
                ordinalMean: ordinalMean,
                ordinalSD: ordinalSD,
                capabilityBattery: batterySummaries[condition],
                reasoningStyle: reasoningStyleReport(rows: conditionRows, style: style))
        }
        return StudyRunReport(
            experiment: experiment.name,
            experimentHash: experimentHash,
            taskPromptsFile: taskPrompts.file,
            taskPromptsHash: taskPrompts.hash,
            promptMode: (experiment.promptMode ?? .chatAssistant).rawValue,
            systemPrompt: experiment.systemPrompt,
            qwenThinkingEnabled: experiment.qwenThinkingEnabled ?? false,
            promptCount: taskPrompts.prompts.count,
            conditionCount: conditionCount,
            seedCount: experiment.seeds.count,
            conditions: conditionReports,
            // Declared exclusions apply to the PAIRED statistics: excluded
            // rows drop here (their baseline partners drop with them —
            // pairwise deletion via the (seed, promptID) join), and so do
            // excluded instrument readouts (scope allRecordTypes — the
            // ordinal pairing by promptID loses its baseline partner the
            // same way), while the per-condition descriptive blocks above
            // keep the raw N — the exclusions stamp carries consideredN vs
            // survivingN.
            effectSizes: effectSizes(
                rows: exclusions.map { outcome in
                    rows.filter {
                        !outcome.excludedKeys.contains(
                            ExclusionEngine.rowKey(
                                condition: $0.condition, seed: $0.seed,
                                promptID: $0.promptID))
                    }
                } ?? rows,
                concepts: concepts,
                styleFeatureIDs: style?.taxonomy.featureIDs ?? [],
                choiceReadouts: exclusions.map { outcome in
                    choiceReadouts.filter { readout in
                        readout.source != "instrument"
                            || !outcome.excludedInstrumentKeys.contains(
                                ExclusionEngine.instrumentKey(
                                    condition: readout.condition,
                                    promptID: readout.promptID))
                    }
                } ?? choiceReadouts,
                phase: experiment.phase),
            numericParser: numericParser,
            exclusions: exclusions?.stamp)
    }

    // MARK: - Paired effect sizes (StudyStatistics wiring)

    /// Paired-to-baseline effect sizes over the run's per-item metrics: for
    /// each non-baseline condition × numeric metric (wordCount, distinct2,
    /// each concept's marker density), the per-item differences against the
    /// SAME (seed, promptID) baseline row feed a percentile bootstrap CI and
    /// a Wilcoxon signed-rank test. When the run carried the ordinalScale
    /// instrument, its per-item ladder positions join as one more numeric
    /// metric ("ordinalPosition" — cross-engine endpoint name) through the
    /// IDENTICAL machinery. Engine-pure and deterministic (fixed bootstrap
    /// seed) — unit-tested on fixture rows, no model involved. Every entry
    /// then passes through the phase's multiple-comparison correction
    /// (`applyCorrection` — the server-analyze twin), so both Swift emission
    /// paths (run-inline report + the analyze verb) carry corrected p-values.
    static func effectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        applyCorrection(
            sampledEffectSizes(
                rows: rows, concepts: concepts,
                styleFeatureIDs: styleFeatureIDs, replicates: replicates)
                + ordinalEffectSizes(choiceReadouts, replicates: replicates),
            phase: phase)
    }

    /// The correction method for a funnel phase — the server's exact rule
    /// (`tasks.py` analyze: `"holm" if manifest.phase == "confirm" else
    /// "bh"`): Holm step-down for the pre-registered confirm family, BH-FDR
    /// for screens and every other/absent phase.
    static func correctionMethod(phase: String?) -> String {
        phase == "confirm" ? "holm" : "bh"
    }

    /// The phase's multiple-comparison correction over effect rows,
    /// mirroring the server's analyze verb exactly: rows are grouped into
    /// one correction family PER METRIC (the server groups per endpoint —
    /// the correction runs across conditions within an endpoint, never
    /// across endpoints), the raw Wilcoxon p feeds the correction, rows
    /// with an undefined p are skipped by the adjustment but still stamped
    /// with the method (server `apply_correction` semantics), and entry
    /// order is preserved.
    static func applyCorrection(
        _ entries: [EffectSizeEntry], phase: String?
    ) -> [EffectSizeEntry] {
        let method = correctionMethod(phase: phase)
        var result = entries
        var families: [String: [Int]] = [:]
        for (index, entry) in entries.enumerated() {
            families[entry.metric, default: []].append(index)
        }
        for indices in families.values {
            let usable = indices.filter { result[$0].wilcoxonP != nil }
            let dense = usable.compactMap { result[$0].wilcoxonP }
            let adjusted =
                method == "holm"
                ? StudyStatistics.holm(dense) : StudyStatistics.bhFDR(dense)
            for (offset, index) in usable.enumerated() {
                result[index].adjustedP = adjusted[offset]
            }
            for index in indices {
                result[index].correction = method
            }
        }
        return result
    }

    /// When `stratum` is set the rows have already been restricted to one
    /// stratum's items; every produced entry carries the stratification
    /// provenance, and `unit` says what one paired difference is: "item"
    /// when each joined item contributes exactly one pair, "sample" when
    /// the pairs resolve within items (multiple seeds of the same item).
    private static func sampledEffectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String],
        replicates: Int, stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        var baselineByKey: [String: MetricRow] = [:]
        for row in rows where row.condition == "baseline" {
            baselineByKey["\(row.seed)::\(row.promptID)"] = row
        }
        guard !baselineByKey.isEmpty else { return [] }

        var metrics: [(name: String, value: (MetricRow) -> Double)] = [
            ("wordCount", { Double($0.wordCount) }),
            ("distinct2", { Double($0.distinct2) }),
        ]
        for concept in concepts.sorted() {
            metrics.append(
                ("\(concept)MarkerDensity", { Double($0.markerDensity[concept] ?? 0) }))
        }
        // Reasoning-style features join the same paired machinery, one
        // numeric metric per feature (declared taxonomy order).
        for id in styleFeatureIDs {
            metrics.append(("rs_\(id)", { $0.reasoningStyle[id] ?? 0 }))
        }

        // Conditions in first-appearance order; items in a deterministic
        // (seed, promptIndex, promptID) order so the bootstrap draws are
        // reproducible for a given run.
        var conditionOrder: [String] = []
        var seen = Set<String>()
        for row in rows where row.condition != "baseline" {
            if seen.insert(row.condition).inserted { conditionOrder.append(row.condition) }
        }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let conditionRows = rows
                .filter { $0.condition == condition }
                .sorted {
                    ($0.seed, $0.promptIndex, $0.promptID)
                        < ($1.seed, $1.promptIndex, $1.promptID)
                }
            for metric in metrics {
                var diffs: [Double] = []
                var pairedItems = Set<String>()
                for row in conditionRows {
                    guard let base = baselineByKey["\(row.seed)::\(row.promptID)"] else {
                        continue
                    }
                    diffs.append(metric.value(row) - metric.value(base))
                    pairedItems.insert(row.promptID)
                }
                guard !diffs.isEmpty else { continue }
                let ci = StudyStatistics.pairedBootstrapCI(
                    diffs, replicates: replicates, seed: 0)
                let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
                entries.append(
                    EffectSizeEntry(
                        condition: condition,
                        metric: metric.name,
                        n: ci.n,
                        meanDiff: ci.mean,
                        ciLower: ci.ciLower,
                        ciUpper: ci.ciUpper,
                        wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                        wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                        stratifyBy: stratum?.family,
                        stratum: stratum?.label,
                        unit: stratum.map {
                            _ in diffs.count == pairedItems.count
                                ? "item" : "sample"
                        },
                        // The unit IS the estimand: one pair per item is the
                        // pooled estimand restricted to this stratum and
                        // belongs in the correction family; several draws of
                        // the same item are a within-item variability read
                        // and are reported as a diagnostic instead.
                        estimand: stratum.map {
                            _ in diffs.count == pairedItems.count
                                ? EffectSizeEstimand.itemLevel
                                : EffectSizeEstimand.withinItemSamples
                        },
                        inference: stratum.map {
                            _ in diffs.count == pairedItems.count
                                ? EffectSizeInference.corrected
                                : EffectSizeInference.diagnostic
                        }))
            }
        }
        return entries
    }

    /// The ordinalScale instrument's paired effects: per-item ladder-position
    /// differences against the SAME-item baseline instrument readout (one
    /// deterministic readout per condition × prompt, so pairing is by
    /// promptID), through the same bootstrap CI + Wilcoxon as every other
    /// metric — no new statistics. The metric name "ordinalPosition" is the
    /// pinned cross-engine contract (server `_endpoint_values` twin).
    private static func ordinalEffectSizes(
        _ readouts: [ReportChoiceReadout], replicates: Int,
        stratum: (family: String, label: String)? = nil
    ) -> [EffectSizeEntry] {
        let ordinal = readouts.filter {
            $0.source == "instrument" && $0.ordinalPosition != nil
        }
        // Defensive last-wins on a duplicated promptID, matching `report`.
        var baselineByItem: [String: Double] = [:]
        for readout in ordinal where readout.condition == "baseline" {
            baselineByItem[readout.promptID] = readout.ordinalPosition
        }
        guard !baselineByItem.isEmpty else { return [] }
        var conditionOrder: [String] = []
        var seen = Set<String>()
        for readout in ordinal where readout.condition != "baseline" {
            if seen.insert(readout.condition).inserted {
                conditionOrder.append(readout.condition)
            }
        }
        var entries: [EffectSizeEntry] = []
        for condition in conditionOrder {
            let diffs: [Double] = ordinal
                .filter { $0.condition == condition }
                .sorted { $0.promptID < $1.promptID }
                .compactMap { readout in
                    guard let base = baselineByItem[readout.promptID],
                        let position = readout.ordinalPosition
                    else { return nil }
                    return position - base
                }
            guard !diffs.isEmpty else { continue }
            let ci = StudyStatistics.pairedBootstrapCI(
                diffs, replicates: replicates, seed: 0)
            let wilcoxon = StudyStatistics.wilcoxonSignedRank(diffs)
            entries.append(
                EffectSizeEntry(
                    condition: condition,
                    metric: "ordinalPosition",
                    n: ci.n,
                    meanDiff: ci.mean,
                    ciLower: ci.ciLower,
                    ciUpper: ci.ciUpper,
                    wilcoxonW: wilcoxon.w.isNaN ? nil : wilcoxon.w,
                    wilcoxonP: wilcoxon.p.isNaN ? nil : wilcoxon.p,
                    stratifyBy: stratum?.family,
                    stratum: stratum?.label,
                    // One deterministic readout per (condition, prompt):
                    // the instrument has no sample axis, so a stratified
                    // ordinal pair is always per-item — item-level, and
                    // therefore always a member of the correction family.
                    unit: stratum.map { _ in "item" },
                    estimand: stratum.map { _ in EffectSizeEstimand.itemLevel },
                    inference: stratum.map { _ in EffectSizeInference.corrected }))
        }
        return entries
    }

    // MARK: - Stratified effect sizes (per-cell strata; server twin)

    /// The stratification families for a run, in a fixed order (server twin:
    /// `tasks._stratification_families`): promptID (always), each factor key
    /// with ≥2 observed levels (marginals), and the full cross of ALL factor
    /// keys when there are ≥2 of them (the per-cell view — e.g. `arm×caseID`
    /// → `notLegal×loan`). Keys sort alphabetically; a constant factor is
    /// skipped (its one stratum would duplicate the pooled row under another
    /// name).
    static func stratificationFamilies(
        factorsByItem: [String: [String: String]], items: Set<String>
    ) -> [(name: String, strata: [(label: String, items: Set<String>)])] {
        var families: [(name: String, strata: [(label: String, items: Set<String>)])] = [
            (name: "promptID",
             strata: items.sorted().map { (label: $0, items: Set([$0])) })
        ]
        let keys = Set(factorsByItem.values.flatMap(\.keys)).sorted()
        for key in keys {
            var strata: [String: Set<String>] = [:]
            for item in items {
                if let level = factorsByItem[item]?[key] {
                    strata[level, default: []].insert(item)
                }
            }
            if strata.count >= 2 {
                families.append(
                    (name: key,
                     strata: strata.sorted { $0.key < $1.key }
                         .map { (label: $0.key, items: $0.value) }))
            }
        }
        if keys.count >= 2 {
            var cells: [String: Set<String>] = [:]
            for item in items {
                let levels = factorsByItem[item] ?? [:]
                let values = keys.compactMap { levels[$0] }
                guard values.count == keys.count else { continue }
                cells[values.joined(separator: "×"), default: []].insert(item)
            }
            if cells.count >= 2 {
                families.append(
                    (name: keys.joined(separator: "×"),
                     strata: cells.sorted { $0.key < $1.key }
                         .map { (label: $0.key, items: $0.value) }))
            }
        }
        return families
    }

    /// The stratified companion rows to the pooled effect sizes — the
    /// analyze verb's answer to pooled choice/numeric endpoints BOTH hiding
    /// a real single-cell effect behind saturated cells AND manufacturing a
    /// pooled effect out of one cell's parse garbage. Pooled rows are
    /// untouched (their statistics and correction family do not change);
    /// these rows ADD per-stratum paired statistics through the identical
    /// machinery, corrected per metric WITHIN each stratification family
    /// (never joined to the pooled family). Server twin:
    /// `tasks._stratified_effect_rows`; identical CSV column vocabulary
    /// (stratifyBy, stratum, unit).
    static func stratifiedEffectSizes(
        rows: [MetricRow], concepts: [String], styleFeatureIDs: [String] = [],
        choiceReadouts: [ReportChoiceReadout] = [],
        factorsByItem: [String: [String: String]],
        replicates: Int = 10_000, phase: String? = nil
    ) -> [EffectSizeEntry] {
        var items = Set(rows.map(\.promptID))
        for readout in choiceReadouts
        where readout.source == "instrument" && readout.ordinalPosition != nil {
            items.insert(readout.promptID)
        }
        guard !items.isEmpty else { return [] }
        var entries: [EffectSizeEntry] = []
        for family in stratificationFamilies(
            factorsByItem: factorsByItem, items: items)
        {
            var familyEntries: [EffectSizeEntry] = []
            for (label, members) in family.strata {
                let stratum = (family: family.name, label: label)
                familyEntries += sampledEffectSizes(
                    rows: rows.filter { members.contains($0.promptID) },
                    concepts: concepts, styleFeatureIDs: styleFeatureIDs,
                    replicates: replicates, stratum: stratum)
                familyEntries += ordinalEffectSizes(
                    choiceReadouts.filter { members.contains($0.promptID) },
                    replicates: replicates, stratum: stratum)
            }
            // The phase's correction, per metric WITHIN this family —
            // applyCorrection groups by metric over exactly the entries it
            // is handed — and ONLY over the item-level rows.
            //
            // A `withinItemSamples` row pairs several draws of the SAME item
            // against that item's baseline draws: it measures within-item
            // variability, not an item-level effect, so it is not an
            // independent test of the pre-registered hypothesis. Correcting
            // across those rows inflated the family (shrinking every real
            // row's adjustedP) AND stamped an `adjustedP` that read as a
            // citable test. They are emitted as `diagnostic` instead — raw
            // Wilcoxon and bootstrap CI kept, no adjustedP, no correction
            // stamp. Order is preserved so the CSV row order is unchanged.
            let corrected = applyCorrection(
                familyEntries.filter { !$0.isWithinItemSamples }, phase: phase)
            var correctedRows = corrected.makeIterator()
            entries += familyEntries.map { entry in
                entry.isWithinItemSamples
                    ? entry : (correctedRows.next() ?? entry)
            }
        }
        return entries
    }

    /// `effect-sizes.csv` — fixed cross-engine column set:
    /// condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,
    /// adjustedP,correction,stratifyBy,stratum,unit,estimand,inference
    /// (Wilcoxon/adjusted cells empty when the test is undefined; the
    /// correction and stratification columns name the server contract keys
    /// exactly, so `RunResults.effectSizes(fromCSV:)` reads either engine's
    /// dialect). Rows without stratification provenance are the historical
    /// pooled rows and are labeled "pooled" with EMPTY estimand/inference;
    /// readers that group by (condition, metric) alone must filter on it.
    ///
    /// `estimand`/`inference` are APPENDED (2026-08-06, cross-engine with
    /// the server's two new columns): every existing column keeps its
    /// position, and both engines' readers are name-keyed, so an older
    /// reader simply ignores them.
    static func effectSizesCSV(_ entries: [EffectSizeEntry]) -> String {
        var lines = [
            "condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,"
                + "adjustedP,correction,stratifyBy,stratum,unit,estimand,"
                + "inference"
        ]
        for entry in entries {
            lines.append(
                [
                    csvEscape(entry.condition),
                    csvEscape(entry.metric),
                    String(entry.n),
                    String(entry.meanDiff),
                    String(entry.ciLower),
                    String(entry.ciUpper),
                    entry.wilcoxonW.map { String($0) } ?? "",
                    entry.wilcoxonP.map { String($0) } ?? "",
                    entry.adjustedP.map { String($0) } ?? "",
                    entry.correction.map(csvEscape) ?? "",
                    csvEscape(entry.stratifyBy ?? "pooled"),
                    entry.stratum.map(csvEscape) ?? "",
                    entry.unit.map(csvEscape) ?? "",
                    entry.estimand.map(csvEscape) ?? "",
                    entry.inference.map(csvEscape) ?? "",
                ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func csvEscape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// The layer a condition pins for this concept, else mid-network.
    static func chosenLayer(
        for concept: String, manifest: ExperimentManifest, extraction: ExtractionResult
    ) -> Int {
        validationLayerResolution(
            for: concept, manifest: manifest,
            layerCount: extraction.vectors.layerCount
        ).layer
    }

    /// Every read layer for one concept, in declared order — the list form
    /// of `validationLayerResolution` (validate-at-the-sweep-layers policy;
    /// server twin `_validation_layer_resolutions`).
    static func validationLayerResolutions(
        for concept: String, manifest: ExperimentManifest, layerCount: Int
    ) throws -> [ValidationLayerRule.Resolution] {
        var conditionLayer: Int?
        outer: for condition in manifest.conditions {
            for slot in condition.slots where slot.concept == concept {
                conditionLayer = slot.layer
                break outer
            }
        }
        try requireValidationLayerInRange(manifest, layerCount: layerCount)
        return try ValidationLayerRule.resolveAll(
            concept: concept,
            declaredLayers: manifest.validationLayers,
            declaredFractions: manifest.validationLayerFractions,
            declaredLayer: manifest.validationLayer,
            declaredFraction: manifest.validationLayerFraction,
            conditionLayer: conditionLayer,
            layerCount: layerCount)
    }

    /// The cross-engine `layerResolution` report block for one depth
    /// (review 2026-08-01, P2: the depth-control help promised a stamped
    /// resolution and only the server delivered one — the log line is not
    /// evidence). Same keys and source vocabulary as the server's writer.
    static func resolutionBlock(
        _ resolution: ValidationLayerRule.Resolution
    ) -> [String: Any] {
        [
            "layer": resolution.layer,
            "layerCount": resolution.layerCount,
            "depthFraction": resolution.depthFraction,
            "source": resolution.source.rawValue,
        ]
    }

    /// One per-depth validation report entry (the elements of `depths`).
    static func depthEntry(
        layer: Int, accuracy: Float,
        diagnostics: ScenarioDiagnostics.Report?,
        resolution: ValidationLayerRule.Resolution
    ) -> [String: Any] {
        var sub: [String: Any] = [
            "layer": layer, "accuracy": accuracy,
            "layerResolution": resolutionBlock(resolution),
        ]
        if let diagnostics,
            let encoded = try? JSONEncoder().encode(diagnostics),
            let object = try? JSONSerialization.jsonObject(with: encoded)
        {
            sub["diagnostics"] = object
        }
        return sub
    }

    /// The ONE layer the cross-concept cosine matrix is computed at.
    ///
    /// A study-wide declaration governs; otherwise mid-network is the
    /// documented canonical fallback. Deliberately NOT the per-concept legacy
    /// rule: a per-row layer makes the matrix asymmetric, and an asymmetric
    /// matrix has no defined reading for the maxCrossConceptCosine gate.
    /// Server twin: `_matrix_layer`.
    /// Every artifact must report the SAME layer count, or refuse.
    ///
    /// All vectors in one study belong to one model at one revision, so
    /// differing depths mean a corrupt or mismatched artifact. Clamping each
    /// row to its own depth is what let the matrix become asymmetric again
    /// through the back door.
    static func requireUniformDepth(
        _ extractions: [String: ConceptExtraction]
    ) throws -> Int {
        let depths = extractions.mapValues { $0.result.vectors.layerCount }
        let distinct = Set(depths.values)
        guard distinct.count <= 1 else {
            let detail = depths.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            throw ExperimentError(
                reason: "validation artifacts disagree about model depth "
                    + "(\(detail)) — every vector in one study belongs to the "
                    + "same model revision, so this is a corrupt or mismatched "
                    + "artifact. Re-extract before validating; clamping each "
                    + "to its own depth would make the cosine matrix asymmetric")
        }
        return distinct.first ?? 0
    }

    static func matrixLayers(
        manifest: ExperimentManifest, extractions: [String: ConceptExtraction]
    ) throws -> [Int] {
        let depth = try requireUniformDepth(extractions)
        guard depth > 0 else { return [0] }
        return try ValidationLayerRule.resolveAll(
            concept: "",
            declaredLayers: manifest.validationLayers,
            declaredFractions: manifest.validationLayerFractions,
            declaredLayer: manifest.validationLayer,
            declaredFraction: manifest.validationLayerFraction,
            // No condition fallback: the matrix needs a STUDY layer, and a
            // per-concept one is exactly what makes it asymmetric.
            conditionLayer: nil,
            layerCount: depth
        ).map(\.layer)
    }

    /// The read layer AND why it is that layer (D4). The legacy rule —
    /// inherit from a steering condition, else mid-network — is preserved as
    /// the fallback, so existing manifests keep their numbers; a declared
    /// `validationLayer` / `validationLayerFraction` now takes precedence and
    /// says so in the run log.
    static func validationLayerResolution(
        for concept: String, manifest: ExperimentManifest, layerCount: Int
    ) -> ValidationLayerRule.Resolution {
        var conditionLayer: Int?
        outer: for condition in manifest.conditions {
            for slot in condition.slots where slot.concept == concept {
                conditionLayer = slot.layer
                break outer
            }
        }
        return ValidationLayerRule.resolve(
            concept: concept,
            declaredLayer: manifest.validationLayer,
            declaredFraction: manifest.validationLayerFraction,
            conditionLayer: conditionLayer,
            layerCount: layerCount)
    }

    /// Refuse an explicit out-of-range declaration once depth is known.
    /// Checked where the layer is resolved, since verify() loads no model.
    static func requireValidationLayerInRange(
        _ manifest: ExperimentManifest, layerCount: Int
    ) throws {
        if let refusal = ValidationLayerRule.rangeRefusal(
            declaredLayer: manifest.validationLayer, layerCount: layerCount)
        {
            throw ExperimentError(reason: refusal)
        }
    }

    /// Classifies never-named scenarios by projection onto the concept
    /// direction, thresholded at the midpoint of the training-class mean
    /// projections (same rule as the panel's held-out stat, but on
    /// distribution-shifted scenarios that played no role in extraction) —
    /// at EVERY declared read depth. Activations are captured once for all
    /// layers; extra depths cost per-layer arithmetic, not forward passes
    /// (server twin: the depth loop in ``_task_validate``).
    static func scenarioAccuracyProfile(
        scenarios: [StimulusSet.ValidationScenario], extraction: ExtractionResult,
        stimuli: StimulusSet, layers: [Int], options: ExtractionOptions,
        container: ModelContainer
    ) async throws
        -> [(layer: Int, accuracy: Float, diagnostics: ScenarioDiagnostics.Report)]
    {
        // Held-out activations must be read where AND rendered how the
        // vector was — a projection onto the direction is only meaningful
        // against samples from the distribution it was read from.
        //
        // Frame-free by the same rule as the grand-mean path above: the study
        // frame (and any composed agent persona) is generation arming and
        // never enters a held-out read. The pinned
        // `extractionRendering.systemPrompt` resolved here is the one
        // sanctioned channel for a persona-conditioned validation.
        let rendering = options.resolvedExtractionRendering
        let positives = try await ConceptExtractor.activations(
            container: container, texts: stimuli.positive,
            position: options.readingPosition, rendering: rendering)
        let negatives = try await ConceptExtractor.activations(
            container: container, texts: stimuli.negative,
            position: options.readingPosition, rendering: rendering)
        let scenarioActivations = try await ConceptExtractor.activations(
            container: container, texts: scenarios.map(\.text),
            position: options.readingPosition, rendering: rendering)

        var profile: [(layer: Int, accuracy: Float,
                       diagnostics: ScenarioDiagnostics.Report)] = []
        for layer in layers {
            let direction = extraction.vectors.perLayer[layer]
            let posMean = try SteeringVectorMath.mean(
                positives.values.map { $0[layer] })
            let negMean = try SteeringVectorMath.mean(
                negatives.values.map { $0[layer] })
            let posProjection = SteeringVectorMath.dot(direction, posMean)
            let negProjection = SteeringVectorMath.dot(direction, negMean)
            let midpoint = (posProjection + negProjection) / 2
            // D1: keep the working. The accuracy is computed FROM these
            // projections and this midpoint, and discarding them left no way
            // to tell "does not read the concept" from "ranks correctly,
            // thresholds badly".
            let projections = scenarioActivations.values.map {
                Double(SteeringVectorMath.dot(direction, $0[layer]))
            }
            let report = try ScenarioDiagnostics.report(
                scenarioIDs: scenarios.map { _ in nil },
                scenarioTexts: scenarios.map(\.text),
                projections: projections,
                labels: scenarios.map(\.expresses),
                threshold: Double(midpoint),
                classMeans: [
                    "positive": Double(posProjection),
                    "negative": Double(negProjection),
                ],
                layer: layer,
                directionNorm: Double(SteeringVectorMath.l2Norm(direction)))
            let accuracy = Float(report.rows.count(where: \.correct))
                / Float(max(1, scenarios.count))
            profile.append((layer, accuracy, report))
        }
        return profile
    }
}
