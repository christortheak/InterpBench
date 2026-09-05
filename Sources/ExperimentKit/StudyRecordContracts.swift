import Foundation
import SteeringKit

// Shared wire records and their pure derived summaries.
// Existing nested type names remain source-compatible.
extension ExperimentTasks {
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
        /// WHY the generation ended — `FinishReason`'s closed vocabulary,
        /// read from the decode stream rather than inferred from the text.
        /// The third thing a record says about its output, and the only one
        /// the output cannot express: a capped generation and a terse one
        /// read the same, and `wordCount` only approximates.
        ///
        /// Optional so a record from an engine that predates the field (or a
        /// path that genuinely cannot know) is ABSENT rather than guessed —
        /// "nobody classified this" is a different claim from "this
        /// finished". Every sampled record this engine writes carries it.
        var finishReason: String? = nil
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
        /// Why this row's generation ended (`FinishReason`), so the run's
        /// per-cell truncation block and its gate read the same in-memory
        /// rows every other per-condition aggregate is built from. NOT a
        /// metrics.csv column: that header is a closed cross-engine contract,
        /// and the reading belongs in report.json's `truncation` block beside
        /// the other aggregates. nil = a row from a path that classified
        /// nothing.
        var finishReason: String? = nil
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
        /// Per-cell truncation (cross-engine contract key "truncation").
        /// Written whether or not the study declared a ceiling: the reading
        /// is evidence about the run, and the gate is only what a study chose
        /// to do about it. nil ⇒ key omitted (legacy report bytes unchanged).
        var truncation: TruncationReport? = nil
    }

    /// One cell's truncation reading — (condition, promptID), the same unit
    /// `summaries.csv` aggregates over and deliberately NOT the run.
    struct TruncationCellReport: Codable, Equatable {
        let condition: String
        let promptID: String
        /// Generations that carry a `finishReason` at all. A record from an
        /// engine that predates the field is a generation nobody classified,
        /// which is a different fact from one that finished.
        let classified: Int
        /// Generations that were CUT OFF anywhere — at the answer cap or, under
        /// a reasoning budget, at the reasoning cap. Both are incomplete.
        let lengthStopped: Int
        /// The subset of `lengthStopped` that never closed its reasoning
        /// block, reported beside it so a reader can tell an answer budget
        /// that is too small from a reasoning budget that is.
        let lengthStoppedInReasoning: Int
        let lengthStoppedFraction: Double
    }

    /// The run's truncation block. Server twin: `truncation_gate.report`.
    struct TruncationReport: Codable {
        /// The declared ceiling, or nil when the study declared none.
        let threshold: Double?
        let classified: Int
        let lengthStopped: Int
        let lengthStoppedInReasoning: Int
        /// Pooled over the run — reported for orientation and never gated on.
        /// The 2026-08-30 incident's pooled fraction looked unremarkable
        /// while one whole arm was truncated; `cells` is the reading that
        /// would have caught it.
        let lengthStoppedFraction: Double
        let cells: [TruncationCellReport]
    }

    /// `(classified, lengthStopped, lengthStoppedInReasoning)` for one cell
    /// of a row set.
    static func truncationCell(
        rows: [MetricRow], condition: String, promptID: String
    ) -> (classified: Int, lengthStopped: Int, inReasoning: Int) {
        var classified = 0
        var stopped = 0
        var inReasoning = 0
        for row in rows where row.condition == condition && row.promptID == promptID {
            guard let reason = row.finishReason else { continue }
            classified += 1
            if FinishReason.isCutOff(reason) { stopped += 1 }
            if reason == FinishReason.lengthInReasoning { inReasoning += 1 }
        }
        return (classified, stopped, inReasoning)
    }

    static func truncationReport(
        rows: [MetricRow], threshold: Double?
    ) -> TruncationReport {
        var order: [String] = []
        var byCell:
            [String: (condition: String, promptID: String, n: Int, stopped: Int,
                inReasoning: Int)] = [:]
        for row in rows {
            guard let reason = row.finishReason else { continue }
            let key = row.condition + "\u{0}" + row.promptID
            if byCell[key] == nil {
                order.append(key)
                byCell[key] = (row.condition, row.promptID, 0, 0, 0)
            }
            byCell[key]?.n += 1
            if FinishReason.isCutOff(reason) { byCell[key]?.stopped += 1 }
            if reason == FinishReason.lengthInReasoning {
                byCell[key]?.inReasoning += 1
            }
        }
        let cells =
            order
            .compactMap { byCell[$0] }
            .map {
                TruncationCellReport(
                    condition: $0.condition, promptID: $0.promptID,
                    classified: $0.n, lengthStopped: $0.stopped,
                    lengthStoppedInReasoning: $0.inReasoning,
                    lengthStoppedFraction: Double($0.stopped) / Double($0.n))
            }
            .sorted {
                ($0.condition, $0.promptID) < ($1.condition, $1.promptID)
            }
        let classified = cells.reduce(0) { $0 + $1.classified }
        let stopped = cells.reduce(0) { $0 + $1.lengthStopped }
        let inReasoning = cells.reduce(0) { $0 + $1.lengthStoppedInReasoning }
        return TruncationReport(
            threshold: threshold, classified: classified, lengthStopped: stopped,
            lengthStoppedInReasoning: inReasoning,
            lengthStoppedFraction: classified == 0
                ? 0 : Double(stopped) / Double(classified),
            cells: cells)
    }

    /// The complete refusal for one cell over the declared ceiling, or nil.
    ///
    /// STRICTLY over: a declared ceiling of 0.25 permits a cell sitting
    /// exactly at a quarter, so the number an author writes down is the
    /// largest fraction they are willing to accept rather than one short of
    /// it. Server twin: `truncation_gate.cell_refusal`.
    static func lengthStoppedRefusal(
        classified: Int, lengthStopped: Int, threshold: Double,
        condition: String, promptID: String, maxTokens: Int,
        lengthStoppedInReasoning: Int = 0, reasoningMaxTokens: Int? = nil
    ) -> String? {
        guard classified > 0 else { return nil }
        let fraction = Double(lengthStopped) / Double(classified)
        guard fraction > threshold else { return nil }
        // Under a reasoning budget the sentence says which cap was hit:
        // raising maxTokens would not have helped a generation that never
        // closed its reasoning block. Server twin: `truncation_gate.cell_refusal`.
        let hit: String
        if let reasoningMaxTokens, lengthStoppedInReasoning > 0 {
            hit =
                "stopped at a token cap instead of finishing — "
                + "\(lengthStoppedInReasoning) inside the reasoning block at the "
                + "\(reasoningMaxTokens)-token reasoning cap, "
                + "\(lengthStopped - lengthStoppedInReasoning) in the answer at "
                + "the \(maxTokens)-token answer cap —"
        } else {
            hit = "stopped at the \(maxTokens)-token cap instead of finishing"
        }
        return
            "condition '\(condition)' item '\(promptID)': \(lengthStopped) of "
            + "\(classified) generation(s) \(hit) "
            + String(format: "(%.1f%%)", fraction * 100)
            + ", over the declared maxLengthStoppedFraction of "
            + String(format: "%.1f%%", threshold * 100)
            + ". A capped generation is cut off, not short — its text is "
            + "missing whatever the model had not written yet, and an endpoint "
            + "computed from this cell is computed from truncated text. "
            + "Truncation is not spread evenly across arms, so a run-wide "
            + "fraction would not have shown this"
    }

    /// The executable repair. A command, not advice. Under a reasoning budget
    /// it names BOTH flags, because the refusal has said which cap was hit.
    static func lengthStoppedRepair(
        experiment: String, maxTokens: Int, reasoningMaxTokens: Int? = nil
    ) -> String {
        let flags: String
        if let reasoningMaxTokens {
            flags =
                "--max-tokens <n> and/or --reasoning-max-tokens <m>  "
                + "(n above \(maxTokens), m above \(reasoningMaxTokens), "
                + "whichever cap the refusal names)"
        } else {
            flags = "--max-tokens <n>  (n above \(maxTokens))"
        }
        return "steerlab-cli experiment set-sampling \(experiment) \(flags), "
            + "then re-run; a frozen study is iterated by duplicating first: "
            + "steerlab-cli experiment duplicate \(experiment) \(experiment)-v2"
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

}
