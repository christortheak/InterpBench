import CryptoKit
import Foundation
import SteeringKit

/// An experiment is a discrete, freezable deployment of data on the same
/// app: it pins stimulus sets BY HASH plus extraction options (the recipe),
/// not vector bytes — every run re-derives vectors deterministically from
/// the pinned inputs. Freezing is the circularity firewall made mechanical:
/// after freeze, the manifest is read-only, runs stamp its hash, and any
/// drift in the underlying stimulus files surfaces as a freeze violation
/// rather than a silent change (CLAUDE.md › Data & reproducibility).
public struct ExperimentManifest: Codable, Sendable, Equatable {
    enum CodingKeys: String, CodingKey {
        case name
        case experimentDescription
        case taskDescription
        case outcomeMeasures
        case taskPromptsFile
        case taskPromptsHash
        case studyKind
        case studyType
        case multiAgentScenarioPath
        case multiAgentScenarioHash
        case multiAgentSemanticScenarioPath
        case multiAgentSemanticScenarioHash
        case multiAgentIncludeBaseline
        case createdAt
        case modelID
        case modelRevision
        case dtype
        case concepts
        case grandMeanCorpus
        case conditions
        case variantConditions
        case neutralCorpusHash
        case sweep
        case pipeline
        case evaluation
        case judgeRubricFile
        case judgeRubricHash
        case judges
        case humanValidation
        case capabilityBatteryFile
        case capabilityBatteryHash
        case markersHash
        case reasoningStyleTaxonomyPath
        case reasoningStyleTaxonomyHash
        case jlensReadout
        case recordTokenIDs
        case saeCandidates
        case maxSAEMixtureFeatures
        case saeLatentConditions
        case phase
        case caseFamily
        case outcomeInstruments
        case outcomeInstrumentScope
        case evaluationSampling
        case validationControls
        case validationLayer
        case validationLayerFraction
        case validationLayers
        case validationLayerFractions
        case ordinalAggregation
        case numericParser
        case parserRegistryHash
        case exclusionRules
        case maxLengthStoppedFraction
        case acknowledgeUnequalOptionLengths
        case samplesPerItem
        case seedPolicy
        case screenTaskPromptsHash
        case humanBaseline
        case readerRefs
        case promotionRule
        case perturbationPolicy
        case promptMode
        case systemPrompt
        case qwenThinkingEnabled
        case reasoningEffort
        case reasoningMaxTokens
        case seeds
        case temperature
        case maxTokens
        case templateProvenance
        case status
        case frozenAt
        case freezeHash
        case frozenBy
        case gitCommit
        case appVersion
        case freezeForced
        case forcedGatesSkipped
        case preregistrationHash
        case preregistrationGeneratedHash
    }

    public enum Status: String, Codable, Sendable {
        case draft
        case frozen
        case complete
    }

    public enum PromptMode: String, Codable, Sendable, CaseIterable {
        case chatAssistant
        case rawCompletion

        public var label: String {
            switch self {
            case .chatAssistant: "chat assistant"
            case .rawCompletion: "raw completion"
            }
        }
    }

    public enum StudyKind: String, Codable, Sendable, CaseIterable {
        case modelOutput
        case multiAgent

        public var label: String {
            switch self {
            case .modelOutput: "Model Output"
            case .multiAgent: "Multi-Agent"
            }
        }
    }

    /// A pinned concept recipe: stimuli by hash + how to extract.
    public struct ConceptRef: Codable, Sendable, Equatable {
        enum CodingKeys: String, CodingKey {
            case name
            case stimulusSetHash
            case options
            case validationHash
            case designatedReference
            case vectorArtifact
        }

        /// designatedReference concepts only: the pinned reference stories
        /// corpus (cross-engine key "designatedReference"). The vector is
        /// mean(concept stories) − mean(reference stories), so drifting
        /// reference bytes are a verify violation exactly like stimuli.
        public struct DesignatedReferencePin: Codable, Sendable, Equatable {
            public var name: String
            public var hash: String

            public init(name: String, hash: String) {
                self.name = name
                self.hash = hash
            }
        }

        /// ARTIFACT-PINNED concepts only (method `pinnedArtifact`;
        /// cross-engine contract key "vectorArtifact", commit af1af0e). A
        /// recipe concept pins stimuli and RE-DERIVES its vector every run;
        /// an artifact-pinned concept pins the VECTOR BYTES — the honest
        /// firewall for post-hoc derived directions (family-grand-mean
        /// centring, OptVec) that no stimulus recipe reproduces. Keys match
        /// the server's `manifest.ConceptRef.vector_artifact` block exactly;
        /// the optvec* keys are additive provenance copied from an OptVec
        /// sidecar's `optvec` block at attach (absent on non-OptVec pins).
        public struct VectorArtifactPin: Codable, Sendable, Equatable {
            /// Workspace-relative, EXTENSION-LESS locator
            /// (`ArtifactIdentity` convention: `<path>.safetensors` +
            /// `<path>.json`).
            public var path: String
            /// SHA-256 of `<path>.safetensors` raw bytes.
            public var sha256TensorHash: String
            /// SHA-256 of `<path>.json` raw bytes.
            public var sha256SidecarHash: String
            /// The extractionMethod recorded in that sidecar (a raw string,
            /// e.g. "optvec" — the DATA method every lifecycle question
            /// resolves through `effectiveMethod`).
            public var sourceMethod: String
            /// The concept whose stimuli and held-out validation.jsonl the
            /// probe reads; equals the manifest concept name for OptVec
            /// pins (which have no source concept at all).
            public var sourceConcept: String
            /// The artifact's norm-denominator provenance, and the neutral
            /// corpus it was measured on.
            public var residualNormSource: String
            public var normCorpusHash: String?
            // MIRRORED POLE linkage (`PoleMirror`). Both absent for every
            // ordinary pin, so existing manifests re-encode byte-identically
            // and keep their content hash.
            /// True when the pinned artifact is a mirrored pole: its sidecar
            /// carries `polesSwappedFromSource`, and this concept's stimulus
            /// directory holds the SOURCE concept's two files with their
            /// positive/negative roles exchanged.
            public var polesSwappedFromSource: Bool?
            /// The sidecar's inherited `stimulusSetHash` — the SOURCE
            /// concept's order-sensitive hash, `sha256(positive ‖ negative)`
            /// of the parent's files. The concept's OWN hash is the pin's
            /// `ConceptRef.stimulusSetHash` (what verify recomputes); this is
            /// the claim that links the two, and verify re-derives it by
            /// hashing this concept's files in the parent's order.
            public var sourceStimulusSetHash: String?
            // OptVec provenance (additive; server writes them from the
            // sidecar's `optvec` block so the manifest is self-describing
            // about what the direction was trained to do and which eval
            // run certifies it — freeze surfaces the latter as an advisory).
            public var optvecLayer: Int?
            public var optvecTrainingRun: String?
            public var optvecSeed: Int?
            public var optvecEvalRun: String?
            /// Whether the eval-run citation was RESOLVED at attach (run
            /// directory found, eval.json certifies this artifact's tensor
            /// hash). Absent = legacy attach, recorded before verification
            /// existed (2026-08-10) — the freeze advisory says so.
            public var optvecEvalRunVerified: Bool?
            public var optvecEvalRunUnverifiedReason: String?

            public init(
                path: String, sha256TensorHash: String,
                sha256SidecarHash: String, sourceMethod: String,
                sourceConcept: String, residualNormSource: String,
                normCorpusHash: String? = nil,
                polesSwappedFromSource: Bool? = nil,
                sourceStimulusSetHash: String? = nil, optvecLayer: Int? = nil,
                optvecTrainingRun: String? = nil, optvecSeed: Int? = nil,
                optvecEvalRun: String? = nil,
                optvecEvalRunVerified: Bool? = nil,
                optvecEvalRunUnverifiedReason: String? = nil
            ) {
                self.path = path
                self.sha256TensorHash = sha256TensorHash
                self.sha256SidecarHash = sha256SidecarHash
                self.sourceMethod = sourceMethod
                self.sourceConcept = sourceConcept
                self.residualNormSource = residualNormSource
                self.normCorpusHash = normCorpusHash
                self.polesSwappedFromSource = polesSwappedFromSource
                self.sourceStimulusSetHash = sourceStimulusSetHash
                self.optvecLayer = optvecLayer
                self.optvecTrainingRun = optvecTrainingRun
                self.optvecSeed = optvecSeed
                self.optvecEvalRun = optvecEvalRun
                self.optvecEvalRunVerified = optvecEvalRunVerified
                self.optvecEvalRunUnverifiedReason = optvecEvalRunUnverifiedReason
            }
        }

        public var name: String
        public var stimulusSetHash: String
        public var options: ExtractionOptions
        /// SHA-256 over the concept's never-named `validation.jsonl` raw
        /// bytes (the convergent-validity scenarios the `validate` gate
        /// reads). Cross-engine key "validationHash", three-state contract:
        /// key ABSENT = legacy attach (verify passes, freeze advises);
        /// key NULL = attach found no validation.jsonl (a file appearing
        /// later is drift); key set = pinned. New attaches always write the
        /// key.
        public var validationHash: String?
        /// True when the manifest carries an EXPLICIT `"validationHash":
        /// null` (attach pinned the file as absent), as opposed to a legacy
        /// manifest with no key at all. Encoded as the null itself, never a
        /// separate key.
        public var validationHashPinnedAbsent: Bool
        public var designatedReference: DesignatedReferencePin?
        public var vectorArtifact: VectorArtifactPin?

        public init(
            name: String, stimulusSetHash: String, options: ExtractionOptions,
            validationHash: String? = nil, validationHashPinnedAbsent: Bool = false,
            designatedReference: DesignatedReferencePin? = nil,
            vectorArtifact: VectorArtifactPin? = nil
        ) {
            self.name = name
            self.stimulusSetHash = stimulusSetHash
            self.options = options
            self.validationHash = validationHash
            self.validationHashPinnedAbsent =
                validationHash == nil && validationHashPinnedAbsent
            self.designatedReference = designatedReference
            self.vectorArtifact = vectorArtifact
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            stimulusSetHash = try container.decode(String.self, forKey: .stimulusSetHash)
            options = try container.decode(ExtractionOptions.self, forKey: .options)
            validationHash = try container.decodeIfPresent(
                String.self, forKey: .validationHash)
            validationHashPinnedAbsent = try validationHash == nil
                && container.contains(.validationHash)
                && container.decodeNil(forKey: .validationHash)
            designatedReference = try container.decodeIfPresent(
                DesignatedReferencePin.self, forKey: .designatedReference)
            vectorArtifact = try container.decodeIfPresent(
                VectorArtifactPin.self, forKey: .vectorArtifact)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            try container.encode(stimulusSetHash, forKey: .stimulusSetHash)
            try container.encode(options, forKey: .options)
            if let validationHash {
                try container.encode(validationHash, forKey: .validationHash)
            } else if validationHashPinnedAbsent {
                try container.encodeNil(forKey: .validationHash)
            }
            try container.encodeIfPresent(
                designatedReference, forKey: .designatedReference)
            try container.encodeIfPresent(vectorArtifact, forKey: .vectorArtifact)
        }

        /// Whether this concept materializes from pinned artifact bytes
        /// (method `pinnedArtifact` AND a pin present — both engines ask
        /// both, so a half-declared concept surfaces in verify instead of
        /// silently branching).
        public var isPinnedArtifact: Bool {
            options.method == .pinnedArtifact && vectorArtifact != nil
        }

        /// The DATA method: what the direction actually is, as opposed to
        /// how it enters the study. For a pinned artifact this resolves the
        /// sidecar-recorded source method ("optvec" resolves to `.optvec`);
        /// nil when the source method is one this engine does not know —
        /// callers must treat that as a violation, never a fallback. Server
        /// twin: `manifest.ConceptRef.effective_method`.
        public var effectiveMethod: ExtractionMethod? {
            guard options.method == .pinnedArtifact else { return options.method }
            guard let source = vectorArtifact?.sourceMethod else { return nil }
            return ExtractionMethod(rawValue: source)
        }

        /// The concept whose stimuli and held-out data answer DATA-side
        /// questions (an artifact-pinned "crit-gm" reads "crit"'s). Server
        /// twin: `manifest.ConceptRef.data_concept`.
        public var dataConcept: String {
            guard options.method == .pinnedArtifact,
                let source = vectorArtifact?.sourceConcept, !source.isEmpty
            else { return name }
            return source
        }
    }

    /// The pinned population for grand-mean extraction. A grand-mean vector
    /// is mean(concept stories) − mean(ALL corpus stories), so the vector
    /// depends on every member of the corpus — membership and every member's
    /// stories.jsonl hash must be pinned, not just the target concept's own
    /// file. JSON shape matches the server's `GrandMeanCorpus` exactly.
    public struct GrandMeanCorpus: Codable, Sendable, Equatable {
        public var concepts: [String]
        public var hashes: [String: String]

        public init(concepts: [String], hashes: [String: String]) {
            self.concepts = concepts
            self.hashes = hashes
        }
    }

    /// The sweep's DECLARED selection criterion — manifest data, not compiled
    /// code (cross-engine contract with the server's `sweep_selection.py`).
    /// Every field is optional so an absent block (or absent subfield)
    /// resolves to the historical hardcoded behavior via
    /// `SweepSelectionRule.resolve`; the RESOLVED criterion is embedded
    /// verbatim in selection provenance (deliberately not hashed — verbatim
    /// embedding avoids cross-engine JSON canonicalization).
    public struct SweepSelection: Codable, Sendable, Equatable {
        public struct Objective: Codable, Sendable, Equatable {
            /// "markerDensity" | "judgeScore" | "logprobShift" (all
            /// implemented on both engines; unknown strings refuse at
            /// declaration AND at sweep start).
            public var metric: String
            /// logprobShift only: the declared dev choice-prompt JSONL
            /// (study-path choice-row schema: prompt + options + optional
            /// target). All other fields stay nil in the DECLARED block —
            /// they are stamped into the RESOLVED criterion at sweep start.
            public var choicePromptsFile: String?
            /// logprobShift only: the PER-CONCEPT form (2026-08-02) —
            /// `{concept: path}`, one instrument per attached concept, so a
            /// multi-concept sweep never scores one concept's cells on
            /// another's items. Exactly one of the two file declarations
            /// may be set; coverage of every attached concept is enforced
            /// at sweep start.
            public var choicePromptsFiles: [String: String]?
            /// SHA-256 of the choice file's raw bytes. In PROVENANCE copies
            /// it is stamped at resolve time; in the DECLARED manifest
            /// block it is the freeze-time pin (review 2026-08-02, P1 — the
            /// files that determine the winning cell were the one sweep
            /// input not pinned at freeze), enforced by verify and refused
            /// on drift at sweep start.
            public var choicePromptsHash: String?
            /// The per-concept pin map for `choicePromptsFiles` — same
            /// freeze-pin contract, keyed by concept.
            public var choicePromptsHashes: [String: String]?
            /// judgeScore only: the manifest's pinned rubric hash, stamped
            /// at resolve time (judge config comes from MANIFEST pins).
            public var judgeRubricHash: String?
            /// judgeScore only: the manifest's judge panel, embedded
            /// verbatim at resolve time.
            public var judges: [JudgeRef]?

            public init(
                metric: String,
                choicePromptsFile: String? = nil,
                choicePromptsFiles: [String: String]? = nil,
                choicePromptsHash: String? = nil,
                choicePromptsHashes: [String: String]? = nil,
                judgeRubricHash: String? = nil,
                judges: [JudgeRef]? = nil
            ) {
                self.metric = metric
                self.choicePromptsFile = choicePromptsFile
                self.choicePromptsFiles = choicePromptsFiles
                self.choicePromptsHash = choicePromptsHash
                self.choicePromptsHashes = choicePromptsHashes
                self.judgeRubricHash = judgeRubricHash
                self.judges = judges
            }
        }

        public struct Constraints: Codable, Sendable, Equatable {
            /// Battery accuracy must stay within this of baseline (default 0.15).
            public var capabilityTolerance: Double?
            /// Distinct-bigram ratio floor. Under the LEGACY rule this is the
            /// whole coherence gate (default 0.45); under the
            /// baseline-relative rule it is the absolute BACKSTOP beneath the
            /// relative bar.
            public var coherenceFloor: Double?
            /// BASELINE-RELATIVE coherence (default for new declarations,
            /// 0.85): a cell passes only when its distinct-2 is at least this
            /// multiple of the α=0 baseline cell's. Its PRESENCE — either this
            /// key or `coherenceAbsoluteBackstop` — is what selects the
            /// relative rule, so a criterion carrying neither means the
            /// absolute rule at `coherenceFloor` and keeps meaning that
            /// forever. Both optional + omitted-when-nil, so every existing
            /// manifest re-encodes byte-identically and keeps its content hash.
            public var coherenceRatioToBaseline: Double?
            /// The absolute floor under the relative bar (default 0.60) — what
            /// stops a degenerate BASELINE from licensing a degenerate winner.
            public var coherenceAbsoluteBackstop: Double?

            public init(
                capabilityTolerance: Double? = nil, coherenceFloor: Double? = nil,
                coherenceRatioToBaseline: Double? = nil,
                coherenceAbsoluteBackstop: Double? = nil
            ) {
                self.capabilityTolerance = capabilityTolerance
                self.coherenceFloor = coherenceFloor
                self.coherenceRatioToBaseline = coherenceRatioToBaseline
                self.coherenceAbsoluteBackstop = coherenceAbsoluteBackstop
            }
        }

        public struct Controls: Codable, Sendable, Equatable {
            /// When set, the winning cell must beat a deterministic
            /// matched-norm random direction (same layer/alpha) by at least
            /// this margin, else no recommendation is made.
            public var matchedNormRandomMargin: Double?
            /// How the control is applied (2026-08-03): "winner"
            /// (historical, absent = this) controls only the argmax cell;
            /// "topK" controls the top `topK` promotable cells in objective
            /// order and promotes the FIRST that beats its own control —
            /// one disruption-artifact corner can no longer veto a grid
            /// containing a legitimate winner.
            public var applyTo: String?
            public var topK: Int?

            public init(
                matchedNormRandomMargin: Double? = nil,
                applyTo: String? = nil, topK: Int? = nil
            ) {
                self.matchedNormRandomMargin = matchedNormRandomMargin
                self.applyTo = applyTo
                self.topK = topK
            }
        }

        public var objective: Objective?
        public var constraints: Constraints?
        public var controls: Controls?

        public init(
            objective: Objective? = nil,
            constraints: Constraints? = nil,
            controls: Controls? = nil
        ) {
            self.objective = objective
            self.constraints = constraints
            self.controls = controls
        }
    }

    /// Selection provenance stamped by the sweep on its `<concept>-recommended`
    /// condition (and copied into a promoted agent's birth certificate): which
    /// run, which resolved criterion, which dev split, which cell, which
    /// metrics — the mechanical record that settings were chosen on dev data
    /// by a predeclared rule. JSON shape is a pinned cross-engine contract.
    public struct SelectionProvenance: Codable, Sendable, Equatable {
        public struct Cell: Codable, Sendable, Equatable {
            public var layer: Int
            public var alpha: Double

            public init(layer: Int, alpha: Double) {
                self.layer = layer
                self.alpha = alpha
            }
        }

        public struct Control: Codable, Sendable, Equatable {
            public var type: String
            public var metricValue: Double
            public var margin: Double
            /// Which random-control recipe generated the control direction
            /// (`SteeringVectorMath.randomVectorAlgorithm`; the server stamps
            /// the identical string). Optional so legacy manifests keep their
            /// content hash — an UNSTAMPED control is legacy: cube-uniform on
            /// Swift, Gaussian on the server.
            public var randomVectorAlgorithm: String?

            public init(
                type: String, metricValue: Double, margin: Double,
                randomVectorAlgorithm: String? = nil
            ) {
                self.type = type
                self.metricValue = metricValue
                self.margin = margin
                self.randomVectorAlgorithm = randomVectorAlgorithm
            }
        }

        public var sweepRun: String
        public var criterion: SweepSelection
        public var devPromptsHash: String
        /// The generation length the sweep's coherence floor was measured at
        /// (cross-engine key "devMaxTokens"; server twin stamps the same).
        ///
        /// The c18 lesson: collapse hidden at 256 tokens was decisive at
        /// 1024, so a winning cell's distinct-2 is only study-relevant
        /// evidence if the dev generations were at least as long as the
        /// study's. Stamping it makes that checkable after the fact instead
        /// of reconstructible only from the sweep spec that has since been
        /// edited. Optional + omit-when-nil so pre-existing manifests decode
        /// unchanged and keep their content hash.
        public var devMaxTokens: Int?
        public var winningCell: Cell
        public var metrics: [String: Double]
        public var control: Control?

        public init(
            sweepRun: String,
            criterion: SweepSelection,
            devPromptsHash: String,
            devMaxTokens: Int? = nil,
            winningCell: Cell,
            metrics: [String: Double],
            control: Control? = nil
        ) {
            self.sweepRun = sweepRun
            self.criterion = criterion
            self.devPromptsHash = devPromptsHash
            self.devMaxTokens = devMaxTokens
            self.winningCell = winningCell
            self.metrics = metrics
            self.control = control
        }
    }

    /// A steering condition: the slot boxes plus the shared globals,
    /// referencing concepts by name (vectors are re-derived at run time).
    public struct Condition: Codable, Sendable, Equatable {
        public struct Slot: Codable, Sendable, Equatable {
            public var concept: String
            public var layer: Int
            /// α when steering, λ when ablating.
            public var alpha: Double
            /// `add` (steer) or `ablate`. Absent means `add`, and an explicit
            /// `add` is never written: manifest bytes are the content hash, so
            /// a key appearing on every existing condition would re-identify
            /// every frozen study in the workspace.
            public var mode: InterventionPlan.Mode?

            public var effectiveMode: InterventionPlan.Mode { mode ?? .add }

            public init(
                concept: String, layer: Int, alpha: Double,
                mode: InterventionPlan.Mode? = nil
            ) {
                self.concept = concept
                self.layer = layer
                self.alpha = alpha
                self.mode = mode
            }

            enum CodingKeys: String, CodingKey {
                case concept, layer, alpha, mode
            }

            public func encode(to encoder: any Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                try container.encode(concept, forKey: .concept)
                try container.encode(layer, forKey: .layer)
                try container.encode(alpha, forKey: .alpha)
                if let mode, mode != .add {
                    try container.encode(mode, forKey: .mode)
                }
            }
        }

        public var name: String
        public var slots: [Slot]
        public var bandWidth: Int
        public var alphaInNormUnits: Bool
        public var neutralPCBasisPath: String?
        public var neutralPCBasisLabel: String?
        public var neutralPCBasisHash: String?
        /// "randomMatchedNorm": each slot injects a deterministic random
        /// direction norm-matched to the named concept's vector at that layer
        /// — the magnitude/noise control cell, expressible as data instead of
        /// ad-hoc code.
        ///
        /// "randomDirectionAblation": the ABLATION analogue. Norm-matching is
        /// meaningless for a projection — ablation removes whatever is present
        /// regardless of the direction's length — so the control substitutes a
        /// deterministic random DIRECTION and removes that instead. It answers
        /// the question the concept ablation raises: is the effect specific to
        /// this direction, or does removing any rank-1 subspace of the
        /// residual stream do it? Uses the same seeding convention, so the
        /// control cell is reproducible across re-runs of a frozen study.
        public var controlType: String?
        /// Present only on sweep-written `<concept>-recommended` conditions:
        /// how this cell was selected (run, resolved criterion, dev split,
        /// metrics, control). Optional so legacy manifests keep their
        /// content hash.
        public var selection: SelectionProvenance?

        public init(
            name: String, slots: [Slot], bandWidth: Int = 1, alphaInNormUnits: Bool = true,
            neutralPCBasisPath: String? = nil,
            neutralPCBasisLabel: String? = nil,
            neutralPCBasisHash: String? = nil,
            controlType: String? = nil,
            selection: SelectionProvenance? = nil
        ) {
            self.name = name
            self.slots = slots
            self.bandWidth = bandWidth
            self.alphaInNormUnits = alphaInNormUnits
            self.neutralPCBasisPath = neutralPCBasisPath
            self.neutralPCBasisLabel = neutralPCBasisLabel
            self.neutralPCBasisHash = neutralPCBasisHash
            self.controlType = controlType
            self.selection = selection
        }
    }

    /// The repair for a condition DOCUMENT that declares no `alphaInNormUnits`
    /// — in BOTH spellings a caller can write it in, because the two authoring
    /// surfaces are the manifest document and the CLI verb, and a client that
    /// only learns one of them is stuck the first time it uses the other.
    ///
    /// Phase-0 gap G6 (`docs/PORTABILITY-CONTRACTS.md`): the key's DEFAULT
    /// disagreed across the engines — `true` from `Condition.init` here,
    /// `False` from the server's `_condition_entry` — so the same client call
    /// authored a different study depending on which engine served it. α units
    /// are dose semantics (`docs/CONDUCTING-A-STUDY.md`: α in norm units is
    /// the standing convention, and a raw α at the same number is a different
    /// intervention), so the repair is not to pick a default: it is to refuse
    /// a NEW declaration that does not say. Server twin:
    /// `experiment_store._condition_entry`'s refusal, whose repair names the
    /// same two spellings.
    public static let alphaUnitsRepairAction =
        "declare the α units explicitly: add \"alphaInNormUnits\": true "
        + "(α in residual-stream-norm units — the project convention) or "
        + "false (raw α) to the condition, or declare the arm with "
        + "`steerlab-cli experiment declare-condition <study> <condition> "
        + "--slots <concept>:<layer>:<alpha> --alpha-units norm|raw`"

    /// Every key a condition document must carry, as a caller would type it.
    static let conditionRequiredKeys = ["alphaInNormUnits", "bandWidth", "name", "slots"]

    /// A condition document this engine cannot read, as a TYPED refusal that
    /// names the arm, the key, and the repair (Phase-0 gaps G4 + G6).
    ///
    /// `DecodingError` is a debug dump whose only actionable content — which
    /// key, in which array element — is buried in `codingPath`; the same
    /// reading `MultiAgentScenarioStore.decodeFailureReason` does for
    /// scenarios.
    /// `raw` is the same `conditions` array re-decoded opaquely, so the arm
    /// can be named by NAME rather than only by index.
    static func conditionDecodeRefusal(
        _ error: DecodingError, raw: [JSONValue]
    ) -> ExperimentError {
        func context(_ error: DecodingError) -> DecodingError.Context? {
            switch error {
            case let .keyNotFound(_, context): context
            case let .typeMismatch(_, context): context
            case let .valueNotFound(_, context): context
            case let .dataCorrupted(context): context
            @unknown default: nil
            }
        }
        let index = context(error)?.codingPath.compactMap(\.intValue).first
        let name: String? = index.flatMap { position in
            guard position < raw.count, case let .object(fields) = raw[position],
                case let .string(value)? = fields["name"], !value.isEmpty
            else { return nil }
            return value
        }
        let arm =
            name.map { "condition '\($0)'" }
            ?? index.map { "conditions[\($0)]" }
            ?? "a condition"

        guard case let .keyNotFound(key, _) = error else {
            return .malformed(
                "\(arm) is malformed: "
                    + MultiAgentScenarioStore.decodeFailureReason(error),
                repair: "repair the condition in the manifest document — it "
                    + "must be {\"name\": …, \"slots\": [{\"concept\": …, "
                    + "\"layer\": …, \"alpha\": …}, …], \"bandWidth\": 1, "
                    + "\"alphaInNormUnits\": true|false}")
        }
        if key.stringValue == "alphaInNormUnits" {
            return .malformed(
                "\(arm) declares no 'alphaInNormUnits', so the α it names has "
                    + "no unit — this engine would read residual-norm units "
                    + "and the server engine raw α for the same document, "
                    + "which is a different intervention at the same number",
                repair: alphaUnitsRepairAction)
        }
        return .malformed(
            "\(arm) is missing the required key '\(key.stringValue)'",
            repair: "every condition declares "
                + conditionRequiredKeys.map { "'\($0)'" }.joined(separator: ", ")
                + " — add '\(key.stringValue)', or declare the arm through "
                + "`steerlab-cli experiment declare-condition`, which writes "
                + "the whole shape for you")
    }

    /// Layer/alpha dose-response grid (alphas in residual-norm units; 0 is
    /// the baseline cell and is always implied). Default alphas recalibrated
    /// 2026-06-10 with the corrected norm-unit conversion (injected norm =
    /// α·r exactly): on the French/Qwen3-4B shakedown, expression appears
    /// around α≈0.3 and capability collapse above α≈1 — the pre-fix default
    /// [0.03, 0.06, 0.1] was implicitly scaled by ‖v‖ and is sub-threshold
    /// in honest units. Recalibrated 2026-07-09 (researcher decision,
    /// live-testing): [0.1, 0.2, 0.4] steers too hard as a STARTING grid.
    /// Recalibrated again 2026-07-14 (researcher decision, live-testing):
    /// stronger alphas routinely push models into wasteful incoherence, and
    /// the live optimum sits late in the network — L28/α0.08 on gemma-3-4b
    /// (≈0.82 depth) lies inside the new grid. Defaults are now depth
    /// fractions [0.5, 0.7, 0.85] × alphas [0.05, 0.08, 0.1, 0.13] on both
    /// engines (server twin: tasks.py `_sweep_with_spec` fallback — keep
    /// identical); stronger grids remain one edit away in the spec editor.
    /// Fractions resolve against the model's layer count at sweep time via
    /// `resolvedLayers(layerCount:)`.
    public struct SweepSpec: Codable, Sendable, Equatable {
        public var layerFractions: [Double]
        public var alphas: [Double]
        public var devPromptsFile: String
        public var batteryFile: String
        public var maxTokens: Int
        /// The declared selection criterion; absent resolves to the
        /// historical defaults (markerDensity, tolerance 0.15, floor 0.45,
        /// no control) via `SweepSelectionRule.resolve`.
        public var selection: SweepSelection?
        /// Sweep-input pins (cross-engine contract keys "devPromptsHash" +
        /// "batteryHash", firewall closure 2026-07-20): SHA-256 over the
        /// raw bytes of `devPromptsFile` / `batteryFile` — the inputs the
        /// sweep SELECTS on, pinned at FREEZE when absent (never silently
        /// re-pinned). Drift after pinning is a verify() violation, and
        /// sweep start refuses to select on drifted inputs — which is also
        /// what keeps these pins in agreement with the ex-post provenance
        /// stamp (`SelectionProvenance.devPromptsHash`). Optional +
        /// omit-when-nil so legacy manifests keep their content hash.
        public var devPromptsHash: String?
        public var batteryHash: String?

        public init(
            layerFractions: [Double] = [0.5, 0.7, 0.85],
            alphas: [Double] = [0.05, 0.08, 0.1, 0.13],
            devPromptsFile: String = "prompts/dev/dev-prompts.jsonl",
            batteryFile: String = "prompts/batteries/basic.jsonl",
            maxTokens: Int = 80,
            selection: SweepSelection? = nil,
            devPromptsHash: String? = nil,
            batteryHash: String? = nil
        ) {
            self.layerFractions = layerFractions
            self.alphas = alphas
            self.devPromptsFile = devPromptsFile
            self.batteryFile = batteryFile
            self.maxTokens = maxTokens
            self.selection = selection
            self.devPromptsHash = devPromptsHash
            self.batteryHash = batteryHash
        }

        /// Depth-fraction → block-index resolution, shared by the sweep run
        /// loop and tests. Truncating (`Int(count·f)`), clamped to a valid
        /// block, deduplicated, sorted — the same rule the server applies in
        /// `_sweep_with_spec` (`int(layer_count * f)`), so a fraction grid
        /// names the same cells on both engines for a given layer count.
        public func resolvedLayers(layerCount: Int) -> [Int] {
            Set(
                layerFractions.map {
                    min(layerCount - 1, max(0, Int(Double(layerCount) * $0)))
                }
            ).sorted()
        }
    }

    public struct EvaluationSpec: Codable, Sendable, Equatable {
        public enum Kind: String, Codable, Sendable {
            case none
            case pairedJudge
        }

        public var kind: Kind
        public var judgeModel: String
        public var judgePrompt: String
        public var structuredPrompt: String?

        public init(
            kind: Kind = .none,
            judgeModel: String = "claude-opus-4-8",
            judgePrompt: String = "",
            structuredPrompt: String? = nil
        ) {
            self.kind = kind
            self.judgeModel = judgeModel
            self.judgePrompt = judgePrompt
            self.structuredPrompt = structuredPrompt
        }
    }

    /// One judge in the evaluation panel. `kind` is "claude" (Anthropic
    /// API), "openrouter" (any OpenRouter-served model, provider-pinned),
    /// or "local" (an MLX model loaded on this substrate); `model` is the
    /// model id for that kind. nil/empty model (cross-engine rule,
    /// 2026-07-08): a claude judge uses the default Claude judge model; a
    /// LOCAL judge uses the STUDY model (manifest.modelID) — it judges with
    /// the same model that generated the outputs. OpenRouter judges have NO
    /// defaults: an explicit model slug AND a pinned `provider` are
    /// required — the same slug can be served by different backends with
    /// different outputs, so an unpinned provider is not a pinned judge.
    /// JSON keys match the server's judge entries exactly:
    /// {"name", "kind", "model", "provider", "revision", "dtype"} — the
    /// last two are LOCAL-judge pins (2026-07-23), omit-when-nil so legacy
    /// manifests keep their bytes. A blank local-judge revision is pinned
    /// from the STUDY pin at freeze when the judge resolves to the study
    /// model; `dtype` is honored by the server engine (the MLX loader takes
    /// no dtype — carried for cross-engine round-trip stability).
    public struct JudgeRef: Codable, Sendable, Equatable {
        public var name: String
        public var kind: String
        public var model: String?
        /// OpenRouter judges only: the pinned serving provider.
        public var provider: String?
        /// Local judges: the pinned model revision (commit hash).
        public var revision: String?
        /// Local judges: loader dtype where the loader takes one (server).
        public var dtype: String?

        public init(name: String, kind: String, model: String? = nil,
                    provider: String? = nil, revision: String? = nil,
                    dtype: String? = nil) {
            self.name = name
            self.kind = kind
            self.model = model
            self.provider = provider
            self.revision = revision
            self.dtype = dtype
        }

        /// This judge with only the fields its KIND owns (field bug
        /// 2026-08-07): switching a row's kind in the Studies panel used to
        /// carry the previous kind's fields into the manifest — a local
        /// judge kept `provider` from its OpenRouter past, a claim about a
        /// pin that does not exist for that kind. Ownership follows the
        /// cross-engine schema above: local → model/revision/dtype;
        /// openrouter → model/provider (revision and dtype are LOCAL-judge
        /// pins on both engines); claude → model. A blank kind resolves to
        /// claude, matching `resolvedJudgeIdentity`; an unrecognized kind
        /// keeps every field — this build cannot know what it owns, and
        /// destroying data is worse than carrying it.
        public func keepingKindOwnedFields() -> JudgeRef {
            var kept = self
            let trimmed = kind.trimmingCharacters(in: .whitespacesAndNewlines)
            switch trimmed.isEmpty ? "claude" : trimmed {
            case "local":
                kept.provider = nil
            case "openrouter":
                kept.revision = nil
                kept.dtype = nil
            case "claude":
                kept.provider = nil
                kept.revision = nil
                kept.dtype = nil
            default:
                break
            }
            return kept
        }
    }

    /// A pinned human-effect table (CSV) for the alien-residual computation
    /// R = delta_model − delta_human. Pinned by hash like every other input.
    public struct HumanBaseline: Codable, Sendable, Equatable {
        public var path: String
        public var hash: String

        public init(path: String, hash: String) {
            self.path = path
            self.hash = hash
        }
    }

    /// A pinned RepE reader artifact (measurement instrument) used as the
    /// `repeReaderScore` outcome instrument. Pinned by file hash like
    /// `humanBaseline`; additionally substrate-gated in `verify()` because a
    /// reader fitted on another engine's activations measures nothing on this
    /// one. JSON shape matches the server's `ReaderRef` exactly.
    public struct ReaderRef: Codable, Sendable, Equatable {
        public var path: String
        public var hash: String
        public var concept: String

        public init(path: String, hash: String, concept: String) {
            self.path = path
            self.hash = hash
            self.concept = concept
        }
    }

    /// Screen→confirm promotion gate (study-guide funnel). All criteria must
    /// hold for a concept to enter the confirm phase.
    public struct PromotionRule: Codable, Sendable, Equatable {
        public var fdrThreshold: Double?
        public var doseMonotone: Bool?
        public var exceedsRandomFloor: Bool?
        public var capabilityGate: String?

        public init(
            fdrThreshold: Double? = nil,
            doseMonotone: Bool? = nil,
            exceedsRandomFloor: Bool? = nil,
            capabilityGate: String? = nil
        ) {
            self.fdrThreshold = fdrThreshold
            self.doseMonotone = doseMonotone
            self.exceedsRandomFloor = exceedsRandomFloor
            self.capabilityGate = capabilityGate
        }
    }

    /// Confirmation-study perturbation policy (cross-engine contract with the
    /// server's `confirmation.py`): the DECLARED rule that expanded into this
    /// manifest's `<agent>-anchor` / `<agent>-minus-δ` / `<agent>-plus-δ` /
    /// `<agent>-control` conditions at AUTHORING time. The expansion is
    /// mechanical (`ConfirmationStudy.expandedConditions`), so the frozen
    /// manifest shows exactly what will run and the pin/freeze/verify
    /// firewall applies unchanged — never hand-picked post-hoc points.
    /// Participates in the content hash (NOT a volatile lifecycle stamp);
    /// optional so legacy manifests keep their content hash.
    public struct PerturbationPolicy: Codable, Sendable, Equatable {
        /// Provenance of the agent under confirmation. `promoted` records
        /// whether the artifact carried a sweep-promotion birth certificate —
        /// hand-created agents are allowed (freeze advisories already surface
        /// them); this field just makes the provenance honest.
        public struct SourceAgent: Codable, Sendable, Equatable {
            public var name: String
            /// Workspace-relative path to the variant artifact JSON.
            public var artifactPath: String
            /// SHA-256 of the artifact file bytes (same convention as
            /// `VariantCondition.artifactHash` / `ModelVariantStore.hash`).
            public var artifactHash: String
            public var promoted: Bool

            public init(
                name: String, artifactPath: String, artifactHash: String,
                promoted: Bool
            ) {
                self.name = name
                self.artifactPath = artifactPath
                self.artifactHash = artifactHash
                self.promoted = promoted
            }
        }

        public var sourceAgent: SourceAgent
        public var concept: String
        /// The agent's anchor cell (alpha in residual-norm units).
        public var cell: SelectionProvenance.Cell
        /// Symmetric offsets: each δ>0 expands to α−δ and α+δ. The whole
        /// operation refuses when any α−δ ≤ 0.
        public var alphaDeltas: [Double]
        public var includeMatchedNormControl: Bool
        public var declaredAt: String

        public init(
            sourceAgent: SourceAgent,
            concept: String,
            cell: SelectionProvenance.Cell,
            alphaDeltas: [Double],
            includeMatchedNormControl: Bool,
            declaredAt: String
        ) {
            self.sourceAgent = sourceAgent
            self.concept = concept
            self.cell = cell
            self.alphaDeltas = alphaDeltas
            self.includeMatchedNormControl = includeMatchedNormControl
            self.declaredAt = declaredAt
        }
    }

    /// Where a template-instantiated study came from: which study template
    /// supplied every setting the study did not choose for itself, at which
    /// content hash, and (for batch mints) which batch it belongs to.
    ///
    /// This DELIBERATELY participates in the manifest content hash. It is not
    /// a lifecycle stamp like `gitCommit`: two studies whose settings are
    /// byte-identical but which descend from different templates are
    /// different preregistrations, and a run that stamps one must not verify
    /// against the other. It is also what makes "load this study back as a
    /// template" answerable without guessing — `StudyTemplateStore` compares
    /// the study's stripped form against the hash recorded here to tell an
    /// unchanged instance from a diverged one.
    ///
    /// Optional + omit-when-nil, so every manifest authored before templates
    /// existed decodes unchanged and keeps its content hash. NOT part of the
    /// cross-engine contract: the Python engine does not know the key, so a
    /// server-side re-save of a manifest drops it (the `JudgeRef.provider`
    /// failure mode). Instantiation is a Mac-authoring act and the stamp is
    /// provenance rather than a pin, so a dropped stamp costs lineage, never
    /// measurement — but re-mint on the Mac if the server has rewritten the
    /// manifest.
    public struct TemplateProvenance: Codable, Sendable, Equatable {
        /// The template's directory name under the workspace's `templates/`.
        public var template: String
        /// `StudyTemplateStore.hash` of the template at instantiation time.
        public var templateHash: String
        /// Set only on studies minted together by `instantiateBatch`: an id
        /// shared by every sibling of that mint, so analysis can group a
        /// composition sweep or a permutation set back together. Sibling
        /// studies are the design for panels (one scenario per study is
        /// hard-wired in both engines), so the batch id is the ONLY thing
        /// tying those siblings to one another.
        public var batchGroup: String?

        public init(
            template: String, templateHash: String, batchGroup: String? = nil
        ) {
            self.template = template
            self.templateHash = templateHash
            self.batchGroup = batchGroup
        }
    }

    /// A model-variant condition pins a full reusable model configuration:
    /// base model, optional adapter, injections, prompt mode/system prompt,
    /// and neutral-basis choices. The artifact is snapshotted into the
    /// manifest and also checked by hash so edits to the library do not
    /// silently change a frozen study.
    public struct VariantCondition: Codable, Sendable, Equatable {
        /// Forward reference (seamless pipeline stage 4, server-resolved):
        /// "the agent this experiment's sweep promotes for CONCEPT under
        /// the declared criterion" — declarable (and freezable) before the
        /// agent exists. The SERVER resolves it at run time and records
        /// the pin in the run directory; this engine carries the
        /// declaration verbatim (an app re-save must never drop it — the
        /// JudgeRef.provider lesson).
        public struct FromPromotion: Codable, Sendable, Equatable {
            public var concept: String

            public init(concept: String) {
                self.concept = concept
            }
        }

        /// Training provenance of a TRAINED ADAPTER arm (cross-engine
        /// contract key `trainingProvenance`; cluster-LoRA readiness §0
        /// amendment 1 + contract §9).
        ///
        /// Adapters enter studies as variants, so the manifest is where a
        /// trained adapter's provenance becomes evidence: the training
        /// dataset manifest joins the freeze `verify()` pin surface (drift
        /// after freeze is a violation, exactly like stimulus drift), and
        /// the matched control — amendment 2 — is DECLARED here, ex ante,
        /// before training, never derived at qualification time.
        public struct TrainingProvenance: Codable, Sendable, Equatable {
            /// The neutralized/shuffled-label control arm this adapter is
            /// measured against, named before training.
            public struct MatchedControl: Codable, Sendable, Equatable {
                /// The control variant's name.
                public var variant: String
                /// How the construct was neutralized, e.g.
                /// `shuffledAssistantPairing`.
                public var kind: String

                public init(variant: String, kind: String) {
                    self.variant = variant
                    self.kind = kind
                }
            }

            public var datasetBundleID: String?
            /// Workspace-relative path of the dataset manifest (the Mac
            /// workspace is the source of truth, so this file IS local even
            /// when the adapter was trained on the cluster).
            public var datasetManifestPath: String?
            public var datasetManifestHash: String?
            /// SHA-256 of the adapter's training sidecar JSON bytes. A
            /// SERVER-side artifact: this engine verifies it only when the
            /// file happens to be present locally (see `verify`).
            public var adapterSidecarHash: String?
            public var evidenceGrade: Bool?
            public var matchedControl: MatchedControl?

            public init(
                datasetBundleID: String? = nil,
                datasetManifestPath: String? = nil,
                datasetManifestHash: String? = nil,
                adapterSidecarHash: String? = nil,
                evidenceGrade: Bool? = nil,
                matchedControl: MatchedControl? = nil
            ) {
                self.datasetBundleID = datasetBundleID
                self.datasetManifestPath = datasetManifestPath
                self.datasetManifestHash = datasetManifestHash
                self.adapterSidecarHash = adapterSidecarHash
                self.evidenceGrade = evidenceGrade
                self.matchedControl = matchedControl
            }
        }

        public var name: String
        public var artifactPath: String
        public var artifactHash: String
        public var artifact: ModelVariantArtifact
        public var fromPromotion: FromPromotion?
        /// Absent on every non-adapter arm and on every manifest written
        /// before the LoRA readiness work — encoded only when present, so
        /// legacy manifest bytes (and their hashes) are unchanged.
        public var trainingProvenance: TrainingProvenance?

        public init(
            name: String,
            artifactPath: String,
            artifactHash: String,
            artifact: ModelVariantArtifact,
            fromPromotion: FromPromotion? = nil,
            trainingProvenance: TrainingProvenance? = nil
        ) {
            self.name = name
            self.artifactPath = artifactPath
            self.artifactHash = artifactHash
            self.artifact = artifact
            self.fromPromotion = fromPromotion
            self.trainingProvenance = trainingProvenance
        }

        enum CodingKeys: String, CodingKey {
            case name, artifactPath, artifactHash, artifact, fromPromotion
            case trainingProvenance
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            fromPromotion = try container.decodeIfPresent(
                FromPromotion.self, forKey: .fromPromotion)
            trainingProvenance = try container.decodeIfPresent(
                TrainingProvenance.self, forKey: .trainingProvenance)
            // A forward-referenced condition carries NO artifact keys —
            // default them instead of failing the whole manifest decode.
            artifactPath =
                try container.decodeIfPresent(String.self, forKey: .artifactPath) ?? ""
            artifactHash =
                try container.decodeIfPresent(String.self, forKey: .artifactHash) ?? ""
            artifact = try container.decodeIfPresent(
                ModelVariantArtifact.self, forKey: .artifact)
                ?? ModelVariantArtifact(
                    name: "", baseModelID: "", promptMode: "",
                    qwenThinkingEnabled: false, temperature: 0,
                    systemPrompt: "")
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(name, forKey: .name)
            if let fromPromotion {
                // Byte-faithful round-trip of the server's shape: a
                // forward reference has exactly ONE identity — writing a
                // placeholder artifact would make the server flag "both
                // identities declared".
                try container.encode(fromPromotion, forKey: .fromPromotion)
                return
            }
            try container.encode(artifactPath, forKey: .artifactPath)
            try container.encode(artifactHash, forKey: .artifactHash)
            try container.encode(artifact, forKey: .artifact)
            try container.encodeIfPresent(
                trainingProvenance, forKey: .trainingProvenance)
        }
    }

    public var name: String
    public var experimentDescription: String
    /// Domain-neutral task/protocol note: what the model, agent, or
    /// institution is asked to do. Judicial opinion writing is one possible
    /// task, not a privileged assumption in ExperimentKit.
    public var taskDescription: String?
    /// Domain-neutral measurement note: what outcomes make the protocol
    /// succeed, fail, flip, degrade, or otherwise move.
    public var outcomeMeasures: String?
    /// JSONL prompt/task file to execute for measured study runs. Lines are
    /// `{"text": "..."}`; domain-specific schemas can be layered later.
    public var taskPromptsFile: String?
    /// SHA-256 over `taskPromptsFile`, pinned so the measured task cannot
    /// drift silently after a protocol is built.
    public var taskPromptsHash: String?
    public var studyKind: StudyKind
    /// The researcher's declared study type (the top-of-page picker's
    /// vocabulary: conceptStudy | agentComparison |
    /// multiAgent) — DURABLE authoring intent, so a baseline-only
    /// comparison stays "Compare agents" across selection changes instead
    /// of re-deriving from content. Optional: absent manifests derive
    /// from content and keep their hash. `studyKind` remains the
    /// engine-facing run-path switch; the store setter keeps them
    /// consistent.
    public var studyType: String?
    public var multiAgentScenarioPath: String?
    public var multiAgentScenarioHash: String?
    /// The SEMANTIC scenario the pinned scenario was compiled from, when this
    /// study's seats were cast rather than hand-bound (`SeatCasting`).
    ///
    /// PROVENANCE, not a pin — the same standing as `templateProvenance`, and
    /// for the same reason. What a run reads is `multiAgentScenarioPath`: the
    /// compiled file, with every seat bound. This records where that casting
    /// came from, so reopening the study re-lists ITS seats and the picker can
    /// show the scenario a researcher actually chose (a compiled file is
    /// deliberately absent from the scenario library). Nothing verifies it,
    /// nothing gates on it, and a drifted semantic scenario is an advisory in
    /// the Seats section — never a verify violation, because the study runs
    /// the compiled bytes either way.
    ///
    /// Optional + omit-when-nil so every manifest written before seat casting
    /// existed decodes unchanged and keeps its content hash.
    public var multiAgentSemanticScenarioPath: String?
    public var multiAgentSemanticScenarioHash: String?
    public var multiAgentIncludeBaseline: Bool
    public var createdAt: String
    public var modelID: String
    public var modelRevision: String?
    /// The numeric precision this study's model is pinned to run in
    /// (`bfloat16`/`float16`/`float32`, or the bf16/fp16/fp32 aliases). Nil
    /// means "let the device decide", which is what every study did before
    /// this key existed.
    ///
    /// Greedy decoding is not precision-proof: at a near-tie between two
    /// tokens, bf16 and fp16 round differently, the argmax flips, and the
    /// continuation diverges. The same argument that requires a JUDGE to
    /// pin its dtype applies at least as strongly to the model that
    /// produced the text being judged.
    ///
    /// SERVER-HONORED, validated-but-unconsumed here — the same shape as
    /// `JudgeRef.dtype`. MLX study models are quantized repos with no loader
    /// dtype to set, but the Mac is the AUTHORING surface, so freeze checks
    /// the pin here: a manifest must not reach the cluster carrying a dtype
    /// that refuses at load after a queue wait. Optional so legacy manifests
    /// decode unchanged and keep their content hash.
    public var dtype: String?
    public var concepts: [ConceptRef]
    /// Pinned when any concept uses grand-mean extraction. Optional so
    /// legacy manifests decode unchanged and keep their content hash.
    public var grandMeanCorpus: GrandMeanCorpus?
    public var conditions: [Condition]
    public var variantConditions: [VariantCondition]
    /// Pinned when any concept uses confound projection.
    public var neutralCorpusHash: String?
    public var sweep: SweepSpec?
    /// The chain-runner block (server stage 3, 2026-07-18) — stages + gates
    /// the `pipeline` verb runs. Data-only PASSTHROUGH on this engine: the
    /// server resolves and enforces it (`pipeline_spec.py`); Swift carries
    /// it verbatim so an app re-save never destroys a server-authored
    /// chain (the JudgeRef.provider lesson, same day).
    public var pipeline: JSONValue?
    /// Server-authored J-lens readout declaration, carried VERBATIM.
    ///
    /// Same reason `pipeline` is opaque here: this engine cannot produce the
    /// block — imported lens artifacts are PyTorch/HF-native, so J-lens work is
    /// server-only by rule — but it can load, duplicate, and re-save a manifest
    /// that has one. Without a passthrough the field is absent from
    /// `CodingKeys`, so `duplicate`'s round-trip would silently drop a
    /// scientific pin: the copy would declare no readout, freeze cleanly, and
    /// measure nothing, with no warning anywhere. Decoded as `JSONValue` rather
    /// than a mirrored struct so the server can extend the block without this
    /// engine needing to learn each new key.
    public var jlensReadout: JSONValue?
    /// Retain the exact sampled token ids on every generation record
    /// (cross-engine contract key "recordTokenIDs"; server twin
    /// `Manifest.record_token_ids`). Authored and consumed on the SERVER —
    /// this engine neither reads nor writes token ids — but carried verbatim
    /// so a Mac-side duplicate, edit, or re-save cannot silently drop a
    /// study's declaration that its runs must stay replayable. Absent
    /// decodes as `false`, which is the historical behaviour of every
    /// manifest written before 2026-08-15.
    public var recordTokenIDs: Bool = false
    /// SAE candidate-roster pin, carried VERBATIM (cross-engine contract key
    /// "saeCandidates" = `{"path": …, "hash": …}`; server twin
    /// `sae_candidates.pin_violations`).
    ///
    /// The roster records WHICH SAE features a study may seat and the
    /// discovery evidence behind each nomination, so it is a measurement-side
    /// input like `markersHash` or the reasoning-style taxonomy. The block is
    /// authored on the server (import and qualification are server verbs by
    /// rule), but this engine loads, duplicates and re-saves the manifests
    /// that carry it — and a dropped pin here would let a duplicate freeze
    /// cleanly while claiming no roster at all.
    ///
    /// Opaque on decode, but NOT unchecked: `verify()` re-checks the pin
    /// mechanically (path present, workspace-relative, file there, SHA-256 of
    /// its bytes unchanged). The roster's SCHEMA is deliberately not
    /// validated here — those are server-only semantics, and a second
    /// validator drifts from the first by construction.
    public var saeCandidates: JSONValue?
    /// Preregistered cap on how many SAE features one mixture condition may
    /// seat (cross-engine key "maxSAEMixtureFeatures"), carried VERBATIM.
    /// The server enforces it; this engine must not silently discard the
    /// declaration, because a lost cap reads downstream as "the default was
    /// declared" rather than "a cap was chosen".
    public var maxSAEMixtureFeatures: JSONValue?
    /// SAE LATENT intervention arms (cross-engine key "saeLatentConditions"),
    /// carried VERBATIM. A distinct mechanism from decoder-direction
    /// addition — encode → clamp/add latent → decode — and deliberately a
    /// SEPARATE top-level list so it can never execute as an ordinary
    /// steering condition. Server-only to run; opaque and preserved here.
    public var saeLatentConditions: JSONValue?
    public var evaluation: EvaluationSpec?
    // Judge-rubric versioning (evidence tier). All optional so pre-existing
    // manifests decode unchanged and keep their content hash.
    /// Rubric file under prompts/rubrics/, relative to the project root.
    /// Freezing a judge-evaluated study requires this pin — inline
    /// `evaluation.judgePrompt` text is a draft-only convenience.
    public var judgeRubricFile: String?
    /// SHA-256 over the rubric file's raw bytes.
    public var judgeRubricHash: String?
    /// The judge panel: >=2 required at freeze for judge-evaluated studies
    /// so agreement statistics (percent agreement, Cohen's kappa) exist.
    public var judges: [JudgeRef]?
    /// Optional pinned human-judgment subset; when present the evaluation
    /// report adds per-judge vs-human agreement.
    public var humanValidation: HumanBaseline?
    /// Capability-battery-as-evidence: the battery `validate` runs through
    /// every variant condition (and baseline). Defaults to the same battery
    /// VariantRobustness's default preset uses when unpinned at validate.
    public var capabilityBatteryFile: String?
    public var capabilityBatteryHash: String?
    /// Combined pin over every attached concept's `markers.json` (the
    /// scoring rubrics the run/sweep marker densities read). Pinned at
    /// freeze from the resolved files — see `liveMarkersHash` for the exact
    /// cross-engine hash construction. nil = legacy manifest or no markers
    /// on disk. Optional + encodeIfPresent so pre-existing manifests decode
    /// unchanged and keep their content hash.
    public var markersHash: String?
    /// Reasoning-style taxonomy pin (cross-engine contract keys
    /// "reasoningStyleTaxonomyPath" + "reasoningStyleTaxonomyHash"): a
    /// versioned feature file under `prompts/taxonomies/`, pinned at set
    /// time (`experiment set-style-taxonomy`) by SHA-256 of its raw bytes.
    /// Drift after pinning is a verify() violation like every other
    /// measurement-side input; ABSENT (both nil) = no reasoning-style
    /// scoring, no violation. Optional + encodeIfPresent so pre-existing
    /// manifests decode unchanged and keep their content hash.
    public var reasoningStyleTaxonomyPath: String?
    public var reasoningStyleTaxonomyHash: String?
    // Science-layer fields (alien-stance program; Phase D). All optional —
    // pre-Phase-D manifests must decode unchanged and, because synthesized
    // encoding omits nil optionals, keep their content hash.
    /// shakedown | screen | confirm | triangulate | panel
    public var phase: String?
    /// A provenance LABEL. Free text, decoded from every manifest that
    /// carries it, printed in report/preregistration output — and
    /// behaviorless, with one DEPRECATED exception: the value `"sentencing"`
    /// still implicitly selects the built-in duration endpoint where no
    /// `numericParser` is declared. That trigger keeps working for manifests
    /// that already depend on it (2026-08-18) and now announces itself at
    /// every site where it fires — see
    /// `ExperimentManifest.implicitCaseFamilyAdvisory`. Declare
    /// `numericParser` instead; the workspace registry's shipped
    /// `sentencing-months` entry reproduces the built-in parser exactly.
    public var caseFamily: String?
    /// e.g. ["answerTokenLogprob", "sampledText"]; nil/empty = sampledText only.
    public var outcomeInstruments: [String]?
    /// REQUIRED companion of `outcomeInstruments: ["ordinalScale"]`: how the
    /// renormalized probability distribution over the item's declared option
    /// ladder collapses to one position — "expectedValue"
    /// (probability-weighted mean of ladder positions 1..K) or "argmax"
    /// (position of the maximum-probability option). An instrument-design
    /// choice, so verify refuses ordinalScale without it — declared, never
    /// silently defaulted. Optional so legacy manifests decode unchanged and
    /// keep their content hash.
    public var ordinalAggregation: String?
    /// Declared, hash-pinned applicability subset for the option-consuming
    /// instruments (cross-engine contract key "outcomeInstrumentScope").
    ///
    /// A mixed task-prompts file may hold a label arm the answer-token
    /// instruments CAN read and a JSON arm they cannot (see
    /// `ResponseFormat`). Measuring only part of a file is legitimate, but
    /// "which rows were measured" is a result-bearing fact, so it must be
    /// DECLARED rather than inferred — and pinned, so the subset is
    /// checkable after the fact instead of recomputed from whatever the file
    /// says later. ABSENT = the instrument applies to every item, which is
    /// today's behavior exactly. Optional so existing manifests decode
    /// unchanged and keep their content hash.
    public var outcomeInstrumentScope: ResponseFormat.Scope?

    /// The study's declared EVALUATION SAMPLING DESIGN (cross-engine contract
    /// key "evaluationSampling"): how many records per condition the judged
    /// coding preregistered, and the seed that draws them.
    ///
    /// The seeded evaluate subsample shipped as CLI flags and run stamps
    /// (2026-08-29). A stamp records what HAPPENED; "preregistered" is a
    /// claim about what was decided BEFORE anything ran, and a claim like
    /// that has to live in the artifact chain to be evidence. Declaring it
    /// here is what puts it there: every run writes the manifest snapshot
    /// into its own `experiment.json`, so the design travels with the
    /// evidence instead of only with the command line that produced it.
    ///
    /// `evaluate` then needs no flags at all, and a flag that DISAGREES with
    /// this refuses (`EvaluateSubsample.reconcile`) — the flags become a
    /// cross-check on a declared study, never an override. ABSENT = no
    /// declared design, which is the flags-only path exactly as it was, and
    /// is what every manifest written before this holds. Optional so those
    /// manifests decode unchanged and keep their content hash.
    public var evaluationSampling: EvaluateSubsample.Declaration?

    /// A discriminant-validity CONTROL concept: a direction the study's own
    /// concepts must NOT collapse into.
    ///
    /// Before 2026-07-26 Swift built its cross-concept cosine matrix from
    /// "every other concept on disk", extracting each with the FIRST pinned
    /// paired concept's options. Two things were wrong with that. The
    /// control SET was ambient — it changed whenever unrelated work landed in
    /// the workspace, so the same manifest produced different discriminant
    /// evidence on two machines, and `worstCosinePair` was not a property of
    /// the study at all. And the control RECIPE was borrowed, so a control
    /// authored for grand-mean extraction was silently read at the wrong
    /// position with the wrong method. Python meanwhile had no controls, so
    /// the two engines disagreed about what validate even measures.
    ///
    /// A control is therefore a COMPLETE PINNED RECIPE REFERENCE: which
    /// concept, which stimulus bytes, and its OWN extraction options.
    /// Declared data, pinned by the manifest hash like everything else.
    public struct ValidationControl: Codable, Sendable, Equatable {
        public var concept: String
        /// SHA-256 of the control's stimulus set — drift refuses validate,
        /// exactly as a pinned concept's does.
        public var stimulusSetHash: String
        /// The control's OWN recipe. Never inherited from a study concept:
        /// a borrowed method reads the control at a position it was not
        /// authored for, and the resulting cosine says nothing.
        public var options: ExtractionOptions
        /// DEPRECATED and inoperative — nothing reads it, and nothing should.
        ///
        /// The cosine matrix compares both vectors of every cell at ONE
        /// layer, because the residual stream drifts with depth: the same
        /// concept a few layers apart can be near-orthogonal to itself, so a
        /// cosine spanning two depths conflates "different concepts" with
        /// "different depths". A per-control layer could therefore only be
        /// ignored (what happens) or honoured — and honouring it would break
        /// that invariant and manufacture false discriminant validity.
        ///
        /// Kept in the type, and still ENCODED, purely so a manifest that
        /// carries it round-trips byte-identically and keeps its content
        /// hash. `verify()` reports it so it is removed deliberately rather
        /// than silently preserved — a field that exists only to be
        /// round-tripped is the `responseFormat` mistake again.
        public var validationLayer: Int?
        /// The model revision this control's recipe assumes. Must equal the
        /// manifest's when declared — a control extracted from a different
        /// revision is not comparable to the study's directions.
        public var modelRevision: String?

        public init(
            concept: String, stimulusSetHash: String,
            options: ExtractionOptions, validationLayer: Int? = nil,
            modelRevision: String? = nil
        ) {
            self.concept = concept
            self.stimulusSetHash = stimulusSetHash
            self.options = options
            self.validationLayer = validationLayer
            self.modelRevision = modelRevision
        }
    }

    /// Declared discriminant-validity controls (cross-engine contract key
    /// "validationControls"). ABSENT = no controls, which is what the Python
    /// engine has always done; the ambient disk scan it replaces is gone.
    /// Validate emits a loud advisory naming any undeclared concepts on disk
    /// so the change is never silent.
    public var validationControls: [ValidationControl]?
    /// The layer convergent validity reads at, as an absolute index (D4).
    ///
    /// The historical rule made this a SIDE EFFECT of the injection
    /// conditions ("the layer a condition steers this concept at, else
    /// mid-network"), so moving the validation read meant editing steering
    /// conditions — a different decision entirely. ABSENT = that legacy rule
    /// exactly, so existing manifests keep their numbers and their content
    /// hash. Mutually exclusive with `validationLayerFraction`.
    public var validationLayer: Int?
    /// The same decision as a depth FRACTION (0…1), for studies that should
    /// read at the same relative depth across model sizes. Resolved with the
    /// truncating, clamped rule `SweepSpec.resolvedLayers` uses, so a
    /// fraction means one thing across the app.
    public var validationLayerFraction: Double?
    /// The same decision as a LIST of absolute indices — one validate run
    /// measures every declared depth (the scenario activations are captured
    /// once for all layers, so extra depths are free). Exists for the
    /// validate-at-the-sweep-layers policy: the reading certificate should
    /// cover every layer the sweep may promote. Exactly one of the four
    /// depth fields may be declared (`ValidationLayerRule.violation`).
    public var validationLayers: [Int]?
    /// The list form of `validationLayerFraction` — same resolution rule per
    /// entry, same exactly-one-of-four exclusivity.
    public var validationLayerFractions: [Double]?
    /// Declared numeric-answer parser (cross-engine contract keys
    /// "numericParser" + "parserRegistryHash"): the name of an entry in the
    /// workspace parser registry (`ParserRegistry.registryFile`) that parses
    /// this study's numeric outcome instead of the built-in duration
    /// parser. Freeze pins the registry file's SHA-256; drift after pinning
    /// is a verify() violation. ABSENT (both nil) = the historical behavior
    /// (caseFamily == "sentencing" → built-in parseMonths), no violation, no
    /// advisory. Optional so pre-existing manifests decode unchanged and
    /// keep their content hash.
    public var numericParser: String?
    public var parserRegistryHash: String?

    /// The one `caseFamily` value that still selects a measurement
    /// instrument. The ONE deprecated implicit selection left in the manifest
    /// vocabulary. Python twin: `manifest.IMPLICIT_ENDPOINT_CASE_FAMILY`.
    public static let implicitEndpointCaseFamily = "sentencing"

    /// What every site that fires the deprecated trigger says, on both engines
    /// and at every site, byte for byte. ONE sentence to match on: an agent
    /// that has to learn four spellings of the same deprecation has not been
    /// told anything. Python twin:
    /// `manifest.IMPLICIT_CASE_FAMILY_ADVISORY`.
    ///
    /// Advisory, never a refusal — the whole point of the deprecation is that
    /// manifests which already depend on the trigger keep producing the same
    /// numbers. What changes is that they say so: in the run log, in the run
    /// directory's `advisories.txt`, and in the CLI envelope under
    /// `CLIAdvisory.deprecatedImplicitSelection`.
    public static let implicitCaseFamilyAdvisory =
        "caseFamily 'sentencing' selected the built-in duration endpoint "
        + "implicitly — declare numericParser instead; this implicit selection "
        + "is deprecated. The shipped registry entry 'sentencing-months' "
        + "(prompts/parsers/parser-registry.json) reproduces this parser "
        + "exactly."

    /// Does the deprecated magic trigger ACTUALLY fire for this manifest?
    ///
    /// ONE definition for every site that asks, because two would drift and
    /// the advisory would then be right at one site and wrong at another.
    ///
    /// Model-output studies: true only when the study declares no
    /// `numericParser` — the declared mechanism always wins — AND names the
    /// one case family with a built-in parser. A study that declares a parser
    /// gets no advisory, because nothing was selected implicitly.
    ///
    /// Multi-agent studies: true on the case family ALONE, mirroring the
    /// server's panel-effects decomposition, which a declared `numericParser`
    /// does not displace. This engine runs no panel-effects decomposition
    /// today; the predicate matches the Python twin regardless, because a
    /// cross-engine predicate that answers differently per engine is a worse
    /// bug than an unreachable branch.
    ///
    /// Python twin: `manifest.implicit_case_family_endpoint`.
    public var usesImplicitCaseFamilyEndpoint: Bool {
        guard caseFamily == Self.implicitEndpointCaseFamily else { return false }
        if studyKind == .multiAgent { return true }
        return (numericParser ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    /// Declared record-exclusion rules (cross-engine contract key
    /// "exclusionRules"): a closed rule vocabulary (failedAttentionCheck /
    /// unparseableEndpoint / outOfRange — see `ExclusionEngine`) applied at
    /// ANALYSIS time, joined against per-item `attentionCheck` declarations
    /// in the task prompts. Manifest data, so freeze pins the rules through
    /// the ordinary content hash. ABSENT = today's behavior exactly: no
    /// exclusion, no stamp. Optional so pre-existing manifests decode
    /// unchanged and keep their content hash.
    public var exclusionRules: [ExclusionRule]?
    /// The largest fraction of ANY ONE CELL's generations (condition ×
    /// promptID) that may have stopped at the token cap rather than
    /// finishing, before the run refuses (`LifecycleGate.lengthStopped`).
    ///
    /// The 2026-08-30 incident: a run capped at a token budget produced
    /// outputs where a large fraction never reached their required final
    /// line, and the loss fell almost entirely on ONE arm — so a run-wide
    /// fraction looked unremarkable while a whole arm was truncated. Hence
    /// per cell, and hence declared IN ADVANCE rather than discovered
    /// afterwards.
    ///
    /// ABSENT = off, which is what every manifest written before this key
    /// says; the per-cell fractions are reported either way. Optional so
    /// pre-existing manifests decode unchanged and keep their content hash
    /// (the `exclusionRules` template — omit-when-nil, no
    /// `defaultElidedFreezeKeys` entry needed).
    public var maxLengthStoppedFraction: Double?
    /// Opt-in acknowledgement that the scored answer options tokenize to
    /// unequal lengths (joint logprobs favor shorter options). nil/false =
    /// the run loop refuses unequal option sets (server
    /// `acknowledge_unequal_option_lengths` twin). Optional so pre-existing
    /// manifests decode unchanged and keep their content hash.
    public var acknowledgeUnequalOptionLengths: Bool?
    /// Stochastic samples per (condition, prompt); nil = 1.
    public var samplesPerItem: Int?
    /// manifestSeeds | derivedSHA256
    public var seedPolicy: String?
    /// Confirm phase: the screen item pool this study must be disjoint from.
    public var screenTaskPromptsHash: String?
    public var humanBaseline: HumanBaseline?
    /// Pinned RepE reader instruments for `outcomeInstruments:
    /// ["repeReaderScore"]`. Optional so legacy manifests decode unchanged
    /// and keep their content hash.
    public var readerRefs: [ReaderRef]?
    public var promotionRule: PromotionRule?
    /// Declared confirmation-study perturbation rule (see the struct doc).
    /// Optional + encodeIfPresent so legacy manifests decode unchanged and
    /// keep their content hash.
    public var perturbationPolicy: PerturbationPolicy?
    public var promptMode: PromptMode?
    public var systemPrompt: String?
    /// The LEGACY spelling of the reasoning effort (false ≡ off, true ≡
    /// xhigh, the template's default). Read from a manifest frozen under it
    /// and never rewritten — the ladder program is frozen under this key and
    /// its hashes stand; a new manifest writes `reasoningEffort` instead.
    public var qwenThinkingEnabled: Bool?
    /// The declared reasoning effort (2026-09-03): off | low | medium |
    /// xhigh, stored as the string it is so a hand-edited value outside the
    /// vocabulary is a `verify` violation on both engines rather than a
    /// decode failure here. nil = absent; `resolvedReasoningEffort` reads
    /// through the legacy boolean. Server twin: `Manifest.reasoning_effort`.
    public var reasoningEffort: String?
    /// The reasoning block's own token cap, REQUIRED beside a non-off effort
    /// and refused beside off (declared, never defaulted). `maxTokens` is then
    /// the ANSWER budget, counted from the token after `</think>`. nil for a
    /// study that predates the key, which keeps it on the single budget it
    /// always ran under. Optional and omit-when-nil, so pre-existing manifests
    /// keep their content hash.
    public var reasoningMaxTokens: Int?
    public var seeds: [UInt64]
    public var temperature: Double
    public var maxTokens: Int
    /// Template lineage — see `TemplateProvenance`. Content-hashed on purpose.
    public var templateProvenance: TemplateProvenance?
    public var status: Status
    public var frozenAt: String?
    public var freezeHash: String?
    public var frozenBy: String?
    public var gitCommit: String?
    /// Engine version that froze the study (cross-engine key: this engine
    /// stamps `SteerLabVersion.current`, the Python server stamps its own).
    /// A lifecycle/provenance stamp like gitCommit — excluded from the
    /// content hash, cleared on duplicate. Optional so pre-existing
    /// manifests decode unchanged and keep their content hash.
    public var appVersion: String?
    /// true only on manifests frozen with `--force` — a lifecycle stamp
    /// (excluded from the content hash, cleared on duplicate) that marks the
    /// freeze as NON-CITABLE: one or more evidence gates were skipped.
    /// Cross-engine key: "freezeForced".
    public var freezeForced: Bool?
    /// The gate ids that were skipped AND would have failed at forced
    /// freeze. Fixed cross-engine vocabulary: "revision",
    /// "validateEvidence", "batteryEvidence", "judgeValidity",
    /// "variantValidity", "gitClean", "measurementPins". Lifecycle stamp
    /// like `freezeForced`.
    public var forcedGatesSkipped: [String]?
    /// SHA-256 of the researcher-authored `preregistration.md` this freeze
    /// PRESERVED (nil when the freeze owned that path and generated the file
    /// itself). A lifecycle stamp like `freezeForced` — excluded from the
    /// content hash, cleared on duplicate — but unlike the others it is
    /// ENFORCED: `verify` re-hashes the file, and the pin surface carries it
    /// into bundles. An authored preregistration is the scientifically
    /// load-bearing kind, so freezing the study must freeze it too.
    /// Cross-engine key: "preregistrationHash".
    public var preregistrationHash: String?
    /// SHA-256 of the settings summary this freeze GENERATED, wherever it
    /// landed. Provenance only — never verified: it is what lets a later
    /// freeze PROVE a file at `preregistration.md` is its own untouched
    /// output rather than guess from the text. Same stamp discipline as
    /// `preregistrationHash`. Cross-engine key:
    /// "preregistrationGeneratedHash".
    public var preregistrationGeneratedHash: String?

    public init(
        name: String, description: String, modelID: String,
        createdAt: Date = Date()
    ) {
        self.name = name
        self.experimentDescription = description
        self.taskDescription = nil
        self.outcomeMeasures = nil
        self.taskPromptsFile = nil
        self.taskPromptsHash = nil
        self.studyKind = .modelOutput
        self.multiAgentScenarioPath = nil
        self.multiAgentScenarioHash = nil
        self.multiAgentSemanticScenarioPath = nil
        self.multiAgentSemanticScenarioHash = nil
        self.multiAgentIncludeBaseline = true
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
        self.modelID = modelID
        self.modelRevision = nil
        self.dtype = nil
        self.concepts = []
        self.grandMeanCorpus = nil
        self.conditions = []
        self.variantConditions = []
        self.neutralCorpusHash = nil
        self.sweep = nil
        self.evaluation = nil
        self.judgeRubricFile = nil
        self.judgeRubricHash = nil
        self.judges = nil
        self.humanValidation = nil
        self.capabilityBatteryFile = nil
        self.capabilityBatteryHash = nil
        self.markersHash = nil
        self.reasoningStyleTaxonomyPath = nil
        self.reasoningStyleTaxonomyHash = nil
        self.phase = nil
        self.caseFamily = nil
        self.outcomeInstruments = nil
        self.outcomeInstrumentScope = nil
        self.evaluationSampling = nil
        self.validationControls = nil
        self.validationLayer = nil
        self.validationLayerFraction = nil
        self.validationLayers = nil
        self.validationLayerFractions = nil
        self.ordinalAggregation = nil
        self.numericParser = nil
        self.parserRegistryHash = nil
        self.exclusionRules = nil
        self.acknowledgeUnequalOptionLengths = nil
        self.samplesPerItem = nil
        self.seedPolicy = nil
        self.screenTaskPromptsHash = nil
        self.humanBaseline = nil
        self.readerRefs = nil
        self.promotionRule = nil
        self.perturbationPolicy = nil
        self.promptMode = .chatAssistant
        self.systemPrompt = nil
        // A NEW manifest spells the effort, never the legacy boolean — the
        // server's `create` writes the same key.
        self.qwenThinkingEnabled = nil
        self.reasoningEffort = ReasoningEffort.off.rawValue
        self.reasoningMaxTokens = nil
        self.seeds = [20260610]
        self.temperature = 0
        self.maxTokens = 2048
        self.templateProvenance = nil
        self.status = .draft
        self.frozenAt = nil
        self.freezeHash = nil
        self.frozenBy = nil
        self.gitCommit = nil
        self.appVersion = nil
        self.freezeForced = nil
        self.forcedGatesSkipped = nil
        self.preregistrationHash = nil
        self.preregistrationGeneratedHash = nil
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        experimentDescription = try container.decode(String.self, forKey: .experimentDescription)
        taskDescription = try container.decodeIfPresent(String.self, forKey: .taskDescription)
        outcomeMeasures = try container.decodeIfPresent(String.self, forKey: .outcomeMeasures)
        taskPromptsFile = try container.decodeIfPresent(String.self, forKey: .taskPromptsFile)
        taskPromptsHash = try container.decodeIfPresent(String.self, forKey: .taskPromptsHash)
        studyKind = try container.decodeIfPresent(StudyKind.self, forKey: .studyKind) ?? .modelOutput
        studyType = try container.decodeIfPresent(String.self, forKey: .studyType)
        multiAgentScenarioPath = try container.decodeIfPresent(String.self, forKey: .multiAgentScenarioPath)
        multiAgentScenarioHash = try container.decodeIfPresent(String.self, forKey: .multiAgentScenarioHash)
        multiAgentSemanticScenarioPath = try container.decodeIfPresent(
            String.self, forKey: .multiAgentSemanticScenarioPath)
        multiAgentSemanticScenarioHash = try container.decodeIfPresent(
            String.self, forKey: .multiAgentSemanticScenarioHash)
        multiAgentIncludeBaseline =
            try container.decodeIfPresent(Bool.self, forKey: .multiAgentIncludeBaseline) ?? true
        createdAt = try container.decode(String.self, forKey: .createdAt)
        modelID = try container.decode(String.self, forKey: .modelID)
        modelRevision = try container.decodeIfPresent(String.self, forKey: .modelRevision)
        dtype = try container.decodeIfPresent(String.self, forKey: .dtype)
        concepts = try container.decodeIfPresent([ConceptRef].self, forKey: .concepts) ?? []
        grandMeanCorpus = try container.decodeIfPresent(
            GrandMeanCorpus.self, forKey: .grandMeanCorpus)
        do {
            conditions =
                try container.decodeIfPresent([Condition].self, forKey: .conditions) ?? []
        } catch let error as DecodingError {
            // Phase-0 gaps G4 + G6 (docs/PORTABILITY-CONTRACTS.md): the
            // obvious client-side shape `{"name": …, "slots": […]}` used to
            // die as a raw `keyNotFound` deep inside `conditions[0]`, naming
            // no arm and offering no repair. A condition document is client
            // input, so it gets a typed refusal like every other client input.
            throw Self.conditionDecodeRefusal(
                error,
                raw: (try? container.decodeIfPresent(
                    [JSONValue].self, forKey: .conditions)) ?? [])
        }
        variantConditions =
            try container.decodeIfPresent([VariantCondition].self, forKey: .variantConditions) ?? []
        neutralCorpusHash = try container.decodeIfPresent(String.self, forKey: .neutralCorpusHash)
        sweep = try container.decodeIfPresent(SweepSpec.self, forKey: .sweep)
        pipeline = try container.decodeIfPresent(JSONValue.self, forKey: .pipeline)
        // Carried verbatim; this engine never authors one (see the property).
        jlensReadout = try container.decodeIfPresent(
            JSONValue.self, forKey: .jlensReadout)
        recordTokenIDs = try container.decodeIfPresent(
            Bool.self, forKey: .recordTokenIDs) ?? false
        // The SAE program's manifest surface: authored on the server, carried
        // verbatim here so a Mac-side duplicate/edit/backfill cannot silently
        // destroy a roster pin, a preregistered mixture cap, or a declared
        // latent arm (see the properties).
        saeCandidates = try container.decodeIfPresent(
            JSONValue.self, forKey: .saeCandidates)
        maxSAEMixtureFeatures = try container.decodeIfPresent(
            JSONValue.self, forKey: .maxSAEMixtureFeatures)
        saeLatentConditions = try container.decodeIfPresent(
            JSONValue.self, forKey: .saeLatentConditions)
        evaluation = try container.decodeIfPresent(EvaluationSpec.self, forKey: .evaluation)
        judgeRubricFile = try container.decodeIfPresent(String.self, forKey: .judgeRubricFile)
        judgeRubricHash = try container.decodeIfPresent(String.self, forKey: .judgeRubricHash)
        judges = try container.decodeIfPresent([JudgeRef].self, forKey: .judges)
        humanValidation = try container.decodeIfPresent(
            HumanBaseline.self, forKey: .humanValidation)
        capabilityBatteryFile = try container.decodeIfPresent(
            String.self, forKey: .capabilityBatteryFile)
        capabilityBatteryHash = try container.decodeIfPresent(
            String.self, forKey: .capabilityBatteryHash)
        markersHash = try container.decodeIfPresent(String.self, forKey: .markersHash)
        reasoningStyleTaxonomyPath = try container.decodeIfPresent(
            String.self, forKey: .reasoningStyleTaxonomyPath)
        reasoningStyleTaxonomyHash = try container.decodeIfPresent(
            String.self, forKey: .reasoningStyleTaxonomyHash)
        phase = try container.decodeIfPresent(String.self, forKey: .phase)
        caseFamily = try container.decodeIfPresent(String.self, forKey: .caseFamily)
        outcomeInstruments = try container.decodeIfPresent(
            [String].self, forKey: .outcomeInstruments)
        outcomeInstrumentScope = try container.decodeIfPresent(
            ResponseFormat.Scope.self, forKey: .outcomeInstrumentScope)
        evaluationSampling = try container.decodeIfPresent(
            EvaluateSubsample.Declaration.self, forKey: .evaluationSampling)
        validationControls = try container.decodeIfPresent(
            [ValidationControl].self, forKey: .validationControls)
        validationLayer = try container.decodeIfPresent(
            Int.self, forKey: .validationLayer)
        validationLayerFraction = try container.decodeIfPresent(
            Double.self, forKey: .validationLayerFraction)
        validationLayers = try container.decodeIfPresent(
            [Int].self, forKey: .validationLayers)
        validationLayerFractions = try container.decodeIfPresent(
            [Double].self, forKey: .validationLayerFractions)
        ordinalAggregation = try container.decodeIfPresent(
            String.self, forKey: .ordinalAggregation)
        numericParser = try container.decodeIfPresent(
            String.self, forKey: .numericParser)
        parserRegistryHash = try container.decodeIfPresent(
            String.self, forKey: .parserRegistryHash)
        exclusionRules = try container.decodeIfPresent(
            [ExclusionRule].self, forKey: .exclusionRules)
        maxLengthStoppedFraction = try container.decodeIfPresent(
            Double.self, forKey: .maxLengthStoppedFraction)
        acknowledgeUnequalOptionLengths = try container.decodeIfPresent(
            Bool.self, forKey: .acknowledgeUnequalOptionLengths)
        samplesPerItem = try container.decodeIfPresent(Int.self, forKey: .samplesPerItem)
        seedPolicy = try container.decodeIfPresent(String.self, forKey: .seedPolicy)
        screenTaskPromptsHash = try container.decodeIfPresent(
            String.self, forKey: .screenTaskPromptsHash)
        humanBaseline = try container.decodeIfPresent(HumanBaseline.self, forKey: .humanBaseline)
        readerRefs = try container.decodeIfPresent([ReaderRef].self, forKey: .readerRefs)
        promotionRule = try container.decodeIfPresent(PromotionRule.self, forKey: .promotionRule)
        perturbationPolicy = try container.decodeIfPresent(
            PerturbationPolicy.self, forKey: .perturbationPolicy)
        promptMode = try container.decodeIfPresent(PromptMode.self, forKey: .promptMode)
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        qwenThinkingEnabled = try container.decodeIfPresent(Bool.self, forKey: .qwenThinkingEnabled)
        reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        reasoningMaxTokens = try container.decodeIfPresent(Int.self, forKey: .reasoningMaxTokens)
        seeds = try container.decodeIfPresent([UInt64].self, forKey: .seeds) ?? [20260610]
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 2048
        templateProvenance = try container.decodeIfPresent(
            TemplateProvenance.self, forKey: .templateProvenance)
        status = try container.decode(Status.self, forKey: .status)
        frozenAt = try container.decodeIfPresent(String.self, forKey: .frozenAt)
        freezeHash = try container.decodeIfPresent(String.self, forKey: .freezeHash)
        frozenBy = try container.decodeIfPresent(String.self, forKey: .frozenBy)
        gitCommit = try container.decodeIfPresent(String.self, forKey: .gitCommit)
        appVersion = try container.decodeIfPresent(String.self, forKey: .appVersion)
        freezeForced = try container.decodeIfPresent(Bool.self, forKey: .freezeForced)
        forcedGatesSkipped = try container.decodeIfPresent(
            [String].self, forKey: .forcedGatesSkipped)
        preregistrationHash = try container.decodeIfPresent(
            String.self, forKey: .preregistrationHash)
        preregistrationGeneratedHash = try container.decodeIfPresent(
            String.self, forKey: .preregistrationGeneratedHash)
    }
}

public struct ExperimentError: Error, CustomStringConvertible {
    public let reason: String
    /// The structured freeze-gate refusal this error carries, when it is one
    /// (WP0 step 2). Additive: `reason`/`description` are byte-identical to
    /// what freeze has always thrown, so every existing catch site, exit
    /// code, and printed message is unchanged, and a machine caller can read
    /// `freezeRefusal?.gate` instead of parsing prose. nil for every error
    /// that is not a freeze-gate refusal.
    public let freezeRefusal: FreezeRefusal?
    /// The structured LIFECYCLE refusal this error carries, when it is one
    /// (WP0 step 7). The second closed vocabulary — every gate on the agent
    /// path that is not a freeze gate. Additive on exactly the same terms as
    /// `freezeRefusal`: `reason` is byte-identical to what the site has always
    /// thrown, so no catch site, printed line, or human exit code moves; a
    /// machine caller reads `lifecycleRefusal?.gate` instead of parsing prose.
    /// nil for errors that are not gate-shaped (a genuine operational failure
    /// must stay distinguishable from a refusal — that is the whole point).
    public let lifecycleRefusal: LifecycleRefusal?
    /// A MALFORMED INVOCATION rather than a refusal: an argument value the
    /// verb cannot accept at all — an out-of-vocabulary enum value typed on
    /// the command line — as opposed to a gate declining a well-formed
    /// request against a healthy system. It lands in the same class an
    /// undeclared flag does (`blocked`, exit 64 in `--json`), because
    /// nothing was run and retrying cannot help; a refusal (65) says the
    /// study needs repairing, which is a different instruction to an agent.
    ///
    /// Gate-5 dry run #2 (P3) measured the gap: an unknown `set-instruments`
    /// value answered `verbFailed`/70, indistinguishable from a crash. It
    /// carries no new state or gate vocabulary — `blocked`/64 and the `usage`
    /// code both predate it; only the CLASSIFICATION of this throw moves.
    public let malformedInvocation: MalformedInvocation?
    public var description: String { reason }

    /// The repair for a malformed invocation: the legal values, as text a
    /// caller can retype.
    public struct MalformedInvocation: Sendable, Equatable {
        public let repairAction: String
        public init(repairAction: String) { self.repairAction = repairAction }
    }

    public init(reason: String) {
        self.reason = reason
        self.freezeRefusal = nil
        self.lifecycleRefusal = nil
        self.malformedInvocation = nil
    }

    public init(refusal: FreezeRefusal) {
        self.reason = refusal.reason
        self.freezeRefusal = refusal
        self.lifecycleRefusal = nil
        self.malformedInvocation = nil
    }

    public init(refusal: LifecycleRefusal) {
        self.reason = refusal.reason
        self.freezeRefusal = nil
        self.lifecycleRefusal = refusal
        self.malformedInvocation = nil
    }

    /// A value the verb's own vocabulary does not contain. `reason` is the
    /// prose the site has always thrown; `repair` names the legal values.
    public static func malformed(_ reason: String, repair: String) -> ExperimentError {
        ExperimentError(reason: reason, malformed: .init(repairAction: repair))
    }

    private init(reason: String, malformed: MalformedInvocation) {
        self.reason = reason
        self.freezeRefusal = nil
        self.lifecycleRefusal = nil
        self.malformedInvocation = malformed
    }
}

/// One capability-battery readout stamped into validation evidence — the
/// pinned battery scored through one condition (baseline or a variant).
/// JSON keys are the pinned cross-engine contract:
/// {"condition", "batteryHash", "total", "correct", "accuracy"}, plus the
/// server's two arming stamps ({"batteryFormat", "armingIsolated"}) when the
/// reading came from a format-aware run — omitted, never null, on legacy
/// evidence, so an existing evidence file still round-trips byte for byte.
public struct CapabilityBatteryConditionResult: Codable, Sendable, Equatable {
    public var condition: String
    public var batteryHash: String
    public var total: Int
    public var correct: Int
    public var accuracy: Double
    /// The format the battery declared (1 legacy, 2 isolated).
    public var batteryFormat: Int?
    /// Whether the reading was armed by the BATTERY rather than by the
    /// surrounding instrument — the difference between a number that is
    /// comparable across instruments and one that is not.
    public var armingIsolated: Bool?

    public init(
        condition: String, batteryHash: String, total: Int, correct: Int,
        accuracy: Double, batteryFormat: Int? = nil,
        armingIsolated: Bool? = nil
    ) {
        self.condition = condition
        self.batteryHash = batteryHash
        self.total = total
        self.correct = correct
        self.accuracy = accuracy
        self.batteryFormat = batteryFormat
        self.armingIsolated = armingIsolated
    }
}
