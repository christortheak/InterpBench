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
        case validationControls
        case validationLayer
        case validationLayerFraction
        case validationLayers
        case validationLayerFractions
        case ordinalAggregation
        case numericParser
        case parserRegistryHash
        case exclusionRules
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
                normCorpusHash: String? = nil, optvecLayer: Int? = nil,
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
            /// Distinct-bigram ratio floor (default 0.45).
            public var coherenceFloor: Double?

            public init(capabilityTolerance: Double? = nil, coherenceFloor: Double? = nil) {
                self.capabilityTolerance = capabilityTolerance
                self.coherenceFloor = coherenceFloor
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
    public var qwenThinkingEnabled: Bool?
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
        self.qwenThinkingEnabled = false
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

public enum ExperimentStore {
    private struct ValidationEvidenceFile: Codable {
        var schemaVersion: Int
        var task: String
        /// Optional: the server's evidence (`tasks.py`) does not stamp it.
        var completedAt: String?
        var validationScopeHash: String
        /// Engine that produced the evidence ("swift-mlx" here,
        /// "python-hf-transformers" on the server). nil = legacy evidence,
        /// necessarily from THIS engine (the historical filename divergence
        /// made cross-engine evidence impossible).
        var substrate: String?
        /// Validation-report filename within the run directory; nil = the
        /// legacy Swift default "report.json".
        var reportFile: String?
        /// Capability-battery-as-evidence: per-condition battery scores for
        /// variant studies (cross-engine evidence key "batteryResults" —
        /// the server's `tasks.py` writes the same). nil = legacy evidence
        /// (pre-battery) or a study with no variant conditions.
        var batteryResults: [CapabilityBatteryConditionResult]?
        /// Pinned concepts for which this validate run scored NO held-out
        /// probe — the concept was eligible for the convergent gate (its
        /// method reads off a source concept's stimuli) but no
        /// `validation.jsonl` existed, it was empty, or the probe could not
        /// be scored (cross-engine evidence key "vacuousConcepts"; the
        /// server's `tasks.py` writes the same).
        ///
        /// Three states, deliberately (the same shape as the
        /// `validationHash` pin): a NON-EMPTY list = this run is vacuous
        /// evidence for those concepts and freeze's `validateEvidence` gate
        /// refuses it; an EMPTY list = the run scored a probe for every
        /// eligible concept; nil = LEGACY evidence written before the stamp
        /// existed, which keeps satisfying the gate exactly as it did
        /// (2026-08-17 firewall repair — only newly vacuous runs stop).
        var vacuousConcepts: [String]?
    }

    /// Stamped into validation evidence so a freeze can enforce the
    /// same-substrate rule explicitly (mirror of the server's
    /// `_THIS_SUBSTRATE = "python-hf-transformers"`). Public so the
    /// cross-substrate advisory can default its perspective parameter.
    public static let evidenceSubstrate = "swift-mlx"

    /// Test seam: when set, experiments live under this root instead of the
    /// project. nonisolated(unsafe) is justified: mutated only by tests,
    /// once, before any concurrent access.
    nonisolated(unsafe) public static var rootOverride: URL?

    public static var directory: URL {
        (rootOverride ?? VectorCatalog.projectRoot).appending(component: "experiments")
    }

    /// Where experiment run directories live — the project's `runs/` in
    /// production, the override root in tests (so freeze's validation-
    /// evidence scan is testable without touching real runs).
    public static var runsDirectory: URL {
        workspaceRoot.appending(component: "runs")
    }

    /// The active workspace's root (test override honoured).
    public static var workspaceRoot: URL {
        rootOverride ?? VectorCatalog.projectRoot
    }

    /// The substrate this WORKSPACE computes on — the one whose artifacts and
    /// evidence it treats as native. Distinct from `evidenceSubstrate`, which
    /// names THIS ENGINE and must stay a constant so foreign evidence stays
    /// detectable. In a cluster workspace the Mac manages data it did not
    /// compute, so the two differ by design.
    public static var computeSubstrate: String {
        WorkspaceCompute.resolved(root: workspaceRoot).substrate
    }

    /// Multi-concept story corpora for grand-mean extraction
    /// (`prompts/emotions/<concept>/stories.jsonl`), under the same test
    /// seam as experiments/runs so lifecycle tests are hermetic.
    public static var emotionsDirectory: URL {
        (rootOverride ?? VectorCatalog.projectRoot).appending(
            components: "prompts", "emotions")
    }

    static func storiesURL(for concept: String) -> URL {
        emotionsDirectory.appending(components: concept, "stories.jsonl")
    }

    /// SHA-256 of a concept's stories.jsonl raw bytes (the firewall pin), or
    /// nil when the file is missing or unreadable.
    static func storiesHash(for concept: String) -> String? {
        (try? StimulusSet.loadMultiConceptTexts(url: storiesURL(for: concept)))?.hash
    }

    /// The text rows of one concept's stories.jsonl, in file order — the
    /// class loader for designated-reference extraction. Throws when the
    /// file is missing or empty: a silent empty class would extract a
    /// garbage mean.
    static func loadStoriesTexts(for concept: String) throws -> [String] {
        try StimulusSet.loadMultiConceptTexts(url: storiesURL(for: concept))
            .rows.map(\.text)
    }

    // MARK: - Measurement-side input pins (markers / validation / PC basis)

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// THE workspace-relative resolution rule for a concept's never-named
    /// validation scenarios — paired concepts under
    /// `prompts/concepts/<name>/validation.jsonl`, grand-mean concepts under
    /// `prompts/emotions/<name>/validation.jsonl`. Single source of truth:
    /// `conceptValidationURL` (the pin/verify reader) and
    /// `StudyDataReadiness` (the checklist) both resolve through it.
    public static func conceptValidationRelativePath(
        name: String, isPaired: Bool
    ) -> String {
        (isPaired ? "prompts/concepts/" : "prompts/emotions/")
            + name + "/validation.jsonl"
    }

    /// THE workspace-relative location of a concept's `markers.json` scoring
    /// rubric. The run loop and the sweep read this path for EVERY attached
    /// concept, grand-mean included (see `liveMarkersHash`).
    public static func markersRelativePath(concept: String) -> String {
        "prompts/concepts/\(concept)/markers.json"
    }

    /// The concept's never-named validation scenarios, resolved exactly as
    /// `ExperimentTasks.validate` reads them (see
    /// `conceptValidationRelativePath` for the rule).
    static func conceptValidationURL(name: String, isPaired: Bool) -> URL {
        // Paired concepts resolve against the live project root; grand-mean
        // concepts honor the test-seam override (matches `emotionsDirectory`).
        let root = isPaired
            ? VectorCatalog.projectRoot
            : (rootOverride ?? VectorCatalog.projectRoot)
        return root.appending(
            path: conceptValidationRelativePath(name: name, isPaired: isPaired))
    }

    /// Where a concept's held-out `validation.jsonl` was actually FOUND.
    ///
    /// `url` is the file the pin hashes and the probe reads; `canonicalURL`
    /// is the home this recipe owns. `usedFallback` is true when the two
    /// differ (the set is filed under the OTHER recipe's root);
    /// `bothHomesPresent` is true when a file sits in each home — ambiguous
    /// filing, worth its own advisory even though the canonical one wins.
    /// Python twin: `manifest.ValidationSetLocation`.
    public struct ValidationSetLocation: Sendable, Equatable {
        public let url: URL
        public let relativePath: String
        public let canonicalURL: URL
        public let canonicalRelativePath: String
        public let fallbackRelativePath: String
        public let usedFallback: Bool
        public let bothHomesPresent: Bool
    }

    /// THE dual-root lookup for a concept's held-out set (2026-08-19).
    ///
    /// Resolution order is deterministic and the canonical home ALWAYS wins:
    ///
    /// 1. the recipe's canonical home (`conceptValidationURL`),
    /// 2. failing that, the OTHER recipe's home.
    ///
    /// nil when neither holds a file — the historical "absent" state,
    /// unchanged. Only WHERE we look changed: a set filed under the wrong
    /// recipe's root used to be silently invisible (no hash pinned, no
    /// error), which is a measurement-side pin quietly missing from the
    /// firewall. Callers that land on the fallback must SAY SO —
    /// `validationLookupAdvisory` is the one wording. Python twin:
    /// `manifest.resolve_validation_file`.
    public static func resolveConceptValidation(
        name: String, isPaired: Bool
    ) -> ValidationSetLocation? {
        let fm = FileManager.default
        let canonical = conceptValidationURL(name: name, isPaired: isPaired)
        let fallback = conceptValidationURL(name: name, isPaired: !isPaired)
        let canonicalRelative = conceptValidationRelativePath(
            name: name, isPaired: isPaired)
        let fallbackRelative = conceptValidationRelativePath(
            name: name, isPaired: !isPaired)
        let canonicalPresent = fm.fileExists(atPath: canonical.path)
        let fallbackPresent =
            fm.fileExists(atPath: fallback.path)
            && fallback.standardizedFileURL != canonical.standardizedFileURL
        guard canonicalPresent || fallbackPresent else { return nil }
        let usedFallback = !canonicalPresent
        return ValidationSetLocation(
            url: usedFallback ? fallback : canonical,
            relativePath: usedFallback ? fallbackRelative : canonicalRelative,
            canonicalURL: canonical,
            canonicalRelativePath: canonicalRelative,
            fallbackRelativePath: fallbackRelative,
            usedFallback: usedFallback,
            bothHomesPresent: canonicalPresent && fallbackPresent)
    }

    /// The LOUD, non-fatal note a dual-root lookup owes its caller: the file
    /// was not in this recipe's home, or it is in both. nil when the filing
    /// is unambiguous (canonical only) or nothing was found.
    ///
    /// One wording, shared by every point of use (validate, freeze,
    /// data-readiness) so the same mistake reads the same everywhere.
    /// Python twin: `manifest.validation_lookup_advisory` — intent-twinned
    /// prose, not a byte-identical literal (these are engine-local
    /// log/advisory lines, like the other freeze advisories).
    public static func validationLookupAdvisory(
        concept: String, location: ValidationSetLocation?
    ) -> String? {
        guard let location else { return nil }
        if location.bothHomesPresent {
            return "concept '\(concept)': validation.jsonl exists under BOTH "
                + "recipe roots (\(location.canonicalRelativePath) and "
                + "\(location.fallbackRelativePath)) — this recipe reads and "
                + "pins the canonical \(location.canonicalRelativePath); "
                + "delete or merge the other so the held-out set is unambiguous"
        }
        if location.usedFallback {
            return "concept '\(concept)': validation.jsonl was found under the "
                + "OTHER recipe's root (\(location.relativePath)) and is being "
                + "read and pinned from there — this recipe's canonical home "
                + "is \(location.canonicalRelativePath); move it there so the "
                + "pin names the file the recipe owns"
        }
        return nil
    }

    /// SHA-256 over the concept's validation.jsonl raw bytes, or nil when
    /// the file does not exist in EITHER home (nothing to pin). Stamped into
    /// new `ConceptRef.validationHash` pins at attach; checked by `verify()`.
    ///
    /// Hashes the file the dual-root lookup actually found, canonical home
    /// first. The three-state pin semantics are untouched: a hash when a
    /// file exists in either home, an explicit null when neither does.
    public static func conceptValidationHash(name: String, isPaired: Bool) -> String? {
        guard let location = resolveConceptValidation(name: name, isPaired: isPaired)
        else { return nil }
        return conceptValidationHash(fileURL: location.url)
    }

    /// The same hash for a validation file the caller has already located
    /// (`DatasetInventory`, which walks the workspace itself). ONE definition
    /// of what the validation pin hashes — bytes of the file, SHA-256 — so a
    /// hash shown in the Data inventory is the hash a manifest would pin.
    public static func conceptValidationHash(fileURL: URL) -> String? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return sha256Hex(data)
    }

    /// The one true attach constructor: pins the concept at the given
    /// stimulus hash AND always writes the measurement-side validation pin
    /// (the hash when validation.jsonl exists, an explicit null otherwise —
    /// cross-engine rule: new attaches always write the key).
    public static func makeConceptRef(
        name: String, stimulusSetHash: String, options: ExtractionOptions
    ) -> ExperimentManifest.ConceptRef {
        let hash = conceptValidationHash(name: name, isPaired: options.method.isPaired)
        return ExperimentManifest.ConceptRef(
            name: name, stimulusSetHash: stimulusSetHash, options: options,
            validationHash: hash, validationHashPinnedAbsent: hash == nil)
    }

    /// Cross-engine markers aggregate (pure; fixture-tested on both
    /// engines): for each DISTINCT concept name, sorted ascending by UTF-8
    /// bytes, that HAS markers bytes, emit the line
    /// `"<name>\t<sha256-hex-of-raw-bytes>\n"`; the aggregate is the SHA-256
    /// hex of the concatenated lines' UTF-8. nil when no entry has bytes.
    static func markersAggregateHash(_ entries: [(name: String, bytes: Data)]) -> String? {
        var byName: [String: Data] = [:]
        for entry in entries where byName[entry.name] == nil {
            byName[entry.name] = entry.bytes
        }
        let names = byName.keys.sorted {
            Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8))
        }
        guard !names.isEmpty else { return nil }
        var lines = ""
        for name in names {
            lines += "\(name)\t\(sha256Hex(byName[name]!))\n"
        }
        return sha256Hex(Data(lines.utf8))
    }

    /// Combined pin over every attached concept's `markers.json`, resolved
    /// exactly where scoring resolves it (the run loop and the sweep both
    /// read `prompts/concepts/<name>/markers.json` for EVERY attached
    /// concept, grand-mean included). nil when no attached concept has a
    /// markers.json (nothing to pin).
    public static func liveMarkersHash(_ manifest: ExperimentManifest) -> String? {
        markersAggregateHash(
            manifest.concepts.compactMap { ref in
                let url = VectorCatalog.projectRoot.appending(
                    path: markersRelativePath(concept: ref.name))
                guard let data = try? Data(contentsOf: url) else { return nil }
                return (ref.name, data)
            })
    }

    /// Raw bytes of a neutral-PC basis artifact. A file path reads the file
    /// itself; a directory path resolves to the basis JSON inside it (this
    /// engine's `neutral-pcs.json`, else the server's
    /// `neutral-pc-basis.json`).
    static func neutralPCBasisBytes(path: String) -> Data? {
        let url =
            path.hasPrefix("/")
            ? URL(filePath: path)
            : (rootOverride ?? VectorCatalog.projectRoot).appending(path: path)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        else { return nil }
        if isDirectory.boolValue {
            for name in ["neutral-pcs.json", "neutral-pc-basis.json"] {
                if let data = try? Data(contentsOf: url.appending(component: name)) {
                    return data
                }
            }
            return nil
        }
        return try? Data(contentsOf: url)
    }

    /// The canonical chain stage order (server `pipeline_spec.VALID_STAGES`).
    static let pipelineValidStages = [
        "extract", "validate", "sweep", "promote", "run", "evaluate",
        "analyze",
    ]

    /// Pure mirror of the server resolver's refusals over the passthrough
    /// `pipeline` block (stage 5): unknown/out-of-order/duplicate stages,
    /// evaluate/analyze without run, unknown gate names/keys, gates naming
    /// absent stages, out-of-range thresholds. The declaration is checked
    /// here; the SERVER remains the resolver of record at pipeline start.
    /// `verify()` for the manifest's optional `saeCandidates` pin — the
    /// MECHANICAL half of the server's `sae_candidates.pin_violations`.
    ///
    /// Same contract as `markersHash` and the reasoning-style taxonomy:
    /// absent pins nothing and violates nothing; a half-pin certifies
    /// nothing; an ABSOLUTE path is a violation (the Mac workspace is the
    /// source of truth, so stored refs are workspace-relative or they resolve
    /// to nothing on the cluster); a missing file or drifted bytes after
    /// pinning is a violation exactly like stimulus drift.
    ///
    /// What this deliberately does NOT do is validate the roster's SCHEMA.
    /// That is server-only semantics (`sae_candidates.CandidateManifest`),
    /// and a second validator of the same file drifts from the first by
    /// construction — the file-hash check is the part that is identical on
    /// both engines, so it is the part that lives here.
    public static func saeCandidatesPinViolations(_ block: JSONValue?) -> [String] {
        guard let block, !isNull(block) else { return [] }
        guard case .object(let pin) = block else {
            return [
                "saeCandidates must be an object {\"path\": …, \"hash\": …} "
                    + "naming the workspace-relative SAE candidate manifest"
            ]
        }
        let unknown = pin.keys.filter { $0 != "path" && $0 != "hash" }.sorted()
        if !unknown.isEmpty {
            return [
                "saeCandidates has unknown key(s) \(unknown.joined(separator: ", ")) "
                    + "— the pin block is path + hash only"
            ]
        }
        guard case .string(let path)? = pin["path"],
            case .string(let pinned)? = pin["hash"],
            !path.isEmpty, !pinned.isEmpty
        else {
            return [
                "SAE candidate manifest pin is incomplete — saeCandidates.path "
                    + "and saeCandidates.hash must both be set"
            ]
        }
        if path.hasPrefix("/") {
            return [
                "SAE candidate manifest path '\(path)' is absolute — pinned "
                    + "inputs are WORKSPACE-RELATIVE so the study resolves on any "
                    + "machine (the Mac workspace is the source of truth)"
            ]
        }
        guard let data = try? Data(contentsOf: resolveProjectPath(path)) else {
            return [
                "SAE candidate manifest '\(path)': file missing — the manifest "
                    + "pins it, so the bytes must be there (restore the file, or "
                    + "drop the saeCandidates block on a duplicate draft)"
            ]
        }
        let live = sha256Hex(data)
        if live != pinned {
            return [
                "SAE candidate manifest '\(path)' changed since pinning "
                    + "(have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)"
            ]
        }
        return []
    }

    public static func pipelineBlockViolations(_ block: JSONValue?) -> [String] {
        guard let block else { return [] }
        guard case .object(let pipeline) = block else {
            return ["pipeline block invalid: must be an object"]
        }
        var violations: [String] = []
        for key in pipeline.keys where key != "stages" && key != "gates" {
            violations.append(
                "pipeline block invalid: unknown pipeline key '\(key)' — a "
                    + "typo'd key silently ignored would un-declare a gate")
        }
        // The effective stage list (absent/empty = the default chain) — the
        // gate-membership check below needs it.
        var stages = ["extract", "validate", "sweep", "promote", "run"]
        if let rawStages = pipeline["stages"], !isNull(rawStages) {
            guard case .array(let items) = rawStages else {
                return violations
                    + ["pipeline block invalid: 'stages' must be a list of stage names"]
            }
            var names: [String] = []
            for item in items {
                guard case .string(let name) = item else {
                    return violations
                        + ["pipeline block invalid: 'stages' must be a list of stage names"]
                }
                names.append(name)
            }
            if !names.isEmpty {
                let order = Dictionary(
                    uniqueKeysWithValues: pipelineValidStages.enumerated()
                        .map { ($1, $0) })
                let unknown = names.filter { order[$0] == nil }
                if !unknown.isEmpty {
                    violations.append(
                        "pipeline block invalid: unknown stage(s) "
                            + unknown.joined(separator: ", "))
                }
                if Set(names).count != names.count {
                    violations.append(
                        "pipeline block invalid: 'stages' contains duplicates")
                }
                let indices = names.compactMap { order[$0] }
                if indices != indices.sorted() {
                    violations.append(
                        "pipeline block invalid: 'stages' must follow the "
                            + "canonical order "
                            + pipelineValidStages.joined(separator: ", "))
                }
                if (names.contains("evaluate") || names.contains("analyze")),
                    !names.contains("run")
                {
                    violations.append(
                        "pipeline block invalid: evaluate/analyze require "
                            + "'run' in the same chain")
                }
                stages = names
            }
        }
        if let rawGates = pipeline["gates"], !isNull(rawGates) {
            guard case .object(let gates) = rawGates else {
                return violations + ["pipeline block invalid: 'gates' must be an object"]
            }
            let knownKeys: [String: Set<String>] = [
                "validate": [
                    "minScenarioAccuracy", "maxCrossConceptCosine",
                    "accuracyFloor",
                ],
                "sweep": ["requireSelectionForEveryConcept"],
            ]
            for (gateName, gate) in gates {
                guard let keys = knownKeys[gateName] else {
                    violations.append(
                        "pipeline block invalid: no gate is defined for "
                            + "stage '\(gateName)'")
                    continue
                }
                if !stages.contains(gateName) {
                    violations.append(
                        "pipeline block invalid: gate '\(gateName)' names a "
                            + "stage that is not in the stage list")
                }
                guard case .object(let fields) = gate else {
                    if !isNull(gate) {
                        violations.append(
                            "pipeline block invalid: gate '\(gateName)' must "
                                + "be an object")
                    }
                    continue
                }
                for (key, value) in fields {
                    guard keys.contains(key) else {
                        violations.append(
                            "pipeline block invalid: unknown \(gateName)-gate "
                                + "key '\(key)'")
                        continue
                    }
                    if gateName == "validate", key == "accuracyFloor",
                        !isNull(value)
                    {
                        violations += accuracyFloorViolations(value)
                        continue
                    }
                    if gateName == "validate", !isNull(value) {
                        guard case .number(let threshold) = value,
                            threshold >= 0, threshold <= 1
                        else {
                            violations.append(
                                "pipeline block invalid: gate "
                                    + "'\(gateName).\(key)' must be a number "
                                    + "in [0, 1]")
                            continue
                        }
                    }
                }
                // The legacy key IS the transferAccuracy floor — declaring
                // it beside an accuracyFloor is one ambiguity, refused on
                // both engines (server resolver twin).
                if gateName == "validate",
                    let legacy = fields["minScenarioAccuracy"], !isNull(legacy),
                    let declared = fields["accuracyFloor"], !isNull(declared)
                {
                    violations.append(
                        "pipeline block invalid: both minScenarioAccuracy "
                            + "and accuracyFloor are declared — declare "
                            + "exactly one")
                }
            }
        }
        return violations
    }

    private static func isNull(_ value: JSONValue) -> Bool {
        if case .null = value { return true }
        return false
    }

    /// The COMPLETE reader↔study binding, in one place (review 2026-08-02:
    /// verify and the runtime scorer each carried their own subset, so the
    /// runtime accepted readers verify would flag — and a forced freeze
    /// made that gap live). A reader is activation-, model-, revision-,
    /// and substrate-specific: substrate must be this engine's, modelID
    /// must equal the study model, a revision is REQUIRED and must equal
    /// the study's pin when one exists, and the artifact's concept must be
    /// the concept the ref claims. Used by `verify()` (violations) and
    /// `ExperimentTasks.loadReaderScorers` (refusals). Server twin:
    /// `repe_reader.binding_problems`.
    static func readerBindingProblems(
        _ artifact: RepEReader.Artifact, refConcept: String,
        manifest: ExperimentManifest
    ) -> [String] {
        var problems: [String] = []
        if artifact.substrate != RepEReader.substrate {
            problems.append(
                "reader '\(refConcept)' was fitted on substrate "
                    + "'\(artifact.substrate)', not this engine "
                    + "('\(RepEReader.substrate)') — reader artifacts are "
                    + "substrate-specific and must be re-fitted")
        }
        if artifact.modelID.isEmpty {
            problems.append(
                "reader '\(refConcept)': artifact carries no modelID — "
                    + "an unattributable reader cannot be bound to a study")
        } else if artifact.modelID != manifest.modelID {
            problems.append(
                "reader '\(refConcept)' was fitted on \(artifact.modelID), "
                    + "not the study model \(manifest.modelID)")
        }
        if artifact.revision?.isEmpty != false {
            problems.append(
                "reader '\(refConcept)': artifact carries no model "
                    + "revision — readers bind to exact fitted bytes")
        } else if let readerRevision = artifact.revision,
            let pinned = manifest.modelRevision, readerRevision != pinned
        {
            problems.append(
                "reader '\(refConcept)' was fitted on revision "
                    + "\(readerRevision.prefix(12))…, not the study's pinned "
                    + "\(pinned.prefix(12))…")
        }
        if artifact.concept != refConcept {
            problems.append(
                "reader '\(refConcept)': the pinned artifact is for "
                    + "concept '\(artifact.concept)' — the ref names the "
                    + "wrong instrument")
        }
        return problems
    }

    /// The declared accuracy-floor metric vocabulary (cross-engine; server
    /// twin `pipeline_spec.ACCURACY_FLOOR_METRICS`). Each name reads ONE
    /// place in the validation report; an entry that cannot produce the
    /// declared metric FAILS the gate — never a fallback.
    public static let pipelineAccuracyFloorMetrics = [
        "transferAccuracy", "calibratedAccuracy",
        "calibratedBalancedAccuracy", "auc",
    ]

    /// Shape/vocabulary check for `gates.validate.accuracyFloor` — the
    /// server resolver's refusals, mirrored.
    private static func accuracyFloorViolations(_ value: JSONValue) -> [String] {
        guard case .object(let floor) = value,
            Set(floor.keys) == ["metric", "minimum"]
        else {
            return [
                "pipeline block invalid: 'validate.accuracyFloor' must be "
                    + "an object {\"metric\": …, \"minimum\": …}"
            ]
        }
        var violations: [String] = []
        if case .string(let metric)? = floor["metric"] {
            if !pipelineAccuracyFloorMetrics.contains(metric) {
                violations.append(
                    "pipeline block invalid: unknown accuracyFloor metric "
                        + "'\(metric)' — declare one of "
                        + pipelineAccuracyFloorMetrics.joined(separator: ", "))
            }
        } else {
            violations.append(
                "pipeline block invalid: accuracyFloor metric must be one of "
                    + pipelineAccuracyFloorMetrics.joined(separator: ", "))
        }
        if case .number(let minimum)? = floor["minimum"],
            minimum >= 0, minimum <= 1
        {
            // In range — fine.
        } else {
            violations.append(
                "pipeline block invalid: gate 'validate.accuracyFloor.minimum' "
                    + "must be a number in [0, 1]")
        }
        return violations
    }

    /// Freeze-time sweep-input pinning (see the call site in `freeze` for
    /// the contract): each absent pin whose file exists gains the SHA-256 of
    /// the file's raw bytes; existing pins are NEVER touched (a drifted pin
    /// must surface as a verify violation, not be silently repaired). A
    /// missing file pins nothing HERE — `missingSweepInputRefusals` (called
    /// right after this in `freeze`) then refuses the freeze, so an
    /// operative sweep can never be frozen with an unpinned input. Server
    /// twin: the sweep branch of `experiment_store.freeze`.
    /// The choice-instrument pin surface (review 2026-08-02, P1: the files
    /// that determine the WINNING CELL were the one sweep input not pinned
    /// at freeze). One entry per declared instrument: concept nil = the
    /// singular `choicePromptsFile` (pin `choicePromptsHash`), a concept
    /// names a `choicePromptsFiles` entry (pin map `choicePromptsHashes`).
    /// Shared by freeze (pin-when-absent), verify (drift), and the
    /// sweep-start refusal. Server twin: `sweep_choice_pin_entries`.
    static func sweepChoicePinEntries(
        _ sweep: ExperimentManifest.SweepSpec
    ) -> [(concept: String?, file: String, pinned: String?, label: String)] {
        guard let objective = sweep.selection?.objective else { return [] }
        // Only logprobShift READS choice instruments (review 2026-08-02
        // round 2, P2): a stale path carried under judgeScore/markerDensity
        // is inert at execution, so pinning it would let dead declarations
        // block freezes over files nothing reads.
        guard objective.metric == "logprobShift" else { return [] }
        var entries: [(String?, String, String?, String)] = []
        if let singular = objective.choicePromptsFile,
            !singular.trimmingCharacters(in: .whitespaces).isEmpty
        {
            entries.append(
                (nil, singular, objective.choicePromptsHash,
                 "sweep choice prompts"))
        }
        for (concept, rel) in (objective.choicePromptsFiles ?? [:])
            .sorted(by: { $0.key < $1.key })
        {
            entries.append(
                (concept, rel, objective.choicePromptsHashes?[concept],
                 "sweep choice prompts '\(concept)'"))
        }
        return entries
    }

    static func pinSweepInputs(into manifest: inout ExperimentManifest) {
        guard var sweep = manifest.sweep else { return }
        if sweep.devPromptsHash == nil,
            let data = try? Data(
                contentsOf: resolveProjectPath(sweep.devPromptsFile))
        {
            sweep.devPromptsHash = sha256Hex(data)
        }
        if sweep.batteryHash == nil,
            let data = try? Data(
                contentsOf: resolveProjectPath(sweep.batteryFile))
        {
            sweep.batteryHash = sha256Hex(data)
        }
        // Choice instruments: same contract — pin-when-absent, never
        // re-pin; a missing file pins nothing here and
        // `missingSweepInputRefusals` then refuses the freeze.
        for entry in sweepChoicePinEntries(sweep) where entry.pinned == nil {
            guard
                let data = try? Data(
                    contentsOf: resolveProjectPath(entry.file))
            else { continue }
            if let concept = entry.concept {
                var hashes = sweep.selection?.objective?
                    .choicePromptsHashes ?? [:]
                hashes[concept] = sha256Hex(data)
                sweep.selection?.objective?.choicePromptsHashes = hashes
            } else {
                sweep.selection?.objective?.choicePromptsHash = sha256Hex(data)
            }
        }
        manifest.sweep = sweep
    }

    /// Freeze-only refusals for sweep inputs that could NOT be pinned
    /// (firewall closure second pass, 2026-07-20): must run AFTER
    /// `pinSweepInputs`, so a still-nil hash means the file was missing or
    /// unreadable at the pin moment. Freezing anyway would leave an
    /// operative sweep with an unpinned input — and if the file appears
    /// AFTER freeze, sweep start's legacy-unpinned fallback would accept
    /// whatever bytes it finds, a preregistration hole. Freeze-time only:
    /// legacy manifests frozen before this rule (absent hashes) keep
    /// verifying clean — their documented epoch is the sweep-start legacy
    /// path, and only NEW freezes refuse. Server twin: the sweep branch of
    /// `experiment_store.freeze`.
    static func missingSweepInputRefusals(
        _ manifest: ExperimentManifest
    ) -> [String] {
        guard let sweep = manifest.sweep else { return [] }
        var refusals: [String] = []
        for entry in sweepChoicePinEntries(sweep) where entry.pinned == nil {
            refusals.append(
                "\(entry.label) file '\(entry.file)' is missing, so freeze "
                    + "cannot pin it — an operative sweep selects on that "
                    + "file; create the file, or remove/repoint the "
                    + "declaration, before freezing")
        }
        for (file, hash, label, fileKey) in [
            (sweep.devPromptsFile, sweep.devPromptsHash,
             "sweep dev prompts", "devPromptsFile"),
            (sweep.batteryFile, sweep.batteryHash,
             "sweep capability battery", "batteryFile"),
        ] where hash == nil {
            refusals.append(
                "\(label) file '\(file)' is missing, so freeze cannot pin "
                    + "it — an operative sweep selects on that file; create "
                    + "the file, or remove/repoint the sweep's \(fileKey), "
                    + "before freezing")
        }
        return refusals
    }

    /// Non-blocking freeze advisory when measurement-side inputs exist on
    /// disk but are not pinned by this manifest — legacy attaches (no
    /// `validationHash`) and legacy freezes (no `markersHash`). New attaches
    /// always pin; freeze pins `markersHash` itself, so post-2026-07-13
    /// drafts only trip the validation half.
    static func measurementPinAdvisory(for manifest: ExperimentManifest) -> String? {
        var unpinned = manifest.markersHash == nil && liveMarkersHash(manifest) != nil
        if !unpinned {
            unpinned = manifest.concepts.contains { ref in
                // An explicit null IS a pin ("no validation set existed") —
                // only the legacy keyless state is unpinned.
                ref.validationHash == nil && !ref.validationHashPinnedAbsent
                    && conceptValidationHash(
                        name: ref.name, isPaired: ref.options.method.isPaired) != nil
            }
        }
        return unpinned
            ? "measurement-side inputs unpinned (markers/validation) — re-attach to pin"
            : nil
    }

    /// Resolves a manifest-pinned path: absolute paths pass through, relative
    /// paths resolve against the project root (the test seam root in tests).
    static func resolveProjectPath(_ path: String) -> URL {
        resolveProjectPath(path, root: workspaceRoot)
    }

    /// The same rule against a NAMED workspace, for the scanners that already
    /// know which tree they are reading (`TaskPromptsStore.list(root:)`,
    /// reached from `DatasetInventory.scan(root:)`). One rule — a second
    /// spelling is how a pinned path and its listing would stop agreeing.
    static func resolveProjectPath(_ path: String, root: URL) -> URL {
        path.hasPrefix("/") ? URL(filePath: path) : root.appending(path: path)
    }

    static func manifestURL(_ name: String) -> URL {
        directory.appending(components: name, "experiment.json")
    }

    public static func list() -> [ExperimentManifest] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .compactMap { try? load(name: $0.lastPathComponent) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public static func load(name: String) throws -> ExperimentManifest {
        let data = try Data(contentsOf: manifestURL(name))
        return try JSONDecoder().decode(ExperimentManifest.self, from: data)
    }

    /// The one name rule for anything that becomes an `experiments/<name>/`
    /// path component: lowercase letters, digits, hyphens. Everything else
    /// (spaces → hyphens; slashes, dots, traversal sequences) is dropped —
    /// a manifest name is never allowed to be a path.
    public static func sanitizedExperimentName(_ name: String) -> String {
        name.lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
    }

    public static func create(
        name: String, description: String, modelID: String,
        modelRevision: String? = nil
    ) throws -> ExperimentManifest {
        let sanitized = sanitizedExperimentName(name)
        guard !sanitized.isEmpty else { throw ExperimentError(reason: "empty name") }
        guard (try? load(name: sanitized)) == nil else {
            throw ExperimentError(reason: "experiment '\(sanitized)' already exists")
        }
        var manifest = ExperimentManifest(
            name: sanitized, description: description, modelID: modelID)
        manifest.modelRevision = modelRevision
        try save(manifest, allowCreate: true)
        return manifest
    }

    /// Whether a manifest holds concepts/conditions — the §8 transition's
    /// WATCHED pair. `variantConditions` is deliberately outside this pair on
    /// the EXISTING side: clearing only the variant arms is a legitimate edit
    /// that has never gone wrong. Server twin:
    /// `experiment_store.ARM_BEARING_KEYS`.
    static func holdsArms(_ manifest: ExperimentManifest) -> Bool {
        !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
    }

    /// Whether the INCOMING document still carries ANY measured surface —
    /// `variantConditions` included. An agentComparison-style save whose whole
    /// surface lives in variant conditions is not a disarm (that study type's
    /// arms LIVE there), and refusing it was the guard's first false positive
    /// (test_transcript_study, caught at landing 2026-08-20). Server twin:
    /// `experiment_store._clears_every_arm`'s incoming check.
    static func holdsAnySurface(_ manifest: ExperimentManifest) -> Bool {
        holdsArms(manifest) || !manifest.variantConditions.isEmpty
    }

    /// Persists a manifest. Frozen manifests are immutable: only the
    /// frozen → complete transition may be written.
    ///
    /// `mayClearArms` is the caller DECLARING that dropping every concept and
    /// condition is the point of this write (open-issues §8). Without it, such
    /// a save is refused. The reason is that `save` writes the WHOLE document
    /// and several callers hold a manifest they read some time ago —
    /// `ExperimentPanel.selected` is a cache refreshed on selection, and the
    /// run paths hold a copy for the length of a sweep — so a document that
    /// arrives at both-empty over a populated draft is far more often a stale
    /// copy than an intentional reset, and the loss is silent: the only reason
    /// §8's arms survived at all is that a run directory had snapshotted them.
    /// Server twin: `experiment_store.save_raw(clearing_arms=…)`.
    public static func save(
        _ manifest: ExperimentManifest, allowCreate: Bool = false,
        mayClearArms: Bool = false
    ) throws {
        if let existing = try? load(name: manifest.name) {
            switch existing.status {
            case .draft:
                if !mayClearArms, holdsArms(existing), !holdsAnySurface(manifest) {
                    throw ExperimentError.refusing(
                        .armsCleared,
                        "refusing to save '\(manifest.name)' with no concepts "
                            + "and no conditions over a draft that has "
                            + "\(existing.concepts.count) concept(s) and "
                            + "\(existing.conditions.count) condition(s) — a "
                            + "manifest does not lose its whole measured "
                            + "surface in one write by accident",
                        repair: clearedArmsRepair(manifest.name))
                }
            case .frozen:
                // Only completion is allowed, and nothing else may change.
                var completed = existing
                completed.status = .complete
                guard manifest == completed else {
                    throw ExperimentError.refusing(
                        .statusImmutable,
                        "experiment '\(manifest.name)' is frozen — duplicate it "
                            + "to iterate",
                        repair: duplicateToIterateRepair(manifest.name))
                }
            case .complete:
                throw ExperimentError.refusing(
                    .statusImmutable,
                    "experiment '\(manifest.name)' is complete and immutable",
                    repair: duplicateToIterateRepair(manifest.name))
            }
        } else if !allowCreate {
            throw ExperimentError(reason: "experiment '\(manifest.name)' does not exist")
        }

        let url = manifestURL(manifest.name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: url)
    }

    // MARK: - Draft science-manifest setters (App gap A2)

    /// Known vocabulary the editors OFFER (pickers), never a validation
    /// wall: the fields stay free strings in the manifest — concepts, case
    /// families, and phases are data (CLAUDE.md design rule), and the server
    /// may know values this engine does not.
    public static let knownPhases = [
        "shakedown", "screen", "confirm", "triangulate", "panel",
    ]
    /// Suggested `caseFamily` LABELS. None of them selects an instrument:
    /// `caseFamily` is provenance, and `numericParser` + the workspace parser
    /// registry are how a study declares how its answers are read. The one
    /// value that still behaves — `"sentencing"`, with no parser declared —
    /// does so through a DEPRECATED implicit rule that announces itself at
    /// every site where it fires (`implicitCaseFamilyAdvisory`).
    public static let knownCaseFamilies = [
        "siliconFormalism", "katzZamir", "sentencing",
    ]
    public static let knownSeedPolicies = ["manifestSeeds", "derivedSHA256"]
    public static let knownOutcomeInstruments = [
        "sampledText", "answerTokenLogprob", "choiceProbability",
        "repeReaderScore", "ordinalScale",
    ]

    /// The declared instruments no engine implements, in declaration order.
    ///
    /// The vocabulary was enforced at DECLARATION only (`setOutcomeInstruments`
    /// refuses at 64), which protects the one path that goes through this
    /// CLI's authoring verb and no other. A hand-edited manifest, an imported
    /// bundle, or a manifest authored against a newer vocabulary carries
    /// whatever string it likes — and every downstream reader is a SET
    /// MEMBERSHIP test (`CHOICE_INSTRUMENTS.contains`, `.contains("ordinalScale")`,
    /// `ExecutionPlan.resolve`), so an unrecognised value dispatches nothing,
    /// raises nothing, and the study completes having measured only the
    /// default sampled text. `sampledTxt` for `sampledText` is the whole
    /// failure. Server twin: `experiment_store.unknown_outcome_instruments`.
    public static func unknownOutcomeInstruments(
        _ manifest: ExperimentManifest
    ) -> [String] {
        (manifest.outcomeInstruments ?? []).filter {
            !knownOutcomeInstruments.contains($0)
        }
    }

    /// The plain-language problem for a run-start refusal, or nil. One rule,
    /// both engines (server twin:
    /// `experiment_store.unknown_outcome_instrument_problem`) — the sentence is
    /// the cross-engine contract because the claim is the same claim.
    static func unknownOutcomeInstrumentProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        let unknown = unknownOutcomeInstruments(manifest)
        guard !unknown.isEmpty else { return nil }
        let named = unknown.map { "'\($0)'" }.joined(separator: ", ")
        return "outcomeInstruments declares \(named), which this engine does "
            + "not implement — the declared instruments are read by set "
            + "membership, so an unrecognised value dispatches nothing and the "
            + "study would complete having measured only the default sampled "
            + "text. Known instruments: "
            + knownOutcomeInstruments.joined(separator: ", ")
    }

    /// THE repair for an unknown instrument, on both engines: `set-instruments`
    /// is authoring, and authoring is Mac-authority (audit §10.x), so the
    /// server's copy of this refusal names this binary too.
    static func unknownOutcomeInstrumentRepair(_ name: String) -> String {
        "steerlab-cli experiment set-instruments \(name) <"
            + knownOutcomeInstruments.joined(separator: "|") + ">[,…]"
    }

    /// The closed `ordinalAggregation` vocabulary (server
    /// `manifest.KNOWN_ORDINAL_AGGREGATIONS` twin) — the declared collapse
    /// of the ladder distribution for the `ordinalScale` instrument.
    public static let knownOrdinalAggregations = ["expectedValue", "argmax"]
    /// The closed judge-`kind` vocabulary verify() enforces (server twin:
    /// the same three strings). Named here so an AUTHORING path can refuse a
    /// typo at the point of writing instead of at the next verify.
    public static let knownJudgeKinds = ["claude", "local", "openrouter"]

    /// The one draft-edit gate every science-manifest setter goes through:
    /// load, refuse non-draft with the immutability line, mutate, save.
    ///
    /// Setters that go through here are load-fresh by construction, so the
    /// stale-copy hazard `save`'s arms guard exists for does not apply to
    /// them — but a setter whose whole job is to REMOVE the last arm still
    /// has to say so, which is what `mayClearArms` passes on.
    @discardableResult
    static func updateDraft(
        name: String, mayClearArms: Bool = false,
        _ mutate: (inout ExperimentManifest) throws -> Void
    ) throws -> ExperimentManifest {
        var manifest = try load(name: name)
        guard manifest.status == .draft else {
            throw ExperimentError.refusing(
                .statusImmutable,
                "experiment '\(name)' is \(manifest.status.rawValue) — "
                    + "duplicate it to iterate",
                repair: duplicateToIterateRepair(name))
        }
        try mutate(&manifest)
        try save(manifest, mayClearArms: mayClearArms)
        return manifest
    }

    /// THE repair for an arms-cleared refusal (open-issues §8), as runnable
    /// commands. The first is diagnostic — it prints what the manifest on
    /// disk still holds, which is the fact a caller writing a stale document
    /// does not know.
    static func clearedArmsRepair(_ name: String) -> String {
        "steerlab-cli experiment verify \(name)  "
            + "# the manifest on disk still holds its arms; re-attach what "
            + "the caller dropped (steerlab-cli experiment attach \(name) "
            + "<concept>… ; steerlab-cli experiment declare-condition "
            + "\(name) …), or author the cleared study as its own draft "
            + "with steerlab-cli experiment duplicate \(name) \(name)-v2"
    }

    /// THE repair for every frozen-manifest refusal, as a runnable command
    /// pair (WP0 step 7). "Duplicate it to iterate" is the correct rule and
    /// was, until now, prose: an agent had to guess the verb, the argument
    /// order, and that the copy is what it then edits.
    static func duplicateToIterateRepair(_ name: String) -> String {
        "steerlab-cli experiment duplicate \(name) \(name)-v2 && "
            + "steerlab-cli experiment <the verb you just ran> \(name)-v2 …  "
            + "(frozen studies are immutable; the duplicate is a draft again)"
    }

    /// Persist the researcher's declared study type (the top-of-page
    /// picker). Draft-only, and it keeps the engine-facing `studyKind`
    /// consistent — the type is authoring vocabulary, but multi-agent vs
    /// model-output selects the run path.
    @discardableResult
    public static func setStudyType(
        _ type: StudyIntent, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            manifest.studyType = type.rawValue
            manifest.studyKind = type.mappedKind
        }
    }

    /// The draft's model-revision pin (audit 2026-08-01: create-time only
    /// before — changing it required duplicate-or-paste-JSON). nil / empty
    /// clears back to the auto-pin path (resolved at freeze from the local
    /// HF cache, or by --revision). Draft-only like every setter, and safe
    /// to change BECAUSE evidence freshness is scope-matched by revision:
    /// validate evidence for the old revision reclassifies as stale rather
    /// than silently carrying over to the new one.
    @discardableResult
    public static func setModelRevision(
        _ revision: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = revision?.trimmingCharacters(in: .whitespacesAndNewlines)
            manifest.modelRevision = trimmed?.isEmpty == false ? trimmed : nil
        }
    }

    /// nil / empty clears the field.
    @discardableResult
    public static func setPhase(
        _ phase: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = phase?.trimmingCharacters(in: .whitespacesAndNewlines)
            manifest.phase = trimmed?.isEmpty == false ? trimmed : nil
        }
    }

    @discardableResult
    public static func setCaseFamily(
        _ caseFamily: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = caseFamily?.trimmingCharacters(in: .whitespacesAndNewlines)
            manifest.caseFamily = trimmed?.isEmpty == false ? trimmed : nil
        }
    }

    /// Declare the sweep's SELECTION CRITERION on a draft (WP0 step 7, punch
    /// list #1 P3).
    ///
    /// `sweep.selection` was manifest data with no authoring path outside the
    /// Optimizations panel: a headless caller could not declare it, so the
    /// sweep always resolved to `markerDensity` — the objective the methods
    /// note forbids for decision studies. The rule the document is most
    /// emphatic about was, in practice, unfollowable headlessly.
    ///
    /// Draft-only through `updateDraft`, and validated at DECLARATION through
    /// the same two checks the panel's save path runs (`validateSelection`
    /// for the criterion's shape and ranges, `validateObjectiveRequirements`
    /// for the instrument's own pins), so a criterion that could never arm is
    /// refused here rather than at sweep start after a model has loaded.
    /// A criterion whose METRIC is legal but unimplemented on this engine
    /// saves — declaring ahead is deliberate — and the sweep refuses at start.
    @discardableResult
    public static func setSweepSelection(
        _ selection: ExperimentManifest.SweepSelection?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            guard let selection else {
                manifest.sweep?.selection = nil
                return
            }
            var spec = manifest.sweep ?? ExperimentManifest.SweepSpec()
            spec.selection = selection
            spec = SweepSpecForm.workspaceRelativeNormalized(spec)
            if case .invalid(let reason) = SweepSpecForm.validateSelection(
                spec.selection)
            {
                throw ExperimentError.refusing(
                    .sweepSelectionRule, reason,
                    repair: "steerlab-cli experiment set-sweep-selection "
                        + "\(experimentName) --objective "
                        + SweepSelectionRule.implementedMetrics.joined(
                            separator: "|"))
            }
            if let problem = SweepSpecForm.validateObjectiveRequirements(
                spec.selection, manifest: manifest)
            {
                throw ExperimentError.refusing(
                    .sweepSelectionRule, problem,
                    repair: "steerlab-cli experiment pin-rubric "
                        + "\(experimentName) <prompts/rubrics/file.md> "
                        + "--judges <name>:local  (judgeScore), or "
                        + "steerlab-cli experiment set-sweep-selection "
                        + "\(experimentName) --objective logprobShift "
                        + "--choice-prompts <a readable choice JSONL> "
                        + "(logprobShift)")
            }
            manifest.sweep = spec
        }
    }

    /// nil / empty = the engine default (sampled text only). The declared
    /// list is provenance: it is written explicitly, never inferred from the
    /// prompt data (fields preserved ≠ measurement enabled).
    @discardableResult
    public static func setOutcomeInstruments(
        _ instruments: [String]?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let cleaned = (instruments ?? [])
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if let unknown = cleaned.first(where: {
                !knownOutcomeInstruments.contains($0)
            }) {
                throw ExperimentError.malformed(
                    "unknown outcome instrument '\(unknown)' — known: "
                        + knownOutcomeInstruments.joined(separator: ", "),
                    repair: "steerlab-cli experiment set-instruments "
                        + "\(experimentName) <"
                        + knownOutcomeInstruments.joined(separator: "|") + ">[,…]")
            }
            manifest.outcomeInstruments = cleaned.isEmpty ? nil : cleaned
        }
    }

    /// The declared aggregation for the `ordinalScale` instrument. nil /
    /// empty clears the field; a non-empty value must be in
    /// `knownOrdinalAggregations`. The declaration is REQUIRED whenever
    /// ordinalScale is declared (verify refuses it otherwise) — the
    /// instrument-design choice is written down, never silently defaulted.
    @discardableResult
    public static func setOrdinalAggregation(
        _ aggregation: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = aggregation?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = trimmed, !value.isEmpty else {
                manifest.ordinalAggregation = nil
                return
            }
            guard knownOrdinalAggregations.contains(value) else {
                throw ExperimentError.malformed(
                    "unknown ordinalAggregation '\(value)' — known: "
                        + knownOrdinalAggregations.joined(separator: ", "),
                    repair: "steerlab-cli experiment set-instruments "
                        + "\(experimentName) ordinalScale --ordinal-aggregation <"
                        + knownOrdinalAggregations.joined(separator: "|") + ">")
            }
            manifest.ordinalAggregation = value
        }
    }

    /// The opt-in acknowledgement that scored options tokenize to unequal
    /// lengths (the run loop refuses unequal option sets otherwise). Stored
    /// as true or ABSENT — never an explicit false (content-hash hygiene).
    @discardableResult
    public static func setAcknowledgeUnequalOptionLengths(
        _ acknowledged: Bool, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            manifest.acknowledgeUnequalOptionLengths = acknowledged ? true : nil
        }
    }

    /// Sampling policy. samplesPerItem ≤ 1 normalizes to ABSENT (the engine
    /// default is 1); seedPolicy must be a known policy or empty (absent).
    /// The server-only-stochastic rule is surfaced by the UI, not enforced
    /// here — a local draft may legitimately declare a stochastic design it
    /// will run on the server.
    @discardableResult
    public static func setSamplingPolicy(
        samplesPerItem: Int?, seedPolicy: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            if let samples = samplesPerItem, samples < 1 {
                throw ExperimentError(
                    reason: "samplesPerItem must be ≥ 1 — got \(samples)")
            }
            let policy = seedPolicy?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let policy, !policy.isEmpty, !knownSeedPolicies.contains(policy) {
                throw ExperimentError(
                    reason: "unknown seedPolicy '\(policy)' — known: "
                        + knownSeedPolicies.joined(separator: ", "))
            }
            manifest.samplesPerItem =
                (samplesPerItem ?? 1) > 1 ? samplesPerItem : nil
            manifest.seedPolicy = policy?.isEmpty == false ? policy : nil
        }
    }

    /// The layer(s) convergent validity reads at (D4) — declared, never
    /// inferred. At most one of the parameters may be non-nil; all nil
    /// clears the declaration back to the legacy rule (condition layer,
    /// else mid-network). A one-element list is stored as the scalar field,
    /// so a UI that always passes lists produces the exact manifests the
    /// scalar era did. Refusals reuse `ValidationLayerRule.violation` — the
    /// same rule verify and both engines' resolvers apply — so the UI
    /// cannot save a declaration the run would refuse. The out-of-depth
    /// check (`rangeRefusal`) stays at resolve time: no model is loaded
    /// here, so depth is unknowable.
    @discardableResult
    public static func setValidationReadDepth(
        layer: Int? = nil, fraction: Double? = nil,
        layers: [Int]? = nil, fractions: [Double]? = nil,
        experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            if let problem = ValidationLayerRule.violation(
                declaredLayer: layer, declaredFraction: fraction,
                declaredLayers: layers, declaredFractions: fractions)
            {
                throw ExperimentError(reason: problem)
            }
            manifest.validationLayer = layers?.count == 1 ? layers?.first : layer
            manifest.validationLayerFraction =
                fractions?.count == 1 ? fractions?.first : fraction
            manifest.validationLayers = (layers?.count ?? 0) > 1 ? layers : nil
            manifest.validationLayerFractions =
                (fractions?.count ?? 0) > 1 ? fractions : nil
        }
    }

    /// The manifest's fixed seed list (audit 2026-08-01: only the default
    /// [20260610] and the confirm-draft carry could ever populate it).
    /// Refuses empty (the fixed-list policy indexes into this list) and
    /// duplicates (two identical seeds generate two identical records that
    /// look like independent samples). Order is data — it is preserved.
    @discardableResult
    public static func setSeeds(
        _ seeds: [UInt64], experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            guard !seeds.isEmpty else {
                throw ExperimentError(
                    reason: "the seed list cannot be empty — the fixed-list "
                        + "seed policy indexes into it")
            }
            guard Set(seeds).count == seeds.count else {
                throw ExperimentError(
                    reason: "duplicate seeds — two identical seeds generate "
                        + "two identical records that would masquerade as "
                        + "independent samples")
            }
            manifest.seeds = seeds
        }
    }

    /// A promotionRule with every field empty normalizes to ABSENT.
    @discardableResult
    public static func setPromotionRule(
        _ rule: ExperimentManifest.PromotionRule?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            guard let rule else {
                manifest.promotionRule = nil
                return
            }
            if let threshold = rule.fdrThreshold,
                !(threshold.isFinite && threshold > 0 && threshold < 1)
            {
                throw ExperimentError(
                    reason: "promotionRule fdrThreshold must be in (0, 1) — "
                        + "got \(threshold)")
            }
            let empty = rule.fdrThreshold == nil && rule.doseMonotone == nil
                && rule.exceedsRandomFloor == nil
                && (rule.capabilityGate?.isEmpty ?? true)
            manifest.promotionRule = empty ? nil : rule
        }
    }

    /// Pin the human-effect table (R = delta_model − delta_human) by path +
    /// SHA-256 of its current bytes — hash pinned at SET time, drift after
    /// pinning is a verify()/readiness finding like every other input.
    /// Shape too (Usability Plan Phase 0 item 2): the header must carry the
    /// columns the analyze loader reads, checked NOW — feedback at the
    /// moment of action, not a failure much later at analyze.
    @discardableResult
    public static func pinHumanBaseline(
        path: String, experimentName: String
    ) throws -> ExperimentManifest.HumanBaseline {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ExperimentError(reason: "human-baseline path is empty")
        }
        guard let data = try? Data(contentsOf: resolveProjectPath(trimmed)) else {
            throw ExperimentError(
                reason: "human-baseline file not found: \(trimmed) — author or "
                    + "scaffold it first (Data readiness › human-baseline CSV)")
        }
        if let problem = PinShapeValidation.humanBaselineShapeProblem(
            data, file: trimmed)
        {
            throw ExperimentError(reason: problem)
        }
        let baseline = ExperimentManifest.HumanBaseline(
            path: trimmed, hash: sha256Hex(data))
        try updateDraft(name: experimentName) { manifest in
            manifest.humanBaseline = baseline
        }
        return baseline
    }

    /// Pin the human-validation subset (per-judge vs-human agreement rows;
    /// audit 2026-08-01: the field had no writer in any Swift target — it
    /// arrived only via pasted JSON). Same discipline as every pin: shape
    /// validated NOW (the cross-engine JSONL row contract), hashed at pin
    /// time, drift afterwards is a verify/readiness finding.
    @discardableResult
    public static func pinHumanValidation(
        path: String, experimentName: String
    ) throws -> ExperimentManifest.HumanBaseline {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ExperimentError(reason: "human-validation path is empty")
        }
        guard let data = try? Data(contentsOf: resolveProjectPath(trimmed)) else {
            throw ExperimentError(
                reason: "human-validation file not found: \(trimmed)")
        }
        if let problem = PinShapeValidation.humanValidationShapeProblem(
            data, file: trimmed)
        {
            throw ExperimentError(reason: problem)
        }
        let pin = ExperimentManifest.HumanBaseline(
            path: trimmed, hash: sha256Hex(data))
        try updateDraft(name: experimentName) { manifest in
            manifest.humanValidation = pin
        }
        return pin
    }

    @discardableResult
    public static func clearHumanValidation(
        experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            manifest.humanValidation = nil
        }
    }

    @discardableResult
    public static func clearHumanBaseline(
        experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            manifest.humanBaseline = nil
        }
    }

    // MARK: - Direct concept attach / detach (App gap A8)

    /// What stimulus data exists on disk for a concept name — drives the
    /// attach picker's method choices (paired methods need a
    /// positive/negative set; grand-mean needs stories.jsonl). Existence
    /// checks only; the attach itself re-validates through the real loaders.
    public struct ConceptSources: Sendable, Equatable {
        public let name: String
        /// `prompts/concepts/<name>/` loads as a paired StimulusSet.
        public let hasPairedStimuli: Bool
        /// `prompts/emotions/<name>/stories.jsonl` exists (grand-mean).
        public let hasStories: Bool

        public init(name: String, hasPairedStimuli: Bool, hasStories: Bool) {
            self.name = name
            self.hasPairedStimuli = hasPairedStimuli
            self.hasStories = hasStories
        }

        /// The RECIPE methods this concept's on-disk data can support.
        /// `pinnedArtifact` and `optvec` are excluded by construction — the
        /// former attaches through `attachArtifact` (bytes, not stimuli),
        /// the latter is only ever a pinned artifact's source method.
        public var supportedMethods: [ExtractionMethod] {
            ExtractionMethod.allCases.filter {
                $0.isRecipeMethod && ($0.isPaired ? hasPairedStimuli : hasStories)
            }
        }

        public var pickerLabel: String {
            var kinds: [String] = []
            if hasPairedStimuli { kinds.append("paired") }
            if hasStories { kinds.append("stories") }
            return kinds.isEmpty ? name : "\(name) (\(kinds.joined(separator: " + ")))"
        }
    }

    public static func conceptSources(name: String) -> ConceptSources {
        let pairedDirectory = VectorCatalog.conceptsDirectory.appending(component: name)
        return ConceptSources(
            name: name,
            hasPairedStimuli: (try? StimulusSet(directory: pairedDirectory)) != nil,
            hasStories: storiesHash(for: name) != nil)
    }

    /// One-step concept attach on a DRAFT manifest — the store twin of the
    /// CLI `experiment attach` verb (and of the web client's
    /// `POST /api/experiment/attach`): pins the stimulus hash at its CURRENT
    /// bytes, always writes the measurement-side `validationHash` pin
    /// (through `makeConceptRef`), pins the grand-mean corpus for
    /// emotionGrandMean, and re-pins the neutral corpus (norm denominator)
    /// exactly like the CLI. Draft-only via `updateDraft` — a frozen or
    /// completed study refuses with the immutability line.
    /// Declare a discriminant-validity CONTROL, computing its pin from disk.
    ///
    /// C2 removed the ambient "every concept on disk" control set and told
    /// researchers to declare controls instead — while giving them no way to
    /// do it, so the instruction meant hand-editing JSON and computing a
    /// SHA-256 by hand. This is that missing operation: pick a concept, and
    /// the store reads its stimuli, pins the hash, and records the control's
    /// OWN extraction options.
    @discardableResult
    public static func attachValidationControl(
        concept: String,
        options: ExtractionOptions = .init(),
        experimentName: String
    ) throws -> ExperimentManifest {
        let name = concept.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "control concept name is empty")
        }
        return try updateDraft(name: experimentName) { manifest in
            guard !manifest.concepts.contains(where: { $0.name == name }) else {
                throw ExperimentError(
                    reason: "'\(name)' is already a STUDY concept — a concept "
                        + "cannot be its own discriminant control")
            }
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            let stimuli = try StimulusSet(directory: directory)
            var controls = manifest.validationControls ?? []
            controls.removeAll { $0.concept == name }
            controls.append(
                .init(
                    concept: name,
                    stimulusSetHash: stimuli.hash,
                    options: options,
                    modelRevision: manifest.modelRevision))
            manifest.validationControls = controls.sorted { $0.concept < $1.concept }
        }
    }

    @discardableResult
    public static func removeValidationControl(
        concept: String, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            var controls = manifest.validationControls ?? []
            controls.removeAll { $0.concept == concept }
            manifest.validationControls = controls.isEmpty ? nil : controls
        }
    }

    /// Declare which response formats the option-consuming instruments apply
    /// to, pinning the resulting row set.
    ///
    /// A2's refusal on a mixed-format file says "declare
    /// outcomeInstrumentScope to apply it to the label rows only" — advice
    /// the app could not follow. This computes the pin from the study's own
    /// task prompts so the researcher picks formats, not hashes.
    @discardableResult
    public static func declareOutcomeInstrumentScope(
        responseFormats: [String], experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            guard let file = manifest.taskPromptsFile, !file.isEmpty else {
                throw ExperimentError(
                    reason: "declare the task prompts first ('steerlab-cli "
                        + "experiment pin-prompts \(experimentName) "
                        + "prompts/…/file.jsonl') — the scope pins which of "
                        + "THEIR rows the instrument reads")
            }
            let data = try Data(contentsOf: resolveProjectPath(file))
            let document = try TaskPromptsDocument.load(data)
            let items = document.responseFormatItems
            guard !responseFormats.isEmpty else {
                manifest.outcomeInstrumentScope = nil
                return
            }
            manifest.outcomeInstrumentScope = ResponseFormat.Scope.pin(
                responseFormats: responseFormats, items: items)
        }
    }

    /// THE agent → condition constructor: an agent record becomes a study arm
    /// pinned by workspace-relative path + artifact hash, with the artifact
    /// itself snapshotted into the manifest so a later edit to the library
    /// cannot silently change a frozen study.
    ///
    /// Extracted from the Studies panel (2026-08-06) because study TEMPLATES
    /// mint the same arms headlessly. Two implementations of "add an agent"
    /// would drift, and the one the researcher clicks is not the one the
    /// paper's mint used — so both go through here.
    public static func agentCondition(
        for record: ModelVariantRecord
    ) throws -> ExperimentManifest.VariantCondition {
        ExperimentManifest.VariantCondition(
            name: record.artifact.name,
            artifactPath: ModelVariantStore.relativePath(for: record),
            artifactHash: try ModelVariantStore.hash(record.url),
            artifact: record.artifact)
    }

    /// Appends (or replaces) one agent arm. Replacement matches the panel's
    /// rule exactly: same artifact path OR same agent name, so re-adding an
    /// agent re-pins it rather than producing two arms that generate twice
    /// and analyze as separate conditions.
    ///
    /// Refuses an agent built on a different base model than the study: the
    /// arms of a comparison must differ by the intervention, not by which
    /// model produced the text.
    public static func attachAgent(
        _ record: ModelVariantRecord, into manifest: inout ExperimentManifest
    ) throws {
        guard record.artifact.baseModelID == manifest.modelID else {
            throw ExperimentError(
                reason: "agent '\(record.artifact.name)' uses "
                    + "\(record.artifact.baseModelID), not this study's base "
                    + "model \(manifest.modelID)")
        }
        let condition = try agentCondition(for: record)
        manifest.variantConditions.removeAll {
            $0.artifactPath == condition.artifactPath || $0.name == condition.name
        }
        manifest.variantConditions.append(condition)
    }

    @discardableResult
    public static func attachConcept(
        _ concept: String,
        method: ExtractionMethod,
        poolFromToken: Int? = nil,
        corpusConcepts: [String] = [],
        reference: String? = nil,
        experimentName: String
    ) throws -> ExperimentManifest {
        let name = concept.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "concept name is empty")
        }
        guard method.isRecipeMethod else {
            // Same rule as the server's attach dispatcher: optvec is a
            // SOURCE method, never an attachable recipe; a pinned artifact
            // attaches through attachArtifact with the bytes as the recipe.
            throw ExperimentError(
                reason: "'\(method.rawValue)' is not an attachable recipe "
                    + "method — attach a vector ARTIFACT (Attach Artifact / "
                    + "attach-artifact), which pins the bytes instead of a "
                    + "stimulus recipe")
        }
        if let token = poolFromToken, token < 0 {
            throw ExperimentError(
                reason: "pool-from expects a token index ≥ 0 — got \(token)")
        }
        return try updateDraft(name: experimentName) { manifest in
            if method == .emotionGrandMean {
                try attachGrandMeanConcepts(
                    [name],
                    corpusConcepts: corpusConcepts,
                    poolFromToken: poolFromToken,
                    into: &manifest)
            } else if method == .designatedReference {
                // mean(concept stories) − mean(REFERENCE stories), both
                // pooled: the reference pins beside the concept, and the
                // pooled reading is the method's POLICY — token 50 unless
                // deliberately overridden — because a last-token read on
                // paragraph stories extracts closing-sentence content.
                let refName = (reference ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !refName.isEmpty else {
                    throw ExperimentError(
                        reason: "designatedReference needs a reference stories "
                            + "concept — the reference corpus is part of the recipe")
                }
                guard let conceptHash = storiesHash(for: name) else {
                    throw ExperimentError(
                        reason: "no stories.jsonl for concept '\(name)' under "
                            + "prompts/emotions/")
                }
                guard let refHash = storiesHash(for: refName) else {
                    throw ExperimentError(
                        reason: "no stories.jsonl for reference '\(refName)' under "
                            + "prompts/emotions/")
                }
                var options = ExtractionOptions(method: method)
                options.readingPosition = .meanFromToken(poolFromToken ?? 50)
                manifest.concepts.removeAll { $0.name == name }
                var ref = makeConceptRef(
                    name: name, stimulusSetHash: conceptHash, options: options)
                ref.designatedReference = .init(name: refName, hash: refHash)
                manifest.concepts.append(ref)
            } else {
                let directory = VectorCatalog.conceptsDirectory.appending(component: name)
                let stimuli = try StimulusSet(directory: directory)
                var options = ExtractionOptions(method: method)
                if let token = poolFromToken {
                    options.readingPosition = .meanFromToken(token)
                }
                manifest.concepts.removeAll { $0.name == name }
                manifest.concepts.append(
                    makeConceptRef(
                        name: name, stimulusSetHash: stimuli.hash, options: options))
            }
            // The corpus is a pinned input whenever it exists: it denominates
            // norm-unit alphas (same rule as the CLI attach).
            pinNeutralCorpus(into: &manifest)
        }
    }

    /// Pin an EXISTING vector artifact into a draft manifest as a concept
    /// (method `pinnedArtifact`; cross-engine contract key `vectorArtifact`).
    /// Swift mirror of the server's `experiment_store.attach_artifact`
    /// (commit af1af0e promised "Swift will mirror them" — this is the
    /// mirror), with one adaptation: the server refuses artifacts from a
    /// substrate other than ITSELF, because it is also the engine that will
    /// steer; here the engine that will steer is the WORKSPACE's declared
    /// compute substrate (`computeSubstrate`), so a python-hf OptVec
    /// artifact attaches cleanly in a cluster workspace and is refused in a
    /// local-MLX one.
    ///
    /// Every other attach form pins a RECIPE and lets each run re-derive the
    /// vector. Some legitimate directions have no such recipe — family-
    /// grand-mean centring, and OPTVEC directions (optimized by backprop
    /// against hashed datasets: no stimuli, no source concept, no
    /// validation.jsonl). Pinning the BYTES is then the honest firewall:
    /// this records the workspace-relative extension-less locator plus the
    /// SHA-256 of BOTH files, and verify/freeze re-check them against the
    /// bytes on disk. The pin must be one verify() could pass the moment it
    /// is written — attach never records a hash the very next verify would
    /// reject.
    @discardableResult
    public static func attachArtifact(
        _ concept: String,
        artifact reference: String,
        sourceConcept: String? = nil,
        evalRun: String? = nil,
        experimentName: String
    ) throws -> ExperimentManifest {
        let name = concept.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "concept name is empty")
        }
        // Extension-less locator: accept either file's path and strip it.
        var trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        for suffix in [".safetensors", ".json"] where trimmed.hasSuffix(suffix) {
            trimmed = String(trimmed.dropFirst(0).dropLast(suffix.count))
        }
        guard !trimmed.isEmpty else {
            throw ExperimentError(reason: "artifact path is empty")
        }
        // Resolve against the ACTIVE workspace root (test seam honoured),
        // then require the reference to come back workspace-relative — the
        // only shape that survives moving the workspace to another machine.
        let resolved = trimmed.hasPrefix("/")
            ? URL(filePath: trimmed).standardizedFileURL
            : workspaceRoot.appending(path: trimmed).standardizedFileURL
        let rel = workspaceRelativePath(resolved.path)
        guard !rel.hasPrefix("/"), !rel.hasPrefix("..") else {
            throw ExperimentError(
                reason: "artifact path '\(reference)' is outside the workspace "
                    + "— pinned inputs must be workspace-relative")
        }
        let tensorURL = resolved.appendingPathExtension("safetensors")
        let sidecarURL = resolved.appendingPathExtension("json")
        for url in [tensorURL, sidecarURL]
        where !FileManager.default.fileExists(atPath: url.path) {
            throw ExperimentError(
                reason: "no vector artifact at '\(rel)' — expected both "
                    + "\(rel).safetensors and \(rel).json (the extension-less "
                    + "path is the artifact locator)")
        }
        // Containment is REAL-path, not lexical (review finding 2026-08-10):
        // a symlink inside the workspace can point at bytes outside it while
        // the relative-path check above passes, and a copy of the workspace
        // would then silently lose the pinned bytes. Server twin: the
        // safe_paths.is_contained check in attach_artifact.
        let realRoot = workspaceRoot.resolvingSymlinksInPath().path
        for url in [tensorURL, sidecarURL] {
            let real = url.resolvingSymlinksInPath().path
            if real != realRoot, !real.hasPrefix(realRoot + "/") {
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' resolves outside the "
                        + "workspace (symlink target \(real)) — a pinned "
                        + "input's bytes must live under the workspace root, "
                        + "or a copy of the workspace silently loses them. "
                        + "Move or copy the artifact into the workspace and "
                        + "re-attach")
            }
        }
        let sidecarData = try Data(contentsOf: sidecarURL)
        let sidecar: SteeringVectorSidecar
        let rawSidecar: [String: Any]
        do {
            sidecar = try JSONDecoder().decode(
                SteeringVectorSidecar.self, from: sidecarData)
            rawSidecar = try JSONSerialization.jsonObject(with: sidecarData)
                as? [String: Any] ?? [:]
        } catch {
            throw ExperimentError(
                reason: "vector artifact sidecar '\(rel).json' is not readable "
                    + "as a sidecar: \(error)")
        }
        return try updateDraft(name: experimentName) { manifest in
            try attachArtifactPin(
                concept: name, rel: rel, tensorURL: tensorURL,
                sidecarData: sidecarData, sidecar: sidecar,
                rawSidecar: rawSidecar, sourceConcept: sourceConcept,
                evalRun: evalRun, into: &manifest)
        }
    }

    /// The refusal-heavy body of `attachArtifact`, split out so the checks
    /// read in the server's order (`attach_artifact`, af1af0e) and stay
    /// diffable against it.
    private static func attachArtifactPin(
        concept name: String, rel: String, tensorURL: URL,
        sidecarData: Data, sidecar: SteeringVectorSidecar,
        rawSidecar: [String: Any], sourceConcept: String?, evalRun: String?,
        into manifest: inout ExperimentManifest
    ) throws {
        if sidecar.modelID != manifest.modelID {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' was extracted on model "
                    + "\(sidecar.modelID), not this study's "
                    + "\(manifest.modelID) — a direction does not transfer "
                    + "between models")
        }
        if let pinned = manifest.modelRevision, let recorded = sidecar.revision,
            recorded != pinned
        {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' was extracted at revision "
                    + "\(recorded.prefix(12))…, not this study's pinned "
                    + "\(pinned.prefix(12))…")
        }
        let native = computeSubstrate
        if let substrate = sidecar.substrate, substrate != native {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' was extracted on substrate "
                    + "'\(substrate)'; this workspace computes on '\(native)' "
                    + "— steering vectors do not transfer across engines")
        }
        guard let sourceMethodRaw = sidecar.extractionMethod,
            !sourceMethodRaw.isEmpty
        else {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' records no extractionMethod "
                    + "— without it the study cannot know what its held-out "
                    + "validation MEANS (contrastive vs population)")
        }
        guard !sidecar.stimulusSetHash.isEmpty else {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' records no stimulusSetHash "
                    + "— the direction's data provenance is unpinnable")
        }
        guard let normSource = sidecar.residualNormSource, !normSource.isEmpty
        else {
            if sourceMethodRaw == ExtractionMethod.optvec.rawValue {
                // Not a defect of the artifact but a missing LIFECYCLE STEP
                // (plan §6): the training driver deliberately writes no
                // norms, so the refusal names the verb that supplies them.
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' is an OptVec vector "
                        + "with no residualNormSource — an optvec vector is "
                        + "BORN without one. Run the residual-norm backfill "
                        + "against the pinned neutral corpus first "
                        + "(POST /api/vectors/backfill-norms) and attach the "
                        + "BACKFILLED artifact: α in norm units is "
                        + "meaningless until the denominator is measured")
            }
            throw ExperimentError(
                reason: "vector artifact '\(rel)' records no "
                    + "residualNormSource — its norm denominator (and so its "
                    + "recipe identity, which promotion matches on) cannot "
                    + "be proved")
        }
        guard let method = ExtractionMethod(rawValue: sourceMethodRaw) else {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' records extractionMethod "
                    + "'\(sourceMethodRaw)', which this engine does not know "
                    + "— it cannot resolve where the concept's held-out data "
                    + "lives")
        }
        if method.isPinnedArtifact {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' is itself a materialized "
                    + "pinned artifact — pin the ORIGINAL it names in "
                    + "pinnedFrom, so the study cites the bytes' actual origin")
        }
        let requestedSource = (sourceConcept ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let dataConcept = requestedSource.isEmpty ? name : requestedSource
        var optvecBlock: [String: Any]?
        if !method.hasSourceConcept {
            // Two families invert (or sidestep) the pipeline and have no
            // source concept anywhere in them: an OptVec direction is
            // behavior → vector with no stimulus set, and an imported Gemma
            // Scope SAE decoder row is a coordinate in a published
            // dictionary. Every data-side question below must be SKIPPED
            // rather than answered with an invention; the artifact's own
            // identity and the provenance that says where it came from are
            // not skipped. Server twin: the `has_source_concept` branch of
            // `attach_artifact`.
            if !requestedSource.isEmpty, requestedSource != name {
                let kind = method.isOptvec
                    ? "an OptVec vector"
                    : "an imported Gemma Scope SAE decoder row"
                let evidence = method.isOptvec
                    ? "the OptVec eval run (eval.json)"
                    : "the pinned SAE candidate roster's discovery snapshot "
                        + "and qualification artifact"
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' is \(kind), which has "
                        + "no source concept — '\(requestedSource)' names "
                        + "stimuli and a held-out validation.jsonl that play "
                        + "no part in it. Attach it under the study's own "
                        + "concept name; its evidence is \(evidence), not a "
                        + "concept's held-out set")
            }
        }
        if method.isOptvec {
            guard let block = rawSidecar["optvec"] as? [String: Any],
                !block.isEmpty
            else {
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' declares "
                        + "extractionMethod 'optvec' but carries no 'optvec' "
                        + "provenance block — a stripped sidecar cannot say "
                        + "WHAT was optimized (objective, λs, datasets, seed, "
                        + "chosen checkpoint, training run), and an optvec "
                        + "vector with no recorded objective certifies "
                        + "nothing. Re-attach the artifact the training run "
                        + "wrote (or its norm-backfilled copy, which "
                        + "preserves the block verbatim)")
            }
            optvecBlock = block
        }
        // The pin must be one verify() could pass the moment it is written.
        let live: String?
        let where_: String
        if !method.hasSourceConcept {
            // The identity hash travels VERBATIM: nothing under prompts/
            // compares against it — for optvec the composite
            // "optvec:<sha256>" over the split files (pinned in the training
            // run), for a Gemma Scope import the
            // "gemmascope:<release>:<saeID>:<feature>" dictionary coordinate.
            live = sidecar.stimulusSetHash
            where_ = method.isOptvec
                ? "the OptVec training run's pinned dataset splits"
                : "the published Gemma Scope dictionary the feature was "
                    + "imported from"
        } else if method.usesStoryCorpus {
            live = storiesHash(for: dataConcept)
            where_ = "prompts/emotions/\(dataConcept)/stories.jsonl"
        } else {
            let directory = VectorCatalog.conceptsDirectory
                .appending(component: dataConcept)
            live = (try? StimulusSet(directory: directory))?.hash
            where_ = "prompts/concepts/\(dataConcept)/"
        }
        guard let liveHash = live else {
            let centring = rawSidecar["familyGrandMeanCentring"]
                as? [String: Any]
            let hint = (centring?["baseConcept"] as? String).map {
                " (the artifact names base concept '\($0)' — try attaching "
                    + "with source concept '\($0)')"
            } ?? ""
            throw ExperimentError(
                reason: "no stimulus data at \(where_) for concept "
                    + "'\(dataConcept)', so the study could never validate "
                    + "'\(name)' — pass the concept the artifact was derived "
                    + "from\(hint)")
        }
        if liveHash != sidecar.stimulusSetHash {
            throw ExperimentError(
                reason: "vector artifact '\(rel)' was extracted from stimuli "
                    + "hashing \(sidecar.stimulusSetHash.prefix(12))…, but "
                    + "\(where_) now hashes \(liveHash.prefix(12))… — restore "
                    + "the bytes the artifact was built on, or pass the right "
                    + "source concept")
        }
        // Reading position: copied from the sidecar so held-out activations
        // are read where the vector was read; an unparseable label refuses
        // (verify compares the labels, and attach must not write a pin the
        // next verify rejects).
        var options = ExtractionOptions(method: .pinnedArtifact)
        if let recorded = sidecar.readingPosition {
            guard let position = ReadingPosition(label: recorded) else {
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' records reading "
                        + "position '\(recorded)', which this engine cannot "
                        + "parse — re-attach on the engine that wrote it")
            }
            options.readingPosition = position
        }
        var pin = ExperimentManifest.ConceptRef.VectorArtifactPin(
            path: rel,
            sha256TensorHash: sha256Hex(try Data(contentsOf: tensorURL)),
            sha256SidecarHash: sha256Hex(sidecarData),
            sourceMethod: sourceMethodRaw,
            sourceConcept: dataConcept,
            residualNormSource: normSource,
            normCorpusHash: sidecar.neutralCorpusHash)
        if let optvecBlock {
            // The optimization's own identity, copied so the manifest is
            // self-describing. Absent keys stay absent — never a guessed
            // reference.
            pin.optvecLayer = optvecBlock["layer"] as? Int
            pin.optvecTrainingRun = optvecBlock["runID"] as? String
            pin.optvecSeed = optvecBlock["seed"] as? Int
            if let recorded = evalRun ?? recordedOptvecEvalRun(optvecBlock) {
                // Evidence is not trusted by name (review finding
                // 2026-08-10): resolve the citation and check the eval run
                // certifies THIS artifact — mirror of the server's
                // `_verify_optvec_eval_run`.
                let verification = try verifyOptvecEvalRun(
                    recorded, tensorHash: pin.sha256TensorHash)
                pin.optvecEvalRun = recorded
                pin.optvecEvalRunVerified = verification.verified
                if !verification.verified {
                    pin.optvecEvalRunUnverifiedReason = verification.reason
                }
            }
        }
        // A source-concept-less concept (optvec, imported SAE feature) pins
        // validation EXPLICITLY NULL (never merely absent, which reads as a
        // legacy attach): there is no held-out validation.jsonl to pin, and
        // a file appearing later under this name would be unrelated to the
        // direction.
        let validationHash = !method.hasSourceConcept
            ? nil
            : conceptValidationHash(
                name: dataConcept, isPaired: !method.usesStoryCorpus)
        var ref = ExperimentManifest.ConceptRef(
            name: name, stimulusSetHash: sidecar.stimulusSetHash,
            options: options, validationHash: validationHash,
            validationHashPinnedAbsent: validationHash == nil,
            vectorArtifact: pin)
        if method == .designatedReference {
            guard let reference = sidecar.designatedReference,
                let refName = reference["name"], !refName.isEmpty,
                let refHash = reference["hash"], !refHash.isEmpty
            else {
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' is a designated-"
                        + "reference direction but records no "
                        + "designatedReference {name, hash} — the reference "
                        + "is part of what validation compares against")
            }
            let liveRef = storiesHash(for: refName)
            guard liveRef == refHash else {
                throw ExperimentError(
                    reason: "the artifact's reference '\(refName)' stories "
                        + "hash \(refHash.prefix(12))… does not match the "
                        + "bytes on disk (\((liveRef ?? "missing").prefix(12))…) "
                        + "— restore them before pinning the artifact")
            }
            ref.designatedReference = .init(name: refName, hash: refHash)
        }
        if method.isGrandMean {
            guard let population = sidecar.grandMeanPopulation,
                !population.isEmpty
            else {
                throw ExperimentError(
                    reason: "vector artifact '\(rel)' is a grand-mean "
                        + "direction but records no grandMeanPopulation — "
                        + "the population IS the comparison, so validation "
                        + "cannot be reproduced without it")
            }
            var members = manifest.grandMeanCorpus?.concepts ?? []
            var hashes = manifest.grandMeanCorpus?.hashes ?? [:]
            for (member, digest) in population.sorted(by: { $0.key < $1.key }) {
                let liveMember = storiesHash(for: member)
                guard liveMember == digest else {
                    throw ExperimentError(
                        reason: "grand-mean population member '\(member)' "
                            + "hashes \((liveMember ?? "missing").prefix(12))… "
                            + "on disk but \(digest.prefix(12))… in the "
                            + "artifact — restore the bytes the artifact was "
                            + "built on")
                }
                if !members.contains(member) { members.append(member) }
                hashes[member] = digest
            }
            manifest.grandMeanCorpus = ExperimentManifest.GrandMeanCorpus(
                concepts: members, hashes: hashes)
        }
        manifest.concepts.removeAll { $0.name == name }
        manifest.concepts.append(ref)
        pinNeutralCorpus(into: &manifest)
    }

    /// An absolute path under the ACTIVE workspace root (test override
    /// honoured — `ArtifactIdentity.workspaceRelative` deliberately keys on
    /// the app-level `WorkspaceRoot`, which the `rootOverride` test seam
    /// does not move), made workspace-relative; anything else passes
    /// through unchanged. Symlinked roots (macOS `/var` → `/private/var`,
    /// which is exactly where test temp roots live) are handled the same
    /// way `ArtifactIdentity.canonical` handles them.
    static func workspaceRelativePath(_ reference: String) -> String {
        guard reference.hasPrefix("/") else { return reference }
        func tail(of path: String, under root: String) -> String? {
            let prefix = root.hasSuffix("/") ? root : root + "/"
            guard path.hasPrefix(prefix), path.count > prefix.count
            else { return nil }
            return String(path.dropFirst(prefix.count))
        }
        let root = workspaceRoot.standardizedFileURL
        let target = URL(filePath: reference).standardizedFileURL
        if let relative = tail(of: target.path, under: root.path) {
            return relative
        }
        let resolvedTarget = target.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appending(component: target.lastPathComponent)
        if let relative = tail(
            of: resolvedTarget.path,
            under: root.resolvingSymlinksInPath().path)
        {
            return relative
        }
        return reference
    }

    /// The eval run an OptVec sidecar's provenance block already names, if
    /// any (server twin: `_recorded_optvec_eval_run` — same key vocabulary).
    static func recordedOptvecEvalRun(_ block: [String: Any]) -> String? {
        for key in ["evalRun", "evalRunID", "evalRunId", "evalRunDirectory"] {
            guard let value = block[key] else { continue }
            if let text = value as? String,
                !text.trimmingCharacters(in: .whitespaces).isEmpty
            {
                return text.trimmingCharacters(in: .whitespaces)
            }
            if let nested = value as? [String: Any] {
                for inner in ["runID", "runDirectory", "path"] {
                    if let text = nested[inner] as? String,
                        !text.trimmingCharacters(in: .whitespaces).isEmpty
                    {
                        return text.trimmingCharacters(in: .whitespaces)
                    }
                }
            }
        }
        return nil
    }

    /// Resolve a named OptVec eval run and check it certifies THIS artifact
    /// (server twin: `_verify_optvec_eval_run`, review finding 2026-08-10 —
    /// evidence must not be trusted by name). Throws on a reference that
    /// names no run directory (a typo'd citation is an input error) and on
    /// an eval.json that certifies a DIFFERENT tensor; returns unverified-
    /// with-reason for a run directory without a readable eval.json (a
    /// crashed or partially imported eval may legitimately complete later).
    static func verifyOptvecEvalRun(
        _ reference: String, tensorHash: String
    ) throws -> (verified: Bool, reason: String?) {
        var candidates: [URL] = []
        if reference.hasPrefix("/") {
            candidates.append(URL(filePath: reference))
        } else {
            candidates.append(workspaceRoot.appending(path: reference))
            let basename = URL(filePath: reference).lastPathComponent
            candidates.append(
                workspaceRoot.appending(components: "runs", basename))
        }
        var isDirectory: ObjCBool = false
        guard
            let runURL = candidates.first(where: {
                FileManager.default.fileExists(
                    atPath: $0.path, isDirectory: &isDirectory)
                    && isDirectory.boolValue
            })
        else {
            throw ExperimentError(
                reason: "OptVec eval run '\(reference)' names no run "
                    + "directory in this workspace — the eval evidence must "
                    + "exist where it is cited. Import the eval run (or fix "
                    + "the reference), then re-attach")
        }
        let evalURL = runURL.appending(component: "eval.json")
        guard FileManager.default.fileExists(atPath: evalURL.path) else {
            return (
                false,
                "run directory exists but has no eval.json (crashed or "
                    + "partially imported eval run)"
            )
        }
        let payload: [String: Any]
        do {
            payload = try JSONSerialization.jsonObject(
                with: Data(contentsOf: evalURL)) as? [String: Any] ?? [:]
        } catch {
            return (false, "eval.json is unreadable: \(error)")
        }
        guard
            let recorded = (payload["artifact"] as? [String: Any])?[
                "tensorSHA256"] as? String,
            !recorded.trimmingCharacters(in: .whitespaces).isEmpty
        else {
            return (
                false,
                "eval.json records no artifact.tensorSHA256 (pre-identity "
                    + "eval schema)"
            )
        }
        let cited = recorded.trimmingCharacters(in: .whitespaces).lowercased()
        guard cited == tensorHash.lowercased() else {
            throw ExperimentError(
                reason: "OptVec eval run '\(reference)' evaluated tensor "
                    + "\(cited.prefix(12))…, not this artifact's "
                    + "\(tensorHash.lowercased().prefix(12))… — it is "
                    + "evidence for a DIFFERENT direction. Name the eval run "
                    + "that read THIS artifact's test split (or run the eval "
                    + "verb on it first)")
        }
        return (true, nil)
    }

    /// Removes a pinned concept from a DRAFT manifest. Refuses while any
    /// condition still references it (verify() would flag the dangling slot
    /// anyway — refusing here keeps the draft sound instead of quietly
    /// breaking it). When the last grand-mean target leaves, the pinned
    /// grand-mean corpus goes with it (nothing left to define); a corpus
    /// wider than the remaining targets is deliberately kept — the
    /// population is part of the remaining vectors' recipe.
    @discardableResult
    public static func detachConcept(
        _ concept: String, experimentName: String
    ) throws -> ExperimentManifest {
        // Declared intent (open-issues §8): detaching the last concept of a
        // condition-less draft legitimately lands on both-empty. It is a
        // researcher removing one named arm, one call at a time — not a
        // document arriving from somewhere stale.
        try updateDraft(name: experimentName, mayClearArms: true) { manifest in
            guard manifest.concepts.contains(where: { $0.name == concept }) else {
                throw ExperimentError(
                    reason: "concept '\(concept)' is not attached to '\(experimentName)'")
            }
            let referencing = manifest.conditions
                .filter { $0.slots.contains { $0.concept == concept } }
                .map(\.name)
            guard referencing.isEmpty else {
                throw ExperimentError(
                    reason: "concept '\(concept)' is referenced by condition(s) "
                        + referencing.joined(separator: ", ")
                        + " — remove those conditions first")
            }
            manifest.concepts.removeAll { $0.name == concept }
            if !manifest.concepts.contains(where: { $0.options.method.isGrandMean }) {
                manifest.grandMeanCorpus = nil
            }
        }
    }

    // MARK: - Run-directory lookup (App gap A11)

    /// Newest run directory the given task wrote for an experiment —
    /// `<stamp>-exp-<name>-<task>` with an optional `-N` collision suffix
    /// (`VectorCatalog.makeUniqueRunDirectory` naming). Used to report the
    /// artifact directory of tasks whose API predates returning one
    /// (`ExperimentTasks.extract` prints it but returns Void).
    public static func newestRunDirectory(
        experimentName: String, task: String
    ) -> URL? {
        let pattern = "-exp-\(experimentName)-\(task)(-\\d+)?$"
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        return entries
            .filter {
                $0.lastPathComponent.range(
                    of: pattern, options: .regularExpression) != nil
            }
            .max { $0.lastPathComponent < $1.lastPathComponent }
    }

    // MARK: - Native condition editing (App gap A4)

    /// Adds (or replaces, by name) a condition on a draft manifest, pinning
    /// the neutral corpus like the capture path does (norm denominator).
    @discardableResult
    public static func upsertCondition(
        _ condition: ExperimentManifest.Condition, experimentName: String
    ) throws -> ExperimentManifest {
        let name = condition.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw ExperimentError(reason: "condition name is empty")
        }
        return try updateDraft(name: experimentName) { manifest in
            for slot in condition.slots {
                guard manifest.concepts.contains(where: { $0.name == slot.concept })
                else {
                    throw ExperimentError(
                        reason: "condition '\(name)' references concept "
                            + "'\(slot.concept)', which is not attached to "
                            + "'\(experimentName)' — attach (pin) it first")
                }
            }
            manifest.conditions.removeAll { $0.name == name }
            manifest.conditions.append(condition)
            if !condition.slots.isEmpty {
                pinNeutralCorpus(into: &manifest)
            }
        }
    }

    @discardableResult
    public static func removeCondition(
        named name: String, experimentName: String
    ) throws -> ExperimentManifest {
        // Declared intent, same rule as `detachConcept` (open-issues §8).
        try updateDraft(name: experimentName, mayClearArms: true) { manifest in
            manifest.conditions.removeAll { $0.name == name }
        }
    }

    /// The negative-α counterpart (direction control, Step-5 item 2): same
    /// slots with every alpha negated, deterministic name `<name>-neg`.
    /// Pure — the scaffold and the one-click button share it.
    public static func signControlCondition(
        for condition: ExperimentManifest.Condition
    ) -> ExperimentManifest.Condition {
        var control = condition
        control.name = condition.name + "-neg"
        control.slots = condition.slots.map { slot in
            .init(concept: slot.concept, layer: slot.layer, alpha: -slot.alpha)
        }
        control.controlType = nil
        control.selection = nil
        return control
    }

    /// The matched-norm random control (coherence control, Step-5 item 4):
    /// same slots/alphas, `controlType: randomMatchedNorm` — the run loop
    /// substitutes a deterministic random direction of the SAME norm per
    /// slot. Deterministic name `<name>-random`. Pure.
    public static func matchedNormRandomCondition(
        for condition: ExperimentManifest.Condition
    ) -> ExperimentManifest.Condition {
        var control = condition
        control.name = condition.name + "-random"
        control.controlType = "randomMatchedNorm"
        control.selection = nil
        return control
    }

    /// The ABLATION control: same slots and λ, `controlType:
    /// randomDirectionAblation` — the run loop removes a deterministic random
    /// DIRECTION instead of the concept's. Deterministic name
    /// `<name>-random`. Pure.
    ///
    /// Norm-matching, the steering control's whole mechanism, means nothing
    /// here: a projection removes whatever the residual stream contains along
    /// the direction, so scaling the direction changes nothing at all. The
    /// question an ablation raises is different — is the effect specific to
    /// THIS direction, or does removing any rank-1 subspace of the residual
    /// stream produce it? — and this is the condition that answers it.
    public static func randomDirectionAblationCondition(
        for condition: ExperimentManifest.Condition
    ) -> ExperimentManifest.Condition {
        var control = condition
        control.name = condition.name + "-random"
        control.controlType = "randomDirectionAblation"
        control.selection = nil
        return control
    }

    /// The right random control for a condition, by what the condition does.
    /// Callers should not have to know the vocabulary; attaching a
    /// matched-norm control to an ablation would silently produce a cell that
    /// ablates the CONCEPT (the substitution never fires) and therefore
    /// duplicates the treatment.
    public static func randomControlCondition(
        for condition: ExperimentManifest.Condition
    ) -> ExperimentManifest.Condition {
        condition.slots.contains(where: { $0.effectiveMode == .ablate })
            ? randomDirectionAblationCondition(for: condition)
            : matchedNormRandomCondition(for: condition)
    }

    /// Whether a condition counts as a TREATMENT for scaffolding: it steers
    /// (has slots), is not itself a control cell, and pushes INTO the
    /// concept (all alphas positive — a negative-α condition is already a
    /// direction control).
    static func isTreatmentCondition(
        _ condition: ExperimentManifest.Condition
    ) -> Bool {
        !condition.slots.isEmpty
            && condition.controlType == nil
            && condition.slots.allSatisfy { $0.alpha > 0 }
            && !condition.name.hasSuffix("-neg")
    }

    public struct ControlMatrixScaffold: Sendable, Equatable {
        /// Conditions the scaffold added, in order.
        public var added: [String]
        /// Deterministically-named controls that already existed (idempotence:
        /// the scaffold never duplicates or overwrites).
        public var skipped: [String]
        /// The Step-5 pieces no scaffold can author — data decisions that
        /// stay with the researcher.
        public var notes: [String]
    }

    /// One-click Step-5 control-matrix scaffold over a draft's TREATMENT
    /// conditions: ensures an explicit no-steer baseline, a negative-α
    /// counterpart per treatment, and a matched-norm random control per
    /// treatment. Idempotent — existing conditions (by name) are never
    /// touched — and honest about what remains manual (the sympathy
    /// positive-control concept, the capability battery pin).
    @discardableResult
    public static func scaffoldControlMatrix(
        experimentName: String
    ) throws -> ControlMatrixScaffold {
        var added: [String] = []
        var skipped: [String] = []
        var notes: [String] = []
        let updated = try updateDraft(name: experimentName) { manifest in
            let treatments = manifest.conditions.filter(isTreatmentCondition)
            guard !treatments.isEmpty else {
                throw ExperimentError(
                    reason: "no treatment conditions to scaffold controls for — "
                        + "add at least one positive-α vector condition first")
            }
            func ensure(_ candidate: ExperimentManifest.Condition) {
                if manifest.conditions.contains(where: { $0.name == candidate.name }) {
                    skipped.append(candidate.name)
                } else {
                    manifest.conditions.append(candidate)
                    added.append(candidate.name)
                }
            }
            // Step-5 item 1: an explicit named baseline (any no-slot
            // condition counts — do not force the name).
            if let existing = manifest.conditions.first(where: { $0.slots.isEmpty }) {
                skipped.append(existing.name)
            } else {
                ensure(.init(name: "baseline", slots: []))
            }
            for treatment in treatments {
                ensure(signControlCondition(for: treatment))  // item 2
                ensure(randomControlCondition(for: treatment))  // item 4
            }
        }
        // Items the scaffold CANNOT author — say so instead of pretending.
        notes.append(
            "manual: the relevant-concept POSITIVE control (e.g. sympathy) is "
                + "its own concept + condition — author its stimuli and attach it")
        notes.append(
            "manual: ≥2 positive alphas per treatment (dose-response) — add "
                + "further vector conditions at other alphas")
        if updated.capabilityBatteryFile == nil {
            notes.append(
                "manual: pin a capability battery so every condition is "
                    + "battery-scored inside run (Step-5 item 5)")
        }
        return ControlMatrixScaffold(added: added, skipped: skipped, notes: notes)
    }

    // MARK: - Draft delete (App gap A12)

    /// Moves a DRAFT experiment's directory to a `.trash-<timestamp>` sibling
    /// under experiments/ — never a destructive delete, and never for frozen
    /// or completed studies (immutability). The trashed directory is nested
    /// (`experiments/.trash-<ts>/<name>/…`) so `list()`/`load(name:)` can
    /// never resolve it as a live experiment.
    @discardableResult
    public static func moveDraftToTrash(name: String) throws -> URL {
        let manifest = try load(name: name)
        guard manifest.status == .draft else {
            throw ExperimentError(
                reason: "experiment '\(name)' is \(manifest.status.rawValue) — "
                    + "frozen/completed studies are immutable and cannot be "
                    + "deleted; duplicate to iterate")
        }
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
        let trashRoot = directory.appending(component: ".trash-\(stamp)")
        try FileManager.default.createDirectory(
            at: trashRoot, withIntermediateDirectories: true)
        var destination = trashRoot.appending(component: name)
        var counter = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            destination = trashRoot.appending(component: "\(name)-\(counter)")
            counter += 1
        }
        try FileManager.default.moveItem(
            at: directory.appending(component: name), to: destination)
        return destination
    }

    // MARK: - Study naming: canonical rename + hash-exempt display label

    /// The display-label sidecar, one plain-text line in the study directory.
    ///
    /// It lives OUTSIDE `experiment.json` on purpose: `name` participates in
    /// the manifest content hash on both engines, so writing a new name into
    /// a frozen manifest would change its epoch — every completed run's
    /// stamped experiment hash would mismatch, `evaluate`/`analyze` would
    /// refuse the study's own runs, and `verify()` would report the frozen
    /// manifest as drifted. A sidecar cannot perturb the hash, is not on the
    /// pin surface (`pinnedInputEntries` enumerates declared inputs plus
    /// `pinned/` — never the study directory itself), and so is neither
    /// git-gated at freeze nor packed into a run bundle. It is workspace
    /// presentation data, not evidence.
    public static let displayLabelFileName = "display-label.txt"

    static func displayLabelURL(name: String) -> URL {
        directory.appending(components: name, displayLabelFileName)
    }

    /// This study's researcher-facing label, or nil when none is set. Never
    /// throws: a missing or unreadable sidecar simply means "no label".
    public static func displayLabel(name: String) -> String? {
        guard let data = try? Data(contentsOf: displayLabelURL(name: name)) else {
            return nil
        }
        let text = normalizedDisplayLabel(String(decoding: data, as: UTF8.self))
        return text.isEmpty ? nil : text
    }

    /// What to show for this study: the label when set, the canonical name
    /// otherwise. The canonical name stays the identity everywhere else —
    /// directory, run stamps, CLI arguments.
    public static func displayName(_ manifest: ExperimentManifest) -> String {
        displayLabel(name: manifest.name) ?? manifest.name
    }

    /// Labels for a whole list, read once (the app renders study pickers
    /// every frame; disk reads belong in a refresh, not a view body).
    public static func displayLabels(_ manifests: [ExperimentManifest]) -> [String: String] {
        var out: [String: String] = [:]
        for manifest in manifests {
            if let label = displayLabel(name: manifest.name) { out[manifest.name] = label }
        }
        return out
    }

    /// A label is one line of free text — newlines and surrounding
    /// whitespace collapse, because it renders in single-line pickers.
    static func normalizedDisplayLabel(_ raw: String) -> String {
        raw.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Sets or clears the display label. Allowed at EVERY status: the label
    /// is not manifest content, so it changes nothing the firewall protects.
    /// Empty (or nil) clears it, removing the sidecar.
    public static func setDisplayLabel(_ label: String?, experimentName: String) throws {
        // Load, so labelling a name that is not a study refuses instead of
        // creating a stray directory.
        _ = try load(name: experimentName)
        let normalized = normalizedDisplayLabel(label ?? "")
        let url = displayLabelURL(name: experimentName)
        guard !normalized.isEmpty else {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            return
        }
        guard let data = (normalized + "\n").data(using: .utf8) else {
            throw ExperimentError(reason: "display label is not valid UTF-8")
        }
        try data.write(to: url)
    }

    /// The directory name a typed rename resolves to. Surrounding
    /// whitespace is trimmed BEFORE sanitizing, because the sanitizer maps
    /// spaces to hyphens and a pasted " my-study " would otherwise become
    /// "-my-study-"; interior spaces still hyphenate, which is the
    /// intended convention. Callers use this to tell "no change" from a real
    /// rename without duplicating the rule.
    public static func resolvedRenameTarget(_ raw: String) -> String {
        sanitizedExperimentName(raw.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// What a rename did, so the caller can tell the researcher the part
    /// that is not obvious: existing runs keep the OLD name.
    public struct RenameOutcome: Sendable, Equatable {
        public let oldName: String
        public let newName: String
        /// Run directories whose manifest snapshot stamps the old name.
        /// They are immutable and are never touched — after the rename they
        /// no longer list under this study.
        public let runsKeepingOldName: Int

        /// The sentence the UI shows when runs were left behind, or nil.
        public var runsNote: String? {
            guard runsKeepingOldName > 0 else { return nil }
            let plural = runsKeepingOldName == 1 ? "run" : "runs"
            return "\(runsKeepingOldName) existing \(plural) still stamp "
                + "'\(oldName)' — runs are immutable and were not touched, so "
                + "they no longer list under '\(newName)'"
        }
    }

    /// How many run directories stamp this experiment name in their manifest
    /// snapshot — the same association `StudyResultStore.list` reads.
    public static func runsStamped(experimentName: String) -> Int {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return 0 }
        return entries.filter { url in
            guard
                let data = try? Data(contentsOf: url.appending(component: "experiment.json")),
                let manifest = try? JSONDecoder().decode(ExperimentManifest.self, from: data)
            else { return false }
            return manifest.name == experimentName
        }.count
    }

    /// Renames a DRAFT: the whole directory moves (`pinned/` snapshots, the
    /// display-label sidecar, everything inside travels with it) and the
    /// manifest's `name` is rewritten.
    ///
    /// Draft-only is a firewall requirement, not a policy preference: `name`
    /// is hashed into the manifest content hash, so renaming a frozen study
    /// would re-epoch it and orphan the runs that stamped the old hash. Use
    /// `setDisplayLabel(_:experimentName:)` for those — a label is hash-exempt and
    /// works at every status.
    ///
    /// Name safety follows the one existing rule
    /// (`sanitizedExperimentName`): the candidate is sanitized down to
    /// lowercase letters/digits/hyphens — path separators, dots and
    /// traversal sequences cannot survive it — and an empty result refuses,
    /// exactly as `create` does.
    @discardableResult
    public static func rename(
        experimentName oldName: String, to newName: String
    ) throws -> RenameOutcome {
        let manifest = try load(name: oldName)
        guard manifest.status == .draft else {
            throw ExperimentError(
                reason: "experiment '\(oldName)' is \(manifest.status.rawValue) — "
                    + "its name is stamped into every run's provenance and cannot "
                    + "change; set a display label instead, or duplicate to iterate")
        }
        let sanitized = resolvedRenameTarget(newName)
        // `create` accepts anything non-empty after sanitizing, which lets
        // "   " through as "---". A rename is a deliberate act on an
        // existing study, so it holds out for a name with substance.
        guard sanitized.contains(where: { $0.isLetter || $0.isNumber }) else {
            throw ExperimentError(
                reason: "empty name — a study name needs at least one letter "
                    + "or digit (it becomes the experiments/<name>/ directory)")
        }
        guard sanitized != oldName else {
            return RenameOutcome(
                oldName: oldName, newName: oldName, runsKeepingOldName: 0)
        }
        let destination = directory.appending(component: sanitized)
        guard (try? load(name: sanitized)) == nil,
            !FileManager.default.fileExists(atPath: destination.path)
        else {
            throw ExperimentError(reason: "experiment '\(sanitized)' already exists")
        }
        let stranded = runsStamped(experimentName: oldName)
        try FileManager.default.moveItem(
            at: directory.appending(component: oldName), to: destination)
        var renamed = manifest
        renamed.name = sanitized
        try save(renamed)
        return RenameOutcome(
            oldName: oldName, newName: sanitized, runsKeepingOldName: stranded)
    }

    /// A study name not yet taken: `base`, else `base-2`, `base-3`, … (the
    /// same suffix shape `duplicate` produces, so a workspace reads
    /// consistently).
    public static func unusedExperimentName(base: String) -> String {
        let sanitized = sanitizedExperimentName(base)
        guard !sanitized.isEmpty else { return sanitized }
        var candidate = sanitized
        var counter = 1
        while (try? load(name: candidate)) != nil
            || FileManager.default.fileExists(
                atPath: directory.appending(component: candidate).path)
        {
            counter += 1
            candidate = "\(sanitized)-\(counter)"
        }
        return candidate
    }

    /// The placeholder a one-click New Study starts from — readable and
    /// unique, meant to be renamed immediately.
    public static func placeholderStudyName(date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd"
        return unusedExperimentName(base: "new-study-\(formatter.string(from: date))")
    }

    // MARK: - Verification & freeze

    /// Checks every pinned input against reality. Empty result = sound.
    /// Coherence-gate wording (2026-07-22 incident: a frozen study whose
    /// pipeline declared evaluate carried no `evaluation` block and died at
    /// the evaluate stage after generation). Cross-engine identical string —
    /// server twin: `manifest.EVALUATE_WITHOUT_JUDGING_MESSAGE`.
    public static let evaluateWithoutJudgingViolation =
        "the pipeline declares evaluate but the study declares no paired "
        + "judging — pin at least one judge and a rubric, or remove evaluate "
        + "from the pipeline"

    /// The paired-judge evaluation this manifest EFFECTIVELY declares
    /// (2026-07-22 incident: the app's rubric-FILE + judges path pinned
    /// `judges` and `judgeRubricFile` but never wrote an `evaluation`
    /// block, so the engines refused a frozen study at evaluate).
    ///
    /// Resolution rule (cross-engine, server twin
    /// `Manifest.effective_evaluation`): an explicit `evaluation` block
    /// always wins (`source == "manifest"`, unchanged semantics — including
    /// an explicit `kind: none`). With no block, at least one pinned judge
    /// PLUS a pinned `judgeRubricFile` ARE a paired-judge declaration: the
    /// spec is synthesized at the documented defaults (kind pairedJudge;
    /// judgeModel/judgePrompt empty — the panel carries the judges and the
    /// pinned file carries the rubric text, hash-verified at read time; no
    /// structured fields) and `source == "pinnedRubric"`. nil = no judging
    /// declared.
    public static func effectiveEvaluation(
        _ manifest: ExperimentManifest
    ) -> (spec: ExperimentManifest.EvaluationSpec, source: String)? {
        if let evaluation = manifest.evaluation {
            return (evaluation, "manifest")
        }
        let rubricFile = (manifest.judgeRubricFile ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !(manifest.judges ?? []).isEmpty, !rubricFile.isEmpty {
            return (
                ExperimentManifest.EvaluationSpec(
                    kind: .pairedJudge, judgeModel: "", judgePrompt: "",
                    structuredPrompt: nil),
                "pinnedRubric"
            )
        }
        return nil
    }

    /// The PURE half of the server's `_check_jlens_readout`, mirrored so the
    /// freeze firewall does not diverge across engines.
    ///
    /// `data check` grew J-lens coverage first, but that is a UI checklist —
    /// freeze runs through `verify()`, so a Swift/CLI freeze could still
    /// accept a readout with no `qualificationID`, one that records nothing,
    /// or one attached to a choice-only plan, and only the server would refuse
    /// it later (external review round 4). Same defect shape as the rest of
    /// this program: mechanism implemented, enforcement one boundary short.
    ///
    /// Deliberately structural only. Resolving the lens, checking its tier,
    /// and matching a qualification against the runtime all need the lens
    /// store and a loaded model, which are SERVER-ONLY by hard requirement —
    /// this engine renders lens artifacts and never produces or resolves them.
    /// Those checks stay server-side; what is mirrored is everything decidable
    /// from the manifest alone.
    public static func jlensReadoutViolations(
        _ manifest: ExperimentManifest
    ) -> [String] {
        guard let declared = manifest.jlensReadout else { return [] }
        guard case .object(let block) = declared else {
            // Present but not an object. Returning no violations let a
            // malformed block through this engine entirely, and the server
            // then assumed a dictionary (external review round 5).
            return ["jlensReadout is present but is not an object — a readout "
                        + "block is a JSON object of pinned declarations"]
        }
        var violations: [String] = []

        func text(_ key: String) -> String? {
            if case .string(let value)? = block[key], !value.isEmpty { return value }
            return nil
        }
        func nonEmptyArray(_ key: String) -> Bool {
            if case .array(let values)? = block[key] { return !values.isEmpty }
            return false
        }

        var missing: [String] = []
        for key in ["lensID", "lensSHA256", "configHash", "tokenizerHash",
                    "qualificationID"] where text(key) == nil {
            missing.append(key)
        }
        if !nonEmptyArray("layers") { missing.append("layers") }
        if !missing.isEmpty {
            violations.append(
                "jlensReadout is missing \(missing.joined(separator: ", ")) — "
                    + "a readout that is not fully pinned cannot be reproduced, "
                    + "and qualificationID names the exact acceptance the study "
                    + "rests on")
        }

        let records = nonEmptyArray("watchlist") || {
            if case .number(let k)? = block["topK"] { return k > 0 }
            return false
        }()
        if !records {
            violations.append(
                "jlensReadout declares neither a token watchlist nor a top-k "
                    + "width, so it would record nothing")
        }

        // The readout observes the GENERATION path. A deterministic
        // choice/logprob study runs none, so an armed readout there records
        // nothing and closes with an empty trace.
        if !ExecutionPlan.resolve(instruments: manifest.outcomeInstruments).generatesSampledText {
            violations.append(
                "jlensReadout is declared but this study's outcome instruments "
                    + "generate no sampled text — the readout observes the "
                    + "generation path, so it would record nothing")
        }
        return violations
    }

    public static func verify(_ manifest: ExperimentManifest) -> [String] {
        var violations: [String] = []
        // F2: pins are checked for the study kind that USES them. A panel
        // study may CARRY a readout block from before a kind switch and never
        // arms one, so validating it would refuse a legal manifest over an
        // instrument that cannot run (round-12 sweep).
        if manifest.studyKind == .modelOutput {
            violations += jlensReadoutViolations(manifest)
        }
        // Condition names are a CROSS-ENGINE record-identity contract — both
        // engines' `record_key` leads with them — so the uniqueness invariant
        // belongs on both (external review round 11). Checked against the
        // conditions a run EXECUTES: the baseline is implicit and in no
        // declared collection, and carried-but-inert configuration is
        // declared but never runs.
        var conditionNameCounts: [String: Int] = [:]
        for name in ExperimentTasks.effectiveConditionNames(for: manifest)
        where !name.isEmpty {
            conditionNameCounts[name, default: 0] += 1
        }
        let duplicateNames = conditionNameCounts.filter { $0.value > 1 }
            .keys.sorted()
        if !duplicateNames.isEmpty {
            violations.append(
                "duplicate condition name(s) \(duplicateNames) among the "
                    + "conditions this study EXECUTES — condition names are record "
                    + "identity (resume keys, shard ownership, merge completeness "
                    + "all lead with them), so two cannot be told apart in the "
                    + "outputs; rename one. Note the baseline condition is "
                    + "implicit and always executes")
        }
        if manifest.studyKind == .multiAgent {
            if let path = manifest.multiAgentScenarioPath,
                let pinned = manifest.multiAgentScenarioHash
            {
                let url = resolveProjectPath(path)
                if let hash = try? MultiAgentScenarioStore.hash(url), hash != pinned {
                    violations.append(
                        "multi-agent scenario changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
                } else if (try? MultiAgentScenarioStore.hash(url)) == nil {
                    violations.append("multi-agent scenario missing (\(path))")
                }
            } else {
                violations.append("multi-agent study needs a pinned scenario")
            }
        } else if manifest.concepts.isEmpty && manifest.variantConditions.isEmpty {
            violations.append("no concepts or variants attached")
        }
        violations += studyTypeContractViolations(manifest)
        // F2 (2026-07-19): pins are checked for the study kind that
        // USES them. A multi-agent study CARRIES model-output
        // configuration untouched (the type picker's never-delete
        // promise) but does not run it, so its pins neither block
        // verification nor enter packaging; the carried state surfaces
        // as a freeze advisory instead.
        if manifest.studyKind == .modelOutput {
            violations += modelOutputPinViolations(manifest)
        }
        // Reasoning-style taxonomy pin: the taxonomy IS a measurement
        // instrument, so drift or disappearance after pinning is a violation
        // exactly like markers drift; a half-pin certifies nothing. Absent
        // (both keys nil) = no reasoning-style scoring, no violation.
        if let path = manifest.reasoningStyleTaxonomyPath {
            if let pinned = manifest.reasoningStyleTaxonomyHash {
                if let data = try? Data(contentsOf: resolveProjectPath(path)) {
                    let live = sha256Hex(data)
                    if live != pinned {
                        violations.append(
                            "reasoning-style taxonomy changed since pinning "
                                + "(have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)")
                    }
                } else {
                    violations.append(
                        "pinned reasoning-style taxonomy missing (\(path))")
                }
            } else {
                violations.append(
                    "reasoning-style taxonomy pin is incomplete — "
                        + "reasoningStyleTaxonomyPath and reasoningStyleTaxonomyHash "
                        + "must both be set")
            }
        } else if manifest.reasoningStyleTaxonomyHash != nil {
            violations.append(
                "reasoningStyleTaxonomyHash pinned without a path — an "
                    + "unresolvable pin certifies nothing")
        }
        // SAE candidate roster: same contract as the taxonomy pin above. The
        // roster fixes WHICH features a study may seat before behaviour is
        // measured, so drift or disappearance after pinning is a violation,
        // and a half-pin certifies nothing. ABSENT = no roster, no violation.
        violations += saeCandidatesPinViolations(manifest.saeCandidates)
        // Pipeline block (chain runner, stage 5): the chain is preregistered
        // DATA, so a malformed declaration is a verify violation on THIS
        // engine too — not a refusal discovered at first cluster submission.
        let pipelineViolations = pipelineBlockViolations(manifest.pipeline)
        violations += pipelineViolations
        // Coherence gate (2026-07-22 incident): a chain declaring 'evaluate'
        // with no effective paired-judge declaration is a GUARANTEED runtime
        // failure after generation — a verify violation here, never a
        // surprise at the evaluate stage. Judges + a pinned rubric file
        // COUNT as the declaration (both engines synthesize the spec from
        // those pins), so only manifests that could never have produced a
        // judged report trip. A malformed pipeline block already has its own
        // violations above; the stage list is only read once it parses.
        if manifest.pipeline != nil, pipelineViolations.isEmpty,
            let draft = PipelineDraft.parse(manifest.pipeline),
            draft.stages.contains("evaluate"),
            effectiveEvaluation(manifest)?.spec.kind != .pairedJudge
        {
            violations.append(evaluateWithoutJudgingViolation)
        }
        // Human-baseline table drift (when pinned) — the alien-residual R is
        // only meaningful against the exact table the study froze.
        if let baseline = manifest.humanBaseline {
            let url = resolveProjectPath(baseline.path)
            if let data = try? Data(contentsOf: url) {
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                if hash != baseline.hash {
                    violations.append(
                        "human baseline '\(baseline.path)' changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(baseline.hash.prefix(12))…)")
                } else if let problem =
                    PinShapeValidation.humanBaselineShapeProblem(
                        data, file: baseline.path)
                {
                    // Shape check ONLY when the hash matches (drift is
                    // already reported above): a present-but-malformed
                    // baseline would otherwise pass verify and die at
                    // analyze. Same checker as the pin — verify is the
                    // backstop for files that entered without it.
                    violations.append(problem)
                }
            } else {
                violations.append("human baseline missing (\(baseline.path))")
            }
        }
        // Judge rubric pin: a judge-evaluated study's rubric is a pinned
        // input like any other — drift or a half-pin is a violation.
        if let file = manifest.judgeRubricFile, let pinned = manifest.judgeRubricHash {
            let url = resolveProjectPath(file)
            if let data = try? Data(contentsOf: url) {
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                if hash != pinned {
                    violations.append(
                        "judge rubric '\(file)' changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
                }
            } else {
                violations.append("judge rubric missing (\(file))")
            }
        } else if manifest.judgeRubricFile != nil || manifest.judgeRubricHash != nil {
            violations.append("judge rubric is incompletely pinned (need file AND hash)")
        }
        // Judge panel sanity: kinds are a closed set. A local judge without
        // a model id is LEGAL — it resolves to the study model
        // (manifest.modelID) at run/sweep start (cross-engine rule,
        // 2026-07-08). An OpenRouter judge has NO defaults: model slug and
        // serving provider are both pins (server rule, 2026-07-19), so a
        // missing one is a violation here, not a mid-sweep surprise.
        for judge in manifest.judges ?? [] {
            if judge.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                violations.append("judge entry has an empty name")
            }
            switch judge.kind {
            case "claude", "local":
                break
            case "openrouter":
                if (judge.model ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    violations.append(
                        "openrouter judge '\(judge.name)' has no model slug "
                            + "— there is no default to resolve")
                }
                if (judge.provider ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    violations.append(
                        "openrouter judge '\(judge.name)' has no pinned "
                            + "provider — an unpinned provider is not a "
                            + "pinned judge")
                }
            default:
                violations.append(
                    "judge '\(judge.name)' has unknown kind '\(judge.kind)' "
                        + "(expected claude|openrouter|local)")
            }
        }
        // Human-validation subset (per-judge vs-human agreement source).
        if let human = manifest.humanValidation {
            let url = resolveProjectPath(human.path)
            if let data = try? Data(contentsOf: url) {
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                if hash != human.hash {
                    violations.append(
                        "human validation set '\(human.path)' changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(human.hash.prefix(12))…)")
                } else if let data = try? Data(
                    contentsOf: resolveProjectPath(human.path))
                {
                    // Hash-clean is not row-clean: a pin authored by pasted
                    // JSON or a bundle can hash a file this engine's parser
                    // refuses — verified through the SAME parser evaluate
                    // uses (review 2026-08-02; server twin parses via
                    // `human_validation.parse_rows` in its verify).
                    do {
                        _ = try ExperimentTasks.parseHumanValidation(data)
                    } catch let error as ExperimentError {
                        violations.append(
                            "human validation set '\(human.path)': "
                                + error.reason)
                    } catch {
                        violations.append(
                            "human validation set '\(human.path)': \(error)")
                    }
                }
            } else {
                violations.append("human validation set missing (\(human.path))")
            }
        }
        // Stochastic sampling policy: samplesPerItem > 1 requires
        // temperature > 0 and a deterministic per-record seed derivation.
        if let samples = manifest.samplesPerItem, samples > 1 {
            if manifest.temperature <= 0 {
                violations.append(
                    "samplesPerItem > 1 requires temperature > 0 "
                        + "(greedy decoding makes every sample identical)")
            }
            if manifest.seedPolicy != "derivedSHA256" {
                violations.append(
                    "samplesPerItem > 1 requires seedPolicy 'derivedSHA256' "
                        + "(per-record seeds derived from condition/prompt/sampleIndex)")
            }
        }
        if let frozen = manifest.freezeHash, manifest.status != .draft {
            if manifest.frozenBy == "server" {
                violations += serverFreezeCanonicalViolations(manifest, freezeHash: frozen)
            } else if frozen != manifestHash(manifest),
                frozen != legacyManifestHash(manifest)
            {
                // The legacy branch accepts manifests frozen before
                // `createdAt` left the hash — same acceptance the old code
                // gave them, nothing wider.
                violations.append("manifest content changed after freeze (hash mismatch)")
            }
        }
        return violations
    }
    /// Whether the CONCEPT MACHINERY (concepts, injection conditions,
    /// markers, grand-mean corpus, neutral corpus, sweep inputs) is
    /// operative for this manifest — the finer rule under `studyKind`
    /// (engineer finding 2026-07-19: the three model-output study types
    /// share a kind, so kind-level gating alone left carried concepts
    /// ACTIVE inside a compare-agents study). Operative for concept
    /// studies and confirmations (whose conditions ARE injections), and
    /// for ANY study carrying forward-referenced agents (its own sweep
    /// must extract and select on those concepts). A compare-agents study
    /// with no forward references derives nothing — its carried concept
    /// configuration is inert. The durable `studyType` decides ties (a
    /// declared agentComparison keeps carried concepts inert even though
    /// content derivation would call it a concept study).
    static func conceptMachineryOperative(_ manifest: ExperimentManifest) -> Bool {
        guard manifest.studyKind == .modelOutput else { return false }
        switch StudyIntent.derive(from: manifest) {
        case .conceptStudy:
            // Both phases: the screen funnel and the confirm phase's
            // perturbation conditions are the same machinery.
            return true
        case .agentComparison:
            return manifest.variantConditions.contains { $0.fromPromotion != nil }
        case .multiAgent:
            return false
        }
    }

    /// A study that would SILENTLY measure baseline only (observed live
    /// 2026-08-11: the c20-* cluster fan-out burned a 4-shard GPU
    /// allocation producing 12 baseline records instead of 96): a
    /// modelOutput manifest whose DECLARED study type keeps the concept
    /// machinery inert — a declared agentComparison with no fromPromotion
    /// arms — while attaching NO variant conditions to run instead, yet
    /// still carrying non-baseline injection conditions. Both engines drop
    /// those conditions by the 2026-07-19 inert-machinery rule, so every
    /// declared arm vanishes from the evidence without a word. Returns the
    /// plain-language problem (shared by verify and run start — one rule,
    /// both surfaces; Python twin `inert_conditions_problem`), or nil.
    ///
    /// Deliberately narrow: a manifest WITH variant conditions keeps the
    /// 2026-07-19 behavior (agents run; carried injection conditions are
    /// inert, surfaced by the mixed-arm check), and a baseline-only
    /// manifest with no injection conditions is a legal, if unusual,
    /// study.
    static func inertConditionsProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        guard manifest.studyKind == .modelOutput,
              !conceptMachineryOperative(manifest),
              manifest.variantConditions.isEmpty
        else { return nil }
        let dropped = manifest.conditions
            .filter { !isCanonicalBaseline($0) }
            .map(\.name)
        guard !dropped.isEmpty else { return nil }
        return "declared studyType '\(manifest.studyType ?? "?")' keeps "
            + "the concept machinery inert and the manifest attaches no "
            + "agent (variant) conditions, so a run would execute BASELINE "
            + "ONLY — silently dropping injection condition(s) "
            + dropped.joined(separator: ", ")
            + ". Remove the studyType declaration (content derivation "
            + "would make this a concept study), declare 'conceptStudy', "
            + "or attach the agent conditions the declaration promises"
    }

    /// A concept study that would measure NOTHING: the concept machinery is
    /// operative, yet the manifest declares no injection condition, no agent
    /// (variant) condition, and no SAE latent arm — so a run executes the
    /// implicit BASELINE alone and `analyze` reports zero effect sizes, both
    /// exiting 0. A pipeline that "succeeds" end to end and measures nothing
    /// (WP0 dry run #0, P0-2).
    ///
    /// The sibling of `inertConditionsProblem` (2026-08-11), which catches
    /// the other way of reaching a silent baseline-only run: there the arms
    /// were DECLARED and dropped, here they were never declared. Same
    /// remedy shape — say what is missing, and name the sanctioned spelling
    /// of a deliberate baseline-only run.
    ///
    /// Deliberately narrow: a study with agents runs them; a declared
    /// `agentComparison` keeps the machinery inert and is the explicit
    /// baseline-only declaration, so it never trips this. Python twin:
    /// `manifest.no_measured_conditions_problem`.
    ///
    /// WP0 step 5½: the "declare a condition" half of the remedy now names
    /// the verb that performs it (`experiment declare-condition`), which did
    /// not exist when this refusal was written — the message described an
    /// operation only the Studies panel could do. The Python twin keeps its
    /// wording until the server's authoring verbs land (audit §7 step 8);
    /// the cross-engine contract both suites assert is the SUBSTANCE
    /// ("BASELINE", "promote", "agentComparison"), not the byte string.
    static func noMeasuredConditionsProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        guard manifest.studyKind == .modelOutput,
              conceptMachineryOperative(manifest),
              manifest.variantConditions.isEmpty,
              saeLatentConditionCount(manifest) == 0,
              !manifest.conditions.contains(where: { !isCanonicalBaseline($0) }),
              // No concepts either: nothing was ever DERIVED, so there is no
              // arm that "vanished" and nothing to promise. That is the plain
              // baseline-only manifest the 2026-08-11 rule already calls
              // legal, if unusual — this refusal is about a manifest that
              // attached concepts and then never gave them an arm.
              !manifest.concepts.isEmpty
        else { return nil }
        return "study '\(manifest.name)' runs the concept machinery "
            + "(\(manifest.concepts.count) concept(s) attached) but declares "
            + "NO injection condition, NO agent (variant) condition and NO "
            + "SAE latent arm, so a run would execute the implicit BASELINE "
            + "only and measure nothing — analyze would then report zero "
            + "effect sizes. Add an arm: sweep layer×alpha and "
            + "'steerlab-cli experiment promote \(manifest.name) <concept>' "
            + "mints one from the winning cell, or declare one directly — "
            + "'steerlab-cli experiment declare-condition \(manifest.name) "
            + "<condition> --slots <concept>:<layer>:<alpha>'. For a DELIBERATE "
            + "baseline-only run, declare studyType 'agentComparison' — the "
            + "manifest then says baseline-only instead of implying it"
    }

    /// How many SAE latent arms the manifest declares (server-only to RUN,
    /// carried verbatim here). Counting them keeps the baseline-only refusal
    /// from firing on a study whose arms are latent interventions.
    private static func saeLatentConditionCount(
        _ manifest: ExperimentManifest
    ) -> Int {
        guard case .array(let entries)? = manifest.saeLatentConditions else { return 0 }
        return entries.count
    }

    /// One-line record of INERT carried concept machinery, for the runs
    /// that legally proceed (agent arms exist, so `inertConditionsProblem`
    /// does not refuse): a declared agent comparison whose manifest still
    /// carries concepts or injection conditions runs agents+baseline only,
    /// by the 2026-07-19 rule. Correct — and it must be LOUD and stamped,
    /// not silent (the 2026-08-11 c20-* incident: a baseline-only result
    /// that looked completed and ordinary cost two GPU rounds). The run
    /// start prints this and stamps it into config.json's `notes` under
    /// `inertConceptMachinery` (Python twin `inert_machinery_note`).
    /// Returns nil when the machinery is operative, the study is not
    /// modelOutput, or nothing is carried.
    static func inertMachineryNote(
        _ manifest: ExperimentManifest
    ) -> String? {
        guard manifest.studyKind == .modelOutput,
              !conceptMachineryOperative(manifest)
        else { return nil }
        let conditions = manifest.conditions
            .filter { !isCanonicalBaseline($0) }
            .map(\.name)
        let concepts = manifest.concepts.map(\.name)
        guard !conditions.isEmpty || !concepts.isEmpty else { return nil }
        return "declared studyType '\(manifest.studyType ?? "?")' keeps "
            + "the concept machinery INERT — \(conditions.count) carried "
            + "injection condition(s) "
            + "(\(conditions.isEmpty ? "none" : conditions.joined(separator: ", "))) "
            + "and \(concepts.count) concept(s) will NOT execute; this run "
            + "measures baseline and agent (variant) conditions only"
    }

    /// The canonical baseline the Add Baseline affordance creates: named
    /// "baseline", no slots. The agent-comparison path executes an
    /// equivalent baseline, so this one never "vanishes".
    static func isCanonicalBaseline(
        _ condition: ExperimentManifest.Condition
    ) -> Bool {
        condition.name == "baseline" && condition.slots.isEmpty
    }

    /// STRUCTURALLY VALIDATED sweep provenance (fourth engineer round —
    /// a decodable `selection` block alone is not proof): the block must
    /// name a sweep run, and the condition must BE the winning cell —
    /// exactly one slot whose layer/alpha equal `winningCell`, named
    /// "<concept>-recommended" (the name the sweep stamps). A fabricated
    /// block now has to forge a fully consistent record; the firewall
    /// guards against accidents, not determined fraud (same stance as
    /// `freeze --force`: loud, stamped, checkable).
    static func isSweepStamped(
        _ condition: ExperimentManifest.Condition
    ) -> Bool {
        guard let selection = condition.selection,
            !selection.sweepRun.trimmingCharacters(in: .whitespaces).isEmpty,
            condition.slots.count == 1,
            let slot = condition.slots.first,
            slot.layer == selection.winningCell.layer,
            slot.alpha == selection.winningCell.alpha,
            condition.name == "\(slot.concept)-recommended"
        else { return false }
        return true
    }

    /// Whether a sweep-stamped "-recommended" condition's EXECUTABLE TWIN
    /// is among the study's agent arms (fifth engineer round, 2026-07-19:
    /// structural self-consistency alone proved nothing about what RUNS —
    /// an unrelated agent could execute while the recommended cell's
    /// effect silently never did; sixth round, same day: the twin now
    /// requires COMPLETE birth-certificate identity). The twin is a
    /// CONCRETE variant whose artifact carries a promotion birth
    /// certificate with all three identity legs:
    /// - concept + cell, as EXECUTED: the artifact carries EXACTLY ONE
    ///   injection, and that injection's concept/layer/alpha equal the
    ///   condition's slot / `selection.winningCell` (seventh round,
    ///   2026-07-20: the certificate alone proved what was CLAIMED, not
    ///   what RUNS — an artifact certifying L4/α2 while injecting L9/α9,
    ///   or carrying a second injection, passed). `promote` on both
    ///   engines mints exactly one injection stamped from the same
    ///   layer/alpha values as the certificate's winningCell, so exact
    ///   Double equality on alpha is correct here (both sides are the
    ///   same stamp, never recomputed). The `Promotion` certificate
    ///   itself carries NO concept field (a known shape gap) — concept
    ///   identity is read from the injection;
    /// - sweepRun: the certificate's sweepRun is NON-EMPTY and equal to
    ///   `selection.sweepRun` (a stamped selection always names its run,
    ///   so "when both carry one" was a hole: an empty certificate run
    ///   matched any recommendation);
    /// - cell: winningCell layer/alpha equal to the condition's
    ///   `selection.winningCell`.
    /// A `fromPromotion` forward reference NO LONGER exempts a STAMPED
    /// condition: it binds to no selection — its future sweep may pick a
    /// different cell. (Pipeline chains are unaffected: frozen chains
    /// never carry mid-chain stamps, and chains resolve forward refs from
    /// their OWN sweep at run time.)
    /// Server twin: `_sweep_stamped_executable_twin` in
    /// `steerlab_server/experiment/manifest.py` — keep identical.
    static func sweepStampedExecutableTwinPresent(
        _ condition: ExperimentManifest.Condition,
        among variants: [ExperimentManifest.VariantCondition]
    ) -> Bool {
        guard let selection = condition.selection,
            let slot = condition.slots.first
        else { return false }
        let conditionRun = selection.sweepRun
            .trimmingCharacters(in: .whitespaces)
        guard !conditionRun.isEmpty else { return false }
        return variants.contains { variant in
            guard variant.fromPromotion == nil,
                let promotion = variant.artifact.promotion,
                variant.artifact.injections.count == 1,
                let injection = variant.artifact.injections.first,
                injection.concept == slot.concept,
                injection.layer == selection.winningCell.layer,
                injection.alpha == selection.winningCell.alpha,
                let cell = promotion.winningCell,
                cell.layer == selection.winningCell.layer,
                cell.alpha == selection.winningCell.alpha
            else { return false }
            let promotedRun = (promotion.sweepRun ?? "")
                .trimmingCharacters(in: .whitespaces)
            return !promotedRun.isEmpty && promotedRun == conditionRun
        }
    }

    /// The studyType contract (2026-07-19): LLM-authored JSON gets loud
    /// feedback — an unknown vocabulary value, or a label contradicting
    /// the engine-facing studyKind, is a violation, never a silent
    /// re-derive (derive() ignores it as a fail-safe; verify SAYS so).
    /// Absent is legal: legacy manifests derive from content.
    static func studyTypeContractViolations(
        _ manifest: ExperimentManifest
    ) -> [String] {
        guard let declared = manifest.studyType else { return [] }
        // Alias-aware parse: legacy "confirmAgent" (a type until
        // 2026-07-19, now the concept study's confirm phase) stays legal
        // in shipped manifests and LLM packs.
        guard let intent = StudyIntent.parse(declared) else {
            return [
                "unknown studyType '\(declared)' — one of "
                    + StudyIntent.allCases.map(\.rawValue)
                        .joined(separator: ", ")
                    + " (legacy alias: confirmAgent)"
            ]
        }
        guard intent.mappedKind == manifest.studyKind else {
            return [
                "studyType '\(declared)' contradicts studyKind "
                    + "'\(manifest.studyKind.rawValue)' — a study runs its "
                    + "studyKind; fix whichever is wrong"
            ]
        }
        return []
    }

    /// Pins OPERATIVE only for model-output studies: concepts, injection
    /// conditions, agents, task prompts, battery, readers, the confirm
    /// pool rule, and the thinking-mode instrument gate. A multi-agent
    /// study carries this configuration untouched but never runs it, so
    /// its drift must not block verify/freeze (it surfaces as a freeze
    /// advisory) and packaging skips it (`pinnedInputEntries`).
    /// Drift/completeness violations for artifact-pinned concepts. The
    /// pinned bytes ARE the recipe here, so both files are re-hashed at
    /// every verify and the refusal names the file and both hashes — the
    /// same rule and the same loudness as stimulus drift. Contradictions
    /// that are pure DATA (a sidecar recorded against another model, or a
    /// sourceMethod that disagrees with the sidecar's own record) are
    /// violations too: they are checkable on either engine and can never
    /// become true later. Substrate is deliberately NOT checked here — a
    /// manifest must stay verifiable on the engine that authored it; the
    /// foreign-substrate refusal belongs at the load path, before any
    /// steering. Server twin: `Manifest._verify_vector_artifact_pins`.
    static func vectorArtifactPinViolations(
        _ manifest: ExperimentManifest, machinery: Bool
    ) -> [String] {
        var violations: [String] = []
        for ref in machinery ? manifest.concepts : [] {
            guard ref.options.method == .pinnedArtifact else {
                if ref.vectorArtifact != nil {
                    violations.append(
                        "concept '\(ref.name)' pins a vectorArtifact but "
                            + "declares method '\(ref.options.method.rawValue)' "
                            + "— a pinned artifact is only read under method "
                            + "'pinnedArtifact'")
                }
                continue
            }
            guard let pin = ref.vectorArtifact, !pin.path.isEmpty else {
                violations.append(
                    "concept '\(ref.name)': method 'pinnedArtifact' with no "
                        + "vectorArtifact.path — there is nothing to "
                        + "materialize")
                continue
            }
            guard !pin.sha256TensorHash.isEmpty, !pin.sha256SidecarHash.isEmpty
            else {
                violations.append(
                    "concept '\(ref.name)' vectorArtifact pin is incomplete — "
                        + "sha256TensorHash and sha256SidecarHash must both "
                        + "be set for '\(pin.path)' (a half-pin certifies "
                        + "nothing)")
                continue
            }
            // pin.path is workspace-relative by construction (attach
            // enforces it), so resolve against the active workspace root
            // directly — the test seam and cross-machine imports both hold.
            let base = workspaceRoot.appending(path: pin.path)
            var sidecarOK = true
            for (suffix, pinned, label) in [
                ("safetensors", pin.sha256TensorHash,
                 "vector artifact '\(pin.path).safetensors'"),
                ("json", pin.sha256SidecarHash,
                 "vector artifact sidecar '\(pin.path).json'"),
            ] {
                let url = base.appendingPathExtension(suffix)
                guard let data = try? Data(contentsOf: url) else {
                    violations.append(
                        "concept '\(ref.name)' \(label) missing")
                    if suffix == "json" { sidecarOK = false }
                    continue
                }
                let live = sha256Hex(data)
                if live != pinned {
                    violations.append(
                        "concept '\(ref.name)' \(label) changed since pinning "
                            + "(have \(live.prefix(12))…, pinned "
                            + "\(pinned.prefix(12))…)")
                    if suffix == "json" { sidecarOK = false }
                }
            }
            guard sidecarOK,
                let data = try? Data(contentsOf: base.appendingPathExtension("json")),
                let sidecar = try? JSONDecoder().decode(
                    SteeringVectorSidecar.self, from: data)
            else { continue }
            if pin.sourceMethod != (sidecar.extractionMethod ?? "") {
                violations.append(
                    "concept '\(ref.name)' vectorArtifact.sourceMethod "
                        + "'\(pin.sourceMethod)' contradicts the pinned "
                        + "sidecar's extractionMethod "
                        + "'\(sidecar.extractionMethod ?? "nil")' — re-attach "
                        + "the artifact")
            }
            if sidecar.modelID != manifest.modelID {
                violations.append(
                    "concept '\(ref.name)' vector artifact was extracted on "
                        + "'\(sidecar.modelID)', not this study's model "
                        + "'\(manifest.modelID)' — a direction does not "
                        + "transfer between models")
            }
            if let pinnedRevision = manifest.modelRevision,
                let recorded = sidecar.revision, recorded != pinnedRevision
            {
                violations.append(
                    "concept '\(ref.name)' vector artifact was extracted at "
                        + "revision \(recorded.prefix(12))…, not this study's "
                        + "pinned \(pinnedRevision.prefix(12))…")
            }
            if let recorded = sidecar.readingPosition,
                recorded != ref.options.readingPosition.label
            {
                violations.append(
                    "concept '\(ref.name)' reading position "
                        + "'\(ref.options.readingPosition.label)' contradicts "
                        + "the pinned artifact's '\(recorded)' — held-out "
                        + "activations must be read where the vector was read")
            }
        }
        return violations
    }

    static func modelOutputPinViolations(
        _ manifest: ExperimentManifest
    ) -> [String] {
        var violations: [String] = []
        // The concept machinery's pins apply only where the machinery is
        // operative (see `conceptMachineryOperative`): a compare-agents
        // study without forward references carries them inert.
        let machinery = conceptMachineryOperative(manifest)
        // Data-side questions ask the EFFECTIVE method and the DATA concept:
        // an artifact-pinned concept keeps the base concept's stimuli and
        // held-out data (crit-gm reads crit's), so its pins are checked
        // against those bytes. A concept with NO SOURCE CONCEPT has no
        // stimuli at all — its stimulusSetHash is the verbatim
        // "optvec:<composite>" dataset hash (pinned in the training run) or
        // the "gemmascope:<release>:<saeID>:<feature>" dictionary
        // coordinate, both re-proved through the artifact-pin hash checks
        // below — so a stimulus-directory lookup here would refuse it for
        // missing data it never had. Server twin: the concept loop of
        // `Manifest.verify`.
        for ref in machinery ? manifest.concepts : [] {
            guard let effective = ref.effectiveMethod else {
                violations.append(
                    "concept '\(ref.name)' pins an artifact whose "
                        + "sourceMethod '\(ref.vectorArtifact?.sourceMethod ?? "")' "
                        + "this engine does not know — it cannot resolve where "
                        + "the concept's held-out data lives")
                continue
            }
            if !effective.hasSourceConcept { continue }
            let dataName = ref.dataConcept
            guard effective.isPaired else {
                // Grand-mean concept: pinned against its stories.jsonl, never
                // a positive/negative StimulusSet directory.
                if let live = storiesHash(for: dataName) {
                    if live != ref.stimulusSetHash {
                        violations.append(
                            "concept '\(ref.name)' stories changed since pinning "
                                + "(have \(live.prefix(12))…, pinned \(ref.stimulusSetHash.prefix(12))…)")
                    }
                } else {
                    violations.append(
                        "concept '\(ref.name)': no stories.jsonl under prompts/emotions/")
                }
                if effective == .designatedReference {
                    if let pin = ref.designatedReference {
                        if let liveRef = storiesHash(for: pin.name) {
                            if liveRef != pin.hash {
                                violations.append(
                                    "concept '\(ref.name)' reference '\(pin.name)' stories "
                                        + "changed since pinning (have \(liveRef.prefix(12))…, "
                                        + "pinned \(pin.hash.prefix(12))…)")
                            }
                        } else {
                            violations.append(
                                "concept '\(ref.name)' reference '\(pin.name)': "
                                    + "stories.jsonl missing under prompts/emotions/")
                        }
                    } else {
                        violations.append(
                            "designated-reference concept '\(ref.name)' has no pinned "
                                + "reference (designatedReference.name/hash) — re-attach "
                                + "with a reference")
                    }
                }
                continue
            }
            let conceptDir = VectorCatalog.conceptsDirectory.appending(component: dataName)
            guard let set = try? StimulusSet(directory: conceptDir) else {
                violations.append("concept '\(ref.name)': stimulus files missing/unreadable")
                continue
            }
            if set.hash != ref.stimulusSetHash {
                violations.append(
                    "concept '\(ref.name)': stimulus files changed since pinning "
                        + "(have \(set.hash.prefix(12))…, pinned \(ref.stimulusSetHash.prefix(12))…)")
            }
        }
        // Artifact-pinned concepts: the VECTOR BYTES are the pinned input,
        // so both artifact files are re-hashed here exactly like a stimulus
        // file (and the block's own completeness is checked — a half-pin
        // certifies nothing).
        violations += vectorArtifactPinViolations(manifest, machinery: machinery)
        // Measurement-side pins: each concept's never-named validation.jsonl
        // (the convergent gate's input) is checked like any other pinned
        // file. Three-state contract: key absent = legacy (verify passes,
        // freeze advises); key null = pinned-as-absent (a file appearing
        // later is drift); key set = hash-checked.
        for ref in machinery ? manifest.concepts : [] {
            // A source-concept-less direction has NOTHING to validate: no
            // stimuli, no held-out validation.jsonl. An optvec vector's
            // evidence is the OptVec eval run's eval.json (plan §6); an
            // imported Gemma Scope SAE decoder row's is the discovery
            // snapshot + qualification artifact in the pinned candidate
            // roster. attach pins the hash explicitly null so neither is
            // mistaken for a legacy unpinned attach; checking a validation
            // FILE that can never exist would be inventing an obligation.
            if ref.effectiveMethod?.hasSourceConcept == false { continue }
            // Dual-root lookup (2026-08-19): the canonical home wins, the
            // other recipe's home is the fallback. A file in the CANONICAL
            // home resolves exactly as before, so verify()'s verdict on an
            // already-frozen manifest is unchanged; only a misfiled set —
            // previously invisible to the pin — now participates.
            let live = conceptValidationHash(
                name: ref.dataConcept,
                isPaired: !(ref.effectiveMethod ?? ref.options.method)
                    .usesStoryCorpus)
            if let pinned = ref.validationHash {
                if let live {
                    if live != pinned {
                        violations.append(
                            "concept '\(ref.name)': validation.jsonl changed since pinning "
                                + "(have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)")
                    }
                } else {
                    violations.append(
                        "concept '\(ref.name)': validation.jsonl missing "
                            + "(pinned \(pinned.prefix(12))…)")
                }
            } else if ref.validationHashPinnedAbsent, live != nil {
                violations.append(
                    "concept '\(ref.name)': validation.jsonl appeared after attach "
                        + "(pinned as absent) — re-attach to pin it")
            }
        }
        // Study-level markers pin (scoring rubrics): drift or disappearance
        // after pinning is a violation like stimulus drift.
        if machinery, let pinned = manifest.markersHash {
            if let live = liveMarkersHash(manifest) {
                if live != pinned {
                    violations.append(
                        "concept markers.json changed since pinning "
                            + "(have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)")
                }
            } else {
                violations.append(
                    "markersHash is pinned but no attached concept has a markers.json")
            }
        }
        // Sweep-input pins (firewall closure, 2026-07-20 — cross-engine
        // contract keys "sweep.devPromptsHash" + "sweep.batteryHash"): the
        // sweep's dev prompts and capability battery decide which cell
        // WINS, so they are measurement-side inputs exactly like markers —
        // a pinned hash is enforced against the file bytes (freeze pins
        // them; sweep start additionally refuses to select on drifted
        // inputs). Legacy manifests without the keys verify clean. Server
        // twin: `Manifest._sweep_input_pin_violations`.
        if machinery, let sweep = manifest.sweep {
            // Choice instruments join the same two-state contract (review
            // 2026-08-02, P1): key absent = legacy/unpinned (clean — freeze
            // pins it); key present = enforced against the bytes.
            let sweepPins: [(String, String?, String)] = [
                (sweep.devPromptsFile, sweep.devPromptsHash,
                 "sweep dev prompts"),
                (sweep.batteryFile, sweep.batteryHash,
                 "sweep capability battery"),
            ] + sweepChoicePinEntries(sweep).map { entry in
                (entry.file, entry.pinned, entry.label)
            }
            for (file, pinned, label) in sweepPins {
                guard let pinned else { continue }
                if let data = try? Data(contentsOf: resolveProjectPath(file)) {
                    let live = sha256Hex(data)
                    if live != pinned {
                        violations.append(
                            "\(label) '\(file)' changed since pinning "
                                + "(have \(live.prefix(12))…, pinned "
                                + "\(pinned.prefix(12))…)")
                    }
                } else {
                    violations.append(
                        "\(label) '\(file)': file missing at \(file)")
                }
            }
        }
        // A grand-mean vector also depends on every OTHER corpus member: the
        // whole pinned population must exist unchanged, and every grand-mean
        // target must be a member of it.
        let grandMeanNames = manifest.concepts
            .filter { $0.options.method.isGrandMean }
            .map(\.name)
        if machinery, !grandMeanNames.isEmpty {
            if let corpus = manifest.grandMeanCorpus {
                for name in grandMeanNames where !corpus.concepts.contains(name) {
                    violations.append(
                        "grand-mean concept '\(name)' is not a member of the pinned "
                            + "grandMeanCorpus")
                }
                for member in corpus.concepts {
                    guard let pinned = corpus.hashes[member], !pinned.isEmpty else {
                        violations.append(
                            "grandMeanCorpus member '\(member)' has no pinned hash")
                        continue
                    }
                    if let live = storiesHash(for: member) {
                        if live != pinned {
                            violations.append(
                                "grandMeanCorpus member '\(member)' stories changed since "
                                    + "pinning (have \(live.prefix(12))…, pinned \(pinned.prefix(12))…)")
                        }
                    } else {
                        violations.append(
                            "grandMeanCorpus member '\(member)': stories.jsonl missing")
                    }
                }
            } else {
                violations.append(
                    "grand-mean concepts attached but no grandMeanCorpus pinned "
                        + "(the population the grand mean is computed over)")
            }
        }
        let legacyProjectionConcepts = manifest.concepts
            .filter { ($0.options.neutralPCCount ?? 0) > 0 }
            .map(\.name)
        if machinery, !legacyProjectionConcepts.isEmpty {
            violations.append(
                "legacy fixed-k neutral PC projection is disabled for verified/frozen "
                    + "experiments until a token-position neutral activation bank exists "
                    + "(remove --project-neutral from: "
                    + legacyProjectionConcepts.joined(separator: ", ") + ")")
        }
        // Mixed arm modes refuse (engineer finding 2026-07-19): when agent
        // conditions exist, BOTH run engines execute agents only, so a
        // HAND-DECLARED injection condition would silently vanish from the
        // evidence. Two exemptions, both narrow:
        // - the canonical empty "baseline" (Add Baseline creates it, and
        //   the agent path executes an equivalent baseline — nothing
        //   vanishes); a CUSTOM-named empty condition still refuses
        //   because its identity would disappear from the evidence.
        // - a sweep-stamped "-recommended" condition whose provenance is
        //   STRUCTURALLY CONSISTENT (fourth round: presence of a selection
        //   block is not proof — the block must name a sweep run and the
        //   condition must BE the winning cell: one slot, layer/alpha
        //   matching winningCell, name "<concept>-recommended") AND whose
        //   EXECUTABLE TWIN is among the agent arms (fifth round:
        //   self-consistency alone did not prove the promoted agent
        //   actually runs; sixth round: the twin needs complete
        //   birth-certificate identity — concept + sweepRun + cell — and
        //   an unbound forward reference no longer counts; seventh
        //   round, 2026-07-20: the artifact must EXECUTE the identity it
        //   certifies — exactly one injection, at exactly that
        //   concept/layer/alpha — see
        //   `sweepStampedExecutableTwinPresent`). The
        //   promoted agent is that cell's executable form. Run-directory
        //   existence is deliberately NOT required here: runs are
        //   per-substrate trees, and a bundle-imported study legitimately
        //   references the other engine's sweep evidence.
        if machinery, !manifest.variantConditions.isEmpty {
            var handDeclared: [String] = []
            for condition in manifest.conditions {
                if isCanonicalBaseline(condition) { continue }
                guard isSweepStamped(condition) else {
                    handDeclared.append(condition.name)
                    continue
                }
                if !sweepStampedExecutableTwinPresent(
                    condition, among: manifest.variantConditions)
                {
                    let concept = condition.slots.first?.concept ?? "?"
                    let sweepRun = condition.selection?.sweepRun
                        .trimmingCharacters(in: .whitespaces) ?? "?"
                    violations.append(
                        "recommended condition '\(condition.name)': the "
                            + "promoted agent for concept '\(concept)' is "
                            + "not among this study's arms — the "
                            + "recommendation is bound to sweep run "
                            + "'\(sweepRun)', and a study with agent "
                            + "conditions runs agents only, so the "
                            + "recommended cell would never execute. "
                            + "Promote that sweep's agent and add it (its "
                            + "birth certificate must match this concept, "
                            + "sweep run, and winning cell, and its "
                            + "artifact must inject exactly that cell — "
                            + "one injection, same concept/layer/alpha; "
                            + "an unbound fromPromotion forward reference "
                            + "does not count), or remove the stale "
                            + "recommendation")
                }
            }
            if !handDeclared.isEmpty {
                violations.append(
                    "mixed arm modes: hand-declared injection condition(s) "
                        + handDeclared.joined(separator: ", ")
                        + " would NOT execute — a study with agent conditions "
                        + "runs agents only. Remove one side or split into "
                        + "two studies")
            }
        }
        // The DEGENERATE inert case (observed live 2026-08-11: the c20-*
        // cluster fan-out produced baseline-only evidence): a declared
        // agentComparison with no agent arms at all but carrying injection
        // conditions runs baseline only, silently — the mixed-arm check
        // above cannot see it because the machinery is inert and there are
        // no variants. The run path refuses through the same rule.
        if let inertProblem = inertConditionsProblem(manifest) {
            violations.append(inertProblem)
        }
        let known = Set(manifest.concepts.map(\.name))
        for condition in machinery ? manifest.conditions : [] {
            for slot in condition.slots where !known.contains(slot.concept) {
                violations.append(
                    "condition '\(condition.name)': references unattached concept "
                        + "'\(slot.concept)'")
            }
            // Neutral-PC basis pin (cross-engine rule): the pinned hash is
            // the SHA-256 of the basis FILE BYTES at neutralPCBasisPath —
            // it was historically written (as the corpus hash) but never
            // checked, so a swapped or regenerated basis silently changed
            // frozen studies. Path-without-hash = legacy, passes;
            // hash-without-path is an unresolvable pin, a violation.
            if let path = condition.neutralPCBasisPath,
                let pinned = condition.neutralPCBasisHash
            {
                if let data = neutralPCBasisBytes(path: path) {
                    let live = sha256Hex(data)
                    if live != pinned {
                        violations.append(
                            "condition '\(condition.name)': neutral-PC basis changed "
                                + "since pinning (have \(live.prefix(12))…, "
                                + "pinned \(pinned.prefix(12))…)")
                    }
                } else {
                    violations.append(
                        "condition '\(condition.name)': neutral-PC basis missing (\(path))")
                }
            } else if condition.neutralPCBasisPath == nil,
                condition.neutralPCBasisHash != nil
            {
                violations.append(
                    "condition '\(condition.name)': neutral-PC basis hash pinned "
                        + "without a path")
            }
        }
        let attachedConcepts = Set(manifest.concepts.map(\.name))
        for variant in manifest.variantConditions {
            if let forward = variant.fromPromotion {
                // Forward-referenced (stage 4, server-resolved): the
                // DECLARATION is what verify can check — a named, attached
                // concept, and exactly one identity.
                let concept = forward.concept
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if concept.isEmpty {
                    violations.append(
                        "variant '\(variant.name)' fromPromotion names no concept")
                } else if !attachedConcepts.contains(concept) {
                    violations.append(
                        "variant '\(variant.name)' forward-references concept "
                            + "'\(concept)', which is not attached to this "
                            + "experiment")
                }
                // An EMBEDDED artifact counts as a concrete identity too
                // (cross-engine parity with the server's check — an
                // artifact-only conflict must not pass here).
                let embedded = variant.artifact
                let hasEmbeddedArtifact = !embedded.name.isEmpty
                    || !embedded.baseModelID.isEmpty
                    || !embedded.injections.isEmpty
                    || !embedded.adapters.isEmpty
                if !variant.artifactPath.isEmpty || !variant.artifactHash.isEmpty
                    || hasEmbeddedArtifact
                {
                    violations.append(
                        "variant '\(variant.name)' declares BOTH fromPromotion "
                            + "and a concrete artifact — a condition has exactly "
                            + "one identity")
                }
                continue
            }
            if variant.artifact.baseModelID != manifest.modelID {
                violations.append(
                    "variant '\(variant.name)' uses \(variant.artifact.baseModelID), "
                        + "not study base model \(manifest.modelID)")
            }
            let url = resolveProjectPath(variant.artifactPath)
            if let hash = try? ModelVariantStore.hash(url), hash != variant.artifactHash {
                violations.append(
                    "variant '\(variant.name)' changed since pinning "
                        + "(have \(hash.prefix(12))…, pinned \(variant.artifactHash.prefix(12))…)")
            } else if (try? ModelVariantStore.hash(url)) == nil {
                violations.append("variant '\(variant.name)' artifact missing (\(variant.artifactPath))")
            }
            // Training-dataset pins (LoRA readiness §0 amendment 1): a
            // trained adapter's dataset manifest is a pinned input like
            // stimuli — drift after freeze is a violation.
            violations += trainingProvenanceViolations(variant)
        }
        // The neutral corpus is pinned whenever it participates — as the
        // norm-unit denominator (pinned at attach when the corpus exists)
        // and/or the confound-projection basis.
        if machinery, let pinned = manifest.neutralCorpusHash {
            let url = VectorCatalog.projectRoot.appending(
                components: "prompts", "neutral", "corpus.jsonl")
            if let (_, hash) = try? StimulusSet.loadTexts(url: url), hash != pinned {
                violations.append(
                    "neutral corpus changed since pinning "
                        + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
            } else if (try? StimulusSet.loadTexts(url: url)) == nil {
                violations.append("neutral corpus missing (prompts/neutral/corpus.jsonl)")
            }
        } else if machinery,
            manifest.concepts.contains(where: { ($0.options.neutralPCCount ?? 0) > 0 })
        {
            violations.append(
                "confound projection requested but neutral corpus not pinned "
                    + "(re-attach with --project-neutral)")
        }
        if let file = manifest.taskPromptsFile, let pinned = manifest.taskPromptsHash {
            // Same resolution + parser as the run loop (`loadTaskPrompts`):
            // task items are `{"prompt": …}` (server style) or `{"text": …}`
            // rows with optional science-layer keys. Verifying through the
            // text-only stimulus loader rejected valid choice files the run
            // loop accepts — a pinned study must verify with the parser that
            // will actually read it.
            let url = resolveProjectPath(file)
            if let data = try? Data(contentsOf: url) {
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                if hash != pinned {
                    violations.append(
                        "task prompts changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
                } else {
                    // Scripted-transcript pin checks ride on the parse: a
                    // transcript schema violation IS the parse error (its
                    // message is the cross-engine contract), and parsed
                    // transcripts are checked against the study model
                    // family's chat-template constraints + the rawCompletion
                    // incompatibility at verify/freeze time, not first at
                    // run start (server `Manifest._transcript_violations`
                    // twin).
                    do {
                        let prompts = try ExperimentTasks.parseTaskPrompts(data)
                        violations.append(
                            contentsOf: ExperimentTasks.transcriptPinViolations(
                                prompts, manifest: manifest))
                    } catch {
                        let reason = (error as? ExperimentError)?.reason ?? "\(error)"
                        violations.append(
                            "task prompts unparseable (\(file)): \(reason)")
                    }
                }
            } else {
                violations.append("task prompts missing (\(file))")
            }
        } else if manifest.taskPromptsFile != nil || manifest.taskPromptsHash != nil {
            violations.append("task prompts are incompletely pinned")
        }
        // Capability battery pin (evidence tier): validate runs the battery
        // through every variant condition, so its file is a pinned input.
        if let file = manifest.capabilityBatteryFile,
            let pinned = manifest.capabilityBatteryHash
        {
            let url = resolveProjectPath(file)
            if let data = try? Data(contentsOf: url) {
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                if hash != pinned {
                    violations.append(
                        "capability battery '\(file)' changed since pinning "
                            + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
                } else if let problem =
                    PinShapeValidation.capabilityBatteryShapeProblem(
                        data, file: file)
                {
                    // Shape check ONLY when the hash matches (drift is
                    // already its own violation): a present-but-malformed
                    // battery would otherwise pass verify and die at task
                    // load. Same checker as the pin (PinShapeValidation)
                    // — the file legally enters a manifest without the
                    // pin path (hand-edit, import, legacy), so verify is
                    // the cross-engine backstop.
                    violations.append(problem)
                }
            } else {
                violations.append("capability battery missing (\(file))")
            }
        } else if manifest.capabilityBatteryFile != nil
            || manifest.capabilityBatteryHash != nil
        {
            violations.append(
                "capability battery is incompletely pinned (need file AND hash)")
        }
        // RepE reader instruments (server `Manifest.verify` twin): each
        // pinned reader must match its hash, be a reader artifact at all, be
        // fitted on THIS substrate (activations do not transfer between
        // engines), and be fitted on the study's model.
        if (manifest.outcomeInstruments ?? []).contains("repeReaderScore"),
            (manifest.readerRefs ?? []).isEmpty
        {
            violations.append(
                "outcomeInstruments includes repeReaderScore but no readerRefs "
                    + "are pinned")
        }
        for ref in manifest.readerRefs ?? [] {
            let url = resolveProjectPath(ref.path)
            guard let data = try? Data(contentsOf: url) else {
                violations.append("reader '\(ref.concept)' missing (\(ref.path))")
                continue
            }
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            if hash != ref.hash {
                violations.append(
                    "reader '\(ref.concept)' changed since pinning "
                        + "(have \(hash.prefix(12))…, pinned \(ref.hash.prefix(12))…)")
            }
            // THROUGH THE REAL LOADER (review 2026-08-02, P1): a loose JSON
            // inspection let an artifact missing `template`/`probe` pass
            // verification and fail much later at evaluate. What the
            // scorer cannot load, verify refuses NOW. Server twin:
            // Manifest.verify's reader block via `repe_reader.load_reader`.
            let artifact: RepEReader.Artifact
            do {
                artifact = try JSONDecoder().decode(
                    RepEReader.Artifact.self, from: data)
            } catch {
                violations.append(
                    "reader '\(ref.concept)': unloadable artifact — "
                        + "verified through the same loader scoring uses "
                        + "(\(error))")
                continue
            }
            // The COMPLETE binding through the ONE helper the runtime
            // scorer also calls (review 2026-08-02: two hand-kept subsets
            // meant the runtime accepted readers verify would flag).
            violations += readerBindingProblems(
                artifact, refConcept: ref.concept, manifest: manifest)
        }
        // Funnel: a confirm-phase study must use an item pool DISJOINT from
        // the screen pool it references (study guide: held-out items for
        // Phase 2).
        if manifest.phase == "confirm" {
            if manifest.screenTaskPromptsHash == nil {
                violations.append(
                    "confirm-phase study must pin screenTaskPromptsHash "
                        + "(the screen pool it is held out from)")
            } else if let pinned = manifest.taskPromptsHash,
                pinned == manifest.screenTaskPromptsHash
            {
                violations.append(
                    "confirm-phase item pool is IDENTICAL to the screen pool "
                        + "(held-out items required)")
            }
        }
        // Thinking-mode gate: a reasoning model's answer distribution is a
        // marginal over sampled reasoning paths, so the answer-token logprob
        // instrument (conditional on NO reasoning prefix) is only valid with
        // thinking off. Thinking-mode arms must use sampled answers instead.
        if manifest.qwenThinkingEnabled == true,
            let instruments = manifest.outcomeInstruments,
            !Set(instruments).isDisjoint(
                with: ["answerTokenLogprob", "choiceProbability", "ordinalScale"])
        {
            violations.append(
                "outcomeInstruments includes an answer-token instrument but "
                    + "qwenThinkingEnabled is true — thinking-mode answers are marginals "
                    + "over reasoning paths; disable thinking or use sampled answers")
        }
        // Ordinal-scale contract: the collapse of the ladder distribution to
        // one position is an instrument-design choice — DECLARED in the
        // manifest, never silently defaulted (workbench rule). Server twin:
        // Manifest.verify in steerlab_server/experiment/manifest.py.
        if (manifest.outcomeInstruments ?? []).contains("ordinalScale"),
            manifest.ordinalAggregation == nil
        {
            violations.append(
                "outcomeInstruments includes ordinalScale but ordinalAggregation "
                    + "is not declared — declare 'expectedValue' "
                    + "(probability-weighted mean ladder position) or 'argmax' "
                    + "(position of the highest-probability option)")
        }
        if let aggregation = manifest.ordinalAggregation,
            !knownOrdinalAggregations.contains(aggregation)
        {
            violations.append(
                "unknown ordinalAggregation '\(aggregation)' — one of "
                    + "expectedValue, argmax")
        }
        // An inoperative per-control layer: reported so it is removed
        // deliberately. See ValidationControl.validationLayer for why it can
        // never be honoured.
        for control in manifest.validationControls ?? []
        where control.validationLayer != nil {
            violations.append(
                "validation control '\(control.concept)' declares "
                    + "validationLayer \(control.validationLayer ?? 0), which "
                    + "nothing reads: the cosine matrix compares every cell at "
                    + "ONE layer, so a per-control layer would conflate "
                    + "concept differences with depth differences. Remove the "
                    + "field; declare the study-wide validationLayer instead")
        }
        // The validation read layer is a measurement declaration (D4):
        // declaring an index AND a fraction is ambiguous, and both disagree
        // on any model whose depth differs from the one assumed.
        if let problem = ValidationLayerRule.violation(
            declaredLayer: manifest.validationLayer,
            declaredFraction: manifest.validationLayerFraction,
            declaredLayers: manifest.validationLayers,
            declaredFractions: manifest.validationLayerFractions)
        {
            violations.append(problem)
        }
        // Declared numeric parser: the registry entry IS a measurement
        // instrument, so a missing registry, an undefined/malformed parser,
        // or registry drift after pinning is a violation like markers drift.
        // A study that names no parser (and pins no hash) gains NOTHING here
        // (server `parser_pin_violations` twin).
        violations.append(contentsOf: ParserRegistry.pinViolations(manifest))
        // Declared exclusion rules are measurement declarations: a typo'd
        // rule id or a malformed range surfaces at verify — before any
        // behavior is measured — never first at analyze. Absent key = no
        // rules = nothing to check (server twin: Manifest.verify →
        // exclusions.rule_violations).
        violations.append(contentsOf: ExclusionEngine.violations(manifest.exclusionRules))
        return violations
    }


    private static let volatileFreezeKeys = [
        "status", "frozenAt", "freezeHash", "gitCommit", "frozenBy", "createdAt",
        "appVersion", "freezeForced", "forcedGatesSkipped",
    ]

    /// Keys this engine's encoder ALWAYS writes and the server OMITS when they
    /// hold their default — with the default itself, which is IDENTICAL on
    /// both engines. `multiAgentIncludeBaseline` defaults to true here
    /// (`ExperimentManifest.init`) and on the server (`Manifest.from_dict`);
    /// `recordTokenIDs` defaults to false on both. So absent-at-default and
    /// present-at-default describe the SAME study, and only a key-set
    /// comparison could disagree.
    ///
    /// Portability gap G1 (`docs/PORTABILITY-CONTRACTS.md`): a study authored
    /// ENTIRELY on the server — which is exactly what the cross-platform
    /// client does — carried neither key, so the post-freeze document
    /// comparison below read it as drifted and refused with the generic
    /// "changed after freeze", naming no field. Mac-authored studies frozen on
    /// the server were invisible to it, because their documents already
    /// carried both keys.
    ///
    /// Server twin: the omission is `experiment_store.create`'s key set (which
    /// stamps neither) plus `Manifest.from_dict`'s two defaults. A key added
    /// here must hold its default on BOTH engines — that is the whole licence
    /// for eliding it — and the pin that proves the server really elides it is
    /// `PortabilityContractTests.aServerAuthoredFrozenManifestVerifiesHere`
    /// over the fixture `test_a_python_authored_manifest_fixture_is_current`
    /// writes.
    static let defaultElidedFreezeKeys: [String: Bool] = [
        "multiAgentIncludeBaseline": true,
        "recordTokenIDs": false,
    ]

    /// Drop each `defaultElidedFreezeKeys` entry that is PRESENT and holds its
    /// default. Applied to both sides, so absent ≡ present-at-default while a
    /// present NON-default value still differs from an absence (the server
    /// omits only at the default, so an omission means the default and a
    /// non-default value really is a change).
    private static func droppingDefaultElidedKeys(
        _ object: [String: Any]
    ) -> [String: Any] {
        var result = object
        for (key, defaultValue) in defaultElidedFreezeKeys
        where (result[key] as? Bool) == defaultValue {
            result.removeValue(forKey: key)
        }
        return result
    }

    /// Post-freeze verification for SERVER-frozen manifests. Swift cannot
    /// reproduce Python's canonical JSON byte-for-byte, so the server writes
    /// `freeze-canonical.json` — the exact bytes it hashed. The contract:
    /// sha256(file bytes) == freezeHash, AND the file's parsed content equals
    /// the manifest minus the volatile freeze stamps (compared as parsed JSON,
    /// with explicit nulls stripped — Python writes `null` where Swift omits
    /// the key — and with `defaultElidedFreezeKeys` dropped at their default,
    /// which is the same class of difference one level up: a key one engine
    /// writes and the other elides, both meaning the same thing).
    ///
    /// The two normalizations are DELIBERATELY NOT part of
    /// `comparableFreezeObject`: that function also backs
    /// `canonicalManifestBodyHash`, a value shown to researchers, and moving
    /// it would change a displayed fingerprint for every existing manifest to
    /// close a gap that only this comparison has.
    private static func serverFreezeCanonicalViolations(
        _ manifest: ExperimentManifest, freezeHash: String
    ) -> [String] {
        let canonicalURL = directory.appending(
            components: manifest.name, "freeze-canonical.json")
        guard let canonicalData = try? Data(contentsOf: canonicalURL) else {
            return [
                "server-frozen manifest has no freeze-canonical.json — legacy server "
                    + "freezes must be re-frozen on the server to restore post-freeze "
                    + "verification"
            ]
        }
        let liveHash = SHA256.hash(data: canonicalData)
            .map { String(format: "%02x", $0) }.joined()
        guard liveHash == freezeHash else {
            return ["manifest content changed after freeze (hash mismatch)"]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard
            let canonicalObject = comparableFreezeObject(canonicalData)
                .map(droppingDefaultElidedKeys),
            let manifestData = try? encoder.encode(manifest),
            let manifestObject = comparableFreezeObject(manifestData)
                .map(droppingDefaultElidedKeys)
        else {
            return ["manifest content changed after freeze (hash mismatch)"]
        }
        guard !(canonicalObject as NSDictionary).isEqual(to: manifestObject) else {
            return []
        }
        // NAME the differing fields. The generic line sent a reader looking
        // for tampering when the difference was two keys nobody had edited
        // (G1); the same reuse the remote-freeze identity check already makes
        // of `manifestFieldMismatches` costs nothing and ends that.
        let mismatches = manifestFieldMismatches(
            local: manifestObject, server: canonicalObject)
        return [
            "manifest content changed after freeze — the manifest differs from "
                + "freeze-canonical.json in: "
                + (mismatches.isEmpty
                    ? "no top-level key (a nested value differs)"
                    : mismatches.joined(separator: "; "))
        ]
    }

    private static func comparableFreezeObject(_ data: Data) -> [String: Any]? {
        guard var object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        for key in volatileFreezeKeys { object.removeValue(forKey: key) }
        return strippingNulls(object) as? [String: Any]
    }

    private static func strippingNulls(_ value: Any) -> Any {
        if let dictionary = value as? [String: Any] {
            var result: [String: Any] = [:]
            for (key, entry) in dictionary where !(entry is NSNull) {
                result[key] = strippingNulls(entry)
            }
            return result
        }
        if let array = value as? [Any] {
            return array.map(strippingNulls)
        }
        return value
    }

    // MARK: - Remote-freeze manifest identity (document-level comparison)

    /// Raw bytes of the named experiment's manifest file, or nil when it does
    /// not exist. This is "the manifest on screen" for the remote-freeze
    /// identity check — the document as stored, not the decoded model (so
    /// nothing is lost or reordered before comparison).
    public static func manifestData(name: String) -> Data? {
        try? Data(contentsOf: manifestURL(name))
    }

    /// Outcome of comparing two manifest DOCUMENTS (raw JSON bytes) with the
    /// SAME canonicalization the frozenBy:"server" freeze-canonical
    /// verification uses: the volatile freeze-stamp keys
    /// (`volatileFreezeKeys`) removed and explicit nulls stripped, then
    /// parsed-JSON equality. Both documents are canonicalized by THIS engine
    /// — same-engine canonicalization of two documents is well-defined; a
    /// cross-engine byte-canonical hash deliberately does not exist.
    public enum ManifestDocumentComparison: Sendable, Equatable {
        case equal
        /// Field-level mismatch lines (top-level keys that differ; array
        /// valued keys carry element counts).
        case different([String])
        /// One side is not a JSON object — nothing meaningful to compare.
        case unparseable
    }

    /// Compare the LOCAL displayed manifest document against the SERVER's
    /// same-named manifest body, modulo the volatile freeze stamps.
    public static func compareManifestDocuments(
        local: Data, server: Data
    ) -> ManifestDocumentComparison {
        guard
            let localObject = comparableFreezeObject(local),
            let serverObject = comparableFreezeObject(server)
        else { return .unparseable }
        if (localObject as NSDictionary).isEqual(to: serverObject) {
            return .equal
        }
        return .different(
            manifestFieldMismatches(local: localObject, server: serverObject))
    }

    /// One human line per differing TOP-LEVEL key, sorted by key: which side
    /// has it, or that both do but the values differ (with element counts for
    /// array-valued keys — "conditions: differs (local 3, server 5)").
    static func manifestFieldMismatches(
        local: [String: Any], server: [String: Any]
    ) -> [String] {
        var lines: [String] = []
        for key in Set(local.keys).union(server.keys).sorted() {
            switch (local[key], server[key]) {
            case (nil, nil):
                continue
            case (let value?, nil):
                lines.append("\(key): local only\(arrayCountSuffix(value))")
            case (nil, let value?):
                lines.append("\(key): server only\(arrayCountSuffix(value))")
            case (let localValue?, let serverValue?):
                if jsonValuesEqual(localValue, serverValue) { continue }
                if let localArray = localValue as? [Any],
                    let serverArray = serverValue as? [Any]
                {
                    lines.append(
                        "\(key): differs (local \(localArray.count), "
                            + "server \(serverArray.count))")
                } else {
                    lines.append("\(key): differs")
                }
            }
        }
        return lines
    }

    private static func arrayCountSuffix(_ value: Any) -> String {
        guard let array = value as? [Any] else { return "" }
        return " (\(array.count) item\(array.count == 1 ? "" : "s"))"
    }

    /// Foundation JSON values (NSDictionary/NSArray/NSNumber/NSString/NSNull)
    /// compare correctly through `isEqual` — both sides came from
    /// JSONSerialization, so this is parsed-JSON equality.
    private static func jsonValuesEqual(_ a: Any, _ b: Any) -> Bool {
        (a as AnyObject).isEqual(b as AnyObject)
    }

    /// SHA-256 over the canonicalized manifest body (volatile stamps removed,
    /// nulls stripped, sorted keys) — a SAME-ENGINE display fingerprint so a
    /// researcher can confirm a server-only copy against something real.
    /// Never a cross-engine freeze hash; never persisted.
    public static func canonicalManifestBodyHash(_ data: Data) -> String? {
        guard
            let object = comparableFreezeObject(data),
            let canonical = try? JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys])
        else { return nil }
        return sha256Hex(canonical)
    }

    /// Canonical content hash: the manifest minus its lifecycle fields,
    /// encoded with sorted keys. `createdAt` is a lifecycle field like the
    /// rest — two manifests differing only in creation time describe the
    /// same study, and every sibling canonicalization (`volatileFreezeKeys`,
    /// the server's `content_hash` and `manifest_diff`) already excludes it;
    /// this hash was the lone holdout until 2026-08-17. Cleared to "" (not a
    /// dropped key) so `StudyTemplate.strippedBody` bodies — already "" —
    /// hash unchanged.
    public static func manifestHash(_ manifest: ExperimentManifest) -> String {
        var canonical = manifest
        canonical.createdAt = ""
        return legacyManifestHash(canonical)
    }

    /// The pre-2026-08-17 canonicalization, which treated `createdAt` as
    /// content. Kept ONLY so artifacts stamped under it — Swift-frozen
    /// `freezeHash`, native runs' `experiment-hash.txt` — keep verifying
    /// without a false "changed after freeze"/"different epoch" alarm.
    /// Swift-only compat: the server's hash never included `createdAt`, so
    /// there is no server twin. Never stamp new artifacts with this.
    static func legacyManifestHash(_ manifest: ExperimentManifest) -> String {
        var canonical = manifest
        canonical.status = .draft
        canonical.frozenAt = nil
        canonical.freezeHash = nil
        canonical.frozenBy = nil
        canonical.gitCommit = nil
        canonical.appVersion = nil
        canonical.freezeForced = nil
        canonical.forcedGatesSkipped = nil
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(canonical)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Pins the neutral corpus's current hash into the manifest when the
    /// corpus exists on disk. Called at attach: extraction denominates
    /// norm-unit alphas against this corpus, so it is a pinned input even
    /// when confound projection is off. Returns the pinned hash, if any.
    @discardableResult
    public static func pinNeutralCorpus(
        into manifest: inout ExperimentManifest
    ) -> String? {
        let url = VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral", "corpus.jsonl")
        guard let (_, hash) = try? StimulusSet.loadTexts(url: url) else { return nil }
        manifest.neutralCorpusHash = hash
        return hash
    }

    @discardableResult
    public static func pinTaskPrompts(
        _ file: String, into manifest: inout ExperimentManifest
    ) throws -> String {
        // Same resolution as verify() (root-override aware — the study-pack
        // import pins under the test seam and any future root override).
        let url = resolveProjectPath(file)
        guard let data = try? Data(contentsOf: url) else {
            // Typed at the PIN, not only at the run (2026-08-18): the run
            // loop's twin (`ExperimentTasks.loadTaskPrompts`) has thrown this
            // exact sentence as a `missingPrerequisite` refusal since step 7,
            // while the pin verb that an agent hits FIRST threw it bare — so
            // the same rule answered `refused`/65 with a runnable repair at
            // run time and `failed`/70 with "read the reason" at pin time.
            // Same gate, same sentence, same repair as the twin.
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "task prompt file not found: \(url.path)",
                repair: "author \(file) as {\"id\": …, \"prompt\": …} JSONL rows, "
                    + "then steerlab-cli experiment pin-prompts "
                    + "\(manifest.name) \(file)")
        }
        // Same parser the run loop uses (`{"prompt": …}` or `{"text": …}`
        // rows plus science-layer keys) — the historical text-only loader
        // refused valid choice files the run loop accepts. The hash is the
        // raw file bytes either way, so existing pins stay valid.
        _ = try ExperimentTasks.parseTaskPrompts(data)
        let hash = sha256Hex(data)
        manifest.taskPromptsFile = file
        manifest.taskPromptsHash = hash
        return hash
    }

    /// The draft-gated entry to the task-prompts pin — the store twin of the
    /// Studies panel's setup save and of the CLI's `experiment pin-prompts`
    /// (WP0 step 5½: until it existed, "pin a prompt set" was a remedy only
    /// the GUI could perform, and a headless agent could not author a study
    /// that measures anything).
    ///
    /// Deliberately the same shape as `setCapabilityBatteryFile`: nil/empty
    /// CLEARS the pin, anything else goes through the ONE validating pin
    /// helper (`pinTaskPrompts`) so a GUI-authored and a CLI-authored study
    /// carry byte-identical `taskPromptsFile` + `taskPromptsHash`.
    @discardableResult
    public static func setTaskPromptsFile(
        _ file: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = file?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let path = trimmed, !path.isEmpty else {
                manifest.taskPromptsFile = nil
                manifest.taskPromptsHash = nil
                return
            }
            _ = try pinTaskPrompts(path, into: &manifest)
        }
    }

    /// The explicit `evaluation` block a draft write produces from the pins
    /// (2026-07-22 incident: the rubric-FILE + judges path never wrote one,
    /// and a frozen study died at the evaluate stage).
    ///
    /// HOISTED from `ExperimentPanel` (WP0 step 5½): it was already static
    /// and pure, but it lived in view-adjacent code, so the headless
    /// `experiment pin-rubric` verb would have had to re-derive the rule —
    /// two implementations of "what did the researcher declare", and the one
    /// the paper's study used would not be the one the panel wrote.
    /// `ExperimentPanel.evaluationDeclaration` now forwards here.
    ///
    /// Pinned judges + a chosen rubric file ARE a paired-judge declaration
    /// (kind pairedJudge; judgeModel and judgePrompt empty — the manifest
    /// carries the judges, the pinned file the rubric). Without the pin
    /// pair, inline text declares a draft-only inline evaluation, and
    /// nothing declares none.
    public static func evaluationDeclaration(
        judges: [ExperimentManifest.JudgeRef],
        rubricFile: String,
        inlineRubric: String,
        structuredPrompt: String?,
        inlineJudgeModel: String
    ) -> ExperimentManifest.EvaluationSpec? {
        let rubric = rubricFile.trimmingCharacters(in: .whitespacesAndNewlines)
        if !judges.isEmpty, !rubric.isEmpty {
            return ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge, judgeModel: "", judgePrompt: "",
                structuredPrompt: structuredPrompt)
        }
        let inline = inlineRubric.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inline.isEmpty else { return nil }
        return ExperimentManifest.EvaluationSpec(
            kind: .pairedJudge, judgeModel: inlineJudgeModel,
            judgePrompt: inline, structuredPrompt: structuredPrompt)
    }

    /// The draft-gated judge-rubric pin — the store twin of the Studies
    /// panel's rubric picker + judge rows, and of `experiment pin-rubric`.
    ///
    /// Pins the rubric FILE at its current bytes through `JudgeRubricStore`
    /// (the same hash the panel writes), optionally replaces the judge panel,
    /// and — exactly as the panel save does — writes the explicit
    /// `evaluation` declaration the pin pair implies, so a CLI-authored
    /// judged study cannot reach freeze in the 2026-07-22 shape (pins
    /// present, declaration absent). A nil/empty `file` clears the pin and
    /// re-derives the declaration from what is left.
    ///
    /// `judges == nil` keeps the manifest's existing panel; an empty array
    /// clears it. Judge rows are normalised through
    /// `keepingKindOwnedFields()`, the one write funnel, so no path can leak
    /// a kind-foreign pin (a local judge carrying `provider`).
    @discardableResult
    public static func setJudgeRubric(
        file: String?, judges: [ExperimentManifest.JudgeRef]? = nil,
        experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed =
                (file ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                manifest.judgeRubricFile = nil
                manifest.judgeRubricHash = nil
            } else {
                try JudgeRubricStore.pin(trimmed, into: &manifest)
            }
            if let judges {
                let cleaned = judges
                    .map { $0.keepingKindOwnedFields() }
                    .filter {
                        !$0.name.trimmingCharacters(in: .whitespacesAndNewlines)
                            .isEmpty
                    }
                manifest.judges = cleaned.isEmpty ? nil : cleaned
            }
            manifest.evaluation = evaluationDeclaration(
                judges: manifest.judges ?? [],
                rubricFile: manifest.judgeRubricFile ?? "",
                inlineRubric: manifest.evaluation?.judgePrompt ?? "",
                structuredPrompt: manifest.evaluation?.structuredPrompt,
                inlineJudgeModel: manifest.evaluation?.judgeModel ?? manifest.modelID)
        }
    }

    /// Pin a NAMED capability battery file by path + SHA-256, validating at
    /// pin time the JSONL row shape the battery runner (`CapabilityBattery`)
    /// will load — the ONE validating pin helper for every path that pins a
    /// battery the manifest names (study-pack auto-pin today; pickers next),
    /// so a shape-garbage battery refuses NOW instead of at run time.
    /// Distinct from `pinCapabilityBattery(into:)` below, which pins the
    /// manifest's existing file-or-default for the validate/freeze evidence
    /// path without changing that path's non-throwing contract.
    @discardableResult
    public static func pinCapabilityBattery(
        _ file: String, into manifest: inout ExperimentManifest
    ) throws -> String {
        let url = resolveProjectPath(file)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(
                reason: "capability battery file not found: \(url.path)")
        }
        if let problem = PinShapeValidation.capabilityBatteryShapeProblem(
            data, file: file)
        {
            throw ExperimentError(reason: problem)
        }
        let hash = sha256Hex(data)
        manifest.capabilityBatteryFile = file
        manifest.capabilityBatteryHash = hash
        return hash
    }

    /// The draft-gated UI entry to the battery pin: pin a named battery file
    /// (shape-validated, hashed) or clear back to the engine default with
    /// nil. Wraps `pinCapabilityBattery(_:into:)` — the ONE validating pin
    /// helper — through the same gate every science-manifest setter uses,
    /// so the picker cannot write what verify would refuse.
    @discardableResult
    public static func setCapabilityBatteryFile(
        _ file: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = file?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let path = trimmed, !path.isEmpty else {
                manifest.capabilityBatteryFile = nil
                manifest.capabilityBatteryHash = nil
                return
            }
            _ = try pinCapabilityBattery(path, into: &manifest)
        }
    }

    /// Pin a fitted RepE reader artifact by workspace path (audit
    /// 2026-08-01: `readerRefs` had no writer, so the reader-instrument
    /// button gated on a list nothing could populate). Decodes through the
    /// REAL `RepEReader.Artifact` loader (review P1: a loose field
    /// inspection pins artifacts the scorer cannot load), then refuses
    /// what is context-free at pin time — a missing file, a non-reader or
    /// unloadable artifact, a reader fitted on a different model, an
    /// artifact carrying no revision, or a revision that contradicts the
    /// study's pin. Substrate is deliberately NOT refused here: a
    /// server-destined study legitimately pins a server-fitted reader the
    /// LOCAL verify flags, and the substrate check belongs to the engine
    /// that freezes. Re-pinning a concept replaces its entry.
    @discardableResult
    public static func pinReader(
        path: String, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let url = resolveProjectPath(path)
            guard let data = try? Data(contentsOf: url) else {
                throw ExperimentError(
                    reason: "reader artifact not found: \(url.path)")
            }
            let artifact: RepEReader.Artifact
            do {
                artifact = try JSONDecoder().decode(
                    RepEReader.Artifact.self, from: data)
            } catch {
                throw ExperimentError(
                    reason: "'\(path)' is not a loadable "
                        + "\(RepEReader.artifactType) artifact — pin the "
                        + "reader JSON a fit run wrote, not a steering "
                        + "vector or sidecar. Details: \(error)")
            }
            guard artifact.modelID == manifest.modelID else {
                throw ExperimentError(
                    reason: "reader '\(artifact.concept)' was fitted on "
                        + "\(artifact.modelID), not the study model "
                        + "\(manifest.modelID) — readers measure nothing on "
                        + "a model they were not fitted on")
            }
            guard let revision = artifact.revision, !revision.isEmpty else {
                throw ExperimentError(
                    reason: "reader '\(artifact.concept)' carries no model "
                        + "revision — readers bind to exact fitted bytes")
            }
            if let pinned = manifest.modelRevision, revision != pinned {
                throw ExperimentError(
                    reason: "reader '\(artifact.concept)' was fitted on "
                        + "revision \(revision.prefix(12))…, not the study's "
                        + "pinned \(pinned.prefix(12))…")
            }
            var refs = manifest.readerRefs ?? []
            refs.removeAll { $0.concept == artifact.concept }
            refs.append(
                ExperimentManifest.ReaderRef(
                    path: path, hash: sha256Hex(data),
                    concept: artifact.concept))
            manifest.readerRefs = refs
        }
    }

    /// Remove one concept's pinned reader; an emptied list stores as ABSENT.
    /// If the reader instrument would be orphaned by the removal (verify
    /// refuses `repeReaderScore` with no pinned readers), it is removed in
    /// the same write — one edit, never an invalid intermediate manifest.
    @discardableResult
    public static func removeReader(
        concept: String, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            var refs = manifest.readerRefs ?? []
            refs.removeAll { $0.concept == concept }
            manifest.readerRefs = refs.isEmpty ? nil : refs
            if refs.isEmpty,
                var instruments = manifest.outcomeInstruments,
                instruments.contains("repeReaderScore")
            {
                instruments.removeAll { $0 == "repeReaderScore" }
                manifest.outcomeInstruments =
                    instruments.isEmpty ? nil : instruments
            }
        }
    }

    /// Pins grand-mean concepts into a draft manifest. Each target pins its
    /// own `prompts/emotions/<name>/stories.jsonl` hash as stimulusSetHash,
    /// AND the manifest pins the grand-mean corpus (population membership +
    /// every member's hash): the vector is mean(concept) − mean(corpus), so
    /// the population is part of the recipe. `corpusConcepts` widens the
    /// population beyond the attached targets (targets are always members);
    /// the emotion paper's default reading position (mean from token 50)
    /// applies when no pooling token is given.
    public static func attachGrandMeanConcepts(
        _ concepts: [String],
        corpusConcepts: [String] = [],
        poolFromToken: Int? = nil,
        into manifest: inout ExperimentManifest
    ) throws {
        var members: [String] = []
        var seen = Set<String>()
        for name in (manifest.grandMeanCorpus?.concepts ?? []) + corpusConcepts + concepts
        where seen.insert(name).inserted {
            members.append(name)
        }
        var hashes: [String: String] = [:]
        for member in members {
            guard let hash = storiesHash(for: member) else {
                throw ExperimentError(
                    reason: "no stories.jsonl for grand-mean corpus member '\(member)' "
                        + "under prompts/emotions/")
            }
            hashes[member] = hash
        }
        for concept in concepts {
            guard let hash = hashes[concept] else { continue }  // targets are members
            let options = ExtractionOptions(
                method: .emotionGrandMean,
                readingPosition: .meanFromToken(poolFromToken ?? 50))
            manifest.concepts.removeAll { $0.name == concept }
            manifest.concepts.append(
                makeConceptRef(name: concept, stimulusSetHash: hash, options: options))
        }
        manifest.grandMeanCorpus = .init(concepts: members, hashes: hashes)
    }

    // MARK: - Validation evidence

    /// The inputs a `validate` run actually certifies: vectors depend on
    /// (model, revision, pinned concepts + options, neutral corpus) and on
    /// nothing else — conditions and sampling settings are deliberately out
    /// of scope, so adding a condition does not invalidate prior validation.
    private struct ValidationScope: Codable {
        var modelID: String
        var modelRevision: String?
        var concepts: [ExperimentManifest.ConceptRef]
        var neutralCorpusHash: String?
        /// The population defines the grand-mean vectors: changing corpus
        /// membership or any member's stories invalidates validation. Nil is
        /// omitted from the encoding, so legacy scope hashes are unchanged.
        var grandMeanCorpus: ExperimentManifest.GrandMeanCorpus?
        /// Appended ONLY when the manifest has variant conditions (the
        /// battery is validation evidence for variants, so a battery change
        /// invalidates their validation). Nil is omitted from the encoding,
        /// so legacy and non-variant scope hashes are unchanged. The server
        /// extends its scope-hash function the same way — equivalent scope
        /// semantics, engine-local hashing.
        var capabilityBatteryHash: String?
        /// Appended ONLY when a validation depth is declared (any of the
        /// four D4 fields): evidence validated at one depth does not certify
        /// a manifest that now declares different depths, so the declaration
        /// enters the scope. Nil is omitted, so undeclared-legacy scope
        /// hashes are unchanged. Server twin: the `validationDepths` payload
        /// key in `validation_scope_hash`.
        var validationDepths: ValidationDepths?

        struct ValidationDepths: Codable {
            var validationLayer: Int?
            var validationLayerFraction: Double?
            var validationLayers: [Int]?
            var validationLayerFractions: [Double]?
        }
    }

    static func validationScopeHash(_ manifest: ExperimentManifest) -> String {
        let scope = ValidationScope(
            modelID: manifest.modelID, modelRevision: manifest.modelRevision,
            concepts: manifest.concepts, neutralCorpusHash: manifest.neutralCorpusHash,
            grandMeanCorpus: manifest.grandMeanCorpus.map {
                .init(concepts: $0.concepts.sorted(), hashes: $0.hashes)
            },
            capabilityBatteryHash: manifest.variantConditions.isEmpty
                ? nil
                : manifest.capabilityBatteryHash,
            validationDepths: manifest.validationLayer == nil
                && manifest.validationLayerFraction == nil
                && manifest.validationLayers == nil
                && manifest.validationLayerFractions == nil
                ? nil
                : .init(
                    validationLayer: manifest.validationLayer,
                    validationLayerFraction: manifest.validationLayerFraction,
                    validationLayers: manifest.validationLayers,
                    validationLayerFractions: manifest.validationLayerFractions))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(scope)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The newest `validate` run whose manifest snapshot matches this
    /// manifest's validation scope, if any. The experiment name is not part
    /// of the scope: a duplicate with identical pins inherits the evidence.
    ///
    /// `runSubstrate` (Mac-authority mode, 2026-07-21) is the engine the
    /// study's measured runs will execute on — the substrate the evidence
    /// must certify. Default: this engine (freeze-where-you-run, the
    /// historical rule). A local freeze of a study that will RUN on the
    /// server passes `WorkspaceScoping.serverSubstrate`, so an imported
    /// server validate run (evidence bundle brought home) counts — it is
    /// substrate-MATCHED for those runs, while this engine's own evidence
    /// then does not count.
    public static func validationEvidence(
        for manifest: ExperimentManifest,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> URL? {
        let scope = validationScopeHash(manifest)
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        for entry in entries.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                entry.lastPathComponent.range(
                    of: #"-validate(-\d+)?$"#, options: .regularExpression) != nil,
                let data = try? Data(
                    contentsOf: entry.appending(component: "experiment.json")),
                let snapshot = try? JSONDecoder().decode(
                    ExperimentManifest.self, from: data),
                validationScopeHash(snapshot) == scope,
                isCompleteValidationRun(entry, scope: scope, runSubstrate: runSubstrate)
            else { continue }
            return entry
        }
        return nil
    }

    /// `vacuousConcepts` names the eligible concepts this run scored no
    /// held-out probe for; pass `[]` (the default) when every eligible
    /// concept was probed. It is ALWAYS stamped, so a run written by this
    /// code is self-describing and an ABSENT key means "legacy evidence".
    public static func writeValidationEvidence(
        for manifest: ExperimentManifest,
        runDirectory: URL,
        capabilityBattery: [CapabilityBatteryConditionResult]? = nil,
        vacuousConcepts: [String] = []
    ) throws {
        let evidence = ValidationEvidenceFile(
            schemaVersion: 1,
            task: "validate",
            completedAt: ISO8601DateFormatter().string(from: Date()),
            validationScopeHash: validationScopeHash(manifest),
            substrate: evidenceSubstrate,
            reportFile: "validation-report.json",
            batteryResults: capabilityBattery,
            vacuousConcepts: vacuousConcepts.sorted())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(evidence).write(
            to: runDirectory.appending(component: "validation-evidence.json"),
            options: .atomic)
    }

    /// Capability-battery results stamped in a validate run's evidence file
    /// (key "batteryResults"), or nil when the evidence is legacy/missing.
    /// Used by the freeze variant gate and surfaced to reports.
    public static func validationEvidenceBatteryResults(
        at runDirectory: URL
    ) -> [CapabilityBatteryConditionResult]? {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "validation-evidence.json")),
            let evidence = try? JSONDecoder().decode(ValidationEvidenceFile.self, from: data)
        else { return nil }
        return evidence.batteryResults
    }

    /// Whether a pinned concept OWES a held-out probe at validate time.
    ///
    /// False for directions that were never read off a concept's stimuli —
    /// an OptVec vector (its evidence is the eval run's eval.json, OptVec
    /// plan §6), an imported Gemma Scope SAE decoder row (its evidence is
    /// the roster's discovery snapshot + qualification), and any artifact
    /// pin whose source method cannot be resolved. Asking those for a
    /// `validation.jsonl` invents an obligation they can never meet.
    /// Server twin: `ExtractionMethod.has_source_concept`.
    static func owesHeldOutProbe(_ ref: ExperimentManifest.ConceptRef) -> Bool {
        guard let effective = ref.effectiveMethod else { return false }
        return effective.hasSourceConcept
    }

    /// Workspace-relative path of the `validation.jsonl` a pinned concept's
    /// held-out probe reads — the DATA concept's file, resolved by the
    /// effective method's family (an artifact-pinned "crit-gm" reads
    /// "crit"'s). Named in refusals so the remedy is a real file path.
    static func heldOutProbePath(_ ref: ExperimentManifest.ConceptRef) -> String {
        conceptValidationRelativePath(
            name: ref.dataConcept,
            isPaired: (ref.effectiveMethod ?? ref.options.method).isPaired)
    }

    /// Concepts the relied-upon validate evidence scored NO held-out probe
    /// for — VACUOUS evidence (2026-08-17 firewall repair).
    ///
    /// The default state of a seeded workspace is that no concept has a
    /// `validation.jsonl`, so `validate` completed as a no-op and its run
    /// satisfied the `validateEvidence` freeze gate: an unforced, unstamped
    /// freeze was indistinguishable from a validated one, while `data check`
    /// called the same missing file a blocker. `validate` now stamps what it
    /// could not probe and this reads the stamp back.
    ///
    /// Returns [] when there is no matching evidence at all (the plain
    /// missing-evidence gate covers that), when the evidence predates the
    /// stamp (legacy runs keep passing, by design), or when every eligible
    /// concept was probed. Server twin:
    /// `experiment_store._vacuous_validate_evidence`.
    static func vacuousValidationEvidence(
        for manifest: ExperimentManifest,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> [String] {
        guard
            let directory = validationEvidence(for: manifest, runSubstrate: runSubstrate),
            let data = try? Data(
                contentsOf: directory.appending(component: "validation-evidence.json")),
            let evidence = try? JSONDecoder().decode(ValidationEvidenceFile.self, from: data),
            let vacuous = evidence.vacuousConcepts
        else { return [] }
        // Only concepts this manifest still pins matter: evidence is matched
        // by SCOPE, and a duplicate that dropped a concept inherits it.
        let pinned = Set(manifest.concepts.map(\.name))
        return vacuous.filter(pinned.contains).sorted()
    }

    /// The `validateEvidence` gate's refusal text when the matching validate
    /// run is VACUOUS, naming the exact files that would make it real.
    /// nil when the evidence is not vacuous. Shared by `freeze`, the forced
    /// path, and `freezeReadiness`, so the three cannot disagree.
    static func vacuousValidationEvidenceProblem(
        for manifest: ExperimentManifest,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> String? {
        let vacuous = vacuousValidationEvidence(
            for: manifest, runSubstrate: runSubstrate)
        guard !vacuous.isEmpty else { return nil }
        guard
            let repair = vacuousValidationRepairAction(
                for: manifest, vacuousConcepts: vacuous)
        else { return nil }
        return "the matching validate run scored NO held-out probe for "
            + "concept(s) \(vacuous.joined(separator: ", ")) — it is VACUOUS "
            + "evidence, not validation. " + repair
    }

    /// The remedy half of the vacuous-evidence refusal, split out (WP0 step
    /// 2) so the `validateEvidence` gate can carry the file-path remedy as a
    /// machine-readable `repairAction` without a second copy of the paths.
    /// Composition is byte-preserving: problem-sentence + " " + this string
    /// is the message this gate has thrown since 2026-08-17.
    static func vacuousValidationRepairAction(
        for manifest: ExperimentManifest, vacuousConcepts: [String]
    ) -> String? {
        guard !vacuousConcepts.isEmpty else { return nil }
        let byName = Dictionary(
            manifest.concepts.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        let paths = vacuousConcepts.compactMap { byName[$0].map(heldOutProbePath) }
        return "Author the never-named scenarios "
            + "(\(paths.joined(separator: ", "))) as "
            + "{\"text\": …, \"expresses\": true|false} rows and re-run "
            + "'steerlab-cli experiment validate \(manifest.name)', or freeze "
            + "--force to record an unvalidated experiment"
    }

    /// The EXECUTABLE form of the vacuous-evidence repair (WP0 step 7, P5).
    ///
    /// Identical to `vacuousValidationRepairAction` plus the step it was
    /// missing: `attach` re-pins `validationHash`, which `attach` set to
    /// explicitly-absent when the file did not exist. Without it the sequence
    /// the refusal asks for — author the file, re-run validate — refuses at
    /// the pin surface instead of validating, which is the loop dry run #1
    /// could not escape.
    ///
    /// Kept SEPARATE from the prose version deliberately: the prose is the
    /// cross-engine refusal string and is asserted byte-for-byte on both
    /// engines, so extending it would be a contract change; the repairAction
    /// is a machine field the envelope carries and can say more.
    static func vacuousValidationMachineRepair(
        for manifest: ExperimentManifest, vacuousConcepts: [String],
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> String? {
        guard let prose = vacuousValidationRepairAction(
            for: manifest, vacuousConcepts: vacuousConcepts)
        else { return nil }
        guard !vacuousConcepts.isEmpty else { return prose }
        // The two halves name DIFFERENT binaries on purpose: authoring
        // (`attach`) is Mac-authority and only `steerlab-cli` has it, while
        // the evidence the gate then reads must come from the RUN substrate
        // — see `validateCLI(forRunSubstrate:)`.
        return prose + " — and note the ORDER: authoring the file makes it "
            + "\"appear after attach\" against a pin that recorded its "
            + "absence, so re-pin it first: steerlab-cli experiment attach "
            + "\(manifest.name) \(vacuousConcepts.joined(separator: " ")) && "
            + "\(validateCLI(forRunSubstrate: runSubstrate)) experiment "
            + "validate \(manifest.name)"
    }

    /// Which CLI can PRODUCE validate evidence for a given run substrate.
    ///
    /// Gate-5 dry run #2 (P2): under `freeze --run-substrate server` the
    /// `validateEvidence` gate reads evidence stamped with the SERVER
    /// substrate, and evidence from this engine will never satisfy it — but
    /// the refusal's repair named `steerlab-cli experiment validate` anyway,
    /// which is a repair that cannot work. A repair must name the engine that
    /// can satisfy the gate it repairs.
    static func validateCLI(forRunSubstrate substrate: String) -> String {
        substrate == WorkspaceScoping.serverSubstrate
            ? "steerlab-server" : "steerlab-cli"
    }

    /// The battery a variant study validates against: the manifest's pin, or
    /// the shared VariantRobustness default preset battery (the same
    /// constant, not retyped). Returns nil when the file cannot be read.
    public static func effectiveCapabilityBattery(
        for manifest: ExperimentManifest
    ) -> (file: String, hash: String)? {
        let file =
            manifest.capabilityBatteryFile ?? VariantRobustness.defaultPreset.batteryFile
        guard let data = try? Data(contentsOf: resolveProjectPath(file)) else { return nil }
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        if let pinned = manifest.capabilityBatteryHash, pinned != hash { return nil }
        return (file, hash)
    }

    /// Pins the capability battery (manifest's file or the shared default)
    /// into the manifest. Called by `validate`/`freeze` for variant studies
    /// so the validation scope hash and the evidence agree on the battery.
    /// Shape-checked like every other battery pin (one universal contract):
    /// a malformed file is treated exactly like an ABSENT one — the pin is
    /// skipped (nil), preserving this path's nil-tolerant freeze-time
    /// contract — and readiness reports the file as invalid with the plain
    /// shape detail, so the skip surfaces instead of pinning bytes the
    /// battery runner would refuse.
    @discardableResult
    public static func pinCapabilityBattery(
        into manifest: inout ExperimentManifest
    ) -> (file: String, hash: String)? {
        guard let battery = effectiveCapabilityBattery(for: manifest) else { return nil }
        guard
            let data = try? Data(contentsOf: resolveProjectPath(battery.file)),
            PinShapeValidation.capabilityBatteryShapeProblem(
                data, file: battery.file) == nil
        else { return nil }
        manifest.capabilityBatteryFile = battery.file
        manifest.capabilityBatteryHash = battery.hash
        return battery
    }

    private static func isCompleteValidationRun(
        _ runDirectory: URL, scope: String,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> Bool {
        // A run whose own status says it did not complete is never evidence
        // (retention 2026-07-24). Today this is belt-and-braces — a partial
        // validate run has no `validation-evidence.json`, so the check below
        // already refuses it — but partial run directories now legitimately
        // EXIST in runs/, imported from failed cluster jobs, and a gate that
        // only refuses them by accident is one refactor away from accepting
        // one.
        if RunStatusFile.isPartial(at: runDirectory) { return false }
        guard let evidence = completeValidationEvidence(runDirectory, scope: scope) else {
            return false
        }
        // Same-substrate rule (explicit, not accidental), re-keyed on the
        // RUN substrate (Mac-authority mode, 2026-07-21): evidence stamped
        // with a substrate other than the one the study's measured runs
        // will execute on never counts — CUDA/HF activations do not match
        // MLX/Metal, so vectors must be validated on the engine that RUNS
        // the study. The freeze may happen elsewhere (the Mac is the
        // authority; the cluster is a runner). Evidence without the stamp
        // is legacy, treated as native to the perspective engine (mirror of
        // the server's `_matching_validate_evidence`, which keeps
        // freeze-engine == run-engine — the server only ever freezes
        // studies that run on it).
        if let substrate = evidence.substrate, substrate != runSubstrate {
            return false
        }
        return true
    }

    /// The evidence file of a COMPLETE validate run whose scope matches,
    /// REGARDLESS of substrate (the same-substrate gate stays in
    /// `isCompleteValidationRun`; the raw read also feeds the cross-substrate
    /// advisory): schema-1 "validate" evidence with the matching scope hash
    /// plus a non-empty validation report.
    private static func completeValidationEvidence(
        _ runDirectory: URL, scope: String
    ) -> ValidationEvidenceFile? {
        let evidenceURL = runDirectory.appending(component: "validation-evidence.json")
        guard
            let evidenceData = try? Data(contentsOf: evidenceURL),
            let evidence = try? JSONDecoder().decode(ValidationEvidenceFile.self, from: evidenceData),
            evidence.schemaVersion == 1,
            evidence.task == "validate"
        else { return nil }
        // Scope proof. For evidence written by THIS engine (or legacy
        // unstamped evidence) the stamped hash must equal this engine's
        // scope hash, byte for byte. Evidence stamped with a FOREIGN
        // substrate (an imported server validate run, Mac-authority mode)
        // carries the OTHER engine's engine-local canonicalization of the
        // same scope — the hashes are deliberately not comparable across
        // engines, so the scope proof for foreign evidence is the run's
        // manifest SNAPSHOT, which every caller re-hashes with THIS
        // engine's function before this check. A foreign stamp must still
        // be present and non-empty: evidence certifying no scope at all is
        // incomplete on any engine.
        let foreignEngine =
            evidence.substrate != nil && evidence.substrate != evidenceSubstrate
        if foreignEngine {
            guard !evidence.validationScopeHash.isEmpty else { return nil }
        } else {
            guard evidence.validationScopeHash == scope else { return nil }
        }
        let reportURL = runDirectory.appending(component: evidence.reportFile ?? "report.json")
        // Non-empty validation content, either engine's report shape: Swift
        // writes a top-level "validation" object, the server's
        // validation-report.json a top-level "concepts" object.
        guard
            let reportData = try? Data(contentsOf: reportURL),
            let report = try? JSONSerialization.jsonObject(with: reportData) as? [String: Any],
            report["validation"] != nil || report["concepts"] != nil
        else { return nil }
        return evidence
    }

    // MARK: Cross-substrate validate-evidence advisory (WS7.1)

    /// Best-known engine of a validate run: the evidence's own `substrate`
    /// stamp, else the run's canonical `config.json` stamp (covers evidence
    /// written before the substrate field existed), else nil (unknowable —
    /// pre-stamp runs from either engine are never accused).
    private static func validateEvidenceEngine(
        at runDirectory: URL, evidence: ValidationEvidenceFile
    ) -> String? {
        if let substrate = evidence.substrate, !substrate.isEmpty {
            return substrate
        }
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: RunMetadata.fileName)),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any],
            let substrate = dictionary["substrate"] as? String,
            !substrate.isEmpty
        else { return nil }
        return substrate
    }

    /// WS7.1 non-blocking advisory: scope-matched validate evidence exists,
    /// but the best evidence the freezing engine can name came from the OTHER
    /// engine — the study should re-validate on the freezing substrate before
    /// its runs count.
    ///
    /// Returns nil when (a) the evidence the freeze gate relies on is
    /// same-engine (or its engine is unknowable — legacy runs are never
    /// accused), or (b) no scope-matched evidence exists at all (the plain
    /// missing-evidence gate already covers that). Wording is byte-identical
    /// to the server's `cross_substrate_validation_advisory` modulo engine
    /// names.
    ///
    /// `perspective` is the substrate about to FREEZE (default: this engine).
    /// Passing `WorkspaceScoping.serverSubstrate` evaluates the same shared
    /// tree from the server's viewpoint — how the panel warns BEFORE a
    /// server-routed freeze on a workspace-PAIRED server, instead of only in
    /// the server's own freeze-time advisory. Unstamped legacy evidence is
    /// treated as native to the perspective engine, mirroring both engines'
    /// own gate rule.
    public static func crossSubstrateValidationAdvisory(
        for manifest: ExperimentManifest,
        perspective: String = ExperimentStore.evidenceSubstrate
    ) -> String? {
        let scope = validationScopeHash(manifest)
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        var candidates: [(directory: URL, evidence: ValidationEvidenceFile)] = []
        for entry in entries.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                entry.lastPathComponent.range(
                    of: #"-validate(-\d+)?$"#, options: .regularExpression) != nil,
                let data = try? Data(
                    contentsOf: entry.appending(component: "experiment.json")),
                let snapshot = try? JSONDecoder().decode(
                    ExperimentManifest.self, from: data),
                validationScopeHash(snapshot) == scope,
                let evidence = completeValidationEvidence(entry, scope: scope)
            else { continue }
            candidates.append((entry, evidence))
        }
        // The newest gate-passing run is what freeze/run would rely on; its
        // config.json can still reveal a foreign engine behind a legacy
        // unstamped evidence file.
        if let reliedUpon = candidates.first(where: {
            $0.evidence.substrate == nil || $0.evidence.substrate == perspective
        }) {
            guard
                let engine = validateEvidenceEngine(
                    at: reliedUpon.directory, evidence: reliedUpon.evidence),
                engine != perspective
            else { return nil }
            return crossSubstrateMessage(engine: engine, perspective: perspective)
        }
        // No gate-passing evidence: advise when known-foreign evidence exists.
        for candidate in candidates {
            if let engine = validateEvidenceEngine(
                at: candidate.directory, evidence: candidate.evidence),
                engine != perspective
            {
                return crossSubstrateMessage(engine: engine, perspective: perspective)
            }
        }
        return nil
    }

    private static func crossSubstrateMessage(
        engine: String, perspective: String = evidenceSubstrate
    ) -> String {
        "validation evidence was produced on \(engine); runs on "
            + "\(perspective) should re-validate on-substrate"
    }

    /// Read-only answer to "would `freeze` succeed right now?" — the same
    /// verify pass and evidence gates `freeze(name:)` enforces, reported as
    /// plain-words unmet gates instead of a thrown error, and WITHOUT any of
    /// freeze's side effects (no revision save, no battery-pin save, no
    /// external-input copying). Empty `unmetGates` means ready to freeze.
    public struct FreezeReadiness: Sendable, Equatable {
        public let unmetGates: [String]
        /// Non-blocking notes — worth surfacing next to the gates but never
        /// a reason freeze would refuse (e.g. hand-created variants without
        /// sweep-selection provenance: fine for exploration, distinguishable
        /// from promoted agents in evidence-grade studies).
        public let advisories: [String]
        public var ready: Bool { unmetGates.isEmpty }

        public init(unmetGates: [String], advisories: [String] = []) {
            self.unmetGates = unmetGates
            self.advisories = advisories
        }

        /// Compact one-liner for UI captions: "ready to freeze", or the
        /// first `maxGates` unmet gates in plain words plus a remainder
        /// count. Pure formatting — unit-testable without a store.
        public func displayLine(maxGates: Int = 3) -> String {
            guard !unmetGates.isEmpty else { return "ready to freeze" }
            let shown = unmetGates.prefix(max(1, maxGates))
            var line = "not ready to freeze: " + shown.joined(separator: " · ")
            let remaining = unmetGates.count - shown.count
            if remaining > 0 {
                line += " · +\(remaining) more"
            }
            return line
        }
    }

    /// Strips freeze's error preamble ("cannot freeze '<name>': ") so gate
    /// reasons read as plain words in a readiness caption. Pure; testable.
    static func plainGateReason(_ reason: String, experimentName: String) -> String {
        let prefix = "cannot freeze '\(experimentName)': "
        guard reason.hasPrefix(prefix) else { return reason }
        return String(reason.dropFirst(prefix.count))
    }

    /// Computes `FreezeReadiness` for a manifest by running `verify` plus
    /// every freeze gate in report-only mode (the throwing gate checks are
    /// wrapped, never re-implemented, so readiness cannot drift from what
    /// `freeze` actually enforces). Pass `violations` when the caller has
    /// already run `verify` to avoid hashing every pinned file twice.
    /// Whether the MODEL-OUTPUT freeze surfaces apply to this manifest.
    ///
    /// The one decision shared by readiness and freeze, so the two cannot
    /// give opposite answers about the same manifest (external review round
    /// 13). A multi-agent study runs a SCENARIO; under the never-delete rule
    /// it may carry concepts, injection conditions, agents and a J-lens
    /// readout from before a kind switch, and executes none of them. Carried
    /// model-output state may ADVISE, and may not block or be auto-pinned.
    ///
    /// Deliberately outside this — a panel loads a model and is judged like
    /// any other study: pinned revision, loadable dtype, judge validity, git
    /// cleanliness. Python twin: `model_output_surfaces_operative`.
    public static func modelOutputSurfacesOperative(
        _ manifest: ExperimentManifest
    ) -> Bool {
        manifest.studyKind == .modelOutput
    }

    public static func freezeReadiness(
        for manifest: ExperimentManifest,
        violations: [String]? = nil,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> FreezeReadiness {
        guard manifest.status == .draft else {
            return FreezeReadiness(
                unmetGates: ["'\(manifest.name)' is already \(manifest.status.rawValue)"])
        }
        var gates: [String] = []
        // Probe copy mirrors freeze's in-memory pinning (revision from the
        // local cache, default capability battery) WITHOUT saving anything.
        var probe = manifest
        if probe.modelRevision == nil {
            probe.modelRevision = SteeredContainerLoader.cachedRevision(for: probe.modelID)
        }
        if modelOutputSurfacesOperative(probe), !probe.variantConditions.isEmpty,
            probe.capabilityBatteryFile == nil
        {
            pinCapabilityBattery(into: &probe)
        }
        // Same in-memory mirror for a trained adapter's dataset pins, so
        // readiness answers "would freeze succeed right now?" rather than
        // reporting a gate that freeze's own auto-pin would have satisfied.
        pinTrainingProvenance(into: &probe)
        // A caller-supplied verify result is only reusable when the probe's
        // pins are the manifest's pins (a freshly probed battery or dataset
        // pin adds hash checks the caller's verify never saw).
        let pinsNewlyProbed =
            probe.capabilityBatteryFile != manifest.capabilityBatteryFile
            || probe.variantConditions.map(\.trainingProvenance)
                != manifest.variantConditions.map(\.trainingProvenance)
        gates += ((pinsNewlyProbed ? nil : violations) ?? verify(probe))
            .map { "pin violation: \($0)" }
        if probe.modelRevision == nil {
            gates.append(
                "model revision is not pinned and \(probe.modelID) is not in the "
                    + "local HF cache — load the model once or run Validate Study")
        }
        let usesLegacyConceptVectors = !probe.concepts.isEmpty || !probe.conditions.isEmpty
        // Readiness answers "would freeze succeed right now?", so it must
        // scope exactly as freeze does — otherwise a panel reads as unready
        // for gates freeze no longer asks of it (external review round 13).
        let modelOutputSurfaces = modelOutputSurfacesOperative(probe)
        if modelOutputSurfaces, usesLegacyConceptVectors,
            !optvecExemptFromValidateGate(probe)
        {
            if validationEvidence(for: probe, runSubstrate: runSubstrate) == nil {
                gates.append(
                    "no validate run matches the current pins (model+revision, concepts, "
                        + "neutral corpus) — run Validate Study")
            } else if let vacuous = vacuousValidationEvidenceProblem(
                for: probe, runSubstrate: runSubstrate)
            {
                gates.append(vacuous)
            }
        }
        func gate(_ check: (ExperimentManifest) throws -> Void) {
            do {
                try check(probe)
            } catch let error as ExperimentError {
                gates.append(plainGateReason(error.reason, experimentName: probe.name))
            } catch {
                gates.append("\(error)")
            }
        }
        if modelOutputSurfaces {
            gate(checkVariantValidity)
            gate { try checkVariantBatteryEvidence($0, runSubstrate: runSubstrate) }
        }
        // Judge validity applies to every kind: a panel's turns are flattened
        // into generations and judged like any other study's output.
        gate(checkJudgeEvaluationValidity)
        // Where freeze would auto-commit the workspace (shipped-app
        // semantics), dirty pins are self-healing — reporting them as an
        // unmet gate would contradict "would freeze succeed right now?".
        // The gate is only a readiness blocker where auto-commit is off
        // (the legacy code-checkout root, where freeze never commits).
        if !freezeAutoCommitIsEnabled() {
            gate(checkGitPinCleanliness)
        }
        return FreezeReadiness(
            unmetGates: gates,
            advisories: freezeAdvisories(for: manifest, runSubstrate: runSubstrate))
    }

    /// Non-blocking freeze advisories. Hand-created variant conditions (no
    /// sweep-promotion birth certificate) are legal — exploration, pilots,
    /// robustness checks — but an evidence-grade study should compare agents
    /// whose settings were SELECTED on dev data by a declared rule, so the
    /// difference is surfaced at the moment of freezing, without forbidding
    /// anything.
    /// Every concept pinned to an OptVec artifact, as (concept name, pin)
    /// pairs. Pure manifest reading, so the freeze gates and advisories can
    /// both ask it. Server twin: `experiment_store.optvec_pinned_concepts`.
    static func optvecPinnedConcepts(
        _ manifest: ExperimentManifest
    ) -> [(name: String, pin: ExperimentManifest.ConceptRef.VectorArtifactPin)] {
        manifest.concepts.compactMap { ref in
            guard ref.options.method == .pinnedArtifact,
                let pin = ref.vectorArtifact,
                pin.sourceMethod == ExtractionMethod.optvec.rawValue
            else { return nil }
            return (ref.name, pin)
        }
    }

    /// Whether the validate-evidence freeze gate has nothing to ask of this
    /// manifest because every concept it declares is an OptVec direction.
    ///
    /// THE RULE (OptVec plan §6): an optvec concept has nothing to validate.
    /// `validate` scores a held-out probe against the recipe's class means;
    /// an optvec vector has no stimuli, no classes and no validation.jsonl,
    /// so a validate run could never exist — gating on one would make an
    /// optvec confirm study freezable only under --force, i.e. permanently
    /// non-citable, a stamp about the FIREWALL rather than the science. The
    /// evidence that certifies the direction is the OptVec eval run's
    /// eval.json (test split, untouched by gradients and by checkpoint
    /// selection), surfaced by `freezeAdvisories`.
    ///
    /// Deliberately narrow: applies only when concepts exist and EVERY one
    /// is optvec-pinned — a mixed study still owes a validate run for its
    /// ordinary concepts, and a variant study keeps the gate. Server twin:
    /// `experiment_store.optvec_exempt_from_validate_gate`.
    static func optvecExemptFromValidateGate(
        _ manifest: ExperimentManifest
    ) -> Bool {
        guard !manifest.concepts.isEmpty, manifest.variantConditions.isEmpty
        else { return false }
        return optvecPinnedConcepts(manifest).count == manifest.concepts.count
    }

    /// Non-blocking advisories about MISFILED held-out sets (2026-08-19).
    ///
    /// A concept's `validation.jsonl` has two possible homes and the RECIPE
    /// decides which is canonical — paired recipes read `prompts/concepts/`,
    /// the grand-mean recipe reads `prompts/emotions/`. The lookup falls
    /// back to the other home rather than reading a misfiled set as absent,
    /// so the hash IS pinned; what the researcher still needs to be told is
    /// that the file is not where its recipe says it lives (or that it is in
    /// both places). Python twin:
    /// `experiment_store._validation_lookup_advisories`.
    static func validationLookupAdvisories(
        for manifest: ExperimentManifest
    ) -> [String] {
        guard conceptMachineryOperative(manifest) else { return [] }
        return manifest.concepts.compactMap { ref in
            // Nothing to validate — no home, no advisory.
            guard ref.effectiveMethod?.hasSourceConcept != false else { return nil }
            let location = resolveConceptValidation(
                name: ref.dataConcept,
                isPaired: !(ref.effectiveMethod ?? ref.options.method)
                    .usesStoryCorpus)
            return validationLookupAdvisory(concept: ref.name, location: location)
        }
    }

    static func freezeAdvisories(
        for manifest: ExperimentManifest,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> [String] {
        var advisories: [String] = []
        // OptVec-pinned concepts: name the evidence run that stands in for
        // validate (plan §6 — eval.json on the test split is the
        // validate-equivalent, advisory-first), or say that none is
        // recorded. Never a gate: an optvec concept has nothing to validate.
        for (name, pin) in optvecPinnedConcepts(manifest) {
            if let evalRun = pin.optvecEvalRun,
                pin.optvecEvalRunVerified == false
            {
                // Named but NOT verifiable at attach: say so instead of
                // describing contents nobody has seen.
                let reason = pin.optvecEvalRunUnverifiedReason ?? "unverified"
                advisories.append(
                    "concept '\(name)' is an OptVec direction naming eval "
                        + "run '\(evalRun)', but the reference could NOT be "
                        + "verified at attach (\(reason)) — treat it as no "
                        + "verifiable eval evidence: confirm the run "
                        + "completed and certifies this artifact's tensor "
                        + "hash, then re-attach so the citation is checked")
            } else if let evalRun = pin.optvecEvalRun {
                let verified =
                    pin.optvecEvalRunVerified == true
                    ? " — verified at attach: its eval.json certifies this "
                        + "artifact's tensor hash"
                    : " — recorded before attach-time verification; the "
                        + "reference was never checked against this artifact "
                        + "(re-attach to verify)"
                advisories.append(
                    "concept '\(name)' is an OptVec direction — its "
                        + "validate-equivalent evidence is the OptVec eval "
                        + "run '\(evalRun)' (test-split shift/anchor/"
                        + "capability/fluency), not a validate run\(verified)")
            } else {
                advisories.append(
                    "concept '\(name)' is an OptVec direction with NO "
                        + "recorded eval run — run optvec eval on the test "
                        + "split and re-attach with the eval run named, or "
                        + "the direction enters the study with no held-out "
                        + "evidence at all")
            }
        }
        // Forward-referenced conditions are BY DEFINITION sweep-promoted
        // (the server resolves them from a criterion promotion at run
        // time) — never "hand-created".
        // An ABLATION-only agent is correctly unpromoted: ablation has no
        // layer×alpha grid, so there was no cell to select and no sweep to
        // promote from. Advising "promote agents from sweeps" for one would
        // name a remedy that does not exist, and an advisory a researcher
        // learns to ignore stops catching the hand-tuned steering agents it
        // is actually for (2026-07-27).
        let handCreated = manifest.variantConditions.filter {
            $0.artifact.promotion == nil && $0.fromPromotion == nil
                && !$0.artifact.ablatesOnly
        }
        if !handCreated.isEmpty {
            advisories.append(
                "\(handCreated.count) variant condition(s) are hand-created (no "
                    + "sweep-selection provenance) — fine for exploration; "
                    + "evidence-grade studies should promote agents from sweeps")
        }
        // Trained-adapter arms (LoRA readiness §0 amendments 1 + 2). Both are
        // non-blocking BY DESIGN: an exploratory adapter is a legitimate pilot
        // arm, and a missing matched control is a design choice the researcher
        // must be able to make loudly rather than be refused for.
        advisories += adapterVariantAdvisories(manifest)
        // Override-promoted agents carry a birth certificate, but the cell
        // was a human's choice, not the declared rule's — surfaced with the
        // documented reason(s) so the deviation is visible at freeze time.
        let overridden = manifest.variantConditions.filter {
            $0.artifact.promotion?.promotedBy == "manualOverride"
        }
        if !overridden.isEmpty {
            var message =
                "\(overridden.count) variant condition(s) were promoted by "
                + "manual override (declared selection bypassed)"
            let reasons = overridden.compactMap { condition in
                condition.artifact.promotion?.overrideReason.map {
                    "\(condition.name): \($0)"
                }
            }
            if !reasons.isEmpty {
                message += " — " + reasons.joined(separator: "; ")
            }
            advisories.append(message)
        }
        // Confirmation studies attach PLAIN conditions expanded from the
        // perturbation policy, so the variant-condition advisories above
        // never fire for them — the policy's own source-agent provenance is
        // the honest place to look.
        if let policy = manifest.perturbationPolicy, !policy.sourceAgent.promoted {
            advisories.append(
                "confirmation policy source agent '\(policy.sourceAgent.name)' "
                    + "is hand-created (no sweep-selection provenance) — "
                    + "confirmation of an undeclared selection is exploratory")
        }
        // WS7.1: scope-matched validate evidence from an engine other than
        // the one the study will RUN on never satisfies the gate — say so
        // here instead of leaving a bare "no validate run matches" mystery
        // (and catch the legacy unstamped-evidence case whose config.json
        // names a foreign engine). The perspective is the RUN substrate
        // (Mac-authority mode, 2026-07-21): a local freeze of a
        // server-bound study with server-produced evidence is
        // substrate-MATCHED and must not draw a false advisory.
        if !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
            || !manifest.variantConditions.isEmpty,
            let crossSubstrate = crossSubstrateValidationAdvisory(
                for: manifest, perspective: runSubstrate)
        {
            advisories.append(crossSubstrate)
        }
        // F1: panel-script authoring problems that fail SILENTLY at run time
        // — duplicate output labels, and {{outputs.X}} references no earlier
        // turn produces. Advisory rather than a gate: they make prompts
        // quietly wrong, not the study invalid, and a draft is exactly where
        // the researcher is still moving turns around.
        if manifest.studyKind == .multiAgent, let path = manifest.multiAgentScenarioPath,
            let data = try? Data(contentsOf: resolveProjectPath(path)),
            let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        {
            advisories.append(contentsOf: MultiAgentRunner.advisories(scenario))
        }
        // Measurement-side inputs (markers.json / validation.jsonl) that
        // exist on disk but carry no pin — legacy attaches; scoring would
        // read files the firewall does not watch.
        if let unpinned = measurementPinAdvisory(for: manifest) {
            advisories.append(unpinned)
        }
        // Misfiled held-out sets (2026-08-19): the dual-root lookup FOUND a
        // concept's validation.jsonl under the other recipe's root, or under
        // both. Non-blocking — the file is read and pinned either way — but
        // freeze is the last moment the filing can still be corrected.
        advisories += validationLookupAdvisories(for: manifest)
        // An indistinct judge panel (finding 4) is a freeze GATE
        // (judgeValidity); surfaced here too so a DRAFT shows the problem
        // before the freeze attempt refuses.
        if let indistinct = judgePanelIndistinctProblem(manifest) {
            advisories.append(indistinct)
        }
        // `adapter_config.json` pinned? — advisory, deliberately not a gate.
        // Weights pinned and config unpinned means the config can be edited
        // (rank, target modules, scaling, WHICH layers the adapter touches)
        // while the agent's declared identity stays byte-identical — a real
        // hole (external review round 6). It is not a freeze refusal because
        // every agent minted before 2026-08-16 lacks the field and the value
        // is RECOVERABLE from bytes already on disk; refusing would push
        // whole studies onto --force, which stamps them non-citable — worse
        // than a loud advisory for a repairable gap. The enforcement that
        // bites is downstream, where it matters: a J-lens report over an
        // unpinned agent is downgraded off "qualified".
        for variant in manifest.variantConditions {
            for adapter in variant.artifact.adapters
            where (adapter.configHash ?? "").isEmpty {
                advisories.append(
                    "variant '\(variant.name)' adapter '\(adapter.name)' pins its "
                        + "WEIGHTS but not its adapter_config.json (no configHash) — "
                        + "the configuration can change without changing the agent's "
                        + "declared identity. The hash is computable from the adapter "
                        + "directory. Repair: re-save the Agent, then RE-ATTACH it to "
                        + "this study — re-saving the library Agent alone does not "
                        + "repair a condition already attached, which carries its own "
                        + "pinned copy. If the study is FROZEN, duplicate it, attach "
                        + "the repaired Agent, verify, and freeze again. A J-lens "
                        + "readout over this agent cannot claim 'qualified'")
            }
        }
        // Kind-foreign judge fields (field bug 2026-08-07) — stale leftovers
        // from a kind switch saved before the kind-owned write filter.
        // Advisory, never a gate: they name pins that do not exist for the
        // kind, but invalidate nothing.
        if let kindForeign = kindForeignJudgeFieldsAdvisory(manifest) {
            advisories.append(kindForeign)
        }
        // A judged SWEEP whose local judge cannot load inside the chain
        // (finding 1, live incident 2026-07-22) is a freeze GATE
        // (judgeValidity); surfaced here too so a DRAFT shows the problem
        // before the freeze attempt refuses.
        if let pipelineJudgeProblem = localJudgePipelineProblem(manifest) {
            advisories.append(pipelineJudgeProblem)
        }
        // An evaluate stage with such judges ROUTES to the server's
        // post-generation judge fan-out (2026-07-23) — informational,
        // never a gate.
        if let fanoutNote = localJudgeFanoutNote(manifest) {
            advisories.append(fanoutNote)
        }
        // WHERE this panel's judging would happen on THIS host
        // (2026-07-24). Informational, never a gate — deferring to the Mac
        // is a legitimate design. Surfaced because the fork was previously
        // invisible until after a run, and the surprising case (a mixed
        // panel deferring wholesale despite a deliberately placed key) is
        // worth seeing before a study is frozen.
        if let custody = judgingCustodyAdvisory(manifest) {
            advisories.append(custody)
        }
        // A foreign local judge with no revision/dtype pin is a freeze GATE
        // (judgeValidity); surfaced here so a DRAFT shows it before the
        // freeze attempt refuses.
        if let unpinned = unpinnedForeignLocalJudgeProblem(manifest) {
            advisories.append(unpinned)
        }
        // A declared chain with NO gates freezes legally but is worth a
        // loud word (stage 5, server twin): the frozen object then runs
        // every stage to completion with no scientific stop conditions.
        if case .object(let pipeline)? = manifest.pipeline {
            let hasGates: Bool
            if case .object(let gates)? = pipeline["gates"], !gates.isEmpty {
                hasGates = true
            } else {
                hasGates = false
            }
            if !hasGates {
                advisories.append(
                    "pipeline declares no gates — the chain will run every "
                        + "stage to completion with no scientific stop "
                        + "conditions; declare pipeline.gates for "
                        + "evidence-grade chains")
            }
        }
        // Carried non-operative configuration (2026-07-19): preserved by
        // the type picker's never-delete promise, but invisible to this
        // kind's verification, snapshot, and bundle — say so at the
        // moment of freezing instead of letting it read as covered.
        var carried: [String] = []
        if manifest.studyKind == .multiAgent {
            if !manifest.concepts.isEmpty {
                carried.append("\(manifest.concepts.count) concept(s)")
            }
            if !manifest.conditions.isEmpty {
                carried.append("\(manifest.conditions.count) injection condition(s)")
            }
            if !manifest.variantConditions.isEmpty {
                carried.append("\(manifest.variantConditions.count) agent condition(s)")
            }
            if manifest.taskPromptsFile != nil {
                carried.append("a task-prompts pin")
            }
        } else {
            if manifest.multiAgentScenarioPath != nil {
                carried.append("a pinned multi-agent scenario")
            }
            // The finer rule within model-output: a compare-agents study
            // without forward references carries its concept machinery
            // inert (engineer finding 2026-07-19).
            if !conceptMachineryOperative(manifest) {
                if !manifest.concepts.isEmpty {
                    carried.append("\(manifest.concepts.count) concept(s)")
                }
                if !manifest.conditions.isEmpty {
                    carried.append(
                        "\(manifest.conditions.count) injection condition(s)")
                }
            }
        }
        if !carried.isEmpty {
            advisories.append(
                "carries configuration for another study type ("
                    + carried.joined(separator: ", ")
                    + ") — preserved, but NOT verified, snapshotted, or "
                    + "bundled for this study kind; switch the study type "
                    + "(and duplicate) to use it")
        }
        // A forced freeze skipped evidence gates — the manifest says so
        // durably, and every readiness/report surface repeats it.
        if manifest.freezeForced == true {
            let skipped = manifest.forcedGatesSkipped ?? []
            advisories.append(
                "forced freeze — gates skipped: "
                    + (skipped.isEmpty ? "none" : skipped.joined(separator: ", "))
                    + " — non-citable")
        }
        return advisories
    }

    // MARK: The pin-surface enumeration (cleanliness gate + run-bundle packer)

    /// One manifest-declared file input: URL, a human-readable label for
    /// packaging errors, and whether the manifest REQUIRES the file to exist
    /// (a declared pin — a study bundle without it is broken) or merely reads
    /// it when present (the pinned/ snapshot, an unpinned validation file).
    public struct PinnedInputEntry: Sendable {
        public let url: URL
        public let label: String
        public let required: Bool
    }

    /// THE pin-surface enumeration: every file input the manifest declares,
    /// as structured entries. Single source of truth shared by the git
    /// cleanliness gate (`checkGitPinCleanliness`) and the run-bundle packer
    /// (`RunBundlePackager.packageExperiment`) — add a new pin kind HERE and
    /// it is git-gated and packed automatically; a pin kind that bypasses
    /// this function is exactly the silent-bundle-gap bug this function
    /// exists to prevent. Pure path assembly (existence is not checked
    /// here); a directory entry means "the whole directory". Parallel to the
    /// server's `experiment_store.pinned_input_entries`.
    static func pinnedInputEntries(_ manifest: ExperimentManifest) -> [PinnedInputEntry] {
        var entries: [PinnedInputEntry] = []
        func add(_ rel: String?, _ label: String, required: Bool) {
            guard let rel, !rel.isEmpty else { return }
            let url =
                rel.hasPrefix("/")
                ? URL(filePath: rel)
                : VectorCatalog.projectRoot.appending(path: rel)
            entries.append(.init(url: url, label: label, required: required))
        }
        // The pin surface is the OPERATIVE surface for the study kind
        // (2026-07-19, engineer finding): configuration CARRIED from
        // another type — the picker's never-delete promise — is neither
        // verified nor packaged, so a stale hidden concept can never
        // block a multi-agent freeze or bloat its evidence bundle.
        // Kind-agnostic pins (judging, taxonomy, human data, snapshot)
        // stay on both surfaces.
        let modelOutputOperative = manifest.studyKind == .modelOutput
        // Finer rule within model-output (engineer finding 2026-07-19):
        // concept machinery is inert for compare-agents studies without
        // forward references — their carried concepts/conditions/sweep
        // inputs are neither git-gated, snapshotted, nor bundled.
        let machinery = conceptMachineryOperative(manifest)
        for concept in machinery ? manifest.concepts : [] {
            let paired = concept.options.method.isPaired
            // Paired concepts pin their whole stimulus directory; grand-mean
            // concepts keep stimuli under prompts/emotions/ but may still
            // carry a markers.json in prompts/concepts/<name>/.
            entries.append(
                .init(
                    url: VectorCatalog.conceptsDirectory.appending(component: concept.name),
                    label: "concept '\(concept.name)' stimulus directory",
                    required: paired))
            if !paired {
                entries.append(
                    .init(
                        url: VectorCatalog.emotionsDirectory
                            .appending(components: concept.name, "stories.jsonl"),
                        label: "concept '\(concept.name)' stories.jsonl",
                        required: true))
            }
            if concept.options.method == .designatedReference,
                let pin = concept.designatedReference
            {
                entries.append(
                    .init(
                        url: VectorCatalog.emotionsDirectory
                            .appending(components: pin.name, "stories.jsonl"),
                        label: "concept '\(concept.name)' designated reference "
                            + "'\(pin.name)' stories.jsonl",
                        required: true))
            }
            // Measurement-side pin: a non-null validationHash names exact
            // bytes (beside the stories for grand-mean concepts, inside the
            // stimulus directory for paired ones).
            let validationURL =
                paired
                ? VectorCatalog.conceptsDirectory
                    .appending(components: concept.name, "validation.jsonl")
                : VectorCatalog.emotionsDirectory
                    .appending(components: concept.name, "validation.jsonl")
            entries.append(
                .init(
                    url: validationURL,
                    label: "concept '\(concept.name)' validation.jsonl",
                    required: concept.validationHash != nil))
        }
        for concept in machinery
            ? (manifest.grandMeanCorpus?.concepts ?? []) : []
        {
            entries.append(
                .init(
                    url: VectorCatalog.emotionsDirectory
                        .appending(components: concept, "stories.jsonl"),
                    label: "grand-mean corpus member '\(concept)' stories.jsonl",
                    required: true))
        }
        if modelOutputOperative {
            add(manifest.taskPromptsFile, "task prompts", required: true)
            add(manifest.capabilityBatteryFile, "capability battery", required: true)
            if manifest.numericParser != nil {
                add(ParserRegistry.registryFile, "parser registry", required: true)
            }
            if machinery, manifest.neutralCorpusHash != nil {
                add("prompts/neutral/corpus.jsonl", "neutral corpus", required: true)
            }
            for condition in machinery ? manifest.conditions : [] {
                add(
                    condition.neutralPCBasisPath,
                    "condition '\(condition.name)' neutral-PC basis", required: true)
            }
            for reader in manifest.readerRefs ?? [] {
                add(reader.path, "reader '\(reader.concept)' artifact", required: true)
            }
            // Declared validation controls are pinned INPUTS (C2): their
            // stimuli must travel with the bundle and be git-gated at
            // freeze, exactly like a study concept's.
            for control in machinery ? (manifest.validationControls ?? []) : [] {
                add(
                    "prompts/concepts/\(control.concept)",
                    "validation control '\(control.concept)' stimulus directory",
                    required: true)
            }
            for variant in manifest.variantConditions
            where variant.fromPromotion == nil {
                // Forward-referenced conditions have no artifact yet — the
                // server resolves them at run time from its own runs tree.
                add(variant.artifactPath, "variant '\(variant.name)' artifact", required: true)
                // ...and everything that artifact POINTS AT. Packing the
                // agent's JSON alone shipped a bundle whose injections named
                // vectors that did not exist on the far side: the study
                // failed on the cluster, after the queue wait and the model
                // load, with a missing-artifact error that read as a server
                // problem. These are `required`, so an unresolvable
                // dependency now refuses packaging here instead (B3,
                // fail closed).
                for dependency in variantArtifactDependencies(variant) {
                    add(dependency.path, dependency.label, required: true)
                }
                // A trained adapter's DATASET MANIFEST is a pinned input like
                // any other (LoRA readiness §0 amendment 1): verify()
                // re-hashes it, so it must be git-gated, snapshotted, and
                // packed with the study. Declared-only — legacy adapter
                // variants gain no entry.
                add(
                    variant.trainingProvenance?.datasetManifestPath,
                    "variant '\(variant.name)' training dataset manifest",
                    required: true)
            }
            // Sweep inputs are manifest-declared data ON the verify()
            // hash-pin surface (firewall closure 2026-07-20:
            // sweep.devPromptsHash + sweep.batteryHash, pinned at freeze):
            // a declared sweep cannot run without them.
            if machinery, let sweep = manifest.sweep {
                add(sweep.devPromptsFile, "sweep dev prompts", required: true)
                add(sweep.batteryFile, "sweep battery", required: true)
                // Choice instruments are inputs only when the objective
                // READS them (logprobShift) — a stale path under another
                // metric is inert, and a required-but-unread file must not
                // refuse packaging. The pin enumeration applies the same
                // rule (`sweepChoicePinEntries`).
                if sweep.selection?.objective?.metric == "logprobShift" {
                    add(
                        sweep.selection?.objective?.choicePromptsFile,
                        "sweep choice prompts", required: true)
                    // Per-concept instruments (choicePromptsFiles,
                    // 2026-08-02): every map value is a declared, required
                    // input — a bundle without one cannot run the sweep it
                    // declares.
                    for (concept, rel) in (sweep.selection?.objective?
                        .choicePromptsFiles ?? [:])
                        .sorted(by: { $0.key < $1.key })
                    {
                        add(rel, "sweep choice prompts '\(concept)'",
                            required: true)
                    }
                }
            }
        } else {
            add(manifest.multiAgentScenarioPath, "multi-agent scenario", required: true)
            // ...and every AGENT artifact the panel script names, plus what
            // those point at. A scenario carries its seats by path INSIDE the
            // JSON, so enumerating the script alone shipped a bundle whose
            // seats named variants absent on the far side — the same failure
            // the variantConditions branch above was already fixed for,
            // reproduced here because this branch never looked inside the
            // script. Required, so an unresolvable seat refuses PACKAGING
            // rather than dying on the GPU after the queue wait.
            for (path, label) in panelAgentDependencies(manifest.multiAgentScenarioPath) {
                add(path, label, required: true)
            }
        }
        add(manifest.judgeRubricFile, "judge rubric", required: true)
        add(manifest.reasoningStyleTaxonomyPath, "reasoning-style taxonomy", required: true)
        add(manifest.humanBaseline?.path, "human baseline", required: true)
        add(manifest.humanValidation?.path, "human validation", required: true)
        entries.append(
            .init(
                url: directory.appending(components: manifest.name, "pinned"),
                label: "pinned-input snapshot", required: false))
        return entries
    }

    /// Concepts sitting in the workspace that the study does NOT declare as
    /// controls — the set the old ambient rule silently folded into the
    /// cosine matrix.
    ///
    /// Removing an implicit behaviour must not be invisible: a researcher who
    /// relied on those cosines needs to be told they are gone and how to keep
    /// them. Advisory only — an undeclared concept on disk is not an error,
    /// it is simply not evidence.
    static func undeclaredControlAdvisories(
        _ manifest: ExperimentManifest, availableConcepts: [String]
    ) -> [String] {
        let declared = Set((manifest.validationControls ?? []).map(\.concept))
        let pinned = Set(manifest.concepts.map(\.name))
        let undeclared = availableConcepts
            .filter { !declared.contains($0) && !pinned.contains($0) }
            .sorted()
        guard !undeclared.isEmpty else { return [] }
        let names = undeclared.joined(separator: ", ")
        return [
            "note: \(undeclared.count) concept(s) in this workspace are not "
                + "declared as validation controls and are NOT in the cosine "
                + "matrix: \(names). Discriminant evidence covers declared "
                + "inputs only — add them to validationControls (each with its "
                + "own stimulus hash and extraction options) to measure "
                + "against them."
        ]
    }

    /// The files a variant condition's agent artifact REFERENCES: each
    /// injection's vector pair, each adapter directory, and the neutral-PC
    /// basis it projects against.
    ///
    /// Unlike the rest of `pinnedInputEntries` this reads a file — it has
    /// to, because the dependency list exists only inside the artifact. A
    /// artifact that cannot be read or decoded returns nothing: the artifact
    /// entry itself is `required`, so packaging already refuses, and
    /// guessing dependency paths from an unparsed file would be worse than
    /// the honest refusal.
    static func variantArtifactDependencies(
        _ variant: ExperimentManifest.VariantCondition
    ) -> [(path: String, label: String)] {
        variantArtifactDependencies(
            artifactPath: variant.artifactPath, label: "variant '\(variant.name)'")
    }

    /// Path-addressed form, so a PANEL SEAT — which names its artifact inside
    /// the scenario rather than as a variant condition — walks the identical
    /// dependency surface instead of a parallel copy of it.
    static func variantArtifactDependencies(
        artifactPath: String, label: String
    ) -> [(path: String, label: String)] {
        guard !artifactPath.isEmpty,
            let data = try? Data(contentsOf: resolveProjectPath(artifactPath)),
            let artifact = try? JSONDecoder().decode(
                ModelVariantArtifact.self, from: data)
        else { return [] }

        var dependencies: [(path: String, label: String)] = []
        for injection in artifact.injections {
            // `vectorArtifactID` is an extension-less locator; the store
            // opens the two files beside it.
            let reference = ArtifactIdentity.canonical(injection.vectorArtifactID)
            for suffix in ["safetensors", "json"] {
                dependencies.append(
                    (
                        "\(reference).\(suffix)",
                        "\(label) vector "
                            + "'\(injection.concept)' (.\(suffix))"
                    ))
            }
        }
        for adapter in artifact.adapters {
            dependencies.append(
                (
                    ArtifactIdentity.canonical(adapter.adapterDirectory),
                    "\(label) adapter"
                ))
        }
        if let basis = artifact.neutralPCBasisPath, !basis.isEmpty {
            dependencies.append(
                (
                    ArtifactIdentity.canonical(basis),
                    "\(label) neutral-PC basis"
                ))
        }
        return dependencies
    }

    /// Agent artifacts a panel script names, and what those artifacts point
    /// at. Reads files, because the dependency list exists only inside them:
    /// the scenario names its seats' variants, each variant names its
    /// vectors, adapters and neutral basis. An unreadable scenario
    /// contributes nothing — its own entry is already required, so the
    /// refusal lands there instead of on a guess. Twin of the server's
    /// `_panel_agent_dependencies`.
    static func panelAgentDependencies(_ scenarioPath: String?) -> [(String, String)] {
        guard let scenarioPath, !scenarioPath.isEmpty,
            let data = try? Data(contentsOf: resolveProjectPath(scenarioPath)),
            let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        else { return [] }

        var out: [(String, String)] = []
        var seen = Set<String>()
        for agent in scenario.agents {
            guard let artifact = agent.variantArtifactPath, !artifact.isEmpty,
                seen.insert(artifact).inserted
            else { continue }
            out.append((artifact, "panel seat '\(agent.name)' agent artifact"))
            // A seat's artifact has the same shape and dependency surface as
            // a variant condition's, so it walks through the same code.
            out.append(
                contentsOf: variantArtifactDependencies(
                    artifactPath: artifact, label: "panel seat '\(agent.name)'"))
        }
        return out
    }


    // MARK: Git pin cleanliness (freeze-time reproducibility gate)

    /// Absolute URLs of every input the manifest declares — what the stamped
    /// gitCommit must actually contain for the frozen manifest's hashes to be
    /// resolvable to bytes forever. Existence-filtered, deduplicated view
    /// over `pinnedInputEntries`; shared by the cleanliness gate and its tests.
    static func pinnedInputPaths(_ manifest: ExperimentManifest) -> [URL] {
        var seen = Set<String>()
        var urls: [URL] = []
        for entry in pinnedInputEntries(manifest) {
            let key = entry.url.standardizedFileURL.path
            guard !seen.contains(key),
                FileManager.default.fileExists(atPath: entry.url.path)
            else { continue }
            seen.insert(key)
            urls.append(entry.url)
        }
        return urls
    }

    /// Paths named by `git status --porcelain` output — anything listed is
    /// modified or untracked. Pure; unit-testable without git.
    static func offendingPorcelainPaths(_ porcelain: String) -> [String] {
        porcelain.split(separator: "\n")
            .map { String($0) }
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { line in
                String(line.dropFirst(min(3, line.count)))
                    .trimmingCharacters(in: .whitespaces)
            }
            .sorted()
    }

    /// Every pinned input must be COMMITTED — tracked and unmodified — or the
    /// gitCommit freeze stamps cannot resolve the pinned hashes to bytes (the
    /// "untracked inputs with completed runs" reproducibility hole, caught at
    /// the only moment it matters). Skipped when the workspace is not a git
    /// work tree or under a test root override; `freeze --force` skips loudly.
    private static func checkGitPinCleanliness(_ manifest: ExperimentManifest) throws {
        guard rootOverride == nil else { return }
        let root = VectorCatalog.projectRoot
        guard runGit(["rev-parse", "--is-inside-work-tree"], in: root) == "true" else {
            return
        }
        let pinned = pinnedInputPaths(manifest)
        guard !pinned.isEmpty else { return }
        guard
            let porcelain = runGit(
                ["status", "--porcelain", "--"] + pinned.map(\.path), in: root)
        else { return }
        let offending = offendingPorcelainPaths(porcelain)
        guard !offending.isEmpty else { return }
        let shown = offending.prefix(8).joined(separator: ", ")
            + (offending.count > 8 ? " …" : "")
        throw ExperimentError(
            reason: "cannot freeze '\(manifest.name)': pinned input(s) not "
                + "committed to git — \(shown) — the stamped gitCommit must "
                + "contain the bytes the manifest's hashes name (commit them, "
                + "or freeze --force)")
    }

    /// Freeze auto-commits only a MANAGED WORKSPACE: never under the test
    /// root override, never the legacy code-checkout fallback (auto-`git add
    /// -A` on the developer's repo would commit unrelated work — dev behavior
    /// stays byte-identical), only when the root is a git work tree
    /// (elsewhere the pinned/ snapshot is the reproducibility floor), and —
    /// since WP1 — never when the workspace is a SUBDIRECTORY of a larger
    /// repository (`nestedWorkspaceRepositoryRoot`, the Swift twin of the
    /// server's `_auto_commit_workspace` safety-skip).
    static func freezeAutoCommitIsEnabled() -> Bool {
        guard rootOverride == nil else { return false }
        let root = VectorCatalog.projectRoot.standardizedFileURL
        guard root.path != VectorCatalog.bundledSeedRoot.standardizedFileURL.path else {
            return false
        }
        guard runGit(["rev-parse", "--is-inside-work-tree"], in: root) == "true"
        else { return false }
        return nestedWorkspaceRepositoryRoot(of: root) == nil
    }

    /// The enclosing git work-tree root when `root` is a SUBDIRECTORY of a
    /// larger repository; nil when `root` IS its own work-tree root, or is
    /// no work tree at all.
    ///
    /// The hazard this closes (WP1, the server's rule since 2026-07-13):
    /// people hand-create workspaces inside bigger repos. Swift's freeze
    /// auto-commit scoped `git add -A .` to the workspace directory, which
    /// avoids sweeping up the parent repo's unrelated work — but it still
    /// lands a commit in the PARENT repo's history as a side effect of
    /// freezing a study. That is someone else's repository. Skip, say so
    /// loudly, and let `checkGitPinCleanliness` refuse dirty pins instead:
    /// the researcher commits the pinned inputs themselves, in the repo
    /// they own.
    static func nestedWorkspaceRepositoryRoot(of root: URL) -> URL? {
        guard let top = runGit(["rev-parse", "--show-toplevel"], in: root),
            !top.isEmpty
        else { return nil }
        let topURL = URL(filePath: top).resolvingSymlinksInPath().standardizedFileURL
        let base = root.resolvingSymlinksInPath().standardizedFileURL
        return topURL.path == base.path ? nil : topURL
    }

    /// The advisory printed when the skip above fires. Pure so it can be
    /// asserted; wording mirrors the server's message family.
    static func nestedWorkspaceAutoCommitAdvisory(
        freezing name: String, workspace: URL, repository: URL
    ) -> String {
        "⚠︎ freeze '\(name)': auto-commit skipped — workspace '\(workspace.path)' "
            + "is a subdirectory of the git repository at '\(repository.path)', "
            + "and committing that repo as a side effect would be hostile. "
            + "Commit the pinned inputs yourself (or freeze --force)."
    }

    // MARK: - The freeze-gate table (one table, both branches)

    /// How one gate would decline, in the two renderings freeze has always
    /// used: the full refusal prose, and the bare reason the `--force`
    /// warning prints. `repairAction` is the machine-readable remedy (WP0
    /// step 2); `underlying` is set only when the check threw something
    /// other than an `ExperimentError`, so the refusal path can rethrow that
    /// error verbatim rather than re-wrapping it.
    struct FreezeGateOutcome {
        let gate: FreezeGate
        let refusal: String
        let forced: String
        let repairAction: String
        var underlying: Error?
    }

    /// One freeze gate: its closed-vocabulary id, and the check that reports
    /// how it would decline (nil = the gate passes).
    struct FreezeGateEntry {
        let gate: FreezeGate
        let evaluate: (ExperimentManifest) -> FreezeGateOutcome?
    }

    /// Wraps a throwing gate check as a table entry. The refusal keeps the
    /// check's own prose verbatim; the forced rendering is the same string
    /// with freeze's preamble stripped — exactly what `forceEvaluate` did
    /// before this table existed.
    private static func freezeGateEntry(
        _ gate: FreezeGate, name: String, repairAction: String,
        _ check: @escaping (ExperimentManifest) throws -> Void
    ) -> FreezeGateEntry {
        FreezeGateEntry(gate: gate) { manifest in
            do {
                try check(manifest)
                return nil
            } catch let error as ExperimentError {
                return FreezeGateOutcome(
                    gate: gate, refusal: error.reason,
                    forced: plainGateReason(error.reason, experimentName: name),
                    repairAction: repairAction)
            } catch {
                return FreezeGateOutcome(
                    gate: gate, refusal: "\(error)", forced: "\(error)",
                    repairAction: repairAction, underlying: error)
            }
        }
    }

    /// The git-cleanliness gate, built once because it is evaluated at one of
    /// TWO points: before the freeze transaction where freeze does not
    /// auto-commit (the legacy code-checkout root), and after the
    /// auto-commit where it does — dirty pins are self-healing there, so the
    /// gate can only speak once the commit has been attempted.
    private static func freezeGitCleanGate(name: String) -> FreezeGateEntry {
        freezeGateEntry(
            .gitClean, name: name,
            repairAction: "commit the pinned inputs to the workspace git repo, "
                + "or freeze --force"
        ) { try checkGitPinCleanliness($0) }
    }

    /// THE freeze-gate table: every force-skippable gate, in freeze's
    /// historical refusal order, each paired with its closed-vocabulary id.
    ///
    /// Hoisted out of the `--force` branch (WP0 step 2) so ONE table drives
    /// both branches — refusal names the gate, force stamps it — instead of
    /// the id existing only where `forcedGatesSkipped` was written. Python
    /// twin: `_evaluate_freeze_gates`, which has evaluated every gate on
    /// both paths since 2026-07-13 (its own historical order differs where
    /// `judgeValidity` sits; the VOCABULARY is what the engines share, not
    /// the evaluation order).
    ///
    /// Read-only: no gate mutates the manifest, so evaluating all of them on
    /// the refusal path — where freeze used to short-circuit at the first —
    /// changes nothing but the completeness of `FreezeRefusal.gates`.
    ///
    /// Never in this table, deliberately: `verify()` pin violations, missing
    /// sweep-input pins, and the frozen-status guard. Those are pin-surface
    /// integrity, the never-skippable class — `--force` does not reach them
    /// and they own no gate id.
    static func freezeGateTable(
        name: String, autoCommit: Bool,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) -> [FreezeGateEntry] {
        var table: [FreezeGateEntry] = []
        table.append(
            FreezeGateEntry(gate: .revision) { manifest in
                guard manifest.modelRevision == nil else { return nil }
                let forced = "model revision is not pinned and \(manifest.modelID) is not in "
                    + "the local HF cache"
                let repair = "create with --revision, load the model once, or freeze --force"
                return FreezeGateOutcome(
                    gate: .revision,
                    refusal: "cannot freeze '\(name)': \(forced) — \(repair)",
                    forced: forced, repairAction: repair)
            })
        table.append(
            FreezeGateEntry(gate: .revision) { manifest in
                guard let symbolic = symbolicRevisionProblem(manifest) else { return nil }
                return FreezeGateOutcome(
                    gate: .revision, refusal: "cannot freeze '\(name)': \(symbolic)",
                    forced: symbolic,
                    repairAction: "pin the immutable commit the symbolic revision "
                        + "resolves to, or freeze --force")
            })
        // "measurementPins": measurement-pin DRIFT stays an unskippable
        // verify() violation and unpinned legacy inputs stay advisory, so
        // this gate covers only pins that are present and invalid. The study
        // dtype is the first (2026-07-24) — an input that determines what
        // gets measured rather than what gets loaded.
        table.append(
            FreezeGateEntry(gate: .measurementPins) { manifest in
                guard let badDtype = unloadableStudyDtypeProblem(manifest) else { return nil }
                return FreezeGateOutcome(
                    gate: .measurementPins, refusal: "cannot freeze '\(name)': \(badDtype)",
                    forced: badDtype,
                    repairAction: "repoint the study dtype at a loadable value, "
                        + "or freeze --force")
            })
        // Model-output surfaces: gated on the study kind that USES them. A
        // panel carrying these from a kind switch executes none of them, and
        // refusing its freeze over them contradicts the advisory this same
        // type emits about carried configuration. The predicate is asked per
        // evaluation, not baked into the table, so the table's SHAPE is the
        // vocabulary and only its answers depend on the manifest.
        table.append(
            FreezeGateEntry(gate: .validateEvidence) { manifest in
                guard modelOutputSurfacesOperative(manifest) else { return nil }
                let usesLegacyConceptVectors =
                    !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
                guard usesLegacyConceptVectors,
                    !optvecExemptFromValidateGate(manifest)
                else { return nil }
                guard
                    validationEvidence(for: manifest, runSubstrate: runSubstrate) != nil
                else {
                    let forced = "no validate run matches its exact pins (model+revision, "
                        + "concepts, neutral corpus) on the run substrate " + runSubstrate
                    // The engine named is the one whose evidence this gate
                    // reads — `steerlab-cli` cannot satisfy a server-substrate
                    // gate (gate-5 dry run #2, P2). Sentence structure is
                    // unchanged; only the binary moves.
                    let repair = "Run '\(validateCLI(forRunSubstrate: runSubstrate)) "
                        + "experiment validate \(name)' first, or "
                        + "freeze --force to record an unvalidated experiment"
                    return FreezeGateOutcome(
                        gate: .validateEvidence,
                        refusal: "cannot freeze '\(name)': \(forced). \(repair)",
                        forced: forced, repairAction: repair)
                }
                // Evidence EXISTS but probed nothing: same gate id, a remedy
                // naming the missing files (2026-08-17).
                guard
                    let vacuous = vacuousValidationEvidenceProblem(
                        for: manifest, runSubstrate: runSubstrate)
                else { return nil }
                // P5 (dry run #1): the gate's own repair FAILED as given.
                // Authoring the named validation.jsonl makes it appear after
                // an attach that pinned it absent, which is a verify()
                // violation — so the very next `validate` refuses. The
                // machine repair now carries the re-attach that re-pins it;
                // the composed PROSE is unchanged (it is the cross-engine
                // refusal string, asserted whole by VacuousValidationTests).
                let vacuousConcepts = vacuousValidationEvidence(
                    for: manifest, runSubstrate: runSubstrate)
                let repair = vacuousValidationMachineRepair(
                    for: manifest, vacuousConcepts: vacuousConcepts,
                    runSubstrate: runSubstrate) ?? vacuous
                return FreezeGateOutcome(
                    gate: .validateEvidence, refusal: "cannot freeze '\(name)': \(vacuous)",
                    forced: vacuous, repairAction: repair)
            })
        let variantValidity = freezeGateEntry(
            .variantValidity, name: name,
            repairAction: "re-save the variant with hashed adapter weights and re-attach "
                + "it so freeze can pin the dataset manifest, or freeze --force"
        ) { try checkVariantValidity($0) }
        table.append(
            FreezeGateEntry(gate: .variantValidity) { manifest in
                guard modelOutputSurfacesOperative(manifest) else { return nil }
                return variantValidity.evaluate(manifest)
            })
        let batteryEvidence = freezeGateEntry(
            .batteryEvidence, name: name,
            // Same class as `validateEvidence` (dry run #2, P2): this gate
            // reads battery evidence stamped with the RUN substrate, so under
            // `--run-substrate server` a `steerlab-cli experiment validate`
            // can never satisfy it. The sentence is unchanged; only the
            // binary moves. Server twin: `_freeze_gate_repair`, which names
            // `steerlab-server` unconditionally because THAT engine's
            // `_check_battery_evidence` has no run-substrate seam — it accepts
            // only `python-hf-transformers` evidence.
            repairAction: "re-run '\(validateCLI(forRunSubstrate: runSubstrate)) "
                + "experiment validate \(name)' (each variant "
                + "condition runs the pinned battery), or freeze --force"
        ) { try checkVariantBatteryEvidence($0, runSubstrate: runSubstrate) }
        table.append(
            FreezeGateEntry(gate: .batteryEvidence) { manifest in
                guard modelOutputSurfacesOperative(manifest) else { return nil }
                return batteryEvidence.evaluate(manifest)
            })
        // Judge validity is NOT model-output-only: a panel's turns are
        // flattened into generations and judged like any other output.
        table.append(
            freezeGateEntry(
                .judgeValidity, name: name,
                repairAction: "steerlab-cli experiment pin-rubric \(name) "
                    + "\(JudgeRubricStore.defaultRubricFile) --judges "
                    + "<name>:<kind>,<name>:<kind> — a rubric FILE and at least "
                    + "2 distinct judges the pipeline can actually run, or freeze --force"
            ) { try checkJudgeEvaluationValidity($0) })
        if !autoCommit {
            table.append(freezeGitCleanGate(name: name))
        }
        return table
    }

    /// Turns the table's failures into the typed refusal freeze throws:
    /// `reason` is the FIRST failure's prose (byte-identical to what freeze
    /// threw before the table existed), `gate` is that failure's id, and
    /// `gates` is every failure in vocabulary order — the same order and the
    /// same ids the `forcedGatesSkipped` stamp uses.
    private static func freezeRefusal(_ failures: [FreezeGateOutcome]) -> Error? {
        guard let first = failures.first else { return nil }
        if let underlying = first.underlying { return underlying }
        let failed = Set(failures.map(\.gate))
        return ExperimentError(
            refusal: FreezeRefusal(
                gate: first.gate,
                gates: FreezeGate.allCases.filter(failed.contains),
                reason: first.refusal, repairAction: first.repairAction))
    }

    /// `git add -A . && git commit -m "freeze <name>"` on the workspace root.
    /// Best-effort: a clean tree is a no-op, and any git failure is left for
    /// `checkGitPinCleanliness` to report loudly — never silently.
    ///
    /// Reached only for a workspace that IS its own git work-tree root:
    /// `freezeAutoCommitIsEnabled` now safety-skips a workspace nested
    /// inside a larger repository, so the `.` pathspec below is no longer
    /// the only thing standing between a freeze and a commit in someone
    /// else's history.
    private static func autoCommitWorkspace(freezing name: String) {
        let root = VectorCatalog.projectRoot
        guard let porcelain = runGit(["status", "--porcelain"], in: root),
            !porcelain.isEmpty
        else { return }
        _ = runGit(["add", "-A", "."], in: root)
        _ = runGit(["commit", "-m", "freeze \(name)"], in: root)
    }

    private static func runGit(_ arguments: [String], in root: URL) -> String? {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = ["-C", root.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// One-way: verifies all pinned inputs, requires a model revision and a
    /// matching `validate` run (the circularity firewall is mechanical, not
    /// remembered — `force` skips the evidence gates but never pin
    /// violations, and since 2026-07-13 a forced freeze is LOUD and STAMPED:
    /// every skipped-and-failing gate is logged and recorded in the manifest
    /// as `freezeForced` + `forcedGatesSkipped`), stamps the content hash
    /// and the current git commit, and makes the manifest read-only.
    @discardableResult
    /// `runSubstrate` (Mac-authority mode, 2026-07-21): the engine the
    /// study's measured runs will execute on — validate/battery evidence
    /// counts on THAT substrate, not on the substrate performing the
    /// freeze. Default keeps the historical freeze-where-you-run rule; the
    /// app passes `WorkspaceScoping.serverSubstrate` when freezing locally
    /// for a server-bound study (unpaired server Compute), and the CLI
    /// exposes `--run-substrate server`. Every other gate — pins, revision,
    /// judges, variants, git cleanliness — is substrate-independent and
    /// unchanged.
    public static func freeze(
        name: String, force: Bool = false,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) throws -> ExperimentManifest {
        var manifest = try load(name: name)
        guard manifest.status == .draft else {
            // Typed since gate-5 dry run #2 (P3): `freeze` was the last
            // manifest-WRITING verb whose immutability refusal arrived as an
            // untyped `verbFailed`/70 — an operational failure, to an agent —
            // while every other writer already answered `statusImmutable`/65
            // with a runnable duplicate-to-iterate repair. Prose unchanged.
            throw ExperimentError.refusing(
                .statusImmutable,
                "'\(name)' is already \(manifest.status.rawValue)",
                repair: duplicateToIterateRepair(name))
        }
        // Pin the revision the local cache would actually run, if not set.
        if manifest.modelRevision == nil {
            manifest.modelRevision = SteeredContainerLoader.cachedRevision(
                for: manifest.modelID)
        }
        // Variant studies pin the capability battery (default preset battery
        // if none chosen) so the frozen manifest names the exact battery its
        // validation evidence scored.
        // Every model-output-only PIN is scoped the same way the gates are: a
        // panel carrying agents or concepts across a kind switch must not have
        // a battery, marker rubric, training provenance or parser registry
        // stamped into its frozen manifest for configuration it never
        // executes (external review round 14). The predicate governs the whole
        // freeze transaction, not just gate evaluation.
        let modelOutputSurfaces = modelOutputSurfacesOperative(manifest)
        if modelOutputSurfaces, !manifest.variantConditions.isEmpty,
            manifest.capabilityBatteryFile == nil
        {
            pinCapabilityBattery(into: &manifest)
        }
        // Local-judge revision pin (cross-engine key "judges[].revision",
        // 2026-07-23): a local judge that resolves to the STUDY model
        // inherits the study's pinned revision when its own is blank — the
        // judging path then loads exactly the pinned bytes. Different-model
        // local judges keep a blank revision (no study pin to inherit); a
        // declared revision is never overwritten.
        if let studyRevision = manifest.modelRevision, var judges = manifest.judges {
            for index in judges.indices
            where judges[index].kind.trimmingCharacters(in: .whitespacesAndNewlines)
                == "local" && (judges[index].revision ?? "").isEmpty
            {
                let declared = (judges[index].model ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if declared.isEmpty || declared == manifest.modelID {
                    judges[index].revision = studyRevision
                }
            }
            manifest.judges = judges
        }
        // Measurement-side markers pin: the frozen manifest names the exact
        // scoring rubrics its runs will read. Only pinned when absent — a
        // pinned-but-drifted markersHash must surface as a verify violation
        // below, never be silently re-pinned.
        if modelOutputSurfaces, manifest.markersHash == nil {
            manifest.markersHash = liveMarkersHash(manifest)
        }
        // Adapter training-provenance pin (cross-engine key
        // "variantConditions[].trainingProvenance", LoRA readiness §0
        // amendment 1): freeze is the pin moment for a trained adapter's
        // DATASET — stamped from the adapter's own v2 sidecar, BEFORE the
        // freeze hash, so the frozen manifest is self-describing and later
        // drift in the training data is a verify violation. Usually a no-op
        // here: the sidecar is a server artifact (see the asymmetry note on
        // `pinTrainingProvenance`).
        if modelOutputSurfaces {
            pinTrainingProvenance(into: &manifest)
        }
        // Numeric-parser registry pin (cross-engine key "parserRegistryHash"):
        // freeze is the pin moment for the registry the named parser reads —
        // stamped only when absent, BEFORE the freeze hash, so later drift is
        // a verify violation, never a silent re-pin. A study that names no
        // parser gets no new key (legacy bytes unchanged).
        if modelOutputSurfaces, manifest.numericParser != nil,
            manifest.parserRegistryHash == nil
        {
            manifest.parserRegistryHash = ParserRegistry.liveHash()
        }
        // Sweep-input pins (cross-engine keys "sweep.devPromptsHash" +
        // "sweep.batteryHash", firewall closure 2026-07-20): freeze is the
        // pin moment for the files the sweep SELECTS on — stamped only when
        // absent and the file exists, BEFORE the freeze hash, so later
        // drift is a verify violation, never a silent re-pin. Only an
        // OPERATIVE declared sweep gains keys (legacy bytes unchanged
        // elsewhere). The ex-post provenance stamp
        // (selection.devPromptsHash) is unchanged — sweep start refuses on
        // a pin mismatch, so pin and provenance can only agree.
        if conceptMachineryOperative(manifest) {
            pinSweepInputs(into: &manifest)
            // A sweep input that could not be pinned (missing file) REFUSES
            // the freeze, force included: this is pin-surface integrity —
            // the never-skippable class, like verify() itself — not an
            // evidence gate. A forced freeze with an absent pin would leave
            // the sweep-start legacy-unpinned fallback open to whatever
            // bytes later appear at the path, and no `forcedGatesSkipped`
            // stamp can neutralize data accepted silently at run time.
            // Carried-inert sweeps (this whole branch) neither pin nor
            // block, exactly as before.
            let missingSweepInputs = missingSweepInputRefusals(manifest)
            guard missingSweepInputs.isEmpty else {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)':\n  - "
                        + missingSweepInputs.joined(separator: "\n  - "))
            }
        }
        let violations = verify(manifest)
        guard violations.isEmpty else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)':\n  - "
                    + violations.joined(separator: "\n  - "))
        }
        let autoCommit = freezeAutoCommitIsEnabled()
        // The nested-workspace safety-skip is LOUD: silently not committing
        // would look identical to "there was nothing to commit", and the
        // cleanliness gate's refusal moments later would read as a mystery.
        if !autoCommit, rootOverride == nil,
            let enclosing = nestedWorkspaceRepositoryRoot(
                of: VectorCatalog.projectRoot.standardizedFileURL)
        {
            print(
                nestedWorkspaceAutoCommitAdvisory(
                    freezing: name,
                    workspace: VectorCatalog.projectRoot.standardizedFileURL,
                    repository: enclosing))
        }
        // ONE gate table drives both branches (WP0 step 2). Under --force
        // every gate is still EVALUATED — each failure is logged loudly and
        // recorded in the frozen manifest ("freezeForced" +
        // "forcedGatesSkipped"), so a forced freeze can never pass as a
        // clean one. Without force the FIRST failure refuses, with its prose
        // unchanged, now carrying the gate id it always computed and dropped.
        var forcedGateFailures: [String] = []
        func recordForcedFailure(_ id: String, _ reason: String) {
            forcedGateFailures.append(id)
            print("⚠︎ freeze --force: skipping gate '\(id)' which would have failed — \(reason)")
        }
        let gateFailures = freezeGateTable(
            name: name, autoCommit: autoCommit, runSubstrate: runSubstrate
        ).compactMap { $0.evaluate(manifest) }
        if !force {
            if let refusal = freezeRefusal(gateFailures) { throw refusal }
        } else {
            for failure in gateFailures {
                recordForcedFailure(failure.gate.rawValue, failure.forced)
            }
        }

        // Frozen studies must be self-contained: pinned scenario/variant
        // inputs that live under gitignored runs/ are copied into the
        // experiment directory (byte-identical, hash-checked by the
        // re-verify) so the git-tracked manifest never points at an
        // unversioned file. Happens BEFORE the freeze hash is stamped
        // because it rewrites the pinned paths.
        try pinExternalInputs(into: &manifest)
        // Full pinned snapshot: every pinned input, byte-copied into
        // experiments/<name>/pinned/ — the no-git reproducibility floor.
        try snapshotPinnedInputs(for: manifest)
        let postPinViolations = verify(manifest)
        guard postPinViolations.isEmpty else {
            throw ExperimentError(
                reason: "cannot freeze (after pinning inputs):\n  - "
                    + postPinViolations.joined(separator: "\n  - "))
        }

        // Freeze = commit + stamp as one gesture (shipped-app semantics):
        // the workspace is committed BEFORE the gitCommit stamp is read, so
        // the stamped commit contains the pinned bytes and the snapshot. If
        // the workspace is not a git tree this is a silent skip (the pinned/
        // snapshot is the floor); if the commit fails, the cleanliness gate
        // below fires loudly.
        if autoCommit {
            autoCommitWorkspace(freezing: manifest.name)
            // The same table entry as the pre-transaction gate, evaluated at
            // the only moment it can speak here — after the auto-commit.
            if let failure = freezeGitCleanGate(name: name).evaluate(manifest) {
                if !force {
                    if let refusal = freezeRefusal([failure]) { throw refusal }
                } else {
                    recordForcedFailure(failure.gate.rawValue, failure.forced)
                }
            }
        }

        manifest.appVersion = SteerLabVersion.current
        if force {
            // Ordered by the fixed vocabulary so the stamp is stable across
            // engines and re-freezes.
            manifest.freezeForced = true
            manifest.forcedGatesSkipped =
                FreezeGate.vocabulary.filter(forcedGateFailures.contains)
        }
        manifest.status = .frozen
        manifest.frozenAt = ISO8601DateFormatter().string(from: Date())
        manifest.frozenBy = "swift"
        manifest.freezeHash = manifestHash(manifest)
        manifest.gitCommit = currentGitCommit()

        // Bypass the frozen-immutability guard for this one transition.
        let url = manifestURL(manifest.name)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: url)
        // A9: export preregistration.md beside the frozen manifest —
        // generated at the freeze instant from the frozen manifest so it
        // cannot disagree with what was frozen (the server's
        // `_write_preregistration` twin: same sections and facts;
        // byte-identity across engines is a non-goal). Best-effort like the
        // server's — a failed export never un-freezes a stamped manifest.
        try? preregistrationMarkdown(manifest).write(
            to: directory.appending(components: manifest.name, "preregistration.md"),
            atomically: true, encoding: .utf8)
        // The stamped manifest cannot be inside the commit it stamps (a
        // commit cannot contain its own hash), so a managed workspace gets a
        // follow-up stamp commit — freeze leaves the tree clean. The
        // gitCommit field still names the commit holding the pinned bytes.
        if autoCommit {
            autoCommitWorkspace(freezing: "\(manifest.name) (stamp)")
        }
        return manifest
    }

    /// The preregistration export's content — the study's settings-chosen-
    /// before-measurement statement, derived from the FROZEN manifest.
    /// Section headings and facts mirror the server's
    /// `_write_preregistration` (experiment_store.py); engines are not
    /// byte-identical (number formatting differs), but a reader of either
    /// file sees the same design. Pure and fixture-testable.
    public static func preregistrationMarkdown(_ manifest: ExperimentManifest) -> String {
        func formatNumber(_ value: Double) -> String {
            value == value.rounded() && abs(value) < 1e15
                ? String(Int(value))
                : "\(value)"
        }
        var lines = [
            "# Preregistration: \(manifest.name)",
            "",
            "- **Frozen at:** \(manifest.frozenAt ?? "None")",
            "- **Freeze hash:** `\(manifest.freezeHash ?? "None")`",
            "- **Git commit:** `\(manifest.gitCommit ?? "None")`",
            "- **Model:** \(manifest.modelID) @ `\(manifest.modelRevision ?? "None")`",
            "- **Phase:** \(manifest.phase ?? "unspecified")",
            "- **Case family:** \(manifest.caseFamily ?? "unspecified")",
            "- **Outcome instruments:** "
                + ((manifest.outcomeInstruments ?? []).isEmpty
                    ? "sampledText"
                    : (manifest.outcomeInstruments ?? []).joined(separator: ", ")),
            "- **Sampling:** temperature \(formatNumber(manifest.temperature)), "
                + "samplesPerItem \(manifest.samplesPerItem ?? 1), "
                + "seedPolicy \(manifest.seedPolicy ?? "manifestSeeds")",
            "",
            "## Conditions",
            "",
        ]
        for condition in manifest.conditions {
            let slots = condition.slots
                .map { "\($0.concept)@L\($0.layer) α=\(formatNumber($0.alpha))" }
                .joined(separator: ", ")
            let suffix = condition.controlType.map { " [\($0)]" } ?? ""
            lines.append(
                "- **\(condition.name)**\(suffix): "
                    + (slots.isEmpty ? "none (baseline)" : slots))
        }
        if let rule = manifest.promotionRule {
            lines += [
                "",
                "## Promotion rule (screen → confirm)",
                "",
                "- FDR threshold: \(formatNumber(rule.fdrThreshold ?? 0.05)) "
                    + "(Benjamini–Hochberg across concepts)",
                "- Dose-monotonicity required: \(rule.doseMonotone ?? true)",
                "- Must exceed matched-norm random floor: "
                    + "\(rule.exceedsRandomFloor ?? true)",
                "- Capability gate: "
                    + ((rule.capabilityGate?.isEmpty ?? true)
                        ? "none" : rule.capabilityGate!),
            ]
        }
        if let baseline = manifest.humanBaseline {
            lines += [
                "",
                "## Human baseline",
                "",
                "- `\(baseline.path)` (SHA-256 `\(baseline.hash)`)",
                "- Residual: R = delta_model − delta_human, per endpoint.",
            ]
        }
        lines += [
            "",
            "## Statistics",
            "",
            "- Paired to each item's same-case baseline; percentile bootstrap CIs "
                + "on the mean paired difference; Wilcoxon signed-rank as the "
                + "nonparametric companion.",
            "- Multiple comparisons: BH-FDR across concepts at screen; Holm within "
                + "the pre-registered family at confirm.",
            "",
            "*Generated at freeze; do not edit. Duplicate the experiment to change "
                + "anything.*",
            "",
        ]
        return lines.joined(separator: "\n")
    }

    // MARK: - Trained-adapter provenance (LoRA readiness §0 amendments 1+2)
    //
    // CROSS-ENGINE ASYMMETRY, deliberate and documented: the training sidecar
    // (`<run>/<adapter>.json`) is a SERVER artifact — evidence-grade LoRA
    // training runs as a Slurm job on the cluster (amendment 3), and this
    // engine may never hold the runs/ tree it lives in. So:
    //
    //   * the DATASET MANIFEST is verified here unconditionally. It lives in
    //     the workspace `adapters/` tree, and the Mac workspace is the source
    //     of truth, so its bytes ARE local;
    //   * the ADAPTER SIDECAR hash is verified only when the file is present
    //     locally. An absent sidecar is not drift, it is a remote artifact,
    //     and refusing on it would make every cluster-trained adapter
    //     unverifiable on the authoring machine. The server verifies it
    //     unconditionally (`manifest._verify_training_provenance`), which is
    //     where the check has teeth;
    //   * the GATE below is armed from the manifest's own
    //     `trainingProvenance.evidenceGrade` in addition to the sidecar, so a
    //     declared evidence-grade adapter is gated on either engine.

    /// The adapter training sidecar's URL for a variant: `<run>/<name>.json`
    /// beside `<run>/<name>/` (server twin: `model_variant.adapter_sidecar_path`).
    /// nil when the variant applies no adapter.
    static func adapterSidecarURL(for variant: ExperimentManifest.VariantCondition)
        -> URL?
    {
        guard
            let adapter = variant.artifact.adapters.first(where: {
                !$0.adapterDirectory.isEmpty || !$0.artifactPath.isEmpty
            })
        else { return nil }
        let reference = adapter.adapterDirectory.isEmpty
            ? adapter.artifactPath : adapter.adapterDirectory
        let directory = resolveProjectPath(reference)
            .standardizedFileURL
        return directory.deletingLastPathComponent()
            .appending(path: directory.lastPathComponent + ".json")
    }

    /// The parsed adapter training sidecar, or nil when the variant has no
    /// adapter or the file is absent/unreadable here (the ordinary case for a
    /// cluster-trained adapter).
    static func adapterTrainingSidecar(
        for variant: ExperimentManifest.VariantCondition
    ) -> [String: Any]? {
        guard let url = adapterSidecarURL(for: variant),
            let data = try? Data(contentsOf: url),
            let parsed = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return nil }
        return parsed
    }

    /// Whether a training sidecar claims EVIDENCE-GRADE provenance: schema v2
    /// or later AND `evidenceGrade: true`. Server twin:
    /// `manifest.sidecar_is_evidence_grade`.
    static func sidecarIsEvidenceGrade(_ sidecar: [String: Any]?) -> Bool {
        guard let sidecar else { return false }
        let version = (sidecar["schemaVersion"] as? Int)
            ?? Int((sidecar["schemaVersion"] as? Double) ?? 0)
        guard version >= trainingProvenanceSchemaVersion else { return false }
        return (sidecar["evidenceGrade"] as? Bool) ?? false
    }

    /// The sidecar schema version at which a trained adapter carries
    /// evidence-grade training provenance (contract §7). Below it — or no
    /// sidecar at all — the adapter is EXPLORATORY: legal in a study, never
    /// evidence.
    static let trainingProvenanceSchemaVersion = 2

    /// Whether an adapter variant is evidence-grade, from EITHER side of the
    /// pin: the manifest's own stamp (checkable anywhere) or the sidecar's
    /// (checkable only where the artifact tree is local). Server twin:
    /// `manifest.variant_is_evidence_grade`.
    static func variantIsEvidenceGrade(
        _ variant: ExperimentManifest.VariantCondition
    ) -> Bool {
        if variant.trainingProvenance?.evidenceGrade == true { return true }
        return sidecarIsEvidenceGrade(adapterTrainingSidecar(for: variant))
    }

    /// Drift/completeness violations for a variant's `trainingProvenance`.
    /// Absent block = no violations; a PRESENT block is checked exactly like
    /// stimuli, and a half-pin (path without hash, or hash without path)
    /// certifies nothing and refuses. Server twin:
    /// `manifest._verify_training_provenance`.
    static func trainingProvenanceViolations(
        _ variant: ExperimentManifest.VariantCondition
    ) -> [String] {
        guard let block = variant.trainingProvenance else { return [] }
        var violations: [String] = []
        let label = "variant '\(variant.name)' trainingProvenance"
        let path = (block.datasetManifestPath ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hash = (block.datasetManifestHash ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty, hash.isEmpty {
            violations.append(
                "\(label) names dataset manifest '\(path)' with no "
                    + "datasetManifestHash — a half-pin certifies nothing")
        } else if path.isEmpty, !hash.isEmpty {
            violations.append(
                "\(label) pins a datasetManifestHash with no "
                    + "datasetManifestPath — a half-pin certifies nothing")
        } else if !path.isEmpty {
            let url = resolveProjectPath(path)
            if let data = try? Data(contentsOf: url) {
                let live = sha256Hex(data)
                if live != hash {
                    violations.append(
                        "\(label) dataset manifest changed since pinning "
                            + "(have \(live.prefix(12))…, pinned \(hash.prefix(12))…)")
                }
            } else {
                violations.append(
                    "\(label) dataset manifest: file missing at \(path)")
            }
        }
        let sidecarHash = (block.adapterSidecarHash ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !sidecarHash.isEmpty {
            guard let url = adapterSidecarURL(for: variant) else {
                violations.append(
                    "\(label) pins an adapterSidecarHash but the variant "
                        + "declares no adapter directory")
                return violations
            }
            // Present-but-different is drift; ABSENT is a remote artifact
            // (see the asymmetry note above), never a violation here.
            if let data = try? Data(contentsOf: url) {
                let live = sha256Hex(data)
                if live != sidecarHash {
                    violations.append(
                        "\(label) adapter sidecar changed since pinning "
                            + "(have \(live.prefix(12))…, pinned "
                            + "\(sidecarHash.prefix(12))…)")
                }
            }
        }
        return violations
    }

    /// Stamp each adapter variant's `trainingProvenance` from the adapter's
    /// own v2 training sidecar, pin-when-absent PER KEY (the `markersHash`
    /// pattern one level down): a value already in the manifest is never
    /// overwritten — that would be a silent re-pin, and it would destroy the
    /// ex ante `matchedControl` declaration, which is researcher data the
    /// trainer knows nothing about.
    ///
    /// On this engine it is usually a no-op (the sidecar is a server
    /// artifact); it exists so a locally-held adapter freezes with the same
    /// pins the server would stamp. Server twin:
    /// `experiment_store._pin_training_provenance`.
    static func pinTrainingProvenance(into manifest: inout ExperimentManifest) {
        for index in manifest.variantConditions.indices {
            let variant = manifest.variantConditions[index]
            guard variant.fromPromotion == nil else { continue }
            guard let url = adapterSidecarURL(for: variant),
                let sidecar = adapterTrainingSidecar(for: variant),
                let data = try? Data(contentsOf: url)
            else { continue }
            let version = (sidecar["schemaVersion"] as? Int)
                ?? Int((sidecar["schemaVersion"] as? Double) ?? 0)
            guard version >= trainingProvenanceSchemaVersion else { continue }
            let dataset = sidecar["dataset"] as? [String: Any] ?? [:]
            var block = variant.trainingProvenance
                ?? ExperimentManifest.VariantCondition.TrainingProvenance()
            if block.datasetBundleID == nil {
                block.datasetBundleID = dataset["bundleID"] as? String
            }
            // Path and hash are ONE pin: stamping a path the sidecar could
            // not hash would mint a half-pin that refuses at verify.
            let datasetPath = (dataset["manifestPath"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let datasetHash = (dataset["manifestHash"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !datasetPath.isEmpty, !datasetHash.isEmpty {
                if block.datasetManifestPath == nil {
                    block.datasetManifestPath = datasetPath
                }
                if block.datasetManifestHash == nil {
                    block.datasetManifestHash = datasetHash
                }
            }
            if block.adapterSidecarHash == nil {
                block.adapterSidecarHash = sha256Hex(data)
            }
            if block.evidenceGrade == nil {
                block.evidenceGrade = (sidecar["evidenceGrade"] as? Bool) ?? false
            }
            manifest.variantConditions[index].trainingProvenance = block
        }
    }

    /// Non-blocking advisories about ADAPTER variant conditions. Server twin:
    /// `experiment_store._adapter_variant_advisories`.
    ///
    /// * an EXPLORATORY adapter — pre-v2 sidecar, no sidecar, or a v2 sidecar
    ///   that does not claim evidence grade. Legal (pilots, robustness arms),
    ///   but it cannot carry a citable intervention;
    /// * an evidence-grade adapter with NO `matchedControl` declared.
    ///   Amendment 2: every evidence-grade stance adapter trains alongside an
    ///   S0-analog control on an identical schedule, and WHICH neutralization
    ///   a study uses is manifest data declared before training. An adapter
    ///   arm with no declared control is an intervention with no
    ///   counterfactual.
    static func adapterVariantAdvisories(
        _ manifest: ExperimentManifest
    ) -> [String] {
        var advisories: [String] = []
        for variant in manifest.variantConditions
        where variant.fromPromotion == nil && !variant.artifact.adapters.isEmpty {
            guard variantIsEvidenceGrade(variant) else {
                advisories.append(
                    "variant '\(variant.name)' uses an EXPLORATORY adapter (no "
                        + "evidence-grade training provenance: the adapter's "
                        + "sidecar is pre-v2 or does not claim evidence grade) — "
                        + "fine for pilots and robustness arms; a citable adapter "
                        + "intervention must be trained through the "
                        + "evidence-grade path")
                continue
            }
            if variant.trainingProvenance?.matchedControl == nil {
                advisories.append(
                    "variant '\(variant.name)' pins an evidence-grade adapter "
                        + "with NO matchedControl declared — an adapter "
                        + "intervention without its matched "
                        + "(neutralized/shuffled-label) control arm has no "
                        + "counterfactual; declare "
                        + "trainingProvenance.matchedControl before training")
            }
        }
        return advisories
    }

    /// Modality arms need a validity story (WORK-PLAN Phase E): a variant
    /// condition's interventions must be fully pinned — adapter content
    /// hashes, system-prompt hash, vector artifact ids — or the condition is
    /// unverifiable the moment it freezes. `freeze --force` skips this
    /// loudly, like the other evidence gates; the always-run verify() of the
    /// artifact-file hash is never skippable.
    private static func checkVariantValidity(_ manifest: ExperimentManifest) throws {
        let name = manifest.name
        for variant in manifest.variantConditions {
            if variant.fromPromotion != nil {
                // Forward-referenced (stage 4): the artifact does not exist
                // at freeze time BY DESIGN — its pins land at run time on
                // the server and are recorded in the run directory.
                // verify() enforces the declaration shape.
                continue
            }
            if variant.artifactHash.isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has no "
                        + "pinned artifactHash")
            }
            for adapter in variant.artifact.adapters
            where (adapter.adapterHash ?? "").isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' adapter "
                        + "'\(adapter.name)' has no adapterHash — re-save the variant "
                        + "with hashed adapter weights, or freeze --force")
            }
            let systemPrompt = (variant.artifact.systemPrompt ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !systemPrompt.isEmpty,
                (variant.artifact.systemPromptHash ?? "").isEmpty
            {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has a "
                        + "system prompt but no systemPromptHash — re-save the variant, "
                        + "or freeze --force")
            }
            for injection in variant.artifact.injections
            where injection.vectorArtifactID.isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has an "
                        + "injection for '\(injection.concept)' without a "
                        + "vectorArtifactID pin")
            }
            // Trained-adapter arms owe the same story about their TRAINING
            // DATA (LoRA readiness §0 amendment 1). An evidence-grade adapter
            // whose dataset is not pinned into the manifest is unverifiable
            // the moment it freezes: the training files could change
            // afterwards with nothing to flag the drift. Exploratory adapters
            // are legal and produce an advisory instead, never a refusal.
            if !variant.artifact.adapters.isEmpty,
                variantIsEvidenceGrade(variant),
                (variant.trainingProvenance?.datasetManifestHash ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' uses an "
                        + "evidence-grade adapter but carries no "
                        + "trainingProvenance.datasetManifestHash — its training data "
                        + "would stay outside the freeze pin surface. Re-attach the "
                        + "variant so freeze can pin the dataset manifest from the "
                        + "adapter's sidecar, or freeze --force")
            }
        }
    }

    /// Capability-battery-as-evidence gate (server `experiment_store` twin):
    /// a variant study freezes only when its scope-matched validate evidence
    /// carries battery results for baseline AND every variant condition,
    /// all scored against the expected battery hash (the manifest's pin, or
    /// the live default battery). `freeze --force` skips this loudly, never
    /// silently.
    private static func checkVariantBatteryEvidence(
        _ manifest: ExperimentManifest,
        runSubstrate: String = ExperimentStore.evidenceSubstrate
    ) throws {
        guard !manifest.variantConditions.isEmpty else { return }
        let name = manifest.name
        guard
            let evidenceDirectory = validationEvidence(
                for: manifest, runSubstrate: runSubstrate)
        else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': no validate run matches its exact pins "
                    + "(model+revision, variant capability battery). Variant studies "
                    + "validate the pinned battery per condition — run 'steerlab-cli "
                    + "experiment validate \(name)' first, or freeze --force")
        }
        let results = Dictionary(
            (validationEvidenceBatteryResults(at: evidenceDirectory) ?? [])
                .map { ($0.condition, $0) },
            uniquingKeysWith: { first, _ in first })
        // Forward-referenced conditions (stage 4) are exempt: their agent
        // does not exist at validate time — their battery evidence is the
        // RUN's per-condition battery, produced after server-side
        // resolution.
        let required = ["baseline"] + manifest.variantConditions
            .filter { $0.fromPromotion == nil }
            .map(\.name)
        let missing = required.filter { results[$0] == nil }
        guard missing.isEmpty else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': matching validate evidence has no "
                    + "capability-battery results for condition(s): "
                    + missing.joined(separator: ", ")
                    + " — re-run 'steerlab-cli experiment validate \(name)' (each "
                    + "variant condition runs the pinned battery), or freeze --force")
        }
        if let expected = manifest.capabilityBatteryHash
            ?? effectiveCapabilityBattery(for: manifest)?.hash
        {
            let drifted = required
                .filter { results[$0]?.batteryHash != expected }
                .sorted()
            guard drifted.isEmpty else {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': capability battery drifted since "
                        + "validation for condition(s): " + drifted.joined(separator: ", ")
                        + " — re-run 'steerlab-cli experiment validate \(name)', or "
                        + "freeze --force")
            }
        }
    }

    /// Judge-rubric gate: a judge-evaluated study (paired-judge evaluation
    /// or an explicit judge panel) freezes only with a pinned rubric FILE
    /// (inline text is draft-only) and >=2 judges, so the report can carry
    /// agreement statistics. `freeze --force` skips loudly, never silently;
    /// the rubric-hash drift check in verify() is never skippable.
    private static func checkJudgeEvaluationValidity(_ manifest: ExperimentManifest) throws {
        let judgeEvaluated =
            manifest.evaluation?.kind == .pairedJudge || !(manifest.judges ?? []).isEmpty
        guard judgeEvaluated else { return }
        let name = manifest.name
        guard manifest.judgeRubricFile != nil, manifest.judgeRubricHash != nil else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': judge-evaluated study has no pinned "
                    + "judge rubric file — pin one: 'steerlab-cli experiment "
                    + "pin-rubric \(name) \(JudgeRubricStore.defaultRubricFile)'; "
                    + "inline rubric text is draft-only. Or freeze --force")
        }
        let judgeCount = (manifest.judges ?? []).count
        guard judgeCount >= 2 else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': judge-evaluated study pins "
                    + "\(judgeCount) judge\(judgeCount == 1 ? "" : "s"); at least 2 are "
                    + "required so the report carries agreement statistics "
                    + "(percent agreement, Cohen's kappa) — pin a panel: "
                    + "'steerlab-cli experiment pin-rubric \(name) <rubric> "
                    + "--judges <name>:<kind>,<name>:<kind>'. Or freeze --force")
        }
        if let indistinct = judgePanelIndistinctProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(indistinct)")
        }
        if let pipelineProblem = localJudgePipelineProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(pipelineProblem)")
        }
        if let unpinned = unpinnedForeignLocalJudgeProblem(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(unpinned)")
        }
        if let conflict = studyModelJudgePinConflict(manifest) {
            throw ExperimentError(reason: "cannot freeze '\(name)': \(conflict)")
        }
    }

    /// Local judges whose declared model differs from the study model,
    /// rendered `'name' (model 'id')` — the server's
    /// `_foreign_local_judges` twin.
    private static func foreignLocalJudges(_ manifest: ExperimentManifest) -> [String] {
        (manifest.judges ?? []).compactMap { judge -> String? in
            guard !judge.name.isEmpty else { return nil }
            let kind = judge.kind.trimmingCharacters(in: .whitespacesAndNewlines)
            guard kind == "local" else { return nil }
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !declared.isEmpty, declared != manifest.modelID else {
                return nil
            }
            return "'\(judge.name)' (model '\(declared)')"
        }
    }

    /// Finding 1 gate, fan-out era (2026-07-23; live incident 2026-07-22:
    /// judge-1=gemma-4b + judge-2=gemma-12b froze fine, then the frozen
    /// pipeline died at runtime because the chain holds ONE model). A
    /// judged SWEEP inside the chain still cannot use a local judge whose
    /// model differs from the study model — sweep judging is interleaved
    /// with the selection, not a separable post-stage, so no fan-out
    /// exists for it. The EVALUATE stage no longer refuses: it routes to
    /// the server's post-generation judge fan-out
    /// (`localJudgeFanoutNote`). Wording matches the server's
    /// `local_judge_pipeline_problem`.
    static func localJudgePipelineProblem(_ manifest: ExperimentManifest) -> String? {
        guard pipelineBlockViolations(manifest.pipeline).isEmpty,
            let draft = PipelineDraft.parse(manifest.pipeline),
            draft.stages.contains("sweep"),
            manifest.sweep?.selection?.objective?.metric == "judgeScore"
        else { return nil }
        let offenders = foreignLocalJudges(manifest)
        guard !offenders.isEmpty else { return nil }
        return "the declared pipeline's sweep stage holds ONE model — "
            + "the study model '\(manifest.modelID)' — but local judge(s) "
            + offenders.joined(separator: ", ")
            + " resolve to a different model, which cannot load inside the "
            + "chain (the judge fan-out covers the evaluate stage only). "
            + "Leave a local judge's model empty to judge with the study "
            + "model, pin claude/openrouter judges, or select on logprobShift"
    }

    /// Routing information (never a gate, 2026-07-23): a declared pipeline
    /// whose EVALUATE stage pins local judges resolving to models other
    /// than the study model judges them as the server's post-generation
    /// judge fan-out — one worker job per distinct judge model, merged
    /// when every judge × response-pair cell appears exactly once.
    /// Wording matches the server's `local_judge_fanout_note`.
    /// Foreign local judges whose model bytes are not pinned.
    ///
    /// A local judge resolving to the STUDY model inherits the study's
    /// pinned revision, so "the same judge" across two sessions is a fact.
    /// One naming a DIFFERENT model has no such pin to inherit, and freeze
    /// deliberately leaves its revision blank. That was tolerable while a
    /// judgment artifact merely RECORDED what loaded — but targeted retry
    /// compares recorded identities to decide whether an earlier session's
    /// verdicts may be REUSED, and two sessions can each load a different
    /// default while both records say nil, where nil == nil passes.
    ///
    /// Python twin: `experiment_store.unpinned_foreign_local_judge_problem`.
    /// Identical wording on both engines.
    /// Canonical dtype names a judge may pin. The Python twin is
    /// `experiment_store.JUDGE_DTYPE_VOCABULARY`; the server's loader
    /// (`model_loader.DTYPE_VOCABULARY`) is the same set, and a test on that
    /// engine asserts the two agree.
    public static let judgeDtypeVocabulary = ["bfloat16", "float16", "float32"]

    private static let judgeDtypeAliases = [
        "bfloat16": "bfloat16", "bf16": "bfloat16",
        "float16": "float16", "fp16": "float16",
        "float32": "float32", "fp32": "float32",
    ]

    /// Canonical spelling of a judge dtype alias, or nil if unrecognized.
    public static func normalizeJudgeDtype(_ value: String?) -> String? {
        judgeDtypeAliases[
            (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()]
    }

    /// The pins to prefill for a local judge row, so a researcher is not
    /// asked to type a 40-character commit hash by hand (external review
    /// round 4, finding 5).
    ///
    /// Returns nil for anything that needs no pins — a blank model, or the
    /// study model, which inherits the study's own pin. Only ever fills a
    /// BLANK field: a pin already set is returned unchanged, so re-picking a
    /// model never silently rewrites a deliberate choice.
    ///
    /// `bfloat16` is the dtype every CUDA judge normally runs in and the
    /// server's own CUDA default. Prefilling it into a visible, editable
    /// field is not the silent defaulting that finding 2 was about — that
    /// was recording a dtype the LOAD never honored.
    public static func judgePinPrefill(
        model: String?,
        previousModel: String? = nil,
        studyModel: String,
        revision: String?,
        dtype: String?,
        resolveRevision: (String) -> String?
    ) -> (revision: String?, dtype: String?)? {
        let declared = (model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !declared.isEmpty, declared != studyModel else { return nil }
        let previous = (previousModel ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // A revision pins nothing on its own — only a (model, revision) PAIR
        // identifies bytes. So when the MODEL changes, the old revision stops
        // being a pin and becomes a wrong one: it would freeze happily and
        // then fail on the compute node, because the new model has no such
        // commit (external review round 5, finding 2).
        //
        // Conditioned on a genuine model→model transition, NOT merely on the
        // value having changed: the same notification fires when a different
        // study is loaded into the editor, and clearing there would wipe a
        // pin the manifest legitimately carries. An empty `previous` means
        // "no prior model to have moved away from" — leave the pins alone.
        let modelChanged = !previous.isEmpty && previous != declared
        let currentRevision = modelChanged ? "" : (revision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let currentDtype = (dtype ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // dtype survives a model change: it describes how to LOAD, not which
        // bytes, so it stays meaningful for a different model.
        return (currentRevision.isEmpty ? resolveRevision(declared) : revision,
                currentDtype.isEmpty ? "bfloat16" : dtype)
    }

    /// Whether a revision names FIXED bytes rather than a moving ref.
    ///
    /// Hexadecimal — the shape of a git commit hash, full or abbreviated.
    /// Branch names (`main`, `master`), `HEAD`, `refs/...` paths, and
    /// conventional tags (`v1.0`, `latest`) all fail it, which is the
    /// point: a branch is re-pointed by definition and a tag can be moved,
    /// so neither identifies the bytes a run used.
    ///
    /// Honest residual: a tag whose name happens to be hexadecimal would
    /// pass. No format check can distinguish that from a short hash — only
    /// asking the hub could — and it is not a shape anyone tags in practice.
    static func isCommitLike(_ revision: String) -> Bool {
        let stripped = revision.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else { return false }
        return stripped.allSatisfy(\.isHexDigit)
    }

    /// Revision pins that name a moving ref instead of a commit.
    ///
    /// Applies to the STUDY revision and to every local judge's. `"main"`
    /// passed the old gate — it only required non-emptiness — and the
    /// loader then recorded the symbolic name it was handed rather than the
    /// commit it resolved to, so two runs a week apart could record the same
    /// "pin" and have run different weights (external review round 5,
    /// finding 4).
    ///
    /// Python twin: `experiment_store.symbolic_revision_problem`.
    static func symbolicRevisionProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        var offenders: [String] = []
        let study = (manifest.modelRevision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !study.isEmpty, !isCommitLike(study) {
            offenders.append("the study model pins '\(study)'")
        }
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            let revision = (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !revision.isEmpty, !isCommitLike(revision) {
                offenders.append("judge '\(judge.name)' pins '\(revision)'")
            }
        }
        guard !offenders.isEmpty else { return nil }
        return "revision pin(s) name a moving reference rather than a commit: "
            + offenders.joined(separator: "; ")
            + ". A branch or tag is re-pointed by definition, so it cannot "
            + "identify the weights a run used — two runs a week apart would "
            + "record the same pin having loaded different bytes. Use the "
            + "commit hash (the Resolve button reads it from whichever "
            + "substrate will run the model)"
    }

    /// A study-model local judge declaring pins that differ from the study's.
    ///
    /// Scoped to studies declaring a **judgeScore sweep** (external review
    /// round 5, finding 1). There, such a judge has no independent identity:
    /// the sweep judges with the already-HELD study model rather than
    /// loading anything, so a divergent `revision`/`dtype` is silently
    /// ignored while remaining in the criterion provenance.
    ///
    /// Deliberately NOT a blanket rule. `evaluate` genuinely LOADS a
    /// declared judge revision, so judging with a different checkpoint of
    /// the study repo is a legitimate design there — freeze has always
    /// preserved a declared revision for exactly that reason. The defect is
    /// the SWEEP path silently ignoring what evaluate honors: one manifest,
    /// two identities, depending on the verb.
    ///
    /// Forbidding divergence is preferred over verify-and-stamp: it is
    /// checkable while authoring, rather than producing an artifact merely
    /// honest about having judged with something else. Pins that AGREE with
    /// the study stay legal — redundant, not wrong.
    ///
    /// Python twin: `experiment_store.study_model_judge_pin_conflict`.
    static func studyModelJudgePinConflict(
        _ manifest: ExperimentManifest
    ) -> String? {
        guard manifest.sweep?.selection?.objective?.metric == "judgeScore" else {
            return nil
        }
        let studyRevision = (manifest.modelRevision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let studyDtype = (manifest.dtype ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var offenders: [String] = []
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Blank model AND explicit study model both resolve to the
            // study model.
            guard declared.isEmpty || declared == manifest.modelID else {
                continue
            }
            let revision = (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !revision.isEmpty, revision != studyRevision {
                offenders.append(
                    "'\(judge.name)' pins revision '\(revision)' but the study "
                        + (studyRevision.isEmpty
                            ? "has no revision pinned"
                            : "is pinned at '\(studyRevision)'"))
            }
            let dtype = (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !dtype.isEmpty,
                normalizeJudgeDtype(dtype) != normalizeJudgeDtype(studyDtype)
            {
                offenders.append(
                    "'\(judge.name)' pins dtype '\(dtype)' but the study "
                        + (studyDtype.isEmpty
                            ? "pins none (the device decides)"
                            : "is pinned at '\(studyDtype)'"))
            }
        }
        guard !offenders.isEmpty else { return nil }
        return "this study selects on judgeScore, and local judge(s) "
            + "resolving to the STUDY model cannot pin a different identity: " + offenders.joined(separator: "; ")
            + ". Such a judge IS the study model — a sweep judges with the "
            + "already-held weights and never loads anything else, so the "
            + "divergent pin would be silently ignored. Drop the pin to "
            + "inherit the study's, or name a different model to make it a "
            + "genuinely separate judge"
    }

    /// A study-level `dtype` outside the closed vocabulary.
    ///
    /// The Mac is the AUTHORING surface and the cluster is the measurement
    /// one, so this is validated here even though only the server consumes
    /// the key — a manifest must not reach the cluster carrying a dtype that
    /// refuses at load after a queue wait. Python twin:
    /// `experiment_store.unloadable_study_dtype_problem`.
    static func unloadableStudyDtypeProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        let spelled = (manifest.dtype ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spelled.isEmpty, normalizeJudgeDtype(spelled) == nil else {
            return nil
        }
        return "study dtype '\(spelled)' is not one this engine can load — "
            + "the loader accepts only "
            + judgeDtypeVocabulary.joined(separator: ", ")
            + " (aliases bf16/fp16/fp32). Leave it unset to let the device "
            + "decide, which is what every study did before this pin existed"
    }

    static func unpinnedForeignLocalJudgeProblem(
        _ manifest: ExperimentManifest
    ) -> String? {
        var offenders: [String] = []
        var unknown: [String] = []
        for judge in manifest.judges ?? [] where judge.kind == "local" {
            // A dtype OUTSIDE the closed vocabulary is checked for every
            // local judge, pinned or not: the server's loader refuses it at
            // run time, and discovering that on a compute node after a queue
            // wait is exactly the failure this firewall exists to move
            // forward in time (external review round 4, finding 2).
            let spelled = (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !spelled.isEmpty, normalizeJudgeDtype(spelled) == nil {
                unknown.append("'\(judge.name)' declares dtype '\(spelled)'")
            }
            let declared = (judge.model ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !declared.isEmpty, declared != manifest.modelID else { continue }
            var missing: [String] = []
            if (judge.revision ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                missing.append("revision")
            }
            if (judge.dtype ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                missing.append("dtype")
            }
            if !missing.isEmpty {
                offenders.append(
                    "'\(judge.name)' (model '\(declared)') is missing "
                        + missing.joined(separator: " and "))
            }
        }
        if !unknown.isEmpty {
            return "local judge(s) declare a dtype this engine cannot load: "
                + unknown.joined(separator: "; ")
                + ". The loader accepts only "
                + judgeDtypeVocabulary.joined(separator: ", ")
                + " (aliases bf16/fp16/fp32). An unrecognized value used to "
                + "load float32 silently, so the pin would be a false claim"
        }
        guard !offenders.isEmpty else { return nil }
        return "local judge(s) naming a model other than the study model "
            + "must pin the exact bytes that will judge: "
            + offenders.joined(separator: "; ")
            + ". Without a revision pin two judging sessions can load "
            + "different defaults while both records say 'none', so a "
            + "resumed evaluation cannot prove its reused verdicts came "
            + "from the same judge. Pin judges[].revision and "
            + "judges[].dtype, or use the study model as judge"
    }

    /// Plain-language note about WHERE this panel's judging would run on
    /// this host, or nil when there is nothing notable to say.
    ///
    /// Silent on the two unsurprising cases — a purely local panel, and a
    /// fully credentialed external panel — because an advisory that fires
    /// on every study is one nobody reads. It speaks up when judging would
    /// DEFER or REFUSE, which a researcher would otherwise discover only
    /// from the artifacts after a run.
    ///
    /// The confusing case it exists for: the judge key holds ONE kind, so a
    /// panel of one claude and one openrouter judge with an openrouter key
    /// credentials half the panel and defers the whole thing — despite a key
    /// having been deliberately placed. Evaluated against the CURRENT host,
    /// so the same manifest honestly says different things on a Mac and on a
    /// cluster node. Python twin: `experiment_store.judging_custody_advisory`.
    /// Non-blocking cleanliness advisory (field bug 2026-08-07): names the
    /// judges whose manifest entries carry fields their kind does not own —
    /// stale leftovers from a kind switch saved before the kind-owned write
    /// filter existed (a local judge keeping `provider` from its OpenRouter
    /// past). Such fields are claims about pins that do not exist for that
    /// kind; they invite verify confusion but invalidate nothing, so this
    /// follows the freeze-advisory precedent — advise, never refuse. A
    /// re-save in the Studies panel (or on a duplicate, for frozen studies)
    /// drops them.
    static func kindForeignJudgeFieldsAdvisory(
        _ manifest: ExperimentManifest
    ) -> String? {
        func present(_ value: String?) -> Bool {
            !(value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let offenders: [String] = (manifest.judges ?? []).compactMap { judge in
            let kept = judge.keepingKindOwnedFields()
            var foreign: [String] = []
            if kept.provider == nil, present(judge.provider) {
                foreign.append("provider")
            }
            if kept.revision == nil, present(judge.revision) {
                foreign.append("revision")
            }
            if kept.dtype == nil, present(judge.dtype) {
                foreign.append("dtype")
            }
            guard !foreign.isEmpty else { return nil }
            return "'\(judge.name)' (\(judge.kind): "
                + foreign.joined(separator: ", ") + ")"
        }
        guard !offenders.isEmpty else { return nil }
        return "judge(s) " + offenders.joined(separator: ", ")
            + " carry fields their kind does not own — stale from a kind "
            + "switch saved before the kind-owned write filter; they name "
            + "pins that do not exist for that kind. Re-save Evaluation "
            + "Settings (or duplicate, for a frozen study) to drop them"
    }

    static func judgingCustodyAdvisory(
        _ manifest: ExperimentManifest,
        keyForKind: (String) -> String? = { JudgeKeyStore.resolveKey(kind: $0) }
    ) -> String? {
        let judges = (manifest.judges ?? []).filter { !$0.name.isEmpty }
        guard !judges.isEmpty else { return nil }
        let external = judges.filter { $0.kind == "claude" || $0.kind == "openrouter" }
        guard !external.isEmpty else { return nil }
        let local = judges.filter { $0.kind == "local" }

        // Credential kinds this host cannot serve. Mirrors the server's
        // `_missing_external_credentials`: a judge KIND maps to a key kind
        // ("claude" needs anthropic), and the two can differ.
        let uncredentialed = external.filter {
            keyForKind($0.kind == "claude" ? "anthropic" : "openrouter") == nil
        }
        guard !uncredentialed.isEmpty else { return nil }
        let named = uncredentialed
            .map { "'\($0.name)' (\($0.kind))" }
            .joined(separator: ", ")
        let kinds = Set(uncredentialed.map(\.kind)).sorted().joined(separator: "/")
        if !local.isEmpty {
            return "judging would REFUSE on this host: the panel mixes local "
                + "judges with uncredentialed \(kinds) judges (\(named)) — a "
                + "split panel cannot defer coherently. Pin an all-local or "
                + "all-external panel, or place a judge key here."
        }
        return "judging would DEFER on this host: no \(kinds) credential for "
            + "\(named) — the whole panel defers, including any judge that IS "
            + "credentialed here. That is a legitimate design (keyless is the "
            + "default posture) — but if you expected inline judging, note "
            + "the key holds ONE kind, so a mixed panel needs a matching "
            + "credential for every external judge."
    }

    static func localJudgeFanoutNote(_ manifest: ExperimentManifest) -> String? {
        guard pipelineBlockViolations(manifest.pipeline).isEmpty,
            let draft = PipelineDraft.parse(manifest.pipeline),
            draft.stages.contains("evaluate")
        else { return nil }
        let offenders = foreignLocalJudges(manifest)
        guard !offenders.isEmpty else { return nil }
        return "the pipeline's evaluate stage will judge local judge(s) "
            + offenders.joined(separator: ", ")
            + " as a post-generation judge fan-out (one worker job per "
            + "distinct judge model; available on Slurm run-first pipeline "
            + "submissions — elsewhere the emitted packets await deferred "
            + "judging)"
    }

    /// A judge's RESOLVED identity — what will actually run, not what the
    /// manifest happens to spell. Cross-engine rules: a LOCAL judge with a
    /// blank model resolves to the STUDY model; a claude judge with a blank
    /// model resolves to the default Claude judge model; openrouter judges
    /// have no defaults (their own verify rules apply).
    struct ResolvedJudgeIdentity: Hashable {
        let kind: String
        let model: String
        let provider: String
    }

    static func resolvedJudgeIdentity(
        _ judge: ExperimentManifest.JudgeRef, studyModelID: String
    ) -> ResolvedJudgeIdentity {
        let rawKind = judge.kind.trimmingCharacters(in: .whitespacesAndNewlines)
        let kind = rawKind.isEmpty ? "claude" : rawKind
        var model = (judge.model ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let rawProvider = (judge.provider ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let provider =
            kind == "openrouter"
            ? OpenRouterProviderIdentity.canonical(rawProvider)
            : rawProvider
        if kind == "local", model.isEmpty {
            model = studyModelID
        } else if kind == "claude", model.isEmpty {
            model = ClaudePairedJudge.defaultModel
        }
        return ResolvedJudgeIdentity(kind: kind, model: model, provider: provider)
    }

    /// External review 2026-07-22 (finding 4): two blank-model local judges
    /// both resolve to the study model at temperature 0 — identical
    /// deterministic judges whose perfect agreement is guaranteed by
    /// construction, satisfying a count-only panel gate while providing
    /// zero independence. Returns the plain-language problem (identical
    /// wording on both engines) when a panel of >= 2 named judges collapses
    /// to fewer than 2 DISTINCT resolved identities, else nil. Shared by
    /// the freeze gate (judgeValidity), the freeze advisories, and the
    /// data-check judge-panel row.
    static func judgePanelIndistinctProblem(_ manifest: ExperimentManifest) -> String? {
        let judges = (manifest.judges ?? []).filter { !$0.name.isEmpty }
        guard judges.count >= 2 else { return nil }
        var identities: [ResolvedJudgeIdentity: [String]] = [:]
        var order: [ResolvedJudgeIdentity] = []
        for judge in judges {
            let identity = resolvedJudgeIdentity(
                judge, studyModelID: manifest.modelID)
            if identities[identity] == nil { order.append(identity) }
            identities[identity, default: []].append(judge.name)
        }
        guard identities.count < 2, let identity = order.first else { return nil }
        let names = identities[identity] ?? []
        let quoted = names.map { "'\($0)'" }
        let joined =
            quoted.count == 2
            ? quoted.joined(separator: " and ")
            : quoted.dropLast().joined(separator: ", ") + " and "
                + (quoted.last ?? "")
        let quantifier = quoted.count == 2 ? "both" : "all"
        let what: String
        if identity.kind == "local", identity.model == manifest.modelID {
            what = "the study model at temperature 0"
        } else if !identity.provider.isEmpty {
            what = "the \(identity.kind) judge '\(identity.model)' via "
                + "'\(identity.provider)'"
        } else {
            what = "the \(identity.kind) judge '\(identity.model)'"
        }
        return "judges \(joined) \(quantifier) resolve to the same "
            + "deterministic judge (\(what)) — they would agree perfectly by "
            + "construction; use judges with different models, kinds, or "
            + "providers"
    }

    /// Full pinned snapshot (freeze, after `pinExternalInputs`): byte-copies
    /// EVERY pinned input into `experiments/<name>/pinned/` — concept
    /// stimulus directories, grand-mean story files (targets and every
    /// corpus member), task prompts, judge rubric, capability battery,
    /// human baseline/validation tables, and the multi-agent scenario.
    /// Manifest paths for prompts-resident inputs are NOT rewritten: the
    /// pinned hashes prove identity, and the snapshot is the no-git
    /// reproducibility floor (a frozen study survives later stimulus edits
    /// even in an unversioned workspace). Inputs already under pinned/
    /// (runs-resident inputs `pinExternalInputs` just repointed) are
    /// skipped. Pure file copies — runs under the test root override, where
    /// the git auto-commit is skipped.
    static func snapshotPinnedInputs(for manifest: ExperimentManifest) throws {
        let fm = FileManager.default
        let pinnedDirectory = directory.appending(components: manifest.name, "pinned")
        let pinnedPrefix = pinnedDirectory.standardizedFileURL.path + "/"

        // destination (relative to pinned/) → source; first plan wins so a
        // grand-mean target that is also a corpus member copies once.
        var planned: [(relative: String, source: URL)] = []
        var seen = Set<String>()
        func plan(source: URL, destination relative: String) {
            guard fm.fileExists(atPath: source.path) else { return }
            guard !source.standardizedFileURL.path.hasPrefix(pinnedPrefix) else { return }
            guard seen.insert(relative).inserted else { return }
            planned.append((relative, source))
        }

        // Same source resolution as verify(): paired concepts read their
        // prompts/concepts/<name> directory, grand-mean concepts and corpus
        // members read prompts/emotions/<name>/stories.jsonl. Snapshot the
        // OPERATIVE surface only (2026-07-19): configuration carried from
        // another study type is neither verified nor snapshotted.
        let machinery = conceptMachineryOperative(manifest)
        let modelOutputOperative = manifest.studyKind == .modelOutput
        for ref in manifest.concepts
        where machinery && ref.options.method.isPaired {
            plan(
                source: VectorCatalog.conceptsDirectory.appending(component: ref.name),
                destination: "concepts/\(ref.name)")
        }
        // Story-corpus concepts + grand-mean members + DESIGNATED REFERENCES
        // (external review 2026-07-31, finding 4: the reference corpus was
        // bundled but not snapshotted — the no-git floor was incomplete).
        // TODO(review finding 4): derive this whole snapshot from
        // pinnedInputEntries so the two enumerations cannot drift again.
        let storyConcepts = machinery
            ? manifest.concepts.filter { $0.options.method.usesStoryCorpus }.map(\.name)
                + (manifest.grandMeanCorpus?.concepts ?? [])
                + manifest.concepts.compactMap { $0.designatedReference?.name }
            : []
        for name in storyConcepts {
            plan(
                source: storiesURL(for: name),
                destination: "emotions/\(name)/stories.jsonl")
        }
        // Measurement-side inputs of grand-mean concepts (a paired concept's
        // markers/validation ride along with its whole directory above).
        for ref in manifest.concepts
        where machinery && ref.options.method.usesStoryCorpus {
            // The SOURCE is the file the dual-root lookup found (canonical
            // home first, the other recipe's home as fallback) so a misfiled
            // set still travels into the no-git floor; the DESTINATION stays
            // the canonical layout, because that is where the snapshot's
            // reader will look for it.
            plan(
                source: resolveConceptValidation(name: ref.name, isPaired: false)?.url
                    ?? conceptValidationURL(name: ref.name, isPaired: false),
                destination: "emotions/\(ref.name)/validation.jsonl")
            plan(
                source: VectorCatalog.conceptsDirectory
                    .appending(components: ref.name, "markers.json"),
                destination: "concepts/\(ref.name)/markers.json")
        }
        func planFile(_ path: String?, role: String) {
            guard let path, !path.isEmpty else { return }
            let source = resolveProjectPath(path)
            plan(source: source, destination: "\(role)-\(source.lastPathComponent)")
        }
        // The neutral corpus DENOMINATES norm-unit α, so a frozen study whose
        // α is in norm units cannot be re-derived without it — and it was the
        // one pinned input `pinnedInputEntries` git-gates and packs that this
        // snapshot omitted (gate-5 dry run #2, P2). The no-git floor was
        // incomplete for the α denominator: an unversioned workspace could
        // edit `prompts/neutral/corpus.jsonl` after freeze and leave the
        // frozen manifest naming a hash whose bytes exist nowhere. The
        // destination mirrors the source layout so the snapshot reads as the
        // workspace it came from.
        //
        // THE PREDICATE IS THE PIN, NOT THE MACHINERY (2026-08-18, WP0
        // residual (c)). The first cut mirrored the pin surface's
        // `machinery && modelOutput` guard, which excluded exactly the
        // studies that keep the field without operating the concept
        // machinery — a compare-agents study whose promoted agents' α is in
        // norm units, and a panel carrying the pin forward. Their manifests
        // still NAME a corpus hash, so the bytes behind that name still have
        // to survive beside the frozen manifest; whether this engine
        // re-derives vectors from them is a different question. Snapshotting
        // is additive and freeze-time-only: no gate, no verify() rule, and no
        // already-frozen study changes. Server twin:
        // `_snapshot_pinned_inputs`.
        if manifest.neutralCorpusHash != nil {
            plan(
                source: resolveProjectPath("prompts/neutral/corpus.jsonl"),
                destination: "neutral/corpus.jsonl")
        }
        if modelOutputOperative {
            planFile(manifest.taskPromptsFile, role: "task-prompts")
            planFile(manifest.capabilityBatteryFile, role: "capability-battery")
        } else {
            planFile(manifest.multiAgentScenarioPath, role: "scenario")
        }
        // A trained adapter's DATASET MANIFEST is a pinned input (LoRA
        // readiness §0 amendment 1): verify() re-hashes it, so it belongs in
        // the no-git reproducibility floor beside the frozen manifest.
        if modelOutputOperative {
            for variant in manifest.variantConditions where variant.fromPromotion == nil {
                planFile(
                    variant.trainingProvenance?.datasetManifestPath,
                    role: "training-dataset")
            }
        }
        planFile(manifest.judgeRubricFile, role: "judge-rubric")
        planFile(manifest.reasoningStyleTaxonomyPath, role: "reasoning-style")
        planFile(manifest.humanBaseline?.path, role: "human-baseline")
        planFile(manifest.humanValidation?.path, role: "human-validation")

        for (relative, source) in planned {
            let destination = pinnedDirectory.appending(path: relative)
            try fm.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.copyItem(at: source, to: destination)
        }
    }

    /// Copies pinned inputs that live under gitignored `runs/` into the
    /// experiment's own directory (`experiments/<name>/pinned/`),
    /// byte-identical, and repoints the manifest. A frozen, git-committed
    /// The ONE filename slug rule, shared across engines: lowercase, every run
    /// of non-alphanumerics becomes a single `-`, leading/trailing `-`
    /// trimmed, empty falls back to "unnamed".
    ///
    /// Byte-identical to the Python `experiment_store._slugify`, and pinned by
    /// a cross-engine test over a shared fixture list. This matters wherever
    /// both engines name files in the SAME directory — panel scripts under
    /// `prompts/panels/` especially, where a disagreement produces two files
    /// for one panel on Linux and a silent overwrite on a case-insensitive
    /// Mac. Do not reach for `FineTuneStore.slugify` here: it passes `_`
    /// through and falls back to "adapter", so it is a different rule.
    public static func canonicalSlug(_ value: String) -> String {
        let slug = value.lowercased()
            .replacingOccurrences(
                of: "[^a-z0-9]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "unnamed" : slug
    }

    /// manifest must never depend on a file the repository does not version.
    private static func pinExternalInputs(
        into manifest: inout ExperimentManifest
    ) throws {
        let base = (rootOverride ?? VectorCatalog.projectRoot).standardizedFileURL
        let runsPrefix = base.appending(component: "runs").path + "/"

        func underRuns(_ path: String) -> Bool {
            let absolute =
                path.hasPrefix("/")
                ? URL(filePath: path).standardizedFileURL.path
                : base.appending(path: path).standardizedFileURL.path
            return absolute.hasPrefix(runsPrefix)
        }

        func copyIn(_ path: String, destinationName: String, name: String) throws -> String {
            let source = path.hasPrefix("/") ? URL(filePath: path) : base.appending(path: path)
            let pinnedDirectory = directory.appending(components: name, "pinned")
            try FileManager.default.createDirectory(
                at: pinnedDirectory, withIntermediateDirectories: true)
            let destination = pinnedDirectory.appending(component: destinationName)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
            return "experiments/\(name)/pinned/\(destinationName)"
        }

        let slugify = canonicalSlug

        // The scenario is the PANEL's input, not a kind-neutral one — the
        // mirror of the carried-agent case below, and the round-14 comment
        // calling it kind-neutral was wrong. A model-output study carrying a
        // scenario across a kind switch executes none of it: relocating a
        // stale one fails the copy, and relocating a present one rewrites and
        // commits inactive configuration (external review round 15).
        if manifest.studyKind == .multiAgent,
            let scenarioPath = manifest.multiAgentScenarioPath,
            underRuns(scenarioPath)
        {
            manifest.multiAgentScenarioPath = try copyIn(
                scenarioPath, destinationName: "scenario.json", name: manifest.name)
        }
        // Agent artifacts are relocated only for the study kind that RUNS
        // them. A panel carrying one across a kind switch would otherwise have
        // its carried configuration silently rewritten and an irrelevant
        // artifact committed — or, when the stale file is simply gone, fail
        // the copy — immediately after freeze emitted the advisory promising
        // that carried state is "not verified, snapshotted, or bundled for
        // this study kind" (external review round 14).
        guard modelOutputSurfacesOperative(manifest) else { return }
        for index in manifest.variantConditions.indices {
            let variant = manifest.variantConditions[index]
            guard !variant.artifactPath.isEmpty, underRuns(variant.artifactPath) else {
                continue
            }
            manifest.variantConditions[index].artifactPath = try copyIn(
                variant.artifactPath,
                destinationName: "variant-\(slugify(variant.name)).json",
                name: manifest.name)
        }
    }

    /// New draft from any experiment — the only way to iterate on a frozen
    /// one.
    @discardableResult
    public static func duplicate(name: String, as newName: String) throws -> ExperimentManifest {
        let source = try load(name: name)
        var copy = source
        copy.name = newName
        copy.createdAt = ISO8601DateFormatter().string(from: Date())
        copy.status = .draft
        copy.frozenAt = nil
        copy.freezeHash = nil
        copy.frozenBy = nil
        copy.gitCommit = nil
        copy.appVersion = nil
        copy.freezeForced = nil
        copy.forcedGatesSkipped = nil
        guard (try? load(name: newName)) == nil else {
            throw ExperimentError(reason: "experiment '\(newName)' already exists")
        }
        try save(copy, allowCreate: true)
        return copy
    }

    /// Confirmation draft from a screen study, built by ALLOWLIST — never
    /// by duplicating the whole manifest (P1 fix 2026-07-19, second pass:
    /// the duplicate-then-patch shortcut inherited the screen's EXECUTION
    /// state — its sweep, pipeline, conditions incl. the sweep-stamped
    /// recommendation, agent arms, promotion rule, and old perturbation
    /// policy — into a study that must preregister its own). The new draft
    /// carries ONLY the scientific pins; anything not listed below starts
    /// fresh, so a FUTURE manifest field defaults to NOT inherited.
    ///
    /// Field dispositions (every CodingKeys member decided explicitly):
    /// INHERITED (scientific pins):
    /// - model identity: `modelID`, `modelRevision`
    /// - concept recipes: `concepts`, `grandMeanCorpus`,
    ///   `neutralCorpusHash`, `markersHash`
    /// - generation settings: `promptMode`, `systemPrompt`,
    ///   `qwenThinkingEnabled`, `temperature`, `maxTokens`, `seeds`,
    ///   `samplesPerItem`, `seedPolicy`
    /// - measurement declarations: `outcomeInstruments`,
    ///   `ordinalAggregation`, `numericParser`+`parserRegistryHash`,
    ///   `exclusionRules`, `caseFamily`,
    ///   `reasoningStyleTaxonomyPath`+`Hash`, `capabilityBatteryFile`+`Hash`,
    ///   `judges`, `judgeRubricFile`+`Hash`, `evaluation`, `humanBaseline`,
    ///   `humanValidation`
    /// - protocol prose (describes the same task family and endpoints, not
    ///   execution state): `taskDescription`, `outcomeMeasures`
    /// - identity of THIS study: `studyType`=conceptStudy,
    ///   `studyKind`=modelOutput, `phase`="confirm",
    ///   `screenTaskPromptsHash` = the SCREEN's `taskPromptsHash` (the
    ///   held-out-pool rule's reference), `experimentDescription` derived
    ///   from the screen's.
    /// NOT inherited (fresh/empty — screen execution state or wrong-kind
    /// config):
    /// - lifecycle: `name` (new), `createdAt` (now), `status` (draft),
    ///   `frozenAt`/`freezeHash`/`frozenBy`/`gitCommit`/`appVersion`/
    ///   `freezeForced`/`forcedGatesSkipped` (nil)
    /// - item pool: `taskPromptsFile`/`taskPromptsHash` — confirm needs a
    ///   HELD-OUT pool; verify's disjointness rule must never be satisfied
    ///   by the inherited screen pool
    /// - screen execution machinery: `sweep`, `pipeline`, `conditions`
    ///   (ALL — including the sweep-stamped `<concept>-recommended`),
    ///   `variantConditions` (ALL — the confirm flow re-declares its agent
    ///   through the perturbation policy), `promotionRule` (the screen →
    ///   confirm gate already applied), `perturbationPolicy` (the OLD
    ///   policy; ConfirmationStudy.attach declares the new one)
    /// - `readerRefs`: fitted reader artifacts are substrate- and
    ///   phase-bound execution products; a confirm study re-pins them
    ///   deliberately
    /// - `acknowledgeUnequalOptionLengths`: acknowledges a property of the
    ///   SCREEN's option set; the held-out pool must earn its own
    /// - multi-agent config: `multiAgentScenarioPath`/`Hash` (nil),
    ///   `multiAgentIncludeBaseline` (default true) — other study kind.
    /// - `templateProvenance`: the confirm draft is minted from a SCREEN, not
    ///   from a study template; claiming the screen's template lineage would
    ///   make "reload as template" offer a study the template never described.
    @discardableResult
    public static func createConfirmationDraft(
        fromScreen screenName: String, named newName: String
    ) throws -> ExperimentManifest {
        let screen = try load(name: screenName)
        guard (try? load(name: newName)) == nil else {
            throw ExperimentError(reason: "experiment '\(newName)' already exists")
        }
        let description = screen.experimentDescription.isEmpty
            ? "Confirmation of '\(screenName)'"
            : "Confirmation of '\(screenName)' — \(screen.experimentDescription)"
        // The plain init IS the allowlist's floor: every field starts at
        // its fresh-draft default, then only the pins below are copied.
        var draft = ExperimentManifest(
            name: newName, description: description, modelID: screen.modelID)
        draft.modelRevision = screen.modelRevision
        draft.concepts = screen.concepts
        draft.grandMeanCorpus = screen.grandMeanCorpus
        draft.neutralCorpusHash = screen.neutralCorpusHash
        draft.markersHash = screen.markersHash
        draft.promptMode = screen.promptMode
        draft.systemPrompt = screen.systemPrompt
        draft.qwenThinkingEnabled = screen.qwenThinkingEnabled
        draft.temperature = screen.temperature
        draft.maxTokens = screen.maxTokens
        draft.seeds = screen.seeds
        draft.samplesPerItem = screen.samplesPerItem
        draft.seedPolicy = screen.seedPolicy
        draft.outcomeInstruments = screen.outcomeInstruments
        draft.ordinalAggregation = screen.ordinalAggregation
        draft.numericParser = screen.numericParser
        draft.parserRegistryHash = screen.parserRegistryHash
        // Exclusion rules are a measurement declaration: the confirm phase
        // inherits the screen's declared exclusions so the two phases
        // measure under the same instrument (change = duplicate + edit,
        // visible in the manifest diff).
        draft.exclusionRules = screen.exclusionRules
        draft.caseFamily = screen.caseFamily
        draft.reasoningStyleTaxonomyPath = screen.reasoningStyleTaxonomyPath
        draft.reasoningStyleTaxonomyHash = screen.reasoningStyleTaxonomyHash
        draft.capabilityBatteryFile = screen.capabilityBatteryFile
        draft.capabilityBatteryHash = screen.capabilityBatteryHash
        draft.judges = screen.judges
        draft.judgeRubricFile = screen.judgeRubricFile
        draft.judgeRubricHash = screen.judgeRubricHash
        draft.evaluation = screen.evaluation
        draft.humanBaseline = screen.humanBaseline
        draft.humanValidation = screen.humanValidation
        draft.taskDescription = screen.taskDescription
        draft.outcomeMeasures = screen.outcomeMeasures
        draft.studyType = StudyIntent.conceptStudy.rawValue
        draft.studyKind = StudyIntent.conceptStudy.mappedKind
        draft.phase = "confirm"
        draft.screenTaskPromptsHash = screen.taskPromptsHash
        try save(draft, allowCreate: true)
        return draft
    }

    private static func currentGitCommit() -> String? {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = [
            "-C", VectorCatalog.projectRoot.path, "rev-parse", "HEAD",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let commit = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return commit.isEmpty ? nil : commit
    }
}
