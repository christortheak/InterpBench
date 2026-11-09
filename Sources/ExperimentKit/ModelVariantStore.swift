import CryptoKit
import Foundation
import SteeringKit

public struct ModelVariantArtifact: Codable, Sendable, Equatable {
    public struct AdapterRef: Codable, Sendable, Equatable {
        public var name: String
        public var artifactPath: String
        public var adapterDirectory: String
        /// SHA-256 of the adapter WEIGHTS file alone — the established
        /// cross-engine contract (Swift's fine-tune panel and the server's
        /// `lora_train` both emit it this way).
        public var adapterHash: String?
        /// SHA-256 of `adapter_config.json` alone.
        ///
        /// Added 2026-08-16. Without it an agent verified its weights and
        /// silently skipped its configuration, so editing `adapter_config.json`
        /// changed what the forward pass does while the agent's declared
        /// identity stayed identical (external review round 5). `nil` is a
        /// legacy agent and is stamped "configuration unpinned" by consumers
        /// rather than read as "configuration verified".
        public var configHash: String?

        public init(
            name: String,
            artifactPath: String,
            adapterDirectory: String,
            adapterHash: String? = nil,
            configHash: String? = nil
        ) {
            self.name = name
            self.artifactPath = artifactPath
            self.adapterDirectory = adapterDirectory
            self.adapterHash = adapterHash
            self.configHash = configHash
        }

        enum CodingKeys: String, CodingKey {
            case name, artifactPath, adapterDirectory, adapterHash, configHash
        }

        /// Cross-engine tolerant decode (open-issues #11, 2026-08-18).
        ///
        /// The server's adapter entries are opaque dicts carried verbatim
        /// through `ModelVariant.to_dict()`, and the server-minted shape is
        /// `{adapterDirectory, adapterHash, configHash}` — no `name`, often no
        /// `artifactPath`. Every server-side reader already tolerates that
        /// (`adapter.get('name') or adapter.get('adapterDirectory')`), but this
        /// decoder REQUIRED both keys, so `steerlab-cli experiment verify`
        /// failed `DecodingError.keyNotFound: 'name'` at
        /// `variantConditions[0].artifact.adapters[0]` on every study whose
        /// adapter agent came from the server/driver path — a study the server
        /// CLI verified clean. A missing name is a fact about the writer, not
        /// a corrupt artifact: derive it from the adapter directory's basename
        /// (the same label the server's `/api/adapters` listing mints) rather
        /// than refuse the whole manifest.
        ///
        /// Encoding stays synthesized, and `artifactHash` is taken over FILE
        /// BYTES (`ModelVariantStore.hash`), never over a re-encode — so a
        /// derived name cannot move a pinned hash.
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let declaredDirectory = try container.decodeIfPresent(
                String.self, forKey: .adapterDirectory)
            let declaredPath = try container.decodeIfPresent(
                String.self, forKey: .artifactPath)
            adapterDirectory = declaredDirectory ?? declaredPath ?? ""
            artifactPath = declaredPath ?? adapterDirectory
            name = try container.decodeIfPresent(String.self, forKey: .name)
                ?? Self.derivedName(adapterDirectory: adapterDirectory)
            adapterHash = try container.decodeIfPresent(String.self, forKey: .adapterHash)
            configHash = try container.decodeIfPresent(String.self, forKey: .configHash)
        }

        /// The label for an adapter entry that declared none: the adapter
        /// directory's last path component, with any trailing slash ignored.
        /// "adapter" when there is nothing to derive from — a name is a
        /// display label here, and an empty one reads as a missing field in
        /// every UI that shows it.
        static func derivedName(adapterDirectory: String) -> String {
            let trimmed = adapterDirectory.hasSuffix("/")
                ? String(adapterDirectory.dropLast())
                : adapterDirectory
            let basename = trimmed.split(separator: "/").last.map(String.init) ?? ""
            return basename.isEmpty ? "adapter" : basename
        }
    }

    public struct InjectionRef: Codable, Sendable, Equatable {
        public var concept: String
        public var vectorArtifactID: String
        public var layer: Int
        /// α when steering, λ when ablating.
        public var alpha: Double
        /// How this edit acts: `add` (steer — `h + α·v`) or `ablate`
        /// (`h − λ·(h·v̂)v̂`).
        ///
        /// OPTIONAL and absent-means-`add` (2026-07-27): every artifact
        /// written before ablation existed decodes unchanged and keeps its
        /// exact behaviour. Encoded only when it is not the default, so a
        /// steering agent's JSON is byte-identical to what it was — which
        /// matters because those bytes are hashed into promotion keys and
        /// artifact hashes.
        public var mode: InterventionPlan.Mode?

        /// Which layers an ABLATION covers. nil = every layer, which is the
        /// default and what the UI offers.
        ///
        /// Kept in the schema even though no control exposes it: ablating a
        /// narrow band is how you ask WHERE a concept is used, and having the
        /// field now means that question costs a CLI flag later instead of an
        /// artifact migration. Ignored when steering, which uses `layer` plus
        /// the variant's band width.
        public var layers: [Int]?

        /// Ablation-direction centering — declared per injection, never
        /// silent (cross-engine contract key `"centering"`; server twin
        /// `model_variant.variant_injections`). `"neutralMean"` projects the
        /// vector artifact's STORED neutral residual mean out of the
        /// direction (`v − (v·m̂)m̂`) before ablating: extracted vectors
        /// routinely share a large component with that mean, and ablating it
        /// raw at λ=1 collapses generation into single-token repetition
        /// (2026-08-06 collapse study). Absent means "none" — the raw
        /// direction, with the mean-alignment preflight warning instead —
        /// and, like `mode`, is never encoded so pre-existing artifact bytes
        /// and hashes are untouched. Ignored (refused at build) for
        /// steering injections.
        public var centering: String?

        /// Spelled out for call sites where the contextual base is not
        /// inferrable (a literal inside a generic argument list).
        public static let ablateMode: InterventionPlan.Mode = .ablate

        /// The mode in force, resolving the absent default.
        public var effectiveMode: InterventionPlan.Mode { mode ?? .add }

        /// The centering in force, resolving the absent default.
        public var effectiveCentering: String { centering ?? "none" }

        public init(
            concept: String,
            vectorArtifactID: String,
            layer: Int,
            alpha: Double,
            mode: InterventionPlan.Mode? = nil,
            layers: [Int]? = nil,
            centering: String? = nil
        ) {
            self.concept = concept
            self.vectorArtifactID = vectorArtifactID
            self.layer = layer
            self.alpha = alpha
            self.mode = mode
            self.layers = layers
            self.centering = centering
        }

        /// The declared ablation layers as both engines read them: sorted,
        /// each layer once, or nil when none are declared (every layer).
        ///
        /// The list is a SET of blocks. `[3, 7, 10]` ablates exactly those
        /// three and never the blocks between them, and neither the order it
        /// was written in nor a repeat changes what is removed. Server twin:
        /// `model_variant.variant_injections`, which ablates the listed
        /// layers and nothing else.
        public var declaredAblationLayers: [Int]? {
            guard let layers, !layers.isEmpty else { return nil }
            return Array(Set(layers)).sorted()
        }

        /// Why this ablation cannot run on a vector of `layerCount` layers,
        /// or nil when it can: it lists a layer the vector does not have.
        ///
        /// Refused, never skipped. Skipping ran a different study from the
        /// one declared (and preregistered), and a list naming only such
        /// layers used to ablate every layer here and none on the server.
        /// One rule and one sentence on both engines (server twin:
        /// `ablation_layers.covered`).
        public func ablationLayerProblem(layerCount: Int) -> String? {
            guard effectiveMode == .ablate, let declared = declaredAblationLayers
            else { return nil }
            let outside = declared.filter { !(0..<layerCount).contains($0) }
            guard !outside.isEmpty else { return nil }
            let one = outside.count == 1
            return "the ablation of '\(concept)' lists \(one ? "layer" : "layers") "
                + outside.map(String.init).joined(separator: ", ")
                + ", which its vector does not have (its layers are 0 to "
                + "\(layerCount - 1)); remove \(one ? "it" : "them") from the "
                + "injection's layers, or delete the list to ablate every layer; "
                + "in a frozen study, save the corrected agent as a new agent, "
                + "duplicate the study, attach the new agent to the copy, then "
                + "validate and freeze it"
        }

        /// The layers this ref actually covers, given the model's depth.
        /// Steering answers with its single declared layer (the caller
        /// widens it by the variant's band); ablation answers with every
        /// layer unless it declared otherwise — a single-layer ablation is
        /// usually undone by the layers above it, so "all" is the honest
        /// default rather than a convenience. A declared list answers with
        /// exactly its in-range layers, ascending (`declaredAblationLayers`);
        /// a caller that runs the ablation refuses first when the list names
        /// a layer the vector does not have (`ablationLayerProblem`).
        public func resolvedLayers(layerCount: Int) -> [Int] {
            guard effectiveMode == .ablate else { return [layer] }
            if let declared = declaredAblationLayers {
                return declared.filter { (0..<layerCount).contains($0) }
            }
            return Array(0..<layerCount)
        }

        enum CodingKeys: String, CodingKey {
            case concept, vectorArtifactID, layer, alpha, mode, layers, centering
        }

        public func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(concept, forKey: .concept)
            try container.encode(vectorArtifactID, forKey: .vectorArtifactID)
            try container.encode(layer, forKey: .layer)
            try container.encode(alpha, forKey: .alpha)
            // Only when non-default: see `mode`.
            if let mode, mode != .add {
                try container.encode(mode, forKey: .mode)
            }
            if let layers, !layers.isEmpty {
                try container.encode(layers, forKey: .layers)
            }
            // Only when non-default: see `centering`.
            if let centering, centering != "none" {
                try container.encode(centering, forKey: .centering)
            }
        }
    }

    /// True when every injection ABLATES — an agent that removes concepts
    /// rather than steering them.
    ///
    /// Such an agent has no sweep-selected cell by construction: ablation has
    /// no layer×alpha grid, so there is nothing to select and nothing to
    /// promote. Gates and advisories that ask "was this promoted?" must not
    /// read its absence as hand-tuning.
    public var ablatesOnly: Bool {
        !injections.isEmpty && injections.allSatisfy { $0.effectiveMode == .ablate }
    }

    /// True when any injection ablates — the flag a UI uses to describe a
    /// mixed agent honestly.
    public var hasAblation: Bool {
        injections.contains { $0.effectiveMode == .ablate }
    }

    /// Sweep-promotion birth certificate (cross-engine contract with the
    /// server's `promote.py`): present only on agents minted by `promote`
    /// from a sweep-selected cell. Absent = hand-created variant — legal,
    /// but permanently distinguishable from an agent whose settings were
    /// selected on dev data by a predeclared rule.
    public struct Promotion: Codable, Sendable, Equatable {
        /// MIRRORED-POLE inheritance (cross-engine contract key
        /// `promotion.poleProvenance`; server twin:
        /// `promote._pole_provenance`): the injected direction is the
        /// NEGATION of another concept's — a mirrored pole (`PoleMirror`),
        /// whose stimuli are the source concept's two files with their
        /// positive/negative roles exchanged.
        ///
        /// Without this record an agent promoted from a mirrored direction
        /// never announced it injects a negated direction: the sidecar's
        /// `polesSwappedFromSource`/`negatedFrom` stamps and the manifest
        /// pin's mirror linkage all stopped one step short of the birth
        /// certificate, and the certificate is the one artifact an agent
        /// carries everywhere. Every field is inherited from PROVABLE
        /// evidence only — the matched artifact's own sidecar, the manifest's
        /// hash-checked artifact pin, or the pinned sidecar's bytes after
        /// re-verifying them against the pin — never inferred from names.
        /// Absent = the promotion's evidence carries no machine-readable
        /// swap claim (including a fresh extraction over role-swapped files,
        /// where the swap lives only in PROVENANCE.md prose: minting the
        /// mirror, or attaching the minted artifact, is what makes the claim
        /// provable).
        public struct PoleProvenance: Codable, Sendable, Equatable {
            /// Always `true` when the record exists — the same never-false
            /// idiom as the sidecar stamp it inherits.
            public var polesSwappedFromSource: Bool
            /// The SOURCE pole's concept name, from the (verified) mirror
            /// sidecar's `negatedFrom.concept`. Absent when only the
            /// manifest pin's facts were provable.
            public var sourceConcept: String?
            /// The SOURCE concept's order-sensitive stimulus hash
            /// (`sha256(positive ‖ negative)` of the parent's files) — the
            /// qualified hash the mirror sidecar carries verbatim and the
            /// manifest pin records as `sourceStimulusSetHash`.
            public var sourceStimulusSetHash: String?
            /// SHA-256 of the SOURCE artifact's tensor bytes (the direction
            /// this agent's injection negates), from `negatedFrom`.
            public var sourceTensorHash: String?

            public init(
                polesSwappedFromSource: Bool,
                sourceConcept: String? = nil,
                sourceStimulusSetHash: String? = nil,
                sourceTensorHash: String? = nil
            ) {
                self.polesSwappedFromSource = polesSwappedFromSource
                self.sourceConcept = sourceConcept
                self.sourceStimulusSetHash = sourceStimulusSetHash
                self.sourceTensorHash = sourceTensorHash
            }
        }

        public var experiment: String
        /// Manifest content hash at promote time.
        public var experimentHash: String
        public var promotedAt: String
        /// "criterion" (the declared selection picked this cell) or
        /// "manualOverride" (--cell bypassed it, loudly).
        public var promotedBy: String
        /// Why the declared selection was bypassed — stamped only on
        /// manualOverride promotions, so the birth certificate documents the
        /// deviation instead of merely flagging it.
        public var overrideReason: String?
        public var sweepRun: String?
        public var criterion: ExperimentManifest.SweepSelection?
        public var devPromptsHash: String?
        public var winningCell: ExperimentManifest.SelectionProvenance.Cell?
        public var metrics: [String: Double]?
        public var control: ExperimentManifest.SelectionProvenance.Control?
        /// What the sweep CONCLUDED when it declined to recommend — the
        /// failure message from `recommendations.json` (e.g. "no cell passed
        /// the capability/coherence gates"). Stamped only on manualOverride
        /// promotions whose sweep evidence was a failure entry; absent
        /// everywhere else.
        public var selectionOutcome: String?
        /// The canonical full-recipe identity (`RecipeIdentity`) the matched
        /// vector artifact satisfied at promote time — the birth certificate
        /// proves WHICH recipe the injected vector embodies, not just which
        /// artifact file was newest. Absent on agents promoted before the
        /// identity contract existed (2026-07-13).
        public var recipeIdentityHash: String?
        /// SHA-256 of the injected vector's tensor bytes at promote time —
        /// WHICH bytes, not merely which recipe they claim. Absent on agents
        /// promoted before the pinned contract (B2, 2026-07-26).
        public var vectorArtifactHash: String?
        /// Deterministic identity of the promotion REQUEST — the idempotency
        /// key. Two promotions agreeing on experiment, epoch, concept, sweep
        /// run, cell, artifact, mode and agent name are the SAME promotion,
        /// so a retried stage returns the existing agent instead of minting a
        /// duplicate that then competes with it in every picker. Deliberately
        /// excludes the timestamp: a retry is the same promotion even though
        /// it happens later. Cross-engine twin of `promote.promotion_key`.
        public var promotionKey: String?
        /// The qualification artifact this promotion CITES, carried VERBATIM
        /// (cross-engine key `promotion.qualification` =
        /// `{path, contentHash, decision}`; server twin
        /// `sae_qualification.citation`).
        ///
        /// A new vector family (SAE imports, OptVec directions) is seated
        /// through sweep → promote like every other family, and its
        /// qualification evidence — held-out probe movement, leakage,
        /// discriminants, dose response — is something the birth certificate
        /// CITES rather than replaces. This engine never mints the block
        /// (qualification runs on the server), but the Agent Library edits
        /// and re-saves agent artifacts in place: without a passthrough,
        /// `update` would rewrite a server-promoted agent MINUS its citation,
        /// leaving a promotion that silently claims no evidence. That is
        /// evidence destruction, not a lost label — hence opaque JSON, so the
        /// server can extend the citation without this engine learning keys.
        public var qualification: JSONValue?
        /// See `PoleProvenance`. Deliberately OUTSIDE `promotionKey` (the
        /// `qualification` reasoning): the key is the cross-engine identity
        /// of the promotion REQUEST, and this record is additional evidence
        /// about an already-identified promotion, not a different promotion.
        /// Absent on every certificate minted before the record existed and
        /// on every non-mirrored promotion — decodable-absent, and never
        /// encoded when nil, so legacy certificate bytes are untouched.
        public var poleProvenance: PoleProvenance?
        public var substrate: String
        public var appVersion: String

        public init(
            experiment: String,
            experimentHash: String,
            promotedAt: String,
            promotedBy: String,
            overrideReason: String? = nil,
            sweepRun: String? = nil,
            criterion: ExperimentManifest.SweepSelection? = nil,
            devPromptsHash: String? = nil,
            winningCell: ExperimentManifest.SelectionProvenance.Cell? = nil,
            metrics: [String: Double]? = nil,
            control: ExperimentManifest.SelectionProvenance.Control? = nil,
            selectionOutcome: String? = nil,
            recipeIdentityHash: String? = nil,
            substrate: String,
            appVersion: String,
            vectorArtifactHash: String? = nil,
            promotionKey: String? = nil,
            qualification: JSONValue? = nil,
            poleProvenance: PoleProvenance? = nil
        ) {
            self.experiment = experiment
            self.experimentHash = experimentHash
            self.promotedAt = promotedAt
            self.promotedBy = promotedBy
            self.overrideReason = overrideReason
            self.sweepRun = sweepRun
            self.criterion = criterion
            self.devPromptsHash = devPromptsHash
            self.winningCell = winningCell
            self.metrics = metrics
            self.control = control
            self.selectionOutcome = selectionOutcome
            self.recipeIdentityHash = recipeIdentityHash
            self.substrate = substrate
            self.appVersion = appVersion
            self.vectorArtifactHash = vectorArtifactHash
            self.promotionKey = promotionKey
            self.qualification = qualification
            self.poleProvenance = poleProvenance
        }
    }

    public var schemaVersion: Int
    public var name: String
    public var baseModelID: String
    public var baseRevision: String?
    public var adapters: [AdapterRef]
    public var injections: [InjectionRef]
    public var bandWidth: Int
    public var alphaInNormUnits: Bool
    public var neutralPCBasisPath: String?
    public var neutralPCBasisLabel: String?
    public var promptMode: String
    public var qwenThinkingEnabled: Bool
    public var temperature: Double
    public var systemPromptHash: String?
    public var systemPrompt: String?
    public var createdAt: String
    public var promotion: Promotion?
    public var interventionPolicies: [JSONValue]?
    /// How hard the agent thinks (public issue #16): the study's own
    /// vocabulary (`ReasoningEffort`), kept on the agent. nil is every agent
    /// file written before the field existed, and every file whose writer
    /// had no level to state: `qwenThinkingEnabled` then answers, as it
    /// always has (true is the template's default effort, false is off).
    /// Read it through `resolvedReasoningEffort`. Encoded only when present,
    /// so every existing agent's bytes and hash are unchanged. Server twin:
    /// `ModelVariant.reasoning_effort`.
    public var reasoningEffort: String?
    /// The record of an agent a researcher made by hand with `agent create`
    /// (public issue #17): the command, its inputs and their hashes, when it
    /// was made, and the stated reason. Kept beside `promotion`; an agent
    /// carries at most one of the two. Opaque JSON carried verbatim, like a
    /// promotion's `qualification`, so either engine's writer can extend it
    /// without the other learning keys.
    public var handmade: JSONValue?
    /// Why the file this artifact was decoded from cannot be read as one
    /// agent (`fileProblem(effortValue:…)`), recorded rather than thrown, so a
    /// study whose inline agent has such a problem still DECODES and its
    /// `verify` names the problem, as the server's does. Never encoded; nil
    /// for an artifact built in code. Read it through `fileProblem`.
    public internal(set) var fileProblemOnRead: String? = nil

    public init(
        name: String,
        baseModelID: String,
        baseRevision: String? = nil,
        adapters: [AdapterRef] = [],
        injections: [InjectionRef] = [],
        bandWidth: Int = 1,
        alphaInNormUnits: Bool = true,
        neutralPCBasisPath: String? = nil,
        neutralPCBasisLabel: String? = nil,
        promptMode: String,
        qwenThinkingEnabled: Bool,
        temperature: Double,
        systemPrompt: String,
        createdAt: Date = Date(),
        promotion: Promotion? = nil,
        interventionPolicies: [JSONValue]? = nil,
        reasoningEffort: ReasoningEffort? = nil,
        handmade: JSONValue? = nil
    ) {
        self.schemaVersion = 1
        self.name = name
        self.baseModelID = baseModelID
        self.baseRevision = baseRevision
        self.adapters = adapters
        self.injections = injections
        self.bandWidth = bandWidth
        self.alphaInNormUnits = alphaInNormUnits
        self.neutralPCBasisPath = neutralPCBasisPath
        self.neutralPCBasisLabel = neutralPCBasisLabel
        self.promptMode = promptMode
        // A stated level decides the boolean, which is written as its shadow
        // so the two fields agree; `off` is written as the boolean alone,
        // byte-identical to an agent saved before the level existed.
        if let reasoningEffort {
            self.qwenThinkingEnabled = reasoningEffort.isOn
            self.reasoningEffort = reasoningEffort.isOn ? reasoningEffort.rawValue : nil
        } else {
            self.qwenThinkingEnabled = qwenThinkingEnabled
            self.reasoningEffort = nil
        }
        self.handmade = handmade
        self.temperature = temperature
        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        self.systemPrompt = trimmedSystem.isEmpty ? nil : systemPrompt
        self.systemPromptHash = trimmedSystem.isEmpty
            ? nil
            : Self.sha256(systemPrompt)
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
        self.promotion = promotion
        self.interventionPolicies = interventionPolicies
    }

    private static func sha256(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion, name, baseModelID, baseRevision, adapters, injections
        case bandWidth, alphaInNormUnits, neutralPCBasisPath, neutralPCBasisLabel
        case promptMode, qwenThinkingEnabled, temperature
        case systemPromptHash, systemPrompt, createdAt, promotion, interventionPolicies
        case reasoningEffort, handmade
    }

    /// Tolerant decoding for server payloads: the Python server's
    /// `variant.to_dict()` omits `createdAt` when it never had one (and
    /// `neutralPCBasisLabel` always — already optional here). Local artifacts
    /// always carry both stamps, so this only relaxes cross-engine reads.
    /// Encoding stays synthesized — the upload/inline wire schema is unchanged.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        name = try container.decode(String.self, forKey: .name)
        baseModelID = try container.decode(String.self, forKey: .baseModelID)
        baseRevision = try container.decodeIfPresent(String.self, forKey: .baseRevision)
        adapters = try container.decodeIfPresent([AdapterRef].self, forKey: .adapters) ?? []
        injections = try container.decodeIfPresent([InjectionRef].self, forKey: .injections) ?? []
        bandWidth = try container.decodeIfPresent(Int.self, forKey: .bandWidth) ?? 1
        // Cross-engine decode rule (mirrored by the Python server's
        // `ModelVariant.from_dict`): an ABSENT flag reads as RAW alpha — the
        // conservative literal α·v meaning for legacy artifacts that predate
        // norm capture. Same bytes must inject the same on both engines; both
        // encoders always WRITE the key explicitly, which makes this safe.
        alphaInNormUnits =
            try container.decodeIfPresent(Bool.self, forKey: .alphaInNormUnits) ?? false
        neutralPCBasisPath = try container.decodeIfPresent(
            String.self, forKey: .neutralPCBasisPath)
        neutralPCBasisLabel = try container.decodeIfPresent(
            String.self, forKey: .neutralPCBasisLabel)
        promptMode =
            try container.decodeIfPresent(String.self, forKey: .promptMode) ?? "chatAssistant"
        let declaredThinking = try container.decodeIfPresent(Bool.self, forKey: .qwenThinkingEnabled)
        // Read as any JSON value, so a non-string level gets the same
        // sentence as the server's, rather than a decoding error.
        let effortValue = try container.decodeIfPresent(JSONValue.self, forKey: .reasoningEffort)
        if case .string(let spelled)? = effortValue { reasoningEffort = spelled } else { reasoningEffort = nil }
        handmade = try container.decodeIfPresent(JSONValue.self, forKey: .handmade)
        // The boolean is the effort's shadow: a file that spells only the
        // effort thinks exactly when the effort is not off.
        qwenThinkingEnabled = declaredThinking
            ?? reasoningEffort.map { $0 != ReasoningEffort.off.rawValue } ?? false
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        systemPromptHash = try container.decodeIfPresent(String.self, forKey: .systemPromptHash)
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
        promotion = try container.decodeIfPresent(Promotion.self, forKey: .promotion)
        interventionPolicies = try container.decodeIfPresent([JSONValue].self, forKey: .interventionPolicies)
        try InterventionPolicyLibrary.validateAttachments(interventionPolicies)
        fileProblemOnRead = Self.fileProblem(
            effortValue: effortValue == .null ? nil : effortValue,
            qwenThinkingEnabled: declaredThinking,
            hasPromotion: promotion != nil, hasHandmade: handmade?.agentRecordObject != nil)
    }
}

/// An agent file whose fields cannot be read as one agent
/// (`ModelVariantArtifact.fileProblem`), as the readers that refuse it throw
/// it: the same sentence everywhere.
public struct AgentFileProblem: Error, CustomStringConvertible, LocalizedError, Sendable {
    public let reason: String
    public var description: String { reason }
    public var errorDescription: String? { reason }
}

extension ModelVariantArtifact {

    /// The two ways an agent comes to exist, as every surface names them.
    /// `swept`: the file carries a `promotion` record, so a sweep selected its
    /// settings and `experiment promote` minted it. `handmade`: it carries
    /// none, so a researcher chose its settings — with `agent create`, which
    /// leaves a `handmade` record, or in the app's agent editor, which leaves
    /// none. Server twin: `agent_identity.AGENT_KIND_SWEPT`/`_HANDMADE`.
    public enum Kind: String, Codable, Sendable {
        case swept, handmade
    }

    public var kind: Kind { promotion != nil ? .swept : .handmade }

    /// Why an agent file cannot be read as one agent, or nil: its
    /// `reasoningEffort` is in the study's vocabulary and agrees with its
    /// legacy `qwenThinkingEnabled` when both are spelled; and it carries at
    /// most one of `promotion` and `handmade`. The sentences are the
    /// server's (`agent_identity.agent_file_problem`), byte for byte.
    public static func fileProblem(
        reasoningEffort: String?, qwenThinkingEnabled: Bool?,
        hasPromotion: Bool, hasHandmade: Bool
    ) -> String? {
        fileProblem(
            effortValue: reasoningEffort.map(JSONValue.string),
            qwenThinkingEnabled: qwenThinkingEnabled,
            hasPromotion: hasPromotion, hasHandmade: hasHandmade)
    }

    /// `fileProblem` for the level as the file spells it, any JSON value: a
    /// non-string level is named in the server's own spelling
    /// (`json.dumps`), so both engines say the same sentence.
    public static func fileProblem(
        effortValue: JSONValue?, qwenThinkingEnabled: Bool?,
        hasPromotion: Bool, hasHandmade: Bool
    ) -> String? {
        if let effortValue {
            guard case .string(let effort) = effortValue,
                let parsed = ReasoningEffort(rawValue: effort)
            else {
                return "This agent file's reasoningEffort \(Self.pythonJSONText(effortValue)) "
                    + "is not a level SteerLab knows (off, on, low, medium, high, xhigh). "
                    + "Make the agent again with agent create --reasoning-effort, or in the "
                    + "app's agent editor."
            }
            if let thinking = qwenThinkingEnabled, thinking != parsed.isOn {
                return "This agent file says reasoningEffort \"\(effort)\" but "
                    + "qwenThinkingEnabled \(thinking): one says the agent thinks and the "
                    + "other says it does not. Make the agent again with agent create "
                    + "--reasoning-effort, or in the app's agent editor, so both fields "
                    + "say the same thing."
            }
        }
        if hasPromotion, hasHandmade {
            return "This agent file carries both a promotion record (a sweep chose its "
                + "settings) and a handmade record (a researcher chose them). An agent is "
                + "made one way: make it again with agent create or experiment promote."
        }
        return nil
    }

    /// Why this agent cannot be read as one agent, or nil: the problem its
    /// file had when it was read, else its current fields' (an artifact built
    /// or edited in code).
    public var fileProblem: String? {
        fileProblemOnRead
            ?? Self.fileProblem(
                reasoningEffort: reasoningEffort, qwenThinkingEnabled: qwenThinkingEnabled,
                hasPromotion: promotion != nil, hasHandmade: handmade?.agentRecordObject != nil)
    }

    /// A JSON value spelled as Python's `json.dumps` spells it, for the
    /// shared sentences that quote one.
    static func pythonJSONText(_ value: JSONValue) -> String {
        switch value {
        case .string(let text):
            return "\"" + text.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"") + "\""
        case .number(let number):
            if number.rounded() == number, abs(number) < 1e15 { return String(Int64(number)) }
            return String(number)
        case .bool(let flag): return flag ? "true" : "false"
        case .null: return "null"
        case .array(let items): return "[" + items.map(pythonJSONText).joined(separator: ", ") + "]"
        case .object(let members):
            return "{" + members.keys.sorted().map {
                "\"\($0)\": " + pythonJSONText(members[$0] ?? .null)
            }.joined(separator: ", ") + "}"
        }
    }

    /// How hard this agent thinks, wherever it is cast — the ONE reading:
    /// its declared level, else the legacy boolean's meaning (true: the
    /// template's default effort, xhigh; false: off). Ordinary studies read
    /// it through `StudySampling.reasoningEffort(_:variant:)`; a panel seat
    /// backed by an agent reads it where the panel runtime resolves the
    /// seat's thinking. Server twin: `agent_identity.variant_reasoning_effort`.
    public var resolvedReasoningEffort: ReasoningEffort {
        reasoningEffort.flatMap(ReasoningEffort.init(rawValue:))
            ?? ReasoningEffort.legacy(qwenThinkingEnabled: qwenThinkingEnabled)
    }

    /// The template rule a study's `set-sampling --reasoning-effort` applies,
    /// for this agent's DECLARED effort: a non-off effort needs a thinking
    /// switch, and a level must be one the model's chat template accepts.
    /// The reasoning budget belongs to the study that casts the agent, so it
    /// is not checked here; a legacy agent with no declared effort is not
    /// judged anew. Server twin: `agent_identity.variant_effort_violations`.
    public func effortViolations(capabilities: ModelCapabilities? = nil) -> [String] {
        guard let declared = reasoningEffort.flatMap(ReasoningEffort.init(rawValue:)),
            declared.isOn
        else { return [] }
        let record = PromptRendering.capabilities(for: baseModelID, explicit: capabilities)
        guard record.hasThinkingSwitch else {
            return [ReasoningEffort.effortWithoutThinkingModeReason(declared, modelID: baseModelID)]
        }
        guard declared.isLevel else { return [] }
        return ReasoningEffort.effortLevelViolations(
            declared, modelID: baseModelID, capabilities: record)
    }

    /// The levels an agent editor offers for a model: off always; on and the
    /// levels the model's chat template accepts (or, with no probed record,
    /// is assumed to accept) when it has a thinking switch. A DECLARED level
    /// the template rejects or ignores is never offered; the reading of an
    /// older file's thinking switch (`currentDeclared` false) is, so such an
    /// agent shows what it is and saves unchanged. In the vocabulary's order.
    public static func offeredReasoningEfforts(
        capabilities: ModelCapabilities, current: ReasoningEffort,
        currentDeclared: Bool = true
    ) -> [ReasoningEffort] {
        var offered: Set<ReasoningEffort> = [.off]
        if !currentDeclared { offered.insert(current) }
        if capabilities.hasThinkingSwitch {
            offered.insert(.on)
            for level in ReasoningEffort.levels {
                switch capabilities.effortVerdict(level.rawValue) {
                case .accepted?, .assumed?: offered.insert(level)
                default: break
                }
            }
        }
        return ReasoningEffort.allCases.filter(offered.contains)
    }

    /// What every surface says about an agent, from the file alone: how it
    /// was made and how hard it thinks. Same keys and sentences as the
    /// server's `agent_identity.agent_origin`.
    public struct Origin: Codable, Sendable, Equatable {
        public var kind: Kind
        public var summary: String
        public var reasoningEffort: String
        /// `agent` when the file declares its level; `thinkingSwitch` when the
        /// level is read from the legacy `qwenThinkingEnabled`.
        public var reasoningEffortSource: String
    }

    public var origin: Origin {
        let declared = reasoningEffort.flatMap(ReasoningEffort.init(rawValue:)) != nil
        return Origin(
            kind: kind, summary: originSummary,
            reasoningEffort: resolvedReasoningEffort.rawValue,
            reasoningEffortSource: declared ? "agent" : "thinkingSwitch")
    }

    private var originSummary: String {
        if let promotion {
            let from = promotion.experiment.isEmpty ? "" : " from study \(promotion.experiment)"
            let how: String
            switch promotion.promotedBy {
            case "manualOverride": how = ", using a cell the researcher picked within the sweep."
            case "criterion": how = ", using the cell the study's declared rule selected."
            default: how = "."
            }
            return "A sweep chose this agent's settings: experiment promote made it" + from + how
        }
        guard let record = handmade?.agentRecordObject else {
            return "This agent carries no promotion record, so nothing records that a sweep "
                + "selected its settings, and no record of how it was made (agents saved "
                + "before such records existed, and new agent versions, carry none)."
        }
        let steered = !injections.isEmpty || !(interventionPolicies ?? []).isEmpty
        let chose = steered
            ? "The researcher chose its steering settings; no sweep selected them."
            : "The researcher chose its settings."
        let madeWith = record["madeWith"]?.agentRecordString.flatMap { $0.isEmpty ? nil : $0 }
        let reason = record["reason"]?.agentRecordString.flatMap { $0.isEmpty ? nil : $0 }
        let edited = record["editedFrom"]?.agentRecordObject
        return Self.madeWithPhrase(madeWith)
            + (edited.map { Self.editedFromPhrase($0, changed: record["changed"]) } ?? chose)
            + (reason.map { " Stated reason: \"\($0)\"." } ?? " No reason was stated.")
    }

    /// How each agent field an edit can change reads in a summary, in the
    /// order a summary lists them. Server twin:
    /// `agent_identity.EDITABLE_FIELD_WORDS`.
    public static let editableFieldWords: [(field: String, words: String)] = [
        ("baseModelID", "base model"), ("baseRevision", "model revision"),
        ("adapters", "adapter"), ("injections", "steering"),
        ("bandWidth", "layer band"), ("alphaInNormUnits", "strength units"),
        ("neutralPCBasisPath", "neutral basis"), ("interventionPolicies", "intervention policies"),
        ("promptMode", "prompt mode"), ("reasoningEffort", "thinking level"),
        ("systemPrompt", "persona"), ("temperature", "Playground temperature"),
    ]

    /// Words in a sentence, with the Oxford comma.
    static func listed(_ words: [String]) -> String {
        guard words.count > 2 else { return words.joined(separator: " and ") }
        return words.dropLast().joined(separator: ", ") + ", and " + words[words.count - 1]
    }

    /// What an editor record says of the agent it was edited from
    /// (`editedFrom`) and the fields the researcher changed. Server twin:
    /// `agent_identity.edited_from_phrase`, the same sentence.
    static func editedFromPhrase(_ edited: [String: JSONValue], changed: JSONValue?) -> String {
        let source: String
        switch edited["kind"]?.agentRecordString {
        case Kind.swept.rawValue?:
            let study = edited["experiment"]?.agentRecordString.flatMap { $0.isEmpty ? nil : $0 }
            let run = edited["sweepRun"]?.agentRecordString.flatMap { $0.isEmpty ? nil : $0 }
            source = "an agent experiment promote made"
                + (study.map { " from study \($0)" } ?? "")
                + (run.map { " (sweep run \($0))" } ?? "")
        case Kind.handmade.rawValue? where edited["handmade"]?.agentRecordObject != nil:
            source = "an agent made by hand"
        default:
            source = "an agent with no record of how it was made"
        }
        var named: Set<String> = []
        if case .array(let items)? = changed {
            named = Set(items.compactMap(\.agentRecordString))
        }
        let fields = editableFieldWords.filter { named.contains($0.field) }.map(\.words)
        return "The researcher edited it from \(source)"
            + (fields.isEmpty ? "." : " and changed its \(listed(fields)).")
    }

    /// The fields of this agent that differ from `source`, by name, in the
    /// order `editableFieldWords` lists them: what an editor record says the
    /// researcher changed. Compared as `AgentEditProvenance.doesTheSame`
    /// compares, field by field.
    public func changedFields(from source: ModelVariantArtifact) -> [String] {
        let a = AgentEditProvenance.behaviour(self)
        let b = AgentEditProvenance.behaviour(source)
        var changed: Set<String> = []
        if a.baseModelID != b.baseModelID { changed.insert("baseModelID") }
        if a.baseRevision ?? "" != b.baseRevision ?? "" { changed.insert("baseRevision") }
        if a.adapters != b.adapters { changed.insert("adapters") }
        if a.injections != b.injections { changed.insert("injections") }
        if a.bandWidth != b.bandWidth { changed.insert("bandWidth") }
        if a.alphaInNormUnits != b.alphaInNormUnits { changed.insert("alphaInNormUnits") }
        if a.neutralPCBasisPath ?? "" != b.neutralPCBasisPath ?? "" {
            changed.insert("neutralPCBasisPath")
        }
        if a.interventionPolicies ?? [] != b.interventionPolicies ?? [] {
            changed.insert("interventionPolicies")
        }
        if a.promptMode != b.promptMode { changed.insert("promptMode") }
        if a.reasoningEffort != b.reasoningEffort { changed.insert("reasoningEffort") }
        if (systemPrompt ?? "") != (source.systemPrompt ?? "") { changed.insert("systemPrompt") }
        if a.temperature != b.temperature { changed.insert("temperature") }
        return Self.editableFieldWords.map(\.field).filter(changed.contains)
    }

    /// A handmade record's `madeWith` values. Server twin:
    /// `agent_identity.MADE_WITH_AGENT_CREATE` / `MADE_WITH_AGENT_EDITOR`.
    public static let madeWithAgentCreate = "agent create"
    public static let madeWithAgentEditor = "agent editor"

    /// The opening of a handmade agent's summary. Server twin:
    /// `agent_identity.made_with_phrase`.
    static func madeWithPhrase(_ madeWith: String?) -> String {
        if madeWith == madeWithAgentEditor { return "Made by hand in the app's agent editor. " }
        if let madeWith { return "Made by hand with \(madeWith). " }
        return "Made by hand. "
    }

    /// The record the app's agent editor writes on an agent it saves that
    /// carries neither record (a new agent, or an edited one whose record no
    /// longer described it): who made it and when, with no paths. An agent
    /// saved from an edited one also records the agent it was edited from
    /// (`editedFrom`, `editedFromRecord`) and the fields the researcher
    /// changed, so its summary never says no sweep chose settings a sweep did.
    public static func editorRecord(
        at date: Date = Date(), editedFrom: JSONValue? = nil, changed: [String] = []
    ) -> JSONValue {
        var record: [String: JSONValue] = [
            "schemaVersion": .number(1), "madeWith": .string(madeWithAgentEditor),
            "client": .string("SteerLab app"),
            "createdAt": .string(ISO8601DateFormatter().string(from: date)),
            "reason": .null,
        ]
        if let editedFrom {
            record["editedFrom"] = editedFrom
            record["changed"] = .array(changed.map(JSONValue.string))
        }
        return .object(record)
    }

    /// What an editor record keeps of the agent an edit started from: its
    /// kind, its file's SHA-256, and for a promoted agent the study and sweep
    /// run (a handmade agent's own record, which names no paths, travels
    /// whole). Never a path. Server twin: read by
    /// `agent_identity.edited_from_phrase`.
    public static func editedFromRecord(
        _ source: ModelVariantArtifact, fileSHA256: String?
    ) -> JSONValue {
        var record: [String: JSONValue] = ["kind": .string(source.kind.rawValue)]
        if let fileSHA256 { record["artifactFileSHA256"] = .string(fileSHA256) }
        if let promotion = source.promotion {
            if !promotion.experiment.isEmpty { record["experiment"] = .string(promotion.experiment) }
            if let run = promotion.sweepRun, !run.isEmpty { record["sweepRun"] = .string(run) }
        } else if let handmade = source.handmade, handmade.agentRecordObject != nil {
            record["handmade"] = handmade
        }
        return .object(record)
    }

    /// This agent with the editor's record when it carries neither record;
    /// unchanged otherwise. `source` is the agent the edit started from and
    /// its file's SHA-256, when there was one.
    public func withEditorRecordIfUnrecorded(
        at date: Date = Date(), editedFrom source: (ModelVariantArtifact, String?)? = nil
    ) -> ModelVariantArtifact {
        guard promotion == nil, handmade == nil else { return self }
        var recorded = self
        recorded.handmade = Self.editorRecord(
            at: date,
            editedFrom: source.map { Self.editedFromRecord($0.0, fileSHA256: $0.1) },
            changed: source.map { changedFields(from: $0.0) } ?? [])
        return recorded
    }

    // MARK: - What reached the renderer

    /// The effort a template actually renders for `effort` under this
    /// record, by `PromptRendering.thinkingContext`'s own rule: off when the
    /// template has no thinking switch, on for a level it ignores or rejects
    /// (or, with a probed record, does not read), else the level. What a
    /// record stamps. Server twin: `agent_identity.rendered_effort` (where a
    /// rejected level never renders at all).
    public static func renderedReasoningEffort(
        _ effort: ReasoningEffort, modelID: String, capabilities: ModelCapabilities? = nil
    ) -> ReasoningEffort {
        let record = PromptRendering.capabilities(for: modelID, explicit: capabilities)
        guard record.hasThinkingSwitch, effort.isOn else { return .off }
        guard effort.isLevel else { return effort }
        let verdict = record.effortVerdict(effort.rawValue)
        if verdict == .ignored || verdict == .rejected
            || (verdict == nil && record.effortVariableRead == false)
        {
            return .on
        }
        return effort
    }

    // MARK: - The rules an attached agent arm must meet

    /// The instruments that read an answer from its first tokens, with no
    /// reasoning before it.
    public static let answerTokenInstruments: Set<String> = [
        "answerTokenLogprob", "choiceProbability", "ordinalScale",
    ]

    /// Whether at least one attached agent arm thinks. Server twin:
    /// `agent_identity.an_attached_agent_arm_thinks`.
    public static func anAttachedAgentArmThinks(_ manifest: ExperimentManifest) -> Bool {
        manifest.variantConditions.contains { $0.artifact.resolvedReasoningEffort.isOn }
    }

    /// Whether a study whose own effort is off may still declare a reasoning
    /// budget: the ONE predicate every gate asks. Today it holds when an
    /// attached agent arm thinks (ruling 2026-11-08); the budget then reaches
    /// only the arms that think. A panel seat cast with a thinking agent joins
    /// it here. Server twin: `agent_identity.budget_permitted_without_effort`.
    public static func budgetPermittedWithoutEffort(_ manifest: ExperimentManifest) -> Bool {
        anAttachedAgentArmThinks(manifest)
    }

    // MARK: - Revisions

    /// How an agent's revision compares with the study's. Server twin:
    /// `agent_identity.REVISIONS_*`.
    public enum RevisionComparison: String, Sendable {
        case match, differ, unchecked
    }

    private static func isCommitLike(_ value: String) -> Bool {
        (7...40).contains(value.utf8.count)
            && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    /// Whether two model revisions name the same commit, as far as their
    /// spelling can tell: `differ` only when both are commit hashes and
    /// neither begins the other; `match` when they are the same; otherwise
    /// `unchecked` (a branch or tag can move, and a shortened hash the other
    /// begins with may name the same commit). Server twin:
    /// `agent_identity.compare_revisions`.
    public static func compareRevisions(
        _ agentRevision: String, _ studyRevision: String
    ) -> RevisionComparison {
        let a = agentRevision.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let b = studyRevision.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if a == b { return .match }
        guard isCommitLike(a), isCommitLike(b) else { return .unchecked }
        return a.hasPrefix(b) || b.hasPrefix(a) ? .unchecked : .differ
    }

    /// Why an agent made for another commit is refused; `agent` names it
    /// where no arm label already does. Server twin:
    /// `agent_identity.revision_mismatch_reason`.
    public static func revisionMismatchReason(
        model: String, agentRevision: String, studyRevision: String, agent: String? = nil
    ) -> String {
        "\(agent.map { "agent '\($0)'" } ?? "the agent") was made for revision "
            + "\(agentRevision) of \(model), but the study pins revision \(studyRevision)"
    }

    public static let revisionMismatchRepair =
        "Make the agent again at the study's revision (agent create --revision with the "
        + "study's modelRevision), or pin the study to the agent's revision before "
        + "attaching it."

    /// What `verify` notes when this agent's revision cannot be checked
    /// against the study's (advisory `agentRevisionNotChecked`), or nil.
    /// Server twin: `agent_identity.revision_note`, the same sentences.
    public func revisionNote(label: String, modelID: String, modelRevision: String?) -> String? {
        guard let studyRevision = modelRevision,
            !studyRevision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        let model = baseModelID.isEmpty ? modelID : baseModelID
        guard let revision = baseRevision,
            !revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return "\(label): the agent records no model revision (agents saved before agents "
                + "recorded one carry none), so SteerLab did not check it against revision "
                + "\(studyRevision) of \(model), which the study pins."
        }
        guard Self.compareRevisions(revision, studyRevision) == .unchecked else { return nil }
        let moving = [revision, studyRevision].filter {
            !Self.isCommitLike($0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
        }
        let why = moving.first.map { "\($0) is a branch or tag name, which can move, not a commit" }
            ?? "one is a shortened commit hash that the other begins with"
        return "\(label): the agent was made for revision \(revision) of \(model) and the "
            + "study pins revision \(studyRevision); SteerLab did not check that they are the "
            + "same commit, because \(why). Make the agent with the full 40-character commit "
            + "to have it checked."
    }

    /// `revisionNote` over every attached agent arm. Server twin:
    /// `agent_identity.manifest_arm_notes`.
    public static func manifestArmNotes(_ manifest: ExperimentManifest) -> [String] {
        guard manifest.studyKind != .multiAgent else { return [] }
        return manifest.variantConditions.filter { $0.fromPromotion == nil }.compactMap {
            $0.artifact.revisionNote(
                label: "agent arm '\($0.name)'", modelID: manifest.modelID,
                modelRevision: manifest.modelRevision)
        }
    }

    // MARK: - What each broken rule needs

    /// What a broken rule's repair is: make the agent again and attach it in
    /// place of the arm; make it again at the study's revision; or change the
    /// study's instruments. Server twin: `agent_identity.FIX_*`.
    public enum ArmFix: String, Sendable {
        case agent, revision, instrument
    }

    /// One broken rule: the arm, the sentence, and what fixes it.
    public struct ArmFinding: Sendable, Equatable {
        public var arm: String
        public var sentence: String
        public var fix: ArmFix
    }

    private func findings(
        label: String, modelID: String, modelRevision: String?, outcomeInstruments: [String],
        capabilities: ModelCapabilities?, workspaceRoot: URL
    ) -> [(String, ArmFix)] {
        if let problem = fileProblem { return [("\(label): \(problem)", .agent)] }
        var found = effortViolations(capabilities: capabilities).map { ("\(label): \($0)", ArmFix.agent) }
        if adapters.count > 1 {
            found.append(("\(label): the agent lists \(adapters.count) adapters, and both "
                + "engines run only the first, so the others would be recorded but never run",
                .agent))
        }
        let effort = resolvedReasoningEffort
        if effort.isOn, !Set(outcomeInstruments).isDisjoint(with: Self.answerTokenInstruments) {
            found.append(("\(label): the agent thinks (reasoning effort \(effort.rawValue)) "
                + "but the study declares an answer-token instrument, which reads the answer "
                + "with no reasoning before it", .instrument))
        }
        if let revision = baseRevision,
            !revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            let studyRevision = modelRevision,
            !studyRevision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            Self.compareRevisions(revision, studyRevision) == .differ
        {
            found.append(("\(label): " + Self.revisionMismatchReason(
                model: baseModelID.isEmpty ? modelID : baseModelID,
                agentRevision: revision, studyRevision: studyRevision), .revision))
        }
        found += injectionProblems(workspaceRoot: workspaceRoot).map { ("\(label): \($0)", .agent) }
        return found
    }

    /// Every settings-only rule this attached agent arm breaks, as the
    /// server's sentences (`agent_identity.arm_problems`): the file's own
    /// problems, a level the template does not accept, more than one adapter,
    /// a thinking agent beside an answer-token instrument, a commit other
    /// than the study's (`compareRevisions` says `differ`), and an injection
    /// at a layer its vector does not have, an ablation that lists one, or a
    /// norm-unit strength where the vector records no norm at a layer it acts
    /// on.
    public func armProblems(
        name: String, modelID: String, modelRevision: String?,
        outcomeInstruments: [String], capabilities: ModelCapabilities?, workspaceRoot: URL
    ) -> [String] {
        findings(
            label: "agent arm '\(name)'", modelID: modelID, modelRevision: modelRevision,
            outcomeInstruments: outcomeInstruments, capabilities: capabilities,
            workspaceRoot: workspaceRoot
        ).map(\.0)
    }

    /// The same rules for an agent a panel casts in a seat, minus the
    /// answer-token rule (a panel reads its seats' generated turns). `label`
    /// names the seat in each sentence (for example `"seat 'judge'"`). For a
    /// panel's verify to call per cast agent. Server twin:
    /// `agent_identity.cast_agent_problems`.
    public func castAgentProblems(
        label: String, modelID: String, modelRevision: String?,
        capabilities: ModelCapabilities?, workspaceRoot: URL
    ) -> [String] {
        findings(
            label: label, modelID: modelID, modelRevision: modelRevision,
            outcomeInstruments: [], capabilities: capabilities, workspaceRoot: workspaceRoot
        ).map(\.0)
    }

    private func injectionProblems(workspaceRoot: URL) -> [String] {
        var problems: [String] = []
        let half = max(1, bandWidth) / 2
        for injection in injections where !injection.vectorArtifactID.isEmpty {
            let reference = injection.vectorArtifactID
            let base = reference.hasPrefix("/")
                ? URL(fileURLWithPath: reference) : workspaceRoot.appending(path: reference)
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: base.path + ".json")),
                let sidecar = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }  // a missing vector is the artifact preflight's refusal
            guard let countNumber = sidecar["layerCount"] as? NSNumber,
                CFGetTypeID(countNumber) != CFBooleanGetTypeID()
            else { continue }
            let count = countNumber.intValue
            let concept = injection.concept.isEmpty ? reference : injection.concept
            if injection.effectiveMode == .ablate {
                // An ablation acts on its listed layers, or every layer; its
                // `layer` is not where it acts.
                var named = injection
                named.concept = concept
                if let problem = named.ablationLayerProblem(layerCount: count) {
                    problems.append(problem)
                }
                continue
            }
            guard (0..<count).contains(injection.layer) else {
                problems.append("its injection of '\(concept)' is at layer \(injection.layer), "
                    + "which its vector does not have (layers 0 to \(count - 1))")
                continue
            }
            guard alphaInNormUnits else { continue }
            let norms = (sidecar["residualNormPerLayer"] as? [NSNumber])?.map(\.floatValue)
            let vectorNorms = (sidecar["normsPerLayer"] as? [NSNumber])?.map(\.doubleValue)
            let band = max(0, injection.layer - half)...min(count - 1, injection.layer + half)
            if let missing = band.first(where: { layer in
                // A layer where the vector is zero adds nothing, so the
                // engine skips it and needs no residual norm there.
                if let vectorNorms, layer < vectorNorms.count, vectorNorms[layer] == 0 {
                    return false
                }
                return ResidualNormConvention.residualNormProblem(
                    norms, layer: layer, artifact: concept) != nil
            }) {
                problems.append("its injection of '\(concept)' gives a strength in "
                    + "residual-norm units, but the vector records no residual norm at layer "
                    + "\(missing)")
            }
        }
        return problems
    }

    /// `armProblems` over every attached agent arm (a forward reference has
    /// no file yet), with what fixes each, against the record the declaration
    /// gates read. Server twin: `agent_identity.manifest_arm_findings`.
    public static func manifestArmFindings(
        _ manifest: ExperimentManifest, workspaceRoot: URL
    ) -> [ArmFinding] {
        guard manifest.studyKind != .multiAgent else { return [] }
        let capabilities = manifest.status == .draft
            ? ModelCapabilitiesStore.resolve(
                modelID: manifest.modelID, revision: manifest.modelRevision, root: workspaceRoot)
            : ModelCapabilities.heuristicView(
                modelID: manifest.modelID, revision: manifest.modelRevision)
        return manifest.variantConditions.filter { $0.fromPromotion == nil }.flatMap { arm in
            arm.artifact.findings(
                label: "agent arm '\(arm.name)'", modelID: manifest.modelID,
                modelRevision: manifest.modelRevision,
                outcomeInstruments: manifest.outcomeInstruments ?? [],
                capabilities: capabilities, workspaceRoot: workspaceRoot
            ).map { ArmFinding(arm: arm.name, sentence: $0.0, fix: $0.1) }
        }
    }

    /// The sentences of `manifestArmFindings`. Server twin:
    /// `agent_identity.manifest_arm_problems`.
    public static func manifestArmProblems(
        _ manifest: ExperimentManifest, workspaceRoot: URL
    ) -> [String] {
        manifestArmFindings(manifest, workspaceRoot: workspaceRoot).map(\.sentence)
    }

    /// The repair for `manifestArmFindings`, as this command line spells it.
    /// A frozen study is duplicated first and repaired on the draft. An arm
    /// whose agent must change is replaced in place with `attach-agent
    /// --replace`; a thinking agent beside an answer-token instrument is fixed
    /// by declaring instruments that read the answer after the reasoning.
    /// Server twin: `agent_identity.arm_problems_repair`, the same steps.
    public static func armProblemsRepair(
        _ manifest: ExperimentManifest, findings: [ArmFinding], frozen: Bool
    ) -> String {
        let program = ExperimentCLIHelp.program
        let target = frozen ? "\(manifest.name)-v2" : manifest.name
        var steps: [String] = []
        if frozen { steps.append("\(program) experiment duplicate \(manifest.name) \(target)") }
        var arms: [String] = []
        var fixes: [String: Set<ArmFix>] = [:]
        for finding in findings {
            if fixes[finding.arm] == nil { arms.append(finding.arm) }
            fixes[finding.arm, default: []].insert(finding.fix)
        }
        let revision = manifest.modelRevision ?? "<sha>"
        for arm in arms {
            let wanted = fixes[arm] ?? []
            guard wanted.contains(.agent) || wanted.contains(.revision) else { continue }
            let at = wanted.contains(.revision) ? "at the study's revision " : ""
            steps.append("make the agent again \(at)and attach it in place of arm '\(arm)': "
                + "\(program) agent create <new-name> --model \(manifest.modelID) --revision "
                + "\(revision) … (or the app's agent editor), then \(program) experiment "
                + "attach-agent \(target) --replace \(arm) --artifact <path> --artifact-sha256 "
                + "<digest> --manifest-sha256 <digest>")
        }
        if fixes.values.contains(where: { $0.contains(.instrument) }) {
            let remaining = (manifest.outcomeInstruments ?? []).filter {
                !answerTokenInstruments.contains($0)
            }
            steps.append("read the answers after the agents' reasoning: \(program) experiment "
                + "set-instruments \(target) "
                + (remaining.isEmpty ? "sampledText" : remaining.joined(separator: ",")))
        }
        return steps.joined(separator: "; then ")
    }

    /// What `verify` says of a frozen study whose agent arms break a rule
    /// added since it froze (advisory `refusedWhenRun`). Server twin:
    /// `agent_identity.frozen_arm_problems_advisory`.
    public static func frozenArmProblemsAdvisory(
        _ manifest: ExperimentManifest, workspaceRoot: URL
    ) -> String? {
        guard manifest.status != .draft else { return nil }
        let findings = manifestArmFindings(manifest, workspaceRoot: workspaceRoot)
        guard !findings.isEmpty else { return nil }
        return "this study will be refused when its run is submitted or started, because "
            + findings.map(\.sentence).joined(separator: "; ") + ". To run it: "
            + armProblemsRepair(manifest, findings: findings, frozen: true)
    }

    /// The refusal a step that runs the study gives on its agent arms, or
    /// nil: at submission and at run start, before the queue and the model.
    public static func armRefusal(
        _ manifest: ExperimentManifest, workspaceRoot: URL
    ) -> ExperimentError? {
        let findings = manifestArmFindings(manifest, workspaceRoot: workspaceRoot)
        guard !findings.isEmpty else { return nil }
        return .refusing(
            .studyDeclaration, findings.map(\.sentence).joined(separator: "; "),
            repair: armProblemsRepair(
                manifest, findings: findings, frozen: manifest.status != .draft))
    }
}

extension JSONValue {
    /// The members of an object value, or nil for any other value.
    var agentRecordObject: [String: JSONValue]? {
        if case .object(let members) = self { return members }
        return nil
    }

    /// The text of a string value, or nil for any other value.
    var agentRecordString: String? {
        if case .string(let text) = self { return text }
        return nil
    }
}

public struct ModelVariantRecord: Identifiable, Sendable {
    public let url: URL
    public let artifact: ModelVariantArtifact

    public var id: String { url.path }
    public var label: String {
        "\(artifact.name) · \(artifact.baseModelID) · \(dateLabel(artifact.createdAt))"
    }

    private func dateLabel(_ iso: String) -> String {
        let trimmed = iso.replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
        return String(trimmed.prefix(min(16, trimmed.count)))
    }
}

public enum ModelVariantStore {
    public static var directory: URL {
        VectorCatalog.runsDirectory.appending(component: "model-variants")
    }

    /// Whether an agent file lives in the editable agent library, as opposed
    /// to a run folder (an agent a run saved, or one imported with a run's
    /// evidence). Run folders are append-only evidence: an agent there is
    /// never rewritten in place, only copied.
    public static func isInLibrary(_ url: URL) -> Bool {
        let library = directory.standardizedFileURL.path
        let file = url.standardizedFileURL.path
        return file.hasPrefix(library.hasSuffix("/") ? library : library + "/")
    }

    /// Every agent artifact this workspace can see, from BOTH engines'
    /// layouts.
    ///
    /// The two engines mint agents into different shapes:
    ///
    /// - Swift `save`: `runs/model-variants/<slug>/model-variant.json`
    /// - Python `save_variant`: `runs/<stamp>-variant-<name>/<name>.json`
    ///
    /// Scanning only for the literal filename `model-variant.json` under the
    /// `model-variants/` library therefore made every server-promoted agent
    /// invisible on the Mac — it would list under the server's own panel
    /// (which queries the server) and be absent from the Agent Library
    /// (which scans locally), so it could never be selected in Studies.
    /// Observed 2026-07-26 with a promoted `neuroticism` agent.
    ///
    /// The fix is to identify an agent by what it IS, not by where it was
    /// filed: a run directory stamped `runType: "variant-save"` in the
    /// canonical `config.json` — a schema-2 key both engines write — whose
    /// JSON payload decodes as a `ModelVariantArtifact`. Imported evidence
    /// is read in place and never moved or rewritten; discovery adapts to
    /// the artifact, not the reverse.
    ///
    /// - Parameters:
    ///   - root: the native `model-variants/` library.
    ///   - importedRoot: the `runs/` tree to additionally sweep for
    ///     `variant-save` run directories. Nil disables the second pass
    ///     (tests that assert only the native layout).
    public static func scan(
        directory root: URL = directory,
        importedRoot: URL? = VectorCatalog.runsDirectory
    ) -> [ModelVariantRecord] {
        var records = scanNativeLibrary(root)
        if let importedRoot {
            records.append(
                contentsOf: scanImportedRuns(importedRoot, excluding: root))
        }
        // Same artifact reachable by two paths (e.g. a library entry that
        // also carries a run stamp) counts once; the native path wins.
        var seen = Set<String>()
        records = records.filter { seen.insert($0.url.standardizedFileURL.path).inserted }
        return records.sorted {
            if $0.artifact.createdAt == $1.artifact.createdAt {
                return $0.url.path < $1.url.path
            }
            return $0.artifact.createdAt > $1.artifact.createdAt
        }
    }

    private static func scanNativeLibrary(_ root: URL) -> [ModelVariantRecord] {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return [] }

        var records: [ModelVariantRecord] = []
        for case let url as URL in enumerator where url.lastPathComponent == "model-variant.json" {
            guard
                let data = try? Data(contentsOf: url),
                let artifact = try? JSONDecoder().decode(ModelVariantArtifact.self, from: data)
            else { continue }
            records.append(ModelVariantRecord(url: url, artifact: artifact))
        }
        return records
    }

    /// Second pass: run directories anywhere under `runs/` that declare
    /// themselves agent mints. One level deep, matching `VectorCatalog.scan`
    /// — imported evidence lands as `runs/<runID>/`, not nested deeper.
    private static func scanImportedRuns(
        _ runsRoot: URL, excluding nativeLibrary: URL
    ) -> [ModelVariantRecord] {
        let fm = FileManager.default
        let nativePath = nativeLibrary.standardizedFileURL.path
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsRoot, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }

        var records: [ModelVariantRecord] = []
        for runDirectory in entries {
            guard runDirectory.standardizedFileURL.path != nativePath,
                (try? runDirectory.resourceValues(forKeys: [.isDirectoryKey]))?
                    .isDirectory == true,
                isVariantSaveRun(runDirectory)
            else { continue }
            guard
                let files = try? fm.contentsOfDirectory(
                    at: runDirectory, includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles])
            else { continue }
            for file in files
            where file.pathExtension == "json" && file.lastPathComponent != "config.json" {
                guard
                    let data = try? Data(contentsOf: file),
                    let artifact = try? JSONDecoder().decode(
                        ModelVariantArtifact.self, from: data)
                else { continue }
                records.append(ModelVariantRecord(url: file, artifact: artifact))
            }
        }
        return records
    }

    /// The cross-engine stamp: canonical `config.json` `runType`.
    private static func isVariantSaveRun(_ runDirectory: URL) -> Bool {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "config.json")),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return false }
        return object["runType"] as? String == "variant-save"
    }

    /// EVERY path-bearing reference in a serialized agent is workspace-
    /// relative when it names something inside this workspace — the portable
    /// shape both engines resolve (`ArtifactIdentity.resolve` /
    /// `paths.resolve`). Enforced at the write funnel because the writers
    /// keep re-learning this one at a time: `promote` serialized the
    /// absolute catalog id (2026-08-04 — every cluster run died on
    /// `/Users/<user>/…`), and the hand-creation UI picks from the same
    /// absolute-id catalog. References outside the workspace pass through:
    /// they are already broken for portability, and rewriting them would
    /// hide it.
    static func workspaceRelativeNormalized(
        _ artifact: ModelVariantArtifact
    ) -> ModelVariantArtifact {
        var normalized = artifact
        normalized.injections = normalized.injections.map { injection in
            var updated = injection
            updated.vectorArtifactID = ArtifactIdentity.workspaceRelative(
                updated.vectorArtifactID)
            return updated
        }
        normalized.adapters = normalized.adapters.map { adapter in
            var updated = adapter
            updated.artifactPath = ArtifactIdentity.workspaceRelative(
                updated.artifactPath)
            updated.adapterDirectory = ArtifactIdentity.workspaceRelative(
                updated.adapterDirectory)
            return updated
        }
        if let basis = normalized.neutralPCBasisPath {
            normalized.neutralPCBasisPath = ArtifactIdentity.workspaceRelative(basis)
        }
        return normalized
    }

    @discardableResult
    public static func save(_ artifact: ModelVariantArtifact) throws -> ModelVariantRecord {
        try save(workspaceRelativeNormalized(artifact), library: self.directory)
    }

    /// Writes `artifact` as a new agent in the library folder `library`
    /// (`<workspace>/runs/model-variants`), exactly as `save` does: a fresh
    /// folder, its run stamp, and the same encoding. For a caller that holds
    /// an explicit workspace and already wrote its references relative to it
    /// (`agent create`).
    @discardableResult
    public static func save(
        _ artifact: ModelVariantArtifact, library: URL
    ) throws -> ModelVariantRecord {
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "model-variant-\(FineTuneStore.slugify(artifact.name))",
            under: library)
        try RunMetadata.write(
            runType: "variant-save", to: directory,
            modelID: artifact.baseModelID, revision: artifact.baseRevision)
        let url = directory.appending(component: "model-variant.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    /// Writes `artifact` over the agent file at `url`.
    ///
    /// Refused for an agent a run saved (a file under `runs/` outside the
    /// library): a run's folder is evidence, and `runs/` is append-only.
    /// `saveChanges` saves such an agent's edits as a new library agent.
    @discardableResult
    public static func update(
        _ artifact: ModelVariantArtifact,
        at url: URL
    ) throws -> ModelVariantRecord {
        if isSavedByRun(url) {
            let run = url.deletingLastPathComponent().lastPathComponent
            throw WorkspaceHousekeeping.Refusal(
                code: WorkspaceHousekeeping.agentIsRunEvidenceWriteCode,
                path: relativePath(for: ModelVariantRecord(url: url, artifact: artifact)),
                reason: "This agent was saved by a run (\(run)), and run folders are kept "
                    + "as evidence: they are never rewritten.",
                repairAction: "Save the edits as a new agent instead; the agent editor's "
                    + "Save as New Agent does this and leaves the run's agent as it is.")
        }
        let artifact = workspaceRelativeNormalized(artifact)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    /// Whether `url` is an agent file in a run's folder: under `runs/`, and
    /// not in the agent library. Compared as spelled and symlink-resolved
    /// (`/var` against `/private/var`), so neither spelling slips past.
    static func isSavedByRun(_ url: URL) -> Bool {
        [false, true].contains { resolved in
            func path(_ location: URL) -> String {
                (resolved ? location.resolvingSymlinksInPath() : location)
                    .standardizedFileURL.path
            }
            let file = path(url)
            return file.hasPrefix(path(VectorCatalog.runsDirectory) + "/")
                && !file.hasPrefix(path(directory) + "/")
        }
    }

    /// What `saveChanges` did.
    public enum SaveChangesOutcome: Sendable {
        /// The library agent's file was rewritten in place.
        /// `droppedPromotion` is `AgentEditProvenance.Merge.droppedPromotion`.
        case updated(ModelVariantRecord, droppedPromotion: Bool)
        /// The agent was saved by a run: the edit is a new library agent, and
        /// the run's file was not written.
        case savedAsNewAgent(ModelVariantRecord)
        /// The agent was saved by a run and there was nothing to save:
        /// nothing was written.
        case nothingToSave
    }

    /// The agent editor's Save Changes for `selected`, as
    /// `AgentEditProvenance.saveChangesPlan` decides it. `takenNames` are the
    /// listed agents' names, which a new agent's name must avoid. The caller
    /// asks about a dropped promotion record before calling.
    @discardableResult
    public static func saveChanges(
        _ edited: ModelVariantArtifact, to selected: ModelVariantRecord, takenNames: [String]
    ) throws -> SaveChangesOutcome {
        switch AgentEditProvenance.saveChangesPlan(edited: edited, over: selected) {
        case .updateInPlace(let merge):
            // An edit that changes what the agent does leaves no record that
            // describes it, so the editor's own record is written (an agent
            // carries exactly one record), naming the agent it was edited
            // from and what changed. An unedited save writes the stored
            // content back, record or none, unchanged.
            let artifact = AgentEditProvenance.doesTheSame(edited, as: selected.artifact)
                ? merge.artifact
                : merge.artifact.withEditorRecordIfUnrecorded(
                    editedFrom: (selected.artifact, try? hash(selected.url)))
            return .updated(
                try update(artifact, at: selected.url),
                droppedPromotion: merge.droppedPromotion)
        case .saveAsNewAgent:
            return .savedAsNewAgent(
                try saveAsNewAgent(edited, takenNames: takenNames, source: selected))
        case .nothingToSave:
            return .nothingToSave
        }
    }

    /// Saves `edited` as a new library agent under a name none of
    /// `takenNames` has (`AgentEditProvenance.copyName`). A new agent has no
    /// promotion record: a record describes the agent it was made for. Its
    /// editor record names `source`, the agent the edit started from, when
    /// there was one.
    @discardableResult
    public static func saveAsNewAgent(
        _ edited: ModelVariantArtifact, takenNames: [String], source: ModelVariantRecord? = nil
    ) throws -> ModelVariantRecord {
        var copy = edited
        copy.name = AgentEditProvenance.copyName(edited.name, taken: takenNames)
        copy.promotion = nil
        copy.handmade = nil
        return try save(copy.withEditorRecordIfUnrecorded(
            editedFrom: source.map { ($0.artifact, try? hash($0.url)) }))
    }

    /// Moves the agent's folder into a `.trash-<time>` folder in the agent
    /// library and returns where it landed. Nothing is erased, and every
    /// listing skips hidden folders, so the agent leaves the library and can
    /// be moved back by hand.
    ///
    /// Callers check first that no study uses the agent
    /// (`WorkspaceHousekeeping.deleteAgent(reviewed:)` is the one that does).
    @discardableResult
    public static func delete(
        _ record: ModelVariantRecord, workspaceRoot: URL = ExperimentStore.workspaceRoot
    ) throws -> URL {
        let folder = try trashableFolder(for: record, workspaceRoot: workspaceRoot)
        return try WorkspaceHousekeeping.moveToTrash(folder, under: folder.deletingLastPathComponent())
    }

    /// The folder deleting this agent moves: its own entry directly inside
    /// the agent library (`runs/model-variants/<slug>/`).
    ///
    /// An agent saved by a run (`runs/<run>/<name>.json`, how the Python
    /// engine and imported evidence store them) is refused: a run folder is
    /// evidence, and `runs/` is append-only. So is anything that would take
    /// more than this agent with it: a file sitting in the library itself, a
    /// folder nested deeper, or a folder that also holds other agents.
    static func trashableFolder(for record: ModelVariantRecord, workspaceRoot: URL) throws -> URL {
        let runsDirectory = VectorCatalog.runsDirectory(root: workspaceRoot)
        let libraryDirectory = runsDirectory.appending(component: "model-variants")
        let folder = record.url.deletingLastPathComponent()
        let parent = try ManifestFileTransaction.canonicalPath(folder.deletingLastPathComponent())
        let library = try ManifestFileTransaction.canonicalPath(libraryDirectory)
        let runs = try ManifestFileTransaction.canonicalPath(runsDirectory)
        let folderPath = try ManifestFileTransaction.canonicalPath(folder)
        if parent == runs, folderPath != library {
            throw WorkspaceHousekeeping.Refusal(
                code: WorkspaceHousekeeping.agentIsRunEvidenceCode,
                path: try WorkspaceHousekeeping.relativePath(record.url, in: workspaceRoot),
                reason: "This agent was saved by a run (\(folder.lastPathComponent)), and run "
                    + "folders are kept as evidence: they are never moved or deleted.",
                repairAction: "Nothing needs repairing. An agent no study uses changes "
                    + "nothing; it stays in the agent list.")
        }
        guard parent == library else {
            throw ExperimentError.refusing(
                .artifactPin,
                "This agent is not stored in a folder of its own, so deleting it would "
                    + "take other files with it.",
                repair: "Move the agent's file into its own folder under "
                    + "runs/model-variants/, then preview the delete again.")
        }
        let fm = FileManager.default
        let siblings = ((try? fm.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? [])
            .filter {
                $0.pathExtension == "json" && $0.lastPathComponent != "config.json"
                    && $0.lastPathComponent != record.url.lastPathComponent
            }
            .filter { url in
                (try? Data(contentsOf: url)).flatMap {
                    try? JSONDecoder().decode(ModelVariantArtifact.self, from: $0)
                } != nil
            }
        guard siblings.isEmpty else {
            throw ExperimentError.refusing(
                .artifactPin,
                "This agent shares its folder with "
                    + siblings.map(\.lastPathComponent).sorted().joined(separator: ", ")
                    + ", which deleting it would also remove.",
                repair: "Move each agent into a folder of its own under "
                    + "runs/model-variants/, then preview the delete again.")
        }
        return folder
    }

    public static func hash(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func relativePath(for record: ModelVariantRecord) -> String {
        // Directory enumeration can return /var while the workspace names
        // /private/var. Compare canonical roots before producing a public pin.
        let root = VectorCatalog.projectRoot.resolvingSymlinksInPath().standardizedFileURL.path
        let path = record.url.resolvingSymlinksInPath().standardizedFileURL.path
        return path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path
    }

    public static func absoluteURL(_ path: String) -> URL {
        FineTuneStore.absoluteURL(path)
    }
}
