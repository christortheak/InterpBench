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

        /// The layers this ref actually covers, given the model's depth.
        /// Steering answers with its single declared layer (the caller
        /// widens it by the variant's band); ablation answers with every
        /// layer unless it declared otherwise — a single-layer ablation is
        /// usually undone by the layers above it, so "all" is the honest
        /// default rather than a convenience.
        public func resolvedLayers(layerCount: Int) -> [Int] {
            guard effectiveMode == .ablate else { return [layer] }
            if let layers, !layers.isEmpty {
                return layers.filter { (0..<layerCount).contains($0) }
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
            qualification: JSONValue? = nil
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
        promotion: Promotion? = nil
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
        self.qwenThinkingEnabled = qwenThinkingEnabled
        self.temperature = temperature
        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        self.systemPrompt = trimmedSystem.isEmpty ? nil : systemPrompt
        self.systemPromptHash = trimmedSystem.isEmpty
            ? nil
            : Self.sha256(systemPrompt)
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
        self.promotion = promotion
    }

    private static func sha256(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion, name, baseModelID, baseRevision, adapters, injections
        case bandWidth, alphaInNormUnits, neutralPCBasisPath, neutralPCBasisLabel
        case promptMode, qwenThinkingEnabled, temperature
        case systemPromptHash, systemPrompt, createdAt, promotion
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
        qwenThinkingEnabled =
            try container.decodeIfPresent(Bool.self, forKey: .qwenThinkingEnabled) ?? false
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        systemPromptHash = try container.decodeIfPresent(String.self, forKey: .systemPromptHash)
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
        promotion = try container.decodeIfPresent(Promotion.self, forKey: .promotion)
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
        let artifact = workspaceRelativeNormalized(artifact)
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "model-variant-\(FineTuneStore.slugify(artifact.name))",
            under: self.directory)
        try RunMetadata.write(
            runType: "variant-save", to: directory,
            modelID: artifact.baseModelID, revision: artifact.baseRevision)
        let url = directory.appending(component: "model-variant.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    @discardableResult
    public static func update(
        _ artifact: ModelVariantArtifact,
        at url: URL
    ) throws -> ModelVariantRecord {
        let artifact = workspaceRelativeNormalized(artifact)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    public static func delete(_ record: ModelVariantRecord) throws {
        try FileManager.default.removeItem(at: record.url.deletingLastPathComponent())
    }

    public static func hash(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func relativePath(for record: ModelVariantRecord) -> String {
        FineTuneStore.relativePath(for: record.url)
    }

    public static func absoluteURL(_ path: String) -> URL {
        FineTuneStore.absoluteURL(path)
    }
}
