import CryptoKit
import Foundation
import SteeringKit

/// Canonical full-recipe identity for extraction artifacts — the identity
/// Promote matches on, closing the provenance hole where two extractions
/// could "match" while representing different recipes (different reading
/// position, neutral projection, norm denominator, or grand-mean population)
/// with the newest silently winning.
///
/// CANONICAL FORM (cross-engine contract; this doc comment is mirrored
/// verbatim in `Server/steerlab_server/experiment/recipe_identity.py`):
///
/// recipeIdentityHash = SHA-256 hex of the UTF-8 canonical JSON of the
/// recipe: sorted keys (recursively), compact separators ("," and ":"),
/// explicit nulls for every absent field, raw UTF-8 (no ASCII escaping, no
/// forward-slash escaping). Top-level keys, in sorted order:
///
/// - "concept": the concept name.
/// - "extractionMethod": the substrate-independent method name in the
///   MANIFEST vocabulary ("meanDifference" | "lat" | "emotionGrandMean").
///   Sidecar recipeMethod values map caaMeanDifference→meanDifference,
///   repeLAT→lat, emotionGrandMean→emotionGrandMean; any other recorded
///   method travels verbatim (it can never equal a manifest method).
/// - "grandMeanPopulation": for emotionGrandMean only — the FULL comparison
///   population as [[conceptName, storiesSha256], …] sorted by conceptName
///   then hash (code-point order); null for every other method.
/// - "methodParameters": method parameters, or null. designatedReference:
///   {"referenceHash": storiesSha256, "referenceName": conceptName} — two
///   vectors built against different references must never share an
///   identity (external review 2026-07-31, finding 2). Null for every other
///   method, preserving pre-existing hashes. (Previously documented as: 
///   substrate-independent parameters; LAT is the fixed first principal
///   component on both engines).
/// - "modelID": the HF model id.
/// - "neutralProjection": {"basisHash": string|null, "count": int|null,
///   "explainedVariance": decimal-string|null, "mode": "none" |
///   "legacyPooled" | "tokenBankFixedCount" | "tokenBankExplainedVariance"}
///   — all inner fields explicit. explainedVariance is the decimal string
///   exactly as recorded (a string, so float formatting can never diverge
///   across engines).
/// - "normCorpusHash": SHA-256 of the pinned neutral corpus when
///   residualNormSource is "neutral-corpus" or "neutral-token-bank"; null
///   otherwise.
/// - "readingPosition": {"mode": "lastToken" | "meanFromToken",
///   "parameter": int|null} (the pool-from token index for meanFromToken).
/// - "residualNormSource": the canonical source token ("neutral-corpus" |
///   "extraction-stimuli" | "neutral-token-bank"). A sidecar value is
///   canonicalized by truncating at the first space (the Swift experiment
///   writer historically embedded a corpus-hash prefix after it), and the
///   Swift grand-mean self-measured label "multi-concept-corpus"
///   canonicalizes to "extraction-stimuli" (the server records the same
///   denominator recipe — norms measured on the extraction stimuli
///   themselves — under that name).
/// - "revision": the pinned model revision, or null.
/// - "schema": 1 (the integer literal).
/// - "stimulusSetHash": the concept's pinned stimulus-set hash.
///
/// Substrate is deliberately OUTSIDE this identity: it remains a separate
/// match criterion (a CUDA artifact must never satisfy an MLX recipe
/// silently), exactly as before.
public enum RecipeIdentity {

    public static let schema = 1

    /// One grand-mean population member: (concept name, stories.jsonl
    /// SHA-256).
    public struct Member: Sendable, Equatable {
        public var concept: String
        public var hash: String

        public init(concept: String, hash: String) {
            self.concept = concept
            self.hash = hash
        }
    }

    /// The recipe fields the identity is computed over. Optionals encode as
    /// explicit JSON nulls — never omitted.
    public struct Components: Sendable, Equatable {
        public var concept: String
        public var modelID: String
        public var revision: String?
        public var extractionMethod: String
        public var stimulusSetHash: String
        public var readingPositionMode: String
        public var readingPositionParameter: Int?
        public var projectionMode: String
        public var projectionCount: Int?
        public var projectionExplainedVariance: String?
        public var projectionBasisHash: String?
        public var residualNormSource: String
        public var normCorpusHash: String?
        public var grandMeanPopulation: [Member]?
        /// designatedReference only: the reference pin (concept = name).
        public var designatedReference: Member?

        public init(
            concept: String, modelID: String, revision: String?,
            extractionMethod: String, stimulusSetHash: String,
            readingPositionMode: String, readingPositionParameter: Int?,
            projectionMode: String, projectionCount: Int?,
            projectionExplainedVariance: String?, projectionBasisHash: String?,
            residualNormSource: String, normCorpusHash: String?,
            grandMeanPopulation: [Member]?,
            designatedReference: Member? = nil
        ) {
            self.concept = concept
            self.modelID = modelID
            self.revision = revision
            self.extractionMethod = extractionMethod
            self.stimulusSetHash = stimulusSetHash
            self.readingPositionMode = readingPositionMode
            self.readingPositionParameter = readingPositionParameter
            self.projectionMode = projectionMode
            self.projectionCount = projectionCount
            self.projectionExplainedVariance = projectionExplainedVariance
            self.projectionBasisHash = projectionBasisHash
            self.residualNormSource = residualNormSource
            self.normCorpusHash = normCorpusHash
            self.grandMeanPopulation = grandMeanPopulation
            self.designatedReference = designatedReference
        }
    }

    /// The outcome of reading a sidecar's recipe: either full components or
    /// the sorted list of canonical field names the artifact cannot prove.
    public struct Candidate: Sendable {
        public let components: Components?
        public let missingFields: [String]
    }

    // MARK: - canonical JSON + hash

    /// The canonical JSON string (see the canonical-form contract above).
    /// Built by hand so the byte output is engine-independent: Python's
    /// `json.dumps(sort_keys=True, separators=(",", ":"),
    /// ensure_ascii=False)` produces exactly these bytes for the same
    /// payload.
    public static func canonicalJSON(_ c: Components) -> String {
        var out = "{"
        out += "\"concept\":\(jsonString(c.concept))"
        out += ",\"extractionMethod\":\(jsonString(c.extractionMethod))"
        out += ",\"grandMeanPopulation\":"
        if let population = c.grandMeanPopulation {
            let sorted = population.sorted {
                ($0.concept, $0.hash) < ($1.concept, $1.hash)
            }
            out += "["
                + sorted.map { "[\(jsonString($0.concept)),\(jsonString($0.hash))]" }
                    .joined(separator: ",")
                + "]"
        } else {
            out += "null"
        }
        if let reference = c.designatedReference {
            out += ",\"methodParameters\":{"
                + "\"referenceHash\":\(jsonString(reference.hash))"
                + ",\"referenceName\":\(jsonString(reference.concept))}"
        } else {
            out += ",\"methodParameters\":null"
        }
        out += ",\"modelID\":\(jsonString(c.modelID))"
        out += ",\"neutralProjection\":{"
        out += "\"basisHash\":\(jsonOptionalString(c.projectionBasisHash))"
        out += ",\"count\":\(c.projectionCount.map(String.init) ?? "null")"
        out += ",\"explainedVariance\":\(jsonOptionalString(c.projectionExplainedVariance))"
        out += ",\"mode\":\(jsonString(c.projectionMode))}"
        out += ",\"normCorpusHash\":\(jsonOptionalString(c.normCorpusHash))"
        out += ",\"readingPosition\":{"
        out += "\"mode\":\(jsonString(c.readingPositionMode))"
        out += ",\"parameter\":\(c.readingPositionParameter.map(String.init) ?? "null")}"
        out += ",\"residualNormSource\":\(jsonString(c.residualNormSource))"
        out += ",\"revision\":\(jsonOptionalString(c.revision))"
        out += ",\"schema\":\(schema)"
        out += ",\"stimulusSetHash\":\(jsonString(c.stimulusSetHash))}"
        return out
    }

    public static func hash(_ components: Components) -> String {
        SHA256.hash(data: Data(canonicalJSON(components).utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - actionable field diff

    /// Human-readable differences between two identities, as dotted canonical
    /// field paths with both values — e.g. `"revision (manifest: null,
    /// artifact: 8f2a66d1c9e0…)"` or `"neutralProjection.mode (manifest:
    /// none, artifact: legacyPooled)"`. Compares the CANONICAL components
    /// (the exact values the hash covers), listed in the canonical JSON's
    /// sorted key order, so both engines emit the same fields in the same
    /// sequence (server twin: `recipe_identity.diff_fields`). Empty means the
    /// identities hash identically.
    public static func diffFields(
        manifest required: Components, artifact candidate: Components
    ) -> [String] {
        var diffs: [String] = []
        func compare(_ path: String, _ a: String?, _ b: String?,
                     compound: Bool = false) {
            guard a != b else { return }
            let show = compound ? displayCompound : display
            diffs.append("\(path) (manifest: \(show(a)), artifact: \(show(b)))")
        }
        compare("concept", required.concept, candidate.concept)
        compare("extractionMethod", required.extractionMethod,
                candidate.extractionMethod)
        compare("grandMeanPopulation",
                populationJSON(required.grandMeanPopulation),
                populationJSON(candidate.grandMeanPopulation), compound: true)
        compare("methodParameters",
                required.designatedReference.map {
                    "{\"referenceHash\":\"\($0.hash)\",\"referenceName\":\"\($0.concept)\"}"
                },
                candidate.designatedReference.map {
                    "{\"referenceHash\":\"\($0.hash)\",\"referenceName\":\"\($0.concept)\"}"
                },
                compound: true)
        // "methodParameters" is the reserved always-null slot: never diffs.
        compare("modelID", required.modelID, candidate.modelID)
        compare("neutralProjection.basisHash", required.projectionBasisHash,
                candidate.projectionBasisHash)
        compare("neutralProjection.count",
                required.projectionCount.map(String.init),
                candidate.projectionCount.map(String.init))
        compare("neutralProjection.explainedVariance",
                required.projectionExplainedVariance,
                candidate.projectionExplainedVariance)
        compare("neutralProjection.mode", required.projectionMode,
                candidate.projectionMode)
        compare("normCorpusHash", required.normCorpusHash,
                candidate.normCorpusHash)
        compare("readingPosition.mode", required.readingPositionMode,
                candidate.readingPositionMode)
        compare("readingPosition.parameter",
                required.readingPositionParameter.map(String.init),
                candidate.readingPositionParameter.map(String.init))
        compare("residualNormSource", required.residualNormSource,
                candidate.residualNormSource)
        compare("revision", required.revision, candidate.revision)
        // "schema" is a shared literal: never diffs.
        compare("stimulusSetHash", required.stimulusSetHash,
                candidate.stimulusSetHash)
        return diffs
    }

    /// Compact scalar rendering for diff messages: explicit `null`, long
    /// strings (hashes, revisions) truncated to a 12-char prefix. Mirrors
    /// `recipe_identity._display` for scalars.
    static func display(_ value: String?) -> String {
        guard let value else { return "null" }
        return value.count <= 16 ? value : String(value.prefix(12)) + "…"
    }

    /// Compound-value rendering (the population's canonical JSON): mirrors
    /// the server's 48/44-char truncation of `json.dumps(...)`.
    static func displayCompound(_ text: String?) -> String {
        guard let text else { return "null" }
        return text.count <= 48 ? text : String(text.prefix(44)) + "…"
    }

    /// The population's canonical JSON fragment (sorted, compact) — the same
    /// bytes `canonicalJSON` embeds, so the diff compares exactly what the
    /// hash covered.
    static func populationJSON(_ population: [Member]?) -> String? {
        guard let population else { return nil }
        let sorted = population.sorted {
            ($0.concept, $0.hash) < ($1.concept, $1.hash)
        }
        return "["
            + sorted.map { "[\(jsonString($0.concept)),\(jsonString($0.hash))]" }
                .joined(separator: ",")
            + "]"
    }

    // MARK: - the identity a MANIFEST requires

    /// The full-recipe identity this manifest's pinned recipe demands for one
    /// concept. Deterministic from pins alone (no filesystem reads): the
    /// extraction paths derive the norm denominator from exactly these pins,
    /// so the prediction here matches what a faithful extraction stamps.
    public static func required(
        manifest: ExperimentManifest, ref: ExperimentManifest.ConceptRef
    ) throws -> Components {
        let readingMode: String
        let readingParameter: Int?
        switch ref.options.readingPosition {
        case .lastToken:
            readingMode = "lastToken"
            readingParameter = nil
        case .meanFromToken(let k):
            readingMode = "meanFromToken"
            readingParameter = k
        }
        let pcCount = ref.options.neutralPCCount ?? 0
        // A pinned neutral corpus is the norm denominator on both engines
        // (extract / extractGrandMean use it whenever present); without one,
        // norms come from the extraction stimuli themselves.
        let source = manifest.neutralCorpusHash != nil
            ? "neutral-corpus" : "extraction-stimuli"
        var designatedReference: Member?
        if ref.options.method == .designatedReference {
            guard let pin = ref.designatedReference else {
                throw ExperimentError(
                    reason: "designated-reference concept '\(ref.name)' has no "
                        + "pinned reference — re-attach with a reference before "
                        + "promoting")
            }
            designatedReference = Member(concept: pin.name, hash: pin.hash)
        }
        var population: [Member]?
        if ref.options.method.isGrandMean {
            guard let corpus = manifest.grandMeanCorpus else {
                throw ExperimentError(
                    reason: "grand-mean concept '\(ref.name)' has no pinned "
                        + "grandMeanCorpus — re-attach with method emotionGrandMean")
            }
            population = try corpus.concepts.map { member in
                guard let hash = corpus.hashes[member] else {
                    throw ExperimentError(
                        reason: "grandMeanCorpus member '\(member)' has no pinned "
                            + "hash — re-attach before promoting")
                }
                return Member(concept: member, hash: hash)
            }
        }
        return Components(
            concept: ref.name,
            modelID: manifest.modelID,
            revision: manifest.modelRevision,
            extractionMethod: ref.options.method.rawValue,
            stimulusSetHash: ref.stimulusSetHash,
            readingPositionMode: readingMode,
            readingPositionParameter: readingParameter,
            projectionMode: pcCount > 0 ? "legacyPooled" : "none",
            projectionCount: pcCount > 0 ? pcCount : nil,
            projectionExplainedVariance: nil,
            projectionBasisHash: nil,
            residualNormSource: source,
            normCorpusHash: source == "neutral-corpus" ? manifest.neutralCorpusHash : nil,
            grandMeanPopulation: population,
            designatedReference: designatedReference)
    }

    // MARK: - the identity a SIDECAR can prove

    /// Compute the identity from an artifact's recorded provenance. Every
    /// canonical field the sidecar cannot prove is NAMED in
    /// `missingFields` (sorted) — the caller must refuse, never guess.
    public static func candidate(sidecar: SteeringVectorSidecar) -> Candidate {
        var missing: Set<String> = []

        let methodMap = [
            "caaMeanDifference": "meanDifference",
            "repeLAT": "lat",
            "emotionGrandMean": "emotionGrandMean",
        ]
        let method: String?
        if let recipe = sidecar.recipeMethod {
            method = methodMap[recipe] ?? recipe
        } else {
            method = sidecar.extractionMethod
        }
        if method == nil { missing.insert("extractionMethod") }

        var readingMode: String?
        var readingParameter: Int?
        if let label = sidecar.readingPosition,
            let position = ReadingPosition(label: label)
        {
            switch position {
            case .lastToken:
                readingMode = "lastToken"
            case .meanFromToken(let k):
                readingMode = "meanFromToken"
                readingParameter = k
            }
        } else {
            missing.insert("readingPosition")
        }

        var projection: (mode: String, count: Int?, explainedVariance: String?)?
        if let description = sidecar.neutralProjection {
            projection = parseProjection(description)
        } else if let legacy = sidecar.confoundProjection {
            // Pre-neutralProjection sidecars recorded the legacy pooled
            // projection as bare "top-K neutral PCs".
            projection = parseProjection(legacy)
        }
        if projection == nil { missing.insert("neutralProjection") }

        var source: String?
        if let recorded = sidecar.residualNormSource,
            let token = recorded.split(separator: " ").first, !token.isEmpty
        {
            // The Swift experiment writer embeds a corpus-hash prefix after a
            // space; the grand-mean self-measured label unifies with the
            // server's (see the canonical-form contract).
            source = token == "multi-concept-corpus" ? "extraction-stimuli" : String(token)
        } else {
            missing.insert("residualNormSource")
        }

        var normCorpusHash: String?
        if let source, source == "neutral-corpus" || source == "neutral-token-bank" {
            if let hash = sidecar.neutralCorpusHash {
                normCorpusHash = hash
            } else {
                missing.insert("normCorpusHash")
            }
        }

        var population: [Member]?
        if method == "emotionGrandMean" {
            if let recorded = sidecar.grandMeanPopulation {
                population = recorded.map { Member(concept: $0.key, hash: $0.value) }
            } else {
                missing.insert("grandMeanPopulation")
            }
        }

        var designatedReference: Member?
        if method == "designatedReference" {
            if let recorded = sidecar.designatedReference,
                let name = recorded["name"], let hash = recorded["hash"],
                !name.isEmpty, !hash.isEmpty
            {
                designatedReference = Member(concept: name, hash: hash)
            } else {
                missing.insert("designatedReference")
            }
        }

        guard missing.isEmpty, let method, let readingMode, let projection,
            let source
        else {
            return Candidate(components: nil, missingFields: missing.sorted())
        }
        return Candidate(
            components: Components(
                concept: sidecar.concept,
                modelID: sidecar.modelID,
                revision: sidecar.revision,
                extractionMethod: method,
                stimulusSetHash: sidecar.stimulusSetHash,
                readingPositionMode: readingMode,
                readingPositionParameter: readingParameter,
                projectionMode: projection.mode,
                projectionCount: projection.count,
                projectionExplainedVariance: projection.explainedVariance,
                projectionBasisHash: nil,
                residualNormSource: source,
                normCorpusHash: normCorpusHash,
                grandMeanPopulation: population,
                designatedReference: designatedReference),
            missingFields: [])
    }

    /// Parse a sidecar's neutral-projection description into the canonical
    /// (mode, count, explainedVariance) triple. Nil = unrecognized — the
    /// caller reports the field as unprovable, never guesses. Mirrors
    /// `recipe_identity._parse_projection` exactly.
    static func parseProjection(
        _ description: String
    ) -> (mode: String, count: Int?, explainedVariance: String?)? {
        if description == "none" { return ("none", nil, nil) }
        var text = description
        if text.hasPrefix("legacy-pooled ") {
            text = String(text.dropFirst("legacy-pooled ".count))
        }
        if text.hasPrefix("top-"), text.hasSuffix(" neutral PCs") {
            let middle = text.dropFirst("top-".count)
                .dropLast(" neutral PCs".count)
            guard let count = Int(middle), count >= 0,
                middle.allSatisfy(\.isNumber)
            else { return nil }
            return ("legacyPooled", count, nil)
        }
        if description.hasPrefix("token-bank fixed-count "),
            description.hasSuffix(" PCs")
        {
            let middle = description.dropFirst("token-bank fixed-count ".count)
                .dropLast(" PCs".count)
            guard let count = Int(middle), count >= 0,
                middle.allSatisfy(\.isNumber)
            else { return nil }
            return ("tokenBankFixedCount", count, nil)
        }
        if description.hasPrefix("token-bank explained-variance ") {
            let value = description.dropFirst("token-bank explained-variance ".count)
                .trimmingCharacters(in: .whitespaces)
            guard !value.isEmpty else { return nil }
            return ("tokenBankExplainedVariance", nil, value)
        }
        return nil
    }

    // MARK: - JSON string encoding (byte-parity with Python json.dumps)

    static func jsonOptionalString(_ value: String?) -> String {
        value.map(jsonString) ?? "null"
    }

    /// Escapes exactly like Python `json.dumps(…, ensure_ascii=False)`:
    /// `"` and `\` escaped, control characters as \b \f \n \r \t or
    /// \u00xx (lowercase hex), everything else — including non-ASCII —
    /// passed through as raw UTF-8.
    static func jsonString(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
