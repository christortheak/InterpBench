import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// Canonical full-recipe identity — the cross-engine contract Promote
/// matches artifacts on. Pure CPU: canonical-JSON byte parity against the
/// committed golden fixture (shared verbatim with the server's
/// `test_recipe_identity.py`), per-field hash sensitivity, and the strict
/// sidecar readers that refuse to guess.
struct RecipeIdentityTests {

    // MARK: - fixture plumbing

    static let fixtureURL = VectorCatalog.bundledSeedRoot.appending(
        components: "prompts", "fixtures", "recipe-identity",
        "recipe-identity-fixture.json")

    private func loadFixtureCases() throws -> [String: (
        components: RecipeIdentity.Components, canonicalJSON: String, sha256: String
    )] {
        let data = try Data(contentsOf: Self.fixtureURL)
        let root = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases = try #require(root["cases"] as? [String: Any])
        var out: [String: (RecipeIdentity.Components, String, String)] = [:]
        for (name, value) in cases {
            let entry = try #require(value as? [String: Any])
            let raw = try #require(entry["components"] as? [String: Any])
            var population: [RecipeIdentity.Member]?
            if let pairs = raw["grandMeanPopulation"] as? [[String]] {
                population = pairs.map { .init(concept: $0[0], hash: $0[1]) }
            }
            let components = RecipeIdentity.Components(
                concept: try #require(raw["concept"] as? String),
                modelID: try #require(raw["modelID"] as? String),
                revision: raw["revision"] as? String,
                extractionMethod: try #require(raw["extractionMethod"] as? String),
                stimulusSetHash: try #require(raw["stimulusSetHash"] as? String),
                readingPositionMode: try #require(raw["readingPositionMode"] as? String),
                readingPositionParameter: raw["readingPositionParameter"] as? Int,
                projectionMode: try #require(raw["projectionMode"] as? String),
                projectionCount: raw["projectionCount"] as? Int,
                projectionExplainedVariance: raw["projectionExplainedVariance"] as? String,
                projectionBasisHash: raw["projectionBasisHash"] as? String,
                residualNormSource: try #require(raw["residualNormSource"] as? String),
                normCorpusHash: raw["normCorpusHash"] as? String,
                grandMeanPopulation: population,
                designatedReference: nil,
                extractionRendering: renderingFragment(
                    raw["extractionRendering"] as? [String: Any]))
            out[name] = (
                components,
                try #require(entry["canonicalJSON"] as? String),
                try #require(entry["sha256"] as? String)
            )
        }
        return out
    }

    /// The fixture stores the rendering as the canonical identity FRAGMENT
    /// (`null` for absent/raw), so this decodes exactly what both engines
    /// hash.
    private func renderingFragment(_ raw: [String: Any]?) -> ExtractionRendering? {
        guard let raw, let mode = raw["mode"] as? String,
            let parsed = ExtractionRendering.Mode(rawValue: mode)
        else { return nil }
        return ExtractionRendering(
            mode: parsed,
            addGenerationPrompt: raw["addGenerationPrompt"] as? Bool,
            qwenThinkingEnabled: raw["qwenThinkingEnabled"] as? Bool,
            systemPrompt: raw["systemPrompt"] as? String)
    }

    // MARK: - cross-engine golden parity

    @Test func canonicalFormMatchesCommittedFixtureByteForByte() throws {
        let cases = try loadFixtureCases()
        #expect(Set(cases.keys) == ["grandMean", "paired", "templatedRendering"])
        for (name, entry) in cases {
            let json = RecipeIdentity.canonicalJSON(entry.components)
            #expect(json == entry.canonicalJSON, "case \(name): canonical JSON drifted")
            #expect(RecipeIdentity.hash(entry.components) == entry.sha256,
                    "case \(name): identity hash drifted")
        }
    }

    @Test func populationOrderNeverChangesTheHash() throws {
        let grand = try #require(try loadFixtureCases()["grandMean"])
        var shuffled = grand.components
        shuffled.grandMeanPopulation = try #require(
            shuffled.grandMeanPopulation?.reversed().map { $0 })
        #expect(RecipeIdentity.hash(shuffled) == grand.sha256)
    }

    // MARK: - per-field sensitivity (each newly covered field must move the hash)

    @Test func everyNewlyCoveredFieldChangesTheHash() throws {
        let base = try #require(try loadFixtureCases()["grandMean"]).components
        let baseHash = RecipeIdentity.hash(base)

        var flips: [(String, RecipeIdentity.Components)] = []
        func flip(_ label: String, _ mutate: (inout RecipeIdentity.Components) -> Void) {
            var c = base
            mutate(&c)
            flips.append((label, c))
        }
        flip("readingPosition parameter") { $0.readingPositionParameter = 49 }
        flip("readingPosition mode") {
            $0.readingPositionMode = "lastToken"
            $0.readingPositionParameter = nil
        }
        flip("projection K") { $0.projectionCount = 2 }
        flip("projection mode") {
            $0.projectionMode = "none"
            $0.projectionCount = nil
        }
        flip("projection explained variance") {
            $0.projectionMode = "tokenBankExplainedVariance"
            $0.projectionCount = nil
            $0.projectionExplainedVariance = "0.5"
        }
        flip("norm source") {
            $0.residualNormSource = "extraction-stimuli"
            $0.normCorpusHash = nil
        }
        flip("norm corpus hash") {
            $0.normCorpusHash = String(repeating: "9", count: 64)
        }
        flip("one population member's stories hash") {
            $0.grandMeanPopulation?[0].hash = String(repeating: "c", count: 64)
        }
        flip("population membership") {
            $0.grandMeanPopulation?.removeLast()
        }
        flip("revision") { $0.revision = "def456" }
        flip("revision pinned vs absent") { $0.revision = nil }
        flip("method") { $0.extractionMethod = "lat" }
        flip("stimulus hash") { $0.stimulusSetHash = String(repeating: "2", count: 64) }

        var seen: Set<String> = [baseHash]
        for (label, components) in flips {
            let hash = RecipeIdentity.hash(components)
            #expect(!seen.contains(hash),
                    "\(label): flip must produce a distinct identity hash")
            seen.insert(hash)
        }
    }

    // MARK: - the identity a manifest requires

    private func manifest(
        neutralCorpusHash: String? = nil
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "ri", description: "", modelID: "test/model")
        manifest.modelRevision = "abc123"
        manifest.neutralCorpusHash = neutralCorpusHash
        return manifest
    }

    @Test func requiredIdentityPredictsTheNormDenominatorFromPins() throws {
        let ref = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: String(repeating: "f", count: 64),
            options: .init())
        let bare = try RecipeIdentity.required(manifest: manifest(), ref: ref)
        #expect(bare.residualNormSource == "extraction-stimuli")
        #expect(bare.normCorpusHash == nil)
        let corpusHash = String(repeating: "0", count: 64)
        let pinned = try RecipeIdentity.required(
            manifest: manifest(neutralCorpusHash: corpusHash), ref: ref)
        #expect(pinned.residualNormSource == "neutral-corpus")
        #expect(pinned.normCorpusHash == corpusHash)
        #expect(RecipeIdentity.hash(bare) != RecipeIdentity.hash(pinned))
    }

    @Test func requiredIdentityForGrandMeanDemandsThePinnedPopulation() throws {
        let ref = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: String(repeating: "f", count: 64),
            options: .init(method: .emotionGrandMean, readingPosition: .meanFromToken(50)))
        var m = manifest()
        // No pinned corpus: the identity is uncomputable, loudly.
        #expect(throws: ExperimentError.self) {
            try RecipeIdentity.required(manifest: m, ref: ref)
        }
        m.grandMeanCorpus = .init(
            concepts: ["fear", "joy"],
            hashes: [
                "fear": String(repeating: "f", count: 64),
                "joy": String(repeating: "a", count: 64),
            ])
        let components = try RecipeIdentity.required(manifest: m, ref: ref)
        #expect(components.grandMeanPopulation?.count == 2)
        #expect(components.readingPositionMode == "meanFromToken")
        #expect(components.readingPositionParameter == 50)
        // A member without a pinned hash refuses instead of hashing a hole.
        m.grandMeanCorpus = .init(concepts: ["fear", "joy"], hashes: ["fear": "x"])
        #expect(throws: ExperimentError.self) {
            try RecipeIdentity.required(manifest: m, ref: ref)
        }
    }

    // MARK: - the identity a sidecar can prove

    private func fullSidecar(
        readingPosition: String? = "last token",
        neutralProjection: String? = "none",
        residualNormSource: String? = "extraction-stimuli",
        neutralCorpusHash: String? = nil,
        extractionMethod: String? = "meanDifference"
    ) throws -> SteeringVectorSidecar {
        var raw: [String: Any] = [
            "modelID": "test/model", "concept": "fear",
            "stimulusSetHash": String(repeating: "f", count: 64),
            "layerCount": 2, "hiddenSize": 2, "normsPerLayer": [1.0, 1.0],
            "extractionDate": "2026-07-13T00:00:00Z", "revision": "abc123",
        ]
        raw["readingPosition"] = readingPosition
        raw["neutralProjection"] = neutralProjection
        raw["residualNormSource"] = residualNormSource
        raw["neutralCorpusHash"] = neutralCorpusHash
        raw["extractionMethod"] = extractionMethod
        return try JSONDecoder().decode(
            SteeringVectorSidecar.self,
            from: JSONSerialization.data(withJSONObject: raw))
    }

    @Test func completeSidecarProvesTheSameIdentityTheManifestRequires() throws {
        let ref = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: String(repeating: "f", count: 64),
            options: .init())
        let required = try RecipeIdentity.required(manifest: manifest(), ref: ref)
        let candidate = RecipeIdentity.candidate(sidecar: try fullSidecar())
        #expect(candidate.missingFields.isEmpty)
        let components = try #require(candidate.components)
        #expect(RecipeIdentity.hash(components) == RecipeIdentity.hash(required))
    }

    @Test func legacySidecarNamesEveryUnprovableField() throws {
        let sidecar = try fullSidecar(
            readingPosition: nil, neutralProjection: nil, residualNormSource: nil)
        let candidate = RecipeIdentity.candidate(sidecar: sidecar)
        #expect(candidate.components == nil)
        #expect(candidate.missingFields
            == ["neutralProjection", "readingPosition", "residualNormSource"])
    }

    @Test func grandMeanSidecarWithoutPopulationIsUnprovable() throws {
        let sidecar = try fullSidecar(
            residualNormSource: "multi-concept-corpus",
            extractionMethod: "emotionGrandMean")
        let candidate = RecipeIdentity.candidate(sidecar: sidecar)
        #expect(candidate.missingFields == ["grandMeanPopulation"])
        var stamped = sidecar
        stamped.grandMeanPopulation = ["fear": String(repeating: "f", count: 64)]
        let proven = RecipeIdentity.candidate(sidecar: stamped)
        let components = try #require(proven.components)
        // The Swift grand-mean self-measured label unifies with the server's.
        #expect(components.residualNormSource == "extraction-stimuli")
        #expect(components.grandMeanPopulation?.count == 1)
    }

    @Test func neutralCorpusNormsRequireTheFullCorpusHash() throws {
        // The historical Swift writer embedded only a 12-char prefix in the
        // source string — that proves nothing; the field is the proof.
        let prefixOnly = try fullSidecar(
            residualNormSource: "neutral-corpus 0123456789ab")
        #expect(RecipeIdentity.candidate(sidecar: prefixOnly).missingFields
            == ["normCorpusHash"])
        let full = try fullSidecar(
            residualNormSource: "neutral-corpus 0123456789ab",
            neutralCorpusHash: String(repeating: "0", count: 64))
        let components = try #require(RecipeIdentity.candidate(sidecar: full).components)
        #expect(components.residualNormSource == "neutral-corpus")
        #expect(components.normCorpusHash == String(repeating: "0", count: 64))
    }

    @Test func projectionDescriptionsParseStrictly() {
        typealias Parsed = (mode: String, count: Int?, explainedVariance: String?)
        func parsed(_ text: String) -> Parsed? { RecipeIdentity.parseProjection(text) }
        #expect(parsed("none")! == ("none", nil, nil))
        #expect(parsed("top-3 neutral PCs")! == ("legacyPooled", 3, nil))
        #expect(parsed("legacy-pooled top-3 neutral PCs")! == ("legacyPooled", 3, nil))
        #expect(parsed("token-bank fixed-count 4 PCs")! == ("tokenBankFixedCount", 4, nil))
        #expect(parsed("token-bank explained-variance 0.5")!
            == ("tokenBankExplainedVariance", nil, "0.5"))
        #expect(parsed("something else") == nil)
        #expect(parsed("top-x neutral PCs") == nil)
        #expect(parsed("token-bank explained-variance ") == nil)
    }

    @Test func recipeMethodVocabularyMapsToTheManifestVocabulary() throws {
        var sidecar = try fullSidecar(extractionMethod: nil)
        sidecar.recipeMethod = "caaMeanDifference"
        var components = try #require(RecipeIdentity.candidate(sidecar: sidecar).components)
        #expect(components.extractionMethod == "meanDifference")
        sidecar.recipeMethod = "repeLAT"
        components = try #require(RecipeIdentity.candidate(sidecar: sidecar).components)
        #expect(components.extractionMethod == "lat")
        sidecar.recipeMethod = nil
        let missing = RecipeIdentity.candidate(sidecar: sidecar)
        #expect(missing.missingFields == ["extractionMethod"])
    }

    @Test func diffFieldsNamesEachDifferingCanonicalFieldWithBothValues() throws {
        let base = try #require(try loadFixtureCases()["grandMean"]).components
        #expect(RecipeIdentity.diffFields(manifest: base, artifact: base).isEmpty)
        // Population order is canonicalized before diffing — never a
        // difference.
        var shuffled = base
        shuffled.grandMeanPopulation = try #require(
            shuffled.grandMeanPopulation?.reversed().map { $0 })
        #expect(RecipeIdentity.diffFields(manifest: base, artifact: shuffled).isEmpty)
        var changed = base
        changed.revision = nil
        changed.projectionMode = "none"
        changed.projectionCount = nil
        let diffs = RecipeIdentity.diffFields(manifest: changed, artifact: base)
        #expect(diffs.contains {
            $0.hasPrefix("revision (manifest: null, artifact: ")
        })
        #expect(diffs.contains(
            "neutralProjection.mode (manifest: none, artifact: legacyPooled)"))
        #expect(diffs.contains {
            $0.hasPrefix("neutralProjection.count (manifest: null, artifact: ")
        })
        // Dotted paths follow the canonical key order (sorted, recursive) —
        // the server's diff walks the same order.
        let paths = diffs.map { String($0.split(separator: " ")[0]) }
        #expect(paths == paths.sorted())
    }

    @Test func diffDisplayTruncatesHashesAndRendersNull() {
        #expect(RecipeIdentity.display(nil) == "null")
        #expect(RecipeIdentity.display("abc") == "abc")
        // 16 chars = verbatim; longer = 12-char prefix + ellipsis.
        #expect(RecipeIdentity.display("0123456789abcdef") == "0123456789abcdef")
        #expect(RecipeIdentity.display(String(repeating: "f", count: 64))
            == String(repeating: "f", count: 12) + "…")
        let population = (0..<9).map {
            RecipeIdentity.Member(
                concept: "concept-\($0)", hash: String(repeating: "a", count: 64))
        }
        let rendered = RecipeIdentity.displayCompound(
            RecipeIdentity.populationJSON(population))
        #expect(rendered.hasSuffix("…"))
        #expect(rendered.count == 45)
    }

    @Test func jsonStringEscapingMatchesPythonJSONDumps() {
        #expect(RecipeIdentity.jsonString("plain") == "\"plain\"")
        #expect(RecipeIdentity.jsonString("a/b") == "\"a/b\"")  // no slash escaping
        #expect(RecipeIdentity.jsonString("é") == "\"é\"")  // raw UTF-8
        #expect(RecipeIdentity.jsonString("a\"b\\c") == "\"a\\\"b\\\\c\"")
        #expect(RecipeIdentity.jsonString("\n\t\r") == "\"\\n\\t\\r\"")
        #expect(RecipeIdentity.jsonString("\u{01}") == "\"\\u0001\"")
    }

    // MARK: - hash compatibility: the rendering option is invisible to legacy

    /// THE HARD CONSTRAINT. A recipe that declares no `extractionRendering` —
    /// every recipe written before the option existed — must produce the SAME
    /// canonical bytes and the SAME hash it always did, so every frozen
    /// experiment keeps verifying. The fixture's two legacy cases carry their
    /// pre-option hashes verbatim; this asserts the key is genuinely ABSENT
    /// rather than present-and-null.
    @Test func aLegacyRecipeHashesIdenticallyWithAndWithoutTheKey() throws {
        for name in ["grandMean", "paired"] {
            let entry = try #require(try loadFixtureCases()[name])
            #expect(!entry.canonicalJSON.contains("extractionRendering"),
                    "case \(name): the legacy canonical form grew a key")
            var explicitlyAbsent = entry.components
            explicitlyAbsent.extractionRendering = nil
            #expect(RecipeIdentity.canonicalJSON(explicitlyAbsent) == entry.canonicalJSON)
            #expect(RecipeIdentity.hash(explicitlyAbsent) == entry.sha256)
        }
    }

    /// Declaring `.raw` says the legacy semantics out loud; it must not fork
    /// the identity away from an otherwise-identical recipe that said nothing.
    @Test func anExplicitlyDeclaredRawRenderingIsTheLegacyIdentity() throws {
        let entry = try #require(try loadFixtureCases()["paired"])
        var declared = entry.components
        declared.extractionRendering =
            RecipeIdentity.Components.canonicalRendering(.raw)
        #expect(RecipeIdentity.hash(declared) == entry.sha256)
    }

    /// The other half: an explicitly declared NON-default rendering is a
    /// different recipe, and every rendering PARAMETER is part of it.
    @Test func aDeclaredChatTemplateRenderingChangesTheIdentity() throws {
        let legacy = try #require(try loadFixtureCases()["paired"]).components
        var templated = legacy
        templated.extractionRendering =
            RecipeIdentity.Components.canonicalRendering(.chatTemplate())
        #expect(RecipeIdentity.hash(templated) != RecipeIdentity.hash(legacy))

        let variants: [ExtractionRendering] = [
            .chatTemplate(addGenerationPrompt: false),
            .chatTemplate(qwenThinkingEnabled: true),
            .chatTemplate(systemPrompt: "be brief"),
        ]
        for variant in variants {
            var varied = legacy
            varied.extractionRendering =
                RecipeIdentity.Components.canonicalRendering(variant)
            #expect(RecipeIdentity.hash(varied) != RecipeIdentity.hash(templated),
                    "rendering parameter \(variant.label) did not move the hash")
        }
    }

    /// `offsetFromEnd(0)` names the identical token, so it must not split an
    /// identity away from a last-token recipe (maintainer ruling: offsets are
    /// the mechanism, roles are the portable form — neither may quietly
    /// renumber a recipe).
    @Test func offsetFromEndZeroIsTheLastTokenIdentity() {
        #expect(RecipeIdentity.canonicalReading(.offsetFromEnd(0)).mode == "lastToken")
        #expect(RecipeIdentity.canonicalReading(.offsetFromEnd(0)).parameter == nil)
        #expect(RecipeIdentity.canonicalReading(.offsetFromEnd(3)).mode == "offsetFromEnd")
        #expect(RecipeIdentity.canonicalReading(.offsetFromEnd(3)).parameter == 3)
        #expect(RecipeIdentity.canonicalReading(.lastContentToken).mode == "lastContentToken")
        #expect(RecipeIdentity.canonicalReading(.turnCloseToken).mode == "turnCloseToken")
        #expect(RecipeIdentity.canonicalReading(.postInstruction(4)).mode == "postInstruction")
        #expect(RecipeIdentity.canonicalReading(.postInstruction(4)).parameter == 4)
    }

    /// A sidecar stamps the LABEL; the identity reads it back strictly.
    @Test func everyNewPositionLabelRoundTripsThroughTheSidecarReader() {
        let positions: [ReadingPosition] = [
            .lastToken, .meanFromToken(50), .offsetFromEnd(3),
            .lastContentToken, .turnCloseToken, .postInstruction(1),
        ]
        for position in positions {
            let parsed = ReadingPosition(label: position.label)
            #expect(parsed == position, "label '\(position.label)' did not round trip")
        }
        #expect(ReadingPosition(label: "somewhere in the middle") == nil)
        #expect(ReadingPosition(label: "post-instruction 9") == nil)
        #expect(ReadingPosition(label: "offset from end -2") == nil)
    }

    /// An artifact whose rendering block this engine cannot read is refused
    /// for promotion. Reading it as raw would be exactly the silent
    /// substitution the option exists to end.
    @Test func aLegacySidecarProvesTheAbsentRenderingAndATemplatedOneProvesItself() throws {
        var sidecar = SteeringVectorSidecar(
            modelID: "test/model", concept: "c", stimulusSetHash: "h",
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            options: ExtractionOptions(),
            residualNormPerLayer: [1], residualNormSource: "extraction-stimuli")
        let legacy = RecipeIdentity.candidate(sidecar: sidecar)
        #expect(legacy.missingFields.isEmpty)
        #expect(legacy.components?.extractionRendering == nil)

        sidecar.extractionRendering = ExtractionRendering.chatTemplate().stamp
        let templated = RecipeIdentity.candidate(sidecar: sidecar)
        #expect(templated.missingFields.isEmpty)
        #expect(templated.components?.extractionRendering?.mode == .chatTemplate)
        // …and the two identities differ, which is the whole point.
        #expect(RecipeIdentity.hash(try #require(legacy.components))
                != RecipeIdentity.hash(try #require(templated.components)))
    }

    /// The diff names the rendering by its canonical field path, so a refused
    /// promotion says WHICH axis differed.
    @Test func theDiffNamesTheRenderingField() throws {
        let legacy = try #require(try loadFixtureCases()["paired"]).components
        var templated = legacy
        templated.extractionRendering =
            RecipeIdentity.Components.canonicalRendering(.chatTemplate())
        let diffs = RecipeIdentity.diffFields(manifest: legacy, artifact: templated)
        #expect(diffs.contains { $0.hasPrefix("extractionRendering (") })
    }
}
