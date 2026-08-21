import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Server-authored SAE keys must survive THIS engine's round trips.
///
/// The SAE program (docs/SAE-VECTOR-INTERVENTION-PROPOSAL-2026-08-13-r2.md)
/// is Python/cluster-side by rule: import, qualification and latent
/// interventions all run there. But the Mac is the AUTHORING surface — it
/// loads, duplicates, edits and re-saves the same manifests, agents and vector
/// sidecars. Swift decodes through explicit `CodingKeys`, so any key it does
/// not name is silently dropped on re-save. For these keys that is not a lost
/// label:
///
/// - a dropped `saeCandidates` pin lets a duplicate freeze cleanly while
///   claiming no roster, so nothing constrains which features it may seat;
/// - a dropped `promotion.qualification` leaves a promotion that claims no
///   evidence — silent destruction of the citation;
/// - a dropped `saeLatentConditions` un-declares an arm the study runs.
@Suite(.serialized) struct SAEManifestPassthroughTests {

    private func manifestJSON(
        name: String = "sae-screen",
        candidates: String = #"{"path":"prompts/sae/candidates.json","hash":"deadbeef"}"#
    ) -> String {
        """
        {"name":"\(name)","experimentDescription":"d",
         "modelID":"google/gemma-3-27b-it",
         "modelRevision":"005ad3404e59d6023443cb575daa05336842228a",
         "status":"draft","createdAt":"2026-08-13T00:00:00Z","concepts":[],
         "saeCandidates":\(candidates),
         "maxSAEMixtureFeatures":3,
         "saeLatentConditions":[
           {"name":"f62389-clamp","mode":"clamp","feature":62389,"layer":40,
            "dose":4.0,"serverOnly":true}]}
        """
    }

    @Test func theThreeSAEKeysDecodeVerbatim() throws {
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(manifestJSON().utf8))
        guard case .object(let pin)? = manifest.saeCandidates else {
            Issue.record("saeCandidates did not decode as an object")
            return
        }
        #expect(pin["path"] == .string("prompts/sae/candidates.json"))
        #expect(pin["hash"] == .string("deadbeef"))
        #expect(manifest.maxSAEMixtureFeatures == .number(3))
        guard case .array(let arms)? = manifest.saeLatentConditions,
            case .object(let arm) = arms[0]
        else {
            Issue.record("saeLatentConditions did not decode as a list")
            return
        }
        #expect(arm["mode"] == .string("clamp"))
        #expect(arm["feature"] == .number(62389))
        #expect(arm["serverOnly"] == .bool(true))
    }

    @Test func reEncodeDropsNothing() throws {
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(manifestJSON().utf8))
        let round = try JSONDecoder().decode(
            ExperimentManifest.self, from: try JSONEncoder().encode(decoded))
        #expect(round.saeCandidates == decoded.saeCandidates)
        #expect(round.maxSAEMixtureFeatures == decoded.maxSAEMixtureFeatures)
        #expect(round.saeLatentConditions == decoded.saeLatentConditions)
    }

    @Test func absentKeysStayAbsentAndAddNoNulls() throws {
        // Otherwise every existing manifest would gain three null keys on
        // re-save and change its content hash.
        let manifest = ExperimentManifest(
            name: "s", description: "", modelID: "google/gemma-3-4b-it")
        let text = String(
            decoding: try JSONEncoder().encode(manifest), as: UTF8.self)
        #expect(!text.contains("saeCandidates"))
        #expect(!text.contains("maxSAEMixtureFeatures"))
        #expect(!text.contains("saeLatentConditions"))
    }

    @Test func duplicatePreservesTheCandidateRoster() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sae") { root in
            let directory = ExperimentStore.directory
                .appending(component: "sae-screen")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try Data(manifestJSON().utf8).write(
                to: directory.appending(component: "experiment.json"))

            let copy = try ExperimentStore.duplicate(
                name: "sae-screen", as: "sae-screen-2")
            #expect(copy.saeCandidates != nil)
            #expect(copy.maxSAEMixtureFeatures == .number(3))
            #expect(copy.saeLatentConditions != nil)

            // …and the copy on DISK carries it, not just the in-memory value.
            let reloaded = try ExperimentStore.load(name: "sae-screen-2")
            guard case .object(let pin)? = reloaded.saeCandidates else {
                Issue.record("the duplicate lost its saeCandidates pin")
                return
            }
            #expect(pin["hash"] == .string("deadbeef"))
            _ = root
        }
    }
}

/// The `saeCandidates` pin is a MEASUREMENT-side input: it fixes which SAE
/// features a study may seat, before behaviour is measured. Swift re-checks it
/// mechanically — the same shape as the reasoning-style taxonomy check. The
/// roster's SCHEMA stays server-only (a second validator drifts from the
/// first by construction).
@Suite(.serialized) struct SAECandidatesPinVerifyTests {

    private func pin(path: String, hash: String) -> JSONValue {
        .object(["path": .string(path), "hash": .string(hash)])
    }

    private func plantRoster(_ text: String, at relative: String) throws -> String {
        let url = ExperimentStore.resolveProjectPath(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
        return ExperimentStore.sha256Hex(Data(text.utf8))
    }

    @Test func anAbsentBlockViolatesNothing() {
        #expect(ExperimentStore.saeCandidatesPinViolations(nil).isEmpty)
        #expect(ExperimentStore.saeCandidatesPinViolations(.null).isEmpty)
    }

    @Test func aMatchingHashPasses() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "saepin") { _ in
            let hash = try plantRoster(
                #"{"candidates":[]}"#, at: "prompts/sae/candidates.json")
            let violations = ExperimentStore.saeCandidatesPinViolations(
                pin(path: "prompts/sae/candidates.json", hash: hash))
            #expect(violations.isEmpty, Comment(rawValue: "\(violations)"))
        }
    }

    @Test func driftedBytesAreAViolation() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "saepin") { _ in
            _ = try plantRoster(
                #"{"candidates":[{"feature":62389}]}"#,
                at: "prompts/sae/candidates.json")
            let violations = ExperimentStore.saeCandidatesPinViolations(
                pin(path: "prompts/sae/candidates.json", hash: "deadbeef"))
            #expect(violations.count == 1)
            #expect(violations.first?.contains("changed since pinning") == true)
        }
    }

    @Test func aMissingFileIsAViolation() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "saepin") { _ in
            let violations = ExperimentStore.saeCandidatesPinViolations(
                pin(path: "prompts/sae/candidates.json", hash: "deadbeef"))
            #expect(violations.first?.contains("file missing") == true)
        }
    }

    @Test func aHalfPinCertifiesNothing() {
        let violations = ExperimentStore.saeCandidatesPinViolations(
            .object(["path": .string("prompts/sae/candidates.json")]))
        #expect(violations.first?.contains("incomplete") == true)
    }

    @Test func anAbsolutePathIsAViolation() {
        let violations = ExperimentStore.saeCandidatesPinViolations(
            pin(path: "/Users/x/candidates.json", hash: "deadbeef"))
        #expect(violations.first?.contains("absolute") == true)
    }

    @Test func unknownKeysAndNonObjectsRefuse() {
        #expect(
            ExperimentStore.saeCandidatesPinViolations(
                .object([
                    "path": .string("p"), "hash": .string("h"),
                    "decision": .string("accept"),
                ])
            ).first?.contains("unknown key") == true)
        #expect(
            ExperimentStore.saeCandidatesPinViolations(.string("p"))
                .first?.contains("must be an object") == true)
    }

    @Test func verifySurfacesTheDriftedRoster() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "saepin") { _ in
            _ = try plantRoster(#"{"candidates":[]}"#, at: "prompts/sae/roster.json")
            var manifest = ExperimentManifest(
                name: "s", description: "", modelID: "google/gemma-3-27b-it")
            manifest.variantConditions = []
            manifest.saeCandidates = pin(path: "prompts/sae/roster.json", hash: "nope")
            #expect(
                ExperimentStore.verify(manifest)
                    .contains { $0.contains("changed since pinning") })
        }
    }
}

/// A promoted agent's birth certificate CITES its qualification evidence. The
/// Agent Library edits and re-saves agent artifacts in place, so a decode that
/// does not name the key rewrites the artifact minus its citation.
@Suite struct PromotionQualificationPassthroughTests {

    private let artifactJSON = """
        {"schemaVersion":1,"name":"f62389-agent",
         "baseModelID":"google/gemma-3-27b-it","adapters":[],
         "injections":[{"concept":"f62389",
           "vectorArtifactID":"runs/x/f62389","layer":40,"alpha":0.08}],
         "bandWidth":1,"alphaInNormUnits":true,"promptMode":"chatAssistant",
         "qwenThinkingEnabled":false,"temperature":0,
         "createdAt":"2026-08-13T00:00:00Z",
         "promotion":{"experiment":"sae-screen","experimentHash":"abc",
           "promotedAt":"2026-08-13T00:00:00Z","promotedBy":"criterion",
           "substrate":"python-hf-transformers","appVersion":"1.0",
           "qualification":{"path":"runs/y/sae-feature-qualification.json",
             "contentHash":"cafebabe","decision":"accept"}}}
        """

    @Test func qualificationSurvivesDecodeAndReEncode() throws {
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(artifactJSON.utf8))
        let round = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: try JSONEncoder().encode(decoded))
        #expect(round.promotion?.qualification == decoded.promotion?.qualification)

        guard case .object(let citation)? = round.promotion?.qualification else {
            Issue.record("promotion.qualification did not decode as an object")
            return
        }
        #expect(citation["decision"] == .string("accept"))
        #expect(citation["contentHash"] == .string("cafebabe"))
        #expect(
            citation["path"] == .string("runs/y/sae-feature-qualification.json"))
    }

    @Test func theEditAndReSaveFunnelKeepsTheCitation() throws {
        // `ModelVariantStore.update` is what the Agent Library calls: decode,
        // change a field, write back. The citation must be on the bytes it
        // writes.
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(artifactJSON.utf8))
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "agent-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "model-variant.json")

        var edited = decoded
        edited.bandWidth = 3
        _ = try ModelVariantStore.update(edited, at: url)

        let reloaded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(contentsOf: url))
        #expect(reloaded.bandWidth == 3)
        #expect(reloaded.promotion?.qualification == decoded.promotion?.qualification)
    }

    @Test func anUncitedPromotionAddsNoKey() throws {
        let promotion = ModelVariantArtifact.Promotion(
            experiment: "e", experimentHash: "h", promotedAt: "t",
            promotedBy: "criterion", substrate: "swift-mlx", appVersion: "1.0")
        #expect(promotion.qualification == nil)
        let text = String(
            decoding: try JSONEncoder().encode(promotion), as: UTF8.self)
        #expect(!text.contains("qualification"))
    }
}

/// The residual-norm BACKFILL is the one decode -> re-encode path for vector
/// sidecars on this engine (`SteeringVectorStore.save` callers all build a
/// fresh sidecar from extraction). It must carry an imported SAE row's
/// identity onto the backfilled artifact: the server's qualification chain
/// matches artifacts BY `gemmascopeSource`, so losing it there would leave a
/// vector nobody can prove is the feature it was imported as.
@Suite struct NormBackfillPreservesSAEIdentityTests {

    @Test func backfilledSidecarsKeepGemmascopeSource() throws {
        let json = """
            {"modelID":"google/gemma-3-27b-it","concept":"f62389",
             "stimulusSetHash":"-","layerCount":2,"hiddenSize":2,
             "normsPerLayer":[1.0,1.0],
             "extractionDate":"2026-08-13T00:00:00Z",
             "gemmascopeConvention":"residual-norm-match",
             "gemmascopeSource":{"release":"gemma-scope-2-27b-it-res",
               "saeID":"layer_40_width_65k_l0_medium","feature":62389,
               "decoderRowSHA256":"abc123"}}
            """
        let original = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(json.utf8))

        let backfilled = NormBackfill.backfilledSidecar(
            from: original, residualNormPerLayer: [12, 14],
            corpusHash: "corpus-hash",
            sourceArtifact: "runs/x/f62389", sourceVectorsHash: "deadbeef")

        // Survives the assembly…
        #expect(backfilled.gemmascopeSource == original.gemmascopeSource)
        #expect(backfilled.gemmascopeConvention == "residual-norm-match")
        // …and the re-encode that writes the new artifact.
        let written = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(backfilled))
        guard case .object(let source)? = written.gemmascopeSource else {
            Issue.record("the backfilled artifact lost gemmascopeSource")
            return
        }
        #expect(source["feature"] == .number(62389))
        #expect(source["decoderRowSHA256"] == .string("abc123"))
        // The norms are what backfill was FOR — still filled.
        #expect(written.residualNormPerLayer == [12, 14])
    }
}

/// `selection.devMaxTokens` is the length the sweep's coherence floor was
/// measured at (the c18 lesson). Preserving it is what makes the study-vs-dev
/// length comparison checkable after the fact.
@Suite struct SelectionDevMaxTokensTests {

    @Test func devMaxTokensRoundTripsFromAServerStampedBlock() throws {
        let json = """
            {"sweepRun":"20260813T000000Z-sweep","criterion":{},
             "devPromptsHash":"abc","devMaxTokens":1024,
             "winningCell":{"layer":40,"alpha":0.08},"metrics":{"judgeScore":0.7}}
            """
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.SelectionProvenance.self, from: Data(json.utf8))
        #expect(decoded.devMaxTokens == 1024)
        let round = try JSONDecoder().decode(
            ExperimentManifest.SelectionProvenance.self,
            from: try JSONEncoder().encode(decoded))
        #expect(round.devMaxTokens == 1024)
    }

    @Test func anUnstampedBlockDecodesNilAndAddsNoKey() throws {
        let provenance = ExperimentManifest.SelectionProvenance(
            sweepRun: "r", criterion: .init(), devPromptsHash: "abc",
            winningCell: .init(layer: 1, alpha: 0.1), metrics: [:])
        #expect(provenance.devMaxTokens == nil)
        let text = String(
            decoding: try JSONEncoder().encode(provenance), as: UTF8.self)
        #expect(!text.contains("devMaxTokens"))
    }
}

/// `recordTokenIDs` must survive this engine's round trips for the same reason
/// the SAE keys must (2026-08-15).
///
/// The key is server-authored and server-consumed — this engine neither reads
/// nor writes token ids. But a dropped `recordTokenIDs` un-declares a study's
/// decision that its runs stay exactly replayable, and retention is NOT
/// retroactive: once a run completes without the ids, teacher-forced replay of
/// it is no longer exact (re-tokenizing stored text loses the terminal
/// `<end_of_turn>` the streamer skipped). A Mac-side duplicate that silently
/// cleared the flag would produce a study that looks identical and quietly
/// cannot be re-read.
@Suite(.serialized) struct RecordTokenIDsPassthroughTests {

    private func manifestJSON(_ body: String) -> String {
        """
        {"name":"jspace-pilot","experimentDescription":"d",
         "modelID":"google/gemma-3-27b-it",
         "status":"draft","createdAt":"2026-08-15T00:00:00Z","concepts":[]
         \(body)}
        """
    }

    @Test func theFlagDecodesAndSurvivesReSave() throws {
        let json = manifestJSON(#","recordTokenIDs":true"#)
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(json.utf8))
        #expect(manifest.recordTokenIDs)

        let reencoded = try JSONEncoder().encode(manifest)
        let round = try JSONDecoder().decode(
            ExperimentManifest.self, from: reencoded)
        #expect(round.recordTokenIDs, "re-save dropped recordTokenIDs")
        let text = String(data: reencoded, encoding: .utf8) ?? ""
        #expect(text.contains("recordTokenIDs"))
    }

    @Test func absentDecodesFalseSoLegacyManifestsAreUnchanged() throws {
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(manifestJSON("").utf8))
        // Every manifest written before this key existed is an absent one, and
        // absent must mean "did not retain", never "unknown".
        #expect(manifest.recordTokenIDs == false)
    }
}
