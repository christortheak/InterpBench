import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// Measurement-side input pins (markers.json / validation.jsonl / neutral-PC
/// basis) and forced-freeze stamping — the 2026-07-13 firewall edge repairs.
/// Declared as an extension of the serialized `ExperimentStoreTests` suite
/// because these tests share its `rootOverride` test seam (a process-global);
/// the markers checks reference the real committed `french` concept
/// read-only.
extension ExperimentStoreTests {

    private func withPinsTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        // Shared cross-suite lock for the process-global rootOverride.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pins", body)
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Plants a grand-mean concept's stories.jsonl (and optionally its
    /// validation.jsonl) under the temp root's prompts/emotions/<name>/.
    private func plantEmotionConcept(
        _ name: String, root: URL, validation: String? = nil
    ) throws {
        let directory = root.appending(components: "prompts", "emotions", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let stories = """
            {"concept": "\(name)", "text": "a story about \(name)"}
            {"concept": "\(name)", "text": "another story about \(name)"}

            """
        try stories.write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
        if let validation {
            try validation.write(
                to: directory.appending(component: "validation.jsonl"),
                atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Cross-engine markers aggregate fixture

    @Test func markersAggregateMatchesCrossEngineFixture() {
        // Shared fixture with the Python engine: fear + calm markers bytes →
        // this exact aggregate. Lines are "<name>\t<sha256hex>\n", names
        // sorted by UTF-8 bytes, aggregate = sha256 of the concatenation.
        let entries: [(name: String, bytes: Data)] = [
            ("fear", Data("{\"words\": [\"dread\", \"terror\"]}\n".utf8)),
            ("calm", Data("{\"words\": [\"serene\"]}\n".utf8)),
        ]
        #expect(
            ExperimentStore.markersAggregateHash(entries)
                == "2bb18ec6b173f139962dc4c4eb982fe76ae35c559123703d852e642102ea4c94")
        // Duplicates collapse; empty input pins nothing.
        #expect(
            ExperimentStore.markersAggregateHash(entries + [entries[0]])
                == ExperimentStore.markersAggregateHash(entries))
        #expect(ExperimentStore.markersAggregateHash([]) == nil)
    }

    // MARK: - ConceptRef three-state validation pin

    @Test func conceptRefValidationHashThreeStateCodable() throws {
        // Legacy fixture: encode a ref with NO validation pin — the key must
        // be absent (legacy manifests keep their content hash).
        let legacyRef = ExperimentManifest.ConceptRef(
            name: "c", stimulusSetHash: "abc", options: .init())
        let legacyEncoded = String(
            decoding: try JSONEncoder().encode(legacyRef), as: UTF8.self)
        #expect(!legacyEncoded.contains("validationHash"))
        let legacy = try JSONDecoder().decode(
            ExperimentManifest.ConceptRef.self, from: Data(legacyEncoded.utf8))
        #expect(legacy.validationHash == nil)
        #expect(!legacy.validationHashPinnedAbsent)

        // Explicit null: pinned-as-absent, round-trips as explicit null.
        let nullJSON = legacyEncoded.replacingOccurrences(
            of: #""stimulusSetHash":"abc""#,
            with: #""stimulusSetHash":"abc","validationHash":null"#)
        let pinnedNull = try JSONDecoder().decode(
            ExperimentManifest.ConceptRef.self, from: Data(nullJSON.utf8))
        #expect(pinnedNull.validationHash == nil)
        #expect(pinnedNull.validationHashPinnedAbsent)
        let nullEncoded = String(
            decoding: try JSONEncoder().encode(pinnedNull), as: UTF8.self)
        #expect(nullEncoded.contains("\"validationHash\":null"))

        // Pinned hash round-trips.
        let pinnedJSON = legacyEncoded.replacingOccurrences(
            of: #""stimulusSetHash":"abc""#,
            with: #""stimulusSetHash":"abc","validationHash":"dd""#)
        let pinned = try JSONDecoder().decode(
            ExperimentManifest.ConceptRef.self, from: Data(pinnedJSON.utf8))
        #expect(pinned.validationHash == "dd")
        #expect(!pinned.validationHashPinnedAbsent)
        #expect(
            String(decoding: try JSONEncoder().encode(pinned), as: UTF8.self)
                .contains("\"validationHash\":\"dd\""))
    }

    @Test func validationPinVerifiesDriftAndAppearance() throws {
        try withPinsTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "val-pin", description: "", modelID: "test/model")
            let validation = #"{"text": "a hidden scenario", "expresses": true}"# + "\n"
            try plantEmotionConcept("serenity", root: root, validation: validation)
            try ExperimentStore.attachGrandMeanConcepts(["serenity"], into: &manifest)

            // Attach pinned the validation file.
            #expect(
                manifest.concepts.first?.validationHash
                    == sha256Hex(Data(validation.utf8)))
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Drift → violation.
            let url = ExperimentStore.emotionsDirectory
                .appending(components: "serenity", "validation.jsonl")
            try (#"{"text": "edited", "expresses": false}"# + "\n").write(
                to: url, atomically: true, encoding: .utf8)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("validation.jsonl changed since pinning")
                })

            // Missing → violation.
            try FileManager.default.removeItem(at: url)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("validation.jsonl missing")
                })
        }
    }

    @Test func validationPinnedAbsentFlagsLateFile() throws {
        try withPinsTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "val-null", description: "", modelID: "test/model")
            try plantEmotionConcept("stoicism", root: root)  // no validation.jsonl
            try ExperimentStore.attachGrandMeanConcepts(["stoicism"], into: &manifest)
            #expect(manifest.concepts.first?.validationHash == nil)
            #expect(manifest.concepts.first?.validationHashPinnedAbsent == true)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // A validation file appearing AFTER attach is drift, not a bonus.
            let url = ExperimentStore.emotionsDirectory
                .appending(components: "stoicism", "validation.jsonl")
            try (#"{"text": "late", "expresses": true}"# + "\n").write(
                to: url, atomically: true, encoding: .utf8)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("appeared after attach")
                })
        }
    }

    // MARK: - markersHash pin

    @Test func markersHashDriftIsAViolation() throws {
        try withPinsTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "markers", description: "", modelID: "test/model")
            // Real committed french concept (has markers.json), read-only.
            let french = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: "french", stimulusSetHash: french.hash, options: .init()))

            // Unpinned (legacy) passes; advisory says re-attach to pin.
            #expect(manifest.markersHash == nil)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Pinned at the live value passes.
            manifest.markersHash = ExperimentStore.liveMarkersHash(manifest)
            #expect(manifest.markersHash != nil)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Pinned at a stale value is drift.
            manifest.markersHash = String(repeating: "0", count: 64)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("markers.json changed since pinning")
                })
        }
    }

    @Test func freezePinsMarkersHashAndAdvisesOnUnpinnedInputs() throws {
        try withPinsTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "markers-freeze", description: "", modelID: "test/model")
            manifest.modelRevision = "deadbeef"
            let french = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: "french", stimulusSetHash: french.hash, options: .init()))
            try ExperimentStore.save(manifest)

            // Pre-freeze: markers exist but are unpinned → advisory.
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest).contains(
                    "measurement-side inputs unpinned (markers/validation) — "
                        + "re-attach to pin"))

            // Fabricate scope-matched validate evidence so a clean freeze
            // works, then freeze: markersHash gets pinned into the frozen
            // manifest and the advisory clears.
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "markers-freeze")
            #expect(frozen.markersHash == ExperimentStore.liveMarkersHash(frozen))
            #expect(frozen.freezeForced == nil)
            #expect(ExperimentStore.verify(frozen).isEmpty)
            #expect(
                !ExperimentStore.freezeAdvisories(for: frozen).contains {
                    $0.contains("measurement-side inputs unpinned")
                })
        }
    }

    // MARK: - Neutral-PC basis pin

    @Test func neutralPCBasisPinVerifiesFileBytes() throws {
        try withPinsTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "basis", description: "", modelID: "test/model")
            let french = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: "french", stimulusSetHash: french.hash, options: .init()))

            // Plant a basis artifact file; the pin is sha256 of FILE BYTES.
            let basisURL = root.appending(component: "neutral-pcs.json")
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            let bytes = Data(#"{"corpusHash": "cafe", "layers": []}"#.utf8)
            try bytes.write(to: basisURL)
            manifest.conditions.append(
                .init(
                    name: "steered",
                    slots: [.init(concept: "french", layer: 3, alpha: 0.1)],
                    neutralPCBasisPath: basisURL.path,
                    neutralPCBasisHash: sha256Hex(bytes)))
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Regenerated basis (different bytes) → drift violation.
            try Data(#"{"corpusHash": "cafe", "layers": [1]}"#.utf8).write(to: basisURL)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("neutral-PC basis changed since pinning")
                })

            // Missing file → violation.
            try FileManager.default.removeItem(at: basisURL)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("neutral-PC basis missing")
                })

            // Hash without a path is an unresolvable pin.
            manifest.conditions = [
                .init(
                    name: "broken",
                    slots: [.init(concept: "french", layer: 3, alpha: 0.1)],
                    neutralPCBasisHash: "aa")
            ]
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("neutral-PC basis hash pinned without a path")
                })

            // Path without a hash stays legacy/clean.
            manifest.conditions = [
                .init(
                    name: "legacy",
                    slots: [.init(concept: "french", layer: 3, alpha: 0.1)],
                    neutralPCBasisPath: basisURL.path)
            ]
            #expect(ExperimentStore.verify(manifest).isEmpty)
        }
    }

    // MARK: - Forced-freeze stamping

    @Test func forcedFreezeStampsSkippedGatesAndStaysNonCitable() throws {
        try withPinsTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "forced", description: "", modelID: "test/model")
            let french = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
            manifest.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: "french", stimulusSetHash: french.hash, options: .init()))
            try ExperimentStore.save(manifest)

            // No revision, no validate evidence: a clean freeze refuses…
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "forced")
            }
            // …and a forced freeze stamps exactly which gates it skipped.
            let frozen = try ExperimentStore.freeze(name: "forced", force: true)
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced == true)
            #expect(frozen.forcedGatesSkipped == ["revision", "validateEvidence"])

            // The stamps are lifecycle metadata: excluded from the content
            // hash and cleared on duplicate.
            var unstamped = frozen
            unstamped.freezeForced = nil
            unstamped.forcedGatesSkipped = nil
            #expect(
                ExperimentStore.manifestHash(frozen)
                    == ExperimentStore.manifestHash(unstamped))
            let copy = try ExperimentStore.duplicate(name: "forced", as: "forced-2")
            #expect(copy.freezeForced == nil)
            #expect(copy.forcedGatesSkipped == nil)

            // Post-freeze verification still passes (the stamps are outside
            // the freeze hash), and the advisory names the forced freeze.
            #expect(ExperimentStore.verify(frozen).isEmpty)
            #expect(
                ExperimentStore.freezeAdvisories(for: frozen).contains(
                    "forced freeze — gates skipped: revision, validateEvidence "
                        + "— non-citable"))
        }
    }
}
