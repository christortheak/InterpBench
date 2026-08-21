import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Cluster-LoRA readiness §0 amendments 1+2 (contract §9): a trained
/// adapter enters a study as a VARIANT, and the manifest is where its
/// training provenance becomes evidence — the dataset manifest joins the
/// freeze `verify()` pin surface, the existing `variantValidity` gate covers
/// evidence-grade adapters, and two non-blocking advisories name an
/// exploratory adapter and a missing ex ante matched control.
///
/// Swift parity for `Server/tests/test_lora_variant_pins.py`, minus the
/// cases that need a SERVER artifact: the adapter training sidecar is
/// produced by the cluster trainer, so this engine verifies its hash only
/// where the file happens to be local (documented asymmetry on
/// `ExperimentStore.trainingProvenanceViolations`).
extension ExperimentStoreTests {

    private func loraSHA256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let datasetManifestRelativePath =
        "adapters/stance-lora-family-v1-manifest.json"

    /// Writes the dataset manifest into the workspace `adapters/` tree (the
    /// Mac workspace is the source of truth for it) and returns its hash.
    @discardableResult
    private func writeDatasetManifest(rows: Int = 8) throws -> String {
        let root = try #require(ExperimentStore.rootOverride)
        let url = root.appending(path: Self.datasetManifestRelativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bytes = Data(
            #"{"bundleID":"stance-lora-family-v1","rows":\#(rows)}"#.utf8)
        try bytes.write(to: url)
        return loraSHA256Hex(bytes)
    }

    private func makeAdapterVariantStudy(
        name: String = "lora-study",
        trainingProvenance: ExperimentManifest.VariantCondition
            .TrainingProvenance? = nil
    ) throws -> ExperimentManifest {
        let root = try #require(ExperimentStore.rootOverride)
        let artifact = ModelVariantArtifact(
            name: "stance-lora",
            baseModelID: "test/model",
            adapters: [
                .init(
                    name: "stance-lora",
                    artifactPath: "runs/2026-08-12-lora/stance-lora-v1",
                    adapterDirectory: "runs/2026-08-12-lora/stance-lora-v1",
                    adapterHash: String(repeating: "ab", count: 32))
            ],
            injections: [],
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "")
        let relativePath = "runs/model-variants/stance-lora/model-variant.json"
        let artifactURL = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: artifactURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        try data.write(to: artifactURL)

        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.variantConditions = [
            .init(
                name: "stance-lora", artifactPath: relativePath,
                artifactHash: loraSHA256Hex(data), artifact: artifact,
                trainingProvenance: trainingProvenance)
        ]
        try ExperimentStore.save(manifest)
        return manifest
    }

    private func fabricateAdapterBatteryEvidence(
        for manifest: ExperimentManifest
    ) throws {
        try fabricateValidationEvidence(
            for: manifest,
            capabilityBattery: [
                .init(
                    condition: "baseline", batteryHash: "bh", total: 10,
                    correct: 10, accuracy: 1),
                .init(
                    condition: "stance-lora", batteryHash: "bh", total: 10,
                    correct: 9, accuracy: 0.9),
            ])
    }

    // MARK: - manifest round-trip

    @Test func trainingProvenanceRoundTripsWithExactJSONSpellings() throws {
        let block = ExperimentManifest.VariantCondition.TrainingProvenance(
            datasetBundleID: "stance-lora-family-v1",
            datasetManifestPath: Self.datasetManifestRelativePath,
            datasetManifestHash: String(repeating: "cd", count: 32),
            adapterSidecarHash: String(repeating: "ef", count: 32),
            evidenceGrade: true,
            matchedControl: .init(
                variant: "stance-lora-control", kind: "shuffledAssistantPairing"))
        let condition = ExperimentManifest.VariantCondition(
            name: "stance-lora", artifactPath: "p", artifactHash: "h",
            artifact: ModelVariantArtifact(
                name: "stance-lora", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: ""),
            trainingProvenance: block)
        let data = try JSONEncoder().encode(condition)
        let text = try #require(String(data: data, encoding: .utf8))
        // The cross-engine key SPELLINGS are the contract.
        for key in [
            "trainingProvenance", "datasetBundleID", "datasetManifestPath",
            "datasetManifestHash", "adapterSidecarHash", "evidenceGrade",
            "matchedControl", "variant", "kind",
        ] {
            #expect(text.contains("\"\(key)\""), "missing key \(key)")
        }
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.VariantCondition.self, from: data)
        #expect(decoded.trainingProvenance == block)
    }

    @Test func absentTrainingProvenanceEncodesNoKey() throws {
        let condition = ExperimentManifest.VariantCondition(
            name: "plain", artifactPath: "p", artifactHash: "h",
            artifact: ModelVariantArtifact(
                name: "plain", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: ""))
        let text = try #require(
            String(data: try JSONEncoder().encode(condition), encoding: .utf8))
        #expect(!text.contains("trainingProvenance"))
    }

    // MARK: - verify(): the dataset manifest is a pinned input

    @Test func datasetManifestDriftIsAVerifyViolation() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash))
            #expect(ExperimentStore.verify(manifest).isEmpty)
            try writeDatasetManifest(rows: 9)  // a row appears after pinning
            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("dataset manifest changed since pinning")
                }, "\(violations)")
        }
    }

    @Test func missingDatasetManifestIsAVerifyViolation() throws {
        try withTempRoot {
            let root = try #require(ExperimentStore.rootOverride)
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash))
            try FileManager.default.removeItem(
                at: root.appending(path: Self.datasetManifestRelativePath))
            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("dataset manifest: file missing at")
                }, "\(violations)")
        }
    }

    @Test func halfPinnedTrainingProvenanceIsAVerifyViolation() throws {
        try withTempRoot {
            try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath))
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("half-pin certifies nothing")
                })
        }
    }

    /// The documented cross-engine asymmetry: the adapter sidecar is a SERVER
    /// artifact, so a pinned-but-locally-absent sidecar is not drift here.
    @Test func absentAdapterSidecarIsNotAVerifyViolationOnThisEngine() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash,
                    adapterSidecarHash: String(repeating: "ef", count: 32)))
            #expect(ExperimentStore.verify(manifest).isEmpty)
        }
    }

    // MARK: - variantValidity gate

    @Test func evidenceGradeAdapterWithoutDatasetPinFailsVariantValidity() throws {
        try withTempRoot {
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(evidenceGrade: true))
            try fabricateAdapterBatteryEvidence(for: manifest)
            do {
                _ = try ExperimentStore.freeze(name: manifest.name)
                Issue.record("expected the variantValidity gate to refuse")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains(
                        "trainingProvenance.datasetManifestHash"))
            }
            // It is a force-skippable EVIDENCE gate, stamped by its existing id.
            let forced = try ExperimentStore.freeze(
                name: manifest.name, force: true)
            #expect(forced.forcedGatesSkipped?.contains("variantValidity") == true)
        }
    }

    @Test func evidenceGradeAdapterWithFullPinsFreezes() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash,
                    evidenceGrade: true,
                    matchedControl: .init(
                        variant: "stance-lora-control",
                        kind: "shuffledAssistantPairing")))
            try fabricateAdapterBatteryEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: manifest.name)
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced != true)
            #expect(
                frozen.variantConditions[0].trainingProvenance?
                    .datasetManifestHash == hash)
        }
    }

    @Test func exploratoryAdapterDoesNotFailVariantValidity() throws {
        try withTempRoot {
            let manifest = try makeAdapterVariantStudy()
            try fabricateAdapterBatteryEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: manifest.name)
            #expect(frozen.status == .frozen)
        }
    }

    // MARK: - advisories (non-blocking)

    @Test func exploratoryAdapterAdvisoryFires() throws {
        try withTempRoot {
            let manifest = try makeAdapterVariantStudy()
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(
                advisories.contains { $0.contains("EXPLORATORY adapter") },
                "\(advisories)")
        }
    }

    @Test func missingMatchedControlAdvisoryFires() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash,
                    evidenceGrade: true))
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(
                advisories.contains { $0.contains("NO matchedControl declared") },
                "\(advisories)")
            #expect(!advisories.contains { $0.contains("EXPLORATORY adapter") })
        }
    }

    @Test func declaredMatchedControlSilencesTheAdvisory() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash,
                    evidenceGrade: true,
                    matchedControl: .init(
                        variant: "stance-lora-control",
                        kind: "shuffledAssistantPairing")))
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(!advisories.contains { $0.contains("matchedControl") })
        }
    }

    // MARK: - pin surface

    @Test func datasetManifestJoinsThePinnedInputSurface() throws {
        try withTempRoot {
            let hash = try writeDatasetManifest()
            let manifest = try makeAdapterVariantStudy(
                trainingProvenance: .init(
                    datasetManifestPath: Self.datasetManifestRelativePath,
                    datasetManifestHash: hash))
            let entries = ExperimentStore.pinnedInputEntries(manifest)
            #expect(
                entries.contains {
                    $0.label
                        == "variant 'stance-lora' training dataset manifest"
                }, "\(entries.map(\.label))")
        }
    }
}
