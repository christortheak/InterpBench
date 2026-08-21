import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The decided cross-engine Gemma Scope import convention (WS7.2,
/// "analyzed-vector-norm-match" — this app's historical behavior, now shared
/// with the server): the SAE decoder row is rescaled AT IMPORT to the
/// analyzed concept vector's L2 norm at the report layer, the pre-transform
/// norm is recorded as `rawDecoderNorm`, and unstamped Gemma-Scope-sourced
/// sidecars surface as `artifacts audit` findings (pre-convention import —
/// re-import before evidence use). Python twins: `test_gemma_scope.py`.
/// Pure CPU — `deriveFeatureArtifact` does no I/O and no MLX.
@Suite struct GemmaScopeImportConventionTests {

    private func sourceSidecar() -> SteeringVectorSidecar {
        // 5 layers × 3 dims, with residual-norm calibration to carry over.
        SteeringVectorSidecar(
            modelID: "mlx-community/gemma-3-4b-it-4bit",
            revision: "abc123",
            concept: "french",
            stimulusSetHash: "stim-hash",
            vectors: ConceptVectors(perLayer: [
                [1, 0, 0], [0, 2, 0], [6, 0, 8], [0, 2, 0], [1, 0, 0],
            ]),
            residualNormPerLayer: [7.0, 7.5, 8.0, 8.5, 9.0],
            residualNormSource: "neutral-token-bank")
    }

    private func report(
        sidecar: SteeringVectorSidecar? = nil, layer: Int = 2, norm: Float = 10
    ) throws -> GemmaScopeFeatureReport {
        let source = sidecar ?? sourceSidecar()
        let info = try #require(
            GemmaScopeCatalog.info(for: source.modelID, layerCount: 34))
        return GemmaScopeFeatureReport(
            jobFile: "/tmp/job.json",
            vector: GemmaScopeReportVector(
                concept: source.concept, modelID: source.modelID, layer: layer,
                hiddenSize: source.hiddenSize, norm: norm),
            gemmaScope: info,
            artifactSidecar: source,
            decoderShape: [16384, source.hiddenSize],
            topPositive: [], topNegative: [], topAbsolute: [])
    }

    private func row(_ feature: Int, _ values: [Float]?) -> GemmaScopeFeatureRow {
        GemmaScopeFeatureRow(
            feature: feature, cosine: 0.9, sparsity: nil, decoderValues: values)
    }

    // MARK: - The convention transform

    @Test func importAppliesAnalyzedVectorNormMatch() throws {
        // Known decoder row [3,0,4] (L2 norm exactly 5) + target norm 10 →
        // exactly ×2, at the report layer of a FULL-depth zero artifact.
        // The Python twin asserts these same numbers.
        let (vectors, sidecar) = try GemmaScopeReportCatalog.deriveFeatureArtifact(
            report: try report(), row: row(7, [3, 0, 4]), source: "unit test")

        #expect(vectors.perLayer.count == 5)
        #expect(vectors.perLayer[2] == [6, 0, 8])
        for layer in [0, 1, 3, 4] {
            #expect(vectors.perLayer[layer] == [0, 0, 0])
        }

        // Convention stamp + transform provenance (pinned cross-engine keys).
        #expect(sidecar.gemmascopeConvention == "analyzed-vector-norm-match")
        #expect(sidecar.gemmascopeConvention == GemmaScopeReportCatalog.importConvention)
        #expect(sidecar.rawDecoderNorm == 5)
        #expect(sidecar.gemmascopeTargetNorm == 10)

        // Identity + calibration follow the ANALYZED artifact.
        #expect(sidecar.modelID == "mlx-community/gemma-3-4b-it-4bit")
        #expect(sidecar.revision == "abc123")
        #expect(sidecar.concept == "sae:french:L2:F7")
        #expect(
            sidecar.stimulusSetHash
                == "gemmascope:gemma-scope-2-4b-it-res:layer_17_width_16k_l0_medium:7")
        #expect(sidecar.extractionMethod == "gemmaScopeSAE")
        #expect(sidecar.residualNormPerLayer == [7.0, 7.5, 8.0, 8.5, 9.0])
        #expect(sidecar.residualNormSource == "neutral-token-bank")
        #expect(
            sidecar.recipeHash
                == "gemma-scope-2-4b-it-res|layer_17_width_16k_l0_medium|feature:7")
    }

    @Test func degenerateRowsStayRawWithNormRecorded() throws {
        // Mirrors the server's `_convention_rescale` guard exactly: zero-norm
        // row → saved unscaled, and the stamped rawDecoderNorm (0) says why.
        let (vectors, sidecar) = try GemmaScopeReportCatalog.deriveFeatureArtifact(
            report: try report(), row: row(9, [0, 0, 0]), source: "unit test")
        #expect(vectors.perLayer[2] == [0, 0, 0])
        #expect(sidecar.rawDecoderNorm == 0)
        #expect(sidecar.gemmascopeConvention == GemmaScopeReportCatalog.importConvention)
    }

    @Test func missingDecoderValuesRefuses() throws {
        #expect(throws: GemmaScopeReportImportError.self) {
            try GemmaScopeReportCatalog.deriveFeatureArtifact(
                report: try report(), row: row(7, nil), source: "unit test")
        }
    }

    @Test func dimensionMismatchRefuses() throws {
        #expect(throws: GemmaScopeReportImportError.self) {
            try GemmaScopeReportCatalog.deriveFeatureArtifact(
                report: try report(), row: row(7, [1, 2, 3, 4]), source: "unit test")
        }
    }

    // MARK: - Pre-convention imports surface in the artifact audit

    private func plantArtifact(
        in runsDirectory: URL, name: String, mutate: (inout SteeringVectorSidecar) -> Void
    ) throws {
        var sidecar = SteeringVectorSidecar(
            modelID: "mlx-community/gemma-3-4b-it-4bit",
            concept: "sae:french:L2:F7",
            stimulusSetHash: "gemmascope:rel:sid:7",
            vectors: ConceptVectors(perLayer: [[0, 1]]))
        mutate(&sidecar)
        let runDir = runsDirectory.appending(component: "run-\(name)")
        try FileManager.default.createDirectory(
            at: runDir, withIntermediateDirectories: true)
        // Stub tensor file: scan() lists artifacts by *.safetensors presence
        // but only decodes the sidecar — the audit never reads tensor bytes.
        try Data().write(to: runDir.appending(component: "\(name).safetensors"))
        try JSONEncoder().encode(sidecar)
            .write(to: runDir.appending(component: "\(name).json"))
    }

    @Test func auditFlagsPreConventionGemmaScopeImports() throws {
        let runsDirectory = FileManager.default.temporaryDirectory
            .appending(component: "parity-audit-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runsDirectory) }

        try plantArtifact(in: runsDirectory, name: "legacy") { _ in }
        try plantArtifact(in: runsDirectory, name: "stamped") { sidecar in
            sidecar.gemmascopeConvention = GemmaScopeReportCatalog.importConvention
            sidecar.rawDecoderNorm = 1
            sidecar.gemmascopeTargetNorm = 5
        }

        let findings = VectorCatalog.auditArtifacts(runsDirectory: runsDirectory)
        let conventionFindings = findings.filter {
            $0.issue.contains("pre-convention import")
        }
        #expect(conventionFindings.count == 1)
        let finding = try #require(conventionFindings.first)
        #expect(finding.artifactPath.contains("legacy"))
        #expect(finding.recommendation.contains("analyzed-vector-norm-match"))
        #expect(finding.recommendation.contains("re-import"))
    }

    @Test func auditIgnoresNonGemmaScopeArtifactsForTheConventionCheck() throws {
        let runsDirectory = FileManager.default.temporaryDirectory
            .appending(component: "parity-audit-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: runsDirectory) }

        try plantArtifact(in: runsDirectory, name: "caa") { sidecar in
            sidecar.concept = "fear"
            sidecar.stimulusSetHash = "plain-stimulus-hash"
            // extractionMethod nil, stimulus hash non-gemmascope → not SAE-sourced
        }
        let findings = VectorCatalog.auditArtifacts(runsDirectory: runsDirectory)
        #expect(findings.allSatisfy { !$0.issue.contains("pre-convention import") })
    }
}

/// Cross-engine attach parity for an IMPORTED Gemma Scope feature.
///
/// The importer above stamps `extractionMethod: "gemmaScopeSAE"` into every
/// feature sidecar, and the server's `ExtractionMethod` has carried that
/// member since the SAE work; Swift's did not, so `attachArtifact`'s
/// unknown-method guard refused on the Mac exactly the artifact the server
/// accepted. These fix the vocabulary AND the data-side semantics that
/// follow from it (server twin: the `has_source_concept` branches of
/// `experiment_store.attach_artifact`).
///
/// Serialized: attach moves the process-global workspace root.
@Suite(.serialized) struct GemmaScopeAttachParityTests {

    /// A real imported feature on disk: the sidecar comes from
    /// `deriveFeatureArtifact` (the actual writer), substrate-stamped by
    /// `SteeringVectorStore.stamped` exactly as save() would. The tensor is
    /// a stub — attach hashes the bytes, it never decodes them.
    private func plantImportedFeature(
        root: URL, feature: Int = 7, name: String = "sae-feature-7"
    ) throws -> (reference: String, sidecar: SteeringVectorSidecar) {
        let source = SteeringVectorSidecar(
            modelID: "mlx-community/gemma-3-4b-it-4bit",
            revision: "abc123",
            concept: "french",
            stimulusSetHash: "stim-hash",
            vectors: ConceptVectors(perLayer: [
                [1, 0, 0], [0, 2, 0], [6, 0, 8], [0, 2, 0], [1, 0, 0],
            ]),
            residualNormPerLayer: [7.0, 7.5, 8.0, 8.5, 9.0],
            residualNormSource: "neutral-token-bank")
        let info = try #require(
            GemmaScopeCatalog.info(for: source.modelID, layerCount: 34))
        let report = GemmaScopeFeatureReport(
            jobFile: "/tmp/job.json",
            vector: GemmaScopeReportVector(
                concept: source.concept, modelID: source.modelID, layer: 2,
                hiddenSize: source.hiddenSize, norm: 10),
            gemmaScope: info,
            artifactSidecar: source,
            decoderShape: [16384, source.hiddenSize],
            topPositive: [], topNegative: [], topAbsolute: [])
        let (_, featureSidecar) = try GemmaScopeReportCatalog.deriveFeatureArtifact(
            report: report,
            row: GemmaScopeFeatureRow(
                feature: feature, cosine: 0.9, sparsity: nil,
                decoderValues: [3, 0, 4]),
            source: "unit test")
        let runDirectory = root.appending(
            components: "runs", "20260818T090000000-sae-feature-import")
        try FileManager.default.createDirectory(
            at: runDirectory, withIntermediateDirectories: true)
        try Data("fake-tensor".utf8).write(
            to: runDirectory.appending(component: "\(name).safetensors"))
        let stamped = SteeringVectorStore.stamped(featureSidecar)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(stamped)
            .write(to: runDirectory.appending(component: "\(name).json"))
        return ("runs/20260818T090000000-sae-feature-import/\(name)", stamped)
    }

    @Test func gemmaScopeFeatureAttachesAndPinsTheDictionaryCoordinate() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "gemmascope") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "sae-screen", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (reference, sidecar) = try plantImportedFeature(root: root)
            #expect(sidecar.extractionMethod == "gemmaScopeSAE")

            let manifest = try ExperimentStore.attachArtifact(
                "sae-french-f7", artifact: reference,
                experimentName: "sae-screen")
            let ref = try #require(
                manifest.concepts.first { $0.name == "sae-french-f7" })
            #expect(ref.options.method == .pinnedArtifact)
            #expect(ref.effectiveMethod == .gemmaScopeSAE)
            // The dictionary coordinate travels VERBATIM — nothing under
            // prompts/ is looked up for it.
            #expect(
                ref.stimulusSetHash
                    == "gemmascope:gemma-scope-2-4b-it-res:layer_17_width_16k_l0_medium:7")
            // No held-out validation.jsonl exists for a decoder row: the
            // hash is pinned EXPLICITLY absent, not merely missing.
            #expect(ref.validationHash == nil)
            #expect(ref.validationHashPinnedAbsent)
            let pin = try #require(ref.vectorArtifact)
            #expect(pin.sourceMethod == "gemmaScopeSAE")
            #expect(pin.sourceConcept == "sae-french-f7")
            #expect(pin.residualNormSource == "neutral-token-bank")
            #expect(!pin.sha256TensorHash.isEmpty)
            #expect(!pin.sha256SidecarHash.isEmpty)
            // No optvec provenance is invented for it.
            #expect(pin.optvecLayer == nil)
            #expect(pin.optvecEvalRun == nil)
            // The pin passes verify the moment it is written — the concept
            // loop must not go looking for prompts/concepts/<name>/.
            #expect(ExperimentStore.verify(manifest).isEmpty)
            // And validate owes it no held-out probe (so it is never
            // counted as VACUOUS evidence).
            #expect(!ExperimentStore.owesHeldOutProbe(ref))
        }
    }

    @Test func gemmaScopeFeatureRefusesASourceConcept() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "gemmascope") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "sae-refusal", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (reference, _) = try plantImportedFeature(root: root)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "sae-french-f7", artifact: reference,
                    sourceConcept: "french", experimentName: "sae-refusal")
                Issue.record("a decoder row has no source concept to name")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no source concept"))
                #expect(error.reason.contains("Gemma Scope"))
            }
        }
    }
}
