import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// One out-of-range rule for the residual-norm denominator table, per verb
/// (2026-08-28 mathematical-soundness audit, F7 + F13) — and the two other
/// silent seams that wave closed: SAE latent arms on this engine (F12) and a
/// condition slot naming an unextracted concept (F8).
///
/// Server twin: `Server/tests/test_denominator_table_gate.py`.
///
/// What the audit measured: ONE artifact with a truncated `residualNormPerLayer`
/// behaved four different ways. The server's condition path substituted 0.0 and
/// refused as `degenerateData`; its sweep and variant paths clamped to the last
/// entry and dosed the deepest layers with a shallower layer's number; this
/// engine's condition path clamped too. An EMPTY table clamped to index `[-1]`
/// here — a crash, not a refusal. Nothing pinned any of it.
@Suite struct DenominatorTableGateTests {

    private let hidden = 3
    private let layers = 4

    // MARK: - the load-time gate on real bytes

    private func writeArtifact(
        norms: [Float]?, name: String = "fear"
    ) throws -> URL {
        let vectors = ConceptVectors(
            perLayer: (0 ..< layers).map { _ in [1, 0, 0] })
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model", concept: "fear", stimulusSetHash: "h",
            vectors: vectors, residualNormPerLayer: norms)
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "denom-gate-\(UUID().uuidString)")
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: dir, name: name)
        return dir
    }

    @Test func loadingATruncatedArtifactRefuses() throws {
        let dir = try writeArtifact(norms: [7, 7.5])
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try SteeringVectorStore.load(from: dir, name: "fear")
            Issue.record("a truncated denominator table must refuse at load")
        } catch let error as ResidualNormTableError {
            #expect(error.reason.contains("carries 2 residual norms for 4 layers"))
            #expect(error.reason.contains("backfill-norms"))
        }
    }

    /// Absent stays legal: OptVec, J-lens and Gemma Scope report imports are
    /// BORN with no norms and acquire them through the backfill. Refusing them
    /// here would strand every artifact in three families.
    @Test func loadingANormsLessOrFullArtifactStillWorks() throws {
        let bare = try writeArtifact(norms: nil)
        defer { try? FileManager.default.removeItem(at: bare) }
        let full = try writeArtifact(norms: [7, 7.5, 8, 8.5])
        defer { try? FileManager.default.removeItem(at: full) }

        #expect(try SteeringVectorStore.load(from: bare, name: "fear")
            .sidecar.residualNormPerLayer == nil)
        #expect(try SteeringVectorStore.load(from: full, name: "fear")
            .vectors.layerCount == 4)
    }

    // MARK: - the condition path

    private func extraction(norms: [Float]) -> ExperimentTasks.ConceptExtraction {
        ExperimentTasks.ConceptExtraction(
            result: ExtractionResult(
                vectors: ConceptVectors(
                    perLayer: (0 ..< layers).map { _ in [1, 0, 0] }),
                residualNormPerLayer: norms,
                residualNormSource: "neutral-corpus",
                options: ExtractionOptions(method: .meanDifference)),
            stimuli: nil)
    }

    private func condition(layer: Int) -> ExperimentManifest.Condition {
        .init(
            name: "steered",
            slots: [.init(concept: "fear", layer: layer, alpha: 1)],
            bandWidth: 1, alphaInNormUnits: true)
    }

    @Test func aConditionAtACoveredLayerStillDoses() throws {
        let cells = try ExperimentTasks.injections(
            for: condition(layer: 1),
            extractions: ["fear": extraction(norms: [2, 4, 6, 8])])
        #expect(cells.map(\.layer) == [1])
        #expect(cells[0].alpha == 4)
    }

    @Test func aConditionPastATruncatedTableRefuses() throws {
        do {
            _ = try ExperimentTasks.injections(
                for: condition(layer: 3),
                extractions: ["fear": extraction(norms: [2, 4])])
            Issue.record("a layer past the denominator table must refuse")
        } catch let error as ExperimentError {
            #expect(
                error.reason == """
                    condition 'steered': 'fear' has no residual norm at layer \
                    3 — its denominator table covers 2 layer(s), so an α in \
                    residual-norm units cannot be denominated there; \
                    re-measure the norms (vectors backfill-norms), or switch \
                    α to raw units
                    """)
        }
    }

    /// The crash the audit found: `min(layer, count - 1)` against an EMPTY
    /// table is `-1`, which is a fatal index here where the server merely
    /// substituted 0.0. A refusal on both engines now.
    @Test func aConditionAgainstAnEmptyTableRefusesInsteadOfCrashing() throws {
        do {
            _ = try ExperimentTasks.injections(
                for: condition(layer: 0),
                extractions: ["fear": extraction(norms: [])])
            Issue.record("an empty denominator table must refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("covers 0 layer(s)"))
        }
    }

    /// F8's Swift half, pinned: this engine has always thrown here, and the
    /// server now throws the byte-identical sentence instead of silently
    /// dropping the slot.
    @Test func aConditionNamingAnUnextractedConceptRefuses() throws {
        do {
            _ = try ExperimentTasks.injections(
                for: condition(layer: 0), extractions: [:])
            Issue.record("an unextracted concept must refuse")
        } catch let error as ExperimentError {
            #expect(
                error.reason
                    == "condition 'steered' references unextracted concept 'fear'")
        }
    }

    // MARK: - the variant path

    private func variant(layer: Int, id: String) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "v", baseModelID: "test/model",
            injections: [
                .init(concept: "fear", vectorArtifactID: id, layer: layer, alpha: 1)
            ],
            bandWidth: 1, alphaInNormUnits: true,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
    }

    /// The variant path clamped, so the same manifest that refused as a
    /// condition ran here with the wrong dose. The truncated table is now
    /// stopped at load — one gate, every verb.
    @Test func aVariantOnATruncatedArtifactRefuses() throws {
        let dir = try writeArtifact(norms: [7, 7.5])
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try ExperimentTasks.injections(
                for: variant(layer: 3, id: dir.appending(component: "fear").path))
            Issue.record("a truncated denominator table must refuse")
        } catch let error as ResidualNormTableError {
            #expect(error.reason.contains("carries 2 residual norms for 4 layers"))
        }
    }

    @Test func aVariantAtACoveredLayerStillDoses() throws {
        let dir = try writeArtifact(norms: [2, 4, 6, 8])
        defer { try? FileManager.default.removeItem(at: dir) }
        let cells = try ExperimentTasks.injections(
            for: variant(layer: 2, id: dir.appending(component: "fear").path))
        #expect(cells.map(\.layer) == [2])
        #expect(cells[0].alpha == 6)
    }

    // MARK: - F12: SAE latent arms do not execute here

    private func latentManifest(arms: Int) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "latent", description: "", modelID: "test/model")
        manifest.concepts = [
            ExperimentManifest.ConceptRef(
                name: "fear", stimulusSetHash: "h",
                options: ExtractionOptions(method: .meanDifference))
        ]
        manifest.saeLatentConditions = .array(
            (0 ..< arms).map { .object(["name": .string("latent-\($0)")]) })
        return manifest
    }

    /// A latent-ONLY manifest ran BASELINE ALONE here: the arms suppress the
    /// baseline-only refusal (they are legitimate arms — on the server), and
    /// no local run path reads them. The 2026-08-11 declared-studyType
    /// incident through a different door.
    @Test func aLatentOnlyManifestRefusesToRunLocally() {
        let manifest = latentManifest(arms: 2)
        // The suppression is real and stays: the count is what silenced the
        // baseline-only refusal, and that behaviour is correct on the engine
        // that runs latent arms.
        #expect(ExperimentStore.noMeasuredConditionsProblem(manifest) == nil)
        let problem = ExperimentStore.latentArmsNotExecutableProblem(manifest)
        #expect(problem?.contains("2 SAE latent arm(s)") == true)
        #expect(problem?.contains("Python server engine only") == true)
        #expect(problem?.contains("BASELINE") == true)
    }

    /// The ruling: MIXED manifests refuse too. A run that executed the
    /// ordinary arms and skipped the latent ones would leave a run directory
    /// that looks complete and says nothing about the arms it dropped —
    /// exactly the failure this family of refusals exists to stop.
    @Test func aMixedManifestRefusesToRunLocallyAsWell() {
        var mixed = latentManifest(arms: 1)
        mixed.conditions = [
            .init(name: "fear-hi",
                  slots: [.init(concept: "fear", layer: 2, alpha: 1)])
        ]
        #expect(ExperimentStore.latentArmsNotExecutableProblem(mixed) != nil)
    }

    /// No latent arms, nothing to say — and a panel (multi-agent) study never
    /// trips this, because latent arms are model-output configuration.
    @Test func manifestsWithoutLatentArmsAreUntouched() {
        var plain = latentManifest(arms: 0)
        #expect(ExperimentStore.latentArmsNotExecutableProblem(plain) == nil)
        plain.saeLatentConditions = nil
        #expect(ExperimentStore.latentArmsNotExecutableProblem(plain) == nil)

        var panel = latentManifest(arms: 2)
        panel.studyKind = .multiAgent
        #expect(ExperimentStore.latentArmsNotExecutableProblem(panel) == nil)
    }
}
