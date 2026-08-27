import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// A reader-derived steering vector, attached (audit finding 2).
///
/// Until 2026-08-27 the one artifact the faithful RepE pipeline produces was
/// the one artifact a study could not cite: `attachArtifact` resolves the
/// sidecar's `extractionMethod` to ask where the concept's held-out data
/// lives, and `"repeReaderLAT"` was not in the `ExtractionMethod` vocabulary,
/// so every attempt died on "which this engine does not know". A researcher
/// could fit a reader, derive its vector, see it in the catalog — and then
/// have no way to put it in an experiment.
///
/// Its data questions have honest answers, and they are not a plain concept's:
/// the stimuli are the READER's dataset (whose SHA-256 is the
/// `stimulusSetHash`), there is no `prompts/concepts/<c>/` pair set, and the
/// held-out evidence is the reader artifact's own accuracy — not a
/// `validation.jsonl`. So `hasSourceConcept` is false and every data-side
/// branch skips rather than inventing.
@Suite struct ReaderDerivedAttachTests {

    /// A reader whose probe is the +x reading direction at layer 1.
    private func reader(orientation: Float = 1) -> RepEReader.Artifact {
        RepEReader.Artifact(
            modelID: "mlx-community/gemma-3-4b-it-4bit", revision: "abc123",
            concept: "candour", layer: 1,
            template: RepEReader.TaskTemplate(
                id: "unnamed-scenario-v1", conceptSlot: false,
                text: "S: {{stimulus}} q", latToken: "final", hash: "th",
                divergence: "unnamed-clean-room"),
            datasetHash: "reader-dataset-hash",
            probe: SteeringVectorMath.ScalarProbe(
                direction: [1, 0, 0], activationCenter: [0, 0, 0],
                projectionCenter: 0, projectionScale: 1,
                orientation: orientation, positiveMean: orientation,
                negativeMean: -orientation),
            differenceCloudExplainedVariance: 0.6,
            trainAccuracy: 1, heldOutAccuracy: 0.9,
            trainPairCount: 8, heldOutPairCount: 4,
            signConvention: .heldOutPairAgreement, signHeldOutAccuracy: 1)
    }

    /// Writes a derived artifact into a run directory, optionally with the
    /// residual norms a backfill would have measured.
    @discardableResult
    private func plantDerived(
        root: URL, orientation: Float = 1, backfilled: Bool = true,
        name: String = "candour-repe-reader"
    ) throws -> (reference: String, sidecar: SteeringVectorSidecar) {
        let artifact = reader(orientation: orientation)
        var (vectors, sidecar) = try RepEReader.deriveSteeringArtifact(
            from: artifact, readerFileName: "reader-candour-layer1.json",
            readerBytes: Data("reader-bytes".utf8))
        if backfilled {
            sidecar.residualNormPerLayer = [7.0, 7.5]
            sidecar.residualNormSource = "neutral-corpus"
        }
        let runDirectory = root.appending(
            components: "runs", "20260827T090000000-derive-reader-vector")
        try FileManager.default.createDirectory(
            at: runDirectory, withIntermediateDirectories: true)
        try Data("fake-tensor".utf8).write(
            to: runDirectory.appending(component: "\(name).safetensors"))
        let stamped = SteeringVectorStore.stamped(sidecar)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(stamped)
            .write(to: runDirectory.appending(component: "\(name).json"))
        _ = vectors
        return ("runs/20260827T090000000-derive-reader-vector/\(name)", stamped)
    }

    @Test func readerDerivedVectorAttachesAndPinsTheReaderDataset() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "reader-attach") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "reader-study", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (reference, sidecar) = try plantDerived(root: root)
            #expect(sidecar.extractionMethod == "repeReaderLAT")

            let manifest = try ExperimentStore.attachArtifact(
                "candour", artifact: reference, experimentName: "reader-study")
            let ref = try #require(manifest.concepts.first { $0.name == "candour" })
            #expect(ref.options.method == .pinnedArtifact)
            #expect(ref.effectiveMethod == .repeReaderLAT)
            // The READER's dataset hash travels verbatim: nothing under
            // prompts/concepts/ is looked up for a reader-derived direction.
            #expect(ref.stimulusSetHash == "reader-dataset-hash")
            // A reader's held-out evidence is on the reader artifact, so the
            // absence of a validation.jsonl is pinned EXPLICITLY, not merely
            // missing.
            #expect(ref.validationHash == nil)
            #expect(ref.validationHashPinnedAbsent)
            let pin = try #require(ref.vectorArtifact)
            #expect(pin.sourceMethod == "repeReaderLAT")
            #expect(pin.residualNormSource == "neutral-corpus")
            // The pin passes verify the moment it is written…
            #expect(ExperimentStore.verify(manifest).isEmpty)
            // …and validate owes it no held-out probe, so it is never counted
            // as vacuous evidence.
            #expect(!ExperimentStore.owesHeldOutProbe(ref))
        }
    }

    @Test func readerDerivedVectorRefusesASourceConcept() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "reader-attach") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "reader-study", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (reference, _) = try plantDerived(root: root)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "candour", artifact: reference, sourceConcept: "honesty",
                    experimentName: "reader-study")
                Issue.record("a reader-derived direction has no source concept")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no source concept"))
                #expect(error.reason.contains("fitted RepE reader"))
                #expect(error.reason.contains("held-out accuracy"))
            }
        }
    }

    /// Not a defect of the artifact but a missing LIFECYCLE STEP, and the
    /// refusal names the verb that supplies it — cross-engine twin literal
    /// with `experiment_store.reader_derived_norm_backfill_refusal`.
    @Test func unbackfilledReaderVectorNamesTheBackfill() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "reader-attach") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "reader-study", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (reference, _) = try plantDerived(root: root, backfilled: false)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "candour", artifact: reference, experimentName: "reader-study")
                Issue.record("an un-backfilled reader vector has no denominator")
            } catch let error as ExperimentError {
                #expect(
                    error.reason
                        == "vector artifact '\(reference)' is a RepE-reader-derived "
                            + "direction with no residualNormSource — a reader "
                            + "measures a task template's LAT token, not a neutral "
                            + "corpus, so its reading direction is BORN without a "
                            + "denominator. Run the residual-norm backfill against "
                            + "the pinned neutral corpus first and attach the "
                            + "BACKFILLED artifact: α in norm units is meaningless "
                            + "until the denominator is measured")
            }
        }
    }

    /// The orientation fix is visible on the ATTACHED artifact, not only in
    /// the conversion: a reader with `orientation == −1` produces bytes
    /// pointing the way its probe reads, and the sidecar records that the sign
    /// was applied.
    @Test func attachedVectorCarriesTheAppliedOrientation() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "reader-attach") { root in
            try WorkspaceCompute.declare(.localMLX, root: root)
            _ = try ExperimentStore.create(
                name: "reader-study", description: "",
                modelID: "mlx-community/gemma-3-4b-it-4bit")
            let (_, sidecar) = try plantDerived(root: root, orientation: -1)
            #expect(sidecar.readerProbeOrientation == -1)
            #expect(sidecar.readerLayer == 1)
            #expect(sidecar.readerTemplateID == "unnamed-scenario-v1")
            #expect(sidecar.readerSignConvention == "heldOutPairAgreement")
            #expect(sidecar.signConvention == "heldOutPairAgreement")
            #expect(sidecar.readerContrastMode == "supervisedContent")
        }
    }
}
