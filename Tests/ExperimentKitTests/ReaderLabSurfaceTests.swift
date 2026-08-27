import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// What the Concept Lab's reader flow SHOWS, tested where it is written.
///
/// The pane is thin by rule (`ConceptsPanelView` renders `ConceptBuilder`
/// state), so every string a researcher reads about a reader — the split
/// preview, the fit-score cells, the artifact's stamps, the derived vector's
/// provenance — is a pure projection on the builder and is asserted here. The
/// two things this suite is watching for:
///
/// - a REFUSAL the pane invented rather than quoted (the engine owns those
///   literals, on both sides of the cross-engine twin), and
/// - a legacy schema-1 artifact silently rendered as if it were schema 2.
@Suite(.serialized) @MainActor
struct ReaderLabSurfaceTests {

    /// A builder needs a workspace to scan; these tests are about the strings,
    /// so the workspace is an empty temp one.
    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "reader-lab") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(root)
        }
    }

    // MARK: - Fixtures

    private static let stancePair = RepEReader.TaskTemplate(
        id: "instructed-stance-pair-v1", conceptSlot: false,
        text: "{{instruction}}\nScenario: {{stimulus}}\nThe described state is",
        latToken: "final", hash: "th",
        instructionPair: RepEReader.TaskTemplate.InstructionPair(
            experimental: "T plus", reference: "T minus"))

    private static let plain = RepEReader.TaskTemplate(
        id: "unnamed-scenario-v1", conceptSlot: false,
        text: "Scenario: {{stimulus}}\nThe intensity is", latToken: "final",
        hash: "th", divergence: "unnamed-clean-room")

    private func artifact(
        layer: Int = 3,
        contrastMode: RepEReader.ContrastMode = .supervisedContent,
        signConvention: RepEReader.SignConvention = .heldOutPairAgreement,
        signHeldOutAccuracy: Float? = 1,
        signFallbackReason: String? = nil,
        explainedVariance: Float? = 0.62,
        explainedVarianceBasis: String? = nil,
        orientationSeed: UInt64? = nil,
        recommendedLayer: Int? = 3,
        rendering: ExtractionRendering? = nil,
        template: RepEReader.TaskTemplate = ReaderLabSurfaceTests.plain
    ) -> RepEReader.Artifact {
        var made = RepEReader.Artifact(
            modelID: "mlx-community/gemma-3-4b-it-4bit", revision: "abc123",
            concept: "candour", layer: layer, template: template,
            datasetHash: "0123456789abcdef0123",
            probe: SteeringVectorMath.ScalarProbe(
                direction: [1, 0, 0], activationCenter: [0, 0, 0],
                projectionCenter: 0, projectionScale: 1,
                orientation: 1, positiveMean: 1, negativeMean: -1),
            differenceCloudExplainedVariance: explainedVariance,
            explainedVarianceBasis: explainedVarianceBasis,
            trainAccuracy: 0.9, heldOutAccuracy: 0.75,
            trainPairCount: 8, heldOutPairCount: 4,
            contrastMode: contrastMode,
            signConvention: signConvention,
            signHeldOutAccuracy: signHeldOutAccuracy,
            signFallbackReason: signFallbackReason,
            orientationSeed: orientationSeed,
            extractionRendering: rendering)
        made.recommendedLayer = recommendedLayer
        made.recommendedLayerAccuracy = 0.75
        made.layerRecommendationBasis = "heldOutAccuracy"
        return made
    }

    // MARK: - Item 3: the held-out split, with the minimum-2 rule visible

    @Test func splitPreviewNamesTheHeldOutRowsAndTheClamp() {
        let preview = ConceptBuilder.readerSplitPreview(
            concept: "candour", rowCount: 6, requestedHeldOut: 2,
            rowShape: .contentPair)
        #expect(preview.trainRows == 4)
        #expect(preview.heldOutRows == 2)
        #expect(!preview.wasClamped)
        #expect(!preview.signSelectionWillFallBack)
        // WHICH rows: the tail, by the ids the encoder will actually write.
        #expect(preview.heldOutRowIDs == ["candour-pair-4", "candour-pair-5"])
        #expect(preview.note.contains("4 train / 2 held out"))
        #expect(preview.note.contains("held-out sign selection"))
    }

    @Test func splitPreviewSaysWhenTheSignRuleWillStandDown() {
        // One held-out row is below the engine's own minimum, so the fit will
        // fall back to train-label majority and stamp a reason. The pane says
        // so BEFORE the fit, not after.
        let preview = ConceptBuilder.readerSplitPreview(
            concept: "candour", rowCount: 5, requestedHeldOut: 1,
            rowShape: .singleStimulus)
        #expect(preview.signSelectionWillFallBack)
        #expect(preview.heldOutRowIDs == ["candour-row-4"])
        #expect(
            preview.note.contains(
                "\(RepEReader.minimumHeldOutPairsForSignSelection)-pair minimum"))
        #expect(preview.note.contains("train-label majority"))
    }

    @Test func splitPreviewClampProtectsTwoTrainRows() {
        let preview = ConceptBuilder.readerSplitPreview(
            concept: "candour", rowCount: 3, requestedHeldOut: 99,
            rowShape: .contentPair)
        #expect(preview.heldOutRows == 1)
        #expect(preview.trainRows == 2)
        #expect(preview.wasClamped)
        #expect(preview.note.contains("clamped from 99"))
    }

    // MARK: - Item 2: single-stimulus rows, and the request that carries them

    @Test func stimulusRowsWriteTheSingleStimulusShape() throws {
        let lines = try ConceptBuilder.readerStimulusRows(
            concept: "candour", stimuli: ["s0", "s1", "s2"],
            heldOutPairCount: 1, templateID: "instructed-stance-pair-v1")
        #expect(
            lines == [
                #"{"concept":"candour","id":"candour-row-0","split":"train","stimulus":"s0","templateID":"instructed-stance-pair-v1"}"#,
                #"{"concept":"candour","id":"candour-row-1","split":"train","stimulus":"s1","templateID":"instructed-stance-pair-v1"}"#,
                #"{"concept":"candour","id":"candour-row-2","split":"test","stimulus":"s2","templateID":"instructed-stance-pair-v1"}"#,
            ])
        // Round-trips through the shared loader as the shape a T+/T− template
        // needs — which is the whole point of authoring it.
        let dataset = try RepEReader.parsePairs(
            Data((lines.joined(separator: "\n") + "\n").utf8), source: "test")
        #expect(dataset.shape == .singleStimulus)
        #expect(dataset.train.count == 2)
        #expect(dataset.heldOut.count == 1)
        #expect(
            try RepEReader.resolveContrastMode(
                dataset: dataset, template: Self.stancePair)
                == .unsupervisedTemplatePair)
    }

    @Test func singleStimulusRequestUsesTheStimulusEncoder() throws {
        let request = try ConceptBuilder.readerFitRequest(
            concept: "candour",
            positives: [], negatives: [],
            heldOutPairCount: 1,
            registryTemplateID: "instructed-stance-pair-v1",
            customTemplateText: nil,
            rowShape: .singleStimulus,
            stimuli: ["s0", "s1", "s2"])
        #expect(request.templateID == "instructed-stance-pair-v1")
        #expect(request.pairsJSONL.contains(#""stimulus":"s0""#))
        #expect(!request.pairsJSONL.contains("positiveStimulus"))
        #expect(request.pairsJSONL.hasSuffix("\n"))
    }

    /// Item 5 + the seed: both ride the request, and both are ABSENT by
    /// default so an untouched panel posts the bytes it always posted.
    @Test func renderingAndSeedRideTheRequestAndDefaultToAbsent() throws {
        let bare = try ConceptBuilder.readerFitRequest(
            concept: "candour", positives: ["p0", "p1"], negatives: ["n0", "n1"],
            heldOutPairCount: 0, registryTemplateID: "unnamed-scenario-v1",
            customTemplateText: nil)
        #expect(bare.extractionRendering == nil)
        #expect(bare.orientationSeed == nil)

        let declared = try ConceptBuilder.readerFitRequest(
            concept: "candour", positives: [], negatives: [],
            heldOutPairCount: 1, registryTemplateID: "instructed-stance-pair-v1",
            customTemplateText: nil,
            rowShape: .singleStimulus, stimuli: ["s0", "s1", "s2"],
            extractionRendering: ExtractionRendering(mode: .chatTemplate),
            orientationSeed: 42)
        #expect(declared.extractionRendering?.mode == .chatTemplate)
        #expect(declared.orientationSeed == 42)
        // The declaration must not perturb the dataset bytes — the hash is the
        // dataset's, not the fit's.
        let undeclared = try ConceptBuilder.readerFitRequest(
            concept: "candour", positives: [], negatives: [],
            heldOutPairCount: 1, registryTemplateID: "instructed-stance-pair-v1",
            customTemplateText: nil,
            rowShape: .singleStimulus, stimuli: ["s0", "s1", "s2"])
        #expect(declared.pairsJSONL == undeclared.pairsJSONL)
    }

    // MARK: - Item 1: the mismatch surfaces the ENGINE's refusal

    @Test func templatePairAgainstContentPairsQuotesTheEngineRefusal() throws {
        // The pane's refusal must be the engine's, character for character —
        // so assert it against the engine call itself rather than a literal
        // copied into the test.
        var engineText = ""
        do {
            _ = try RepEReader.resolveContrastMode(
                shape: .contentPair, template: Self.stancePair)
            Issue.record("a T+/T− template against content pairs must refuse")
        } catch let error as RepEReader.ReaderError {
            engineText = error.reason
        }
        #expect(engineText.contains("a second stimulus would be a confound"))

        try withTempWorkspace { _ in
            let builder = ConceptBuilder()
            builder.useCustomReaderTemplate = true
            // A custom scaffold can never declare an instructionPair, so the
            // reverse mismatch is reachable from the custom path too.
            builder.customReaderTemplateText = "Scenario: {{stimulus}} —"
            builder.readerRowShape = .singleStimulus
            let refusal = try #require(builder.readerContrastRefusal)
            #expect(refusal.contains("nothing to contrast the stimulus against"))
            #expect(refusal.contains("choose a template-pair template"))
            #expect(builder.readerContrastMode == nil)
        }
    }

    @Test func customTemplateSchemaRefusalIsTheEngineLoadersOwnWords() {
        withTempWorkspace { _ in
            let builder = ConceptBuilder()
            builder.useCustomReaderTemplate = true
            // An {{instruction}} slot with nothing to fill it: `loadTemplate`'s
            // own validation, reached through the same path the fit reaches it.
            builder.customReaderTemplateText =
                "{{instruction}}\nScenario: {{stimulus}} —"
            let refusal = builder.readerTemplateRefusal ?? ""
            #expect(refusal.contains("declares no instructionPair"))
            #expect(refusal.contains("nothing would fill it"))
            #expect(builder.resolvedReaderTemplate == nil)
        }
    }

    /// Item 5's gate. A rendering refusal must stop a LOCAL fit (it would
    /// otherwise fit raw under a panel showing a chat template), must NOT stop
    /// a SERVER fit when it is this engine's own limit naming the server as
    /// the repair, and must never be joined by the reading-position refusal —
    /// a reader reads at its template's LAT token, so that declaration is not
    /// its business.
    @Test func renderingRefusalGatesTheLocalFitButNotTheServerOne() throws {
        try withTempWorkspace { _ in
            let builder = ConceptBuilder()
            builder.useCustomReaderTemplate = true
            builder.customReaderTemplateText = "Scenario: {{stimulus}} —"
            builder.readerRowShape = .contentPair
            #expect(builder.readerFitRefusal(onServer: false) == nil)

            // The assistant voice: MLXLMCommon exposes only the
            // generation-prompt form, so this engine refuses and names the one
            // that can.
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate, voice: .assistant)
            let local = try #require(builder.readerFitRefusal(onServer: false))
            #expect(local == builder.extractionRenderingRefusal)
            #expect(builder.extractionRenderingRefusalIsLocalEngineLimit)
            #expect(builder.readerFitRefusal(onServer: true) == nil)

            // A reading-position refusal is not the reader's business.
            builder.extractionRenderingChoice = ExtractionRenderingChoice()
            #expect(builder.readerFitRefusal(onServer: false) == nil)
        }
    }

    @Test func templateMenuMarksThePairTemplatesRatherThanHidingThem() {
        #expect(
            ConceptBuilder.readerTemplateMenuLabel(Self.stancePair)
                == "instructed-stance-pair-v1 — T+/T− pair")
        #expect(
            ConceptBuilder.readerTemplateMenuLabel(Self.plain)
                == "unnamed-scenario-v1")
    }

    // MARK: - Item 4: per-layer cells and set stamps

    @Test func layerCellsCarryTheBasisAndTheSignRule() {
        let held = ConceptBuilder.ReaderLayerScore(artifact: artifact(layer: 3))
        #expect(
            ConceptBuilder.readerLayerCells(held)
                == ["3 ★", "90%", "75%", "0.62", "diffs", "held-out 100%"])

        let fell = ConceptBuilder.ReaderLayerScore(
            artifact: artifact(
                layer: 4, signConvention: .trainMajority,
                signHeldOutAccuracy: nil,
                signFallbackReason: "no held-out pairs",
                recommendedLayer: 3))
        #expect(ConceptBuilder.readerLayerCells(fell)[0] == "4")
        #expect(ConceptBuilder.readerLayerCells(fell)[5] == "train")

        // A degenerate cloud prints "—", never "0.00": PC1's share of nothing
        // is undefined, and 0 would say the opposite of what it means.
        let degenerate = ConceptBuilder.ReaderLayerScore(
            artifact: artifact(layer: 5, explainedVariance: nil, recommendedLayer: 3))
        #expect(ConceptBuilder.readerLayerCells(degenerate)[3] == "—")
        #expect(ConceptBuilder.readerLayerCells(degenerate)[4] == "degenerate")

        #expect(ConceptBuilder.readerLayerColumns.count
            == ConceptBuilder.readerLayerCells(held).count)
    }

    @Test func signFallbackReasonsAreDeduplicatedInLayerOrder() {
        let reason = "held-out pairs split evenly (2 for, 2 against)"
        let scores = [
            ConceptBuilder.ReaderLayerScore(artifact: artifact(layer: 0)),
            ConceptBuilder.ReaderLayerScore(
                artifact: artifact(
                    layer: 1, signConvention: .trainMajority,
                    signHeldOutAccuracy: nil, signFallbackReason: reason)),
            ConceptBuilder.ReaderLayerScore(
                artifact: artifact(
                    layer: 2, signConvention: .trainMajority,
                    signHeldOutAccuracy: nil, signFallbackReason: reason)),
        ]
        #expect(ConceptBuilder.readerSignFallbackReasons(scores) == [reason])
    }

    @Test func fitStampsComeOffTheArtifactsAndSayRecommendation() throws {
        let stamps = try #require(
            ConceptBuilder.readerFitStamps(from: [
                artifact(
                    layer: 3, contrastMode: .unsupervisedTemplatePair,
                    orientationSeed: 231_001_405,
                    rendering: ExtractionRendering(mode: .chatTemplate),
                    template: Self.stancePair)
            ]))
        #expect(stamps.contrastMode == .unsupervisedTemplatePair)
        #expect(stamps.orientationSeed == 231_001_405)
        #expect(stamps.orientationSeedLine?.contains("SEEDED") == true)
        #expect(!stamps.hasLegacyVarianceBasis)
        let line = try #require(stamps.recommendationLine)
        #expect(line.contains("recommended layer 3"))
        // The artifact's OWN sentence, not a paraphrase — the pane must never
        // read a recommendation as a selection.
        #expect(line.contains(RepEReader.Artifact.layerRecommendationNote))
        #expect(line.contains("never selected automatically"))
    }

    /// A pre-2026-08-27 artifact keeps its own semantics on screen: its
    /// variance number was measured over the ± alternated rows, and the basis
    /// cell says so instead of letting it be compared with a schema-2 number.
    @Test func legacySchemaOneArtifactShowsItsLegacyBasis() throws {
        let legacy = artifact(
            layer: 2, contrastMode: .supervisedContent,
            signConvention: .trainMajority, signHeldOutAccuracy: nil,
            explainedVariance: 0.98, explainedVarianceBasis: "alternatedRows",
            recommendedLayer: nil)
        let score = ConceptBuilder.ReaderLayerScore(artifact: legacy)
        #expect(score.explainedVarianceIsLegacyBasis)
        #expect(ConceptBuilder.readerLayerCells(score)[4] == "alternated (legacy)")
        #expect(score.explainedVarianceBasisLabel.contains("legacy schema-1 basis"))
        #expect(score.explainedVarianceBasisLabel.contains("not comparable"))

        let stamps = try #require(ConceptBuilder.readerFitStamps(from: [legacy]))
        #expect(stamps.hasLegacyVarianceBasis)
        #expect(stamps.recommendationLine == nil)
        #expect(stamps.renderingLabel == ConceptBuilder.renderingLabel(nil))

        let details = ConceptBuilder.readerArtifactDetails(legacy)
        let variance = try #require(
            details.first { $0.label == "PC1 explained variance" })
        #expect(variance.isCaution)
        #expect(variance.value.contains("ALTERNATED"))
        // An absent rendering stamp is shown as the raw default it decodes to,
        // not as a blank.
        let rendering = try #require(
            details.first { $0.label == "extraction rendering" })
        #expect(rendering.value.contains("absent stamp = raw"))
        // Legacy defaults appear as themselves, and train-majority is a
        // caution — it is not the paper's rule.
        let sign = try #require(details.first { $0.label == "sign convention" })
        #expect(sign.isCaution)
        #expect(sign.value.contains("train-label majority"))
    }

    @Test func artifactDetailsQuoteTheDivergenceAndTheFallbackReason() throws {
        let reason = "no held-out pairs (every row is split 'train')"
        let details = ConceptBuilder.readerArtifactDetails(
            artifact(
                signConvention: .trainMajority, signHeldOutAccuracy: nil,
                signFallbackReason: reason))
        #expect(details.first { $0.label == "sign fallback" }?.value == reason)
        #expect(details.first { $0.label == "sign fallback" }?.isCaution == true)
        #expect(
            details.first { $0.label == "template divergence" }?.value
                == "unnamed-clean-room")
        #expect(
            details.first { $0.label == "binding" }?.value.contains("swift-mlx")
                == true)
    }

    // MARK: - Item 6: derive → backfill → attach

    private func derivedSidecar(backfilled: Bool, orientation: Float = 1) throws
        -> SteeringVectorSidecar
    {
        var reader = artifact(layer: 3, contrastMode: .unsupervisedTemplatePair)
        reader.probe = SteeringVectorMath.ScalarProbe(
            direction: [1, 0, 0], activationCenter: [0, 0, 0],
            projectionCenter: 0, projectionScale: 1,
            orientation: orientation, positiveMean: orientation,
            negativeMean: -orientation)
        var (_, sidecar) = try RepEReader.deriveSteeringArtifact(
            from: reader, readerFileName: "reader-candour-layer3.json",
            readerBytes: Data("bytes".utf8))
        if backfilled {
            sidecar.residualNormPerLayer = [7, 7.5, 8, 8.5]
            sidecar.residualNormSource = "neutral-corpus"
        }
        return sidecar
    }

    @Test func derivedSummaryShowsTheProvenanceTheSidecarStamps() throws {
        let summary = ConceptBuilder.derivedReaderVectorSummary(
            sidecar: try derivedSidecar(backfilled: true, orientation: -1),
            reference: "runs/20260827T090000000-derive/candour-repe-reader")
        let values = Dictionary(
            uniqueKeysWithValues: summary.details.map { ($0.label, $0.value) })
        #expect(values["reader layer"] == "3")
        #expect(values["control mode"] == RepEReader.controlMode)
        #expect(values["source"] == RepEReader.artifactType)
        #expect(values["reader template"]?.hasPrefix("unnamed-scenario-v1") == true)
        #expect(
            values["contrast mode"]
                == RepEReader.ContrastMode.unsupervisedTemplatePair.label)
        #expect(
            values["sign convention"]
                == RepEReader.SignConvention.heldOutPairAgreement.label)
        // This reader was signed by HELD-OUT pair agreement, so the fitted
        // direction ships unflipped and the row says whose sign the bytes
        // carry — not that an orientation was folded into them (review round
        // 6, finding 1).
        #expect(
            values["probe orientation"]?
                .contains("the held-out split signed this vector") == true)
        #expect(values["train/held-out sign"]?.hasPrefix("DISAGREE") == true)
    }

    /// The refusal is not composed by the pane: it is the literal
    /// `attachArtifact` raises, so the missing lifecycle step reads the same
    /// in the Concept Lab as it does at the attach that would refuse.
    @Test func unbackfilledDerivedVectorQuotesTheAttachRefusalVerbatim() throws {
        let reference = "runs/20260827T090000000-derive/candour-repe-reader"
        let owed = ConceptBuilder.derivedReaderVectorSummary(
            sidecar: try derivedSidecar(backfilled: false), reference: reference)
        #expect(!owed.isAttachable)
        #expect(
            owed.normBackfillRefusal
                == ExperimentStore.readerDerivedNormBackfillRefusal(artifact: reference))
        #expect(owed.attachabilityNote.contains("measure norms first"))

        let ready = ConceptBuilder.derivedReaderVectorSummary(
            sidecar: try derivedSidecar(backfilled: true), reference: reference)
        #expect(ready.isAttachable)
        #expect(ready.normBackfillRefusal == nil)
        // "attachable" is a claim about the vocabulary, and it names the
        // reason it is true.
        #expect(ready.attachabilityNote.contains("repeReaderLAT"))
        #expect(ready.attachabilityNote.contains("no source concept"))
        #expect(
            ExtractionMethod(rawValue: "repeReaderLAT") == .repeReaderLAT)
    }
}
