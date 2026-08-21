import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// App gap A2 (science-manifest setters), A7 (revision at create), and A12
/// (draft delete → trash). Every setter is draft-only through the store's
/// one gate; frozen manifests refuse with the immutability line.
@Suite(.serialized) struct ScienceManifestEditorTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sci", body)
    }

    @discardableResult
    private func makeDraft(_ name: String = "draft-a") throws -> ExperimentManifest {
        try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
    }

    /// Flip a draft to frozen on disk (save allows draft → frozen edits).
    private func freezeOnDisk(_ name: String) throws {
        var manifest = try ExperimentStore.load(name: name)
        manifest.status = .frozen
        try ExperimentStore.save(manifest)
    }

    // MARK: A7 — revision at create

    @Test func createAcceptsAnUpfrontRevisionPin() throws {
        try withTempRoot { _ in
            let manifest = try ExperimentStore.create(
                name: "pinned", description: "d", modelID: "test/model",
                modelRevision: "abc123def456")
            #expect(manifest.modelRevision == "abc123def456")
            #expect(try ExperimentStore.load(name: "pinned").modelRevision
                == "abc123def456")
            // Empty stays the auto-pin path: nil in the manifest.
            let auto = try ExperimentStore.create(
                name: "auto", description: "d", modelID: "test/model")
            #expect(auto.modelRevision == nil)
        }
    }

    // MARK: A2 — setters round-trip

    @Test func phaseAndCaseFamilyRoundTripAndClear() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.setPhase("screen", experimentName: "draft-a")
            try ExperimentStore.setCaseFamily(
                "sentencing", experimentName: "draft-a")
            var loaded = try ExperimentStore.load(name: "draft-a")
            #expect(loaded.phase == "screen")
            #expect(loaded.caseFamily == "sentencing")
            try ExperimentStore.setPhase("", experimentName: "draft-a")
            try ExperimentStore.setCaseFamily(nil, experimentName: "draft-a")
            loaded = try ExperimentStore.load(name: "draft-a")
            #expect(loaded.phase == nil)
            #expect(loaded.caseFamily == nil)
        }
    }

    @Test func outcomeInstrumentsValidateAgainstKnownVocabulary() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.setOutcomeInstruments(
                ["sampledText", "answerTokenLogprob"], experimentName: "draft-a")
            #expect(try ExperimentStore.load(name: "draft-a").outcomeInstruments
                == ["sampledText", "answerTokenLogprob"])
            // Unknown instrument refuses, naming the vocabulary.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setOutcomeInstruments(
                    ["vibes"], experimentName: "draft-a")
            }
            // Empty list clears to ABSENT (the engine default).
            try ExperimentStore.setOutcomeInstruments([], experimentName: "draft-a")
            #expect(try ExperimentStore.load(name: "draft-a").outcomeInstruments == nil)
        }
    }

    @Test func samplingPolicyNormalizesAndValidates() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.setSamplingPolicy(
                samplesPerItem: 5, seedPolicy: "derivedSHA256",
                experimentName: "draft-a")
            var loaded = try ExperimentStore.load(name: "draft-a")
            #expect(loaded.samplesPerItem == 5)
            #expect(loaded.seedPolicy == "derivedSHA256")
            // 1 normalizes to absent (content-hash hygiene).
            try ExperimentStore.setSamplingPolicy(
                samplesPerItem: 1, seedPolicy: "", experimentName: "draft-a")
            loaded = try ExperimentStore.load(name: "draft-a")
            #expect(loaded.samplesPerItem == nil)
            #expect(loaded.seedPolicy == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSamplingPolicy(
                    samplesPerItem: 0, seedPolicy: nil, experimentName: "draft-a")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSamplingPolicy(
                    samplesPerItem: 2, seedPolicy: "diceRoll",
                    experimentName: "draft-a")
            }
        }
    }

    @Test func acknowledgeUnequalOptionLengthsStoresTrueOrAbsent() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.setAcknowledgeUnequalOptionLengths(
                true, experimentName: "draft-a")
            #expect(
                try ExperimentStore.load(name: "draft-a")
                    .acknowledgeUnequalOptionLengths == true)
            try ExperimentStore.setAcknowledgeUnequalOptionLengths(
                false, experimentName: "draft-a")
            // Never an explicit false — absent.
            #expect(
                try ExperimentStore.load(name: "draft-a")
                    .acknowledgeUnequalOptionLengths == nil)
        }
    }

    @Test func promotionRuleRoundTripsValidatesAndNormalizesEmpty() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.setPromotionRule(
                .init(
                    fdrThreshold: 0.05, doseMonotone: true,
                    exceedsRandomFloor: true, capabilityGate: "within 0.05"),
                experimentName: "draft-a")
            let loaded = try ExperimentStore.load(name: "draft-a")
            #expect(loaded.promotionRule?.fdrThreshold == 0.05)
            #expect(loaded.promotionRule?.doseMonotone == true)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setPromotionRule(
                    .init(fdrThreshold: 1.5), experimentName: "draft-a")
            }
            // All-empty rule normalizes to ABSENT.
            try ExperimentStore.setPromotionRule(
                .init(), experimentName: "draft-a")
            #expect(try ExperimentStore.load(name: "draft-a").promotionRule == nil)
        }
    }

    @Test func humanBaselinePinHashesTheFileAtSetTime() throws {
        try withTempRoot { root in
            try makeDraft()
            let path = "prompts/baselines/draft-a-human-baseline.csv"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.2,0.1,0.3\n".write(
                to: url, atomically: true, encoding: .utf8)
            let pinned = try ExperimentStore.pinHumanBaseline(
                path: path, experimentName: "draft-a")
            #expect(pinned.path == path)
            #expect(pinned.hash.count == 64)
            #expect(try ExperimentStore.load(name: "draft-a").humanBaseline == pinned)
            // A missing file refuses with a pointer at the readiness path.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinHumanBaseline(
                    path: "prompts/baselines/absent.csv", experimentName: "draft-a")
            }
            try ExperimentStore.clearHumanBaseline(experimentName: "draft-a")
            #expect(try ExperimentStore.load(name: "draft-a").humanBaseline == nil)
        }
    }

    // MARK: A2 — frozen refusal (every setter goes through the one gate)

    @Test func settersRefuseOnFrozenManifests() throws {
        try withTempRoot { _ in
            try makeDraft("frozen-a")
            try freezeOnDisk("frozen-a")
            func expectImmutability(_ body: () throws -> Void) {
                do {
                    try body()
                    Issue.record("expected the frozen refusal")
                } catch let error as ExperimentError {
                    #expect(error.reason.contains("frozen"))
                    #expect(error.reason.contains("duplicate"))
                } catch {
                    Issue.record("unexpected error type: \(error)")
                }
            }
            expectImmutability {
                try ExperimentStore.setPhase("screen", experimentName: "frozen-a")
            }
            expectImmutability {
                try ExperimentStore.setOutcomeInstruments(
                    ["sampledText"], experimentName: "frozen-a")
            }
            expectImmutability {
                try ExperimentStore.setSamplingPolicy(
                    samplesPerItem: 3, seedPolicy: nil, experimentName: "frozen-a")
            }
            expectImmutability {
                try ExperimentStore.setPromotionRule(
                    .init(fdrThreshold: 0.05), experimentName: "frozen-a")
            }
            expectImmutability {
                try ExperimentStore.upsertCondition(
                    .init(name: "x", slots: []), experimentName: "frozen-a")
            }
        }
    }

    // MARK: A12 — draft delete to trash

    @Test func draftDeleteMovesToTrashSiblingAndLeavesTheList() throws {
        try withTempRoot { _ in
            try makeDraft("doomed")
            #expect(ExperimentStore.list().contains { $0.name == "doomed" })
            let destination = try ExperimentStore.moveDraftToTrash(name: "doomed")
            // Nested under experiments/.trash-<timestamp>/<name> — a sibling
            // of the live experiment directories, never resolvable as one.
            #expect(destination.lastPathComponent == "doomed")
            #expect(
                destination.deletingLastPathComponent().lastPathComponent
                    .hasPrefix(".trash-"))
            #expect(
                destination.deletingLastPathComponent().deletingLastPathComponent()
                    .path == ExperimentStore.directory.path)
            // The manifest bytes survive (recoverable), the listing forgets it.
            #expect(
                FileManager.default.fileExists(
                    atPath: destination.appending(component: "experiment.json").path))
            #expect(!ExperimentStore.list().contains { $0.name == "doomed" })
            #expect(throws: (any Error).self) {
                try ExperimentStore.load(name: "doomed")
            }
        }
    }

    @Test func frozenAndCompletedRefuseDeletionWithTheImmutabilityLine() throws {
        try withTempRoot { _ in
            try makeDraft("keep")
            try freezeOnDisk("keep")
            do {
                try ExperimentStore.moveDraftToTrash(name: "keep")
                Issue.record("expected the immutability refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("immutable"))
                #expect(error.reason.contains("frozen"))
            }
            // Still listed, untouched.
            #expect(ExperimentStore.list().contains { $0.name == "keep" })
        }
    }

    // MARK: validation read depth (2026-08-01 — the field the first 27B
    // validates silently defaulted past)

    @Test func validationDepthDeclarationRoundTripsAndClears() throws {
        try withTempRoot { _ in
            try makeDraft("depth")
            try ExperimentStore.setValidationReadDepth(
                layer: nil, fraction: 0.35, experimentName: "depth")
            var loaded = try ExperimentStore.load(name: "depth")
            #expect(loaded.validationLayerFraction == 0.35)
            #expect(loaded.validationLayer == nil)
            // Switching to an index replaces the fraction — the setter
            // writes both fields, so a stale sibling can never linger.
            try ExperimentStore.setValidationReadDepth(
                layer: 21, fraction: nil, experimentName: "depth")
            loaded = try ExperimentStore.load(name: "depth")
            #expect(loaded.validationLayer == 21)
            #expect(loaded.validationLayerFraction == nil)
            // nil/nil is the legacy rule, stored as ABSENT.
            try ExperimentStore.setValidationReadDepth(
                layer: nil, fraction: nil, experimentName: "depth")
            loaded = try ExperimentStore.load(name: "depth")
            #expect(loaded.validationLayer == nil)
            #expect(loaded.validationLayerFraction == nil)
        }
    }

    @Test func validationDepthRefusesAmbiguousAndOutOfRangeDeclarations() throws {
        try withTempRoot { _ in
            try makeDraft("depth2")
            // Both at once is the ambiguity ValidationLayerRule names.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    layer: 21, fraction: 0.35, experimentName: "depth2")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    layer: -3, fraction: nil, experimentName: "depth2")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    layer: nil, fraction: 1.5, experimentName: "depth2")
            }
            // Nothing was written by the refusals.
            let loaded = try ExperimentStore.load(name: "depth2")
            #expect(loaded.validationLayer == nil)
            #expect(loaded.validationLayerFraction == nil)
        }
    }

    @Test func validationDepthListsStoreScalarForOneAndListForSeveral() throws {
        try withTempRoot { _ in
            try makeDraft("depthList")
            // A depth LIST (validate-at-the-sweep-layers): several fractions
            // store the list field; ONE stores the scalar field, so a UI
            // that always passes lists produces the exact manifests the
            // scalar era did.
            try ExperimentStore.setValidationReadDepth(
                fractions: [0.5, 0.6, 0.7, 0.8], experimentName: "depthList")
            var loaded = try ExperimentStore.load(name: "depthList")
            #expect(loaded.validationLayerFractions == [0.5, 0.6, 0.7, 0.8])
            #expect(loaded.validationLayerFraction == nil)
            try ExperimentStore.setValidationReadDepth(
                fractions: [0.65], experimentName: "depthList")
            loaded = try ExperimentStore.load(name: "depthList")
            #expect(loaded.validationLayerFraction == 0.65)
            #expect(loaded.validationLayerFractions == nil)
            // Switching shapes replaces every sibling field.
            try ExperimentStore.setValidationReadDepth(
                layers: [21, 31, 40], experimentName: "depthList")
            loaded = try ExperimentStore.load(name: "depthList")
            #expect(loaded.validationLayers == [21, 31, 40])
            #expect(loaded.validationLayerFraction == nil)
            #expect(loaded.validationLayerFractions == nil)
            // Refusals: duplicates, empties, nonsense entries.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    layers: [3, 3], experimentName: "depthList")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    fractions: [], experimentName: "depthList")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setValidationReadDepth(
                    fractions: [0.5, 1.5], experimentName: "depthList")
            }
            // The refused writes left the last declaration in place.
            loaded = try ExperimentStore.load(name: "depthList")
            #expect(loaded.validationLayers == [21, 31, 40])
        }
    }

    @Test func validationDepthRefusesOnFrozen() throws {
        try withTempRoot { _ in
            try makeDraft("depth3")
            try freezeOnDisk("depth3")
            do {
                try ExperimentStore.setValidationReadDepth(
                    layer: nil, fraction: 0.5, experimentName: "depth3")
                Issue.record("expected the draft-only refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("duplicate it to iterate"))
            }
        }
    }

    // MARK: capability battery picker setter (2026-08-01 — the pin helper
    // existed since the study-pack importer; this is its first UI door)

    @Test func batteryFilePinsShapeValidatedAndClears() throws {
        try withTempRoot { root in
            try makeDraft("batt")
            let directory = root.appending(components: "prompts", "batteries")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let file = directory.appending(component: "tiny.jsonl")
            try #"{"prompt": "What is 2 + 2?", "answer": "4"}"#.write(
                to: file, atomically: true, encoding: .utf8)
            try ExperimentStore.setCapabilityBatteryFile(
                "prompts/batteries/tiny.jsonl", experimentName: "batt")
            var loaded = try ExperimentStore.load(name: "batt")
            #expect(loaded.capabilityBatteryFile == "prompts/batteries/tiny.jsonl")
            #expect(loaded.capabilityBatteryHash?.count == 64)
            // nil clears BOTH fields — a hash without its file is a lie.
            try ExperimentStore.setCapabilityBatteryFile(
                nil, experimentName: "batt")
            loaded = try ExperimentStore.load(name: "batt")
            #expect(loaded.capabilityBatteryFile == nil)
            #expect(loaded.capabilityBatteryHash == nil)
        }
    }

    // MARK: reader pins (2026-08-01 — the button that gated on a list
    // nothing could populate)

    /// A REAL, loadable artifact — encoded by the same type the scorer
    /// decodes (review 2026-08-01, P1: a hand-built JSON stub proved the
    /// pin accepted artifacts the evaluate path could not load).
    private func writeReaderArtifact(
        _ root: URL, concept: String, modelID: String = "test/model",
        revision: String? = "abc123"
    ) throws -> String {
        let artifact = RepEReader.Artifact(
            modelID: modelID, revision: revision, concept: concept, layer: 21,
            template: RepEReader.TaskTemplate(
                id: "t1", conceptSlot: false,
                text: "Consider the following. {stimulus}", hash: "th"),
            datasetHash: "dh",
            probe: SteeringVectorMath.ScalarProbe(
                direction: [1, 0], projectionCenter: 0, projectionScale: 1,
                orientation: 1, positiveMean: 1, negativeMean: -1),
            pc1ExplainedVariance: 0.5, trainAccuracy: 0.9,
            heldOutAccuracy: nil, trainPairCount: 10, heldOutPairCount: 2,
            substrate: "python-hf-transformers")
        let directory = root.appending(
            components: "runs", "20260801T000000-reader-fit")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let file = directory.appending(component: "\(concept)-reader.json")
        try JSONEncoder().encode(artifact).write(to: file)
        return "runs/20260801T000000-reader-fit/\(concept)-reader.json"
    }

    @Test func readerPinsReplacePerConceptAndRemoveClearsTheInstrument() throws {
        try withTempRoot { root in
            try makeDraft("readers")
            let path = try writeReaderArtifact(root, concept: "fear")
            try ExperimentStore.pinReader(path: path, experimentName: "readers")
            var loaded = try ExperimentStore.load(name: "readers")
            #expect(loaded.readerRefs?.count == 1)
            #expect(loaded.readerRefs?.first?.concept == "fear")
            #expect(loaded.readerRefs?.first?.hash.count == 64)
            // Re-pinning the same concept replaces, never duplicates.
            try ExperimentStore.pinReader(path: path, experimentName: "readers")
            loaded = try ExperimentStore.load(name: "readers")
            #expect(loaded.readerRefs?.count == 1)
            // A foreign SUBSTRATE pins fine (the freezing engine judges
            // substrate); removing the last reader also removes the
            // instrument it would orphan.
            try ExperimentStore.setOutcomeInstruments(
                ["repeReaderScore"], experimentName: "readers")
            try ExperimentStore.removeReader(
                concept: "fear", experimentName: "readers")
            loaded = try ExperimentStore.load(name: "readers")
            #expect(loaded.readerRefs == nil)
            #expect(loaded.outcomeInstruments == nil)
        }
    }

    @Test func readerPinRefusesNonReadersAndForeignModels() throws {
        try withTempRoot { root in
            try makeDraft("readers2")
            // Wrong model: readers measure nothing off their fit model.
            let foreign = try writeReaderArtifact(
                root, concept: "fear", modelID: "other/model")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinReader(
                    path: foreign, experimentName: "readers2")
            }
            // Not a reader artifact at all — refused by the REAL loader,
            // not a field sniff.
            let sidecar = root.appending(
                components: "runs", "20260801T000000-reader-fit", "vec.json")
            try #"{"concept": "fear", "modelID": "test/model"}"#.write(
                to: sidecar, atomically: true, encoding: .utf8)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinReader(
                    path: "runs/20260801T000000-reader-fit/vec.json",
                    experimentName: "readers2")
            }
            let loaded = try ExperimentStore.load(name: "readers2")
            #expect(loaded.readerRefs == nil)
        }
    }

    /// Review 2026-08-01 (P1): a reader binds to exact fitted bytes —
    /// model AND revision. No revision refuses; a revision contradicting
    /// the study's pin refuses; a matching one pins.
    @Test func readerPinBindsTheRevision() throws {
        try withTempRoot { root in
            _ = try ExperimentStore.create(
                name: "readers3", description: "d", modelID: "test/model",
                modelRevision: "abc123")
            let unattributed = try writeReaderArtifact(
                root, concept: "calm", revision: nil)
            do {
                try ExperimentStore.pinReader(
                    path: unattributed, experimentName: "readers3")
                Issue.record("expected the no-revision refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no model revision"))
            }
            let mismatched = try writeReaderArtifact(
                root, concept: "calm", revision: "beefcafe0000")
            do {
                try ExperimentStore.pinReader(
                    path: mismatched, experimentName: "readers3")
                Issue.record("expected the revision-mismatch refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("revision"))
            }
            let matching = try writeReaderArtifact(
                root, concept: "calm", revision: "abc123")
            try ExperimentStore.pinReader(
                path: matching, experimentName: "readers3")
            #expect(
                try ExperimentStore.load(name: "readers3")
                    .readerRefs?.first?.concept == "calm")
        }
    }

    /// Review 2026-08-02 (P1): verify and the runtime scorer each carried
    /// their own binding subset — the runtime checked only substrate. ONE
    /// helper now serves both; this pins its rules directly.
    @Test func readerBindingHelperEnforcesTheCompleteBinding() throws {
        try withTempRoot { _ in
            let manifest = try ExperimentStore.create(
                name: "bind", description: "d", modelID: "test/model",
                modelRevision: "abc123")
            func artifact(
                model: String = "test/model", revision: String? = "abc123",
                concept: String = "fear", substrate: String = RepEReader.substrate
            ) -> RepEReader.Artifact {
                RepEReader.Artifact(
                    modelID: model, revision: revision, concept: concept,
                    layer: 1,
                    template: RepEReader.TaskTemplate(
                        id: "t", conceptSlot: false,
                        text: "x {stimulus}", hash: "h"),
                    datasetHash: "d",
                    probe: SteeringVectorMath.ScalarProbe(
                        direction: [1], projectionCenter: 0,
                        projectionScale: 1, orientation: 1,
                        positiveMean: 1, negativeMean: -1),
                    pc1ExplainedVariance: 0.5, trainAccuracy: 0.9,
                    heldOutAccuracy: nil, trainPairCount: 4,
                    heldOutPairCount: 1, substrate: substrate)
            }
            #expect(ExperimentStore.readerBindingProblems(
                artifact(), refConcept: "fear", manifest: manifest).isEmpty)
            #expect(ExperimentStore.readerBindingProblems(
                artifact(revision: nil), refConcept: "fear",
                manifest: manifest).contains { $0.contains("no model revision") })
            #expect(ExperimentStore.readerBindingProblems(
                artifact(revision: "beef"), refConcept: "fear",
                manifest: manifest).contains { $0.contains("pinned") })
            #expect(ExperimentStore.readerBindingProblems(
                artifact(concept: "anger"), refConcept: "fear",
                manifest: manifest).contains { $0.contains("wrong instrument") })
            #expect(ExperimentStore.readerBindingProblems(
                artifact(substrate: "python-hf-transformers"),
                refConcept: "fear",
                manifest: manifest).contains { $0.contains("substrate") })
        }
    }

    /// The verify side of the same binding: a hand-written ref whose
    /// concept disagrees with its artifact names the wrong instrument.
    @Test func readerVerifyFlagsConceptMismatchAndMissingRevision() throws {
        try withTempRoot { root in
            try makeDraft("readers4")
            let path = try writeReaderArtifact(
                root, concept: "fear", revision: nil)
            let url = ExperimentStore.resolveProjectPath(path)
            let data = try Data(contentsOf: url)
            let hash = ExperimentStore.sha256Hex(data)
            var manifest = try ExperimentStore.load(name: "readers4")
            manifest.readerRefs = [
                ExperimentManifest.ReaderRef(
                    path: path, hash: hash, concept: "anger")
            ]
            try ExperimentStore.save(manifest)
            let violations = ExperimentStore.verify(
                try ExperimentStore.load(name: "readers4"))
            #expect(violations.contains { $0.contains("wrong instrument") })
            #expect(violations.contains { $0.contains("no model revision") })
        }
    }

    // MARK: human-validation pin (2026-08-01 — the field with no writer)

    @Test func humanValidationPinsShapeValidatedAndClears() throws {
        try withTempRoot { root in
            try makeDraft("hv")
            let directory = root.appending(component: "prompts")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let file = directory.appending(component: "human-validation.jsonl")
            try """
                {"condition": "fear-steered", "promptID": "p1", "outcome": "variant"}
                {"condition": "fear-steered", "promptID": "p2", "outcome": "tie", "sampleIndex": 0}
                """.write(to: file, atomically: true, encoding: .utf8)
            let pin = try ExperimentStore.pinHumanValidation(
                path: "prompts/human-validation.jsonl", experimentName: "hv")
            #expect(pin.hash.count == 64)
            var loaded = try ExperimentStore.load(name: "hv")
            #expect(loaded.humanValidation?.path == "prompts/human-validation.jsonl")
            try ExperimentStore.clearHumanValidation(experimentName: "hv")
            loaded = try ExperimentStore.load(name: "hv")
            #expect(loaded.humanValidation == nil)
        }
    }

    @Test func humanValidationRefusesBadOutcomeVocabulary() throws {
        try withTempRoot { root in
            try makeDraft("hv2")
            let directory = root.appending(component: "prompts")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let file = directory.appending(component: "bad.jsonl")
            // "condition" is this ENGINE'S label; the file contract says
            // "variant" — the validator must hold the cross-engine line.
            try #"{"condition": "c", "promptID": "p", "outcome": "condition"}"#
                .write(to: file, atomically: true, encoding: .utf8)
            do {
                _ = try ExperimentStore.pinHumanValidation(
                    path: "prompts/bad.jsonl", experimentName: "hv2")
                Issue.record("expected the outcome-vocabulary refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("outcome"))
                #expect(error.reason.contains("baseline|variant|tie"))
            }
        }
    }

    // MARK: revision + seeds (2026-08-01 — the last Study Setup orphans)

    @Test func modelRevisionEditsOnDraftsAndEmptyClears() throws {
        try withTempRoot { _ in
            try makeDraft("rev")
            try ExperimentStore.setModelRevision(
                "deadbeef1234", experimentName: "rev")
            #expect(try ExperimentStore.load(name: "rev").modelRevision
                == "deadbeef1234")
            try ExperimentStore.setModelRevision("  ", experimentName: "rev")
            #expect(try ExperimentStore.load(name: "rev").modelRevision == nil)
            try freezeOnDisk("rev")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setModelRevision(
                    "cafe", experimentName: "rev")
            }
        }
    }

    @Test func seedsRoundTripAndRefuseEmptyOrDuplicates() throws {
        try withTempRoot { _ in
            try makeDraft("seeds")
            try ExperimentStore.setSeeds(
                [20260610, 20260611, 7], experimentName: "seeds")
            #expect(try ExperimentStore.load(name: "seeds").seeds
                == [20260610, 20260611, 7])
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSeeds([], experimentName: "seeds")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSeeds([7, 7], experimentName: "seeds")
            }
            // Refusals wrote nothing.
            #expect(try ExperimentStore.load(name: "seeds").seeds
                == [20260610, 20260611, 7])
        }
    }

    @Test func batteryFileRefusesMissingAndShapeGarbage() throws {
        try withTempRoot { root in
            try makeDraft("batt2")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setCapabilityBatteryFile(
                    "prompts/batteries/nonexistent.jsonl",
                    experimentName: "batt2")
            }
            let directory = root.appending(components: "prompts", "batteries")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let garbage = directory.appending(component: "garbage.jsonl")
            try "not json at all".write(
                to: garbage, atomically: true, encoding: .utf8)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setCapabilityBatteryFile(
                    "prompts/batteries/garbage.jsonl", experimentName: "batt2")
            }
            // Refusals wrote nothing.
            let loaded = try ExperimentStore.load(name: "batt2")
            #expect(loaded.capabilityBatteryFile == nil)
            #expect(loaded.capabilityBatteryHash == nil)
        }
    }
}
