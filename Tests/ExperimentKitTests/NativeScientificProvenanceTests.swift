import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

@MainActor @Suite(.serialized) struct NativeScientificProvenanceTests {
    @Test func scopeSidecarReportsActualRankDoseAndUnresolvedConditionsWithoutReplacement() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let baseline = RunInterventionScope.entry(name: "baseline") { [] }
        let configured = RunInterventionScope.entry(name: "configured") {
            [.init(layer: 2, vector: [1, 0], alpha: 0.5, mode: .ablate, concept: "a", centering: "neutralMean"),
             .init(layer: 2, vector: [2, 0], alpha: 0.5, mode: .ablate, concept: "b"),
             .init(layer: 2, vector: [1, 0], alpha: 3)]
        }
        let missing = RunInterventionScope.entry(name: "missing") { throw ExperimentError(reason: "missing vector") }
        try RunInterventionScope.write(experiment: "example", entries: [baseline, configured, missing], to: root)
        let path = root.appending(component: RunInterventionScope.filename)
        let bytes = try Data(contentsOf: path)
        let doc = try JSONDecoder().decode(RunInterventionScope.Document.self, from: bytes)
        #expect(doc.conditions[0].scopes.isEmpty)
        #expect(doc.conditions[1].scopes[0].detail["rankPerLayer"] == .object(["2": .integer(1)]))
        #expect(doc.conditions[1].scopes[1].detail["promptTokenCount"] == .string(RunInterventionScope.promptCountPerItem))
        #expect(doc.conditions[2].unresolved?.contains("missing vector") == true)
        try RunInterventionScope.write(experiment: "changed", entries: [], to: root)
        #expect(try Data(contentsOf: path) == bytes)
    }

    @Test func trainingScaleIsDirectAndLegacyFieldsRemainAbsent() throws {
        var artifact = FineTuneArtifact(name: "example", baseModelID: "example/model", adapterDirectory: "adapter", rank: 8, scale: 10)
        let before = try JSONSerialization.jsonObject(with: JSONEncoder().encode(artifact)) as! [String: Any]
        #expect(before["adapterScaleConvention"] == nil && before["effectiveAdapterScale"] == nil)
        // The trainer's returned multiplier wins over editable panel state.
        artifact.recordTrainingScale(3)
        let bytes = try JSONEncoder().encode(artifact)
        let decoded = try JSONDecoder().decode(FineTuneArtifact.self, from: bytes)
        #expect(decoded.effectiveAdapterScale == 3)
        #expect(decoded.requestedAdapterScale == 3)
        #expect(decoded.adapterScaleConvention == "direct")
        #expect(decoded.requestedAdapterScaleConvention == "direct")
        let legacy = try JSONDecoder().decode(FineTuneArtifact.self, from: JSONSerialization.data(withJSONObject: before))
        #expect(legacy.effectiveAdapterScale == nil)
    }

    @Test func diagnosticWritesExactSeedsAndSeparateDrawPopulationsWithoutEditingTheStudy() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "native-stability") { root in
            var manifest = try ExperimentStore.create(name: "example", description: "fixture", modelID: "example/model")
            let ref = ExperimentManifest.ConceptRef(name: "signal", stimulusSetHash: "pinned",
                options: ExtractionOptions(method: .meanDifference, readingPosition: .lastToken))
            manifest.concepts = [ref]
            try ExperimentStore.save(manifest)
            let path = root.appending(path: "experiments/example/experiment.json")
            let original = try Data(contentsOf: path)
            let prepared = ExtractStability.Prepared(manifest: manifest, ref: ref,
                positive: ["a", "b", "c", "d"], negative: ["e", "f", "g", "h"],
                stimulus: ["stimulusHashPinned": .string("pinned"), "stimulusHashLive": .string("changed")])
            let p = StimulusActivations(values: [[[2,0]],[[3,1]],[[4,2]],[[1,-1]]], residualNormPerLayer: [1])
            let n = StimulusActivations(values: [[[0,0]],[[1,1]],[[2,2]],[[-1,-1]]], residualNormPerLayer: [1])
            let first = try ExtractStability.finish(prepared: prepared, positive: p, negative: n,
                modelRevision: "revision", resamples: 4, fraction: 0.5, seed: UInt64.max, orderShuffles: 3)
            let second = try ExtractStability.finish(prepared: prepared, positive: p, negative: n,
                modelRevision: "revision", resamples: 4, fraction: 0.5, seed: UInt64.max, orderShuffles: 3)
            #expect(first.path != second.path)
            #expect(first.directory.deletingLastPathComponent().resolvingSymlinksInPath().path == root.appending(component: "diagnostics").resolvingSymlinksInPath().path)
            #expect(first.summary["stimulusDrift"] == .bool(true))
            let document = try JSONSerialization.jsonObject(with: Data(contentsOf: first.path)) as! [String: Any]
            let shared = document["resample"] as! [String: Any]
            #expect((shared["seed"] as? NSNumber)?.uint64Value == UInt64.max)
            let layer = (document["layers"] as! [[String: Any]])[0]
            #expect((layer["resampleCosines"] as? [Double])?.count == 4)
            #expect((layer["orderShuffleCosines"] as? [Double])?.count == 3)
            #expect(Set(shared.keys).isDisjoint(with: ExtractStability.layerKeys))
            #expect(document["neutralProjectionApplied"] as? Bool == false)
            #expect(try Data(contentsOf: path) == original)
        }
    }

    @Test func unsupportedMethodAndBadDrawCountsAreExplainedBeforeCompute() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "stability-preflight") { _ in
            var manifest = try ExperimentStore.create(name: "example", description: "fixture", modelID: "example/model")
            manifest.concepts = [.init(name: "signal", stimulusSetHash: "fixture", options: .init(method: .emotionGrandMean))]
            try ExperimentStore.save(manifest)
            do {
                _ = try ExtractStability.preflight(experiment: "example", concept: "signal")
                Issue.record("unsupported method reached capture")
            } catch let failure as ExtractStability.Failure {
                #expect(failure.code == "unsupportedMethod")
                #expect(!failure.repairAction.isEmpty)
            }
            manifest.concepts[0].options.method = .meanDifference
            try ExperimentStore.save(manifest)
            let directory = VectorCatalog.conceptsDirectory.appending(component: "signal")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let rows = "{\"text\":\"first\"}\n{\"text\":\"second\"}\n"
            for filename in ["positive.jsonl", "negative.jsonl"] {
                try rows.write(to: directory.appending(component: filename), atomically: true, encoding: .utf8)
            }
            do {
                _ = try ExtractStability.preflight(experiment: "example", concept: "signal", resamples: 1)
                Issue.record("invalid draw count reached capture")
            } catch let failure as ExtractStability.Failure {
                #expect(failure.code == "usage" && failure.state == .blocked)
                #expect(failure.repairAction.contains("2 resamples"))
            }
        }
    }

    @Test func diagnosticCLIRejectsMalformedInputsBeforeLoadingWeights() async {
        let runner = ExperimentCLIRunner(sink: .discarding)
        let outcome = await runner.run(namespace: "experiment",
            ["extract-stability", "example", "signal", "--seed", "-1", "--json"])
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.state == .blocked)
        #expect(outcome.envelope.error?.code == "usage")
    }
}
