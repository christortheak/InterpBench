import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Pin-surface closure for the run-bundle packer: `RunBundlePackager` must
/// include EVERY pinned input the manifest declares, mechanically, by
/// iterating the same enumeration the git cleanliness gate uses
/// (`ExperimentStore.pinnedInputEntries`). These tests iterate that
/// enumeration too — never a literal file list — so a new pin kind added to
/// the enumeration is asserted into the bundle automatically. Parallel to
/// `Server/tests/test_bundles.py::test_run_bundle_packs_the_entire_pin_surface`.
@Suite(.serialized) struct RunBundleClosureTests {

    /// Both process-global root seams under the one shared override lock:
    /// the packer resolves pins against `VectorCatalog.projectRoot`
    /// (WorkspaceRoot), the manifest store against
    /// `ExperimentStore.rootOverride`.
    private func withTempWorkspace<T>(_ body: (URL) throws -> T) throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "bundle-closure-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        WorkspaceRoot.programmaticOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    @discardableResult
    private func write(_ root: URL, _ relative: String, _ text: String) throws -> String {
        let url = root.appending(path: relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = Data(text.utf8)
        try data.write(to: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A draft study declaring every pinnable input kind the enumeration
    /// knows, with fixture bytes on disk for each.
    private func everyPinKindStudy(in root: URL) throws -> ExperimentManifest {
        // Paired concept with stimuli, held-out validation, and markers.
        try write(root, "prompts/concepts/fair/positive.jsonl", "{\"text\":\"fair\"}\n")
        try write(root, "prompts/concepts/fair/negative.jsonl", "{\"text\":\"unfair\"}\n")
        let fairValidation = try write(
            root, "prompts/concepts/fair/validation.jsonl",
            "{\"text\":\"equal treatment\",\"expresses\":true}\n")
        try write(root, "prompts/concepts/fair/markers.json", "{\"markers\":[\"equit\"]}")
        // Grand-mean concept: stories + validation beside them.
        let warmStories = try write(
            root, "prompts/emotions/warm/stories.jsonl", "{\"text\":\"a warm tale\"}\n")
        let warmValidation = try write(
            root, "prompts/emotions/warm/validation.jsonl",
            "{\"text\":\"kind act\",\"expresses\":true}\n")
        let neutralHash = try write(
            root, "prompts/neutral/corpus.jsonl", "{\"text\":\"the sky is blue\"}\n")

        var manifest = try ExperimentStore.create(
            name: "closure", description: "", modelID: "test/model",
            modelRevision: "abc123")
        manifest.concepts = [
            .init(
                name: "fair", stimulusSetHash: "x", options: .init(),
                validationHash: fairValidation),
            .init(
                name: "warm", stimulusSetHash: warmStories,
                options: .init(method: .emotionGrandMean),
                validationHash: warmValidation),
        ]
        manifest.grandMeanCorpus = .init(concepts: ["warm"], hashes: ["warm": warmStories])
        manifest.neutralCorpusHash = neutralHash
        manifest.taskPromptsFile = "prompts/tasks/cases.jsonl"
        manifest.taskPromptsHash = try write(
            root, "prompts/tasks/cases.jsonl", "{\"id\":\"c1\",\"text\":\"decide\"}\n")
        manifest.judgeRubricFile = "prompts/rubrics/r1.md"
        manifest.judgeRubricHash = try write(root, "prompts/rubrics/r1.md", "# rubric\n")
        manifest.capabilityBatteryFile = "prompts/batteries/b1.jsonl"
        manifest.capabilityBatteryHash = try write(
            root, "prompts/batteries/b1.jsonl", "{\"prompt\":\"2+2\",\"answer\":\"4\"}\n")
        manifest.reasoningStyleTaxonomyPath = "prompts/taxonomies/style.json"
        manifest.reasoningStyleTaxonomyHash = try write(
            root, "prompts/taxonomies/style.json",
            "{\"schemaVersion\":1,\"name\":\"t\",\"features\":[]}")
        manifest.humanBaseline = .init(
            path: "prompts/baselines/human.csv",
            hash: try write(
                root, "prompts/baselines/human.csv",
                "endpoint,deltaHuman,ciLower,ciUpper\naffirmRate,0.1,0.05,0.15\n"))
        manifest.humanValidation = .init(
            path: "prompts/validation/human.jsonl",
            hash: try write(
                root, "prompts/validation/human.jsonl",
                "{\"promptID\":\"c1\",\"condition\":\"baseline\",\"outcome\":\"affirm\"}\n"))
        manifest.multiAgentScenarioPath = "prompts/scenarios/panel.json"
        manifest.multiAgentScenarioHash = try write(
            root, "prompts/scenarios/panel.json", "{\"agents\":[]}")
        // Neutral-PC basis pinned on a condition (runs/-resident library).
        let basisHash = try write(
            root, "runs/neutral-pcs/basis1/neutral-pc-basis.json",
            "{\"componentsByLayer\":{\"1\":[[1.0]]}}")
        manifest.conditions = [
            .init(
                name: "steered",
                slots: [.init(concept: "fair", layer: 1, alpha: 0.1)],
                neutralPCBasisPath: "runs/neutral-pcs/basis1",
                neutralPCBasisHash: basisHash)
        ]
        // RepE reader instrument artifact.
        let readerHash = try write(
            root, "prompts/readers/fair/reader.json",
            "{\"artifactType\":\"repe-reader-lat\",\"modelID\":\"test/model\"}")
        manifest.readerRefs = [
            .init(path: "prompts/readers/fair/reader.json", hash: readerHash, concept: "fair")
        ]
        // Variant condition backed by a runs/-resident agent artifact.
        //
        // The agent carries a REAL injection with its vector pair on disk
        // (B3). Until 2026-07-26 this fixture's agent had `injections: []`,
        // so the closure test passed while the packer shipped agents whose
        // vectors were silently left behind — the study then failed on the
        // cluster after the queue wait and the model load.
        try write(
            root, "runs/extract-1/fair.safetensors", "not-really-safetensors")
        try write(
            root, "runs/extract-1/fair.json",
            "{\"modelID\":\"test/model\",\"layers\":[1]}")
        let artifact = ModelVariantArtifact(
            name: "agent1", baseModelID: "test/model", adapters: [],
            injections: [
                .init(
                    concept: "fair",
                    // Workspace-relative, as the server's `promote` writes it.
                    vectorArtifactID: "runs/extract-1/fair",
                    layer: 1, alpha: 0.1)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false, temperature: 0,
            systemPrompt: "")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let artifactData = try encoder.encode(artifact)
        let artifactHash = try write(
            root, "runs/model-variants/agent1/model-variant.json",
            String(decoding: artifactData, as: UTF8.self))
        manifest.variantConditions = [
            .init(
                name: "agent1",
                artifactPath: "runs/model-variants/agent1/model-variant.json",
                artifactHash: artifactHash, artifact: artifact)
        ]
        // Sweep inputs: declared dev prompts, battery, and choice prompts.
        try write(root, "prompts/dev/dev.jsonl", "{\"text\":\"say something\"}\n")
        try write(
            root, "prompts/dev/choice.jsonl",
            "{\"prompt\":\"pick\",\"options\":[\"a\",\"b\"]}\n")
        try write(root, "prompts/batteries/sweep.jsonl", "{\"prompt\":\"1+1\",\"answer\":\"2\"}\n")
        manifest.sweep = .init(
            devPromptsFile: "prompts/dev/dev.jsonl",
            batteryFile: "prompts/batteries/sweep.jsonl",
            selection: .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: "prompts/dev/choice.jsonl")))
        try ExperimentStore.save(manifest)
        return manifest
    }

    private func filesUnder(_ url: URL) -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        else { return [] }
        guard isDirectory.boolValue else { return [url] }
        guard
            let enumerator = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: [.isRegularFileKey])
        else { return [] }
        var out: [URL] = []
        for case let file as URL in enumerator {
            if (try? file.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
                out.append(file)
            }
        }
        return out
    }

    private func extractBundle(_ bundle: URL) throws -> URL {
        let destination = FileManager.default.temporaryDirectory
            .appending(component: "bundle-extract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: true)
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.arguments = ["-xzf", bundle.path, "-C", destination.path]
        try process.run()
        process.waitUntilExit()
        try #require(process.terminationStatus == 0)
        return destination
    }

    @Test func runBundlePacksTheEntirePinSurface() throws {
        try withTempWorkspace { root in
            let manifest = try everyPinKindStudy(in: root)
            let entries = ExperimentStore.pinnedInputEntries(manifest)
            // Fixture honesty: the study must actually exercise the REQUIRED
            // pin kinds with on-disk bytes (a fixture with missing files
            // tests nothing).
            for entry in entries where entry.required {
                #expect(
                    FileManager.default.fileExists(atPath: entry.url.path),
                    "fixture gap: \(entry.label)")
            }

            let bundle = try RunBundlePackager.packageExperiment(manifest)
            let extracted = try extractBundle(bundle)
            defer { try? FileManager.default.removeItem(at: extracted) }
            let metaData = try Data(
                contentsOf: extracted.appending(component: "steerlab-bundle.json"))
            let meta = try #require(
                try JSONSerialization.jsonObject(with: metaData) as? [String: Any])
            let packedEntries = try #require(meta["entries"] as? [[String: Any]])
            let packed = Set(packedEntries.compactMap { $0["path"] as? String })

            let rootPath = root.standardizedFileURL.path
            for entry in entries {
                for file in filesUnder(entry.url) {
                    let path = file.standardizedFileURL.path
                    try #require(path.hasPrefix(rootPath + "/"))
                    let relative = String(path.dropFirst(rootPath.count + 1))
                    #expect(
                        packed.contains(relative),
                        "pin-surface closure violated: \(entry.label) file \(relative) is enumerated but not in the bundle"
                    )
                    // The extracted bundle actually contains the bytes.
                    #expect(
                        FileManager.default.fileExists(
                            atPath: extracted.appending(path: relative).path),
                        "bundle entry \(relative) missing from extracted payload")
                }
            }
            // The manifest itself rides along.
            #expect(packed.contains("experiments/closure/experiment.json"))
            // B3, stated rather than merely implied by the loop above: an
            // agent condition's VECTORS travel with it. Shipping the agent
            // JSON alone produced a bundle that could not run.
            #expect(packed.contains("runs/extract-1/fair.safetensors"))
            #expect(packed.contains("runs/extract-1/fair.json"))
        }
    }

    /// Review 2026-08-29, P1: a bundle is a workspace with no git and no
    /// neighbours, so a preregistration that does not travel in it has no
    /// integrity at all on the far side. The freeze stamps the authored file,
    /// which puts it on the pin surface — and the surface is what the packer
    /// walks. Parallel to
    /// `test_bundle_carries_a_preserved_authored_preregistration`.
    @Test func runBundleCarriesAPreservedAuthoredPreregistration() throws {
        try withTempWorkspace { root in
            _ = try ExperimentStore.create(
                name: "prereg-bundle", description: "", modelID: "test/model",
                modelRevision: "abc123")
            try write(
                root, "prompts/emotions/calm/stories.jsonl",
                "{\"concept\":\"calm\",\"text\":\"a calm tale\"}\n")
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean,
                experimentName: "prereg-bundle")
            let authored = "# Analysis preregistration\n\nOne estimator, chosen now.\n"
            try authored.write(
                to: ExperimentStore.directory.appending(
                    components: "prereg-bundle",
                    ExperimentStore.preregistrationFilename),
                atomically: true, encoding: .utf8)
            let frozen = try ExperimentStore.freeze(name: "prereg-bundle", force: true)

            let bundle = try RunBundlePackager.packageExperiment(frozen)
            let extracted = try extractBundle(bundle)
            defer { try? FileManager.default.removeItem(at: extracted) }
            let packedPreregistration = extracted.appending(
                path: "experiments/prereg-bundle/preregistration.md")
            #expect(
                try String(contentsOf: packedPreregistration, encoding: .utf8)
                    == authored)
            #expect(
                FileManager.default.fileExists(
                    atPath: extracted.appending(
                        path: "experiments/prereg-bundle/pinned/preregistration.md"
                    ).path))
        }
    }

    /// Fail closed: an agent whose vector is absent must refuse packaging
    /// HERE, not fail on the cluster after the queue wait and model load.
    @Test func packagingRefusesAnAgentWhoseVectorIsMissing() throws {
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "closure-agent", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.taskPromptsFile = "prompts/tasks/task.jsonl"
            try write(root, "prompts/tasks/task.jsonl", "{\"prompt\":\"hi\"}\n")
            manifest.capabilityBatteryFile = "prompts/batteries/basic.jsonl"
            try write(root, "prompts/batteries/basic.jsonl", "{\"prompt\":\"1+1\",\"answer\":\"2\"}\n")

            let artifact = ModelVariantArtifact(
                name: "ghost", baseModelID: "test/model", adapters: [],
                injections: [
                    .init(
                        concept: "fair",
                        vectorArtifactID: "runs/never-extracted/fair",
                        layer: 1, alpha: 0.1)
                ],
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: "")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let artifactHash = try write(
                root, "runs/model-variants/ghost/model-variant.json",
                String(decoding: try encoder.encode(artifact), as: UTF8.self))
            manifest.variantConditions = [
                .init(
                    name: "ghost",
                    artifactPath: "runs/model-variants/ghost/model-variant.json",
                    artifactHash: artifactHash, artifact: artifact)
            ]

            // The dependency IS enumerated...
            let labels = ExperimentStore.pinnedInputEntries(manifest)
                .filter { $0.required }
                .map(\.label)
            #expect(labels.contains { $0.contains("vector 'fair'") })
            // ...and because it is required and absent, packaging refuses.
            #expect(throws: (any Error).self) {
                try RunBundlePackager.packageExperiment(manifest)
            }
        }
    }

    @Test func packagingRefusesMissingPinnedInput() throws {
        try withTempWorkspace { root in
            try write(root, "prompts/concepts/fair/positive.jsonl", "{\"text\":\"fair\"}\n")
            try write(root, "prompts/concepts/fair/negative.jsonl", "{\"text\":\"unfair\"}\n")
            var manifest = try ExperimentStore.create(
                name: "closure-missing", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.concepts = [
                .init(name: "fair", stimulusSetHash: "x", options: .init())
            ]
            manifest.judgeRubricFile = "prompts/rubrics/ghost.md"
            manifest.judgeRubricHash = String(repeating: "0", count: 64)
            try ExperimentStore.save(manifest)
            #expect(throws: ChatServiceError.self) {
                _ = try RunBundlePackager.packageExperiment(manifest)
            }
            do {
                _ = try RunBundlePackager.packageExperiment(manifest)
            } catch let error as ChatServiceError {
                // The failure names the missing pin — never a child-side
                // verify failure hours later on the cluster.
                #expect(error.reason.contains("judge rubric"))
            }
        }
    }
}
