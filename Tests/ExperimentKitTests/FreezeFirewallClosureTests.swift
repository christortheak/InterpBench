import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// Phase E — freeze-firewall closure: variant validity gate, external-input
/// pinning at freeze, thinking-mode gate, run-time prompt drift rules, and
/// canonical-bytes verification of server-frozen manifests. Ports the
/// fixture semantics of `Server/tests/test_freeze_firewall_closure.py`.
/// Declared as an extension of the serialized `ExperimentStoreTests` suite
/// because these tests share its `rootOverride` test seam.
extension ExperimentStoreTests {

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func makeVariantArtifact(
        adapterHash: String?, systemPrompt: String = ""
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "fear-lora",
            baseModelID: "test/model",
            adapters: [
                .init(
                    name: "a1", artifactPath: "x", adapterDirectory: "y",
                    adapterHash: adapterHash)
            ],
            injections: [],
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: systemPrompt)
    }

    /// Writes the artifact under gitignored `runs/` and pins it into a
    /// fresh variant-only study (no concepts, so freeze needs no validate
    /// evidence — matching the server fixture).
    @discardableResult
    private func makeVariantStudy(
        name: String = "vs", artifact: ModelVariantArtifact
    ) throws -> ExperimentManifest {
        let root = try #require(ExperimentStore.rootOverride)
        let relativePath = "runs/model-variants/fear-lora/model-variant.json"
        let artifactURL = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        try data.write(to: artifactURL)
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model", modelRevision: "abc123")
        manifest.variantConditions = [
            .init(
                name: "fear-lora", artifactPath: relativePath,
                artifactHash: sha256Hex(data), artifact: artifact)
        ]
        try ExperimentStore.save(manifest)
        return manifest
    }

    @Test func variantFreezeRequiresAdapterHashUnlessForced() throws {
        try withTempRoot {
            try makeVariantStudy(artifact: makeVariantArtifact(adapterHash: nil))
            do {
                _ = try ExperimentStore.freeze(name: "vs")
                Issue.record("expected freeze to require an adapterHash")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("adapterHash"))
                #expect(error.reason.contains("re-save the variant"))
            }
            // --force skips the evidence gate loudly, like the other gates.
            let forced = try ExperimentStore.freeze(name: "vs", force: true)
            #expect(forced.status == .frozen)
        }
    }

    @Test func variantFreezeRequiresSystemPromptHash() throws {
        try withTempRoot {
            var artifact = makeVariantArtifact(
                adapterHash: String(repeating: "ab", count: 32),
                systemPrompt: "You are calm.")
            artifact.systemPromptHash = nil  // simulate a pre-hash artifact
            try makeVariantStudy(artifact: artifact)
            do {
                _ = try ExperimentStore.freeze(name: "vs")
                Issue.record("expected freeze to require a systemPromptHash")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("systemPromptHash"))
            }
        }
    }

    @Test func variantFreezeRequiresInjectionVectorArtifactPin() throws {
        try withTempRoot {
            var artifact = makeVariantArtifact(
                adapterHash: String(repeating: "ab", count: 32))
            artifact.injections = [
                .init(concept: "fear", vectorArtifactID: "", layer: 10, alpha: 1)
            ]
            try makeVariantStudy(artifact: artifact)
            do {
                _ = try ExperimentStore.freeze(name: "vs")
                Issue.record("expected freeze to require a vectorArtifactID pin")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("vectorArtifactID"))
            }
        }
    }

    @Test func variantFreezePassesWithFullPinsAndRepointsPathOutOfRuns() throws {
        try withTempRoot {
            // Full pins: adapter hash present; systemPromptHash auto-computed
            // by the artifact initializer. Battery evidence per condition is
            // now a freeze requirement for variant studies.
            let artifact = makeVariantArtifact(
                adapterHash: String(repeating: "ab", count: 32),
                systemPrompt: "You are calm.")
            let manifest = try makeVariantStudy(artifact: artifact)
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    .init(
                        condition: "baseline", batteryHash: "bh", total: 10,
                        correct: 10, accuracy: 1),
                    .init(
                        condition: "fear-lora", batteryHash: "bh", total: 10,
                        correct: 9, accuracy: 0.9),
                ])
            let frozen = try ExperimentStore.freeze(name: "vs")
            #expect(frozen.status == .frozen)

            // The artifact lived under gitignored runs/ — freeze copied it
            // into the experiment directory (byte-identical) and repointed
            // the manifest.
            let newPath = frozen.variantConditions[0].artifactPath
            #expect(newPath == "experiments/vs/pinned/variant-fear-lora.json")
            let root = try #require(ExperimentStore.rootOverride)
            let copied = root.appending(path: newPath)
            #expect(FileManager.default.fileExists(atPath: copied.path))
            #expect(
                sha256Hex(try Data(contentsOf: copied))
                    == frozen.variantConditions[0].artifactHash)
            #expect(ExperimentStore.verify(frozen).isEmpty)
        }
    }

    @Test func freezePinsScenarioOutOfRuns() throws {
        try withTempRoot {
            let root = try #require(ExperimentStore.rootOverride)
            let relativePath = "runs/multi-agent-scenarios/panel/scenario.json"
            let scenarioURL = root.appending(path: relativePath)
            try FileManager.default.createDirectory(
                at: scenarioURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let scenarioBytes = Data(
                #"{"name": "panel", "baseModelID": "test/model"}"#.utf8)
            try scenarioBytes.write(to: scenarioURL)

            var manifest = try ExperimentStore.create(
                name: "mas", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.studyKind = .multiAgent
            manifest.multiAgentScenarioPath = relativePath
            manifest.multiAgentScenarioHash = sha256Hex(scenarioBytes)
            try ExperimentStore.save(manifest)

            let frozen = try ExperimentStore.freeze(name: "mas")
            #expect(frozen.multiAgentScenarioPath == "experiments/mas/pinned/scenario.json")
            let copied = root.appending(path: "experiments/mas/pinned/scenario.json")
            #expect(FileManager.default.fileExists(atPath: copied.path))
            #expect(try Data(contentsOf: copied) == scenarioBytes)
            // The pinned hash still verifies against the copied file.
            #expect(ExperimentStore.verify(frozen).isEmpty)
        }
    }

    @Test func thinkingModeGateBlocksAnswerTokenInstruments() throws {
        var manifest = ExperimentManifest(
            name: "think-gate", description: "", modelID: "test/model")
        manifest.qwenThinkingEnabled = true
        manifest.outcomeInstruments = ["answerTokenLogprob"]
        #expect(ExperimentStore.verify(manifest).contains { $0.contains("thinking") })

        manifest.outcomeInstruments = ["choiceProbability"]
        #expect(ExperimentStore.verify(manifest).contains { $0.contains("thinking") })

        manifest.outcomeInstruments = ["sampledText"]
        #expect(!ExperimentStore.verify(manifest).contains { $0.contains("thinking") })

        manifest.outcomeInstruments = ["answerTokenLogprob"]
        manifest.qwenThinkingEnabled = false
        #expect(!ExperimentStore.verify(manifest).contains { $0.contains("thinking") })
    }

    @Test func loadTaskPromptsEnforcesDriftAndFrozenRules() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "prompts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let promptsURL = directory.appending(component: "items.jsonl")
        let pinnedBytes = Data(#"{"id": "a", "prompt": "p"}"#.utf8) + Data("\n".utf8)
        try pinnedBytes.write(to: promptsURL)

        var manifest = ExperimentManifest(
            name: "pm", description: "", modelID: "test/model")
        manifest.taskPromptsFile = promptsURL.path
        manifest.taskPromptsHash = sha256Hex(pinnedBytes)

        // (a) No override: the pinned hash must match at RUN time…
        #expect(try ExperimentTasks.loadTaskPrompts(for: manifest).prompts.count == 1)
        try Data(#"{"id": "a", "prompt": "EDITED"}"#.utf8).write(to: promptsURL)
        do {
            _ = try ExperimentTasks.loadTaskPrompts(for: manifest)
            Issue.record("expected run-time drift detection")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("drifted"))
        }
        try pinnedBytes.write(to: promptsURL)

        // …and a frozen study with no pin at all cannot run.
        var unpinned = manifest
        unpinned.status = .frozen
        unpinned.taskPromptsHash = nil
        do {
            _ = try ExperimentTasks.loadTaskPrompts(for: unpinned)
            Issue.record("expected frozen-without-pin refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("no pinned task prompts"))
        }

        // (b) Override + frozen: byte-identical override is fine; a
        // different one is refused.
        manifest.status = .frozen
        let same = directory.appending(component: "same.jsonl")
        try pinnedBytes.write(to: same)
        #expect(
            try ExperimentTasks.loadTaskPrompts(for: manifest, override: same.path)
                .prompts.count == 1)
        let other = directory.appending(component: "other.jsonl")
        let otherBytes = Data(#"{"id": "b", "prompt": "different"}"#.utf8)
        try otherBytes.write(to: other)
        do {
            _ = try ExperimentTasks.loadTaskPrompts(for: manifest, override: other.path)
            Issue.record("expected frozen-override refusal")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("FROZEN"))
        }

        // (c) Override + draft: allowed, with the hash still returned so it
        // is recorded on every generation.
        manifest.status = .draft
        let loaded = try ExperimentTasks.loadTaskPrompts(
            for: manifest, override: other.path)
        #expect(loaded.hash == sha256Hex(otherBytes))
        #expect(loaded.prompts.count == 1)
    }

    @Test func serverFrozenManifestVerifiesAgainstCanonicalBytes() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "server-canon", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            manifest.status = .frozen
            manifest.frozenAt = "2026-07-01T00:00:00Z"
            manifest.frozenBy = "server"

            // Fabricate the server contract: freeze-canonical.json holds the
            // manifest minus the volatile freeze stamps, and freezeHash is
            // the sha256 of exactly those bytes.
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            var payload = try #require(
                try JSONSerialization.jsonObject(with: encoder.encode(manifest))
                    as? [String: Any])
            for key in ["status", "frozenAt", "freezeHash", "gitCommit", "frozenBy", "createdAt"] {
                payload.removeValue(forKey: key)
            }
            let canonical = try JSONSerialization.data(
                withJSONObject: payload, options: [.sortedKeys])
            manifest.freezeHash = sha256Hex(canonical)
            let canonicalURL = ExperimentStore.directory.appending(
                components: "server-canon", "freeze-canonical.json")
            try canonical.write(to: canonicalURL)

            // Matching bytes → verified.
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Manifest content drift after freeze → violation, even though
            // Swift cannot recompute the Python hash itself.
            var mutated = manifest
            mutated.temperature = 0.9
            #expect(
                ExperimentStore.verify(mutated).contains {
                    $0.contains("manifest content changed after freeze")
                })

            // Tampered canonical bytes → hash mismatch.
            try (canonical + Data(" ".utf8)).write(to: canonicalURL)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("manifest content changed after freeze")
                })

            // Missing canonical file (legacy server freeze) → named violation.
            try FileManager.default.removeItem(at: canonicalURL)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("freeze-canonical.json")
                })
        }
    }

    // MARK: - WP0 step 2: gate ids on the refusal path

    /// The gate vocabulary used to exist only where `forcedGatesSkipped` was
    /// stamped — the refusal path computed no id at all, so an agent could
    /// not tell "no validate evidence" from "dirty git tree" without parsing
    /// prose. Both halves of that closure are asserted here: the ONE table
    /// covers the whole closed vocabulary (no gate can lose its id in a
    /// refactor without failing this), and a live refusal carries the id,
    /// the full failure list, and a repair — with its prose byte-unchanged.
    @Test func everyGateIDIsReachableOnTheRefusalPath() throws {
        // Exhaustiveness over the closed vocabulary, structurally: this is
        // the check a fixture-per-gate test cannot make, because `gitClean`
        // is unreachable under the test root override by design.
        let table = ExperimentStore.freezeGateTable(name: "any", autoCommit: false)
        #expect(Set(table.map(\.gate)) == Set(FreezeGate.allCases))
        // `gitClean` is the one gate that moves: where freeze auto-commits,
        // dirty pins are self-healing, so it is evaluated after the commit
        // instead of from the table.
        let autoCommitting = ExperimentStore.freezeGateTable(name: "any", autoCommit: true)
        #expect(
            Set(autoCommitting.map(\.gate))
                == Set(FreezeGate.allCases).subtracting([.gitClean]))

        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "gates", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)
            do {
                _ = try ExperimentStore.freeze(name: "gates")
                Issue.record("expected a freeze-gate refusal")
            } catch let error as ExperimentError {
                // Prose byte-stability: the exact string freeze threw before
                // the gate table existed.
                #expect(
                    error.reason == "cannot freeze 'gates': model revision is not pinned "
                        + "and test/model is not in the local HF cache — create with "
                        + "--revision, load the model once, or freeze --force")
                let refusal = try #require(error.freezeRefusal)
                #expect(refusal.reason == error.reason)
                // The reported gate is the one the prose describes …
                #expect(refusal.gate == .revision)
                #expect(
                    refusal.repairAction
                        == "create with --revision, load the model once, or freeze --force")
                // … and `gates` carries EVERY failure, in vocabulary order —
                // the same list, in the same order, that --force stamps for
                // the same manifest. Before this, the refusal path leaked the
                // number of failed gates as well as their names.
                #expect(refusal.gateIDs == ["revision", "validateEvidence"])
            }
            let forced = try ExperimentStore.freeze(name: "gates", force: true)
            #expect(forced.forcedGatesSkipped == ["revision", "validateEvidence"])
        }
    }

    /// Prose stability, asserted per gate rather than inferred: the two
    /// renderings the table produces — the refusal message and the bare
    /// reason the `⚠︎ freeze --force` line prints — are the byte-identical
    /// strings the two hand-written branches produced before the hoist.
    /// These literals were copied from the pre-hoist source; the `--force`
    /// reason is deliberately NOT the refusal with a prefix stripped
    /// (`revision` and `validateEvidence` drop their remedy clause), which
    /// is why the table carries both.
    @Test func bothRenderingsOfEveryInlineGateAreByteStable() throws {
        try withTempRoot {
            let table = ExperimentStore.freezeGateTable(name: "gates", autoCommit: false)
            func outcome(
                _ gate: FreezeGate, _ manifest: ExperimentManifest
            ) throws -> ExperimentStore.FreezeGateOutcome {
                try #require(
                    table.lazy.filter { $0.gate == gate }
                        .compactMap { $0.evaluate(manifest) }.first)
            }

            var manifest = try ExperimentStore.create(
                name: "gates", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            let revision = try outcome(.revision, manifest)
            #expect(
                revision.refusal == "cannot freeze 'gates': model revision is not pinned "
                    + "and test/model is not in the local HF cache — create with "
                    + "--revision, load the model once, or freeze --force")
            #expect(
                revision.forced == "model revision is not pinned and test/model is not in "
                    + "the local HF cache")

            manifest.modelRevision = "abc123def456"
            let evidence = try outcome(.validateEvidence, manifest)
            #expect(
                evidence.refusal == "cannot freeze 'gates': no validate run matches its "
                    + "exact pins (model+revision, concepts, neutral corpus) on the run "
                    + "substrate swift-mlx. Run 'steerlab-cli experiment validate gates' "
                    + "first, or freeze --force to record an unvalidated experiment")
            #expect(
                evidence.forced == "no validate run matches its exact pins "
                    + "(model+revision, concepts, neutral corpus) on the run substrate "
                    + "swift-mlx")
        }
    }

    /// A refusal from a gate whose check throws its own `ExperimentError`
    /// (rather than being computed inline) carries the id too, and keeps that
    /// check's message verbatim.
    @Test func aWrappedGateCheckKeepsItsProseAndGainsAnID() throws {
        try withTempRoot {
            try makeVariantStudy(
                name: "wrapped", artifact: makeVariantArtifact(adapterHash: nil))
            do {
                _ = try ExperimentStore.freeze(name: "wrapped")
                Issue.record("expected the variantValidity gate to refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("adapterHash"))
                #expect(error.reason.hasPrefix("cannot freeze 'wrapped': "))
                let refusal = try #require(error.freezeRefusal)
                #expect(refusal.gate == .variantValidity)
                #expect(refusal.repairAction.contains("freeze --force"))
                // The documented divergence, asserted rather than papered
                // over: `gates` is in VOCABULARY order (the stamp's order,
                // which puts batteryEvidence first), while `gate` follows the
                // prose in `reason` (freeze's historical refusal order, which
                // reports variantValidity first). `gate` is always a member
                // of `gates`; it is not necessarily `gates.first`.
                #expect(refusal.gateIDs == ["batteryEvidence", "variantValidity"])
            }
        }
    }
}
