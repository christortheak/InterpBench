import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// Lifecycle tests run against a temp experiments root; concept-hash checks
/// reference the real committed `french` stimulus set read-only.
@Suite(.serialized) struct ExperimentStoreTests {

    func withTempRoot<T>(_ body: () throws -> T) rethrows -> T {
        // rootOverride is process-global and other serialized SUITES also
        // use it; the shared lock keeps override windows from interleaving
        // across concurrently-running suites (ExperimentRootOverrideLock).
        try ExperimentRootOverrideLock.withTempRoot(prefix: "exp") { _ in
            try body()
        }
    }

    func realFrenchHash() throws -> String {
        try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french")
        ).hash
    }

    /// Plants a fake `validate` run directory whose manifest snapshot has
    /// this manifest's validation scope — what `ExperimentTasks.validate`
    /// would leave behind (canonical `validation-report.json` plus the
    /// byte-identical legacy `report.json`, and substrate-stamped evidence).
    func fabricateValidationEvidence(
        for manifest: ExperimentManifest,
        capabilityBattery: [CapabilityBatteryConditionResult]? = nil
    ) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260610T000000000Z-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        let report = #"{"experiment":"\#(manifest.name)","validation":{}}"#
        try report.write(
            to: dir.appending(component: "validation-report.json"),
            atomically: true,
            encoding: .utf8)
        try report.write(
            to: dir.appending(component: "report.json"),
            atomically: true,
            encoding: .utf8)
        try ExperimentStore.writeValidationEvidence(
            for: manifest, runDirectory: dir, capabilityBattery: capabilityBattery)
    }

    /// Hand-rolled evidence with an explicit substrate stamp — the Swift twin
    /// of the server test fixture `_validate_evidence_run`: `substrate: nil`
    /// fabricates legacy (pre-substrate) evidence, `reportFile: nil` omits the
    /// field so the matcher's legacy default ("report.json") applies.
    func fabricateValidationEvidence(
        for manifest: ExperimentManifest,
        substrate: String?,
        reportFile: String? = "validation-report.json",
        stamp: String = "20260610T000000000Z"
    ) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "\(stamp)-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        try #"{"experiment":"\#(manifest.name)","validation":{}}"#.write(
            to: dir.appending(component: reportFile ?? "report.json"),
            atomically: true,
            encoding: .utf8)
        var evidence: [String: Any] = [
            "schemaVersion": 1,
            "task": "validate",
            "validationScopeHash": ExperimentStore.validationScopeHash(manifest),
        ]
        if let substrate { evidence["substrate"] = substrate }
        if let reportFile { evidence["reportFile"] = reportFile }
        try JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys])
            .write(to: dir.appending(component: "validation-evidence.json"))
    }

    private func fabricateIncompleteValidationDirectory(for manifest: ExperimentManifest) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260610T000000000Z-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
    }

    @Test func createSaveLoadRoundTripAndStableHash() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "Round Trip", description: "d", modelID: "test/model")
            #expect(manifest.name == "round-trip")  // sanitized
            #expect(manifest.status == .draft)

            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: "abc", options: .init()))
            try ExperimentStore.save(manifest)
            let loaded = try ExperimentStore.load(name: "round-trip")
            #expect(loaded.concepts.count == 1)

            // Content hash ignores lifecycle fields.
            var frozenish = loaded
            frozenish.status = .frozen
            frozenish.frozenAt = "2026-06-10"
            frozenish.frozenBy = "swift"
            frozenish.gitCommit = "deadbeef"
            #expect(
                ExperimentStore.manifestHash(loaded)
                    == ExperimentStore.manifestHash(frozenish))

            // …but not content changes.
            var changed = loaded
            changed.temperature = 0.123
            #expect(
                ExperimentStore.manifestHash(loaded)
                    != ExperimentStore.manifestHash(changed))

            var changedEvaluation = loaded
            changedEvaluation.evaluation = .init(
                kind: .pairedJudge,
                judgeModel: "judge/model",
                judgePrompt: "Choose the response that better satisfies the rubric.")
            #expect(
                ExperimentStore.manifestHash(loaded)
                    != ExperimentStore.manifestHash(changedEvaluation))

            var changedPromptMode = loaded
            changedPromptMode.promptMode = .rawCompletion
            changedPromptMode.systemPrompt = "Use a terse style."
            #expect(
                ExperimentStore.manifestHash(loaded)
                    != ExperimentStore.manifestHash(changedPromptMode))
        }
    }

    @Test func freezeFailsOnStimulusHashMismatch() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "bad-pin", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: "not-the-real-hash", options: .init()))
            try ExperimentStore.save(manifest)

            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("changed since pinning") })
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "bad-pin")
            }
        }
    }

    @Test func freezeLocksManifestAndDuplicateReopens() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "lockme", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(
                    name: "french", stimulusSetHash: try realFrenchHash(),
                    options: .init()))
            manifest.conditions.append(
                .init(
                    name: "french-mid",
                    slots: [.init(concept: "french", layer: 14, alpha: 0.05)]))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)

            let frozen = try ExperimentStore.freeze(name: "lockme")
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeHash != nil)
            #expect(frozen.frozenBy == "swift")
            #expect(ExperimentStore.verify(frozen).isEmpty)

            // Frozen manifests refuse mutation…
            var mutated = frozen
            mutated.temperature = 0.9
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.save(mutated)
            }
            // …and double-freeze is rejected.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "lockme")
            }

            // Duplicate is the iteration path: fresh draft, no freeze residue.
            let copy = try ExperimentStore.duplicate(name: "lockme", as: "lockme-2")
            #expect(copy.status == .draft)
            #expect(copy.freezeHash == nil)
            #expect(copy.frozenBy == nil)
            #expect(copy.concepts == frozen.concepts)
        }
    }

    /// Server-frozen manifests are no longer exempt from post-freeze content
    /// verification: the server writes `freeze-canonical.json` (the exact
    /// bytes it hashed), and Swift verifies against that. A legacy server
    /// freeze without the canonical file is a violation, never a skip.
    @Test func serverFrozenManifestWithoutCanonicalFileIsAViolation() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "server-frozen", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            manifest.status = .frozen
            manifest.frozenAt = "2026-06-20T00:00:00Z"
            manifest.frozenBy = "server"
            manifest.freezeHash = "server-side-hash-that-swift-does-not-recompute"

            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("freeze-canonical.json") })

            // Pin drift is still caught alongside the canonical-file gap.
            manifest.concepts[0].stimulusSetHash = "bad"
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("changed since pinning")
                })
        }
    }

    /// The freeze gates the audit asked for: no validate run → no freeze;
    /// no pinned revision → no freeze; --force records an unvalidated
    /// experiment deliberately, never silently.
    @Test func freezeRequiresValidationEvidenceAndRevision() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "gated", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            // Pins are sound, but no validate run exists for this scope.
            #expect(ExperimentStore.verify(manifest).isEmpty)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "gated")
            }

            try fabricateValidationEvidence(for: manifest)
            // Adding a condition does NOT invalidate the evidence: conditions
            // are out of validation scope (they don't change the vectors).
            manifest.conditions.append(
                .init(name: "c1", slots: [.init(concept: "french", layer: 10, alpha: 0.05)]))
            try ExperimentStore.save(manifest)
            let frozen = try ExperimentStore.freeze(name: "gated")
            #expect(frozen.status == .frozen)
        }
    }

    @Test func freezeRequiresPinnedRevisionUnlessForced() throws {
        try withTempRoot {
            // "test/model" is not in the local HF cache, so the revision
            // cannot be resolved at freeze time.
            var manifest = try ExperimentStore.create(
                name: "norev", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)

            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "norev")
            }
            let forced = try ExperimentStore.freeze(name: "norev", force: true)
            #expect(forced.status == .frozen)
        }
    }

    @Test func evidenceForDifferentPinsDoesNotCount() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "drifted", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            // Evidence from a validate run against DIFFERENT options.
            var other = manifest
            other.concepts[0].options.neutralPCCount = 3
            other.neutralCorpusHash = "differenthash"
            try fabricateValidationEvidence(for: other)

            #expect(ExperimentStore.validationEvidence(for: manifest) == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "drifted")
            }
        }
    }

    @Test func incompleteValidationDirectoryDoesNotCount() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "partial", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            try fabricateIncompleteValidationDirectory(for: manifest)

            #expect(ExperimentStore.validationEvidence(for: manifest) == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "partial")
            }
        }
    }

    /// Same-substrate rule (twin of the server's
    /// `test_validate_evidence_is_substrate_scoped`): evidence stamped with a
    /// foreign substrate never satisfies this engine's freeze — CUDA/HF
    /// activations do not match MLX/Metal — while own-substrate evidence
    /// (even with a custom reportFile) does.
    @Test func validateEvidenceIsSubstrateScoped() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "sub", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            // Foreign-substrate evidence never counts.
            try fabricateValidationEvidence(
                for: manifest, substrate: "python-hf-transformers",
                stamp: "20260610T000000001Z")
            #expect(ExperimentStore.validationEvidence(for: manifest) == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "sub")
            }

            // Own-substrate evidence (with a custom reportFile) does.
            try fabricateValidationEvidence(
                for: manifest, substrate: "swift-mlx",
                reportFile: "report.json", stamp: "20260610T000000002Z")
            #expect(ExperimentStore.validationEvidence(for: manifest) != nil)
            let frozen = try ExperimentStore.freeze(name: "sub")
            #expect(frozen.status == .frozen)
        }
    }

    /// Pre-substrate evidence was necessarily produced by THIS engine (the
    /// historical filename divergence made cross-engine evidence impossible),
    /// so unstamped legacy evidence still counts (twin of the server's
    /// `test_legacy_unstamped_evidence_still_counts`).
    @Test func legacyUnstampedEvidenceStillCounts() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "leg", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            try ExperimentStore.save(manifest)

            // No substrate stamp, no reportFile field: the matcher must fall
            // back to the legacy Swift default "report.json".
            try fabricateValidationEvidence(for: manifest, substrate: nil, reportFile: nil)
            #expect(ExperimentStore.validationEvidence(for: manifest) != nil)
            let frozen = try ExperimentStore.freeze(name: "leg")
            #expect(frozen.status == .frozen)
        }
    }

    @Test func legacyNeutralProjectionBlocksVerifiedExperiments() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "legacy-projection", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(
                    name: "french",
                    stimulusSetHash: try realFrenchHash(),
                    options: .init(neutralPCCount: 3)))
            manifest.neutralCorpusHash = "placeholder"
            try ExperimentStore.save(manifest)

            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("legacy fixed-k neutral PC projection is disabled")
                })
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "legacy-projection", force: true)
            }
        }
    }

    /// Run directories are immutable; two same-instant invocations must not
    /// alias one directory ("never overwrite or mutate a run").
    @Test func runDirectoriesNeverCollide() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "runs-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        var seen = Set<String>()
        for _ in 0 ..< 5 {
            let url = try VectorCatalog.makeUniqueRunDirectory(
                slug: "exp-collide-validate", under: temp)
            #expect(!seen.contains(url.path))
            seen.insert(url.path)
            #expect(FileManager.default.fileExists(atPath: url.path))
        }
    }

    /// The same guarantee under CONCURRENCY — the production case: generation
    /// shards create run directories at once on a shared filesystem, so many
    /// callers resolve one millisecond stamp. A check-then-create loop lets
    /// two of them agree on a free name, and since
    /// `createDirectory(withIntermediateDirectories: true)` does not fail on
    /// an existing directory, both would be handed the SAME run directory and
    /// silently write over each other. Creation must be the exclusivity test.
    @Test func concurrentRunDirectoriesNeverAlias() throws {
        /// Collector across the concurrent iterations; `@unchecked Sendable`
        /// because the mutable state is guarded by the lock it carries.
        final class Collector: @unchecked Sendable {
            private let lock = NSLock()
            private(set) var paths: [String] = []
            private(set) var errors: [Error] = []
            func record(_ path: String) {
                lock.lock(); defer { lock.unlock() }
                paths.append(path)
            }
            func record(_ error: Error) {
                lock.lock(); defer { lock.unlock() }
                errors.append(error)
            }
        }

        let temp = FileManager.default.temporaryDirectory
            .appending(component: "runs-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        let workers = 16
        let collector = Collector()
        DispatchQueue.concurrentPerform(iterations: workers) { _ in
            do {
                let url = try VectorCatalog.makeUniqueRunDirectory(
                    slug: "exp-shard-run", under: temp)
                collector.record(url.path)
            } catch {
                collector.record(error)
            }
        }
        #expect(collector.errors.isEmpty)
        #expect(collector.paths.count == workers)
        // The assertion that matters: no two shards were handed one directory.
        #expect(Set(collector.paths).count == workers)
        for path in collector.paths {
            #expect(FileManager.default.fileExists(atPath: path))
        }
    }

    /// Phase-D science fields: populated manifests round-trip, pre-Phase-D
    /// manifests decode unchanged, and nil science fields are omitted from
    /// the encoding so existing manifests keep their content hash.
    @Test func scienceFieldsRoundTripAndOldManifestsDecode() throws {
        var manifest = ExperimentManifest(
            name: "science", description: "", modelID: "test/model")
        manifest.phase = "screen"
        manifest.caseFamily = "sentencing"
        manifest.outcomeInstruments = ["answerTokenLogprob", "sampledText"]
        manifest.samplesPerItem = 8
        manifest.seedPolicy = "derivedSHA256"
        manifest.screenTaskPromptsHash = "screen-pool-hash"
        manifest.humanBaseline = .init(path: "prompts/human/anchoring.csv", hash: "abc123")
        manifest.promotionRule = .init(
            fdrThreshold: 0.05, doseMonotone: true, exceedsRandomFloor: true,
            capabilityGate: "battery")
        manifest.conditions.append(
            .init(name: "random-floor", slots: [], controlType: "randomMatchedNorm"))

        let encoder = JSONEncoder()
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(decoded == manifest)
        let again = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(decoded))
        #expect(again == manifest)
        #expect(again.conditions.first?.controlType == "randomMatchedNorm")

        // A pre-Phase-D manifest (no science keys) still decodes, with nils.
        let legacy = """
            {"name": "old", "experimentDescription": "", "createdAt": "2026-01-01",
             "modelID": "test/model", "status": "draft"}
            """
        let old = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(legacy.utf8))
        #expect(old.phase == nil)
        #expect(old.samplesPerItem == nil)
        #expect(old.humanBaseline == nil)
        #expect(old.promotionRule == nil)

        // Nil science fields must not appear in the encoding — otherwise the
        // content hash of every existing manifest would silently change.
        let plain = ExperimentManifest(
            name: "plain", description: "", modelID: "test/model")
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(plain))
                as? [String: Any])
        for key in [
            "phase", "caseFamily", "outcomeInstruments", "samplesPerItem",
            "seedPolicy", "screenTaskPromptsHash", "humanBaseline", "promotionRule",
        ] {
            #expect(json[key] == nil, "nil \(key) must be omitted from encoding")
        }

        // …and populating them DOES change the content hash.
        #expect(
            ExperimentStore.manifestHash(plain) != ExperimentStore.manifestHash(manifest))
    }

    @Test func confirmPhaseRequiresDisjointScreenPool() throws {
        var manifest = ExperimentManifest(
            name: "confirm-gate", description: "", modelID: "test/model")
        manifest.phase = "confirm"
        var violations = ExperimentStore.verify(manifest)
        #expect(violations.contains { $0.contains("must pin screenTaskPromptsHash") })

        manifest.screenTaskPromptsHash = "pool-hash"
        manifest.taskPromptsHash = "pool-hash"
        violations = ExperimentStore.verify(manifest)
        #expect(violations.contains { $0.contains("IDENTICAL to the screen pool") })

        manifest.taskPromptsHash = "held-out-pool-hash"
        violations = ExperimentStore.verify(manifest)
        #expect(!violations.contains { $0.contains("screen pool") })
    }

    @Test func stochasticSamplingRequiresTemperatureAndDerivedSeeds() throws {
        var manifest = ExperimentManifest(
            name: "stochastic-gate", description: "", modelID: "test/model")
        manifest.samplesPerItem = 4
        var violations = ExperimentStore.verify(manifest)
        #expect(violations.contains { $0.contains("requires temperature > 0") })
        #expect(violations.contains { $0.contains("requires seedPolicy 'derivedSHA256'") })

        manifest.temperature = 0.7
        manifest.seedPolicy = "derivedSHA256"
        violations = ExperimentStore.verify(manifest)
        #expect(!violations.contains { $0.contains("samplesPerItem") })
    }

    @Test func verifyAcceptsServerStyleChoicePromptFiles() throws {
        // Regression (found live 2026-07-05): verify() checked task prompts
        // through the text-only stimulus loader, so a valid `{"prompt":…,
        // "options":…}` choice file passed the RUN loop but failed
        // verification — which would have blocked freeze. Verify must use the
        // same parser that will actually read the file.
        let file = FileManager.default.temporaryDirectory
            .appending(component: "choice-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: file) }
        let contents = Data(
            """
            {"id":"a","prompt":"Pick one.","options":[" Yes"," No"],"target":" Yes"}
            """.utf8)
        try contents.write(to: file)
        let hash = SHA256.hash(data: contents)
            .map { String(format: "%02x", $0) }.joined()

        var manifest = ExperimentManifest(
            name: "choice-gate", description: "", modelID: "test/model")
        manifest.taskPromptsFile = file.path
        manifest.taskPromptsHash = hash
        #expect(!ExperimentStore.verify(manifest).contains { $0.contains("task prompts") })

        manifest.taskPromptsHash = "not-the-hash"
        #expect(
            ExperimentStore.verify(manifest).contains {
                $0.contains("task prompts changed since pinning")
            })

        manifest.taskPromptsFile = file.path + ".missing"
        manifest.taskPromptsHash = hash
        #expect(
            ExperimentStore.verify(manifest).contains {
                $0.contains("task prompts missing")
            })

        let malformed = Data("not json\n".utf8)
        try malformed.write(to: file)
        manifest.taskPromptsFile = file.path
        manifest.taskPromptsHash = SHA256.hash(data: malformed)
            .map { String(format: "%02x", $0) }.joined()
        #expect(
            ExperimentStore.verify(manifest).contains {
                $0.contains("task prompts unparseable")
            })
    }

    @Test func humanBaselineHashIsVerified() throws {
        let file = FileManager.default.temporaryDirectory
            .appending(component: "human-baseline-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: file) }
        // Loader-shaped contents: verify also SHAPE-checks a hash-clean
        // baseline (2026-07-19), so the clean case must be a table the
        // analyze loader accepts.
        let contents = Data(
            "endpoint,deltaHuman,ciLower,ciUpper\naffirmRate,0.12,0.05,0.19\n"
                .utf8)
        try contents.write(to: file)
        let hash = SHA256.hash(data: contents)
            .map { String(format: "%02x", $0) }.joined()

        var manifest = ExperimentManifest(
            name: "human-gate", description: "", modelID: "test/model")
        manifest.humanBaseline = .init(path: file.path, hash: hash)
        #expect(!ExperimentStore.verify(manifest).contains { $0.contains("human baseline") })
        #expect(!ExperimentStore.verify(manifest).contains { $0.contains("analyze step") })

        manifest.humanBaseline = .init(path: file.path, hash: "not-the-hash")
        #expect(
            ExperimentStore.verify(manifest).contains {
                $0.contains("human baseline") && $0.contains("changed since pinning")
            })

        manifest.humanBaseline = .init(path: file.path + ".missing", hash: hash)
        #expect(
            ExperimentStore.verify(manifest).contains {
                $0.contains("human baseline missing")
            })
    }

    @Test func conditionsMustReferenceAttachedConcepts() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "dangling", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            manifest.conditions.append(
                .init(name: "bad", slots: [.init(concept: "fear", layer: 10, alpha: 1)]))
            try ExperimentStore.save(manifest)
            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("unattached concept 'fear'") })
        }
    }

    // MARK: - RepE reader pins (readerRefs + repeReaderScore instrument)

    private func manualReaderArtifact(
        substrate: String = RepEReader.substrate,
        modelID: String = "test/model",
        layer: Int = 0
    ) -> RepEReader.Artifact {
        RepEReader.Artifact(
            modelID: modelID, revision: "abc", concept: "fear", layer: layer,
            template: RepEReader.TaskTemplate(
                id: "unnamed-scenario-v1", conceptSlot: false,
                text: "S: {{stimulus}} q", latToken: "final", hash: "th",
                divergence: "unnamed-clean-room"),
            datasetHash: "dh",
            probe: SteeringVectorMath.ScalarProbe(
                direction: [1, 0], activationCenter: [1, 0],
                projectionCenter: 0.5, projectionScale: 2, orientation: 1,
                positiveMean: 1, negativeMean: -1),
            differenceCloudExplainedVariance: 0.9, trainAccuracy: 1, heldOutAccuracy: 0.8,
            trainPairCount: 4, heldOutPairCount: 2, substrate: substrate)
    }

    /// Writes one reader artifact under the temp runs root and returns a
    /// pinned ref (relative path + raw-byte hash) — the server test fixture
    /// `_reader_tree`'s twin.
    private func plantReader(
        substrate: String = RepEReader.substrate,
        modelID: String = "test/model",
        runName: String = "20260703T000000000-reader-fit"
    ) throws -> ExperimentManifest.ReaderRef {
        let runDir = ExperimentStore.runsDirectory.appending(component: runName)
        let url = try RepEReader.saveArtifact(
            manualReaderArtifact(substrate: substrate, modelID: modelID), to: runDir)
        let hash = SHA256.hash(data: try Data(contentsOf: url))
            .map { String(format: "%02x", $0) }.joined()
        return .init(
            path: "runs/\(runName)/\(url.lastPathComponent)",
            hash: hash, concept: "fear")
    }

    @Test func readerRefsRoundTripAndLegacyHashStable() throws {
        var manifest = ExperimentManifest(
            name: "reader-pins", description: "", modelID: "test/model")
        let plainHash = ExperimentStore.manifestHash(manifest)

        // Nil readerRefs must be omitted from the encoding — otherwise the
        // content hash of every existing manifest would silently change.
        let encoder = JSONEncoder()
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(manifest))
                as? [String: Any])
        #expect(json["readerRefs"] == nil, "nil readerRefs must be omitted")
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(ExperimentStore.manifestHash(decoded) == plainHash)

        // …and pinning readers round-trips and changes the hash.
        manifest.readerRefs = [
            .init(path: "runs/r/reader-fear-layer0.json", hash: "h", concept: "fear")
        ]
        manifest.outcomeInstruments = ["repeReaderScore", "sampledText"]
        let repinned = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(repinned == manifest)
        #expect(repinned.readerRefs?.first?.concept == "fear")
        #expect(ExperimentStore.manifestHash(manifest) != plainHash)
    }

    @Test func verifyAcceptsCleanReaderPin() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "reader-ok", description: "", modelID: "test/model")
            manifest.outcomeInstruments = ["repeReaderScore", "sampledText"]
            manifest.readerRefs = [try plantReader()]
            let violations = ExperimentStore.verify(manifest)
            #expect(!violations.contains { $0.contains("reader") }, "\(violations)")
        }
    }

    @Test func verifyRejectsForeignSubstrateReader() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "reader-foreign", description: "", modelID: "test/model")
            manifest.outcomeInstruments = ["repeReaderScore"]
            manifest.readerRefs = [try plantReader(substrate: "python-hf-transformers")]
            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("python-hf-transformers") && $0.contains("substrate-specific")
                }, "\(violations)")
        }
    }

    @Test func verifyFlagsReaderDriftMissingRefsAndModelMismatch() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "reader-drift", description: "", modelID: "test/model")
            manifest.outcomeInstruments = ["repeReaderScore"]
            let ref = try plantReader()
            manifest.readerRefs = [ref]

            // Hash drift after pinning — the firewall catches byte changes.
            let url = ExperimentStore.resolveProjectPath(ref.path)
            var bytes = try Data(contentsOf: url)
            bytes.append(Data("\n".utf8))
            try bytes.write(to: url)
            var violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("reader 'fear'") && $0.contains("changed since pinning")
                }, "\(violations)")

            // Instrument requested but nothing pinned.
            manifest.readerRefs = []
            violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("no readerRefs") })

            // A non-reader JSON (e.g. a steering-vector sidecar) is refused
            // as an artifact-type violation even when its hash matches.
            let sidecarPath = "runs/20260703T000000000-reader-fit/not-a-reader.json"
            let sidecarURL = ExperimentStore.resolveProjectPath(sidecarPath)
            let sidecarBytes = Data(#"{"modelID": "test/model"}"#.utf8)
            try sidecarBytes.write(to: sidecarURL)
            let sidecarHash = SHA256.hash(data: sidecarBytes)
                .map { String(format: "%02x", $0) }.joined()
            manifest.readerRefs = [
                .init(path: sidecarPath, hash: sidecarHash, concept: "fear")
            ]
            violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains { $0.contains("not a repe-reader-lat artifact") },
                "\(violations)")

            // Reader fitted on a different model than the study's.
            manifest.readerRefs = [
                try plantReader(
                    modelID: "org/other", runName: "20260703T000000001-reader-fit")
            ]
            violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("org/other") && $0.contains("study model")
                }, "\(violations)")

            // Missing file.
            manifest.readerRefs = [
                .init(path: "runs/nowhere/reader.json", hash: "h", concept: "fear")
            ]
            violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("reader 'fear' missing") })
        }
    }

    @Test func generationRecordStampsReaderScoresKey() throws {
        // Record-shape parity with the server: the sampled-output reader
        // readout is keyed "readerScores" ({concept: score}) and omitted when
        // the instrument is off — legacy records keep their exact shape.
        let record = ExperimentTasks.GenerationRecord(
            experiment: "e", experimentHash: "h", modelID: "m", modelRevision: nil,
            taskPromptsFile: "f", taskPromptsHash: "th", promptMode: "chatAssistant",
            systemPrompt: nil, qwenThinkingEnabled: false, condition: "baseline",
            seed: 1, seedInert: true, promptIndex: 1, promptID: "p1", prompt: "q",
            output: "a", wordCount: 1, distinct2: 0, markerDensity: [:],
            variantArtifactPath: nil, variantArtifactHash: nil, target: nil,
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
            parsedMonths: .none, parsedChoice: .none,
            readerScores: ["fear": 0.25])
        let json = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(record))
                as? [String: Any])
        let scores = try #require(json["readerScores"] as? [String: Double])
        #expect(abs((scores["fear"] ?? 0) - 0.25) < 1e-6)

        var plain = record
        plain.readerScores = nil
        let plainJSON = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(plain))
                as? [String: Any])
        #expect(plainJSON["readerScores"] == nil, "nil readerScores must be omitted")
    }

    @Test func generationRecordStampsRandomVectorAlgorithm() throws {
        // Random-control provenance: records of a randomMatchedNorm condition
        // carry "randomVectorAlgorithm": "gaussian-isotropic-v1" (the server
        // stamps the identical string inside interventionState). Non-control
        // records omit the key — and so do legacy runs, which is exactly how
        // old Swift cube-uniform controls stay distinguishable from new
        // Gaussian ones.
        var record = ExperimentTasks.GenerationRecord(
            experiment: "e", experimentHash: "h", modelID: "m", modelRevision: nil,
            taskPromptsFile: "f", taskPromptsHash: "th", promptMode: "chatAssistant",
            systemPrompt: nil, qwenThinkingEnabled: false, condition: "fear-control",
            seed: 1, seedInert: true, promptIndex: 1, promptID: "p1", prompt: "q",
            output: "a", wordCount: 1, distinct2: 0, markerDensity: [:],
            variantArtifactPath: nil, variantArtifactHash: nil, target: nil,
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
            parsedMonths: .none, parsedChoice: .none)
        record.randomVectorAlgorithm = SteeringVectorMath.randomVectorAlgorithm
        let json = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(record))
                as? [String: Any])
        #expect(json["randomVectorAlgorithm"] as? String == "gaussian-isotropic-v1")

        var plain = record
        plain.randomVectorAlgorithm = nil
        let plainJSON = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(plain))
                as? [String: Any])
        #expect(
            plainJSON["randomVectorAlgorithm"] == nil,
            "non-control records must omit the stamp")
    }
}

/// A server-only manifest key must survive this engine's round trips.
///
/// J-lens work is server-only by rule (imported lens artifacts are
/// PyTorch/HF-native), but manifests are shared: this engine loads, duplicates,
/// and re-saves them. Before the passthrough, `jlensReadout` was absent from
/// `CodingKeys`, so `duplicate` silently dropped a scientific pin — the copy
/// declared no readout, would freeze cleanly, and would measure nothing.
@Suite(.serialized) struct JLensReadoutPassthroughTests {

    private func manifestJSON(name: String) -> String {
        """
        {"name":"\(name)","experimentDescription":"d","modelID":"google/gemma-3-27b-it",
         "modelRevision":"005ad3404e59d6023443cb575daa05336842228a","dtype":"bfloat16",
         "status":"draft","createdAt":"2026-07-29T00:00:00Z","concepts":[],
         "jlensReadout":{"lensID":"google--gemma-3-27b-it--jlens-wikitext",
           "lensSHA256":"abc123","layers":[16,24],"watchlist":[23648],"topK":5,
           "configHash":"deadbeef","qualificationID":"q1",
           "logitLensCompanion":true}}
        """
    }

    @Test func decodePreservesTheServerAuthoredBlock() throws {
        let data = Data(manifestJSON(name: "s").utf8)
        let manifest = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        guard case .object(let block)? = manifest.jlensReadout else {
            Issue.record("jlensReadout did not decode as an object")
            return
        }
        #expect(block["lensID"] == .string("google--gemma-3-27b-it--jlens-wikitext"))
        #expect(block["configHash"] == .string("deadbeef"))
        // Nested arrays survive too — layers is what a run actually arms.
        #expect(block["layers"] == .array([.number(16), .number(24)]))
    }

    @Test func reEncodeDoesNotDropThePin() throws {
        let data = Data(manifestJSON(name: "s").utf8)
        let decoded = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        let round = try JSONDecoder().decode(
            ExperimentManifest.self, from: try JSONEncoder().encode(decoded))
        #expect(round.jlensReadout == decoded.jlensReadout)
    }

    @Test func anAbsentBlockStaysAbsentAndAddsNoKey() throws {
        // Otherwise every existing manifest would gain `jlensReadout: null` on
        // re-save and change its content hash.
        var manifest = ExperimentManifest(
            name: "s", description: "", modelID: "google/gemma-3-4b-it")
        manifest.jlensReadout = nil
        let encoded = try JSONEncoder().encode(manifest)
        let text = String(decoding: encoded, as: UTF8.self)
        #expect(!text.contains("jlensReadout"))
    }
}
