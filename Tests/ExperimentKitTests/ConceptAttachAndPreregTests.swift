import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// A8 (direct concept attach/detach through the store) and A9 (local
/// freeze's preregistration.md export) plus the A11 helpers. Declared as an
/// extension of the serialized `ExperimentStoreTests` suite because these
/// tests share its `rootOverride` test seam (a process-global); the paired
/// checks reference the real committed `french` concept read-only, exactly
/// like the suite's other stimulus-hash tests.
extension ExperimentStoreTests {

    private func withAttachTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "attach", body)
    }

    private func plantStories(
        _ name: String, root: URL, validation: String? = nil
    ) throws {
        let directory = root.appending(components: "prompts", "emotions", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try """
        {"concept": "\(name)", "text": "a story about \(name)"}
        {"concept": "\(name)", "text": "another story about \(name)"}

        """.write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
        if let validation {
            try validation.write(
                to: directory.appending(component: "validation.jsonl"),
                atomically: true, encoding: .utf8)
        }
    }

    // MARK: - A8: store attach pins stimulus hash + validationHash

    @Test func attachConceptPinsGrandMeanRecipeAndValidation() throws {
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "attach-gm", description: "", modelID: "test/model")
            let validation =
                #"{"text": "a hidden scenario", "expresses": true}"# + "\n"
            try plantStories("serenity", root: root, validation: validation)
            try plantStories("dread", root: root)

            let manifest = try ExperimentStore.attachConcept(
                "serenity",
                method: .emotionGrandMean,
                poolFromToken: 40,
                corpusConcepts: ["dread"],
                experimentName: "attach-gm")

            let ref = try #require(
                manifest.concepts.first { $0.name == "serenity" })
            #expect(ref.options.method == .emotionGrandMean)
            #expect(ref.options.readingPosition.label == "mean from token 40")
            // Stimulus pin is the stories hash.
            #expect(ref.stimulusSetHash == ExperimentStore.storiesHash(for: "serenity"))
            // Measurement-side pin: the validation.jsonl bytes.
            let expectedValidation = SHA256.hash(data: Data(validation.utf8))
                .map { String(format: "%02x", $0) }.joined()
            #expect(ref.validationHash == expectedValidation)
            // Grand-mean population pinned, target always a member.
            let corpus = try #require(manifest.grandMeanCorpus)
            #expect(Set(corpus.concepts) == ["serenity", "dread"])
            #expect(corpus.hashes["dread"] == ExperimentStore.storiesHash(for: "dread"))
            #expect(ExperimentStore.verify(manifest).isEmpty)
        }
    }

    @Test func attachConceptPinsPairedRecipeWithOptions() throws {
        try withAttachTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "attach-paired", description: "", modelID: "test/model")
            let manifest = try ExperimentStore.attachConcept(
                "french", method: .lat, poolFromToken: 7,
                experimentName: "attach-paired")
            let ref = try #require(manifest.concepts.first { $0.name == "french" })
            #expect(ref.stimulusSetHash == (try realFrenchHash()))
            #expect(ref.options.method == .lat)
            #expect(ref.options.readingPosition.label == "mean from token 7")
            // The validationHash key is ALWAYS written by new attaches:
            // either a real pin or an explicit pinned-absent null.
            #expect(ref.validationHash != nil || ref.validationHashPinnedAbsent)
        }
    }

    @Test func attachConceptRefusesFrozenStudy() throws {
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "attach-frozen", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "attach-frozen")
            _ = try ExperimentStore.freeze(name: "attach-frozen", force: true)

            try plantStories("panic", root: root)
            #expect {
                try ExperimentStore.attachConcept(
                    "panic", method: .emotionGrandMean,
                    experimentName: "attach-frozen")
            } throws: { error in
                "\(error)".contains("frozen")
                    && "\(error)".contains("duplicate")
            }
        }
    }

    @Test func detachConceptRemovesPinAndGuardsConditions() throws {
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "detach", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            var manifest = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "detach")
            manifest = try ExperimentStore.upsertCondition(
                .init(
                    name: "calm-low",
                    slots: [.init(concept: "calm", layer: 10, alpha: 0.1)]),
                experimentName: "detach")

            // Referenced by a condition: refuse with the condition named.
            #expect {
                try ExperimentStore.detachConcept("calm", experimentName: "detach")
            } throws: { error in
                "\(error)".contains("calm-low")
            }

            _ = try ExperimentStore.removeCondition(
                named: "calm-low", experimentName: "detach")
            let detached = try ExperimentStore.detachConcept(
                "calm", experimentName: "detach")
            #expect(detached.concepts.isEmpty)
            // Last grand-mean target gone → the pinned population goes too.
            #expect(detached.grandMeanCorpus == nil)

            // Detaching something unattached refuses.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.detachConcept("calm", experimentName: "detach")
            }
        }
    }

    @MainActor
    @Test func conceptPinStatusLineShowsThreeStateValidationPin() {
        let pinned = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: "abcdef0123456789", options: .init(),
            validationHash: "1234567890abcdef")
        #expect(
            ExperimentPanel.conceptPinStatusLine(pinned)
                == "stimuli @ abcdef012345… · meanDifference · last token · "
                + "validation @ 1234567890ab…")
        let pinnedAbsent = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: "abcdef0123456789", options: .init(),
            validationHash: nil, validationHashPinnedAbsent: true)
        #expect(
            ExperimentPanel.conceptPinStatusLine(pinnedAbsent)
                .contains("validation pinned absent"))
        let legacy = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: "abcdef0123456789", options: .init())
        #expect(
            ExperimentPanel.conceptPinStatusLine(legacy)
                .contains("validation unpinned (legacy attach)"))
    }

    // MARK: - A11: extract helpers

    @MainActor
    @Test func extractAllowedForDraftAndFrozenAlike() {
        // The gate is deliberately status-blind: only busy/violations/
        // residency refuse — never the lifecycle state.
        #expect(
            ExperimentPanel.extractDisabledReason(
                busy: false, hasViolations: false, missingOnServer: false) == nil)
        #expect(
            ExperimentPanel.extractDisabledReason(
                busy: true, hasViolations: false, missingOnServer: false) != nil)
        #expect(
            ExperimentPanel.extractDisabledReason(
                busy: false, hasViolations: true, missingOnServer: false) != nil)
        #expect(
            ExperimentPanel.extractDisabledReason(
                busy: false, hasViolations: false, missingOnServer: true)?
                .contains("Submit Bundle") == true)
    }

    @Test func newestRunDirectoryMatchesTaskAndSuffix() throws {
        try withAttachTempRoot { _ in
            let runs = ExperimentStore.runsDirectory
            let names = [
                "20260101T000000000Z-exp-x-extract",
                "20260102T000000000Z-exp-x-extract-2",
                "20260103T000000000Z-exp-xy-extract",  // different study
                "20260104T000000000Z-exp-x-validate",  // different task
            ]
            for name in names {
                try FileManager.default.createDirectory(
                    at: runs.appending(component: name),
                    withIntermediateDirectories: true)
            }
            #expect(
                ExperimentStore.newestRunDirectory(experimentName: "x", task: "extract")?
                    .lastPathComponent == "20260102T000000000Z-exp-x-extract-2")
            #expect(
                ExperimentStore.newestRunDirectory(experimentName: "xy", task: "extract")?
                    .lastPathComponent == "20260103T000000000Z-exp-xy-extract")
            #expect(
                ExperimentStore.newestRunDirectory(experimentName: "none", task: "extract")
                    == nil)
        }
    }

    // MARK: - A9: preregistration export

    private func preregFixtureManifest() -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "prereg-fixture", description: "fixture", modelID: "test/model")
        manifest.modelRevision = "abc123"
        manifest.phase = "screen"
        manifest.caseFamily = "sentencing"
        manifest.outcomeInstruments = ["answerTokenLogprob", "sampledText"]
        manifest.temperature = 0.7
        manifest.samplesPerItem = 5
        manifest.seedPolicy = "derivedSHA256"
        manifest.conditions = [
            .init(name: "baseline", slots: []),
            .init(
                name: "fear-mid",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)]),
            .init(
                name: "fear-mid-random",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)],
                controlType: "randomMatchedNorm"),
        ]
        manifest.promotionRule = .init(
            fdrThreshold: 0.05, doseMonotone: true, exceedsRandomFloor: true,
            capabilityGate: "battery within 0.05 of baseline")
        manifest.humanBaseline = .init(
            path: "prompts/baselines/sentencing.csv", hash: "feedbeef")
        manifest.status = .frozen
        manifest.frozenAt = "2026-07-13T00:00:00Z"
        manifest.freezeHash = "cafe0123"
        manifest.gitCommit = "deadbeef"
        return manifest
    }

    @Test func preregistrationMarkdownCarriesServerSectionsAndFacts() {
        let text = ExperimentStore.preregistrationMarkdown(preregFixtureManifest())
        // Section headings — the cross-engine parity contract (server:
        // experiment_store._write_preregistration).
        #expect(text.contains("# Preregistration: prereg-fixture"))
        #expect(text.contains("## Conditions"))
        #expect(text.contains("## Promotion rule (screen → confirm)"))
        #expect(text.contains("## Human baseline"))
        #expect(text.contains("## Statistics"))
        // Key facts.
        #expect(text.contains("- **Frozen at:** 2026-07-13T00:00:00Z"))
        #expect(text.contains("- **Freeze hash:** `cafe0123`"))
        #expect(text.contains("- **Git commit:** `deadbeef`"))
        #expect(text.contains("- **Model:** test/model @ `abc123`"))
        #expect(text.contains("- **Phase:** screen"))
        #expect(text.contains("- **Case family:** sentencing"))
        #expect(
            text.contains("- **Outcome instruments:** answerTokenLogprob, sampledText"))
        #expect(
            text.contains(
                "- **Sampling:** temperature 0.7, samplesPerItem 5, "
                    + "seedPolicy derivedSHA256"))
        #expect(text.contains("- **baseline**: none (baseline)"))
        #expect(text.contains("- **fear-mid**: fear@L14 α=0.8"))
        #expect(
            text.contains("- **fear-mid-random** [randomMatchedNorm]: fear@L14 α=0.8"))
        #expect(text.contains("- FDR threshold: 0.05"))
        #expect(text.contains("- Capability gate: battery within 0.05 of baseline"))
        #expect(
            text.contains("- `prompts/baselines/sentencing.csv` (SHA-256 `feedbeef`)"))
        #expect(text.contains("R = delta_model − delta_human"))
        #expect(text.contains("*Generated at freeze; do not edit."))
    }

    @Test func preregistrationMarkdownOmitsAbsentSections() {
        var manifest = ExperimentManifest(
            name: "prereg-min", description: "", modelID: "test/model")
        manifest.status = .frozen
        let text = ExperimentStore.preregistrationMarkdown(manifest)
        #expect(text.contains("## Conditions"))
        #expect(text.contains("## Statistics"))
        #expect(!text.contains("## Promotion rule"))
        #expect(!text.contains("## Human baseline"))
        // Absent science fields render the server's defaults.
        #expect(text.contains("- **Phase:** unspecified"))
        #expect(text.contains("- **Outcome instruments:** sampledText"))
        #expect(
            text.contains(
                "- **Sampling:** temperature 0, samplesPerItem 1, "
                    + "seedPolicy manifestSeeds"))
    }

    @Test func localFreezeWritesPreregistrationFile() throws {
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-freeze", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-freeze")
            let frozen = try ExperimentStore.freeze(name: "prereg-freeze", force: true)
            let url = ExperimentStore.directory.appending(
                components: "prereg-freeze", "preregistration.md")
            let text = try String(contentsOf: url, encoding: .utf8)
            #expect(text.contains("# Preregistration: prereg-freeze"))
            #expect(text.contains("`\(frozen.freezeHash ?? "?")`"))
            #expect(text.contains("## Statistics"))
        }
    }
}
