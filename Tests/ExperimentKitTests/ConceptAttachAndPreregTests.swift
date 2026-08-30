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
                "french", method: .pairedDifferencePCA, poolFromToken: 7,
                experimentName: "attach-paired")
            let ref = try #require(manifest.concepts.first { $0.name == "french" })
            #expect(ref.stimulusSetHash == (try realFrenchHash()))
            #expect(ref.options.method == .pairedDifferencePCA)
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

    @Test func localFreezePreservesHandAuthoredPreregistration() throws {
        // Field incident 2026-08-29: a researcher-authored ANALYSIS
        // preregistration at experiments/<name>/preregistration.md —
        // commitments written before any data existed — was silently
        // destroyed by the first freeze. It must survive byte-for-byte,
        // with the generated settings summary landing beside it under its
        // own name.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-authored", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean,
                experimentName: "prereg-authored")
            let experimentDirectory = ExperimentStore.directory.appending(
                component: "prereg-authored")
            let authored = "# Analysis preregistration\n\n"
                + "We commit to the paired-difference estimator before any "
                + "data exists.\n"
            try authored.write(
                to: experimentDirectory.appending(component: "preregistration.md"),
                atomically: true, encoding: .utf8)
            _ = try ExperimentStore.freeze(name: "prereg-authored", force: true)
            let preserved = try String(
                contentsOf: experimentDirectory.appending(
                    component: "preregistration.md"),
                encoding: .utf8)
            #expect(preserved == authored)  // preserved untouched
            let displaced = try String(
                contentsOf: experimentDirectory.appending(
                    component: ExperimentStore.preregistrationFrozenSettingsFilename),
                encoding: .utf8)
            #expect(displaced.contains("# Preregistration: prereg-authored"))
            #expect(displaced.contains(
                ExperimentStore.preregistrationGeneratedMarker))
        }
    }

    @Test func localFreezeRefreshesStaleGeneratedPreregistration() throws {
        // LEGACY structural path: a file with no stamp to check it against is
        // the freeze's own prior output only when it LOOKS exactly like one —
        // generated header on line 1, marker line last. Such a file (copied
        // in by hand from another study, or written by a freeze predating the
        // stamps) is regenerated in sync with THIS freeze instead of
        // displacing the summary.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-stale", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-stale")
            let experimentDirectory = ExperimentStore.directory.appending(
                component: "prereg-stale")
            try ("# Preregistration: other-study\n\nstale facts\n\n"
                + ExperimentStore.preregistrationGeneratedMarker
                + " Duplicate the experiment to change anything.*\n").write(
                    to: experimentDirectory.appending(
                        component: "preregistration.md"),
                    atomically: true, encoding: .utf8)
            _ = try ExperimentStore.freeze(name: "prereg-stale", force: true)
            let text = try String(
                contentsOf: experimentDirectory.appending(
                    component: "preregistration.md"),
                encoding: .utf8)
            #expect(!text.contains("stale facts"))
            #expect(text.contains("# Preregistration: prereg-stale"))
            #expect(
                !FileManager.default.fileExists(
                    atPath: experimentDirectory.appending(
                        component: ExperimentStore.preregistrationFrozenSettingsFilename
                    ).path))
        }
    }

    @Test func localFreezePreservesAPreregistrationThatQuotesTheFooter() throws {
        // Review 2026-08-29, P1: the marker test was a SUBSTRING check, so the
        // most natural way to write a real preregistration — start from the
        // generated summary, add the commitments above it, leave the footer
        // alone — was classified as generated and destroyed. Position, not
        // presence: the marker is only ours when it is the final non-empty
        // line AND the generated header is line 1.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-quoting", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-quoting")
            let experimentDirectory = ExperimentStore.directory.appending(
                component: "prereg-quoting")
            let authored = "# Analysis preregistration\n\n"
                + "We commit to the paired-difference estimator.\n\n"
                + "The settings summary a previous freeze wrote follows:\n\n"
                + ExperimentStore.preregistrationGeneratedMarker
                + " Duplicate the experiment to change anything.*\n\n"
                + "## Deviations\n\nNone so far.\n"
            try authored.write(
                to: experimentDirectory.appending(
                    component: ExperimentStore.preregistrationFilename),
                atomically: true, encoding: .utf8)
            let frozen = try ExperimentStore.freeze(
                name: "prereg-quoting", force: true)
            let preserved = try String(
                contentsOf: experimentDirectory.appending(
                    component: ExperimentStore.preregistrationFilename),
                encoding: .utf8)
            #expect(preserved == authored)  // footer quote and all
            #expect(
                FileManager.default.fileExists(
                    atPath: experimentDirectory.appending(
                        component: ExperimentStore.preregistrationFrozenSettingsFilename
                    ).path))
            #expect(
                frozen.preregistrationHash
                    == ExperimentStore.sha256Hex(Data(authored.utf8)))
        }
    }

    @Test func localFreezeStampsAndSnapshotsThePreservedPreregistration() throws {
        // Review 2026-08-29, P1: preserving the file froze NOTHING about it.
        // The freeze now stamps its sha256, snapshots it into pinned/ for the
        // no-git floor, and puts it on the pin surface — without disturbing
        // the freeze hash, because the stamps are outside the canonical
        // payload like frozenAt.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-stamped", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-stamped")
            let experimentDirectory = ExperimentStore.directory.appending(
                component: "prereg-stamped")
            let authored = "# Analysis preregistration\n\nOne estimator, chosen now.\n"
            try authored.write(
                to: experimentDirectory.appending(
                    component: ExperimentStore.preregistrationFilename),
                atomically: true, encoding: .utf8)
            let frozen = try ExperimentStore.freeze(
                name: "prereg-stamped", force: true)
            let digest = ExperimentStore.sha256Hex(Data(authored.utf8))
            #expect(frozen.preregistrationHash == digest)
            #expect(frozen.freezeHash == ExperimentStore.manifestHash(frozen))
            let snapshot = try String(
                contentsOf: experimentDirectory.appending(
                    components: "pinned", ExperimentStore.preregistrationFilename),
                encoding: .utf8)
            #expect(snapshot == authored)  // no-git reproducibility floor
            let labels = ExperimentStore.pinnedInputEntries(frozen).map(\.label)
            #expect(labels.contains("researcher-authored preregistration"))
            #expect(ExperimentStore.verify(frozen).isEmpty)
            _ = root
        }
    }

    @Test func editingAPreservedPreregistrationAfterFreezeFailsVerify() throws {
        // The whole point of the stamp: an authored preregistration is a
        // frozen artifact, so a post-freeze edit is drift like any other
        // pinned input.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-drift", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-drift")
            let prereg = ExperimentStore.directory.appending(
                components: "prereg-drift", ExperimentStore.preregistrationFilename)
            try "# Analysis preregistration\n\nWe predict a positive shift.\n"
                .write(to: prereg, atomically: true, encoding: .utf8)
            let frozen = try ExperimentStore.freeze(name: "prereg-drift", force: true)
            #expect(ExperimentStore.verify(frozen).isEmpty)

            try "# Analysis preregistration\n\nWe predicted the shift we got.\n"
                .write(to: prereg, atomically: true, encoding: .utf8)
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("researcher-authored preregistration")
                })
            try FileManager.default.removeItem(at: prereg)
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("researcher-authored preregistration")
                })
        }
    }

    @Test func aLegacyFrozenExperimentVerifiesWithoutThePreregistrationStamp() throws {
        // Frozen directories are immutable, so experiments frozen before the
        // stamp existed have none: absence is not a violation, and their
        // authored preregistration is free to sit there unhashed.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-legacy", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-legacy")
            var frozen = try ExperimentStore.freeze(name: "prereg-legacy", force: true)
            try "# Analysis preregistration\n\nWritten in 2026.\n".write(
                to: ExperimentStore.directory.appending(
                    components: "prereg-legacy",
                    ExperimentStore.preregistrationFilename),
                atomically: true, encoding: .utf8)
            frozen.preregistrationHash = nil
            frozen.preregistrationGeneratedHash = nil
            #expect(ExperimentStore.verify(frozen).isEmpty)
            #expect(
                !ExperimentStore.pinnedInputEntries(frozen).contains {
                    $0.label == "researcher-authored preregistration"
                })
        }
    }

    @Test func localFreezeRefreshesItsOwnStampedGeneratedPreregistration() throws {
        // Provenance-plus-hash: when the manifest carries the stamp of the
        // summary this instrument generated and the bytes still match it, the
        // file is provably ours and is refreshed rather than displaced — no
        // structural guessing required.
        try withAttachTempRoot { root in
            _ = try ExperimentStore.create(
                name: "prereg-own", description: "", modelID: "test/model")
            try plantStories("calm", root: root)
            _ = try ExperimentStore.attachConcept(
                "calm", method: .emotionGrandMean, experimentName: "prereg-own")
            let experimentDirectory = ExperimentStore.directory.appending(
                component: "prereg-own")
            var frozen = try ExperimentStore.freeze(name: "prereg-own", force: true)
            let generated = try Data(
                contentsOf: experimentDirectory.appending(
                    component: ExperimentStore.preregistrationFilename))
            #expect(
                frozen.preregistrationGeneratedHash
                    == ExperimentStore.sha256Hex(generated))
            // Re-open the study as a draft carrying its stamps (what copying
            // an experiment directory by hand produces) and freeze again: the
            // untouched generated file is recognized by its hash.
            frozen.status = .draft
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(frozen).write(
                to: ExperimentStore.directory.appending(
                    components: "prereg-own", "experiment.json"))
            let refrozen = try ExperimentStore.freeze(name: "prereg-own", force: true)
            #expect(refrozen.preregistrationHash == nil)
            #expect(
                !FileManager.default.fileExists(
                    atPath: experimentDirectory.appending(
                        component: ExperimentStore.preregistrationFrozenSettingsFilename
                    ).path))
            _ = root
        }
    }

    @Test func preregistrationClassificationRule() throws {
        // The classifier itself, at the boundaries the freeze tests cannot
        // isolate: a stored hash decides when there is one, structure decides
        // when there is not, and doubt always resolves to "preserve".
        try withAttachTempRoot { root in
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            let url = root.appending(component: "preregistration.md")
            var manifest = ExperimentManifest(
                name: "rule", description: "", modelID: "test/model")
            @discardableResult
            func writeFile(_ text: String) throws -> String {
                try text.write(to: url, atomically: true, encoding: .utf8)
                return ExperimentStore.sha256Hex(Data(text.utf8))
            }
            let footer = ExperimentStore.preregistrationGeneratedMarker
                + " Duplicate the experiment to change anything.*"

            // 1. A stamped generated hash wins over a file that looks nothing
            //    like ours.
            manifest.preregistrationGeneratedHash = try writeFile(
                "not remotely our shape\n")
            #expect(
                ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 2. A stamp that does NOT match the bytes preserves, even when
            //    the file is structurally perfect: something edited it.
            try writeFile("# Preregistration: s\n\nbody\n\n" + footer + "\n")
            manifest.preregistrationGeneratedHash = String(repeating: "0", count: 64)
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 3. The preserved-authored stamp names it as the researcher's.
            manifest.preregistrationGeneratedHash = nil
            manifest.preregistrationHash = try writeFile(
                "# Preregistration: s\n\nbody\n\n" + footer + "\n")
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 4. No stamps: structure decides. Header first, marker last.
            manifest.preregistrationHash = nil
            try writeFile("# Preregistration: s\n\nbody\n\n" + footer + "\n")
            #expect(
                ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 5. …the marker quoted mid-document is NOT ours.
            try writeFile("# Analysis preregistration\n\n" + footer + "\n\nmore\n")
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 6. …nor is one whose header is right but whose footer moved.
            try writeFile(
                "# Preregistration: s\n\n" + footer + "\n\ntrailing commitments\n")
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 7. …nor a footer-terminated file under someone else's heading.
            try writeFile("# Analysis preregistration\n\nbody\n\n" + footer + "\n")
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
            // 8. Unreadable-as-text bytes are authored — when in doubt,
            //    preserve.
            try Data([0xff, 0xfe, 0x20, 0x6e, 0x6f, 0x74]).write(to: url)
            #expect(
                !ExperimentStore.preregistrationIsGenerated(
                    at: url, manifest: manifest))
        }
    }

    @Test func exportPreregistrationNoopsWithoutExperimentDirectory() throws {
        // The server's legacy flat-file guard, twinned: no experiment
        // directory means nothing is written and none is created.
        try withAttachTempRoot { _ in
            let missing = ExperimentStore.directory.appending(
                component: "never-created")
            var manifest = ExperimentManifest(
                name: "never-created", description: "", modelID: "test/model")
            manifest.status = .frozen
            ExperimentStore.exportPreregistration(into: missing, &manifest)
            #expect(!FileManager.default.fileExists(atPath: missing.path))
            // Nothing written means nothing stamped: a stamp naming bytes
            // that do not exist would fail verify forever.
            #expect(manifest.preregistrationHash == nil)
            #expect(manifest.preregistrationGeneratedHash == nil)
        }
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
