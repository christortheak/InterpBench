import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Sweep-input hash pins (firewall closure, 2026-07-20): the sweep's dev
/// prompts and capability battery decide which cell wins, so they enter the
/// same drift firewall as markers — freeze pins them (cross-engine contract
/// keys `sweep.devPromptsHash` + `sweep.batteryHash`), verify() reports drift
/// as a violation, sweep start refuses to select on drifted inputs, and
/// legacy manifests without the keys keep their content hash byte-for-byte.
/// Second pass (same day): freeze REFUSES — force included — when an
/// operative sweep's input file is missing (an absent pin would leave sweep
/// start's legacy-unpinned fallback open to later-created bytes);
/// carried-inert sweeps neither pin nor block.
/// Declared as an extension of the serialized `ExperimentStoreTests` suite
/// because these tests share its `rootOverride` test seam.
extension ExperimentStoreTests {

    private func withSweepPinTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sweep-pins", body)
    }

    private func sweepSha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let devRows = Data(
        (#"{"text": "Write about the town."}"# + "\n").utf8)
    private static let batteryRows = Data(
        (#"{"prompt": "What is 1+1?", "answer": "2"}"# + "\n").utf8)

    /// A draft study with the committed french concept attached and a
    /// declared sweep whose input files live under the temp root.
    private func plantSweepStudy(
        name: String, root: URL
    ) throws -> (manifest: ExperimentManifest, devHash: String, batteryHash: String) {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        manifest.modelRevision = "deadbeef"
        let french = try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
        manifest.concepts.append(
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: french.hash, options: .init()))
        let devURL = root.appending(components: "prompts", "dev", "dev.jsonl")
        let batteryURL = root.appending(components: "prompts", "batteries", "b.jsonl")
        for url in [devURL, batteryURL] {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
        }
        try Self.devRows.write(to: devURL)
        try Self.batteryRows.write(to: batteryURL)
        manifest.sweep = ExperimentManifest.SweepSpec(
            devPromptsFile: "prompts/dev/dev.jsonl",
            batteryFile: "prompts/batteries/b.jsonl")
        try ExperimentStore.save(manifest)
        return (manifest, sweepSha256Hex(Self.devRows), sweepSha256Hex(Self.batteryRows))
    }

    // MARK: - Codable contract: legacy bytes unchanged

    @Test func sweepSpecPinKeysAreOmitWhenNil() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Legacy spec (no pins): the new keys must be ABSENT, so a
        // pre-existing manifest's content hash is byte-for-byte unchanged.
        let legacy = String(
            decoding: try encoder.encode(ExperimentManifest.SweepSpec()),
            as: UTF8.self)
        #expect(!legacy.contains("devPromptsHash"))
        #expect(!legacy.contains("batteryHash"))
        // Legacy JSON without the keys decodes to nil pins.
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.SweepSpec.self, from: Data(legacy.utf8))
        #expect(decoded.devPromptsHash == nil)
        #expect(decoded.batteryHash == nil)
        // Pinned specs round-trip both keys.
        var pinned = ExperimentManifest.SweepSpec()
        pinned.devPromptsHash = "aa"
        pinned.batteryHash = "bb"
        let roundTripped = try JSONDecoder().decode(
            ExperimentManifest.SweepSpec.self, from: try encoder.encode(pinned))
        #expect(roundTripped.devPromptsHash == "aa")
        #expect(roundTripped.batteryHash == "bb")
    }

    @Test func legacyManifestHashUnchangedByThePinFields() throws {
        try withSweepPinTempRoot { root in
            let (manifest, _, _) = try plantSweepStudy(
                name: "sweep-legacy", root: root)
            // A manifest whose sweep carries no pins verifies clean even
            // against drifted files (the legacy state exactly)…
            let before = ExperimentStore.manifestHash(manifest)
            try Data(#"{"text": "EDITED"}"#.utf8).write(
                to: root.appending(components: "prompts", "dev", "dev.jsonl"))
            #expect(ExperimentStore.verify(manifest).isEmpty)
            // …and its content hash is untouched by this feature.
            #expect(ExperimentStore.manifestHash(manifest) == before)
        }
    }

    // MARK: - Choice instruments are freeze pins too (review 2026-08-02, P1)

    private func plantChoiceInstruments(
        name: String, root: URL, mapped: Bool
    ) throws -> (manifest: ExperimentManifest, hashes: [String: String]) {
        var (manifest, _, _) = try plantSweepStudy(name: name, root: root)
        let dev = root.appending(components: "prompts", "dev")
        var hashes: [String: String] = [:]
        func write(_ file: String, id: String) throws -> String {
            let data = Data(
                (#"{"id": "\#(id)", "prompt": "p", "options": ["A", "B"]}"#
                    + "\n").utf8)
            try data.write(to: dev.appending(component: file))
            return sweepSha256Hex(data)
        }
        var objective = ExperimentManifest.SweepSelection.Objective(
            metric: "logprobShift")
        if mapped {
            // The map form with the one attached concept — coverage is the
            // rule, not a minimum count.
            hashes["french"] = try write("french-choices.jsonl", id: "f-1")
            objective.choicePromptsFiles = [
                "french": "prompts/dev/french-choices.jsonl"
            ]
        } else {
            hashes[""] = try write("choices.jsonl", id: "c-1")
            objective.choicePromptsFile = "prompts/dev/choices.jsonl"
        }
        manifest.sweep?.selection = .init(objective: objective)
        try ExperimentStore.save(manifest)
        return (manifest, hashes)
    }

    @Test func freezePinsChoiceInstrumentsAndVerifyFlagsDrift() throws {
        // The files that determine the WINNING CELL were the one sweep
        // input not pinned at freeze — a choice file could change between
        // freeze and execution with no manifest drift.
        try withSweepPinTempRoot { root in
            let (manifest, hashes) = try plantChoiceInstruments(
                name: "choice-pin", root: root, mapped: false)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "choice-pin")
            #expect(
                frozen.sweep?.selection?.objective?.choicePromptsHash
                    == hashes[""])
            #expect(ExperimentStore.verify(frozen).isEmpty)
            try Data(#"{"id": "c-1", "prompt": "EDITED", "options": ["A", "B"]}"#.utf8)
                .write(to: root.appending(
                    components: "prompts", "dev", "choices.jsonl"))
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("sweep choice prompts")
                        && $0.contains("changed since pinning")
                })
        }
    }

    @Test func freezePinsThePerConceptChoiceMap() throws {
        try withSweepPinTempRoot { root in
            let (manifest, hashes) = try plantChoiceInstruments(
                name: "choice-map", root: root, mapped: true)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "choice-map")
            #expect(
                frozen.sweep?.selection?.objective?.choicePromptsHashes
                    == ["french": hashes["french"]!])
            #expect(ExperimentStore.verify(frozen).isEmpty)
            try Data(#"{"id": "f-1", "prompt": "EDITED", "options": ["A", "B"]}"#.utf8)
                .write(to: root.appending(
                    components: "prompts", "dev", "french-choices.jsonl"))
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("sweep choice prompts 'french'")
                        && $0.contains("changed since pinning")
                })
        }
    }

    @Test func freezeRefusesWhenAChoiceInstrumentIsMissing() throws {
        try withSweepPinTempRoot { root in
            let (manifest, _) = try plantChoiceInstruments(
                name: "choice-missing", root: root, mapped: true)
            try fabricateValidationEvidence(for: manifest)
            try FileManager.default.removeItem(
                at: root.appending(
                    components: "prompts", "dev", "french-choices.jsonl"))
            do {
                _ = try ExperimentStore.freeze(name: "choice-missing")
                Issue.record("expected the missing-instrument refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("sweep choice prompts 'french'"))
                #expect(error.reason.contains("freeze cannot pin it"))
            }
        }
    }

    @Test func inertChoiceFieldsNeitherPinNorBlock() throws {
        // Review 2026-08-02 round 2 (P2): only logprobShift READS choice
        // instruments, so a stale path carried under markerDensity must
        // not block freezing (or packaging) over a file nothing reads.
        try withSweepPinTempRoot { root in
            var (manifest, _) = try plantChoiceInstruments(
                name: "choice-inert", root: root, mapped: false)
            manifest.sweep?.selection?.objective?.metric = "markerDensity"
            try ExperimentStore.save(manifest)
            try FileManager.default.removeItem(
                at: root.appending(components: "prompts", "dev", "choices.jsonl"))
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "choice-inert")
            #expect(
                frozen.sweep?.selection?.objective?.choicePromptsHash == nil)
            #expect(ExperimentStore.verify(frozen).isEmpty)
            #expect(
                !ExperimentStore.pinnedInputEntries(frozen)
                    .contains { $0.label.contains("choice prompts") })
        }
    }

    // MARK: - Freeze pins; drift is a violation

    @Test func freezePinsSweepInputsAndVerifyFlagsDrift() throws {
        try withSweepPinTempRoot { root in
            let (manifest, devHash, batteryHash) = try plantSweepStudy(
                name: "sweep-freeze", root: root)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "sweep-freeze")
            #expect(frozen.sweep?.devPromptsHash == devHash)
            #expect(frozen.sweep?.batteryHash == batteryHash)
            #expect(ExperimentStore.verify(frozen).isEmpty)

            // Post-freeze dev-prompt drift → verify violation.
            let devURL = root.appending(components: "prompts", "dev", "dev.jsonl")
            try Data(#"{"text": "EDITED"}"#.utf8).write(to: devURL)
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("sweep dev prompts 'prompts/dev/dev.jsonl' "
                        + "changed since pinning")
                })
            // Missing file → violation.
            try FileManager.default.removeItem(at: devURL)
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("sweep dev prompts") && $0.contains("file missing")
                })
            // Battery drift → violation.
            try Data(#"{"prompt": "EDITED", "answer": "x"}"#.utf8).write(
                to: root.appending(components: "prompts", "batteries", "b.jsonl"))
            #expect(
                ExperimentStore.verify(frozen).contains {
                    $0.contains("sweep capability battery "
                        + "'prompts/batteries/b.jsonl' changed since pinning")
                })
        }
    }

    @Test func freezeNeverSilentlyRepinsADriftedSweepInput() throws {
        try withSweepPinTempRoot { root in
            var (manifest, _, _) = try plantSweepStudy(
                name: "sweep-repin", root: root)
            manifest.sweep?.devPromptsHash = String(repeating: "0", count: 64)
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            // The pre-existing (drifted) pin must surface as a verify
            // violation at freeze — never be silently repaired.
            do {
                _ = try ExperimentStore.freeze(name: "sweep-repin")
                Issue.record("expected freeze to refuse the drifted sweep pin")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    "sweep dev prompts 'prompts/dev/dev.jsonl' changed since "
                        + "pinning"))
            }
        }
    }

    @Test func freezeRefusesWhenASweepInputFileIsMissing() throws {
        // Firewall closure second pass (2026-07-20): a missing sweep input
        // used to pin NOTHING at freeze, and the file could then be created
        // AFTER freeze — sweep start's legacy-unpinned fallback would accept
        // the unpinned bytes. Freeze now refuses instead.
        try withSweepPinTempRoot { root in
            let (manifest, _, _) = try plantSweepStudy(
                name: "sweep-nofile", root: root)
            try FileManager.default.removeItem(
                at: root.appending(components: "prompts", "batteries", "b.jsonl"))
            try fabricateValidationEvidence(for: manifest)
            do {
                _ = try ExperimentStore.freeze(name: "sweep-nofile")
                Issue.record("expected freeze to refuse the missing battery")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    "sweep capability battery file "
                        + "'prompts/batteries/b.jsonl' is missing"))
                #expect(error.reason.contains(
                    "create the file, or remove/repoint the sweep's "
                        + "batteryFile, before freezing"))
            }
            // Still a draft, still unpinned — nothing was half-frozen.
            let reloaded = try ExperimentStore.load(name: "sweep-nofile")
            #expect(reloaded.status == .draft)
            #expect(reloaded.sweep?.batteryHash == nil)
        }
    }

    @Test func missingSweepInputRefusalIsNotForceSkippable() throws {
        // Pin-surface integrity, not an evidence gate: `freeze --force`
        // skips evidence gates but never pin violations — a forced freeze
        // with an absent pin would leave the run-time legacy fallback open,
        // which no forcedGatesSkipped stamp can neutralize.
        try withSweepPinTempRoot { root in
            _ = try plantSweepStudy(name: "sweep-force", root: root)
            try FileManager.default.removeItem(
                at: root.appending(components: "prompts", "dev", "dev.jsonl"))
            do {
                _ = try ExperimentStore.freeze(name: "sweep-force", force: true)
                Issue.record("expected forced freeze to refuse missing dev prompts")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    "sweep dev prompts file 'prompts/dev/dev.jsonl' is "
                        + "missing"))
                #expect(error.reason.contains("devPromptsFile"))
            }
        }
    }

    @Test func carriedInertSweepNeitherPinsNorBlocksFreeze() throws {
        // Operative-surface carve-out: a declared agentComparison study
        // carries its sweep INERT — a missing sweep input must neither
        // refuse the freeze nor gain a pin (exactly the pre-existing rule
        // for inert machinery).
        try withSweepPinTempRoot { root in
            var (manifest, _, _) = try plantSweepStudy(
                name: "sweep-inert", root: root)
            manifest.studyType = "agentComparison"
            try ExperimentStore.save(manifest)
            try FileManager.default.removeItem(
                at: root.appending(components: "prompts", "batteries", "b.jsonl"))
            let frozen = try ExperimentStore.freeze(
                name: "sweep-inert", force: true)
            #expect(frozen.status == .frozen)
            #expect(frozen.sweep?.devPromptsHash == nil)
            #expect(frozen.sweep?.batteryHash == nil)
            #expect(ExperimentStore.verify(frozen).isEmpty)
        }
    }

    // MARK: - Sweep-start refusal (pure helper; the run loop calls it with
    // the just-loaded bytes, which is also what keeps the ex-post provenance
    // stamp and the manifest pin in agreement)

    @Test func sweepStartRefusesOnlyPinnedDrift() {
        // Unpinned (legacy) and matching pins pass.
        #expect(
            ExperimentTasks.sweepInputPinViolation(
                label: "sweep dev prompts", file: "prompts/dev/dev.jsonl",
                liveHash: "aa", pinned: nil) == nil)
        #expect(
            ExperimentTasks.sweepInputPinViolation(
                label: "sweep dev prompts", file: "prompts/dev/dev.jsonl",
                liveHash: "aa", pinned: "aa") == nil)
        // Drift refuses with a plain-language remedy.
        let problem = ExperimentTasks.sweepInputPinViolation(
            label: "sweep capability battery",
            file: "prompts/batteries/b.jsonl",
            liveHash: String(repeating: "1", count: 64),
            pinned: String(repeating: "0", count: 64))
        #expect(problem?.contains(
            "sweep capability battery 'prompts/batteries/b.jsonl' do not "
                + "match the manifest's pinned hash") == true)
        #expect(problem?.contains(
            "restore the pinned file, or duplicate the study and re-declare "
                + "the sweep") == true)
    }
}
