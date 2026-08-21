import Foundation
import Testing

@testable import ExperimentKit

/// The epoch guard must compare something that CAN match, and say what
/// changed when it doesn't.
///
/// Live failure, 2026-07-26: a cluster workspace's sweep run was refused with
/// "epoch stamps do not cross substrates — run promote on the engine that
/// produced the run". True, unactionable, and beside the point — the run
/// carries its own manifest snapshot, which this engine can hash itself. The
/// study HAD changed (a resolved `modelRevision` and an added pipeline stage),
/// but nothing in the message said so, so there was no way to tell a real edit
/// from an artifact of cross-engine hashing.
struct ManifestEpochDiagnosisTests {

    private func manifest(
        name: String = "s", revision: String? = nil
    ) -> ExperimentManifest {
        var m = ExperimentManifest(name: name, description: "", modelID: "org/m")
        m.modelRevision = revision
        return m
    }

    private func withRun<T>(
        substrate: String?, snapshot: ExperimentManifest?, stamp: String = "stale",
        _ body: (URL) throws -> T
    ) rethrows -> T {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "epoch-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var config: [String: Any] = ["runType": "sweep", "experimentHash": stamp]
        if let substrate { config["substrate"] = substrate }
        try? JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appending(component: "config.json"))
        if let snapshot {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try? encoder.encode(snapshot).write(
                to: directory.appending(component: "experiment.json"))
        }
        return try body(directory)
    }

    // MARK: The guard

    /// The core fix. A foreign stamp is no evidence at all; the snapshot is
    /// evidence, and here it proves the epochs agree.
    @Test func aForeignRunWhoseSnapshotMatchesIsAccepted() {
        let live = manifest()
        withRun(substrate: "python-hf-transformers", snapshot: live) { directory in
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live) == nil)
        }
    }

    @Test func aForeignRunWhoseSnapshotDiffersNamesTheChangedField() throws {
        let live = manifest(revision: String(repeating: "a", count: 40))
        let snapshot = manifest()
        withRun(substrate: "python-hf-transformers", snapshot: snapshot) { directory in
            let refusal = RunEpoch.refusal(
                verb: "promote", experiment: "s",
                liveHash: ExperimentStore.manifestHash(live),
                runDirectory: directory, liveManifest: live)
            let message = try! #require(refusal)
            #expect(message.contains("modelRevision"))
            #expect(message.contains(String(repeating: "a", count: 40)))
            #expect(!message.contains("do not cross substrates"))
        }
    }

    /// Without a snapshot there is genuinely nothing comparable, and the
    /// original diagnosis is the correct one.
    @Test func aForeignRunWithNoSnapshotStillRefusesOnTheSubstrate() throws {
        let live = manifest()
        withRun(substrate: "python-hf-transformers", snapshot: nil) { directory in
            let message = try! #require(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live))
            #expect(message.contains("no manifest snapshot"))
            #expect(message.contains("python-hf-transformers"))
        }
    }

    /// A SAME-engine mismatch is a genuine epoch difference and keeps
    /// refusing — the fix widens the diagnosis, never the gate.
    @Test func selfProjectedConditionsAreTheRunsOwnEpoch() throws {
        // Field incident 2026-08-04: the sweep projects <concept>-recommended
        // AFTER its start snapshot (and the Mac adopts the same projection
        // locally) — the guard then refused the state the run itself
        // produced. Self-projected conditions are stripped from BOTH sides;
        // a projection from any OTHER run still refuses. Server twin:
        // test_self_projected_conditions_are_the_runs_own_epoch.
        let base = manifest()
        var live = base
        try withRun(substrate: "python-hf-transformers", snapshot: base) { directory in
            let runName = directory.lastPathComponent
            live.conditions.append(
                .init(
                    name: "fear-recommended",
                    slots: [.init(concept: "fear", layer: 31, alpha: 0.2)],
                    bandWidth: 1, alphaInNormUnits: true,
                    selection: .init(
                        sweepRun: runName,
                        criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                        devPromptsHash: String(repeating: "d", count: 64),
                        winningCell: .init(layer: 31, alpha: 0.2),
                        metrics: [:])))
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live)
                    == nil)
            // The Studies panel stamps studyType on open — declared-as-
            // derived is the same study; a DIFFERENT type still refuses.
            live.studyType = "conceptStudy"
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live)
                    == nil)
            live.studyType = "agentComparison"
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live)
                    != nil)
            live.studyType = nil
            live.conditions.append(
                .init(
                    name: "anger-recommended",
                    slots: [.init(concept: "anger", layer: 37, alpha: 0.1)],
                    bandWidth: 1, alphaInNormUnits: true,
                    selection: .init(
                        sweepRun: "20260101T000000000-exp-s-sweep",
                        criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                        devPromptsHash: String(repeating: "d", count: 64),
                        winningCell: .init(layer: 37, alpha: 0.1),
                        metrics: [:])))
            let refusal = RunEpoch.refusal(
                verb: "promote", experiment: "s",
                liveHash: ExperimentStore.manifestHash(live),
                runDirectory: directory, liveManifest: live)
            #expect(refusal?.contains("anger-recommended") == true)
        }
    }

    @Test func aNativeRunWithAStaleStampStillRefuses() throws {
        let live = manifest(revision: String(repeating: "b", count: 40))
        withRun(substrate: nil, snapshot: manifest()) { directory in
            let message = try! #require(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live))
            #expect(message.contains("different manifest epoch"))
            #expect(message.contains("modelRevision"))
        }
    }

    @Test func aMatchingStampIsAcceptedWithoutConsultingTheSnapshot() {
        let live = manifest()
        let hash = ExperimentStore.manifestHash(live)
        withRun(substrate: "python-hf-transformers", snapshot: nil, stamp: hash) {
            directory in
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s", liveHash: hash,
                    runDirectory: directory, liveManifest: live) == nil)
        }
    }

    // MARK: The diff

    @Test func differencesAreDottedPathsWithBothValues() {
        let a = manifest()
        let b = manifest(revision: "abc")
        let diffs = ManifestDiff.differences(a, b)
        #expect(diffs.count == 1)
        #expect(diffs.first?.path == ".modelRevision")
        #expect(diffs.first?.left == "absent")
        #expect(diffs.first?.right == "abc")
    }

    @Test func identicalManifestsDifferNowhere() {
        #expect(ManifestDiff.differences(manifest(), manifest()).isEmpty)
        #expect(ManifestDiff.summary(manifest(), manifest()).isEmpty)
    }

    /// The diff must ignore exactly what `manifestHash` ignores. Naming a
    /// field the gate never checked would contradict its own verdict.
    @Test func volatileFreezeStampsAreNotReportedAsChanges() {
        var frozen = manifest()
        frozen.status = .frozen
        frozen.frozenAt = "2026-07-26T00:00:00Z"
        frozen.freezeHash = "abc"
        frozen.gitCommit = "def"
        frozen.frozenBy = "server"
        frozen.appVersion = "9.9"
        // `createdAt` is lifecycle too (2026-08-17): the hash used to treat
        // it as content while every sibling canonicalization ignored it, and
        // this test only caught the drift when two `manifest()` calls
        // straddled a second boundary. Set it explicitly so the contract is
        // pinned deterministically, not by timing.
        frozen.createdAt = "2001-01-01T00:00:00Z"
        #expect(ManifestDiff.differences(manifest(), frozen).isEmpty)
        #expect(
            ExperimentStore.manifestHash(manifest())
                == ExperimentStore.manifestHash(frozen),
            "the diff's canonicalization has drifted from the hash's")
    }

    /// A run stamped before `createdAt` left the hash canonicalization must
    /// stay eligible — the stamp is the legacy hash of the SAME manifest.
    @Test func aRunStampedUnderTheLegacyCanonicalizationStaysEligible() {
        let live = manifest()
        let legacy = ExperimentStore.legacyManifestHash(live)
        withRun(substrate: nil, snapshot: nil, stamp: legacy) { directory in
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live) == nil)
        }
    }

    @Test func theSummaryTruncatesAndSaysHowMany() {
        let a = ExperimentManifest(name: "s", description: "one", modelID: "org/m")
        var b = a
        b.modelID = "org/other"
        b.capabilityBatteryHash = "c"
        b.modelRevision = "r"
        b.neutralCorpusHash = "n"
        b.markersHash = "m"
        let summary = ManifestDiff.summary(a, b, limit: 2)
        #expect(summary.contains("; and "))
        #expect(summary.contains(" more"))
    }
}
