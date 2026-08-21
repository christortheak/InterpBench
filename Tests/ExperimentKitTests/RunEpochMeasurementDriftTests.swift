import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The epoch guard's measurement-side tolerance, and the strictness that
/// survives it (server twin: `Server/tests/test_run_epoch_substrate.py`).
///
/// The Swift guard had hardened on the server only, so the SAME run behaved
/// differently per engine: `evaluate`/`analyze`/`rescore-style` here refused a
/// source run whose manifest had drifted only in judges/evaluation/rubric/
/// human-labels — fields that cannot have touched a byte of the run's
/// generations — and demanded a full GPU re-run to swap a judge whose model
/// had died at its provider (2026-08-05). The rule now has ONE implementation
/// (`RunEpoch.check`) with one result type:
///
/// - measurement-only drift is ACCEPTED and STAMPED (`measurementDrift`);
/// - generation-side drift still REFUSES, on either substrate;
/// - `promote` never tolerates — a judge swap changes what a judged sweep's
///   evidence means;
/// - a live manifest missing a revision pin its own run snapshot carries is
///   the same epoch one write behind (bundle re-import), not drift.
struct RunEpochMeasurementDriftTests {

    private func manifest(
        name: String = "s", revision: String? = nil
    ) -> ExperimentManifest {
        var m = ExperimentManifest(name: name, description: "", modelID: "org/m")
        m.modelRevision = revision
        return m
    }

    /// A run directory stamped with `stamp`, carrying `snapshot` and
    /// (optionally) a foreign substrate.
    private func withRun<T>(
        substrate: String? = nil,
        snapshot: ExperimentManifest?,
        stamp: String? = "stale",
        _ body: (URL) throws -> T
    ) rethrows -> T {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "epoch-drift-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var config: [String: Any] = ["runType": "run"]
        if let stamp { config["experimentHash"] = stamp }
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

    private func check(
        live: ExperimentManifest, runDirectory: URL,
        verb: String = "evaluate", tolerate: Bool = true,
        allowUnverified: Bool = false
    ) -> RunEpoch.Check {
        RunEpoch.check(
            verb: verb, experiment: live.name,
            liveHash: ExperimentStore.manifestHash(live),
            runDirectory: runDirectory, liveManifest: live,
            allowUnverified: allowUnverified,
            tolerateMeasurementDrift: tolerate)
    }

    // MARK: - The tolerated field list

    /// Every field in the cross-engine list, one at a time. The list itself
    /// is the contract (`run_epoch.MEASUREMENT_FIELDS`), so a field that
    /// stops being tolerated here has silently diverged from the server.
    @Test(arguments: RunEpoch.measurementFields)
    func measurementOnlyDriftIsAcceptedAndStamped(field: String) throws {
        let snapshot = manifest()
        var live = snapshot
        switch field {
        case "judges":
            live.judges = [.init(name: "j1", kind: "local")]
        case "evaluation":
            live.evaluation = .init(kind: .pairedJudge, judgePrompt: "r")
        case "pipeline":
            live.pipeline = .object(["stages": .array([.string("evaluate")])])
        case "judgeRubricFile":
            live.judgeRubricFile = "prompts/rubrics/coding.md"
        case "judgeRubricHash":
            live.judgeRubricHash = String(repeating: "e", count: 64)
        case "humanValidation":
            live.humanValidation = .init(
                path: "prompts/human/labels.csv",
                hash: String(repeating: "f", count: 64))
        default:
            Issue.record("unhandled measurement field '\(field)'")
            return
        }
        try withRun(snapshot: snapshot) { directory in
            let accepted = check(live: live, runDirectory: directory)
            #expect(accepted.refusal == nil)
            #expect(accepted.unverified == false)
            let drift = try #require(accepted.measurementDrift)
            #expect(drift.contains(field))

            // …and promote — which never tolerates — still refuses it.
            let strict = RunEpoch.refusal(
                verb: "promote", experiment: live.name,
                liveHash: ExperimentStore.manifestHash(live),
                runDirectory: directory, liveManifest: live)
            #expect(strict?.contains(field) == true)
        }
    }

    /// The verb that does not opt in gets the old, strict behavior.
    @Test func aVerbThatDoesNotTolerateStillRefusesMeasurementDrift() {
        let snapshot = manifest()
        var live = snapshot
        live.judges = [.init(name: "j1", kind: "local")]
        withRun(snapshot: snapshot) { directory in
            #expect(
                check(live: live, runDirectory: directory, tolerate: false)
                    .refusal != nil)
        }
    }

    // MARK: - Generation-side drift is NOT tolerated

    @Test func generationSideDriftRefusesEvenForMeasurementVerbs() throws {
        let snapshot = manifest()
        var live = snapshot
        live.modelRevision = String(repeating: "a", count: 40)
        // A judge swap RIDES ALONG: one generation-side field is enough to
        // refuse the whole comparison — tolerance is all-or-nothing.
        live.judges = [.init(name: "j1", kind: "local")]
        try withRun(snapshot: snapshot) { directory in
            let refused = check(live: live, runDirectory: directory)
            let message = try #require(refused.refusal)
            #expect(message.contains("modelRevision"))
            #expect(refused.measurementDrift == nil)
        }
    }

    @Test func aForeignRunToleratesMeasurementDriftAndRefusesTheRest() throws {
        let snapshot = manifest()
        var live = snapshot
        live.judgeRubricFile = "prompts/rubrics/perResponse.md"
        try withRun(substrate: "python-hf-transformers", snapshot: snapshot) {
            directory in
            let accepted = check(live: live, runDirectory: directory)
            #expect(accepted.refusal == nil)
            #expect(accepted.measurementDrift?.contains("judgeRubricFile") == true)

            // Same run, a generation-side edit: refused, and the message
            // names the field rather than blaming the substrate.
            var edited = live
            edited.temperature = 0.7
            let refused = check(live: edited, runDirectory: directory)
            let message = try #require(refused.refusal)
            #expect(message.contains("temperature"))
            #expect(!message.contains("do not cross substrates"))
        }
    }

    // MARK: - The revision repair

    @Test func aLiveManifestMissingItsOwnRunsRevisionPinIsTheSameEpoch() {
        // A bundle re-import overwrites the live file with the bundle's
        // UNPINNED copy, so the run's snapshot carries a pin the live
        // manifest lost — the same manifest, one write behind its own
        // machinery. Server twin: `_with_snapshot_revision`.
        let snapshot = manifest(revision: String(repeating: "c", count: 40))
        let live = manifest()
        withRun(snapshot: snapshot) { directory in
            #expect(check(live: live, runDirectory: directory).refusal == nil)
            // Strict promote accepts it too: this is not tolerance, it is
            // the same epoch.
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: "s",
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live) == nil)
        }
    }

    @Test func aDifferentRevisionPinIsRealDriftAndIsNotRepaired() {
        let snapshot = manifest(revision: String(repeating: "c", count: 40))
        let live = manifest(revision: String(repeating: "d", count: 40))
        withRun(snapshot: snapshot) { directory in
            #expect(check(live: live, runDirectory: directory).refusal != nil)
        }
    }

    // MARK: - The unstamped-legacy axis is unchanged

    @Test func anUnstampedRunRefusesUnlessExplicitlyAllowed() throws {
        let live = manifest()
        try withRun(snapshot: live, stamp: nil) { directory in
            let refused = check(live: live, runDirectory: directory)
            let message = try #require(refused.refusal)
            #expect(message.contains("carries no experiment-hash stamp"))

            let accepted = check(
                live: live, runDirectory: directory, allowUnverified: true)
            #expect(accepted.refusal == nil)
            #expect(accepted.unverified)
            #expect(accepted.measurementDrift == nil)
        }
    }

    /// The flag NEVER bypasses a stamped mismatch — it is for legacy
    /// unstamped runs only.
    @Test func allowUnverifiedDoesNotExcuseAStampedMismatch() {
        let snapshot = manifest()
        var live = snapshot
        live.temperature = 0.9
        withRun(snapshot: snapshot) { directory in
            #expect(
                check(
                    live: live, runDirectory: directory, allowUnverified: true
                ).refusal != nil)
        }
    }

    // MARK: - The foreign-substrate refusal (WP0 dry run #2, P0)

    /// The snapshot rescue above answers "do the SETTINGS match?". For a
    /// measurement verb that is the wrong question: what fails is READING
    /// the records, whose schema and pairing keys are per-engine. Dry run
    /// #2 measured the consequence — Swift `analyze` on a server run exited
    /// 0, wrote a durable empty analysis, and advised that the run (two
    /// conditions, 24 records) "has no non-baseline condition".
    @Test func aMeasurementVerbRefusesAForeignRunEvenWhenTheSnapshotMatches()
        throws
    {
        let live = manifest()
        try withRun(
            substrate: "python-hf-transformers", snapshot: live, stamp: "server-side"
        ) { directory in
            for verb in ["analyze", "evaluate", "rescore-style"] {
                let refused = RunEpoch.check(
                    verb: verb, experiment: live.name,
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live,
                    tolerateMeasurementDrift: true, refuseForeignSubstrate: true)
                let message = try #require(refused.refusal)
                #expect(
                    message
                        == RunEpoch.foreignSubstrateRefusal(
                            verb: verb, runName: directory.lastPathComponent,
                            substrate: "python-hf-transformers"))
                #expect(
                    message.contains(
                        "Run \(verb) on the engine that produced the run"))
            }
            // promote is the OTHER family and is deliberately untouched: it
            // reads engine-neutral selection metadata, and a cluster
            // workspace's every sweep is foreign.
            #expect(
                RunEpoch.refusal(
                    verb: "promote", experiment: live.name,
                    liveHash: ExperimentStore.manifestHash(live),
                    runDirectory: directory, liveManifest: live) == nil)
        }
    }

    /// Byte-identical to the server's `run_epoch.foreign_substrate_refusal`.
    /// The two engines' agents read the same sentence, so it is pinned on
    /// both sides (server twin:
    /// `test_the_foreign_refusal_sentence_is_the_cross_engine_literal`).
    @Test func theForeignRefusalSentenceIsTheCrossEngineLiteral() {
        #expect(
            RunEpoch.foreignSubstrateRefusal(
                verb: "analyze", runName: "r", substrate: "swift-mlx")
                == "analyze: source run 'r' was produced on swift-mlx, and "
                + "analyze reads that run's own records — record schemas and "
                + "pairing keys are per-engine, so measuring them here reports "
                + "an empty result as a success. Run analyze on the engine "
                + "that produced the run")
    }

    /// `--allow-unverified-epoch` forgives a MISSING stamp, never a
    /// substrate this engine cannot read — the run stays just as unreadable.
    @Test func allowUnverifiedDoesNotExcuseAForeignSubstrate() throws {
        let live = manifest()
        try withRun(
            substrate: "python-hf-transformers", snapshot: live, stamp: nil
        ) { directory in
            let refused = RunEpoch.check(
                verb: "analyze", experiment: live.name,
                liveHash: ExperimentStore.manifestHash(live),
                runDirectory: directory, liveManifest: live,
                allowUnverified: true, tolerateMeasurementDrift: true,
                refuseForeignSubstrate: true)
            let message = try #require(refused.refusal)
            #expect(message.contains("was produced on python-hf-transformers"))
            #expect(!refused.unverified)
        }
    }

    /// A NATIVE run is untouched by the new flag — the guard must not have
    /// become stricter for the engine that wrote the run.
    @Test func aNativeRunIsUnaffectedByTheForeignRefusal() {
        let live = manifest()
        let hash = ExperimentStore.manifestHash(live)
        withRun(substrate: RepEReader.substrate, snapshot: live, stamp: hash) {
            directory in
            #expect(
                RunEpoch.check(
                    verb: "analyze", experiment: live.name, liveHash: hash,
                    runDirectory: directory, liveManifest: live,
                    tolerateMeasurementDrift: true, refuseForeignSubstrate: true)
                    == .eligible)
        }
    }

    // MARK: - A matching epoch stamps nothing

    @Test func aMatchingEpochIsCleanOnBothAxes() {
        let live = manifest()
        let hash = ExperimentStore.manifestHash(live)
        withRun(snapshot: live, stamp: hash) { directory in
            #expect(check(live: live, runDirectory: directory) == .eligible)
        }
    }
}
