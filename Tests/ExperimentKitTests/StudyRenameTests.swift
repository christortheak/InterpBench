import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Study naming: the DRAFT-only canonical rename (directory move + manifest
/// `name`) and the hash-exempt display label that works at every status.
///
/// The split is a firewall requirement, not ergonomics. `name` participates
/// in the manifest content hash on both engines, so rewriting it on a frozen
/// study would re-epoch it: every completed run's stamped experiment hash
/// would mismatch and `evaluate`/`analyze` would refuse the study's own
/// runs. The label lives in a sidecar precisely so it cannot do that.
@Suite(.serialized) struct StudyRenameTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "rename", body)
    }

    @discardableResult
    private func makeDraft(_ name: String) throws -> ExperimentManifest {
        try ExperimentStore.create(name: name, description: "d", modelID: "test/model")
    }

    /// Flip a draft to a terminal status on disk (save allows draft edits).
    private func setStatus(
        _ status: ExperimentManifest.Status, _ name: String
    ) throws {
        var manifest = try ExperimentStore.load(name: name)
        manifest.status = status
        try ExperimentStore.save(manifest)
    }

    /// Plants a run directory stamped with this experiment name — the same
    /// association `StudyResultStore.list` reads.
    private func fabricateRun(for manifest: ExperimentManifest, stamp: String) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "\(stamp)-exp-\(manifest.name)-run")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
    }

    // MARK: - Canonical rename (drafts only)

    @Test func renameMovesTheWholeDirectoryAndRewritesTheManifestName() throws {
        try withTempRoot { _ in
            try makeDraft("vg-study-arma-2-2-2")
            // Anything already inside the study directory travels with it —
            // `pinned/` snapshots are the case that matters at freeze.
            let pinned = ExperimentStore.directory.appending(
                components: "vg-study-arma-2-2-2", "pinned")
            try FileManager.default.createDirectory(
                at: pinned, withIntermediateDirectories: true)
            try "stimuli".write(
                to: pinned.appending(component: "snapshot.txt"),
                atomically: true, encoding: .utf8)

            let outcome = try ExperimentStore.rename(
                experimentName: "vg-study-arma-2-2-2", to: "Vignette Replication")

            #expect(outcome.newName == "vignette-replication")  // sanitized
            #expect(outcome.runsKeepingOldName == 0)
            #expect(outcome.runsNote == nil)
            #expect(try ExperimentStore.load(name: "vignette-replication").name
                == "vignette-replication")
            #expect(throws: (any Error).self) {
                try ExperimentStore.load(name: "vg-study-arma-2-2-2")
            }
            #expect(ExperimentStore.list().map(\.name) == ["vignette-replication"])
            // The old directory is gone; the pinned snapshot came along.
            #expect(
                !FileManager.default.fileExists(
                    atPath: ExperimentStore.directory
                        .appending(component: "vg-study-arma-2-2-2").path))
            #expect(
                FileManager.default.fileExists(
                    atPath: ExperimentStore.directory
                        .appending(components: "vignette-replication", "pinned", "snapshot.txt")
                        .path))
        }
    }

    @Test func renameRefusesFrozenAndCompletedStudies() throws {
        try withTempRoot { _ in
            for (name, status) in [
                ("frozen-study", ExperimentManifest.Status.frozen),
                ("done-study", .complete),
            ] {
                try makeDraft(name)
                try setStatus(status, name)
                do {
                    try ExperimentStore.rename(experimentName: name, to: "renamed-\(name)")
                    Issue.record("expected the \(status.rawValue) refusal")
                } catch let error as ExperimentError {
                    #expect(error.reason.contains(status.rawValue))
                    // The refusal names the way forward, not just the wall.
                    #expect(error.reason.contains("display label"))
                }
                // Untouched: still listed under its own name.
                #expect(try ExperimentStore.load(name: name).name == name)
                #expect(throws: (any Error).self) {
                    try ExperimentStore.load(name: "renamed-\(name)")
                }
            }
        }
    }

    @Test func renameRefusesNamesWithNoSubstanceAndNeverAcceptsAPath() throws {
        try withTempRoot { root in
            try makeDraft("subject")
            for candidate in ["", "   ", ".", "..", "///", "../.."] {
                #expect(throws: ExperimentError.self) {
                    try ExperimentStore.rename(experimentName: "subject", to: candidate)
                }
            }
            #expect(try ExperimentStore.load(name: "subject").name == "subject")

            // A path-shaped name is not a path: sanitizing strips separators
            // and dots, so the result can only ever be one directory
            // component directly under experiments/.
            let outcome = try ExperimentStore.rename(
                experimentName: "subject", to: "../../escaped")
            #expect(outcome.newName == "escaped")
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(components: "experiments", "escaped").path))
        }
    }

    @Test func renameRefusesANameAlreadyInUse() throws {
        try withTempRoot { _ in
            try makeDraft("first")
            try makeDraft("second")
            do {
                try ExperimentStore.rename(experimentName: "first", to: "second")
                Issue.record("expected the collision refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("already exists"))
            }
            // Both survive, unmoved.
            #expect(try ExperimentStore.load(name: "first").name == "first")
            #expect(try ExperimentStore.load(name: "second").name == "second")
        }
    }

    @Test func renameToTheSameNameIsANoOp() throws {
        try withTempRoot { _ in
            try makeDraft("stable")
            let outcome = try ExperimentStore.rename(
                experimentName: "stable", to: "  Stable  ")
            #expect(outcome.newName == "stable")
            #expect(outcome.oldName == "stable")
            #expect(try ExperimentStore.load(name: "stable").name == "stable")
        }
    }

    @Test func renameReportsRunsThatKeepTheOldNameAndNeverRewritesThem() throws {
        try withTempRoot { _ in
            let draft = try makeDraft("has-runs")
            try fabricateRun(for: draft, stamp: "20260610T000000000Z")
            try fabricateRun(for: draft, stamp: "20260611T000000000Z")
            #expect(ExperimentStore.runsStamped(experimentName: "has-runs") == 2)

            let outcome = try ExperimentStore.rename(
                experimentName: "has-runs", to: "has-runs-renamed")
            #expect(outcome.runsKeepingOldName == 2)
            #expect(outcome.runsNote?.contains("immutable") == true)
            #expect(outcome.runsNote?.contains("has-runs") == true)

            // Runs are immutable: their snapshots still stamp the OLD name,
            // which is exactly why they stop listing under the new one.
            #expect(ExperimentStore.runsStamped(experimentName: "has-runs") == 2)
            #expect(ExperimentStore.runsStamped(experimentName: "has-runs-renamed") == 0)
            #expect(StudyResultStore.list(experimentName: "has-runs-renamed").isEmpty)
            #expect(StudyResultStore.list(experimentName: "has-runs").count == 2)
        }
    }

    // MARK: - Display label (every status)

    @Test func displayLabelRoundTripsSetsClearsAndSurvivesARename() throws {
        try withTempRoot { _ in
            let draft = try makeDraft("labelled")
            // Absent sidecar = no label; the display name falls back to the
            // canonical one.
            #expect(ExperimentStore.displayLabel(name: "labelled") == nil)
            #expect(ExperimentStore.displayName(draft) == "labelled")

            try ExperimentStore.setDisplayLabel(
                "Vignette replication", experimentName: "labelled")
            #expect(ExperimentStore.displayLabel(name: "labelled")
                == "Vignette replication")
            #expect(
                ExperimentStore.displayLabels(ExperimentStore.list())
                    == ["labelled": "Vignette replication"])

            // The sidecar lives INSIDE the study directory, so a rename
            // carries it.
            let outcome = try ExperimentStore.rename(
                experimentName: "labelled", to: "vg-replication")
            #expect(ExperimentStore.displayLabel(name: outcome.newName)
                == "Vignette replication")
            #expect(ExperimentStore.displayLabel(name: "labelled") == nil)

            // A label renders in one line: newlines collapse to spaces.
            try ExperimentStore.setDisplayLabel(
                "  first line \n\n  second line  ", experimentName: "vg-replication")
            #expect(ExperimentStore.displayLabel(name: "vg-replication")
                == "first line second line")

            // Empty clears it, sidecar and all.
            try ExperimentStore.setDisplayLabel("  ", experimentName: "vg-replication")
            #expect(ExperimentStore.displayLabel(name: "vg-replication") == nil)
            #expect(
                !FileManager.default.fileExists(
                    atPath: ExperimentStore.displayLabelURL(name: "vg-replication").path))
            // Clearing an already-absent label is not an error.
            try ExperimentStore.setDisplayLabel(nil, experimentName: "vg-replication")
        }
    }

    @Test func labellingAFrozenStudyLeavesItsManifestBytesAndContentHashUntouched()
        throws
    {
        try withTempRoot { _ in
            try makeDraft("frozen-label")
            try setStatus(.frozen, "frozen-label")
            let manifestURL = ExperimentStore.manifestURL("frozen-label")
            let before = try Data(contentsOf: manifestURL)
            let hashBefore = ExperimentStore.manifestHash(
                try ExperimentStore.load(name: "frozen-label"))

            try ExperimentStore.setDisplayLabel(
                "the one we cite", experimentName: "frozen-label")

            #expect(try Data(contentsOf: manifestURL) == before)
            #expect(
                ExperimentStore.manifestHash(try ExperimentStore.load(name: "frozen-label"))
                    == hashBefore)
            #expect(ExperimentStore.displayLabel(name: "frozen-label") == "the one we cite")
            // The identity the runs stamp is unchanged.
            #expect(try ExperimentStore.load(name: "frozen-label").name == "frozen-label")
        }
    }

    @Test func theLabelSidecarIsNotOnThePinSurface() throws {
        try withTempRoot { _ in
            let draft = try makeDraft("pin-surface")
            try ExperimentStore.setDisplayLabel("a label", experimentName: "pin-surface")
            let manifest = try ExperimentStore.load(name: draft.name)
            let paths = ExperimentStore.pinnedInputEntries(manifest).map(\.url.path)
            #expect(
                !paths.contains(ExperimentStore.displayLabelURL(name: "pin-surface").path))
            // Nor does it come along inside the pinned/ snapshot entry.
            #expect(!paths.contains { $0.hasSuffix(ExperimentStore.displayLabelFileName) })
        }
    }

    @Test func labellingSomethingThatIsNotAStudyRefuses() throws {
        try withTempRoot { root in
            #expect(throws: (any Error).self) {
                try ExperimentStore.setDisplayLabel("x", experimentName: "no-such-study")
            }
            #expect(
                !FileManager.default.fileExists(
                    atPath: root.appending(components: "experiments", "no-such-study").path))
        }
    }

    // MARK: - One-click New Study

    @Test func placeholderStudyNameIsReadableAndSuffixesOnCollision() throws {
        try withTempRoot { _ in
            var components = DateComponents()
            components.year = 2026
            components.month = 8
            components.day = 6
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = TimeZone.current
            let date = try #require(calendar.date(from: components))

            #expect(ExperimentStore.placeholderStudyName(date: date)
                == "new-study-2026-08-06")
            try makeDraft("new-study-2026-08-06")
            #expect(ExperimentStore.placeholderStudyName(date: date)
                == "new-study-2026-08-06-2")
            try makeDraft("new-study-2026-08-06-2")
            #expect(ExperimentStore.placeholderStudyName(date: date)
                == "new-study-2026-08-06-3")
        }
    }

    @Test func aDuplicatedDraftIsImmediatelyRenamable() throws {
        try withTempRoot { _ in
            try makeDraft("source")
            // The duplicate flow's auto-suffix stays as it is; what matters
            // is that its output is a draft, so rename accepts it.
            let copy = try ExperimentStore.duplicate(name: "source", as: "source-2")
            #expect(copy.status == .draft)
            let outcome = try ExperimentStore.rename(
                experimentName: "source-2", to: "sympathy-screen")
            #expect(outcome.newName == "sympathy-screen")
            #expect(try ExperimentStore.load(name: "sympathy-screen").name
                == "sympathy-screen")
        }
    }
}
