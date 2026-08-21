import Foundation
import Testing

@testable import ExperimentKit

/// Cross-engine `run-status.json` reader (retention 2026-07-24). The Python
/// twin's contract is fixed in `Server/tests/test_partial_evidence_retention.py`;
/// these assert the Swift side agrees about what counts as partial, because
/// the two engines now both write directories into the same `runs/` tree.
struct RunStatusFileTests {

    private func directory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-tests-runstatus",
                       "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ json: String, to directory: URL) throws {
        try Data(json.utf8).write(
            to: directory.appending(component: RunStatusFile.filename))
    }

    @Test func absentStatusIsNotPartial() throws {
        // Legacy runs carry no status file. They are unannotated, NOT
        // incomplete — reading them as partial would retroactively
        // invalidate every result produced before this contract existed.
        let url = try directory("legacy")
        #expect(RunStatusFile.read(at: url) == nil)
        #expect(!RunStatusFile.isPartial(at: url))
        #expect(RunStatusFile.partialSummary(at: url) == nil)
    }

    @Test func completedStatusIsNotPartial() throws {
        let url = try directory("done")
        try write(#"""
            {"schemaVersion": 1, "stage": "evaluate", "status": "completed",
             "evidenceComplete": true, "itemsWritten": 4, "itemLabel": "judgment",
             "pendingUnits": []}
            """#, to: url)
        #expect(!RunStatusFile.isPartial(at: url))
        #expect(RunStatusFile.partialSummary(at: url) == nil)
        #expect(RunStatusFile.read(at: url)?.itemsWritten == 4)
    }

    @Test func failedStatusIsPartialAndExplainsItself() throws {
        let url = try directory("failed")
        try write(#"""
            {"schemaVersion": 1, "stage": "evaluate", "status": "failed",
             "evidenceComplete": false, "itemsWritten": 2, "itemLabel": "judgment",
             "error": "judge j2 returned an invalid verdict twice",
             "errorType": "RuntimeError",
             "completedUnits": ["j1"], "pendingUnits": ["j2"]}
            """#, to: url)
        #expect(RunStatusFile.isPartial(at: url))
        let summary = try #require(RunStatusFile.partialSummary(at: url))
        #expect(summary.contains("evaluate did not complete"))
        #expect(summary.contains("invalid verdict twice"))
        #expect(summary.contains("missing: j2"))
    }

    @Test func inProgressStatusIsPartial() throws {
        // A process killed mid-stage (SIGKILL, node eviction) never gets to
        // write "failed". The work is no more complete for it.
        let url = try directory("killed")
        try write(#"""
            {"schemaVersion": 1, "stage": "run", "status": "inProgress",
             "evidenceComplete": false}
            """#, to: url)
        #expect(RunStatusFile.isPartial(at: url))
        #expect(RunStatusFile.partialSummary(at: url)?
            .contains("run did not complete") == true)
    }

    @Test func partialRunOutranksAFrozenStamp() throws {
        // The invariant that makes retention safe: a partial run of a
        // FROZEN, epoch-matched study is still a failure record. If the
        // frozen stamp were read first, this directory would be labelled
        // evidence-grade — which is exactly the claim a failure must not
        // be able to make.
        let url = try directory("frozen-but-partial")
        try write(#"""
            {"schemaVersion": 1, "stage": "run", "status": "failed",
             "evidenceComplete": false, "error": "node evicted"}
            """#, to: url)
        try Data(#"{"name": "s", "modelID": "org/m", "status": "frozen"}"#.utf8)
            .write(to: url.appending(component: "experiment.json"))
        try Data("abc123\n".utf8)
            .write(to: url.appending(component: "experiment-hash.txt"))

        let classification = RunResults.classification(
            runDirectory: url, records: [])
        #expect(classification.runClass == .partialFailedRun)
        #expect(!classification.isCitableResult)
        #expect(classification.label == "Incomplete — failure record")
        #expect(classification.detail.contains("failure record, not a result"))
    }

    @Test func completedRunStillClassifiesNormally() throws {
        // The status file must not disturb the existing rule for runs that
        // DID complete — retention adds a class, it does not reinterpret
        // the ones already there.
        let url = try directory("complete")
        try write(#"""
            {"schemaVersion": 1, "stage": "run", "status": "completed",
             "evidenceComplete": true}
            """#, to: url)
        let classification = RunResults.classification(
            runDirectory: url, records: [])
        #expect(classification.runClass == .draftPilot)
    }

    @Test func tornStatusFailsClosedAsPartial() throws {
        // FAIL-CLOSED (external review 2026-07-24, finding 4). This used to
        // read as "no status" → "legacy" → citable, which meant the crash
        // that made a run partial could also make it look complete. A torn
        // file is evidence a writer died — exactly what the marker exists
        // to record — so "I cannot tell" resolves to "not citable".
        let url = try directory("garbage")
        try write("{not js", to: url)
        #expect(RunStatusFile.reading(at: url) == .unreadable)
        #expect(RunStatusFile.isPartial(at: url))
        #expect(RunStatusFile.read(at: url) == nil)
        #expect(
            RunStatusFile.partialSummary(at: url)?.contains("unreadable")
                == true)
    }

    @Test func absentAndTornAreDistinguishable() throws {
        // The distinction the fail-open collapsed: no file at all is a
        // LEGACY run (trusted, governed by its completion artifacts);
        // a broken file is a SUSPECT run.
        let legacy = try directory("legacy-vs-torn-absent")
        #expect(RunStatusFile.reading(at: legacy) == .absent)
        #expect(!RunStatusFile.isPartial(at: legacy))

        let torn = try directory("legacy-vs-torn-present")
        try write("", to: torn)  // zero-byte: a classic torn write
        #expect(RunStatusFile.reading(at: torn) == .unreadable)
        #expect(RunStatusFile.isPartial(at: torn))
    }

    @Test func tornStatusOutranksAFrozenStampToo() throws {
        // The dominance rule has to hold for the fail-closed case as well,
        // or a torn status on a frozen study would read as evidence-grade.
        let url = try directory("frozen-but-torn")
        try write("{trunc", to: url)
        try Data(#"{"name": "s", "modelID": "org/m", "status": "frozen"}"#.utf8)
            .write(to: url.appending(component: "experiment.json"))
        let classification = RunResults.classification(
            runDirectory: url, records: [])
        #expect(classification.runClass == .partialFailedRun)
        #expect(!classification.isCitableResult)
    }
}
