import Foundation
import Testing

@testable import ExperimentKit

/// Model-revision adoption from imported validation evidence (external
/// review 2026-07-22): a revision-less local draft adopts the revision the
/// server resolved (carried home in the evidence run's manifest snapshot),
/// a conflicting local pin is flagged loudly, and a matching pin is a
/// silent no-op.
@Suite(.serialized) struct EvidenceRevisionAdoptionTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "rev-adopt", body)
    }

    /// Plants an imported-run directory carrying a server-shaped manifest
    /// snapshot (raw JSON — the server writes `manifest.raw`).
    private func plantImportedRun(
        name: String, modelID: String, revision: String?
    ) throws -> URL {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260722T000000000Z-exp-\(name)-validate-imported")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        var snapshot: [String: Any] = ["name": name, "modelID": modelID]
        if let revision { snapshot["modelRevision"] = revision }
        try JSONSerialization.data(withJSONObject: snapshot).write(
            to: dir.appending(component: "experiment.json"))
        return dir
    }

    @Test func nilLocalRevisionAdoptsAndSavesTheDraft() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "adopt", description: "", modelID: "test/model")
            let run = try plantImportedRun(
                name: "adopt", modelID: "test/model", revision: "abc123")
            let outcome = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: run)
            #expect(outcome == .adopted(experiment: "adopt", revision: "abc123"))
            // The draft is SAVED — freeze readiness sees the pin.
            let reloaded = try ExperimentStore.load(name: "adopt")
            #expect(reloaded.modelRevision == "abc123")
            // And the note is emitted, as information rather than warning.
            let notice = try #require(EvidenceRevisionAdoption.notice(for: outcome))
            #expect(!notice.isWarning)
            #expect(notice.message.contains("pinned model revision abc123"))
            #expect(notice.message.contains("validation evidence"))
        }
    }

    @Test func conflictingLocalPinFlagsLoudlyAndChangesNothing() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "clash", description: "", modelID: "test/model",
                modelRevision: "local-rev")
            let run = try plantImportedRun(
                name: "clash", modelID: "test/model", revision: "server-rev")
            let outcome = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: run)
            #expect(
                outcome
                    == .conflict(
                        experiment: "clash", localRevision: "local-rev",
                        evidenceRevision: "server-rev"))
            #expect(try ExperimentStore.load(name: "clash").modelRevision == "local-rev")
            let notice = try #require(EvidenceRevisionAdoption.notice(for: outcome))
            #expect(notice.isWarning)
            #expect(notice.message.contains("cannot satisfy freeze"))
            #expect(notice.message.contains("server-rev"))
            #expect(notice.message.contains("local-rev"))
        }
    }

    @Test func matchingPinIsASilentNoOp() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "match", description: "", modelID: "test/model",
                modelRevision: "same-rev")
            let run = try plantImportedRun(
                name: "match", modelID: "test/model", revision: "same-rev")
            let outcome = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: run)
            #expect(outcome == .alreadyPinned(experiment: "match", revision: "same-rev"))
            #expect(EvidenceRevisionAdoption.notice(for: outcome) == nil)
        }
    }

    @Test func modelMismatchFlagsInsteadOfAdopting() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "wrongmodel", description: "", modelID: "test/model")
            let run = try plantImportedRun(
                name: "wrongmodel", modelID: "other/model", revision: "abc")
            let outcome = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: run)
            #expect(
                outcome
                    == .modelMismatch(
                        experiment: "wrongmodel", localModel: "test/model",
                        evidenceModel: "other/model"))
            #expect(try ExperimentStore.load(name: "wrongmodel").modelRevision == nil)
            let notice = try #require(EvidenceRevisionAdoption.notice(for: outcome))
            #expect(notice.isWarning)
        }
    }

    @Test func revisionlessSnapshotAndUnknownExperimentStaySilent() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "quiet", description: "", modelID: "test/model")
            // Snapshot with no revision: nothing to adopt.
            let bare = try plantImportedRun(
                name: "quiet", modelID: "test/model", revision: nil)
            let noRevision = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: bare)
            #expect(noRevision == .noEvidenceRevision)
            #expect(EvidenceRevisionAdoption.notice(for: noRevision) == nil)
            // Snapshot naming an experiment this workspace does not have.
            let orphan = try plantImportedRun(
                name: "elsewhere", modelID: "test/model", revision: "abc")
            let noLocal = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: orphan)
            #expect(noLocal == .noLocalExperiment)
            #expect(EvidenceRevisionAdoption.notice(for: noLocal) == nil)
        }
    }
}
