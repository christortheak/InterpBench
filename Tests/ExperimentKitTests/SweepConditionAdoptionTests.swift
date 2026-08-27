import Foundation
import Testing

@testable import ExperimentKit

/// Projected-condition adoption (2026-08-04, the server-manifest-mutation
/// family): a server-executed sweep writes `<concept>-recommended` into ITS
/// manifest copy; discovery adopts the run's own recommendations into the
/// local draft — idempotently, and never over a conflicting local cell.
/// Declared as an extension of the serialized `ExperimentStoreTests` suite
/// (ambient-root override).
extension ExperimentStoreTests {

    /// Same shape as SweepPromotionTests' helper (file-private there): the
    /// workspace root is process-global, serialized by the shared lock.
    private func withAdoptionTempWorkspace<T>(
        _ body: (URL) throws -> T
    ) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "adopt-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    @Test func sweepConditionAdoptionProjectsRecommendationsIntoTheDraft() throws {
        try withAdoptionTempWorkspace { root in
            _ = try ExperimentStore.create(
                name: "ad", description: "", modelID: "test/model",
                modelRevision: "abc123")
            let directory = root.appending(
                components: "runs", "20260804T000000000-exp-ad-sweep")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try Data(#"{"name": "ad"}"#.utf8).write(
                to: directory.appending(component: "experiment.json"))
            try Data(
                (SweepRunCatalog.csvHeader
                    + "\nfear,31,0.2,0.0,0.9,0.91,50,false,0.9").utf8
            ).write(to: directory.appending(component: "sweep.csv"))
            let provenance = ExperimentManifest.SelectionProvenance(
                sweepRun: "20260804T000000000-exp-ad-sweep",
                criterion: try SweepSelectionRule.resolve(nil).asCriterion,
                devPromptsHash: String(repeating: "d", count: 64),
                winningCell: .init(layer: 31, alpha: 0.2),
                metrics: ["logprobShift": 1.18])
            let fragment = String(
                decoding: try JSONEncoder().encode(provenance), as: UTF8.self)
            try Data(
                ("{\"fear\": " + fragment
                    + ", \"courage\": \"no eligible cell\"}").utf8
            ).write(to: directory.appending(component: "recommendations.json"))

            let outcome = SweepConditionAdoption.adoptProjectedConditions(
                fromSweepRun: directory)
            #expect(
                outcome
                    == .adopted(experiment: "ad", added: 1, alreadyPresent: 0))
            let loaded = try ExperimentStore.load(name: "ad")
            #expect(loaded.conditions.map(\.name) == ["fear-recommended"])
            #expect(loaded.conditions.first?.slots.first?.layer == 31)
            #expect(loaded.conditions.first?.selection?.winningCell.alpha == 0.2)

            // Idempotent: a second discovery adopts nothing.
            #expect(
                SweepConditionAdoption.adoptProjectedConditions(
                    fromSweepRun: directory)
                    == .alreadyProjected(experiment: "ad"))

            // A conflicting local cell is loud and untouched.
            var mutated = loaded
            mutated.conditions[0].slots[0].alpha = 0.9
            try ExperimentStore.save(mutated)
            #expect(
                SweepConditionAdoption.adoptProjectedConditions(
                    fromSweepRun: directory)
                    == .conflict(
                        experiment: "ad", condition: "fear-recommended"))
            let untouched = try ExperimentStore.load(name: "ad")
            #expect(untouched.conditions.first?.slots.first?.alpha == 0.9)
        }
    }
}
