import Foundation
import Testing

@testable import ExperimentKit

/// App gap A4 — the native condition editor's store writes: upsert/remove,
/// the pure sign-control and matched-norm-random derivations, and the
/// Step-5 control-matrix scaffold (idempotent, never duplicating, honest
/// about what stays manual).
@Suite(.serialized) struct ConditionEditorTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "cond", body)
    }

    /// A draft with one attached concept ("fear" @ fake hash — the editor
    /// only checks attachment, not stimulus reality).
    @discardableResult
    private func makeDraft(_ name: String = "cond-a") throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(name: "fear", stimulusSetHash: "abc", options: .init()))
        try ExperimentStore.save(manifest)
        return manifest
    }

    private func treatment(
        name: String = "fear-mid", alpha: Double = 0.08
    ) -> ExperimentManifest.Condition {
        .init(
            name: name,
            slots: [.init(concept: "fear", layer: 18, alpha: alpha)],
            bandWidth: 1, alphaInNormUnits: true)
    }

    // MARK: upsert / remove

    @Test func upsertAddsReplacesAndRemoves() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.upsertCondition(treatment(), experimentName: "cond-a")
            var loaded = try ExperimentStore.load(name: "cond-a")
            #expect(loaded.conditions.map(\.name) == ["fear-mid"])
            // Same name replaces, never duplicates.
            try ExperimentStore.upsertCondition(
                treatment(alpha: 0.12), experimentName: "cond-a")
            loaded = try ExperimentStore.load(name: "cond-a")
            #expect(loaded.conditions.count == 1)
            #expect(loaded.conditions[0].slots[0].alpha == 0.12)
            try ExperimentStore.removeCondition(
                named: "fear-mid", experimentName: "cond-a")
            #expect(try ExperimentStore.load(name: "cond-a").conditions.isEmpty)
        }
    }

    @Test func upsertRefusesUnattachedConceptsAndEmptyNames() throws {
        try withTempRoot { _ in
            try makeDraft()
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.upsertCondition(
                    .init(
                        name: "ghost",
                        slots: [.init(concept: "not-attached", layer: 3, alpha: 0.1)]),
                    experimentName: "cond-a")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.upsertCondition(
                    .init(name: "  ", slots: []), experimentName: "cond-a")
            }
        }
    }

    // MARK: pure control derivations

    @Test func signControlNegatesEveryAlpha() {
        let control = ExperimentStore.signControlCondition(for: treatment())
        #expect(control.name == "fear-mid-neg")
        #expect(control.slots.map(\.alpha) == [-0.08])
        #expect(control.slots.map(\.layer) == [18])
        #expect(control.controlType == nil)
        #expect(control.selection == nil)
        #expect(control.alphaInNormUnits)
    }

    @Test func matchedNormRandomControlStampsControlType() {
        let control = ExperimentStore.matchedNormRandomCondition(for: treatment())
        #expect(control.name == "fear-mid-random")
        #expect(control.controlType == "randomMatchedNorm")
        // Same layers/alphas — the run loop substitutes the random direction.
        #expect(control.slots.map(\.alpha) == [0.08])
        #expect(control.selection == nil)
    }

    @Test func treatmentClassificationExcludesControls() {
        #expect(ExperimentStore.isTreatmentCondition(treatment()))
        #expect(
            !ExperimentStore.isTreatmentCondition(
                ExperimentStore.signControlCondition(for: treatment())))
        #expect(
            !ExperimentStore.isTreatmentCondition(
                ExperimentStore.matchedNormRandomCondition(for: treatment())))
        #expect(
            !ExperimentStore.isTreatmentCondition(.init(name: "baseline", slots: [])))
    }

    // MARK: control-matrix scaffold

    @Test func scaffoldBuildsTheStepFiveSetOnce() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.upsertCondition(treatment(), experimentName: "cond-a")

            let first = try ExperimentStore.scaffoldControlMatrix(
                experimentName: "cond-a")
            #expect(
                Set(first.added) == ["baseline", "fear-mid-neg", "fear-mid-random"])
            let names = try ExperimentStore.load(name: "cond-a").conditions.map(\.name)
            #expect(
                Set(names) == ["fear-mid", "baseline", "fear-mid-neg", "fear-mid-random"])
            // The scaffold is honest about the manual Step-5 pieces.
            #expect(first.notes.contains { $0.contains("sympathy") })
            #expect(first.notes.contains { $0.contains("battery") })

            // Idempotence: a second scaffold adds nothing and duplicates
            // nothing — everything reports as kept.
            let second = try ExperimentStore.scaffoldControlMatrix(
                experimentName: "cond-a")
            #expect(second.added.isEmpty)
            #expect(
                Set(second.skipped)
                    == ["baseline", "fear-mid-neg", "fear-mid-random"])
            #expect(
                try ExperimentStore.load(name: "cond-a").conditions.count == 4)
        }
    }

    @Test func scaffoldNeverTreatsScaffoldedControlsAsTreatments() throws {
        try withTempRoot { _ in
            try makeDraft()
            try ExperimentStore.upsertCondition(treatment(), experimentName: "cond-a")
            try ExperimentStore.scaffoldControlMatrix(experimentName: "cond-a")
            try ExperimentStore.scaffoldControlMatrix(experimentName: "cond-a")
            // No -neg-neg / -random-neg explosion.
            let names = try ExperimentStore.load(name: "cond-a").conditions.map(\.name)
            #expect(!names.contains { $0.contains("-neg-neg") })
            #expect(!names.contains { $0.contains("-random-neg") })
            #expect(!names.contains { $0.contains("-neg-random") })
        }
    }

    @Test func scaffoldWithoutTreatmentsRefusesLoudly() throws {
        try withTempRoot { _ in
            try makeDraft()
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.scaffoldControlMatrix(experimentName: "cond-a")
            }
        }
    }

    @Test func scaffoldSkipsBatteryNoteWhenBatteryPinned() throws {
        try withTempRoot { _ in
            var manifest = try makeDraft()
            manifest.capabilityBatteryFile = "prompts/batteries/basic.jsonl"
            manifest.capabilityBatteryHash = "abc"
            try ExperimentStore.save(manifest)
            try ExperimentStore.upsertCondition(treatment(), experimentName: "cond-a")
            let result = try ExperimentStore.scaffoldControlMatrix(
                experimentName: "cond-a")
            #expect(!result.notes.contains { $0.contains("battery") })
        }
    }

    @Test func scaffoldIsDraftOnly() throws {
        try withTempRoot { _ in
            try makeDraft("frozen-c")
            try ExperimentStore.upsertCondition(
                treatment(), experimentName: "frozen-c")
            var manifest = try ExperimentStore.load(name: "frozen-c")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.scaffoldControlMatrix(experimentName: "frozen-c")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.removeCondition(
                    named: "fear-mid", experimentName: "frozen-c")
            }
        }
    }
}
