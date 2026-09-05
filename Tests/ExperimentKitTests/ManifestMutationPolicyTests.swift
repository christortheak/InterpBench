import Testing
@testable import ExperimentKit

/// Value-only policy tests: no selected workspace, files or global overrides.
@Suite struct ManifestMutationPolicyTests {
    @Test func creationRequiresExplicitIntentWhenNoRecordExists() throws {
        let draft = ExperimentManifest(name: "policy", description: "", modelID: "test/model")
        #expect(throws: ExperimentError.self) {
            try ManifestMutationPolicy.admitSave(draft, existing: nil)
        }
        try ManifestMutationPolicy.admitSave(draft, existing: nil, allowCreate: true)
    }

    @Test func frozenCompletionAllowsOnlyTheStatusChange() throws {
        var frozen = ExperimentManifest(name: "policy", description: "original", modelID: "test/model")
        frozen.status = .frozen
        var completed = frozen
        completed.status = .complete
        try ManifestMutationPolicy.admitSave(completed, existing: frozen)
        completed.experimentDescription = "changed"
        #expect(throws: ExperimentError.self) {
            try ManifestMutationPolicy.admitSave(completed, existing: frozen,
                                                allowCreate: true, mayClearArms: true)
        }
        #expect(throws: ExperimentError.self) {
            try ManifestMutationPolicy.admitSave(completed, existing: completed,
                                                allowCreate: true, mayClearArms: true)
        }
    }

    @Test func explicitArmClearingDoesNotOverrideFrozenImmutability() throws {
        var armed = ExperimentManifest(name: "policy", description: "", modelID: "test/model")
        armed.conditions = [.init(name: "arm", slots: [])]
        var cleared = armed
        cleared.conditions = []
        do {
            try ManifestMutationPolicy.admitSave(cleared, existing: armed)
            Issue.record("an undeclared removal of every arm must refuse")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .armsCleared)
        }
        try ManifestMutationPolicy.admitSave(cleared, existing: armed, mayClearArms: true)
        armed.status = .frozen
        do {
            try ManifestMutationPolicy.admitSave(cleared, existing: armed, mayClearArms: true)
            Issue.record("explicit arm clearing cannot override a frozen manifest")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .statusImmutable)
        }
    }
}
