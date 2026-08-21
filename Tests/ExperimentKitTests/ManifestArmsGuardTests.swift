import Foundation
import Testing

@testable import ExperimentKit

/// open-issues §8 — a manifest must not lose its whole measured surface in one
/// silent write.
///
/// The reported symptom (`s4x-kz-s1-noreasons`: a workspace manifest with
/// `concepts: []` + `conditions: []` beside a run snapshot carrying 16 of each)
/// turned out NOT to be a rewrite — that file was born a shell and never
/// touched again. But the writer census found the shape is reachable here, and
/// by a route with no equivalent on the server: `ExperimentStore.save` writes
/// the WHOLE document, and the app's `ExperimentPanel.selected` is an
/// in-memory cache with no file watcher, refreshed only on selection. A
/// manifest attached to from a terminal while the panel is open, then saved
/// from the panel, is written back as the panel last read it.
///
/// Python twin: `Server/tests/test_manifest_arms_guard.py`.
@Suite(.serialized) struct ManifestArmsGuardTests {

    func withTempRoot<T>(_ body: () throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "arms") { _ in
            try body()
        }
    }

    /// A draft holding both a concept and a condition — the state the study
    /// was in on the engine that ran it.
    @discardableResult
    func armedDraft(named name: String = "armed") throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        manifest.concepts.append(
            .init(name: "french", stimulusSetHash: "abc", options: .init()))
        manifest.conditions.append(
            .init(
                name: "french-mid",
                slots: [.init(concept: "french", layer: 14, alpha: 0.05)]))
        try ExperimentStore.save(manifest)
        return manifest
    }

    /// The same manifest with its arms gone — a stale panel copy, or a
    /// skeleton document.
    func shell(of manifest: ExperimentManifest) -> ExperimentManifest {
        var copy = manifest
        copy.concepts = []
        copy.conditions = []
        return copy
    }

    // MARK: - Reproduction

    /// THE reproduction: a whole-document save of a copy read before the arms
    /// existed. Before the guard this succeeded silently and the manifest on
    /// disk came back with `concepts: []` and `conditions: []`.
    @Test func aStaleWholeDocumentSaveCannotEmptyAPopulatedDraft() throws {
        try withTempRoot {
            // 1. The document a panel (or any long-lived holder) read.
            let stale = try ExperimentStore.create(
                name: "stale-cache", description: "", modelID: "test/model")
            #expect(stale.concepts.isEmpty && stale.conditions.isEmpty)

            // 2. Meanwhile, the study is attached to and declared elsewhere —
            //    a terminal `experiment attach` / `declare-condition`, the CLI
            //    the GUI is documented to call into.
            var live = try ExperimentStore.load(name: "stale-cache")
            live.concepts.append(
                .init(name: "french", stimulusSetHash: "abc", options: .init()))
            live.conditions.append(
                .init(
                    name: "french-mid",
                    slots: [.init(concept: "french", layer: 14, alpha: 0.05)]))
            try ExperimentStore.save(live)

            // 3. The holder saves what it read in step 1.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.save(stale)
            }
            let onDisk = try ExperimentStore.load(name: "stale-cache")
            #expect(onDisk.concepts.count == 1)
            #expect(onDisk.conditions.count == 1)
        }
    }

    @Test func theRefusalCarriesItsGateAndARunnableRepair() throws {
        try withTempRoot {
            let armed = try armedDraft()
            do {
                try ExperimentStore.save(shell(of: armed))
                Issue.record("expected the arms guard to refuse")
            } catch let error as ExperimentError {
                #expect(error.lifecycleRefusal?.gate == .armsCleared)
                #expect(error.lifecycleRefusal?.gateID == "armsCleared")
                // Names what is being lost, so a reader can tell a stale
                // document from an intended reset without opening the file.
                #expect(error.reason.contains("1 concept(s)"))
                #expect(error.reason.contains("1 condition(s)"))
                #expect(error.lifecycleRefusal?.repairAction.contains("steerlab-cli ") == true)
            }
        }
    }

    // MARK: - The guard triggers on the TRANSITION only

    @Test func aNewEmptyManifestIsAllowed() throws {
        try withTempRoot {
            // Creation is the legitimate both-empty write: no transition,
            // because there is nothing on disk to lose.
            var manifest = try ExperimentStore.create(
                name: "fresh", description: "", modelID: "test/model")
            #expect(manifest.concepts.isEmpty && manifest.conditions.isEmpty)
            // …and every setter that runs BEFORE the first attach — the
            // whole authoring warm-up — keeps working on a still-empty draft.
            manifest.temperature = 0.7
            try ExperimentStore.save(manifest)
            #expect(try ExperimentStore.load(name: "fresh").temperature == 0.7)
        }
    }

    @Test func attachBeforeAnyConditionIsAllowed() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "attaching", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: "abc", options: .init()))
            try ExperimentStore.save(manifest)
            #expect(try ExperimentStore.load(name: "attaching").conditions.isEmpty)
        }
    }

    @Test func losingOnlyTheConceptsIsNotTheGuardedTransition() throws {
        try withTempRoot {
            // The rule is BOTH-empty, deliberately. A study that drops its
            // concepts but keeps its conditions still has a measured surface,
            // and narrowing an arm set is ordinary authoring.
            let armed = try armedDraft()
            var partial = armed
            partial.concepts = []
            try ExperimentStore.save(partial)
            #expect(try ExperimentStore.load(name: "armed").conditions.count == 1)
        }
    }

    @Test func variantConditionsAreOutsideTheGuardedSurface() throws {
        try withTempRoot {
            // Documented scope: `variantConditions` is not part of the
            // concepts+conditions pair the guard watches, so an arm-less
            // variant study is unaffected in both directions.
            var manifest = try ExperimentStore.create(
                name: "variants-only", description: "", modelID: "test/model")
            // A forward-referenced variant arm (no artifact bytes) is the
            // cheapest legal VariantCondition to build here.
            manifest.variantConditions.append(
                try JSONDecoder().decode(
                    ExperimentManifest.VariantCondition.self,
                    from: Data(
                        #"{"name":"agent-a","fromPromotion":{"concept":"french"}}"#
                            .utf8)))
            try ExperimentStore.save(manifest)
            var cleared = try ExperimentStore.load(name: "variants-only")
            cleared.variantConditions.removeAll()
            try ExperimentStore.save(cleared)
            #expect(try ExperimentStore.load(name: "variants-only").variantConditions.isEmpty)
        }
    }

    @Test func aVariantOnlyIncomingDocumentIsNotADisarm() throws {
        try withTempRoot {
            // The guard's first false positive (caught at landing
            // 2026-08-20): a save whose whole measured surface lives in
            // variantConditions — the agentComparison shape — must not be
            // refused over a draft that held concepts/conditions. The
            // surface moved; it did not die.
            let armed = try armedDraft(named: "meta-variant")
            var incoming = shell(of: armed)
            incoming.variantConditions.append(
                try JSONDecoder().decode(
                    ExperimentManifest.VariantCondition.self,
                    from: Data(
                        #"{"name":"agent-a","fromPromotion":{"concept":"french"}}"#
                            .utf8)))
            try ExperimentStore.save(incoming)
            #expect(
                try !ExperimentStore.load(name: "meta-variant")
                    .variantConditions.isEmpty)
        }
    }

    // MARK: - Declared destructive intent

    @Test func detachingTheLastConceptDeclaresItsIntent() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "one-concept", description: "", modelID: "test/model")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: "abc", options: .init()))
            try ExperimentStore.save(manifest)
            let after = try ExperimentStore.detachConcept(
                "french", experimentName: "one-concept")
            #expect(after.concepts.isEmpty)
            #expect(try ExperimentStore.load(name: "one-concept").concepts.isEmpty)
        }
    }

    @Test func removingTheLastConditionDeclaresItsIntent() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "one-condition", description: "", modelID: "test/model")
            manifest.conditions.append(.init(name: "baseline", slots: []))
            try ExperimentStore.save(manifest)
            let after = try ExperimentStore.removeCondition(
                named: "baseline", experimentName: "one-condition")
            #expect(after.conditions.isEmpty)
            #expect(try ExperimentStore.load(name: "one-condition").conditions.isEmpty)
        }
    }

    @Test func theExplicitFlagIsTheOnlyWayThrough() throws {
        try withTempRoot {
            let armed = try armedDraft()
            try ExperimentStore.save(shell(of: armed), mayClearArms: true)
            let after = try ExperimentStore.load(name: "armed")
            #expect(after.concepts.isEmpty && after.conditions.isEmpty)
        }
    }

    // MARK: - The other immutability rules keep answering for themselves

    @Test func aFrozenManifestStillAnswersStatusImmutable() throws {
        try withTempRoot {
            var manifest = try armedDraft(named: "to-freeze")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            do {
                try ExperimentStore.save(shell(of: manifest))
                Issue.record("expected the frozen guard to refuse")
            } catch let error as ExperimentError {
                // An agent switching on `statusImmutable` must not start
                // seeing `armsCleared` instead: the frozen rule is checked
                // first and answers for itself.
                #expect(error.lifecycleRefusal?.gate == .statusImmutable)
            }
        }
    }

    @Test func theGateIsInTheClosedVocabularyAndClaimedByASite() {
        #expect(LifecycleGate.vocabulary.contains("armsCleared"))
        #expect(RefusalSiteRegistry.site(for: .armsCleared) != nil)
        #expect(!LifecycleGate.collidesWithFreezeVocabulary)
    }
}
