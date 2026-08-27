import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `experiment detach` — the inverse of `attach`, and the writer the authoring
/// surface shipped without.
///
/// The gap this closes: every other pin on the authoring surface could be
/// replaced or cleared, and a CONCEPT pin could only ever be ADDED. A draft
/// therefore carried whatever it was first attached with — a mis-pinned
/// concept rode along to freeze as a passenger nothing cites, and re-pointing
/// one concept across a shelf of drafts was a mechanical edit no command
/// expressed.
///
/// The half that is not about convenience is the dependent audit. A detach
/// that quietly orphaned a declaration would leave a dangling reference only
/// the next `verify` names — and a run in between would have measured a study
/// nobody declared. That is the silent-drop class this engine refuses on
/// principle, so it is a typed refusal (`conceptInUse`) naming every dependent.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ConceptDetachVerbTests {

    // MARK: Harness

    /// A WORKSPACE override, not just an experiment-root override: the pins
    /// here are planted stimulus files, and `VectorCatalog.conceptsDirectory`
    /// resolves through `WorkspaceRoot`. Overriding only the narrower seam
    /// would read the CHECKOUT's committed concepts instead.
    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "detach-verb-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    /// Two paired stimulus files with fixed bytes, so a pin hash is stable.
    @discardableResult
    func plantConcept(_ name: String, root: URL) throws -> URL {
        let directory = root.appending(components: "prompts", "concepts", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try "{\"text\":\"a sentence about \(name)\"}\n"
            .write(
                to: directory.appending(component: "positive.jsonl"),
                atomically: true, encoding: .utf8)
        try "{\"text\":\"a matched sentence\"}\n"
            .write(
                to: directory.appending(component: "negative.jsonl"),
                atomically: true, encoding: .utf8)
        return directory
    }

    /// The manifest's bytes on disk — the comparison that proves detach
    /// touched exactly what it named.
    func manifestBytes(_ name: String) throws -> Data {
        try Data(contentsOf: ExperimentStore.manifestURL(name))
    }

    /// The encoded bytes of ONE concept pin, so an untouched entry can be
    /// compared byte for byte across a detach of its neighbour.
    func encodedPin(
        _ concept: String, in manifest: ExperimentManifest
    ) throws -> Data {
        let ref = try #require(manifest.concepts.first { $0.name == concept })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(ref)
    }

    func draft(
        _ name: String, concepts: [String], root: URL
    ) throws -> ExperimentManifest {
        _ = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        var manifest = try ExperimentStore.load(name: name)
        for concept in concepts {
            try plantConcept(concept, root: root)
            manifest = try ExperimentStore.attachConcept(
                concept, method: .meanDifference, experimentName: name)
        }
        return manifest
    }

    // MARK: - It removes exactly the named pin, and nothing else

    @Test func detachRemovesOnlyTheNamedPinAndLeavesTheRestByteIdentical() throws {
        try withTempRoot { root in
            let before = try draft(
                "d", concepts: ["alpha", "beta", "gamma"], root: root)
            let keptAlpha = try encodedPin("alpha", in: before)
            let keptGamma = try encodedPin("gamma", in: before)

            let after = try ExperimentStore.detachConcepts(
                ["beta"], experimentName: "d")

            #expect(after.concepts.map(\.name) == ["alpha", "gamma"])
            // The surviving entries are byte-identical: detach re-encodes the
            // whole document, so "it only removed one" is a claim about
            // BYTES, not about a name list.
            #expect(try encodedPin("alpha", in: after) == keptAlpha)
            #expect(try encodedPin("gamma", in: after) == keptGamma)
            // Every other pinned field survives untouched.
            #expect(after.modelID == before.modelID)
            #expect(after.neutralCorpusHash == before.neutralCorpusHash)
            #expect(after.status == .draft)
        }
    }

    @Test func detachTakesSeveralConceptsAtOnce() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha", "beta", "gamma"], root: root)
            let after = try ExperimentStore.detachConcepts(
                ["alpha", "gamma"], experimentName: "d")
            #expect(after.concepts.map(\.name) == ["beta"])
        }
    }

    /// Detaching the last pin of a condition-less draft is a legitimate
    /// arrival at both-empty — declared intent, not the stale-document shape
    /// the arms guard exists for (open-issues §8).
    @Test func detachingTheLastPinIsDeclaredIntentNotAnArmsClear() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            let after = try ExperimentStore.detachConcepts(
                ["alpha"], experimentName: "d")
            #expect(after.concepts.isEmpty)
        }
    }

    // MARK: - Refusals

    @Test func aFrozenManifestRefusesWithTheStatusRefusal() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            let bytes = try manifestBytes("d")

            #expect {
                try ExperimentStore.detachConcepts(["alpha"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .statusImmutable
                    && refusal.reason
                        == "experiment 'd' is frozen — duplicate it to iterate"
            }
            // Refused means NOTHING was written.
            #expect(try manifestBytes("d") == bytes)
        }
    }

    @Test func aConceptTheDraftDoesNotPinRefusesAndNamesWhatIsPinned() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha", "beta"], root: root)
            #expect {
                try ExperimentStore.detachConcepts(["delta"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .missingPrerequisite
                    && refusal.reason
                        == "concept 'delta' is not pinned to 'd' — pinned: "
                        + "alpha, beta"
                    && refusal.repairAction
                        == ExperimentStore.conceptNotPinnedRepair("d")
            }
        }
    }

    @Test func anInjectionConditionMakesDetachRefuseAndNamesIt() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            _ = try ExperimentStore.upsertCondition(
                .init(
                    name: "alpha-low",
                    slots: [.init(concept: "alpha", layer: 10, alpha: 0.1)]),
                experimentName: "d")
            let bytes = try manifestBytes("d")

            #expect {
                try ExperimentStore.detachConcepts(["alpha"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .conceptInUse
                    && refusal.reason
                        == "concept 'alpha' is still declared by condition "
                        + "'alpha-low' — remove or re-declare those conditions "
                        + "first"
                    && refusal.repairAction
                        == ExperimentStore.conceptInUseRepair("d")
            }
            #expect(try manifestBytes("d") == bytes)
        }
    }

    /// The reference the GUI-only predecessor did not audit: the sweep's
    /// per-concept choice instrument is keyed BY concept, so detaching the key
    /// leaves an instrument declaration pointing at nothing.
    @Test func aPerConceptSweepSelectionInstrumentMakesDetachRefuse() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.sweep = .init(
                selection: .init(
                    objective: .init(
                        metric: "logprobShift",
                        choicePromptsFiles: ["alpha": "prompts/dev/alpha.jsonl"])))
            try ExperimentStore.save(manifest)

            #expect {
                try ExperimentStore.detachConcepts(["alpha"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .conceptInUse
                    && refusal.reason.contains(
                        "sweep selection instrument "
                            + "'sweep.selection.objective.choicePromptsFiles[alpha]'")
            }
        }
    }

    /// A composite condition naming the concept: a variant condition's
    /// forward reference ("the agent this study's sweep promotes for alpha"),
    /// which `verify()` already flags when the concept is not attached.
    @Test func aVariantConditionForwardReferenceMakesDetachRefuse() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.variantConditions = [
                .init(
                    name: "alpha-agent", artifactPath: "", artifactHash: "",
                    artifact: .init(
                        name: "", baseModelID: "", promptMode: "",
                        qwenThinkingEnabled: false, temperature: 0,
                        systemPrompt: ""),
                    fromPromotion: .init(concept: "alpha")),
            ]
            try ExperimentStore.save(manifest)

            #expect {
                try ExperimentStore.detachConcepts(["alpha"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .conceptInUse
                    && refusal.reason.contains("variant condition 'alpha-agent'")
            }
        }
    }

    @Test func aPerturbationPolicyMakesDetachRefuse() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.perturbationPolicy = .init(
                sourceAgent: .init(
                    name: "alpha-agent", artifactPath: "runs/model-variants/a.json",
                    artifactHash: String(repeating: "0", count: 64),
                    promoted: true),
                concept: "alpha",
                cell: .init(layer: 10, alpha: 0.1),
                alphaDeltas: [0.02],
                includeMatchedNormControl: false,
                declaredAt: "2026-01-01T00:00:00Z")
            try ExperimentStore.save(manifest)

            #expect {
                try ExperimentStore.detachConcepts(["alpha"], experimentName: "d")
            } throws: { error in
                guard let refusal = (error as? ExperimentError)?.lifecycleRefusal
                else { return false }
                return refusal.gate == .conceptInUse
                    && refusal.reason.contains(
                        "perturbation policy 'perturbationPolicy'")
            }
        }
    }

    /// All-or-nothing: a two-concept detach whose SECOND name is refused must
    /// not have landed the first. A partial detach is exactly the silent
    /// half-edit the verb exists to make impossible.
    @Test func aRefusalOnTheSecondConceptWritesNothingAtAll() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha", "beta"], root: root)
            _ = try ExperimentStore.upsertCondition(
                .init(
                    name: "beta-low",
                    slots: [.init(concept: "beta", layer: 10, alpha: 0.1)]),
                experimentName: "d")
            let bytes = try manifestBytes("d")

            #expect(throws: ExperimentError.self) {
                try ExperimentStore.detachConcepts(
                    ["alpha", "beta"], experimentName: "d")
            }
            #expect(try manifestBytes("d") == bytes)
            #expect(
                try ExperimentStore.load(name: "d").concepts.map(\.name)
                    == ["alpha", "beta"])
        }
    }

    @Test func detachWithNoConceptNamedIsMalformedNotADisarm() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.detachConcepts([], experimentName: "d")
            }
            #expect(try ExperimentStore.load(name: "d").concepts.count == 1)
        }
    }

    // MARK: - The audited-but-not-gated references

    /// The audit's two footguns, pinned so a later reader does not "fix" them
    /// into gates: a designated reference and a pinned artifact's source
    /// concept name an on-disk DATA concept, which need never be attached, so
    /// detaching a PIN cannot dangle them.
    @Test func aDataConceptReferenceIsNotADependent() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.concepts[0].designatedReference = .init(
                name: "beta", hash: String(repeating: "1", count: 64))
            #expect(ExperimentStore.conceptDependents("beta", in: manifest).isEmpty)
        }
    }

    /// A discriminant control is refused if it IS a study concept, so it can
    /// never be a dependent of one.
    @Test func aValidationControlIsNotADependent() throws {
        try withTempRoot { root in
            _ = try draft("d", concepts: ["alpha"], root: root)
            var manifest = try ExperimentStore.load(name: "d")
            manifest.validationControls = [
                .init(
                    concept: "beta", stimulusSetHash: String(repeating: "2", count: 64),
                    options: .init(), modelRevision: nil),
            ]
            #expect(ExperimentStore.conceptDependents("beta", in: manifest).isEmpty)
        }
    }

    // MARK: - The refusal registry knows the verb

    @Test func bothNewRefusalSitesAreRegistered() {
        let inUse = RefusalSiteRegistry.site(for: .conceptInUse)
        #expect(inUse?.verbs == ["experiment detach"])
        #expect(
            RefusalSiteRegistry.site(for: .missingPrerequisite)?.verbs
                .contains("experiment detach") == true)
        #expect(
            RefusalSiteRegistry.site(for: .statusImmutable)?.verbs
                .contains("experiment detach") == true)
    }
}
