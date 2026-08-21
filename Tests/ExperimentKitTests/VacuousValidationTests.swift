import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Two circularity-firewall repairs found by the WP0 gate-5 probe
/// (2026-08-17).
///
/// **Vacuous validate evidence.** A `validate` run for a pinned concept with
/// no `validation.jsonl` — the DEFAULT state, since workspace seeding creates
/// none — scored no held-out probe, exited 0, and SATISFIED freeze's
/// `validateEvidence` gate: an unforced, unstamped freeze indistinguishable
/// from a validated one, while `data check` called the same missing file a
/// blocker. `validate` now stamps `vacuousConcepts` and the gate refuses it
/// under the same gate id.
///
/// **A study that measures nothing.** `run` on a concept study with zero
/// injection conditions produced a baseline-only run at exit 0 and `analyze`
/// then reported zero effect sizes at exit 0.
///
/// Declared as an extension of the serialized `ExperimentStoreTests` suite
/// because the store tests share its `rootOverride` test seam (a
/// process-global). Python twins:
/// `Server/tests/test_vacuous_validation_and_empty_study.py`.
extension ExperimentStoreTests {

    private func withVacuousTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "vacuous", body)
    }

    /// A grand-mean concept under the temp root's `prompts/emotions/<name>/`.
    /// Grand-mean (not paired) deliberately: paired concepts resolve their
    /// `validation.jsonl` against the REAL project root, so only the story
    /// family can be planted inside a test root.
    private func plantStoryConcept(
        _ name: String, root: URL, validation: String? = nil
    ) throws {
        let directory = root.appending(components: "prompts", "emotions", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try """
            {"concept": "\(name)", "text": "a story about \(name)"}
            {"concept": "\(name)", "text": "another story about \(name)"}

            """.write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
        if let validation {
            try validation.write(
                to: directory.appending(component: "validation.jsonl"),
                atomically: true, encoding: .utf8)
        }
    }

    /// Scope-matched validate evidence carrying an explicit vacuity verdict.
    /// `vacuousConcepts: nil` writes LEGACY evidence — no key at all.
    private func fabricateEvidence(
        for manifest: ExperimentManifest, vacuousConcepts: [String]?
    ) throws {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260817T000000000Z-exp-\(manifest.name)-validate")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        try #"{"experiment":"\#(manifest.name)","validation":{}}"#.write(
            to: dir.appending(component: "validation-report.json"),
            atomically: true, encoding: .utf8)
        if let vacuousConcepts {
            try ExperimentStore.writeValidationEvidence(
                for: manifest, runDirectory: dir,
                vacuousConcepts: vacuousConcepts)
        } else {
            let evidence: [String: Any] = [
                "schemaVersion": 1,
                "task": "validate",
                "substrate": ExperimentStore.evidenceSubstrate,
                "reportFile": "validation-report.json",
                "validationScopeHash": ExperimentStore.validationScopeHash(manifest),
            ]
            try JSONSerialization.data(
                withJSONObject: evidence, options: [.sortedKeys]
            ).write(to: dir.appending(component: "validation-evidence.json"))
        }
    }

    private func draftWithStoryConcept(
        named name: String, concept: String, root: URL, validation: String? = nil
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        try plantStoryConcept(concept, root: root, validation: validation)
        try ExperimentStore.attachGrandMeanConcepts([concept], into: &manifest)
        manifest.modelRevision = "deadbeef"
        try ExperimentStore.save(manifest)
        return manifest
    }

    // MARK: - The evidence stamp

    @Test func validationEvidenceAlwaysStampsAVacuityVerdict() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-stamp", concept: "stoicism", root: root)
            try fabricateEvidence(for: manifest, vacuousConcepts: ["stoicism"])
            #expect(
                ExperimentStore.vacuousValidationEvidence(for: manifest)
                    == ["stoicism"])
            // An empty verdict is a POSITIVE statement (every eligible
            // concept was probed), not an absent one.
            try fabricateEvidence(for: manifest, vacuousConcepts: [])
            #expect(ExperimentStore.vacuousValidationEvidence(for: manifest).isEmpty)
        }
    }

    @Test func legacyEvidenceWithoutTheStampIsNeverConvicted() throws {
        try withVacuousTempRoot { root in
            // Only NEWLY vacuous runs stop: evidence written before the stamp
            // existed carries no verdict, and freeze must keep accepting it.
            let manifest = try draftWithStoryConcept(
                named: "vac-legacy", concept: "stoicism", root: root)
            try fabricateEvidence(for: manifest, vacuousConcepts: nil)
            #expect(ExperimentStore.vacuousValidationEvidence(for: manifest).isEmpty)
            #expect(
                ExperimentStore.vacuousValidationEvidenceProblem(for: manifest)
                    == nil)
            let frozen = try ExperimentStore.freeze(name: "vac-legacy")
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced == nil)
        }
    }

    @Test func aStampNamingADroppedConceptSaysNothingAboutThisManifest() throws {
        try withVacuousTempRoot { root in
            // Evidence is matched by SCOPE, not by name.
            let manifest = try draftWithStoryConcept(
                named: "vac-dropped", concept: "stoicism", root: root)
            try fabricateEvidence(
                for: manifest, vacuousConcepts: ["some-other-concept"])
            #expect(ExperimentStore.vacuousValidationEvidence(for: manifest).isEmpty)
        }
    }

    // MARK: - The freeze gate

    @Test func freezeRefusesVacuousValidateEvidence() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-freeze", concept: "stoicism", root: root)
            try fabricateEvidence(for: manifest, vacuousConcepts: ["stoicism"])

            let problem = ExperimentStore.vacuousValidationEvidenceProblem(
                for: manifest)
            #expect(problem?.contains("VACUOUS evidence") == true)
            // The remedy names the exact missing file, not a category.
            #expect(
                problem?.contains("prompts/emotions/stoicism/validation.jsonl")
                    == true)

            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "vac-freeze")
            }
            #expect(try ExperimentStore.load(name: "vac-freeze").status == .draft)
            // Readiness cannot disagree with freeze about the same manifest.
            #expect(
                ExperimentStore.freezeReadiness(for: manifest).unmetGates
                    .contains { $0.contains("VACUOUS evidence") })
        }
    }

    @Test func forcedFreezeStampsTheVacuousEvidenceGate() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-forced", concept: "stoicism", root: root)
            try fabricateEvidence(for: manifest, vacuousConcepts: ["stoicism"])
            let frozen = try ExperimentStore.freeze(name: "vac-forced", force: true)
            #expect(frozen.freezeForced == true)
            #expect(frozen.forcedGatesSkipped == ["validateEvidence"])
        }
    }

    @Test func realValidateEvidenceFreezesUnchanged() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-real", concept: "stoicism", root: root,
                validation: #"{"text": "a held-out scene", "expresses": true}"# + "\n")
            try fabricateEvidence(for: manifest, vacuousConcepts: [])
            let frozen = try ExperimentStore.freeze(name: "vac-real")
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced == nil)
        }
    }

    /// Defense in depth: the gate reads a STAMP, so the second line is the
    /// pre-existing `validationHash` pin — deleting a validation set after a
    /// real validate run is drift, caught by the unskippable `verify()`.
    @Test func deletingTheValidationSetAfterValidateIsStillAPinViolation() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-deleted", concept: "stoicism", root: root,
                validation: #"{"text": "a held-out scene", "expresses": true}"# + "\n")
            #expect(ExperimentStore.verify(manifest).isEmpty)
            try FileManager.default.removeItem(
                at: root.appending(
                    components: "prompts", "emotions", "stoicism",
                    "validation.jsonl"))
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("validation.jsonl missing")
                })
        }
    }

    // MARK: - Probe eligibility and the path a refusal names

    @Test func heldOutProbePathFollowsTheEffectiveMethodsFamily() {
        let paired = ExperimentStore.makeConceptRef(
            name: "french", stimulusSetHash: "h",
            options: ExtractionOptions(method: .meanDifference))
        #expect(ExperimentStore.owesHeldOutProbe(paired))
        #expect(
            ExperimentStore.heldOutProbePath(paired)
                == "prompts/concepts/french/validation.jsonl")

        let grandMean = ExperimentManifest.ConceptRef(
            name: "fear", stimulusSetHash: "h",
            options: ExtractionOptions(method: .emotionGrandMean))
        #expect(ExperimentStore.owesHeldOutProbe(grandMean))
        #expect(
            ExperimentStore.heldOutProbePath(grandMean)
                == "prompts/emotions/fear/validation.jsonl")

        // An OptVec direction was never read off a concept's stimuli: asking
        // it for a validation.jsonl invents an obligation it cannot meet.
        let optvec = ExperimentManifest.ConceptRef(
            name: "opt", stimulusSetHash: "h",
            options: ExtractionOptions(method: .optvec))
        #expect(!ExperimentStore.owesHeldOutProbe(optvec))
    }

    // MARK: - A study that measures nothing

    @Test func aConceptStudyWithNoArmsRefusesToRun() {
        var manifest = ExperimentManifest(
            name: "nothing", description: "", modelID: "test/model")
        manifest.concepts = [
            ExperimentManifest.ConceptRef(
                name: "fear", stimulusSetHash: "h",
                options: ExtractionOptions(method: .meanDifference))
        ]
        let problem = ExperimentStore.noMeasuredConditionsProblem(manifest)
        #expect(problem?.contains("BASELINE") == true)
        // Both remedies are named: mint an arm, or declare baseline-only.
        #expect(problem?.contains("promote") == true)
        #expect(problem?.contains("agentComparison") == true)
        // The run path really would drop to baseline — which is why it must
        // refuse rather than proceed.
        #expect(
            ExperimentTasks.ordinaryRunConditions(for: manifest).map(\.name)
                == ["baseline"])

        // A canonical baseline alone is still nothing measured.
        var baselineOnly = manifest
        baselineOnly.conditions = [.init(name: "baseline", slots: [])]
        #expect(ExperimentStore.noMeasuredConditionsProblem(baselineOnly) != nil)

        // An injection condition is an arm.
        var withCondition = manifest
        withCondition.conditions = [
            .init(name: "fear-hi",
                  slots: [.init(concept: "fear", layer: 12, alpha: 1)])
        ]
        #expect(ExperimentStore.noMeasuredConditionsProblem(withCondition) == nil)

        // So is an agent.
        var withAgent = manifest
        withAgent.variantConditions = [
            .init(name: "a", artifactPath: "p", artifactHash: "h",
                  artifact: .init(
                      name: "a", baseModelID: "test/model",
                      promptMode: "chatAssistant",
                      qwenThinkingEnabled: false, temperature: 0,
                      systemPrompt: ""))
        ]
        #expect(ExperimentStore.noMeasuredConditionsProblem(withAgent) == nil)

        // So is a (server-executed) SAE latent arm.
        var withLatent = manifest
        withLatent.saeLatentConditions = .array([.object(["name": .string("l1")])])
        #expect(ExperimentStore.noMeasuredConditionsProblem(withLatent) == nil)

        // The sanctioned spelling of a DELIBERATE baseline-only run: the
        // declaration keeps the concept machinery inert, and the manifest
        // then says baseline-only instead of implying it.
        var declared = manifest
        declared.studyType = "agentComparison"
        #expect(ExperimentStore.noMeasuredConditionsProblem(declared) == nil)
        #expect(ExperimentStore.inertConditionsProblem(declared) == nil)

        // Nothing derived at all: no promise, no silence to break.
        var noConcepts = manifest
        noConcepts.concepts = []
        #expect(ExperimentStore.noMeasuredConditionsProblem(noConcepts) == nil)

        // A panel runs a scenario, not conditions.
        var panel = manifest
        panel.studyKind = .multiAgent
        #expect(ExperimentStore.noMeasuredConditionsProblem(panel) == nil)
    }

    /// WP0 step 2 split this message into problem + remedy so the
    /// `validateEvidence` gate can hand an agent the file-path remedy as a
    /// machine-readable `repairAction`. The split is byte-preserving: the
    /// message is asserted whole here against the literal this gate has
    /// thrown since the 2026-08-17 firewall repair, and the refusal carries
    /// the same paths in its repair.
    @Test func theVacuousRefusalProseSurvivesTheGateTableSplit() throws {
        try withVacuousTempRoot { root in
            let manifest = try draftWithStoryConcept(
                named: "vac-prose", concept: "stoicism", root: root)
            try fabricateEvidence(for: manifest, vacuousConcepts: ["stoicism"])
            let expected = "the matching validate run scored NO held-out probe for "
                + "concept(s) stoicism — it is VACUOUS evidence, not validation. "
                + "Author the never-named scenarios "
                + "(prompts/emotions/stoicism/validation.jsonl) as "
                + "{\"text\": …, \"expresses\": true|false} rows and re-run "
                + "'steerlab-cli experiment validate vac-prose', or freeze --force to "
                + "record an unvalidated experiment"
            #expect(
                ExperimentStore.vacuousValidationEvidenceProblem(for: manifest) == expected)
            do {
                _ = try ExperimentStore.freeze(name: "vac-prose")
                Issue.record("expected the validateEvidence gate to refuse")
            } catch let error as ExperimentError {
                #expect(error.reason == "cannot freeze 'vac-prose': \(expected)")
                let refusal = try #require(error.freezeRefusal)
                #expect(refusal.gate == .validateEvidence)
                #expect(
                    refusal.repairAction.hasPrefix(
                        "Author the never-named scenarios "
                            + "(prompts/emotions/stoicism/validation.jsonl)"))
            }
        }
    }

    @Test func analyzeWarnsWhenEveryRecordIsBaseline() {
        #expect(
            ExperimentTasks.baselineOnlyAnalysisWarning(
                runName: "r", conditions: ["baseline"])?
                .contains("only BASELINE records") == true)
        #expect(
            ExperimentTasks.baselineOnlyAnalysisWarning(
                runName: "r", conditions: ["baseline", "fear-hi"]) == nil)
        // No records at all is a different (existing) refusal, not this one.
        #expect(
            ExperimentTasks.baselineOnlyAnalysisWarning(
                runName: "r", conditions: []) == nil)
    }
}
