import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A study that declares a J-lens readout never completes on the engine
/// built into this app as though the readout had been taken.
///
/// What happened before, observed by running one (release review 2026-10-04,
/// F1): a study with a declared readout verified clean, ran to completion on
/// this engine, wrote every response, and reported itself complete — with no
/// readout trace in the run and nothing to say one was owed. No run path on
/// this engine reads the declaration; the Python engine records it.
///
/// Now, matching the probe-measurement and SAE-latent-arm siblings:
///
/// * the RUN refuses at its start, before prompts are read or a model is
///   loaded, and says where the study can run;
/// * verify and freeze stay open — a study authored on a Mac and run on the
///   Python engine is legitimate;
/// * the researcher is told while still authoring, without being blocked:
///   in the readiness checklist and among the freeze advisories, and only
///   when the workspace runs studies on the engine built into this app.
///
/// Pure CPU by construction: the refusal is thrown before the GPU runtime
/// is touched or a model is looked for. Were it ever to regress, the
/// end-to-end test below names a task-prompt file that does not exist, so
/// the run stops there instead — still long before any model.
@Suite(.serialized) struct MacReadoutGapTests {

    private static let readout: JSONValue = .object([
        "lensID": .string("a-lens"),
        "lensSHA256": .string(String(repeating: "a", count: 64)),
        "configHash": .string(String(repeating: "b", count: 64)),
        "tokenizerHash": .string(String(repeating: "c", count: 64)),
        "qualificationID": .string("a-qualification"),
        "layers": .array([.number(10)]),
        "topK": .number(5),
    ])

    /// A model-output study with one steering arm, fully pinned for a
    /// readout: nothing about its pins is incomplete.
    private func study(readout: JSONValue? = MacReadoutGapTests.readout) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "readout-study", description: "", modelID: "test/model")
        manifest.modelRevision = "abc123"
        manifest.dtype = "float32"
        manifest.recordTokenIDs = true
        manifest.concepts = [
            .init(name: "formal", stimulusSetHash: "h", options: .init())
        ]
        manifest.conditions = [
            .init(name: "formal-up", slots: [.init(concept: "formal", layer: 10, alpha: 1)])
        ]
        manifest.jlensReadout = readout
        return manifest
    }

    private func refusal(_ manifest: ExperimentManifest) -> LifecycleRefusal? {
        do {
            try ExperimentTasks.refuseUnrecordableReadout(manifest)
            return nil
        } catch let error as ExperimentError {
            return error.lifecycleRefusal
        } catch {
            Issue.record("unexpected error type: \(error)")
            return nil
        }
    }

    // MARK: - The run refuses

    @Test func aDeclaredReadoutRefusesTheRunAndSaysWhereItCanRun() throws {
        let refused = try #require(refusal(study()))

        // Typed like its latent-arm sibling, so a caller reading the
        // envelope sees a refusal with a repair, not a crash.
        #expect(refused.gate == .inertConditions)
        // What is wrong, in plain words…
        #expect(refused.reason.contains("declares a J-lens readout"))
        #expect(refused.reason.contains("does not take that measurement"))
        #expect(refused.reason.contains("no model was loaded"))
        // …and how to get there: the workspace's compute choice, by the
        // names the app shows.
        #expect(refused.reason.contains(ComputeChoice.macFullCapabilities.title))
        #expect(refused.reason.contains(ComputeChoice.anotherMachine.title))
        #expect(refused.reason.contains("Workspace menu"))
        // The command-line route is the repair, and it is runnable.
        #expect(
            refused.repairAction.hasPrefix(
                "steerlab-cli remote package readout-study && steerlab-cli "
                    + "remote submit-bundle <bundle> --verb run"))
    }

    @Test func studiesThatDeclareNoReadoutAreUntouched() {
        #expect(refusal(study(readout: nil)) == nil)
        #expect(ExperimentStore.jlensReadoutNotExecutableProblem(study(readout: nil)) == nil)
    }

    /// A multi-agent study runs a scenario and never arms a readout. It may
    /// carry a block from before a kind switch, which is preserved and never
    /// executed — on either engine — so there is nothing to refuse.
    @Test func aMultiAgentStudyCarryingABlockIsNotRefused() {
        var panel = study()
        panel.studyKind = .multiAgent
        #expect(refusal(panel) == nil)
        #expect(ExperimentStore.jlensReadoutEngineAdvisory(panel) == nil)
    }

    /// The real run entry point, in a temporary workspace: the refusal comes
    /// first, and nothing is written.
    ///
    /// The run is pointed at a task-prompt file that does not exist. That is
    /// the safety net, not the subject: without the refusal the run's next
    /// step is to read its prompts, so a regression fails here on the missing
    /// file rather than going on to look for a model.
    @Test func theRunEntryPointRefusesBeforeAnythingIsReadOrWritten() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "readout-gap-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }

        // One plain agent as the arm: pinned by an artifact file this test
        // writes, so the draft verifies without any concept data.
        var manifest = try ExperimentStore.create(
            name: "readout-run", description: "", modelID: "test/model",
            modelRevision: "abc123")
        let artifact = ModelVariantArtifact(
            name: "agent", baseModelID: "test/model", adapters: [],
            injections: [], promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        let relativePath = "runs/model-variants/agent/model-variant.json"
        let url = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        manifest.variantConditions = [
            .init(
                name: "agent", artifactPath: relativePath,
                artifactHash: SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined(),
                artifact: artifact)
        ]
        manifest.dtype = "float32"
        manifest.recordTokenIDs = true
        manifest.jlensReadout = Self.readout
        try ExperimentStore.save(manifest)

        // Verify stays open: the declaration is legitimate, and this study
        // may yet run on the Python engine.
        #expect(ExperimentStore.verify(try ExperimentStore.load(name: "readout-run")) == [])
        let before = try FileManager.default.subpathsOfDirectory(atPath: root.path).sorted()

        var thrown: ExperimentError?
        do {
            _ = try await ExperimentTasks.run(
                experimentName: "readout-run",
                promptsFile: "prompts/tasks/no-such-prompts-\(UUID().uuidString).jsonl")
            Issue.record("a run with a declared readout was allowed to start")
        } catch let error as ExperimentError {
            thrown = error
        }

        let error = try #require(thrown)
        let refused = try #require(error.lifecycleRefusal)
        #expect(refused.gate == .inertConditions)
        #expect(refused.reason.contains("declares a J-lens readout"))
        // Nothing was written: no run directory, no model record, no change
        // to the study.
        let after = try FileManager.default.subpathsOfDirectory(atPath: root.path).sorted()
        #expect(after == before)
    }

    // MARK: - The researcher is told while authoring, and not blocked

    @Test func theAdvisoryAppliesOnlyWhereTheStudyWouldRunOnThisEngine() throws {
        let manifest = study()

        let advisory = try #require(
            ExperimentStore.jlensReadoutEngineAdvisory(
                manifest, runSubstrate: ExperimentStore.evidenceSubstrate))
        #expect(advisory.contains("declares a J-lens readout"))
        #expect(advisory.contains(ComputeChoice.macQuickStart.title))
        // It says the study is not blocked, and where to run it.
        #expect(advisory.contains("Designing and freezing the study are not affected"))
        #expect(advisory.contains(ComputeChoice.macFullCapabilities.title))
        #expect(advisory.contains(ComputeChoice.anotherMachine.title))

        // Headed for the Python engine: nothing to say.
        #expect(
            ExperimentStore.jlensReadoutEngineAdvisory(
                manifest, runSubstrate: WorkspaceScoping.serverSubstrate) == nil)
        #expect(
            ExperimentStore.jlensReadoutEngineAdvisory(study(readout: nil)) == nil)
    }

    @Test func freezeListsItAsAnAdvisoryAndNeverAsAGate() throws {
        let manifest = study()
        let advisory = try #require(ExperimentStore.jlensReadoutEngineAdvisory(manifest))

        #expect(
            ExperimentStore.freezeAdvisories(for: manifest).contains(advisory))
        #expect(
            !ExperimentStore.freezeAdvisories(
                for: manifest, runSubstrate: WorkspaceScoping.serverSubstrate
            ).contains { $0.contains("J-lens readout") })
        // Not a verification violation on any engine: verify stays open.
        #expect(ExperimentStore.jlensReadoutViolations(manifest) == [])
    }

    @Test func theChecklistRowSaysSoWithoutBecomingABlocker() throws {
        let manifest = study()
        let advisory = try #require(ExperimentStore.jlensReadoutEngineAdvisory(manifest))

        // Headed for the Python engine: the pins are complete, so the row
        // reads as present, exactly as before.
        let elsewhere = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest, runsOnBuiltInEngine: false)
        #expect(elsewhere == StudyDataReadiness.jlensReadoutRequirement(manifest: manifest))
        #expect(elsewhere.status == .present)

        // On the engine built into this app: said first, shown as partial
        // (never "present": the measurement will not be taken here), and
        // the pin grading is still there to read.
        let here = StudyDataReadiness.jlensReadoutRequirement(
            manifest: manifest, runsOnBuiltInEngine: true)
        #expect(here.status == .partial)
        #expect(here.detail.hasPrefix(advisory))
        #expect(here.detail.hasSuffix(elsewhere.detail))

        // An incomplete declaration keeps its own blocker and its own
        // sentence first; the advisory is added, not substituted.
        var unpinned = manifest
        unpinned.jlensReadout = .object(["lensID": .string("a-lens")])
        let incomplete = StudyDataReadiness.jlensReadoutRequirement(
            manifest: unpinned, runsOnBuiltInEngine: true)
        #expect(incomplete.status == .missing)
        #expect(incomplete.detail.hasPrefix("jlensReadout is missing"))
        #expect(incomplete.detail.hasSuffix(advisory))
    }

    /// The whole checklist reads the workspace's own compute choice.
    @Test func theChecklistFollowsTheWorkspacesComputeChoice() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "readout-checklist-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = study()

        func row() throws -> DataRequirement {
            try #require(
                StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                    .first { $0.id == "jlensReadout" })
        }

        try WorkspaceCompute.declare(ComputeChoice.macQuickStart, root: root)
        let quickStart = try row()
        #expect(quickStart.status == .partial)
        #expect(quickStart.detail.contains("does not take that measurement"))
        // Not a blocker: `data check` counts only missing and invalid rows.
        #expect(
            !StudyDataReadiness.summary([quickStart]).blockers.contains(quickStart))

        for choice in [ComputeChoice.macFullCapabilities, .anotherMachine] {
            try WorkspaceCompute.declare(choice, root: root)
            let python = try row()
            #expect(python.status == .present)
            #expect(!python.detail.contains("does not take that measurement"))
        }
    }
}
