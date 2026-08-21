import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Effective-evaluation declaration (2026-07-22 incident). A researcher's
/// FROZEN study carried pinned judges + a pinned rubric file, `pipeline:
/// {stages: [run, evaluate, analyze]}` — and NO `evaluation` block, because
/// the app's rubric-FILE path never wrote one. The pipeline generated
/// successfully, then died at evaluate. Contract under test (cross-engine —
/// server twin: `Server/tests/test_evaluation_declaration.py`):
///
/// - pinned judges + a pinned rubric file ARE a paired-judge declaration:
///   evaluate synthesizes the spec from those pins and stamps the judge
///   report `evaluationSource: "pinnedRubric"`; an explicit `evaluation`
///   block always wins and stamps `"manifest"`;
/// - the researcher's exact manifest shape now evaluates successfully
///   WITHOUT re-freezing (regression fixture mirrors it);
/// - verify() carries the coherence gate (identical wording both engines);
/// - `data check` surfaces the same incoherence as a blocker row;
/// - the draft save writes the explicit block from the pin pair.
@Suite(.serialized) @MainActor struct EvaluationDeclarationTests {

    // MARK: - Workspace fixture

    /// A temp WORKSPACE (programmatic override) so concept stimuli, rubric
    /// files, experiments, and runs all resolve inside it.
    private func withWorkspace<T>(
        _ body: (URL) async throws -> T
    ) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-decl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Plants a paired concept in the workspace and returns its live hash.
    private func plantConcept(_ root: URL, name: String) throws -> String {
        let directory = root.appending(
            components: "prompts", "concepts", name)
        try write(
            "{\"text\": \"I feel dread\"}\n",
            to: directory.appending(component: "positive.jsonl"))
        try write(
            "{\"text\": \"calm morning\"}\n",
            to: directory.appending(component: "negative.jsonl"))
        return try StimulusSet(directory: directory).hash
    }

    private static let rubricText = "Which response expresses more dread?\n"

    /// The researcher's manifest shape: judges + rubric pinned, NO
    /// evaluation block, a run → evaluate → analyze pipeline — saved with
    /// the given status. Returns the saved manifest.
    private func plantStudy(
        _ root: URL, name: String, status: ExperimentManifest.Status
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "fear",
                stimulusSetHash: try plantConcept(root, name: "fear"),
                options: .init()))
        try write(
            Self.rubricText,
            to: root.appending(
                components: "prompts", "rubrics", "default-paired-v1.md"))
        try JudgeRubricStore.pin(
            "prompts/rubrics/default-paired-v1.md", into: &manifest)
        manifest.judges = [
            .init(name: "j-1", kind: "claude", model: "claude-a"),
            .init(name: "j-2", kind: "claude", model: "claude-b"),
        ]
        manifest.evaluation = nil
        manifest.pipeline = .object([
            "stages": .array(
                [.string("run"), .string("evaluate"), .string("analyze")])
        ])
        manifest.status = status
        try ExperimentStore.save(manifest)
        return manifest
    }

    /// A source run with paired generations, stamped with the live manifest
    /// hash so the epoch guard passes.
    private func plantSourceRun(
        manifest: ExperimentManifest
    ) throws -> URL {
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260722T000000000Z-exp-\(manifest.name)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: source.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(manifest).write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        var rows: [String] = []
        for condition in ["baseline", "fear-steered"] {
            for promptID in ["p0", "p1"] {
                rows.append(
                    "{\"experiment\":\"\(manifest.name)\","
                        + "\"condition\":\"\(condition)\",\"seed\":0,"
                        + "\"promptID\":\"\(promptID)\","
                        + "\"prompt\":\"q-\(promptID)\","
                        + "\"output\":\"\(condition)-\(promptID)\"}")
            }
        }
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        return source
    }

    private func withFakeJudge<T>(_ body: () async throws -> T) async rethrows -> T {
        ExperimentTasks.judgeOverrideForTesting = { _, _, _, _ in
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }
        defer { ExperimentTasks.judgeOverrideForTesting = nil }
        return try await body()
    }

    // MARK: - Fix 1: pins ARE the declaration (the regression)

    /// THE regression: the researcher's exact manifest shape (frozen,
    /// evaluation nil, judges + rubric pinned, run→evaluate→analyze
    /// pipeline) evaluates successfully without re-freezing, and the
    /// report says where the spec came from.
    @Test func frozenManifestWithPinsAndNoEvaluationBlockEvaluates() async throws {
        try await withWorkspace { root in
            let manifest = try plantStudy(
                root, name: "agent-comparison", status: .frozen)
            let source = try plantSourceRun(manifest: manifest)
            let out = try await withFakeJudge {
                try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: "agent-comparison",
                    sourceRunDirectory: source)
            }
            let report = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(
                            component: "judge-report.json")))
                    as? [String: Any])
            #expect(report["evaluationSource"] as? String == "pinnedRubric")
            #expect(
                report["judgeRubricFile"] as? String
                    == "prompts/rubrics/default-paired-v1.md")
            let judges = try #require(report["judges"] as? [String])
            #expect(judges == ["j-1", "j-2"])
        }
    }

    /// An explicit evaluation block keeps winning and stamps "manifest".
    @Test func explicitEvaluationBlockWinsAndStampsManifest() async throws {
        try await withWorkspace { root in
            var manifest = try plantStudy(root, name: "explicit", status: .draft)
            manifest.evaluation = .init(
                kind: .pairedJudge, judgeModel: "claude-test",
                judgePrompt: "inline")
            try ExperimentStore.save(manifest)
            let source = try plantSourceRun(manifest: manifest)
            let out = try await withFakeJudge {
                try await ExperimentTasks.evaluatePairedJudge(
                    experimentName: "explicit", sourceRunDirectory: source)
            }
            let report = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(
                            component: "judge-report.json")))
                    as? [String: Any])
            #expect(report["evaluationSource"] as? String == "manifest")
        }
    }

    @Test func effectiveEvaluationResolutionRule() {
        var manifest = ExperimentManifest(
            name: "e", description: "", modelID: "test/model")

        // Nothing declared → nil.
        #expect(ExperimentStore.effectiveEvaluation(manifest) == nil)

        // Judges alone / rubric alone → still nil (the pair declares).
        manifest.judges = [.init(name: "j", kind: "claude", model: nil)]
        #expect(ExperimentStore.effectiveEvaluation(manifest) == nil)
        manifest.judges = nil
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        #expect(ExperimentStore.effectiveEvaluation(manifest) == nil)

        // The pin pair → synthesized pairedJudge spec, source pinnedRubric.
        manifest.judges = [.init(name: "j", kind: "claude", model: nil)]
        let effective = ExperimentStore.effectiveEvaluation(manifest)
        #expect(effective?.source == "pinnedRubric")
        #expect(effective?.spec.kind == .pairedJudge)
        #expect(effective?.spec.judgeModel == "")
        #expect(effective?.spec.judgePrompt == "")
        #expect(effective?.spec.structuredPrompt == nil)

        // An explicit block always wins — even kind .none.
        manifest.evaluation = .init(kind: .none)
        let explicit = ExperimentStore.effectiveEvaluation(manifest)
        #expect(explicit?.source == "manifest")
        #expect(explicit?.spec.kind == ExperimentManifest.EvaluationSpec.Kind.none)
    }

    // MARK: - Fix 2: verify coherence gate + data check

    private func pipelineBlock(_ stages: [String]) -> JSONValue {
        .object(["stages": .array(stages.map { .string($0) })])
    }

    @Test func verifyFlagsEvaluateStageWithNoJudging() {
        var manifest = ExperimentManifest(
            name: "coherence", description: "", modelID: "test/model")
        manifest.pipeline = pipelineBlock(["run", "evaluate", "analyze"])
        #expect(
            ExperimentStore.verify(manifest)
                .contains(ExperimentStore.evaluateWithoutJudgingViolation))

        // Judges + rubric pin pair = effective declaration → coherent.
        manifest.judges = [.init(name: "j", kind: "claude", model: nil)]
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = "00"
        #expect(
            !ExperimentStore.verify(manifest)
                .contains(ExperimentStore.evaluateWithoutJudgingViolation))

        // An explicit kind-none block declares NO judging — the gate fires
        // even next to the pins (the explicit block always wins).
        manifest.evaluation = .init(kind: .none)
        #expect(
            ExperimentStore.verify(manifest)
                .contains(ExperimentStore.evaluateWithoutJudgingViolation))

        // No evaluate stage (or no pipeline) → nothing to gate.
        manifest.evaluation = nil
        manifest.judges = nil
        manifest.judgeRubricFile = nil
        manifest.judgeRubricHash = nil
        manifest.pipeline = pipelineBlock(["run", "analyze"])
        #expect(
            !ExperimentStore.verify(manifest)
                .contains(ExperimentStore.evaluateWithoutJudgingViolation))
        manifest.pipeline = nil
        #expect(
            !ExperimentStore.verify(manifest)
                .contains(ExperimentStore.evaluateWithoutJudgingViolation))
    }

    @Test func dataCheckSurfacesPipelineEvaluateBlocker() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-decl-dc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        var manifest = ExperimentManifest(
            name: "dc", description: "", modelID: "test/model")
        manifest.pipeline = pipelineBlock(["run", "evaluate", "analyze"])
        let rows = StudyDataReadiness.requirements(
            for: manifest, workspaceRoot: temp)
        let row = try #require(rows.first { $0.id == "pipelineEvaluate" })
        #expect(row.status == .missing)
        #expect(row.detail == ExperimentStore.evaluateWithoutJudgingViolation)
        #expect(
            StudyDataReadiness.summary(rows).blockers
                .contains { $0.id == "pipelineEvaluate" })

        // The pin pair resolves it — no row.
        manifest.judges = [.init(name: "j", kind: "claude", model: nil)]
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = "00"
        #expect(
            !StudyDataReadiness.requirements(for: manifest, workspaceRoot: temp)
                .contains { $0.id == "pipelineEvaluate" })
    }

    // MARK: - Fix 3: the draft save writes the declaration

    @Test func evaluationDeclarationWritesExplicitBlockFromPinPair() {
        let judges: [ExperimentManifest.JudgeRef] = [
            .init(name: "j-1", kind: "claude", model: nil),
            .init(name: "j-2", kind: "local", model: nil),
        ]
        // Judges + rubric → the explicit pairedJudge block, mirroring the
        // engines' synthesis (empty judgeModel/judgePrompt), preserving the
        // structured-fields declaration the file path used to drop.
        let declared = ExperimentPanel.evaluationDeclaration(
            judges: judges, rubricFile: "prompts/rubrics/r.md",
            inlineRubric: "", structuredPrompt: "holding_changed boolean",
            inlineJudgeModel: "claude-test")
        #expect(declared?.kind == .pairedJudge)
        #expect(declared?.judgeModel == "")
        #expect(declared?.judgePrompt == "")
        #expect(declared?.structuredPrompt == "holding_changed boolean")

        // Removing the last judge (or clearing the rubric) falls back to
        // the scratchpad rule: inline text → inline spec; nothing → nil.
        #expect(
            ExperimentPanel.evaluationDeclaration(
                judges: [], rubricFile: "prompts/rubrics/r.md",
                inlineRubric: "", structuredPrompt: nil,
                inlineJudgeModel: "claude-test") == nil)
        #expect(
            ExperimentPanel.evaluationDeclaration(
                judges: judges, rubricFile: "", inlineRubric: "",
                structuredPrompt: nil, inlineJudgeModel: "claude-test") == nil)
        let inline = ExperimentPanel.evaluationDeclaration(
            judges: [], rubricFile: "", inlineRubric: "prefer the calmer one",
            structuredPrompt: nil, inlineJudgeModel: "claude-test")
        #expect(inline?.kind == .pairedJudge)
        #expect(inline?.judgeModel == "claude-test")
        #expect(inline?.judgePrompt == "prefer the calmer one")
    }

    /// End to end through the panel: a draft with pinned judges and a
    /// chosen rubric file saves an explicit evaluation block; removing the
    /// judges clears it on the next save.
    @Test func saveEvaluationSettingsWritesAndClearsTheBlock() async throws {
        try await withWorkspace { root in
            _ = try plantConcept(root, name: "fear")
            try write(
                Self.rubricText,
                to: root.appending(components: "prompts", "rubrics", "r.md"))
            var draft = try ExperimentStore.create(
                name: "ui-decl", description: "d", modelID: "test/model")
            draft.concepts.append(
                .init(
                    name: "fear",
                    stimulusSetHash: try StimulusSet(
                        directory: root.appending(
                            components: "prompts", "concepts", "fear")).hash,
                    options: .init()))
            try ExperimentStore.save(draft)

            let panel = ExperimentPanel()
            panel.selectedName = "ui-decl"
            panel.taskPromptsFile = ""  // no prompt pin in this fixture
            panel.judges = [
                .init(name: "j-1", kind: "claude", model: nil),
                .init(name: "j-2", kind: "claude", model: nil),
            ]
            panel.judgeRubricFile = "prompts/rubrics/r.md"
            panel.saveProtocol()

            var saved = try ExperimentStore.load(name: "ui-decl")
            #expect(saved.evaluation?.kind == .pairedJudge)
            #expect(saved.evaluation?.judgePrompt == "")
            #expect(saved.judgeRubricFile == "prompts/rubrics/r.md")
            #expect(saved.judges?.count == 2)

            // Removing every judge clears the declaration coherently.
            panel.judges = []
            panel.saveProtocol()
            saved = try ExperimentStore.load(name: "ui-decl")
            #expect(saved.evaluation == nil)
            #expect(saved.judges == nil)
        }
    }
}
