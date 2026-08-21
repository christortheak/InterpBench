import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Evaluate's local-judge model resolution (unified with the sweep rule,
/// 2026-07-22 incident): a researcher's evaluate run died on an offline
/// compute node because the SERVER's evaluate path resolved a local judge's
/// model as `model or name` — the judge's NAME ('judge-1') was sent to
/// HuggingFace as a model id. Swift's `resolvedJudges` already carried the
/// correct rule for both paths; these tests pin the evaluate-path shape
/// end to end (the researcher's exact panel judges through the study model
/// and stamps it — never a judge name as a model id), plus the cross-engine
/// evaluate-start log line and the plain-language load-failure wording.
/// Server twin: `Server/tests/test_evaluate_local_judge.py`.
@Suite(.serialized) struct EvaluateLocalJudgeResolutionTests {

    // MARK: pure wording seams (cross-engine contract)

    @Test func resolutionLogLineNamesTheStudyModel() {
        let defaulted = ExperimentTasks.ResolvedJudge(
            name: "judge-1", kind: "local", model: "test/model",
            modelDefaulted: true)
        #expect(
            ExperimentTasks.localJudgeResolutionLogLine(
                defaulted, studyModelID: "test/model")
                == "local judge 'judge-1' resolves to the study model test/model")
        // An explicitly-named study model logs the same resolution line.
        let named = ExperimentTasks.ResolvedJudge(
            name: "judge-2", kind: "local", model: "test/model")
        #expect(
            ExperimentTasks.localJudgeResolutionLogLine(
                named, studyModelID: "test/model")
                == "local judge 'judge-2' resolves to the study model test/model")
        // A genuinely different local model is named as such.
        let other = ExperimentTasks.ResolvedJudge(
            name: "judge-3", kind: "local", model: "other/model")
        #expect(
            ExperimentTasks.localJudgeResolutionLogLine(
                other, studyModelID: "test/model")
                == "local judge 'judge-3' judges with local model 'other/model'")
    }

    @Test func loadFailureMessageNamesJudgeModelAndRemedies() {
        let message = ExperimentTasks.localJudgeLoadFailureMessage(
            judgeName: "judge-1", model: "other/model")
        #expect(message.contains("local judge 'judge-1'"))
        #expect(message.contains("declares model 'other/model'"))
        #expect(message.contains("install the model"))
        #expect(message.contains("leave the judge's model empty"))
        #expect(message.contains("judge with the study model"))
        // Plain language only — never the raw hub/network dump.
        #expect(!message.contains("huggingface"))
        #expect(!message.contains("internet connection"))
    }

    // MARK: the researcher's exact shape, end to end

    /// Two pinned local judges named judge-1/judge-2 with EMPTY model (the
    /// UI's "study model (default)") judge through the STUDY model: every
    /// judgment stamps the study model as the judge model — the judge names
    /// never appear as model ids anywhere in the artifacts.
    @Test func emptyModelLocalJudgesEvaluateThroughTheStudyModel() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-local-judge-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.judgeOverrideForTesting = { _, _, _, _ in
            PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "A", confidence: 0.9, briefReason: "fake")
        }
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        let study = "local-judge-panel"
        var manifest = try ExperimentStore.create(
            name: study, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french",
                stimulusSetHash: try StimulusSet(
                    directory: VectorCatalog.conceptsDirectory
                        .appending(component: "french")
                ).hash,
                options: .init()))
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "local", model: "  "),
        ]
        try ExperimentStore.save(manifest)
        let liveHash = ExperimentStore.manifestHash(manifest)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260722T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try liveHash.write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        let rows = [
            "{\"experiment\":\"\(study)\",\"condition\":\"baseline\","
                + "\"seed\":1,\"promptID\":\"p0\",\"prompt\":\"q\","
                + "\"output\":\"calm\"}",
            "{\"experiment\":\"\(study)\",\"condition\":\"fear\","
                + "\"seed\":1,\"promptID\":\"p0\",\"prompt\":\"q\","
                + "\"output\":\"scared\"}",
        ]
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: study,
            sourceRunDirectory: source,
            evaluation: ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge, judgeModel: "",
                judgePrompt: "prefer the better response"))

        let lines = try String(
            contentsOf: out.appending(component: "judgments.jsonl"),
            encoding: .utf8
        ).split(separator: "\n")
        #expect(lines.count == 2)  // one pair × two judges
        var judgeNames: Set<String> = []
        for line in lines {
            let object = try #require(
                try JSONSerialization.jsonObject(with: Data(line.utf8))
                    as? [String: Any])
            // The judge's model IS the study model — never 'judge-1' or
            // 'judge-2' treated as a model id.
            #expect(object["judgeModel"] as? String == "test/model")
            judgeNames.insert(try #require(object["judge"] as? String))
        }
        #expect(judgeNames == ["judge-1", "judge-2"])
        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "judge-report.json")))
                as? [String: Any])
        #expect(report["judgeModel"] as? String == "test/model, test/model")
        #expect((report["judges"] as? [String]) == ["judge-1", "judge-2"])
    }
}
