import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// What a run's detail says about how much there is to review (release
/// review finding E6): the counts are the files' own, a bounded preview is
/// never the count, and an evaluation that stopped is shown, not absent.
@Suite struct StudyResultsRecordCountTests {

    private func line(_ object: [String: Any]) throws -> String {
        String(
            decoding: try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys]),
            as: UTF8.self)
    }

    /// The detail's preview is the first lines only; its counts are not.
    @Test func aRunOfAThousandRecordsIsCountedInFull() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: "20260101T000000000-exp-tone-study-run")
        defer { workspace.remove() }
        var lines: [String] = []
        for index in 0..<1_000 {
            lines.append(
                try line([
                    "condition": index % 2 == 0 ? "baseline" : "warm",
                    "promptID": "p\(index / 2)",
                    "prompt": "Describe room \(index / 2).",
                    "output": "An answer about room \(index / 2).",
                    "wordCount": 5, "distinct2": 1.0,
                ]))
        }
        // A failure record: one line no strict decoder reads.
        lines.append(
            try line([
                "condition": "cool", "error": "the saved settings could not be loaded",
            ]))
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: workspace.run.appending(component: "generations.jsonl"))

        let item = try #require(
            workspace.repository.list(experimentName: "tone-study").first)
        #expect(item.generationCount == 1_001)
        let detail = workspace.repository.detail(for: item)
        #expect(detail.responseRecordCount == 1_001)
        // A preview of 80, beside a count of 1,001.
        #expect(detail.generations.count == StudyResultRepository.previewResponseLimit)
        #expect(detail.judgmentRecordCount == 0)

        // And the file the count describes is the one the review pages.
        let review = StudyRecordReview<StudyResponseRow>(
            url: StudyResultRepository.responsesURL(runDirectory: workspace.run))
        #expect(review.counts.total == detail.responseRecordCount)
        #expect(review.counts.count(.response) == 1_000)
        #expect(review.counts.count(.failed) == 1)
    }

    /// A run whose file holds only a failure record still has something to
    /// review: the count, not the preview, decides whether the view offers
    /// the review.
    @Test func aRunHoldingOnlyAFailureRecordIsStillReviewable() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: "20260101T000000000-exp-tone-study-run")
        defer { workspace.remove() }
        let failure = try line([
            "condition": "cool", "error": "the saved settings could not be loaded",
        ])
        try Data((failure + "\n").utf8).write(
            to: workspace.run.appending(component: "generations.jsonl"))
        let detail = try workspace.detail()
        #expect(detail.generations.isEmpty)
        #expect(detail.responseRecordCount == 1)
    }

    @Test func everyJudgeRowIsCountedTheNoncompliantOneIncluded() throws {
        for (engine, runName, evaluateName) in [
            (
                "python-engine", JudgeArtifactFixture.pythonSourceRun,
                "20260102T000000000-exp-tone-study-evaluate"
            ),
            (
                "mac-engine", JudgeArtifactFixture.macSourceRun,
                "20260102T000000000Z-exp-tone-study-evaluate"
            ),
        ] {
            let workspace = try JudgedSectionWorkspace(runName: runName)
            defer { workspace.remove() }
            try workspace.plantEvaluation(evaluateName, engine: engine)
            let detail = try workspace.detail()
            #expect(detail.pairedJudgeReport != nil, "\(engine)")
            #expect(detail.judgmentRecordCount == 8, "\(engine)")
            #expect(detail.unfinishedEvaluations.isEmpty, "\(engine)")
        }
    }

    @Test func anEvaluationThatStoppedIsShownWithTheRowsItKept() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: JudgeArtifactFixture.pythonSourceRun)
        defer { workspace.remove() }

        // Stopped mid-panel: rows on disk, a status file, no report.
        let stopped = try workspace.evaluateDirectory(
            "20260102T000000000-exp-tone-study-evaluate")
        try JudgeArtifactFixture.data("python-engine", "judgments.jsonl")
            .write(to: stopped.appending(component: "judgments.jsonl"))
        try Data(
            """
            {"schemaVersion": 1, "stage": "evaluate", "status": "failed",
             "evidenceComplete": false, "itemLabel": "judgment",
             "itemsWritten": 8, "invalidResponses": 0,
             "sourceRun": "\(JudgeArtifactFixture.pythonSourceRun)",
             "expectedUnits": ["judge-a", "judge-b"],
             "completedUnits": ["judge-a"], "pendingUnits": ["judge-b"],
             "error": "the judge service did not answer",
             "errorType": "RuntimeError"}
            """.utf8
        ).write(to: stopped.appending(component: RunStatusFile.filename))

        // A stopped evaluation of ANOTHER run is not this run's.
        let unrelated = try workspace.evaluateDirectory(
            "20260103T000000000-exp-tone-study-evaluate")
        try Data(
            #"{"stage": "evaluate", "status": "failed", "sourceRun": "some-other-run"}"#.utf8
        ).write(to: unrelated.appending(component: RunStatusFile.filename))

        // An evaluation waiting for judgments made elsewhere has no status
        // file and is not "stopped".
        _ = try workspace.evaluateDirectory(
            "20260104T000000000-exp-tone-study-evaluate")

        let detail = try workspace.detail()
        #expect(detail.pairedJudgeReport == nil)
        #expect(
            detail.unfinishedEvaluations.map(\.directoryName)
                == ["20260102T000000000-exp-tone-study-evaluate"])
        let evaluation = try #require(detail.unfinishedEvaluations.first)
        #expect(
            evaluation.summary
                == "Stopped with an error: the judge service did not answer "
                + "Judges that did not finish: judge-b.")
        #expect(evaluation.judgmentRecordCount == 8)

        // The rows it kept are reviewable from its own directory.
        let review = StudyRecordReview<StudyJudgmentRow>(
            url: StudyResultRepository.judgmentsURL(
                directory: URL(filePath: evaluation.path)))
        #expect(review.counts.total == 8)
        #expect(review.counts.count(.noncompliant) == 1)
    }

    /// A finished evaluation and a stopped one of the same run: the report
    /// is shown, and the stopped one is still mentioned.
    @Test func aStoppedEvaluationBesideAFinishedOneIsStillShown() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: JudgeArtifactFixture.pythonSourceRun)
        defer { workspace.remove() }
        try workspace.plantEvaluation(
            "20260102T000000000-exp-tone-study-evaluate", engine: "python-engine")
        let stopped = try workspace.evaluateDirectory(
            "20260105T000000000-exp-tone-study-evaluate")
        try Data(
            """
            {"stage": "evaluate", "status": "failed", "errorType": "Cancelled",
             "error": "cancelled by user",
             "sourceRun": "\(JudgeArtifactFixture.pythonSourceRun)"}
            """.utf8
        ).write(to: stopped.appending(component: RunStatusFile.filename))

        let detail = try workspace.detail()
        #expect(detail.pairedJudgeReport != nil)
        #expect(
            detail.unfinishedEvaluations.map(\.summary)
                == ["Cancelled before it finished."])
        #expect(detail.unfinishedEvaluations.first?.judgmentRecordCount == 0)
    }

    @Test func theStoppedSummarySaysWhatTheStatusFileSays() {
        func summary(_ status: RunStatusFile.Status) -> String? {
            StudyResultRepository.unfinishedSummary(.present(status))
        }
        #expect(StudyResultRepository.unfinishedSummary(.absent) == nil)
        #expect(summary(.init(status: "completed")) == nil)
        #expect(
            StudyResultRepository.unfinishedSummary(.unreadable)
                == "Its status file cannot be read, so it is treated as unfinished.")
        #expect(
            summary(
                .init(status: "failed", error: "cancelled by user", errorType: "Cancelled"))
                == "Cancelled before it finished.")
        #expect(
            summary(.init(status: "checkpointed"))
                == "Paused at a checkpoint; it can be resumed.")
        #expect(
            summary(.init(status: "inProgress"))
                == "Still running, or stopped without recording why.")
        #expect(summary(.init(status: "failed")) == "Stopped before it finished.")
    }
}

/// End to end against the Mac engine's real writer: an evaluation it runs
/// today, with a scripted panel, is fully reviewable through the repository.
/// The committed fixtures pin each dialect; this pins the live writer.
@Suite(.serialized) struct StudyResultsMacEvaluationTests {

    @Test func theMacEnginesOwnEvaluationIsFullyReviewable() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "mac-evaluation-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        // The concept files live under the WORKSPACE root, a second
        // process-global seam. Point it here too and give the study its own
        // copy of the example concept, so the test never depends on whatever
        // another test left that root resolving to.
        let previousWorkspace = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        ExperimentTasks.judgeOverrideForTesting = { judge, prompt, _, _ in
            if prompt.contains("room 3"), judge == "judge-a" {
                // Not A, B, or tie, on both attempts: noncompliant.
                return PairedJudgeResponse(
                    winner: "both", confidence: 0.5, briefReason: "cannot decide")
            }
            if prompt.contains("room 1") {
                return PairedJudgeResponse(
                    winner: "tie", confidence: 0.5,
                    briefReason: "Both are equally warm.")
            }
            return PairedJudgeResponse(
                winner: "B", confidence: 0.8,
                briefReason: "The second reply is warmer.")
        }
        defer {
            ExperimentTasks.judgeOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = previousWorkspace
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        let conceptCopy = VectorCatalog.conceptsDirectory.appending(component: "french")
        try FileManager.default.createDirectory(
            at: conceptCopy.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: CodeResources.compiledCheckoutPath.appending(path: "prompts/concepts/french"),
            to: conceptCopy)

        let study = "tone-study"
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
        manifest.judges = ["judge-a", "judge-b"].map {
            .init(name: $0, kind: "local", model: nil)
        }
        try ExperimentStore.save(manifest)

        let source = ExperimentStore.runsDirectory.appending(
            component: "20260101T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: source.appending(component: "experiment.json"))
        try ExperimentStore.manifestHash(manifest).write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        var rows: [String] = []
        var seed = 0
        for condition in ["baseline", "warm"] {
            for index in 0..<4 {
                seed += 1
                rows.append(
                    "{\"experiment\":\"\(study)\",\"condition\":\"\(condition)\","
                        + "\"seed\":\(seed),\"promptID\":\"p\(index)\","
                        + "\"prompt\":\"Describe room \(index).\","
                        + "\"output\":\"\(condition) answer \(index)\"}")
            }
        }
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        let evaluate = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: study, sourceRunDirectory: source,
            evaluation: .init(
                kind: .pairedJudge, judgeModel: "claude-test",
                judgePrompt: "Which response is warmer in tone?"))

        let repository = StudyResultRepository(workspaceRoot: root)
        let item = try #require(
            repository.list(experimentName: study).first { $0.kind == .run })
        let detail = repository.detail(for: item)

        // The judged section, from what the engine just wrote.
        let report = try #require(detail.pairedJudgeReport)
        #expect(report.dialect == .macEngine)
        #expect(
            URL(filePath: try #require(detail.judgeArtifactDirectory))
                .lastPathComponent == evaluate.lastPathComponent)
        #expect(report.judgeNames == ["judge-a", "judge-b"])
        #expect(report.noncompliantJudgments == 1)
        #expect(report.conditions.first?.pairs == 7)
        #expect(report.judgeAgreement.first?.items == 3)
        #expect(report.epochUnverified == nil)
        let noncompliant = try #require(
            report.reliabilityLines.first { $0.id == "noncompliant" })
        #expect(noncompliant.value == "1")
        #expect(noncompliant.tone == .caution)

        // Every row the engine wrote is counted and reviewable — the
        // noncompliant row as a row, which the bounded strict preview
        // (seven verdicts) does not hold.
        #expect(detail.judgmentRecordCount == 8)
        #expect(detail.judgments.count == 7)
        let review = StudyRecordReview<StudyJudgmentRow>(
            url: StudyResultRepository.judgmentsURL(directory: evaluate))
        #expect(review.counts.total == 8)
        #expect(review.counts.count(.verdict) == 7)
        #expect(review.counts.count(.noncompliant) == 1)
        #expect(review.counts.count(.unreadable) == 0)
        let hole = try #require(
            review.page(0, kind: .noncompliant).rows.first)
        #expect(hole.judge == "judge-a")
        #expect(hole.promptID == "p3")
        #expect(hole.reason?.contains("invalid verdict twice") == true)
        #expect(detail.unfinishedEvaluations.isEmpty)

        // The source run's own records, counted the same way.
        #expect(detail.responseRecordCount == 8)
    }
}
