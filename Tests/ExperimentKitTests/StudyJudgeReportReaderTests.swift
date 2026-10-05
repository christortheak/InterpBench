import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The judged section reads judge reports from both engines (release review
/// finding E6): a strict decoder for the Mac engine's keys read a
/// Python-engine report as "no report", so the section was missing for
/// every server and cluster run.
@Suite struct StudyJudgeReportReaderTests {

    // MARK: - the Python engine's dialect

    @Test func aPythonEngineReportProducesTheJudgedSection() throws {
        let report = try #require(
            StudyJudgeReportReader.read(
                try JudgeArtifactFixture.data("python-engine", "judge-report.json")))
        #expect(report.dialect == .pythonEngine)
        // `sourceRun`, a directory name.
        #expect(report.sourceRunDirectory == JudgeArtifactFixture.pythonSourceRun)
        #expect(StudyJudgeReportReader.sourceRunName(report)
            == JudgeArtifactFixture.pythonSourceRun)
        #expect(report.judgeNames == ["judge-a", "judge-b"])
        #expect(report.judgeModel == "claude-opus-4-8, claude-opus-4-8")

        // Per-judge tallies, `variantWins` read as the condition's wins and
        // `n` as the verdicts counted. judge-a has one verdict fewer: its
        // fourth pair is the noncompliant one.
        #expect(report.judgeBlocks.map(\.name) == ["judge-a", "judge-b"])
        let a = try #require(report.judgeBlocks.first?.conditions.first)
        #expect(a.name == "warm")
        #expect(a.pairs == 3)
        #expect(a.conditionWins == 1)
        #expect(a.baselineWins == 1)
        #expect(a.ties == 1)
        #expect(a.meanConfidence == nil)
        let b = try #require(report.judgeBlocks.last?.conditions.first)
        #expect(b.pairs == 4)
        #expect(b.baselineWins == 2)
        #expect(report.judgeBlocks.first?.pairs == 4)
        #expect(report.judgeBlocks.first?.noncompliantJudgments == 1)
        #expect(report.judgeBlocks.last?.noncompliantJudgments == nil)

        // The top-level tally is the FIRST judge's, as that engine writes
        // it; the section must not present it as the panel's.
        #expect(report.conditions == report.judgeBlocks.first?.conditions)
        #expect(report.tallyGroups.map(\.title) == ["judge-a", "judge-b"])
        #expect(
            report.tallyGroups.first?.conditions.first?.tallyLine
                == "condition 1 · baseline 1 · ties 1 · mean confidence not stored")
        #expect(
            report.tallyGroups.last?.conditions.first?.tallyLine
                == "condition 1 · baseline 2 · ties 1 · mean confidence not stored")

        // Reliability values, read from the report.
        #expect(
            report.judgeAgreement == [
                .init(
                    judgeA: "judge-a", judgeB: "judge-b", items: 3,
                    percentAgreement: 1.0, kappa: 1.0)
            ])
        #expect(report.humanAgreement?.map(\.judge) == ["judge-a", "judge-b"])
        #expect(report.humanAgreement?.first?.items == 3)
        #expect(report.humanAgreement?.first?.kappa == 0.0)
        #expect(report.noncompliantJudgments == 1)
        #expect(report.epochUnverified == true)
        #expect(report.measurementDrift == nil)
        #expect(report.excludedRecords == nil)
        #expect(report.judgingSessions == nil)
    }

    @Test func aPythonEngineReportsReliabilityLinesCarryItsNumbers() throws {
        let report = try #require(
            StudyJudgeReportReader.read(
                try JudgeArtifactFixture.data("python-engine", "judge-report.json")))
        let lines = report.reliabilityLines
        #expect(
            lines.map(\.id) == [
                "agreement:judge-a:judge-b", "human:judge-a", "human:judge-b",
                "noncompliant", "epochUnverified", "salvagedVerdicts",
            ])
        let byID = Dictionary(uniqueKeysWithValues: lines.map { ($0.id, $0) })
        #expect(
            byID["agreement:judge-a:judge-b"]?.label
                == "Agreement between judges: judge-a and judge-b")
        #expect(
            byID["agreement:judge-a:judge-b"]?.value
                == "100% of 3 pairs both judged · kappa 1.00")
        #expect(
            byID["human:judge-a"]?.value == "33% of 3 rated pairs · kappa 0.00")
        #expect(byID["noncompliant"]?.label == "Pairs with no verdict")
        #expect(byID["noncompliant"]?.value == "1 (judge-a 1)")
        #expect(byID["noncompliant"]?.tone == .caution)
        #expect(
            byID["epochUnverified"]?.label
                == "Source run not checked against the study")
        #expect(byID["epochUnverified"]?.tone == .caution)
        #expect(byID["salvagedVerdicts"]?.value == "judge-a 1, judge-b 1")
    }

    // MARK: - the Mac engine's dialect

    @Test func aMacEngineReportProducesTheJudgedSection() throws {
        let report = try #require(
            StudyJudgeReportReader.read(
                try JudgeArtifactFixture.data("mac-engine", "judge-report.json")))
        #expect(report.dialect == .macEngine)
        // `sourceRunDirectory`, a full path.
        #expect(
            report.sourceRunDirectory
                == "/workspace/runs/" + JudgeArtifactFixture.macSourceRun)
        #expect(
            StudyJudgeReportReader.sourceRunName(report)
                == JudgeArtifactFixture.macSourceRun)
        #expect(report.judgeNames == ["judge-a", "judge-b"])
        #expect(report.judgeModel == "test/model, test/model")

        // One tally for the whole panel: 7 verdicts from two judges over
        // four pairs, one of them noncompliant for one judge.
        #expect(report.judgeBlocks.isEmpty)
        let warm = try #require(report.conditions.first)
        #expect(report.conditions.count == 1)
        #expect(warm.name == "warm")
        #expect(warm.pairs == 7)
        #expect(warm.conditionWins == 2)
        #expect(warm.baselineWins == 3)
        #expect(warm.ties == 2)
        // (0.8 + 0.5 + 0.6) + (0.8 + 0.5 + 0.6 + 0.8) = 4.6, over 7.
        #expect(abs(try #require(warm.meanConfidence) - 4.6 / 7) < 1e-12)
        #expect(
            warm.tallyLine
                == "condition 2 · baseline 3 · ties 2 · mean confidence 0.66")
        #expect(report.tallyGroups.count == 1)
        #expect(
            report.tallyGroups.first?.note
                == "These counts add up the verdicts of all 2 judges (judge-a, judge-b).")

        #expect(
            report.judgeAgreement == [
                .init(
                    judgeA: "judge-a", judgeB: "judge-b", items: 3,
                    percentAgreement: 1.0, kappa: 1.0)
            ])
        #expect(report.humanAgreement?.map(\.judge) == ["judge-a", "judge-b"])
        #expect(report.humanAgreement?.first?.items == 3)
        #expect(report.noncompliantJudgments == 1)
        #expect(report.epochUnverified == true)

        let lines = report.reliabilityLines
        #expect(
            lines.map(\.id) == [
                "agreement:judge-a:judge-b", "human:judge-a", "human:judge-b",
                "noncompliant", "epochUnverified",
            ])
        // A Mac-engine report stores the panel's total only.
        #expect(lines.first { $0.id == "noncompliant" }?.value == "1")
    }

    /// The Mac fixture decodes with the engine's OWN strict report and row
    /// types, so it cannot drift into this test's idea of the dialect.
    @Test func theMacFixtureIsInTheMacEnginesOwnFormat() throws {
        let report = try JSONDecoder().decode(
            ExperimentTasks.PairedJudgeReport.self,
            from: try JudgeArtifactFixture.data("mac-engine", "judge-report.json"))
        #expect(report.judges == ["judge-a", "judge-b"])
        #expect(report.noncompliantJudgments == 1)
        #expect(report.epochUnverified == true)
        #expect(report.conditions["warm"]?.pairs == 7)

        let rows = String(
            decoding: try JudgeArtifactFixture.data("mac-engine", "judgments.jsonl"),
            as: UTF8.self
        ).split(separator: "\n")
        #expect(rows.count == 8)
        let verdicts = rows.compactMap {
            try? JSONDecoder().decode(
                ExperimentTasks.PairedJudgeRecord.self, from: Data($0.utf8))
        }
        // Seven verdict rows; the eighth is the noncompliant row, which is
        // not a verdict record and is exactly what a strict reader drops.
        #expect(verdicts.count == 7)
    }

    /// The two fixtures are the same evaluation design run on each engine,
    /// so the facts both reports store must read the same.
    @Test func theSameEvaluationReadsTheSameFromBothEngines() throws {
        func lines(_ engine: String) throws -> [String: String] {
            let report = try #require(
                StudyJudgeReportReader.read(
                    try JudgeArtifactFixture.data(engine, "judge-report.json")))
            return Dictionary(
                uniqueKeysWithValues: report.reliabilityLines.map { ($0.id, $0.value) })
        }
        let mac = try lines("mac-engine")
        let python = try lines("python-engine")
        for id in [
            "agreement:judge-a:judge-b", "human:judge-a", "human:judge-b",
            "epochUnverified",
        ] {
            #expect(mac[id] != nil, "\(id)")
            #expect(mac[id] == python[id], "\(id)")
        }
        #expect(mac["agreement:judge-a:judge-b"] == "100% of 3 pairs both judged · kappa 1.00")
        #expect(mac["human:judge-a"] == "33% of 3 rated pairs · kappa 0.00")
    }

    // MARK: - shapes a fixture does not reach

    @Test func stampsPresentInAReportAreShown() throws {
        let data = Data(
            """
            {"sourceRun": "run-a", "judgeModel": "judge-model",
             "judges": [{"name": "solo", "requestedModel": "judge-model",
                         "actualModel": "judge-model-0420", "pairs": 1240,
                         "conditions": {"warm": {"baselineWins": 400,
                             "variantWins": 800, "ties": 40, "n": 1240}}}],
             "agreement": [],
             "conditions": {"warm": {"baselineWins": 400, "variantWins": 800,
                                     "ties": 40, "n": 1240}},
             "measurementDrift": "maxTokens: 256 -> 512",
             "exclusions": {"excludedRecords": 1500, "rules": []},
             "judgingSessions": {"resumedFrom": "earlier-evaluate",
                                 "reusedJudgments": 1000,
                                 "freshJudgments": 240,
                                 "unpinnedExternalJudges": ["solo"]}}
            """.utf8)
        let report = try #require(StudyJudgeReportReader.read(data))
        #expect(report.dialect == .pythonEngine)
        #expect(
            report.tallyGroups.first?.note == "judge-model, answered by judge-model-0420")
        #expect(
            report.tallyGroups.first?.conditions.first?.tallyLine
                == "condition 800 · baseline 400 · ties 40 · mean confidence not stored")
        let byID = Dictionary(
            uniqueKeysWithValues: report.reliabilityLines.map { ($0.id, $0) })
        // One judge: there is no second verdict to compare.
        #expect(byID["agreement"]?.value == "Not available: one judge")
        #expect(byID["noncompliant"]?.value == "None recorded")
        #expect(byID["noncompliant"]?.tone == .plain)
        #expect(byID["epochUnverified"] == nil)
        #expect(byID["measurementDrift"]?.value == "maxTokens: 256 -> 512")
        #expect(byID["excludedRecords"]?.value == "1,500")
        #expect(
            byID["judgingSessions"]?.value
                == "1,000 kept from earlier-evaluate, 240 new")
    }

    @Test func agreementThatIsUndefinedOrEmptyIsSaidNotShownAsZero() throws {
        let data = Data(
            """
            {"sourceRunDirectory": "/workspace/runs/run-a", "judgeModel": "m, m",
             "judges": ["one", "two"],
             "judgeAgreement": [
                {"judgeA": "one", "judgeB": "two", "items": 12,
                 "percentAgreement": 0.5},
                {"judgeA": "one", "judgeB": "three", "items": 0,
                 "percentAgreement": 0}],
             "humanAgreement": [
                {"judge": "one", "items": 0, "percentAgreement": 0}],
             "conditions": {"warm": {"pairs": 24, "conditionWins": 12,
                 "baselineWins": 10, "ties": 2, "meanConfidence": 0.8125,
                 "structuredSummaries": {}}}}
            """.utf8)
        let report = try #require(StudyJudgeReportReader.read(data))
        #expect(report.dialect == .macEngine)
        let byID = Dictionary(
            uniqueKeysWithValues: report.reliabilityLines.map { ($0.id, $0) })
        // No stored kappa is "not defined", never 0.00.
        #expect(
            byID["agreement:one:two"]?.value
                == "50% of 12 pairs both judged · kappa not defined")
        // The Mac engine writes 0 for an empty comparison; "0%" would be a
        // claim about pairs that do not exist.
        #expect(byID["agreement:one:three"]?.value == "No pair was judged by both")
        #expect(
            byID["human:one"]?.value == "No pair this judge judged has a human rating")
        // A panel-wide tally of more than one judge says that it is a sum.
        #expect(report.tallyGroups.count == 1)
        #expect(report.tallyGroups.first?.title == nil)
        #expect(
            report.tallyGroups.first?.note
                == "These counts add up the verdicts of all 2 judges (one, two).")
        #expect(
            report.conditions.first?.tallyLine
                == "condition 12 · baseline 10 · ties 2 · mean confidence 0.81")
    }

    @Test func everyReliabilityLineHasOnePlainSentenceOfExplanation() throws {
        var lines: [StudyJudgeReliabilityLine] = []
        for engine in ["python-engine", "mac-engine"] {
            lines += try #require(
                StudyJudgeReportReader.read(
                    try JudgeArtifactFixture.data(engine, "judge-report.json"))
            ).reliabilityLines
        }
        #expect(!lines.isEmpty)
        for line in lines {
            #expect(!line.label.isEmpty)
            #expect(!line.value.isEmpty)
            #expect(line.explanation.hasSuffix("."), "\(line.id)")
            // One sentence: no full stop before the last character.
            #expect(
                !line.explanation.dropLast().contains(". "),
                "\(line.id) explains itself in more than one sentence")
            #expect(line.explanation.count > 60, "\(line.id)")
        }
    }

    @Test func bytesThatAreNotAJudgeReportReadAsNoReport() {
        #expect(StudyJudgeReportReader.read(Data("not JSON".utf8)) == nil)
        #expect(StudyJudgeReportReader.read(Data("[1, 2]".utf8)) == nil)
        #expect(StudyJudgeReportReader.read(Data("{\"experiment\": \"x\"}".utf8)) == nil)
        // A run's report.json also has a `conditions` object, with no judge
        // tally in it: not a judge report.
        #expect(
            StudyJudgeReportReader.read(
                Data(
                    #"{"experiment": "x", "conditions": {"warm": {"generations": 4, "meanWordCount": 12.5}}}"#
                        .utf8)) == nil)
    }
}

/// A temporary workspace holding one run of a study named `tone-study`, for
/// the repository-level tests of the judged section.
struct JudgedSectionWorkspace {
    let root: URL
    let run: URL
    var repository: StudyResultRepository { StudyResultRepository(workspaceRoot: root) }

    init(runName: String) throws {
        root = FileManager.default.temporaryDirectory
            .appending(component: "judged-section-\(UUID().uuidString)")
        run = root.appending(components: "runs", runName)
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(
            ExperimentManifest(
                name: "tone-study", description: "", modelID: "example/model")
        ).write(to: run.appending(component: "experiment.json"))
    }

    func evaluateDirectory(_ name: String) throws -> URL {
        let url = root.appending(components: "runs", name)
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    /// An evaluate directory holding one engine's committed artifacts.
    @discardableResult
    func plantEvaluation(_ name: String, engine: String) throws -> URL {
        let url = try evaluateDirectory(name)
        for file in ["judge-report.json", "judgments.jsonl"] {
            try JudgeArtifactFixture.data(engine, file)
                .write(to: url.appending(component: file))
        }
        return url
    }

    func detail() throws -> StudyRunDetail {
        let item = try #require(
            repository.list(experimentName: "tone-study").first { $0.kind == .run })
        return repository.detail(for: item)
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

/// Which judge directory belongs to a run — the step that decides whether
/// a run has a judged section at all.
@Suite struct StudyResultsJudgeDirectoryTests {

    @Test func aPythonEngineJudgeDirectoryIsFoundForItsRun() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: JudgeArtifactFixture.pythonSourceRun)
        defer { workspace.remove() }
        let evaluate = try workspace.plantEvaluation(
            "20260102T000000000-exp-tone-study-evaluate", engine: "python-engine")

        let detail = try workspace.detail()
        // The section that was missing for server and cluster runs.
        let report = try #require(detail.pairedJudgeReport)
        #expect(report.dialect == .pythonEngine)
        #expect(report.judgeBlocks.count == 2)
        #expect(
            URL(filePath: try #require(detail.judgeArtifactDirectory)).lastPathComponent
                == evaluate.lastPathComponent)
    }

    @Test func aMacEngineJudgeDirectoryIsFoundAfterTheWorkspaceMoves() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: JudgeArtifactFixture.macSourceRun)
        defer { workspace.remove() }
        // The fixture's stored path is `/workspace/runs/<run>`: not this
        // temporary workspace, as after a move or a copy.
        try workspace.plantEvaluation(
            "20260102T000000000Z-exp-tone-study-evaluate", engine: "mac-engine")

        let detail = try workspace.detail()
        let report = try #require(detail.pairedJudgeReport)
        #expect(report.dialect == .macEngine)
        #expect(report.sourceRunDirectory != workspace.run.path)
        #expect(detail.judgeArtifactDirectory != nil)
    }

    @Test func aReportAboutAnotherRunIsNotAttached() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: "20260103T000000000Z-exp-tone-study-run")
        defer { workspace.remove() }
        try workspace.plantEvaluation(
            "20260104T000000000Z-exp-tone-study-evaluate", engine: "mac-engine")
        try workspace.plantEvaluation(
            "20260105T000000000-exp-tone-study-evaluate", engine: "python-engine")
        let detail = try workspace.detail()
        #expect(detail.pairedJudgeReport == nil)
        #expect(detail.judgeArtifactDirectory == nil)
    }

    @Test func theNewestFinishedEvaluationOfARunIsTheOneShown() throws {
        let workspace = try JudgedSectionWorkspace(
            runName: JudgeArtifactFixture.pythonSourceRun)
        defer { workspace.remove() }
        try workspace.plantEvaluation(
            "20260102T000000000-exp-tone-study-evaluate", engine: "python-engine")
        let newest = try workspace.plantEvaluation(
            "20260109T000000000-exp-tone-study-evaluate", engine: "python-engine")
        #expect(
            URL(filePath: try #require(try workspace.detail().judgeArtifactDirectory))
                .lastPathComponent == newest.lastPathComponent)
    }
}
