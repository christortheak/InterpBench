import Foundation
import Testing

@testable import ExperimentKit

/// Home's first-study checklist, judged from real workspace folders: none,
/// a new workspace, a copy of the Demo Workspace, and a workspace whose study
/// has been run, analyzed, and exported.
///
/// Every workspace here is a temporary folder named explicitly; nothing reads
/// or changes the process-wide workspace.
@Suite struct FirstStudyChecklistTests {

    static let repository = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let demoFixtures = repository.appending(path: "Tests/Fixtures/DemoWorkspaces")

    private let fm = FileManager.default

    private func temporary(_ name: String) -> URL {
        fm.temporaryDirectory.appending(component: "steerlab-checklist-\(name)-\(UUID().uuidString)")
    }

    private func done(_ items: [FirstStudyChecklist.Item]) -> [FirstStudyChecklist.Step] {
        items.filter(\.isDone).map(\.step)
    }

    private func item(
        _ step: FirstStudyChecklist.Step, _ items: [FirstStudyChecklist.Item]
    ) throws -> FirstStudyChecklist.Item {
        try #require(items.first { $0.step == step })
    }

    /// A completed run directory: responses, report, and its study's stamp.
    private func writeRun(_ name: String, study: String, in root: URL, complete: Bool = true) throws {
        let directory = root.appending(components: "runs", name)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{\"condition\":\"baseline\"}\n".utf8)
            .write(to: directory.appending(component: "generations.jsonl"))
        try Data("{\"experiment\":\"\(study)\"}".utf8)
            .write(to: directory.appending(component: "config.json"))
        if complete {
            try Data("{}".utf8).write(to: directory.appending(component: "report.json"))
        }
    }

    private func writeAnalysis(_ name: String, of run: String, in root: URL) throws {
        let directory = root.appending(components: "runs", name)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{\"sourceRun\":\"\(run)\"}".utf8)
            .write(to: directory.appending(component: "analysis.json"))
        try Data("condition,effect\n".utf8)
            .write(to: directory.appending(component: "effect-sizes.csv"))
    }

    private func writeExport(_ name: String, of run: String, in root: URL) throws {
        let directory = root.appending(components: "exports", name)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest = "{\"kind\":\"steerlab.resultsExport\",\"run\":\"runs/\(run)\"}"
        try Data(manifest.utf8).write(to: directory.appending(component: "manifest.json"))
    }

    // MARK: No workspace

    @Test func withNoWorkspaceNothingIsDoneAndOnlyTheWaysInAreOffered() throws {
        let items = FirstStudyChecklist.items(.noWorkspace)
        #expect(items.map(\.step) == FirstStudyChecklist.Step.allCases)
        #expect(done(items).isEmpty)
        #expect(FirstStudyChecklist.nextStep(items) == .workspace)
        #expect(FirstStudyChecklist.progress(items) == "0 of 7 steps done")
        // The ways to get a workspace work without one; nothing else does.
        let available = items.flatMap(\.actions).filter { !$0.needsWorkspace }
        #expect(Set(available) == [.newWorkspace, .openWorkspace, .openDemoWorkspace])
        #expect(try item(.study, items).actions.contains(.openDemoWorkspace))
    }

    // MARK: A new workspace

    @Test func aNewWorkspaceHasItsFolderAndItsPlaceAndNothingElse() throws {
        let root = temporary("new")
        defer { try? fm.removeItem(at: root) }
        try WorkspaceStore.create(at: root)
        try WorkspaceCompute.declare(.macQuickStart, root: root)

        let facts = FirstStudyChecklist.scan(root: root, carriedDemos: Self.demoFixtures)
        #expect(facts.hasWorkspace)
        #expect(facts.computeChoice == .macQuickStart)
        #expect(!facts.isDemoCopy)
        #expect(facts.studies.isEmpty)
        #expect(facts.completedRuns.isEmpty)

        let items = FirstStudyChecklist.items(facts)
        #expect(done(items) == [.workspace, .compute])
        #expect(FirstStudyChecklist.nextStep(items) == .model)
        #expect(try item(.compute, items).detail.contains("This Mac, quick start"))

        // The model count comes from the app's engine; one model is enough.
        var withModel = facts
        withModel.modelCount = 1
        #expect(done(FirstStudyChecklist.items(withModel)) == [.workspace, .compute, .model])
    }

    @Test func aWorkspaceThatDeclaresNoPlaceHasThatStepOpen() throws {
        let root = temporary("undeclared")
        defer { try? fm.removeItem(at: root) }
        try WorkspaceStore.create(at: root)

        let items = FirstStudyChecklist.items(
            FirstStudyChecklist.scan(root: root, carriedDemos: nil))
        #expect(done(items) == [.workspace])
        #expect(try item(.compute, items).actions == [.chooseCompute])
    }

    // MARK: A Demo Workspace copy

    @Test func aDemoCopyHasItsStudiesButNotARunOfYourOwn() throws {
        let target = temporary("demo")
        defer { try? fm.removeItem(at: target) }
        let opened = try DemoWorkspace.open(
            .mlx, at: target, in: Self.demoFixtures, verifyingStudies: false)

        var facts = FirstStudyChecklist.scan(root: opened.root, carriedDemos: Self.demoFixtures)
        #expect(facts.isDemoCopy)
        #expect(facts.demoModelID == "example/placeholder-model")
        #expect(facts.computeChoice == .macQuickStart)
        #expect(facts.studies.map(\.name) == ["placeholder-study", "placeholder-study-draft"])
        #expect(facts.studies.map(\.status) == ["frozen", "draft"])
        // The finished study's run came with the demo: it was not run here.
        #expect(facts.completedRuns.isEmpty)
        #expect(facts.studiesWithAnyRun == ["placeholder-study"])

        let items = FirstStudyChecklist.items(facts)
        #expect(done(items) == [.workspace, .compute, .study])
        #expect(try item(.model, items).detail.contains("example/placeholder-model"))
        // The Run step opens the draft, the study that has not run.
        #expect(
            try item(.run, items).actions == [.openStudies(select: "placeholder-study-draft")])
        #expect(try item(.run, items).detail.contains("demo's draft"))
        #expect(try item(.results, items).detail.contains("demo's finished study"))

        facts.modelCount = 1
        #expect(FirstStudyChecklist.nextStep(FirstStudyChecklist.items(facts)) == .run)
    }

    /// A build that no longer carries the demo a copy came from cannot tell
    /// the demo's runs from the researcher's, so every run counts.
    @Test func withoutTheCarriedDemoEveryRunCounts() throws {
        let target = temporary("demo-uncarried")
        defer { try? fm.removeItem(at: target) }
        let opened = try DemoWorkspace.open(
            .mlx, at: target, in: Self.demoFixtures, verifyingStudies: false)

        let facts = FirstStudyChecklist.scan(root: opened.root, carriedDemos: nil)
        #expect(facts.isDemoCopy)
        #expect(facts.completedRuns.map(\.study) == ["placeholder-study"])
    }

    // MARK: Run, results, and export

    @Test func aRunItsAnalysisAndItsExportEachCompleteTheirStep() throws {
        let target = temporary("demo-run")
        defer { try? fm.removeItem(at: target) }
        let opened = try DemoWorkspace.open(
            .mlx, at: target, in: Self.demoFixtures, verifyingStudies: false)
        let root = opened.root
        func items() -> [FirstStudyChecklist.Item] {
            var facts = FirstStudyChecklist.scan(root: root, carriedDemos: Self.demoFixtures)
            facts.modelCount = 1
            return FirstStudyChecklist.items(facts)
        }

        // An unfinished run (no report yet) is not a completed run.
        let run = "2026-10-05T120000000Z-exp-placeholder-study-draft-run"
        try writeRun(run, study: "placeholder-study-draft", in: root, complete: false)
        #expect(!done(items()).contains(.run))

        try Data("{}".utf8).write(
            to: root.appending(components: "runs", run, "report.json"))
        #expect(done(items()) == [.workspace, .compute, .model, .study, .run])
        #expect(try item(.run, items()).detail.hasPrefix("1 completed run."))

        // An analysis of the demo's own run does not count; one of yours does.
        try writeAnalysis(
            "2026-10-05T120100000Z-exp-placeholder-study-analyze",
            of: "2026-01-01T000000000Z-exp-placeholder-study-run", in: root)
        #expect(!done(items()).contains(.results))
        try writeAnalysis(
            "2026-10-05T120500000Z-exp-placeholder-study-draft-analyze", of: run, in: root)
        #expect(done(items()).contains(.results))

        // Likewise an export: the demo's run does not count, yours does.
        try writeExport(
            "placeholder-study-results-20261005-120900",
            of: "2026-01-01T000000000Z-exp-placeholder-study-run", in: root)
        #expect(!done(items()).contains(.export))
        try writeExport("placeholder-study-draft-results-20261005-121000", of: run, in: root)

        let finished = items()
        #expect(done(finished) == FirstStudyChecklist.Step.allCases)
        #expect(FirstStudyChecklist.nextStep(finished) == nil)
        #expect(FirstStudyChecklist.progress(finished) == "All 7 steps done")
        // Export opens the study whose run is newest.
        #expect(
            try item(.export, finished).actions
                == [.openStudies(select: "placeholder-study-draft")])
    }

    @Test func anOrdinaryWorkspacesRunsAllCountAndEvaluationsAreResults() throws {
        let root = temporary("ordinary-run")
        defer { try? fm.removeItem(at: root) }
        try WorkspaceStore.create(at: root)
        let study = root.appending(components: "experiments", "first-study")
        try fm.createDirectory(at: study, withIntermediateDirectories: true)
        try Data("{\"name\":\"first-study\",\"status\":\"frozen\"}".utf8)
            .write(to: study.appending(component: "experiment.json"))
        let run = "2026-10-05T090000000Z-exp-first-study-run-2"
        try writeRun(run, study: "first-study", in: root)
        // Directories that are not study runs are not counted as runs.
        try fm.createDirectory(
            at: root.appending(components: "runs", "2026-10-05T080000000Z-exp-first-study-sweep"),
            withIntermediateDirectories: true)

        var facts = FirstStudyChecklist.scan(root: root, carriedDemos: Self.demoFixtures)
        #expect(!facts.isDemoCopy)
        #expect(facts.completedRuns == [.init(directoryName: run, study: "first-study")])
        #expect(facts.runsWithResults.isEmpty)

        // A judged evaluation of the run is results to read, too.
        let evaluation = root.appending(
            components: "runs", "2026-10-05T091000000Z-exp-first-study-evaluate")
        try fm.createDirectory(at: evaluation, withIntermediateDirectories: true)
        try Data("{\"sourceRunDirectory\":\"/elsewhere/runs/\(run)/\"}".utf8)
            .write(to: evaluation.appending(component: "judge-report.json"))
        facts = FirstStudyChecklist.scan(root: root, carriedDemos: Self.demoFixtures)
        #expect(facts.runsWithResults == [run])
    }

    // MARK: Names and words

    @Test func runAnalysisAndEvaluationDirectoriesAreToldApart() {
        let run = FirstStudyChecklist.runPattern
        #expect(FirstStudyChecklist.matched("2026-x-exp-a-study-run", run) == "a-study")
        #expect(FirstStudyChecklist.matched("2026-x-exp-a-study-run-3", run) == "a-study")
        #expect(FirstStudyChecklist.matched("2026-x-exp-panel-multi-agent-run", run) == "panel")
        #expect(FirstStudyChecklist.matched("2026-x-exp-my-run-study-run", run) == "my-run-study")
        #expect(FirstStudyChecklist.matched("2026-x-exp-a-study-analyze", run) == nil)
        #expect(FirstStudyChecklist.matched("2026-x-exp-a-study-sweep", run) == nil)
        #expect(
            FirstStudyChecklist.matched(
                "2026-x-exp-a-study-analyze-2", FirstStudyChecklist.analysisPattern) == "a-study")
        #expect(
            FirstStudyChecklist.matched(
                "2026-x-exp-a-study-evaluate-judgment", FirstStudyChecklist.evaluationPattern)
                == "a-study")
    }

    /// The checklist is read by someone with no machine-learning background:
    /// no command, flag, path, or file name, in any state.
    @Test func theWordsArePlainInEveryState() throws {
        var everything = FirstStudyChecklist.Facts(
            hasWorkspace: true, computeChoice: .anotherMachine, modelCount: 2,
            studies: [.init(name: "a", status: "frozen")], isDemoCopy: false,
            completedRuns: [.init(directoryName: "r", study: "a")], runsWithResults: ["r"],
            exportCount: 1)
        var states = [FirstStudyChecklist.Facts.noWorkspace, everything]
        everything.isDemoCopy = true
        everything.completedRuns = []
        everything.runsWithResults = []
        everything.exportCount = 0
        everything.modelCount = 0
        everything.demoModelID = "owner/model"
        states.append(everything)

        var lines = [
            FirstStudyChecklist.title, FirstStudyChecklist.introduction,
            FirstStudyChecklist.finished,
        ]
        for facts in states {
            for item in FirstStudyChecklist.items(facts) {
                lines += [item.title, item.detail] + item.actions.map(\.title)
            }
        }
        let developerText = [
            "steerlab", "STEERLAB_", "--", "runs/", "exports/", ".json", "Python", "CLI",
        ]
        for line in lines {
            let found = developerText.filter { line.contains($0) }
            #expect(found.isEmpty, "\(found) in: \(line)")
        }
    }
}
