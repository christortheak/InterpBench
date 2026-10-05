import Foundation
import Testing

@testable import ExperimentKit

/// `results report` from the Mac side: the bridge call, the command-line verb,
/// and when the app's "Open Report" button is available.
///
/// The page itself is the Python client's and is tested there
/// (`Server/tests/test_study_results_page.py`). What is held here is that the
/// Mac side reaches that one implementation, that a real Mac analysis comes
/// out as a page outside `runs/` with its numbers shown, and that a refusal
/// arrives typed, in the Mac command line's own words.
///
/// The tests that start the Python client need `STEERLAB_TEST_PYTHON`, like
/// every other bridge test. Every workspace here is a temporary directory.
@Suite(.serialized) struct ResultsReportTests {

    private let repository = URL(filePath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    private var payloadSource: URL { repository.appending(component: "Server") }

    private func testPython() throws -> URL {
        URL(
            filePath: try #require(
                ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
                "Set STEERLAB_TEST_PYTHON to run the real Python client."))
    }

    private func temporaryWorkspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "results-report-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath()
    }

    private static let study = "tone-study"
    private static let runName = "20261004T000000000-exp-tone-study-run"
    private static let items = ["item-1", "item-2", "item-3", "item-4"]

    private struct Record: Encodable {
        let experiment: String
        let modelID: String
        let condition: String
        let seed: UInt64
        let promptIndex: Int
        let promptID: String
        let prompt: String
        let output: String
        let wordCount: Int
        let distinct2: Double
    }

    private func manifest() -> ExperimentManifest {
        ExperimentManifest(
            name: Self.study, description: "Does a formal register change the answers?",
            modelID: "test/model", createdAt: Date(timeIntervalSince1970: 0))
    }

    /// A completed run written the way the Mac engine writes one, analyzed by
    /// the Mac engine's own `analyze` workflow. Returns the analysis directory.
    @discardableResult
    private func writeAnalyzedRun(in root: URL) throws -> URL {
        let manifest = manifest()
        let run = root.appending(path: "runs/\(Self.runName)")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        var lines: [String] = []
        for condition in ["baseline", "formal"] {
            for (index, item) in Self.items.enumerated() {
                let record = Record(
                    experiment: Self.study, modelID: "test/model", condition: condition, seed: 0,
                    promptIndex: index, promptID: item, prompt: "Describe a quiet weekend (\(item)).",
                    output: "Dear reader.", wordCount: 6 + index + (condition == "formal" ? 3 + index : 0),
                    distinct2: condition == "formal" ? 0.75 : 0.5)
                lines.append(String(decoding: try JSONEncoder().encode(record), as: UTF8.self))
            }
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
        return try StudyAnalysisWorkflow.analyze(
            manifest: manifest,
            repository: StudyAnalysisRepository(workspaceRoot: root, promptRoot: root),
            allowUnverifiedEpoch: false)
    }

    private func canonical(_ path: String) -> String {
        URL(filePath: path).resolvingSymlinksInPath().path
    }

    private func tree(_ directory: URL) throws -> [String: Data] {
        var found: [String: Data] = [:]
        let enumerator = try #require(
            FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]))
        for case let url as URL in enumerator {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            found[url.path.replacingOccurrences(of: directory.path + "/", with: "")] = try Data(contentsOf: url)
        }
        return found
    }

    // MARK: - The bridge call

    @Test func theBridgeCarriesTheReportAction() {
        #expect(ResultsReport.action == "results-report")
        #expect(DiagnosticWorkspace.actions.contains(ResultsReport.action))
        // The same request shape as the export, so the Python client checks both alike.
        let root = URL(filePath: "/tmp/example-workspace")
        #expect(
            ResultsReport.payload(workspaceRoot: root, study: "tone-study", run: "runs/a", client: .app)
                == ResultsExport.payload(workspaceRoot: root, study: "tone-study", run: "runs/a", client: .app))
    }

    @Test func aRealMacAnalysisBecomesAPageOutsideRuns() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let analysis = try writeAnalyzedRun(in: root)
        let stored = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: analysis.appending(component: "analysis.json")))
        let before = try tree(root.appending(component: "runs"))

        let written = try await ResultsReport.write(
            workspaceRoot: root, study: Self.study, client: .app, python: python, source: payloadSource)

        // Compared as files, since the temporary directory is reached through
        // a symbolic link and the Python client answers with the resolved path.
        #expect(
            canonical(written.page.path)
                == canonical(root.appending(path: "reports/\(Self.study)/\(Self.runName)/report.html").path))
        #expect(written.run == "runs/\(Self.runName)")
        #expect(written.analysis == "runs/\(analysis.lastPathComponent)")
        #expect(written.changed)
        let page = try String(contentsOf: written.page, encoding: .utf8)
        #expect(page.hasPrefix("<!doctype html>\n<!-- SteerLab report page -->\n"))
        #expect(page.contains("<title>Study results: \(Self.study)</title>"))
        #expect(!page.contains("<script") && !page.contains("http://") && !page.contains("https://"))
        // Every pooled effect the Mac engine stored is on the page, under its own name.
        let pooled = stored.effectSizes.filter { $0.stratifyBy == nil }
        #expect(!pooled.isEmpty)
        for entry in pooled {
            #expect(page.contains("<td><code>\(entry.metric)</code></td>"))
        }
        // The page names the analysis it was drawn from, and no machine path.
        #expect(page.contains(analysis.lastPathComponent))
        #expect(!page.contains(root.path))
        // The run and its analysis are exactly as they were.
        #expect(try tree(root.appending(component: "runs")) == before)

        // Asked again, the same page is already there.
        let again = try await ResultsReport.write(
            workspaceRoot: root, study: Self.study, client: .app, python: python, source: payloadSource)
        #expect(!again.changed && again.page == written.page)
        #expect(again.message.hasPrefix("The results page was already up to date at "))
        #expect(again.sheetPage == ScienceReport.Page(url: written.page, inRunDirectory: false))
    }

    // MARK: - The command-line verb

    @Test func theVerbIsDeclaredAndOwnsItsOutFlag() throws {
        let spec = try #require(ExperimentCLIParser.spec(namespace: "results", verb: "report"))
        #expect(spec.valueFlags == ["--run", "--out"])
        #expect(spec.ownsOutFlag)
        #expect(
            ExperimentCLIHelp.synopsis(spec, includeSharedFlags: false)
                == "steerlab-cli results report <study> [--out <file>] [--run <run-dir>]")
        let invocation = try ExperimentCLIParser.parse(
            namespace: "results", ["report", "tone-study", "--out", "/tmp/page.html", "--json"])
        #expect(invocation.outPath == nil && invocation.json)
        #expect(invocation.args == ["report", "tone-study", "--out", "/tmp/page.html"])
        #expect(ExperimentCLIHelp.topLevelText.contains("results report <study>"))
        #expect(ExperimentCLIRunner.needsWorkspace(namespace: "results", verb: "report"))
    }

    @Test func aMissingStudyIsAUsageErrorBeforeAnythingRuns() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        for arguments in [["report"], ["report", "one", "two"]] {
            do {
                _ = try await ResultsExportCLI.run(
                    try ExperimentCLIParser.parse(namespace: "results", arguments), sink: .discarding,
                    workspaceRoot: root)
                Issue.record("expected a usage refusal for \(arguments)")
            } catch let error as ExperimentError {
                #expect(error.reason == "Name one study to make a results page for.")
                #expect(error.malformedInvocation?.repairAction.hasPrefix(ResultsReportCLI.usage) == true)
            }
        }
        let unknown = ExperimentCLIInvocation(namespace: "results", verb: nil, args: ["import"])
        do {
            _ = try await ResultsExportCLI.run(unknown, sink: .discarding, workspaceRoot: root)
            Issue.record("expected the verb roster")
        } catch let error as ExperimentError {
            #expect(error.reason.hasPrefix("verbs: export | report"))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    @Test func theVerbWritesThePageAndAnswersWithIt() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeAnalyzedRun(in: root)
        let destination = root.appending(component: "for-a-colleague.html")
        let recorder = ExperimentCLIRecorder()

        let result = try await ResultsExportCLI.run(
            try ExperimentCLIParser.parse(
                namespace: "results",
                ["report", Self.study, "--run", "runs/\(Self.runName)", "--out", destination.path]),
            sink: recorder.sink, workspaceRoot: root, python: python, source: payloadSource)

        #expect(result.changed)
        guard case .string(let page) = result.payload["htmlPath"] else {
            Issue.record("the answer names no page")
            return
        }
        #expect(canonical(page) == canonical(destination.path))
        #expect(result.payload["written"] == .bool(true))
        #expect(result.message.hasPrefix("The results page was written to \(page)."))
        #expect(result.message.hasSuffix("Nothing under runs/ was changed, and no statistic was recalculated."))
        #expect(recorder.standardOutput.contains("Results page for \(Self.study): \(page)"))
        #expect(FileManager.default.fileExists(atPath: destination.path))
        #expect(!FileManager.default.fileExists(atPath: root.appending(component: "reports").path))
    }

    @Test func aRefusalArrivesTypedAndInTheMacCommandLinesWords() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeAnalyzedRun(in: root)
        let before = try tree(root.appending(component: "runs"))
        let recorder = ExperimentCLIRecorder()
        let inside = root.appending(path: "runs/\(Self.runName)/report.html")

        do {
            _ = try await ResultsExportCLI.run(
                try ExperimentCLIParser.parse(
                    namespace: "results", ["report", Self.study, "--out", inside.path]),
                sink: recorder.sink, workspaceRoot: root, python: python, source: payloadSource)
            Issue.record("expected a refusal: a page is never written into runs/")
        } catch let stop as ExperimentCLIStop {
            #expect(stop.state == .refused)
            #expect(stop.code == "reportDestinationRefused")
            #expect(stop.reason.hasPrefix("A page is never written inside runs/."))
            #expect(!stop.changed)
        }
        #expect(recorder.standardError.contains("steerlab-cli results report: A page is never written inside runs/."))
        #expect(try tree(root.appending(component: "runs")) == before)

        do {
            _ = try await ResultsReport.write(
                workspaceRoot: root, study: "no-such-study", client: .commandLine, python: python,
                source: payloadSource)
            Issue.record("expected a refusal: no such study")
        } catch let refusal as ResultsExport.Refusal {
            #expect(refusal.state == .notFound && refusal.code == "notFound")
            #expect(refusal.repairAction.hasPrefix("steerlab-cli experiment list"))
        }
    }

    @Test func anAnswerIsReadIntoAPageOrARefusal() throws {
        let written = try ResultsReport.written(
            from: .object([
                "written": .bool(true), "htmlPath": .string("/tmp/reports/s/r/report.html"),
                "run": .string("runs/r"), "analysis": .null, "changed": .bool(true),
                "freezeForced": .bool(true),
                "notAvailable": .array([
                    .object(["what": .string("the bootstrap settings"), "why": .string("not recorded")])
                ]),
            ]))
        #expect(written.page.path == "/tmp/reports/s/r/report.html")
        #expect(written.analysis == nil && written.changed && written.freezeForced)
        #expect(written.message.contains("Not available: the bootstrap settings."))
        #expect(
            throws: ResultsExport.Refusal(
                code: "noCompletedRun", reason: "No run yet.", repairAction: "Run the study.",
                state: .refused)
        ) {
            try ResultsReport.written(
                from: .object([
                    "written": .bool(false),
                    "refusal": .object([
                        "code": .string("noCompletedRun"), "reason": .string("No run yet."),
                        "repairAction": .string("Run the study."), "state": .string("refused"),
                    ]),
                ]))
        }
        #expect(throws: ExperimentError.self) { try ResultsReport.written(from: .object([:])) }
    }

    // MARK: - The app's button

    private func item(
        _ name: String, kind: StudyRunListItem.Kind = .run, generations: Int = 8, hasReport: Bool = true
    ) -> StudyRunListItem {
        StudyRunListItem(
            directoryName: name, path: "/tmp/runs/\(name)", kind: kind,
            createdAt: String(name.prefix(24)), generationCount: generations, hasReport: hasReport)
    }

    @Test func theButtonIsAvailableOnlyWhenTheStudyHasACompletedRun() {
        let completed = item("20261004T000000000-exp-tone-study-run")
        let unfinished = item("20261005T000000000-exp-tone-study-run", hasReport: false)
        #expect(ResultsReport.canOpen(runs: [unfinished, completed], isOpening: false))
        #expect(ResultsReport.unavailableReason(runs: [completed], isOpening: false) == nil)
        #expect(!ResultsReport.canOpen(runs: [unfinished], isOpening: false))
        #expect(
            ResultsReport.unavailableReason(runs: [], isOpening: false)
                == "This study has no completed run yet. Run it first; the results page can be "
                + "opened once a run has finished.")
        #expect(!ResultsReport.canOpen(runs: [completed], isOpening: true))
        #expect(ResultsReport.runToShow(selected: completed) == completed)
        #expect(ResultsReport.runToShow(selected: unfinished) == nil)
    }
}
