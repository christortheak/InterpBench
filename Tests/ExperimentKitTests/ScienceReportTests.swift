import Foundation
import Testing
@testable import ExperimentKit

/// The Mac's side of `science report`: one bridge call to the shared Python
/// owner, and no renderer of its own. The fixture report and its golden page
/// are the Python suite's (`Server/tests/fixtures/science-report`), so both
/// clients are held to the same bytes.
struct ScienceReportTests {
    private var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private var fixtures: URL { repository.appending(path: "Server/tests/fixtures/science-report") }

    private func workspace(run: String = "jlens-assessment-example") throws -> (root: URL, run: URL) {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        let directory = root.appending(path: "runs/" + run)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixtures.appending(component: "assessment-list.json"),
                                         to: directory.appending(component: ScienceReport.reportName))
        try Data("jlens-fit-assess\n".utf8).write(to: directory.appending(component: "COMPLETED"))
        return (root, directory)
    }

    private func names(in directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    private func python() throws -> URL {
        URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
                                   "Set TEST_RUNNER_STEERLAB_TEST_PYTHON before Xcode tests."))
    }

    @Test func theBridgeDrawsThePythonOwnersPageOutsideTheRun() async throws {
        let (root, run) = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let before = try names(in: run)
        let page = try await ScienceReport.page(for: run, root: root, python: python(), source: repository.appending(path: "Server"))
        // The Python owner answers with the fully resolved path (/private/var…); compare resolved forms.
        let expected = root.appending(path: "reports/jlens-assessment-example/" + ScienceReport.pageName).resolvingSymlinksInPath()
        #expect(page.url.resolvingSymlinksInPath().path == expected.path)
        #expect(page.inRunDirectory == false)
        // The same bytes the Python client's own test pins: one implementation.
        #expect(try Data(contentsOf: page.url) == Data(contentsOf: fixtures.appending(component: "assessment-list.html")))
        #expect(try names(in: run) == before)

        let result = try await DiagnosticWorkspace.perform(
            ScienceReport.action, payload: ["workspaceRoot": .string(root.path), "path": .string("runs/jlens-assessment-example")],
            python: python(), source: repository.appending(path: "Server"))
        guard case .object(let object) = result, case .string(let written) = object["htmlPath"] else {
            Issue.record("The bridge returned no page path."); return
        }
        #expect(object["kind"] == .string("jlens-assessment"))
        #expect(URL(filePath: written).resolvingSymlinksInPath().path == expected.path)
        #expect(object["changed"] == .bool(false))  // the second render found the page already current
        #expect(object["sourceReport"] == .string("runs/jlens-assessment-example/assessment-report.json"))
    }

    @Test func aRefusalArrivesWithItsReasonAndRepair() async throws {
        let (root, run) = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let before = try names(in: run)
        do {
            _ = try await ScienceReport.render(path: "runs/jlens-assessment-example", out: "runs/jlens-assessment-example/page.html",
                                               root: root, python: python(), source: repository.appending(path: "Server"))
            Issue.record("A page inside a run folder must be refused.")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("never written inside runs/"))
            #expect(error.malformedInvocation?.repairAction.contains("reports/") == true)
        }
        #expect(try names(in: run) == before)
    }

    @Test func aPageStoredWithTheRunIsOpenedWithoutPython() async throws {
        let (root, run) = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(ScienceReport.hasReport(in: run))
        #expect(ScienceReport.storedPage(in: run) == nil)
        let stored = run.appending(component: ScienceReport.pageName)
        try FileManager.default.copyItem(at: fixtures.appending(component: "assessment-list.html"), to: stored)
        // No interpreter exists at this path, so reaching the bridge would throw.
        let page = try await ScienceReport.page(for: run, root: root, python: URL(filePath: "/nonexistent/python"))
        #expect(page.url.path == stored.path)
        #expect(page.inRunDirectory)
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "reports").path))
    }

    @Test func aRunWithoutAReportHasNoPage() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        let run = root.appending(path: "runs/plain")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // A page beside nothing is not a report's page.
        try Data("<p>stray</p>".utf8).write(to: run.appending(component: ScienceReport.pageName))
        #expect(!ScienceReport.hasReport(in: run))
        #expect(ScienceReport.storedPage(in: run) == nil)
        await #expect(throws: ExperimentError.self) {
            _ = try await ScienceReport.page(for: run, root: root, python: URL(filePath: "/nonexistent/python"))
        }
    }

    @Test func theVerbIsDeclaredAndOwnsItsOutFlag() throws {
        #expect(DiagnosticWorkspace.actions.contains(ScienceReport.action))
        let spec = try #require(ExperimentCLIParser.spec(namespace: "science", verb: "report"))
        #expect(spec.ownsOutFlag)
        #expect(spec.valueFlags == ["--out"])
        #expect(spec.requiredFlags.isEmpty)
        #expect(spec.purpose == ScienceReport.purpose)
        // `--out` reaches the verb as the page's destination; it is not taken as the envelope's file.
        let invocation = try ExperimentCLIParser.parse(namespace: "science", ["report", "runs/example", "--out", "reports/page.html", "--json"])
        #expect(invocation.json)
        #expect(invocation.outPath == nil)
        #expect(invocation.args == ["report", "runs/example", "--out", "reports/page.html"])
        let arguments = try DiagnosticArguments(invocation.args, namespace: "science", takesValue: true)
        #expect(arguments.positional == "runs/example")
        #expect(arguments.flags["--out"] == "reports/page.html")
        // The Python client declares the same sentence.
        let python = try String(contentsOf: repository.appending(path: "Server/steerlab_server/client/science_commands.py"), encoding: .utf8)
        for sentence in ["Turn a stored J-lens assessment report into one self-contained HTML page a person can read.",
                         "A run folder is never written to"] {
            #expect(ScienceReport.purpose.contains(sentence))
            #expect(python.contains(sentence.prefix(40)))
        }
    }

    @Test func aMixedLensListsEveryTextItWasFittedOn() throws {
        let mixed = JLensFitProvenance(
            modelID: "example/tiny-model", revision: nil, revisionKnown: false, dtype: "bfloat16",
            corpus: "sha256:" + String(repeating: "e3", count: 32), promptsFitted: 500, maxSeqLen: 128,
            corpora: [JLensCorpusContribution(corpusSHA256: String(repeating: "c2", count: 32), promptsFitted: 200, rowsConsidered: 210),
                      JLensCorpusContribution(corpusSHA256: String(repeating: "f2", count: 32), promptsFitted: 300, rowsConsidered: nil)])
        #expect(mixed.corpusSummary == "mixed: 2 texts (combined digest sha256:" + String(repeating: "e3", count: 32) + ")")
        #expect(mixed.corpusContributionLines == [
            String(repeating: "c2", count: 32) + ": 200 rows fitted of 210 considered",
            String(repeating: "f2", count: 32) + ": 300 rows fitted of an unrecorded number considered",
        ])
        var single = mixed
        single.corpora = nil
        single.corpus = "sha256:" + String(repeating: "f1", count: 32)
        #expect(single.corpusSummary == single.corpus)
        #expect(single.corpusContributionLines.isEmpty)
        single.corpus = nil
        #expect(single.corpusSummary == nil)
        // The stored record decodes into the same lines.
        let stored = try JSONDecoder().decode(JLensFitProvenance.self, from: Data("""
            {"modelID":"example/tiny-model","corpus":"sha256:abc","promptsFitted":3,
             "corpora":[{"corpusSHA256":"c2","promptsFitted":1,"rowsConsidered":2},{"corpusSHA256":"f2","promptsFitted":2,"rowsConsidered":2}]}
            """.utf8))
        #expect(stored.corpusContributionLines == ["c2: 1 rows fitted of 2 considered", "f2: 2 rows fitted of 2 considered"])
    }
}
