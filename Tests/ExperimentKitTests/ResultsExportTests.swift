import Foundation
import Testing

@testable import ExperimentKit

/// `results export` from the Mac side: the bridge call, the command-line verb,
/// and when the app's "Export Results…" button is available.
///
/// The export itself is the Python client's and is tested there
/// (`Server/tests/test_results_export.py`). What is held here is that the Mac
/// side reaches that one implementation, that a real Mac analysis comes out
/// under the shared column names with its numbers untouched, and that a
/// refusal arrives typed, in the Mac command line's own words.
///
/// The tests that start the Python client need `STEERLAB_TEST_PYTHON`, like
/// every other bridge test. Every workspace here is a temporary directory.
@Suite(.serialized) struct ResultsExportTests {

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
            component: "results-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // Compare resolved paths: the temporary directory is reached through
        // a symbolic link on macOS.
        return root.resolvingSymlinksInPath()
    }

    // MARK: A Mac-engine run, analyzed by the Mac engine

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
        let markerDensity: [String: Double]
    }

    private func manifest() -> ExperimentManifest {
        ExperimentManifest(
            name: Self.study, description: "Does a formal register change the answers?",
            modelID: "test/model", createdAt: Date(timeIntervalSince1970: 0))
    }

    /// A completed run written the way the Mac engine writes one, then
    /// analyzed by the Mac engine's own `analyze` workflow. Returns the
    /// analysis directory.
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
                let words = 6 + index + (condition == "formal" ? 3 + index : 0)
                let record = Record(
                    experiment: Self.study, modelID: "test/model", condition: condition,
                    seed: 0, promptIndex: index, promptID: item,
                    prompt: "Describe a quiet weekend (\(item)).",
                    output: "Dear reader, \"\(item)\".\nA second line.", wordCount: words,
                    distinct2: condition == "formal" ? 0.75 : 0.5,
                    markerDensity: ["formality": condition == "formal" ? 0.25 : 0.0])
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

    /// Every file under a directory, with its bytes.
    private func tree(_ directory: URL) throws -> [String: Data] {
        var found: [String: Data] = [:]
        let enumerator = try #require(
            FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]))
        for case let url as URL in enumerator {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                continue
            }
            found[url.path.replacingOccurrences(of: directory.path + "/", with: "")] =
                try Data(contentsOf: url)
        }
        return found
    }

    /// A CSV with no quoted line breaks, which is what the export writes:
    /// the header, and each row as cells.
    private func table(_ url: URL) throws -> (header: [String], rows: [[String: String]]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        func cells(_ line: Substring) -> [String] {
            var cells: [String] = []
            var cell = ""
            var quoted = false
            var index = line.startIndex
            while index < line.endIndex {
                let character = line[index]
                let next = line.index(after: index)
                if quoted {
                    if character == "\"" {
                        if next < line.endIndex, line[next] == "\"" {
                            cell.append("\"")
                            index = line.index(after: next)
                            continue
                        }
                        quoted = false
                    } else {
                        cell.append(character)
                    }
                } else if character == "\"" {
                    quoted = true
                } else if character == "," {
                    cells.append(cell)
                    cell = ""
                } else {
                    cell.append(character)
                }
                index = next
            }
            cells.append(cell)
            return cells
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        let header = cells(try #require(lines.first))
        let rows = lines.dropFirst().map { line in
            Dictionary(uniqueKeysWithValues: zip(header, cells(line)))
        }
        return (header, rows)
    }

    // MARK: - The bridge call

    @Test func theBridgeCarriesTheExportAction() {
        #expect(ResultsExport.action == "results-export")
        #expect(DiagnosticWorkspace.actions.contains(ResultsExport.action))
    }

    @Test func theRequestCarriesOnlyWhatWasAsked() {
        let root = URL(filePath: "/tmp/example-workspace")
        #expect(
            ResultsExport.payload(workspaceRoot: root, study: "tone-study", client: .app)
                == [
                    "workspaceRoot": .string("/tmp/example-workspace"),
                    "study": .string("tone-study"), "client": .string("app"),
                ])
        // A named run and destination travel as given; empty ones are left
        // out, so the Python client applies its own defaults.
        #expect(
            ResultsExport.payload(
                workspaceRoot: root, study: "tone-study", run: "runs/older", out: "/tmp/out",
                client: .commandLine)
                == [
                    "workspaceRoot": .string("/tmp/example-workspace"),
                    "study": .string("tone-study"), "client": .string("steerlab-cli"),
                    "run": .string("runs/older"), "out": .string("/tmp/out"),
                ])
        #expect(
            ResultsExport.payload(
                workspaceRoot: root, study: "tone-study", run: "", out: "", client: .app
            ).keys.sorted() == ["client", "study", "workspaceRoot"])
    }

    @Test func aRealMacAnalysisExportsThroughTheBridgeUnderTheSharedColumns() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let analysis = try writeAnalyzedRun(in: root)
        let stored = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: analysis.appending(component: "analysis.json")))
        let before = try tree(root.appending(component: "runs"))

        let outcome = try await ResultsExport.export(
            workspaceRoot: root, study: Self.study, client: .commandLine, python: python,
            source: payloadSource)

        #expect(outcome.run == "runs/\(Self.runName)")
        #expect(outcome.analysis == "runs/\(analysis.lastPathComponent)")
        #expect(outcome.directory.deletingLastPathComponent().path == root.appending(component: "exports").path)
        #expect(
            Set(outcome.files.map(\.file)).isSuperset(of: [
                "responses.csv", "effects.csv", "methods.md", "codebook.md", "manifest.json",
            ]))
        #expect(outcome.files.first { $0.file == "responses.csv" }?.rows == 8)

        // One set of column names, whichever engine made the run: these are
        // the names the Python suite pins for a Python-engine run.
        let effects = try table(outcome.directory.appending(component: "effects.csv"))
        #expect(
            effects.header == [
                "study", "run", "analysis", "outcome", "condition", "estimate", "ci_lower",
                "ci_upper", "n_pairs", "unit_of_analysis", "unit_of_analysis_source", "test",
                "test_statistic", "p_value", "adjusted_p_value", "correction", "modality",
            ])
        // The Mac engine's numbers, copied and not recalculated.
        let pooled = stored.effectSizes.filter { $0.stratifyBy == nil }
        #expect(!pooled.isEmpty)
        #expect(effects.rows.count == pooled.count)
        for (row, entry) in zip(effects.rows, pooled) {
            #expect(row["outcome"] == entry.metric)
            #expect(row["condition"] == entry.condition)
            #expect(row["estimate"] == String(entry.meanDiff))
            #expect(row["ci_lower"] == String(entry.ciLower))
            #expect(row["ci_upper"] == String(entry.ciUpper))
            #expect(row["n_pairs"] == String(entry.n))
            #expect(row["p_value"] == (entry.wilcoxonP.map { String($0) } ?? ""))
            #expect(row["adjusted_p_value"] == (entry.adjustedP.map { String($0) } ?? ""))
            #expect(row["correction"] == (entry.correction ?? ""))
            #expect(row["analysis"] == analysis.lastPathComponent)
        }
        #expect(effects.rows.contains { $0["outcome"] == "wordCount" && $0["n_pairs"] == "4" })

        let responses = try table(outcome.directory.appending(component: "responses.csv"))
        #expect(responses.rows.count == 8)
        #expect(responses.rows.first?["marker_density_formality"] == "0")
        #expect(responses.rows.last?["marker_density_formality"] == "0.25")
        #expect(responses.rows.first?["response"] == "Dear reader, \"item-1\". ¶ A second line.")

        // The run and its analysis are exactly as they were.
        #expect(try tree(root.appending(component: "runs")) == before)
    }

    // MARK: - The command-line verb

    @Test func theVerbIsDeclaredAndOwnsItsOutFlag() throws {
        #expect(ExperimentCLIRunner.namespaces.contains("results"))
        let spec = try #require(ExperimentCLIParser.spec(namespace: "results", verb: "export"))
        #expect(spec.valueFlags == ["--run", "--out"])
        #expect(spec.ownsOutFlag)
        #expect(
            ExperimentCLIHelp.synopsis(spec, includeSharedFlags: false)
                == "steerlab-cli results export <study> [--out <dir>] [--run <run-dir>]")
        // `--out` is the export's folder here, never the envelope's file.
        let invocation = try ExperimentCLIParser.parse(
            namespace: "results", ["export", "tone-study", "--out", "/tmp/elsewhere", "--json"])
        #expect(invocation.outPath == nil && invocation.json)
        #expect(invocation.args == ["export", "tone-study", "--out", "/tmp/elsewhere"])
        #expect(throws: ExperimentCLIUsageError.self) {
            try ExperimentCLIParser.parse(namespace: "results", ["export", "tone-study", "--force"])
        }
        #expect(ExperimentCLIHelp.topLevelText.contains("results export <study>"))
        // Exporting reads a workspace, so with none resolved it is refused.
        #expect(ExperimentCLIRunner.needsWorkspace(namespace: "results", verb: "export"))
    }

    @Test func aMissingStudyOrVerbIsAUsageErrorBeforeAnythingRuns() async throws {
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        for arguments in [["export"], ["export", "one", "two"]] {
            let invocation = try ExperimentCLIParser.parse(namespace: "results", arguments)
            do {
                _ = try await ResultsExportCLI.run(
                    invocation, sink: .discarding, workspaceRoot: root)
                Issue.record("expected a usage refusal for \(arguments)")
            } catch let error as ExperimentError {
                #expect(error.reason == "Name one study to export.")
                #expect(error.malformedInvocation?.repairAction.hasPrefix(ResultsExportCLI.usage) == true)
            }
        }
        let unknown = ExperimentCLIInvocation(namespace: "results", verb: nil, args: ["import"])
        do {
            _ = try await ResultsExportCLI.run(unknown, sink: .discarding, workspaceRoot: root)
            Issue.record("expected the verb roster")
        } catch let error as ExperimentError {
            #expect(ExperimentCLIRunner.usageShape(of: error.reason) == "unknownVerb")
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    @Test func theVerbExportsAndAnswersWithTheFolder() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeAnalyzedRun(in: root)
        let destination = root.appending(component: "for-my-paper")
        let recorder = ExperimentCLIRecorder()

        let result = try await ResultsExportCLI.run(
            try ExperimentCLIParser.parse(
                namespace: "results",
                ["export", Self.study, "--run", "runs/\(Self.runName)", "--out", destination.path]),
            sink: recorder.sink, workspaceRoot: root, python: python, source: payloadSource)

        #expect(result.changed)
        #expect(result.payload["exportDirectory"] == .string(destination.path))
        #expect(result.payload["run"] == .string("runs/\(Self.runName)"))
        #expect(result.message.hasPrefix("Results exported to \(destination.path)."))
        #expect(result.message.hasSuffix("Nothing under runs/ was changed, and no statistic was recalculated."))
        #expect(FileManager.default.fileExists(atPath: destination.appending(component: "methods.md").path))
        #expect(recorder.standardOutput.contains("responses.csv  (8 rows)"))
        let methods = try String(contentsOf: destination.appending(component: "methods.md"), encoding: .utf8)
        #expect(methods.contains("# Methods summary: \(Self.study)"))
        #expect(methods.contains("- Model: `test/model`"))
    }

    @Test func aRefusalArrivesTypedAndInTheMacCommandLinesWords() async throws {
        let python = try testPython()
        let root = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let study = root.appending(path: "experiments/\(Self.study)")
        try FileManager.default.createDirectory(at: study, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest()).write(to: study.appending(component: "experiment.json"))
        let recorder = ExperimentCLIRecorder()

        do {
            _ = try await ResultsExportCLI.run(
                try ExperimentCLIParser.parse(namespace: "results", ["export", Self.study]),
                sink: recorder.sink, workspaceRoot: root, python: python, source: payloadSource)
            Issue.record("expected a refusal: the study has no run")
        } catch let stop as ExperimentCLIStop {
            #expect(stop.state == .refused)
            #expect(stop.code == "noCompletedRun")
            #expect(stop.reason.hasPrefix("Study '\(Self.study)' has no completed run in this workspace yet"))
            #expect(stop.repairAction.hasPrefix("steerlab-cli experiment run \(Self.study), then steerlab-cli results export \(Self.study)."))
            #expect(!stop.repairAction.contains("--runner"))
            #expect(!stop.changed)
        }
        // A person at a terminal reads the reason and the repair, once.
        #expect(recorder.standardError.contains("steerlab-cli results export: Study '\(Self.study)' has no completed run"))
        #expect(recorder.standardError.contains("\n  steerlab-cli experiment run \(Self.study)"))
        #expect(!FileManager.default.fileExists(atPath: root.appending(component: "exports").path))

        // A study that does not exist is not found, which is a different answer.
        await #expect(throws: ResultsExport.Refusal.self) {
            try await ResultsExport.export(
                workspaceRoot: root, study: "no-such-study", client: .commandLine,
                python: python, source: payloadSource)
        }
        do {
            _ = try await ResultsExport.export(
                workspaceRoot: root, study: "no-such-study", client: .app, python: python,
                source: payloadSource)
        } catch let refusal as ResultsExport.Refusal {
            #expect(refusal.state == .notFound && refusal.code == "notFound")
            #expect(refusal.repairAction == "Choose a study on the Studies page.")
        }
    }

    @Test func anAnswerIsReadIntoAnOutcomeOrARefusal() throws {
        let outcome = try ResultsExport.outcome(
            from: .object([
                "exported": .bool(true), "exportDirectory": .string("/tmp/exports/one"),
                "run": .string("runs/a-run"), "analysis": .null, "freezeForced": .bool(true),
                "files": .array([
                    .object(["file": .string("responses.csv"), "rows": .number(12)]),
                    .object(["file": .string("methods.md")]),
                ]),
                "notAvailable": .array([
                    .object([
                        "what": .string("effects.csv"), "why": .string("no analysis was found"),
                        "repair": .string("steerlab-cli experiment analyze a"),
                    ])
                ]),
            ]))
        #expect(outcome.directory.path == "/tmp/exports/one")
        #expect(outcome.analysis == nil && outcome.freezeForced)
        #expect(
            outcome.files == [
                .init(file: "responses.csv", rows: 12), .init(file: "methods.md", rows: nil),
            ])
        #expect(outcome.notAvailable.first?.repair == "steerlab-cli experiment analyze a")
        #expect(outcome.message.contains("Not available: effects.csv."))
        #expect(outcome.summaryLines.contains("  responses.csv  (12 rows)"))
        #expect(outcome.summaryLines.last == "  note: this study was frozen with force; methods.md says so.")

        #expect(
            throws: ResultsExport.Refusal(
                code: "exportDestinationRefused", reason: "The folder already contains files.",
                repairAction: "Choose a new or empty folder.", state: .refused)
        ) {
            try ResultsExport.outcome(
                from: .object([
                    "exported": .bool(false),
                    "refusal": .object([
                        "code": .string("exportDestinationRefused"),
                        "reason": .string("The folder already contains files."),
                        "repairAction": .string("Choose a new or empty folder."),
                        "state": .string("refused"),
                    ]),
                ]))
        }
        #expect(throws: ExperimentError.self) { try ResultsExport.outcome(from: .object([:])) }
    }

    // MARK: - The app's button

    private func item(
        _ name: String, kind: StudyRunListItem.Kind = .run, generations: Int = 8,
        hasReport: Bool = true
    ) -> StudyRunListItem {
        StudyRunListItem(
            directoryName: name, path: "/tmp/runs/\(name)", kind: kind,
            createdAt: String(name.prefix(24)), generationCount: generations,
            hasReport: hasReport)
    }

    @Test func theButtonIsAvailableOnlyWhenTheStudyHasACompletedRun() {
        let completed = item("20261004T000000000-exp-tone-study-run")
        let unfinished = item("20261005T000000000-exp-tone-study-run", hasReport: false)
        let empty = item("20261006T000000000-exp-tone-study-run", generations: 0)
        let validation = item("20261007T000000000-exp-tone-study-validate", kind: .validate)

        #expect(ResultsExport.canExport(runs: [completed], isExporting: false))
        #expect(ResultsExport.canExport(runs: [validation, unfinished, completed], isExporting: false))
        #expect(ResultsExport.unavailableReason(runs: [completed], isExporting: false) == nil)

        // Nothing to export: no run at all, or none that finished.
        for runs in [[], [unfinished], [empty], [validation], [unfinished, empty, validation]] {
            #expect(!ResultsExport.canExport(runs: runs, isExporting: false))
            #expect(
                ResultsExport.unavailableReason(runs: runs, isExporting: false)
                    == "This study has no completed run yet. Run it first; results can be "
                    + "exported once a run has finished.")
        }
        // One export at a time.
        #expect(!ResultsExport.canExport(runs: [completed], isExporting: true))
        #expect(
            ResultsExport.unavailableReason(runs: [completed], isExporting: true)
                == "An export is in progress.")
    }

    @Test func theSelectedRunIsExportedOnlyWhenItIsACompletedStudyRun() {
        let completed = item("20261004T000000000-exp-tone-study-run")
        #expect(ResultsExport.runToExport(selected: completed) == completed)
        // Anything else selected asks for the newest completed run instead.
        #expect(ResultsExport.runToExport(selected: nil) == nil)
        #expect(ResultsExport.runToExport(selected: item("a-validate", kind: .validate)) == nil)
        #expect(ResultsExport.runToExport(selected: item("a-run", hasReport: false)) == nil)
        #expect(ResultsExport.runToExport(selected: item("a-run", generations: 0)) == nil)
    }
}
