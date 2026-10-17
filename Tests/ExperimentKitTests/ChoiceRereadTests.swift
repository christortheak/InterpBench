import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Re-reading a run's stored choices with the current choice parser
/// (`ChoiceReread`, `analyze --reread-choices`).
///
/// Everything is held to the hand-written cross-engine fixture
/// `Tests/Fixtures/cross-engine/choice-reread.json`: the counts a re-read
/// reports and the choiceRate rows this engine's analysis writes with and
/// without the re-read. The Python engine reads the same file
/// (`Server/tests/test_choice_reread.py`) and must produce the same.
///
/// Every test binds an explicit temporary workspace; none touches
/// `rootOverride`.
struct ChoiceRereadTests {

    // MARK: - The fixture

    private struct Fixture: Decodable {
        let parserVersion: String
        let parserVersionNote: String
        let parsingFixtureSHA256: String
        let study: String
        let taskPromptsFile: String
        let taskPromptsText: String
        let expected: Expected
    }

    private struct Expected: Decodable {
        let total: ChoiceReread.Counts
        let conditions: [String: ChoiceReread.Counts]
        let examples: [String: [ChoiceReread.Example]]
        let sentence: String
        let note: String
        let choiceRate: ChoiceRate
    }

    private struct ChoiceRate: Decodable {
        struct Row: Decodable {
            let estimate: Double
            let n: Int
        }
        let stored: Row
        let reread: Row
    }

    private static let directory = CodeResources.compiledCheckoutPath.appending(
        components: "Tests", "Fixtures", "cross-engine")

    private func fixture() throws -> Fixture {
        try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: Self.directory.appending(component: "choice-reread.json")))
    }

    /// The fixture's records as generations.jsonl lines, null values kept.
    private func generationLines(mutating change: (inout [[String: Any]]) -> Void = { _ in })
        throws -> String
    {
        let data = try Data(
            contentsOf: Self.directory.appending(component: "choice-reread.json"))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var records = try #require(object["records"] as? [[String: Any]])
        change(&records)
        return try records.map {
            String(
                decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]),
                as: UTF8.self)
        }.joined(separator: "\n") + "\n"
    }

    private func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - A workspace holding one completed run

    private struct Workspace {
        let root: URL
        let run: URL
        let manifest: ExperimentManifest
    }

    private static let runName = "20261017T000000000Z-exp-reread-run"

    private func workspace(
        inWorkspace: Bool = true, pinnedAs pinned: String? = nil,
        exclusionRules: [ExclusionRule]? = nil,
        records change: (inout [[String: Any]]) -> Void = { _ in }
    ) throws -> Workspace {
        let value = try fixture()
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(component: "choice-reread-\(UUID())")
        var manifest = ExperimentManifest(
            name: value.study, description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.taskPromptsFile = value.taskPromptsFile
        manifest.taskPromptsHash = digest(value.taskPromptsText)
        manifest.exclusionRules = exclusionRules
        if inWorkspace {
            let file = root.appending(path: value.taskPromptsFile)
            try fm.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(value.taskPromptsText.utf8).write(to: file)
        }
        if let pinned {
            let file = root.appending(components: "experiments", value.study, "pinned")
                .appending(path: pinned)
            try fm.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(value.taskPromptsText.utf8).write(to: file)
        }
        let run = root.appending(path: "runs/\(Self.runName)")
        try fm.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        try Data(try generationLines(mutating: change).utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
        return Workspace(root: root, run: run, manifest: manifest)
    }

    private func runBytes(_ run: URL) throws -> [String: String] {
        var found: [String: String] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: run.path) {
            let data = try Data(contentsOf: run.appending(component: name))
            found[name] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        return found
    }

    private func analyze(_ space: Workspace, reread: Bool) throws -> URL {
        try StudyAnalysisWorkflow.analyze(
            manifest: space.manifest,
            repository: StudyAnalysisRepository(workspaceRoot: space.root, promptRoot: space.root),
            allowUnverifiedEpoch: false, rereadChoices: reread)
    }

    private func pooled(_ out: URL, metric: String) throws -> ExperimentTasks.EffectSizeEntry {
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: out.appending(component: "analysis.json")))
        let rows = report.effectSizes.filter {
            $0.metric == metric && $0.condition == "steered" && $0.stratifyBy == nil
        }
        #expect(rows.count == 1, "expected one pooled \(metric) row, found \(rows.count)")
        return try #require(rows.first)
    }

    // MARK: - The parser version

    /// The constant names the parsing rules; when the parsing fixture's bytes
    /// change the rules have changed, and the version must move with them.
    @Test func theParserVersionIsTheFixturesAndMovesWithTheRules() throws {
        let value = try fixture()
        #expect(Judicial.choiceParserVersion == value.parserVersion)
        let parsing = try Data(
            contentsOf: Self.directory.appending(component: "choice-parsing.json"))
        let current = SHA256.hash(data: parsing).map { String(format: "%02x", $0) }.joined()
        #expect(current == value.parsingFixtureSHA256, "\(value.parserVersionNote)")
    }

    // MARK: - The re-read

    @Test func theFixtureRecordsRereadToTheHandWrittenCounts() throws {
        let value = try fixture()
        let options = Dictionary(
            uniqueKeysWithValues: try StudyPromptParsing.parseTaskPrompts(
                Data(value.taskPromptsText.utf8)
            ).compactMap { prompt in prompt.options.map { (prompt.id, $0) } })
        let result = try ChoiceReread.reread(
            generations: try generationLines(), options: options, promptsHash: nil)
        #expect(result.total == value.expected.total)
        #expect(result.conditions == value.expected.conditions)
        #expect(result.examples == value.expected.examples)
        #expect(result.fresh.keys.sorted() == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])
    }

    @Test func theSentenceIsBuiltFromTheStampedNumbers() throws {
        let value = try fixture()
        #expect(
            ChoiceReread.sentence(parserVersion: value.parserVersion, total: value.expected.total)
                == value.expected.sentence)
    }

    // MARK: - Analysis, with and without the re-read

    @Test func anOrdinaryAnalysisUsesTheStoredChoices() throws {
        let value = try fixture()
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        let before = try runBytes(space.run)
        let out = try analyze(space, reread: false)
        let row = try pooled(out, metric: "choiceRate")
        #expect(abs(row.meanDiff - value.expected.choiceRate.stored.estimate) < 1e-9)
        #expect(row.n == value.expected.choiceRate.stored.n)
        #expect(
            !FileManager.default.fileExists(
                atPath: out.appending(component: ChoiceReread.stampFile).path))
        #expect(try runBytes(space.run) == before)
    }

    @Test func rereadChoicesUsesTheCurrentReadingAndStampsIt() throws {
        let value = try fixture()
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        let before = try runBytes(space.run)
        let out = try analyze(space, reread: true)
        let row = try pooled(out, metric: "choiceRate")
        #expect(abs(row.meanDiff - value.expected.choiceRate.reread.estimate) < 1e-9)
        #expect(row.n == value.expected.choiceRate.reread.n)
        let stamp = try #require(ChoiceReread.stamp(inRunAt: out))
        #expect(stamp.parserVersion == value.parserVersion)
        #expect(stamp.sourceRun == Self.runName)
        #expect(stamp.runUnchanged)
        #expect(stamp.total == value.expected.total)
        #expect(stamp.conditions == value.expected.conditions)
        #expect(stamp.examples == value.expected.examples)
        #expect(stamp.taskPrompts.sha256 == digest(value.taskPromptsText))
        #expect(stamp.taskPrompts.location == "workspace")
        #expect(stamp.note == value.expected.note)
        // Written with explicit nulls, as the Python engine writes them.
        let text = try String(
            contentsOf: out.appending(component: ChoiceReread.stampFile), encoding: .utf8)
        #expect(text.contains("\"reread\" : null"))
        #expect(text.contains("\"stored\" : null"))
        #expect(try runBytes(space.run) == before)
    }

    /// An unparseableEndpoint rule on parsedChoice reads the re-read value.
    /// Word-count differences per item (steered minus baseline): q1 5, q2 5,
    /// q3 5, q4 4, q5 0, q6 2, q7 -1. Stored, q4 (no answer) is excluded:
    /// 16 over 6 items. Re-read, q3 is excluded instead: 15 over 6 items.
    @Test func rereadChoicesReachesTheExclusions() throws {
        let space = try workspace(
            exclusionRules: [ExclusionRule(rule: "unparseableEndpoint", endpoint: "parsedChoice")])
        defer { try? FileManager.default.removeItem(at: space.root) }
        let stored = try pooled(try analyze(space, reread: false), metric: "wordCount")
        #expect(abs(stored.meanDiff - 16.0 / 6.0) < 1e-9)
        #expect(stored.n == 6)
        let reread = try pooled(try analyze(space, reread: true), metric: "wordCount")
        #expect(abs(reread.meanDiff - 15.0 / 6.0) < 1e-9)
        #expect(reread.n == 6)
    }

    // MARK: - Finding the options

    @Test(arguments: ["prompts/tasks/reread.jsonl", "task-prompts-reread.jsonl"])
    func aMatchingPinnedCopyIsUsedWhenTheWorkspaceFileIsGone(pinned: String) throws {
        let value = try fixture()
        let space = try workspace(inWorkspace: false, pinnedAs: pinned)
        defer { try? FileManager.default.removeItem(at: space.root) }
        let stamp = try #require(ChoiceReread.stamp(inRunAt: try analyze(space, reread: true)))
        #expect(stamp.taskPrompts.location == "pinned")
        #expect(stamp.taskPrompts.readFrom == "experiments/\(value.study)/pinned/\(pinned)")
        #expect(stamp.total == value.expected.total)
    }

    @Test func aChangedFileIsNeverUsed() throws {
        let value = try fixture()
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        try Data(
            value.taskPromptsText.replacingOccurrences(
                of: "\"target\": \"B\"", with: "\"target\": \"A\"").utf8
        ).write(to: space.root.appending(path: value.taskPromptsFile))
        do {
            _ = try ChoiceReread.forAnalysis(
                runDirectory: space.run, generations: try generationLines(),
                workspaceRoot: space.root, study: value.study)
            Issue.record("a file with other bytes was used")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .pinDrift)
            #expect(error.lifecycleRefusal?.repairAction.contains("experiments/reread/pinned/") == true)
        }
    }

    @Test func recordsFromAnotherPromptsFileAreRefused() throws {
        let value = try fixture()
        let space = try workspace { records in records[0]["promptID"] = "elsewhere" }
        defer { try? FileManager.default.removeItem(at: space.root) }
        do {
            _ = try ChoiceReread.forAnalysis(
                runDirectory: space.run,
                generations: try String(
                    contentsOf: space.run.appending(component: "generations.jsonl"),
                    encoding: .utf8),
                workspaceRoot: space.root, study: value.study)
            Issue.record("a record from another prompts file was re-read")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .pinDrift)
            #expect(error.reason.contains("'elsewhere'"))
        }
    }

    // MARK: - The advisory, and the check through the bridge

    @Test func theAdvisoryCountsWhatWouldReadDifferently() throws {
        let value = try fixture()
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        let found = try #require(
            ChoiceReread.advisoryCount(
                runDirectory: space.run, workspaceRoot: space.root, study: value.study))
        #expect(found.changed == 5)
        #expect(found.storedChoices == 12)
        let detail = ChoiceReread.advisoryDetail(
            changed: found.changed, storedChoices: found.storedChoices, run: Self.runName,
            checkCommand: "C", analyzeCommand: "A")
        #expect(detail.hasPrefix("5 of 12 stored choices in run '\(Self.runName)'"))
        #expect(CLIAdvisory.storedChoicesReadDifferently.rawValue == "storedChoicesReadDifferently")
    }

    @Test func theAdvisoryNeverAnswersForAnUnverifiableRun() throws {
        let value = try fixture()
        let space = try workspace(inWorkspace: false)
        defer { try? FileManager.default.removeItem(at: space.root) }
        #expect(
            ChoiceReread.advisoryCount(
                runDirectory: space.run, workspaceRoot: space.root, study: value.study) == nil)
    }

    @Test func theCheckAnswerIsReadAndARefusalArrivesAsData() throws {
        let answer: JSONValue = .object([
            "checked": .bool(true), "run": .string("runs/\(Self.runName)"),
            "parserVersion": .string("2026-10-17"),
            "total": .object(["storedChoices": .number(12), "changed": .number(5)]),
            "conditions": .object(["steered": .object(["changed": .number(5)])]),
        ])
        let report = try ChoiceCheck.report(from: answer)
        #expect(report.total.storedChoices == 12)
        #expect(report.conditions["steered"]?.changed == 5)
        #expect(ChoiceCheck.message(study: "reread", report: report).contains("--reread-choices"))
        let refused: JSONValue = .object([
            "checked": .bool(false),
            "refusal": .object([
                "code": .string("pinDrift"), "reason": .string("no copy"),
                "repairAction": .string("restore it"), "state": .string("refused"),
            ]),
        ])
        #expect(throws: ResultsExport.Refusal.self) { try ChoiceCheck.report(from: refused) }
        #expect(DiagnosticWorkspace.actions.contains(ChoiceCheck.action))
    }

    /// The Mac command line's `results check-choices` reaches the Python
    /// client's check through the real bridge, over a run this engine wrote,
    /// and answers with the fixture's counts; a refusal arrives typed. Needs
    /// `STEERLAB_TEST_PYTHON`, like every other bridge test.
    @Test func theMacVerbAnswersThroughTheBridge() async throws {
        let value = try fixture()
        let python = URL(
            filePath: try #require(
                ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
                "Set STEERLAB_TEST_PYTHON to run the real Python client."))
        let source = CodeResources.compiledCheckoutPath.appending(component: "Server")
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        let before = try runBytes(space.run)
        let result = try await ChoiceCheckCLI.run(
            try ExperimentCLIParser.parse(
                namespace: "results",
                ["check-choices", value.study, "--run", "runs/\(Self.runName)"]),
            sink: .discarding, workspaceRoot: space.root, python: python, source: source)
        #expect(!result.changed)
        guard case .object(let total)? = result.payload["total"] else {
            Issue.record("no total in \(result.payload)")
            return
        }
        #expect(total["changed"] == .number(5))
        #expect(total["storedChoices"] == .number(12))
        #expect(total["noFinishReason"] == .number(2))
        #expect(result.message.contains("steerlab-cli experiment analyze reread --reread-choices"))
        #expect(try runBytes(space.run) == before)

        let missing = try workspace(inWorkspace: false)
        defer { try? FileManager.default.removeItem(at: missing.root) }
        do {
            _ = try await ChoiceCheckCLI.run(
                try ExperimentCLIParser.parse(namespace: "results", ["check-choices", value.study]),
                sink: .discarding, workspaceRoot: missing.root, python: python, source: source)
            Issue.record("a run whose options cannot be verified was checked")
        } catch let stop as ExperimentCLIStop {
            #expect(stop.code == "pinDrift")
            #expect(stop.repairAction.contains("pinned"))
        }
    }

    // MARK: - The app

    /// The app's results read the stamp as stored, locally and from a remote
    /// listing, and say the engines' sentence; an ordinary analysis shows none.
    @Test func theAppsResultsShowTheStampAsStored() throws {
        let value = try fixture()
        let space = try workspace()
        defer { try? FileManager.default.removeItem(at: space.root) }
        let out = try analyze(space, reread: true)
        let stamp = try #require(RunResults.load(runDirectory: out).choiceReread)
        #expect(ChoiceReread.sentence(stamp) == value.expected.sentence)
        #expect(RunResults.ArtifactBytes.fileNames.contains(ChoiceReread.stampFile))
        var artifacts = RunResults.ArtifactBytes()
        artifacts.assign(
            name: ChoiceReread.stampFile,
            data: try Data(contentsOf: out.appending(component: ChoiceReread.stampFile)))
        #expect(RunResults.remoteModel(runID: "remote", artifacts: artifacts).choiceReread == stamp)
        let ordinary = try analyze(space, reread: false)
        #expect(RunResults.load(runDirectory: ordinary).choiceReread == nil)
    }

    @Test func theAppSaysWhatTheControlDoesAndWhenItWouldMatter() {
        #expect(ChoiceReread.appToggleTitle == "Re-read choices with the current parser")
        #expect(ChoiceReread.appToggleHelp.contains("The run itself is never changed."))
        #expect(
            ChoiceReread.appAdvice(changed: 5, storedChoices: 12)
                == "5 of 12 stored choices would be read differently by the current "
                + "choice parser; this analysis used the choices as the run stored "
                + "them. To use the current reading, turn on \"Re-read choices with "
                + "the current parser\" and analyze again.")
    }

    @Test func theFlagsAreDeclared() throws {
        let analyze = try #require(
            ExperimentCLIParser.specs.first { $0.namespace == "experiment" && $0.verb == "analyze" })
        #expect(analyze.booleanFlags.contains("--reread-choices"))
        #expect(
            ExperimentCLIParser.specs.contains {
                $0.namespace == "results" && $0.verb == "check-choices"
            })
    }
}
