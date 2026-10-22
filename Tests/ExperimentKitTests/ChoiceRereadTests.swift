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
        #expect(result.fresh.keys.sorted() == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16])
    }

    /// The fixture's malformed records, given as exact lines: read as the
    /// Python twin reads them (`test_malformed_records_are_read_as_the_swift_twin_reads_them`).
    @Test func malformedRecordsAreReadAsThePythonTwinReadsThem() throws {
        struct EdgeCases: Decodable {
            let options: [String]
            let lines: [String]
            let expected: Expected
            struct Expected: Decodable {
                let total: ChoiceReread.Counts
                let conditions: [String: ChoiceReread.Counts]
                let examples: [String: [ChoiceReread.Example]]
            }
        }
        struct Holder: Decodable { let edgeCases: EdgeCases }
        let cases = try JSONDecoder().decode(
            Holder.self,
            from: Data(contentsOf: Self.directory.appending(component: "choice-reread.json"))
        ).edgeCases
        let result = try ChoiceReread.reread(
            generations: cases.lines.joined(separator: "\n") + "\n",
            options: ["q1": cases.options], promptsHash: nil)
        #expect(result.total == cases.expected.total)
        #expect(result.conditions == cases.expected.conditions)
        #expect(result.examples == cases.expected.examples)
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
        // The analyze envelope's summary says so too, in the engines' words.
        guard case .object(let summary)? = ExperimentCLIRunner.analysisPayload(inRunAt: out)["choiceReread"]
        else {
            Issue.record("no choiceReread block in the analyze payload")
            return
        }
        #expect(summary["sentence"] == .string(value.expected.sentence))
        #expect(summary["changed"] == .number(6))
        #expect(summary["storedChoices"] == .number(14))
        #expect(summary["parserVersion"] == .string(value.parserVersion))
        #expect(summary["stamp"] == .string(ChoiceReread.stampFile))
        #expect(ExperimentCLIRunner.analysisPayload(inRunAt: try analyze(space, reread: false))["choiceReread"] == nil)
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
        #expect(found.changed == 6)
        #expect(found.storedChoices == 14)
        let detail = ChoiceReread.advisoryDetail(
            changed: found.changed, storedChoices: found.storedChoices, run: Self.runName,
            checkCommand: "C", analyzeCommand: "A")
        #expect(detail.hasPrefix("6 of 14 stored choices in run '\(Self.runName)'"))
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
        #expect(total["changed"] == .number(6))
        #expect(total["storedChoices"] == .number(14))
        #expect(total["noFinishReason"] == .number(4))
        #expect(
            result.message.hasSuffix(
                "steerlab-cli experiment analyze reread --run runs/\(Self.runName) --reread-choices"))
        let report = try await ChoiceCheck.check(
            workspaceRoot: space.root, study: value.study, client: .commandLine,
            python: python, source: source)
        #expect(report.examples["changedToNoAnswer"] == value.expected.examples["changedToNoAnswer"])
        let lines = ChoiceCheck.summaryLines(study: value.study, report: report)
        #expect(lines.contains("  e.g. to no answer: line 10, steered, item q3, sample 0: A -> no answer"))
        #expect(lines.contains("  e.g. gained an answer: line 11, steered, item q4, sample 0: no answer -> B"))
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

/// `steerlab-cli experiment analyze` end to end, with and without
/// `--reread-choices`, through the command runner and the workspace it binds.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ChoiceRereadCLITests {

    private struct Fixture: Decodable {
        let study: String
        let taskPromptsFile: String
        let taskPromptsText: String
    }

    private func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "choice-reread-cli-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
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

    @Test func analyzeSaysWhenChoicesReadDifferentlyAndRereadsOnRequest() async throws {
        let directory = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine")
        let data = try Data(contentsOf: directory.appending(component: "choice-reread.json"))
        let value = try JSONDecoder().decode(Fixture.self, from: data)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let records = try #require(object["records"] as? [[String: Any]])
        try await withTempRoot { root in
            let prompts = root.appending(path: value.taskPromptsFile)
            try FileManager.default.createDirectory(
                at: prompts.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(value.taskPromptsText.utf8).write(to: prompts)
            // A study that measures something: one attached concept.
            let concept = root.appending(path: "prompts/concepts/fair")
            try FileManager.default.createDirectory(at: concept, withIntermediateDirectories: true)
            try Data("{\"text\": \"fair\"}\n".utf8).write(to: concept.appending(component: "positive.jsonl"))
            try Data("{\"text\": \"unfair\"}\n".utf8).write(to: concept.appending(component: "negative.jsonl"))
            _ = try ExperimentStore.create(name: value.study, description: "", modelID: "test/model")
            let attached = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", ["attach", value.study, "fair", "--json"])
            #expect(attached.envelope.exitCode == 0, "\(attached.envelope)")
            var manifest = try ExperimentStore.load(name: value.study)
            manifest.taskPromptsFile = value.taskPromptsFile
            manifest.taskPromptsHash = SHA256.hash(data: Data(value.taskPromptsText.utf8))
                .map { String(format: "%02x", $0) }.joined()
            try ExperimentStore.save(manifest)
            let run = root.appending(path: "runs/20261017T000000000Z-exp-\(value.study)-run")
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
            try Data(ExperimentStore.manifestHash(manifest).utf8).write(
                to: run.appending(component: "experiment-hash.txt"))
            let lines = try records.map {
                String(
                    decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]),
                    as: UTF8.self)
            }
            try Data((lines.joined(separator: "\n") + "\n").utf8).write(
                to: run.appending(component: "generations.jsonl"))
            try Data("{}".utf8).write(to: run.appending(component: "report.json"))

            let ordinary = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", ["analyze", value.study, "--json"])
            #expect(ordinary.envelope.exitCode == 0, "\(ordinary.envelope)")
            let advice = (ordinary.envelope.advisories ?? []).filter {
                $0.code == CLIAdvisory.storedChoicesReadDifferently.rawValue
            }
            #expect(advice.count == 1)
            #expect(advice.first?.detail.hasPrefix("6 of 14 stored choices in run '") == true)
            #expect(advice.first?.detail.contains("steerlab-cli results check-choices") == true)

            let reread = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", ["analyze", value.study, "--reread-choices", "--json"])
            #expect(reread.envelope.exitCode == 0, "\(reread.envelope)")
            #expect(
                !(reread.envelope.advisories ?? []).contains {
                    $0.code == CLIAdvisory.storedChoicesReadDifferently.rawValue
                })
            guard let result = reread.envelope.result,
                case .object(let summary)? = result["choiceReread"]
            else {
                Issue.record("no choiceReread summary in \(reread.envelope)")
                return
            }
            #expect(summary["changed"] == .number(6))
        }
    }

    // MARK: - The checked run is the analyzed run (external review F1)

    private static let older = "20261017T000000000Z-exp-reread-run"
    private static let newer = "20261018T000000000Z-exp-reread-run"

    /// The fixture's study in `root`, with one concept attached, and a run
    /// for each `(name, records)`: same manifest, same epoch.
    private func study(in root: URL, runs: [(String, [[String: Any]])]) async throws -> Fixture {
        let directory = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine")
        let value = try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: directory.appending(component: "choice-reread.json")))
        let prompts = root.appending(path: value.taskPromptsFile)
        try FileManager.default.createDirectory(
            at: prompts.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(value.taskPromptsText.utf8).write(to: prompts)
        let concept = root.appending(path: "prompts/concepts/fair")
        try FileManager.default.createDirectory(at: concept, withIntermediateDirectories: true)
        try Data("{\"text\": \"fair\"}\n".utf8).write(to: concept.appending(component: "positive.jsonl"))
        try Data("{\"text\": \"unfair\"}\n".utf8).write(to: concept.appending(component: "negative.jsonl"))
        _ = try ExperimentStore.create(name: value.study, description: "", modelID: "test/model")
        let attached = await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "experiment", ["attach", value.study, "fair", "--json"])
        #expect(attached.envelope.exitCode == 0, "\(attached.envelope)")
        var manifest = try ExperimentStore.load(name: value.study)
        manifest.taskPromptsFile = value.taskPromptsFile
        manifest.taskPromptsHash = SHA256.hash(data: Data(value.taskPromptsText.utf8))
            .map { String(format: "%02x", $0) }.joined()
        try ExperimentStore.save(manifest)
        for (name, records) in runs { try addRun(name, records: records, manifest: manifest, root: root) }
        return value
    }

    private func addRun(
        _ name: String, records: [[String: Any]], manifest: ExperimentManifest, root: URL
    ) throws {
        let run = root.appending(path: "runs/\(name)")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        let lines = try records.map {
            String(
                decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]),
                as: UTF8.self)
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
    }

    /// The fixture's records as stored, and as the current parser reads them:
    /// each example's line holds its re-read value.
    private func records() throws -> (stored: [[String: Any]], current: [[String: Any]]) {
        let data = try Data(
            contentsOf: CodeResources.compiledCheckoutPath.appending(
                components: "Tests", "Fixtures", "cross-engine", "choice-reread.json"))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let stored = try #require(object["records"] as? [[String: Any]])
        let expected = try #require(object["expected"] as? [String: Any])
        let examples = try #require(expected["examples"] as? [String: [[String: Any]]])
        var current = stored
        for entry in examples.values.joined() {
            let line = try #require(entry["line"] as? Int)
            current[line - 1]["parsedChoice"] = entry["reread"] ?? NSNull()
        }
        return (stored, current)
    }

    /// Every file in a run, in name order: compared before and after.
    private func runFiles(_ run: URL) throws -> [Data] {
        try FileManager.default.contentsOfDirectory(atPath: run.path).sorted().map {
            try Data(contentsOf: run.appending(component: $0))
        }
    }

    private func stamp(_ envelope: SteerLabCLIEnvelope) throws -> (source: String, changed: Double) {
        let result = try #require(envelope.result, "\(envelope)")
        guard case .string(let source)? = result["sourceRun"],
            case .object(let summary)? = result["choiceReread"],
            case .number(let changed)? = summary["changed"]
        else { throw ExperimentError(reason: "no source or re-read summary in \(result)") }
        return (source, changed)
    }

    /// Check the older run while it is the only one; a newer run arrives;
    /// follow the check's command exactly as printed. It analyzes the checked
    /// run with its six changes. Without --run, analyze still reads the
    /// newest, which has none. Neither run's bytes change.
    @Test func theCheckedRunIsTheRunItsCommandAnalyzes() async throws {
        let python = URL(
            filePath: try #require(
                ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
                "Set STEERLAB_TEST_PYTHON to run the real Python client."))
        let source = CodeResources.compiledCheckoutPath.appending(component: "Server")
        let (stored, current) = try records()
        try await withTempRoot { root in
            let value = try await study(in: root, runs: [(Self.older, stored)])
            let check = try await ChoiceCheckCLI.run(
                try ExperimentCLIParser.parse(namespace: "results", ["check-choices", value.study]),
                sink: .discarding, workspaceRoot: root, python: python, source: source)
            let command = try #require(
                check.message.components(separatedBy: "current reading: ").last)
            #expect(
                command
                    == "steerlab-cli experiment analyze \(value.study) --run runs/\(Self.older) "
                    + "--reread-choices")

            try addRun(
                Self.newer, records: current,
                manifest: try ExperimentStore.load(name: value.study), root: root)
            let olderRun = root.appending(path: "runs/\(Self.older)")
            let newerRun = root.appending(path: "runs/\(Self.newer)")
            let before = try [olderRun, newerRun].map(runFiles)
            var words = command.split(separator: " ").map(String.init)
            #expect(words.removeFirst() == "steerlab-cli")
            #expect(words.removeFirst() == "experiment")
            let followed = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", words + ["--json"])
            #expect(followed.envelope.exitCode == 0, "\(followed.envelope)")
            let checked = try stamp(followed.envelope)
            #expect(checked.source == Self.older && checked.changed == 6)

            let byDefault = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", ["analyze", value.study, "--reread-choices", "--json"])
            let newest = try stamp(byDefault.envelope)
            #expect(newest.source == Self.newer && newest.changed == 0)
            let after = try [olderRun, newerRun].map(runFiles)
            #expect(after == before)
        }
    }

    /// `--run` names a completed run of this study, or is refused, typed,
    /// with a repair: it is never widened to another run.
    @Test func analyzeRunRefusesWhatIsNotACompletedRunOfTheStudy() async throws {
        let (stored, _) = try records()
        try await withTempRoot { root in
            let value = try await study(in: root, runs: [(Self.older, stored)])
            let unfinished = root.appending(path: "runs/20261019T000000000Z-exp-reread-run")
            try FileManager.default.createDirectory(at: unfinished, withIntermediateDirectories: true)
            var other = try ExperimentStore.load(name: value.study)
            other.name = "another-study"
            try addRun(
                "20261020T000000000Z-exp-another-study-run", records: stored, manifest: other,
                root: root)
            let outside = root.appending(path: "elsewhere")
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            for (reference, words) in [
                ("runs/no-such-run", "not a run directory"),
                (outside.path, "not a run directory"),
                ("runs/20261019T000000000Z-exp-reread-run", "not a completed study run"),
                ("runs/20261020T000000000Z-exp-another-study-run", "a run of the study 'another-study'"),
            ] {
                let result = await ExperimentCLIRunner(sink: .discarding).run(
                    namespace: "experiment", ["analyze", value.study, "--run", reference, "--json"])
                #expect(result.envelope.exitCode == 65, "\(reference): \(result.envelope)")
                #expect(result.envelope.error?.code == "missingPrerequisite")
                #expect(result.envelope.error?.reason.contains(words) == true, "\(result.envelope)")
                #expect(result.envelope.error?.repairAction.isEmpty == false)
            }
            // All three spellings of the same run analyze it.
            for reference in ["runs/\(Self.older)", Self.older, root.appending(path: "runs/\(Self.older)").path] {
                let result = await ExperimentCLIRunner(sink: .discarding).run(
                    namespace: "experiment", ["analyze", value.study, "--run", reference, "--json"])
                #expect(result.envelope.exitCode == 0, "\(reference): \(result.envelope)")
                #expect(result.envelope.result?["sourceRun"] == .string(Self.older))
            }
        }
    }
}
