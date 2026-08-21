import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Duplicate task-prompt item ids (review finding F1).
///
/// Duplicate ids silently corrupt pairing on both engines (choice readouts
/// and paired statistics key on `promptID`) and used to TRAP Swift report
/// assembly (`Dictionary(uniqueKeysWithValues:)`) after all generation
/// compute was spent. The contract under test:
///
/// 1. **Refusal at load, identical messages cross-engine** — replayed from
///    the committed fixture
///    `prompts/fixtures/task-prompts-validation/cases.json` (Python twin:
///    `Server/tests/test_task_prompt_ids.py`). Every task-prompt consumer
///    (run, validate, sweep, logprob, pin, import) inherits the gate from
///    `parseTaskPrompts` / `_load_prompts`.
/// 2. **Report assembly never traps regardless** — a (hypothetically
///    bypassed) duplicate readout key merges last-wins, the server's dict
///    semantics.
/// 3. **`data check` names every duplicate** — INVALID (blocking) status on
///    the task-prompts row, matching the run's refusal to load the file, so
///    the researcher fixes the file in one edit.
/// 4. **Import preview flags the duplicate** — a passing preview must keep
///    guaranteeing the pin's parse succeeds.
struct TaskPromptDuplicateIDTests {

    // MARK: - Fixture helpers

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    struct ValidationCase: Decodable {
        let name: String
        let jsonl: [String]
        let expect: String?
    }

    struct ValidationFixture: Decodable {
        let cases: [ValidationCase]
    }

    private static func loadCases() throws -> [ValidationCase] {
        let url = repoRoot.appending(
            components: "prompts", "fixtures", "task-prompts-validation", "cases.json")
        return try JSONDecoder()
            .decode(ValidationFixture.self, from: Data(contentsOf: url)).cases
    }

    private let duplicateJSONL = """
        {"id": "case-a", "prompt": "First framing.", "options": ["yes", "no"], "target": "yes"}
        {"id": "case-b", "prompt": "Another item."}
        {"id": "case-a", "prompt": "Repeat framing.", "options": ["yes", "no"], "target": "no"}
        """

    private let duplicateMessage =
        "task prompts: duplicate item id 'case-a' (items 1 and 3) — "
        + "ids must be unique for pairing and reporting"

    // MARK: - 1. Refusal at load: the committed cross-engine cases

    @Test func loadValidationMatchesCommittedFixture() throws {
        let cases = try Self.loadCases()
        #expect(!cases.isEmpty)
        for c in cases {
            let data = Data((c.jsonl.joined(separator: "\n") + "\n").utf8)
            var thrown: String?
            do {
                _ = try ExperimentTasks.parseTaskPrompts(data)
            } catch {
                thrown = (error as? ExperimentError)?.reason ?? "\(error)"
            }
            #expect(
                thrown == c.expect,
                """
                \(c.name): this engine's refusal diverged from the \
                cross-engine fixture (the Python twin replays the same file).
                expected: \(c.expect ?? "nil")
                got:      \(thrown ?? "nil")
                """)
        }
    }

    @Test func fixtureCoversTheDuplicateRuleAndAValidCase() throws {
        let expects = try Self.loadCases().map(\.expect)
        #expect(
            expects.contains {
                $0?.contains("ids must be unique for pairing and reporting") == true
            })
        #expect(expects.contains { $0 == nil }, "fixture needs a valid case too")
    }

    /// Acceptance test 1 (review doc): the run path either completes with
    /// report.json or REFUSES at load — validation now blocks the duplicate
    /// before any generation compute, through the same `loadTaskPrompts`
    /// every run/validate/sweep/logprob path calls.
    @Test func runPathLoaderRefusesDuplicateIDsWithExactMessage() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "dup-ids-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appending(component: "items.jsonl")
        try (duplicateJSONL + "\n").write(to: file, atomically: true, encoding: .utf8)

        var manifest = ExperimentManifest(
            name: "dup-study", description: "", modelID: "test/model")
        manifest.taskPromptsFile = file.path

        #expect {
            try ExperimentTasks.loadTaskPrompts(for: manifest)
        } throws: { error in
            (error as? ExperimentError)?.reason == duplicateMessage
        }
    }

    // MARK: - 2. Report assembly is crash-proof regardless

    /// A (hypothetically bypassed) duplicate readout key must NOT trap at
    /// report time — the defensive last-wins merge matches the server's
    /// `_choice_readouts` dict semantics. Before the fix this call was
    /// `Fatal error: Duplicate values for key` after all compute was spent.
    @Test func reportAssemblyMergesDuplicateReadoutKeysLastWins() {
        let manifest = ExperimentManifest(
            name: "dup-study", description: "", modelID: "test/model")
        func readout(
            _ condition: String, id: String, selected: String
        ) -> ExperimentTasks.ReportChoiceReadout {
            ExperimentTasks.ReportChoiceReadout(
                condition: condition, promptID: id, sampleIndex: nil,
                source: "parsed", selected: selected, target: "yes")
        }
        // Duplicate ChoiceKey (promptID, sampleIndex, source) in BOTH the
        // baseline map and a condition's map.
        let readouts = [
            readout("baseline", id: "case-a", selected: "yes"),
            readout("baseline", id: "case-a", selected: "no"),  // last wins
            readout("steered", id: "case-a", selected: "yes"),
            readout("steered", id: "case-a", selected: "no"),  // last wins
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hash",
            taskPrompts: (file: "items.jsonl", hash: "h", prompts: []),
            rows: [], conditionCount: 2, concepts: [],
            choiceReadouts: readouts)
        // Last-wins on both sides: baseline "no" vs steered "no" agree over
        // the single shared key.
        let agreement = report.conditions["steered"]?.agreementWithBaseline
        #expect(agreement?.n == 1)
        #expect(agreement?.agreement == 1.0)
    }

    // MARK: - 3. data check names the duplicates

    @Test func readinessRowNamesEveryDuplicateIDAsInvalidBlocker() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "dup-readiness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appending(path: "prompts/tasks"),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let lines = [
            #"{"id": "case-a", "prompt": "one"}"#,
            #"{"id": "case-b", "prompt": "two"}"#,
            #"{"id": "case-a", "prompt": "three"}"#,
            #"{"id": "case-b", "prompt": "four"}"#,
            #"{"id": "case-b", "prompt": "five"}"#,
        ]
        try (lines.joined(separator: "\n") + "\n").write(
            to: root.appending(path: "prompts/tasks/items.jsonl"),
            atomically: true, encoding: .utf8)
        var m = ExperimentManifest(
            name: "dup-study", description: "", modelID: "test/model")
        m.taskPromptsFile = "prompts/tasks/items.jsonl"

        let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
        let prompts = try #require(rows.first { $0.id == "taskPrompts" })
        // A file the run refuses to load is a BLOCKER (`.invalid`), not a
        // degraded-run `.partial` — preflight must match execution.
        #expect(prompts.status == .invalid)
        #expect(prompts.detail.contains("'case-a' (items 1, 3)"))
        #expect(prompts.detail.contains("'case-b' (items 2, 4, 5)"))
        #expect(prompts.detail.contains("ids must be unique"))
        #expect(prompts.detail.contains("refuses to load"))

        // The summary counts it among the blockers, so `data check` exits 2
        // (the CLI's exit path is `!summary.isReady`).
        let summary = StudyDataReadiness.summary(rows)
        #expect(!summary.isReady)
        #expect(summary.invalidCount == 1)
        #expect(summary.blockers.contains { $0.id == "taskPrompts" })
    }

    // MARK: - 4. Import preview flags the duplicate

    @Test func importPreviewRefusesDuplicateIDsWithTheContractMessage() {
        let outcome = TaskPromptsImport.preview(duplicateJSONL)
        #expect(outcome == .failure(line: 3, message: duplicateMessage))
    }

    @Test func importPreviewStillPassesUniqueIDs() {
        let jsonl = """
            {"id": "case-a", "prompt": "First framing."}
            {"id": "case-b", "prompt": "Another item."}
            {"prompt": "Auto-numbered item."}
            """
        guard case .preview(let preview) = TaskPromptsImport.preview(jsonl) else {
            Issue.record("unique ids must keep previewing cleanly")
            return
        }
        #expect(preview.recordCount == 3)
    }

    @Test func emptyIDRefusesAndNullSharesTheFallback() throws {
        // Round 2026-08-03 P2 (cross-engine contract with _load_prompts):
        // `id: null` and an ABSENT id share the prompt-<ordinal> fallback;
        // an explicit id must be a non-empty string.
        let ok = "{\"id\": null, \"prompt\": \"p\"}\n{\"prompt\": \"q\"}"
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(ok.utf8))
        #expect(prompts.map(\.id) == ["prompt-1", "prompt-2"])
        for bad in ["{\"id\": \"\", \"prompt\": \"p\"}",
                    "{\"id\": \"   \", \"prompt\": \"p\"}"] {
            do {
                _ = try ExperimentTasks.parseTaskPrompts(Data(bad.utf8))
                Issue.record("an empty id must refuse at load")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("empty or non-string 'id'"))
            }
        }
    }
}
