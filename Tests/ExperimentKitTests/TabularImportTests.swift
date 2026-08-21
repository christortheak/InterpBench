import Foundation
import Testing

@testable import ExperimentKit

/// Tabular import (Usability Plan Phase 3, item 13): JSON-array/CSV tables
/// become pinned task prompts and human baselines through a column mapping.
/// Pure conversion rules are tested without IO; the write/pin path runs
/// under the shared temp-root override (never the real workspace).
@Suite(.serialized) struct TabularImportTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "tabular", body)
    }

    // MARK: - Parsing

    @Test func parsesJSONArrayPreservingTypes() throws {
        let json = #"""
        [
          {"prompt": "Affirm or reverse?", "options": ["affirm", "reverse"],
           "target": "affirm", "id": "case-1"},
          {"prompt": "Guilty or not?", "options": ["guilty", "not guilty"],
           "target": "not guilty", "id": "case-2"}
        ]
        """#
        let table = try TabularImport.parseTable(Data(json.utf8), fileName: "cases.json")
        #expect(table.rows.count == 2)
        #expect(Set(table.columns) == ["prompt", "options", "target", "id"])
        #expect(table.rows[0]["prompt"] == .string("Affirm or reverse?"))
        #expect(
            table.rows[0]["options"]
                == .array([.string("affirm"), .string("reverse")]))
    }

    @Test func parsesCSVWithQuotedCommasAndBlankLines() throws {
        let csv = "text,id\n\"Affirm, or reverse?\",case-1\n\nSecond prompt,case-2\n"
        let table = try TabularImport.parseTable(Data(csv.utf8), fileName: "cases.csv")
        #expect(table.columns == ["text", "id"])
        #expect(table.rows.count == 2)
        #expect(table.rows[0]["text"] == .string("Affirm, or reverse?"))
        #expect(table.rows[1]["id"] == .string("case-2"))
    }

    @Test func parseRefusesNonTableJSONPlainly() {
        #expect(throws: TabularImport.Problem.self) {
            try TabularImport.parseTable(
                Data(#"{"not": "an array"}"#.utf8), fileName: "x.json")
        }
        do {
            _ = try TabularImport.parseTable(
                Data("header-only\n".utf8), fileName: "x.csv")
            Issue.record("header-only CSV should refuse")
        } catch {
            #expect("\(error)".contains("no data rows"))
        }
    }

    // MARK: - Auto-guess mapping

    @Test func guessesMappingCaseAndPunctuationInsensitively() {
        let prompts = TabularImport.guessMapping(
            columns: ["Prompt", "Choices", "TARGET", "item_id"],
            target: .taskPrompts)
        #expect(prompts["text"] == "Prompt")
        #expect(prompts["options"] == "Choices")
        #expect(prompts["target"] == "TARGET")
        #expect(prompts["id"] == "item_id")

        let baseline = TabularImport.guessMapping(
            columns: ["endpoint", "Delta_Human", "CI_Lower", "ci upper"],
            target: .humanBaseline)
        #expect(baseline["deltaHuman"] == "Delta_Human")
        #expect(baseline["ciLower"] == "CI_Lower")
        #expect(baseline["ciUpper"] == "ci upper")
    }

    @Test func ambiguousColumnsGuessNothing() {
        // Both columns normalize to a "text" alias — guessing between them
        // would be silent coercion, so the picker decides.
        let mapping = TabularImport.guessMapping(
            columns: ["text", "Prompt"], target: .taskPrompts)
        #expect(mapping["text"] == nil)
    }

    // MARK: - Task-prompt conversion

    @Test func jsonArrayConvertsWithOptionsAndTargetPreserved() throws {
        let json = #"""
        [{"prompt": "Affirm or reverse?", "options": ["affirm", "reverse"],
          "target": "affirm", "id": "case-1"}]
        """#
        let table = try TabularImport.parseTable(Data(json.utf8), fileName: "t.json")
        let mapping = TabularImport.guessMapping(
            columns: table.columns, target: .taskPrompts)
        let jsonl = try TabularImport.taskPromptsJSONL(table: table, mapping: mapping)
        let lines = jsonl.split(separator: "\n")
        #expect(lines.count == 1)
        let object =
            try JSONSerialization.jsonObject(with: Data(lines[0].utf8))
            as? [String: Any]
        #expect(object?["text"] as? String == "Affirm or reverse?")
        #expect(object?["options"] as? [String] == ["affirm", "reverse"])
        #expect(object?["target"] as? String == "affirm")
        #expect(object?["id"] as? String == "case-1")
        // The converted file passes the pin parser's own preview.
        #expect(TaskPromptsImport.looksLikeJSONL(jsonl))
    }

    @Test func csvConvertsToTaskPrompts() throws {
        let csv = "question,answer,choices\nAffirm or reverse?,affirm,affirm|reverse\n"
        let table = try TabularImport.parseTable(Data(csv.utf8), fileName: "t.csv")
        // Paper-style names — mapped by hand, as the sheet would.
        let jsonl = try TabularImport.taskPromptsJSONL(
            table: table,
            mapping: ["text": "question", "target": "answer", "options": "choices"])
        let object =
            try JSONSerialization.jsonObject(
                with: Data(jsonl.split(separator: "\n")[0].utf8))
            as? [String: Any]
        #expect(object?["text"] as? String == "Affirm or reverse?")
        #expect(object?["options"] as? [String] == ["affirm", "reverse"])
        #expect(object?["target"] as? String == "affirm")
    }

    @Test func delimitedOptionsParseBothDelimiters() {
        #expect(
            TabularImport.optionStrings(.string("yes|no|maybe"))
                == ["yes", "no", "maybe"])
        #expect(
            TabularImport.optionStrings(.string("affirm; reverse"))
                == ["affirm", "reverse"])
        #expect(TabularImport.optionStrings(.string("guilty")) == ["guilty"])
        #expect(
            TabularImport.optionStrings(.array([.string("a"), .number(3)]))
                == ["a", "3"])
    }

    @Test func missingRequiredColumnRefusesNamingIt() throws {
        let table = TabularImport.Table(
            columns: ["question"], rows: [["question": .string("Q?")]])
        do {
            _ = try TabularImport.taskPromptsJSONL(table: table, mapping: [:])
            Issue.record("unmapped required text should refuse")
        } catch {
            #expect("\(error)".contains("text"))
            #expect("\(error)".contains("pick the column"))
        }
        do {
            _ = try TabularImport.humanBaselineCSV(
                table: table, mapping: ["endpoint": "question"])
            Issue.record("unmapped baseline numerics should refuse")
        } catch {
            let message = "\(error)"
            #expect(message.contains("deltaHuman"))
            #expect(message.contains("ciLower"))
            #expect(message.contains("ciUpper"))
        }
    }

    @Test func emptyRequiredCellAndDuplicateIDsRefuse() throws {
        let empty = TabularImport.Table(
            columns: ["text"], rows: [[:]])
        do {
            _ = try TabularImport.taskPromptsJSONL(
                table: empty, mapping: ["text": "text"])
            Issue.record("empty text cell should refuse")
        } catch {
            #expect("\(error)".contains("row 1"))
        }
        // The pin parser's duplicate-id rule surfaces at conversion, not
        // later at the pin.
        let duplicates = TabularImport.Table(
            columns: ["text", "id"],
            rows: [
                ["text": .string("A?"), "id": .string("case-1")],
                ["text": .string("B?"), "id": .string("case-1")],
            ])
        #expect(throws: TabularImport.Problem.self) {
            try TabularImport.taskPromptsJSONL(
                table: duplicates, mapping: ["text": "text", "id": "id"])
        }
    }

    // MARK: - Human-baseline conversion

    @Test func paperStyleCSVConvertsToLoaderColumns() throws {
        let csv = """
            outcome,Delta_Human,CI_Lower,CI_Upper,notes
            holding_flip,0.31,0.12,0.48,"source paper, table 2"
            severity_months,-2.5,-4.1,-0.9,
            """
        let table = try TabularImport.parseTable(Data(csv.utf8), fileName: "paper.csv")
        var mapping = TabularImport.guessMapping(
            columns: table.columns, target: .humanBaseline)
        mapping["endpoint"] = "outcome"  // paper name — picked in the sheet
        let converted = try TabularImport.humanBaselineCSV(
            table: table, mapping: mapping)
        #expect(
            converted == """
                endpoint,deltaHuman,ciLower,ciUpper
                holding_flip,0.31,0.12,0.48
                severity_months,-2.5,-4.1,-0.9

                """)
        // By construction the loader's shape check passes.
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(converted.utf8), file: "converted.csv") == nil)
    }

    @Test func nonNumericEffectRefusesPlainly() throws {
        let csv = "endpoint,deltaHuman,ciLower,ciUpper\nflip,large,0.1,0.2\n"
        let table = try TabularImport.parseTable(Data(csv.utf8), fileName: "b.csv")
        let mapping = TabularImport.guessMapping(
            columns: table.columns, target: .humanBaseline)
        do {
            _ = try TabularImport.humanBaselineCSV(table: table, mapping: mapping)
            Issue.record("non-numeric deltaHuman should refuse")
        } catch {
            #expect("\(error)".contains("deltaHuman"))
            #expect("\(error)".contains("not a number"))
        }
    }

    // MARK: - Write + pin (temp root)

    @Test func importWritesPinsAndRefusesDifferingExistingFile() throws {
        try withTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "tab-import", description: "t", modelID: "test-model")
            let table = TabularImport.Table(
                columns: ["prompt"],
                rows: [["prompt": .string("Affirm or reverse?")]])
            let result = try TabularImport.importTaskPrompts(
                table: table, mapping: ["text": "prompt"], manifest: &manifest)
            try ExperimentStore.save(manifest)
            #expect(result.file == "prompts/tasks/tab-import-prompts.jsonl")
            #expect(manifest.taskPromptsFile == result.file)
            #expect(manifest.taskPromptsHash == result.hash)
            // Identical bytes: idempotent re-import re-pins.
            let again = try TabularImport.importTaskPrompts(
                table: table, mapping: ["text": "prompt"], manifest: &manifest)
            #expect(again.hash == result.hash)
            // Different rows → different bytes at the same destination:
            // refuse with the remedy, and the pin stays untouched.
            let changed = TabularImport.Table(
                columns: ["prompt"],
                rows: [["prompt": .string("Guilty or not?")]])
            do {
                _ = try TabularImport.importTaskPrompts(
                    table: changed, mapping: ["text": "prompt"],
                    manifest: &manifest)
                Issue.record("differing existing file should refuse")
            } catch {
                #expect("\(error)".contains("never overwrite"))
            }
            #expect(manifest.taskPromptsHash == result.hash)
        }
    }

    @Test func humanBaselineImportLandsAtDestinationAndPins() throws {
        try withTempRoot { root in
            _ = try ExperimentStore.create(
                name: "tab-base", description: "t", modelID: "test-model")
            let json = #"""
            [{"endpoint": "holding_flip", "deltaHuman": 0.31,
              "ciLower": 0.12, "ciUpper": 0.48}]
            """#
            let table = try TabularImport.parseTable(Data(json.utf8), fileName: "b.json")
            let mapping = TabularImport.guessMapping(
                columns: table.columns, target: .humanBaseline)
            let pinned = try TabularImport.importHumanBaseline(
                table: table, mapping: mapping, experimentName: "tab-base")
            #expect(pinned.path == "prompts/baselines/tab-base-human-baseline.csv")
            let written = try Data(
                contentsOf: root.appending(path: pinned.path))
            #expect(
                PinShapeValidation.humanBaselineShapeProblem(
                    written, file: pinned.path) == nil)
            let reloaded = try ExperimentStore.load(name: "tab-base")
            #expect(reloaded.humanBaseline?.hash == pinned.hash)
            // Identical bytes at the destination: the re-import is
            // idempotent — same pin, file untouched.
            let again = try TabularImport.importHumanBaseline(
                table: table, mapping: mapping, experimentName: "tab-base")
            #expect(again.hash == pinned.hash)
            #expect(
                (try Data(contentsOf: root.appending(path: pinned.path)))
                    == written)
        }
    }

    // MARK: - Transactional import (a pin refusal rolls the write back)

    private let baselineTable = TabularImport.Table(
        columns: ["endpoint", "deltaHuman", "ciLower", "ciUpper"],
        rows: [[
            "endpoint": .string("rate"), "deltaHuman": .string("0.1"),
            "ciLower": .string("0.0"), "ciUpper": .string("0.2"),
        ]])
    private let baselineMapping = [
        "endpoint": "endpoint", "deltaHuman": "deltaHuman",
        "ciLower": "ciLower", "ciUpper": "ciUpper",
    ]

    private func createFrozen(_ name: String) throws {
        _ = try ExperimentStore.create(
            name: name, description: "t", modelID: "test-model")
        var manifest = try ExperimentStore.load(name: name)
        manifest.status = .frozen
        try ExperimentStore.save(manifest)
    }

    /// Finding A: the write used to land BEFORE the pin, so a pin refusal
    /// (frozen manifest) left an unpinned file behind. Now the failed
    /// import rolls its own write back — no new file survives the refusal.
    @Test func frozenStudyPinRefusalLeavesNoNewFile() throws {
        try withTempRoot { root in
            try createFrozen("tab-frozen")
            do {
                _ = try TabularImport.importHumanBaseline(
                    table: baselineTable, mapping: baselineMapping,
                    experimentName: "tab-frozen")
                Issue.record("frozen study should refuse the pin")
            } catch {
                #expect("\(error)".contains("frozen"))
            }
            let destination = root.appending(
                path: "prompts/baselines/tab-frozen-human-baseline.csv")
            #expect(!FileManager.default.fileExists(atPath: destination.path))
            #expect(try ExperimentStore.load(name: "tab-frozen").humanBaseline == nil)
        }
    }

    /// Finding A (2026-07-14 review): the pin only mutates the in-memory
    /// manifest, so a panel-side save AFTER the import used to be outside
    /// the transaction — a save refusal orphaned the freshly written
    /// prompts file. The save now runs INSIDE the import (its `persist`
    /// step): the realistic race — the manifest frozen ON DISK after the
    /// panel loaded its draft copy — rolls the write back.
    @Test func taskPromptsSaveRefusalRollsBackTheWrittenFile() throws {
        try withTempRoot { root in
            var draftCopy = try ExperimentStore.create(
                name: "tab-race", description: "t", modelID: "test-model")
            // The race: someone froze the study on disk after this draft
            // copy was loaded.
            var onDisk = try ExperimentStore.load(name: "tab-race")
            onDisk.status = .frozen
            try ExperimentStore.save(onDisk)
            let table = TabularImport.Table(
                columns: ["prompt"],
                rows: [["prompt": .string("Affirm or reverse?")]])
            do {
                _ = try TabularImport.importTaskPrompts(
                    table: table, mapping: ["text": "prompt"],
                    manifest: &draftCopy,
                    persist: { try ExperimentStore.save($0) })
                Issue.record("the frozen-on-disk save should refuse")
            } catch {
                #expect("\(error)".contains("frozen"))
            }
            // The whole import rolled back: no orphan file, no pin.
            let destination = root.appending(
                path: "prompts/tasks/tab-race-prompts.jsonl")
            #expect(!FileManager.default.fileExists(atPath: destination.path))
            #expect(
                try ExperimentStore.load(name: "tab-race").taskPromptsHash == nil)
        }
    }

    /// The success path is unchanged with `persist` supplied: one save,
    /// after the pin, and the file + pin both land.
    @Test func taskPromptsImportPersistsThroughTheTransaction() throws {
        try withTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "tab-persist", description: "t", modelID: "test-model")
            let table = TabularImport.Table(
                columns: ["prompt"],
                rows: [["prompt": .string("Affirm or reverse?")]])
            var persisted = 0
            let result = try TabularImport.importTaskPrompts(
                table: table, mapping: ["text": "prompt"], manifest: &manifest,
                persist: { saved in
                    persisted += 1
                    try ExperimentStore.save(saved)
                })
            #expect(persisted == 1)
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(path: result.file).path))
            let reloaded = try ExperimentStore.load(name: "tab-persist")
            #expect(reloaded.taskPromptsFile == result.file)
            #expect(reloaded.taskPromptsHash == result.hash)
        }
    }

    /// Rollback discipline holds for the persist step too: a pre-existing
    /// identical prompts file (the idempotent case) survives a failed save
    /// byte-for-byte.
    @Test func preExistingTaskPromptsFileSurvivesSaveRefusal() throws {
        try withTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "tab-keepfile", description: "t", modelID: "test-model")
            let table = TabularImport.Table(
                columns: ["prompt"],
                rows: [["prompt": .string("Affirm or reverse?")]])
            let jsonl = try TabularImport.taskPromptsJSONL(
                table: table, mapping: ["text": "prompt"])
            let path = "prompts/tasks/tab-keepfile-prompts.jsonl"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(jsonl.utf8).write(to: url)
            struct SaveFailed: Error {}
            do {
                _ = try TabularImport.importTaskPrompts(
                    table: table, mapping: ["text": "prompt"],
                    manifest: &manifest,
                    persist: { _ in throw SaveFailed() })
                Issue.record("the injected save failure should surface")
            } catch is SaveFailed {
            } catch {
                Issue.record("unexpected error: \(error)")
            }
            // The import did not create the file, so it must not delete it.
            #expect(try Data(contentsOf: url) == Data(jsonl.utf8))
        }
    }

    /// The rollback removes ONLY what the import created: a pre-existing
    /// identical file (the idempotent case) survives a later pin failure
    /// byte-for-byte.
    @Test func preExistingIdenticalFileSurvivesPinRefusal() throws {
        try withTempRoot { root in
            try createFrozen("tab-keep")
            let csv = try TabularImport.humanBaselineCSV(
                table: baselineTable, mapping: baselineMapping)
            let path = "prompts/baselines/tab-keep-human-baseline.csv"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(csv.utf8).write(to: url)
            do {
                _ = try TabularImport.importHumanBaseline(
                    table: baselineTable, mapping: baselineMapping,
                    experimentName: "tab-keep")
                Issue.record("frozen study should refuse the pin")
            } catch {
                #expect("\(error)".contains("frozen"))
            }
            // The import did not create the file, so it must not delete it.
            #expect(try Data(contentsOf: url) == Data(csv.utf8))
        }
    }
}
