import Foundation
import Testing

@testable import ExperimentKit

/// The logprobShift authoring preview (`SweepSpecForm.previewChoicePrompts`)
/// — an advisory over the ENGINE's own loader (`SweepSelectionRule.
/// loadChoiceRows` + `ExperimentTasks.parseTaskPrompts`), so its verdicts
/// can never diverge from what the sweep enforces at start. Pure-CPU:
/// fixtures are temp files written by each test, never live model output.
struct ChoicePromptsPreviewTests {

    // MARK: fixtures

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "choice-preview-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    @discardableResult
    private func write(
        _ content: String, to root: URL, name: String = "choices.jsonl"
    ) throws -> String {
        let url = root.appending(path: name)
        try content.write(to: url, atomically: true, encoding: .utf8)
        return name
    }

    // MARK: happy path

    @Test func validFileCountsRowsOptionsAndTargets() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            """
            {"id": "a", "prompt": "P1", "options": ["yes", "no"]}
            {"id": "b", "prompt": "P2", "options": ["1", "2", "3"], "target": "2"}
            {"id": "c", "prompt": "P3", "options": ["w", "x", "y", "z"]}
            """,
            to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        #expect(
            outcome
                == .ok(
                    SweepSpecForm.ChoicePromptsPreview(
                        rowCount: 3, minOptions: 2, maxOptions: 4,
                        explicitTargetRows: 1, defaultedTargetRows: 2)))
    }

    @Test func explicitTargetEqualToFirstOptionStillCountsAsExplicit() throws {
        // The reason the preview re-reads the RAW rows: a resolved ChoiceRow
        // whose target equals options[0] is indistinguishable from a
        // defaulted one, but the file declared it explicitly.
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            """
            {"prompt": "P1", "options": ["A", "B"], "target": "A"}
            {"prompt": "P2", "options": ["A", "B"]}
            """,
            to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        guard case .ok(let preview) = outcome else {
            Issue.record("expected .ok, got \(outcome)")
            return
        }
        #expect(preview.explicitTargetRows == 1)
        #expect(preview.defaultedTargetRows == 1)
        #expect(preview.minOptions == 2)
        #expect(preview.maxOptions == 2)
    }

    @Test func summaryLineCollapsesAUniformOptionRange() {
        let uniform = SweepSpecForm.ChoicePromptsPreview(
            rowCount: 1, minOptions: 2, maxOptions: 2,
            explicitTargetRows: 0, defaultedTargetRows: 1)
        #expect(
            SweepSpecForm.choicePromptsSummary(uniform)
                == "1 row · 2 options per row · 0 with explicit target, "
                + "1 defaulting to options[0]")
        let ranged = SweepSpecForm.ChoicePromptsPreview(
            rowCount: 3, minOptions: 2, maxOptions: 4,
            explicitTargetRows: 1, defaultedTargetRows: 2)
        #expect(SweepSpecForm.choicePromptsSummary(ranged).contains("2–4 options"))
    }

    // MARK: refusals — always the engine's own message

    @Test func emptyFileIsARowlessProblem() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write("", to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        guard case .problem(let reason) = outcome else {
            Issue.record("expected .problem, got \(outcome)")
            return
        }
        #expect(reason.contains("no rows"))
    }

    @Test func malformedRowNamesTheLine() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            """
            {"prompt": "P1", "options": ["A", "B"]}
            not json at all
            """,
            to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        guard case .problem(let reason) = outcome else {
            Issue.record("expected .problem, got \(outcome)")
            return
        }
        #expect(reason.contains("line 2"))
    }

    @Test func missingFileIsANamedProblem() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let outcome = SweepSpecForm.previewChoicePrompts(
            file: "does-not-exist.jsonl", root: root)
        guard case .problem(let reason) = outcome else {
            Issue.record("expected .problem, got \(outcome)")
            return
        }
        #expect(reason.contains("not found"))
    }

    @Test func fewerThanTwoOptionsNamesTheRow() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            """
            {"id": "ok-row", "prompt": "P1", "options": ["A", "B"]}
            {"id": "solo", "prompt": "P2", "options": ["only"]}
            """,
            to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        guard case .problem(let reason) = outcome else {
            Issue.record("expected .problem, got \(outcome)")
            return
        }
        #expect(reason.contains("'solo'"))
        #expect(reason.contains("at least 2 options"))
    }

    @Test func targetOutsideItsOptionsIsAProblem() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            #"{"id": "t", "prompt": "P", "options": ["A", "B"], "target": "C"}"#,
            to: root)
        let outcome = SweepSpecForm.previewChoicePrompts(file: file, root: root)
        guard case .problem(let reason) = outcome else {
            Issue.record("expected .problem, got \(outcome)")
            return
        }
        #expect(reason.contains("'t'"))
        #expect(reason.contains("not one of"))
    }

    // MARK: edge inputs

    @Test func blankOrNilPathIsNoFileNotAProblem() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(SweepSpecForm.previewChoicePrompts(file: nil, root: root) == .noFile)
        #expect(SweepSpecForm.previewChoicePrompts(file: "   ", root: root) == .noFile)
    }

    @Test func absolutePathResolvesLikeTheEngine() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let name = try write(
            #"{"prompt": "P", "options": ["A", "B"]}"#, to: root)
        let absolute = root.appending(path: name).path
        // Resolve against an unrelated root: absolute paths pass through,
        // exactly as `SweepSelectionRule.choicePromptsURL` resolves them.
        let outcome = SweepSpecForm.previewChoicePrompts(
            file: absolute, root: FileManager.default.temporaryDirectory)
        guard case .ok(let preview) = outcome else {
            Issue.record("expected .ok, got \(outcome)")
            return
        }
        #expect(preview.rowCount == 1)
    }

    @Test func previewProblemMatchesTheEngineVerbatim() throws {
        // The never-disagree guarantee: whatever reason the sweep's own
        // loader raises is byte-identical to the advisory's.
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(
            #"{"prompt": "P", "options": ["only"]}"#, to: root)
        var engineReason: String?
        do {
            _ = try SweepSelectionRule.loadChoiceRows(file: file, root: root)
            Issue.record("engine unexpectedly accepted the fixture")
        } catch let error as ExperimentError {
            engineReason = error.reason
        }
        let reason = try #require(engineReason)
        #expect(
            SweepSpecForm.previewChoicePrompts(file: file, root: root)
                == .problem(reason))
    }
}
