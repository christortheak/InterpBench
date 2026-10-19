import Foundation
import Testing

@testable import ExperimentKit

/// A sweep's replies, read back cell by cell for the Optimizations grid.
///
/// The gap this closes: both engines record every dev-prompt generation in
/// `dev-generations.jsonl` (`SweepDevGenerationsTests`), and the Demo
/// Workspace's guide tells researchers to choose a cell by reading its
/// replies, but the grid showed only the numbers. `SweepReplies` reads the
/// record back and pairs one cell's replies with the unsteered baseline's
/// replies to the same prompts. Temp directories and fixture lines only, plus
/// the shipped Demo Workspace's own sweep run when one is carried.
struct SweepRepliesTests {

    private let fm = FileManager.default

    private func temporary() throws -> URL {
        let url = fm.temporaryDirectory
            .appending(component: "sweep-replies-\(UUID().uuidString)")
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func line(
        _ kind: String, _ concept: String?, _ layer: Int, _ alpha: Double,
        _ prompt: Int, _ text: String
    ) throws -> String {
        try SweepRunCatalog.devGenerationLine(
            kind: kind, concept: concept, layer: layer, alpha: alpha,
            promptIndex: prompt, text: text)
    }

    private func record(_ lines: [String]) -> SweepReplies.Record {
        SweepReplies.parse(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    // MARK: Pairing

    @Test func pairsTheSelectedCellWithTheBaselineByPrompt() throws {
        let replies = record([
            try line("baseline", nil, -1, 0, 0, "calm library"),
            try line("baseline", nil, -1, 0, 1, "weekend at home"),
            try line("cell", "fear", 2, 0.1, 0, "the shelves watched me"),
            try line("cell", "fear", 2, 0.1, 1, "I will stay inside"),
            try line("cell", "fear", 2, 0.2, 0, "run run run"),
            try line("cell", "hungry", 2, 0.1, 0, "pancakes"),
        ])
        let prompts = SweepReplies.DevPrompts(
            texts: ["Describe a library.", "Plan a weekend."], status: .matched)
        let comparison = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 2, alpha: 0.1,
            prompts: prompts)

        #expect(comparison.pairs.map(\.promptIndex) == [0, 1])
        #expect(comparison.pairs.map(\.prompt) == ["Describe a library.", "Plan a weekend."])
        #expect(comparison.pairs.map(\.baseline?.text) == ["calm library", "weekend at home"])
        #expect(
            comparison.pairs.map(\.steered?.text)
                == ["the shelves watched me", "I will stay inside"])
        #expect(comparison.hasCellReplies && comparison.hasBaseline)
        #expect(!comparison.hasControl)
    }

    @Test func aCellCutShortShowsOnlyThePromptsItAnswered() throws {
        // A cancel can stop a cell mid-prompts; the baseline answered every
        // prompt, and must not make the cell look as if it did too.
        let replies = record([
            try line("baseline", nil, -1, 0, 0, "b0"),
            try line("baseline", nil, -1, 0, 1, "b1"),
            try line("baseline", nil, -1, 0, 2, "b2"),
            try line("cell", "fear", 2, 0.1, 0, "c0"),
        ])
        let partial = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 2, alpha: 0.1,
            prompts: SweepReplies.DevPrompts(status: .notNamed))
        #expect(partial.pairs.map(\.promptIndex) == [0])
        #expect(partial.pairs.first?.prompt == nil)
        let absent = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 3, alpha: 0.1,
            prompts: SweepReplies.DevPrompts(status: .notNamed))
        #expect(!absent.hasCellReplies)
        #expect(absent.pairs.isEmpty)
    }

    @Test func aRunWithoutABaselineStillShowsTheCell() throws {
        // The server's legacy sweep (no sweep spec) writes one cell reply per
        // cell and no baseline.
        let comparison = SweepReplies.comparison(
            record: record([try line("cell", "fear", 4, 0.5, 0, "alone")]),
            concept: "fear", layer: 4, alpha: 0.5,
            prompts: SweepReplies.DevPrompts(status: .notNamed))
        #expect(comparison.hasCellReplies)
        #expect(!comparison.hasBaseline)
        #expect(comparison.pairs.first?.baseline == nil)
    }

    @Test func theControlJoinsOnlyTheCellItWasRunAt() throws {
        let replies = record([
            try line("baseline", nil, -1, 0, 0, "b0"),
            try line("cell", "fear", 2, 0.1, 0, "c0"),
            try line("cell", "fear", 2, 0.2, 0, "d0"),
            try line("control", "fear", 2, 0.1, 0, "random breeze"),
        ])
        let at = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 2, alpha: 0.1,
            prompts: SweepReplies.DevPrompts(status: .notNamed))
        #expect(at.hasControl)
        #expect(at.pairs.first?.control?.text == "random breeze")
        let elsewhere = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 2, alpha: 0.2,
            prompts: SweepReplies.DevPrompts(status: .notNamed))
        #expect(!elsewhere.hasControl)
        #expect(elsewhere.pairs.first?.steered?.text == "d0")
    }

    // MARK: Reading the record

    @Test func readsBothEnginesEncodingsOfTheSameRecord() throws {
        // Python's `json.dumps(record, sort_keys=True)`: spaced separators,
        // `0.0` for the baseline's alpha, ASCII escapes, and a `truncated`
        // flag. The Swift writer's form is compact with raw UTF-8. Same
        // schema, one reader.
        let python = [
            #"{"alpha": 0.0, "concept": null, "kind": "baseline", "layer": -1, "promptIndex": 0, "text": "café au lait"}"#,
            #"{"alpha": 0.1, "concept": "fear", "kind": "cell", "layer": 2, "promptIndex": 0, "text": "dread\nat dawn", "truncated": true}"#,
        ]
        let swift = [
            try line("baseline", nil, -1, 0, 0, "café au lait"),
            try line("cell", "fear", 2, 0.1, 0, "dread\nat dawn"),
        ]
        let fromPython = record(python)
        let fromSwift = record(swift)
        #expect(fromPython.unreadableLines == 0)
        #expect(fromPython.replies.map(\.text) == fromSwift.replies.map(\.text))
        #expect(fromPython.replies.map(\.kind) == [.baseline, .cell])
        #expect(fromPython.replies[0].concept == nil)
        #expect(fromPython.replies[1].truncated)
        #expect(!fromSwift.replies[1].truncated)
        let pair = SweepReplies.comparison(
            record: fromPython, concept: "fear", layer: 2, alpha: 0.1,
            prompts: SweepReplies.DevPrompts(status: .notLocal)
        ).pairs.first
        #expect(pair?.baseline?.text == "café au lait")
        #expect(pair?.steered?.text == "dread\nat dawn")
    }

    @Test func aTornTailAndARepeatAreReadAround() throws {
        // A kill mid-write leaves half a line; the record is prose, not a
        // ledger, so the rest still reads. A repeated generation keeps its
        // first text, as the server's resume dedupe does.
        let replies = record([
            try line("cell", "fear", 2, 0.1, 0, "first"),
            try line("cell", "fear", 2, 0.1, 0, "second"),
            #"{"kind": "mystery", "concept": "fear", "layer": 2, "alpha": 0.1, "promptIndex": 1, "text": "?"}"#,
            #"{"alpha": 0.1, "concept": "fear", "kind": "cell", "lay"#,
        ])
        #expect(replies.replies.map(\.text) == ["first"])
        #expect(replies.unreadableLines == 2)
    }

    @Test func aRunWithoutTheFileSaysItWasNotRecorded() throws {
        let run = try temporary()
        defer { try? fm.removeItem(at: run) }
        #expect(SweepReplies.load(runDirectory: run) == .notRecorded)
    }

    @Test func theSweepsOwnWriterAndThisReaderAgree() throws {
        // Round trip through the writer `ExperimentTasks.sweep` calls on
        // both the app's and the command line's route.
        let run = try temporary()
        defer { try? fm.removeItem(at: run) }
        try SweepRunCatalog.appendDevGeneration(
            runDirectory: run, kind: "baseline", concept: nil, layer: -1,
            alpha: 0, promptIndex: 0, text: "calm morning")
        try SweepRunCatalog.appendDevGeneration(
            runDirectory: run, kind: "cell", concept: "fear", layer: 17,
            alpha: 0.125, promptIndex: 0, text: "dread at dawn")
        guard case .recorded(let replies) = SweepReplies.load(runDirectory: run) else {
            Issue.record("the record did not load")
            return
        }
        let pair = SweepReplies.comparison(
            record: replies, concept: "fear", layer: 17, alpha: 0.125,
            prompts: SweepReplies.DevPrompts(status: .notNamed)
        ).pairs.first
        #expect(pair?.baseline?.text == "calm morning")
        #expect(pair?.steered?.text == "dread at dawn")
    }

    // MARK: Dev prompts

    /// A workspace with one sweep run whose snapshot names `file`.
    private func workspace(
        devPromptsFile file: String?, pin: String? = nil, prompts: String?
    ) throws -> (root: URL, run: URL) {
        let root = try temporary()
        let run = root.appending(path: "runs/2026-01-01T000000000Z-exp-demo-sweep")
        try fm.createDirectory(at: run, withIntermediateDirectories: true)
        var sweep: [String: Any] = [:]
        if let file { sweep["devPromptsFile"] = file }
        if let pin { sweep["devPromptsHash"] = pin }
        let snapshot: [String: Any] = ["name": "demo", "sweep": sweep]
        try JSONSerialization.data(withJSONObject: snapshot)
            .write(to: run.appending(component: "experiment.json"))
        if let prompts {
            let url = root.appending(path: "prompts/dev/dev.jsonl")
            try fm.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(prompts.utf8).write(to: url)
        }
        return (root, run)
    }

    private let promptsText = """
        {"text": "Describe a library."}
        {"text": "Plan a weekend."}

        """

    private var promptsHash: String { ExperimentStore.sha256Hex(Data(promptsText.utf8)) }

    @Test func promptTextIsShownWhenTheFileHoldsTheSweepsBytes() throws {
        let (root, run) = try workspace(
            devPromptsFile: "prompts/dev/dev.jsonl", prompts: promptsText)
        defer { try? fm.removeItem(at: root) }
        let matched = SweepReplies.devPrompts(
            runDirectory: run, workspaceRoot: root, recordedHashes: [promptsHash])
        #expect(matched.status == .matched)
        #expect(matched.texts == ["Describe a library.", "Plan a weekend."])
        // No hash recorded anywhere: shown, and marked unchecked.
        let unchecked = SweepReplies.devPrompts(
            runDirectory: run, workspaceRoot: root, recordedHashes: [])
        #expect(unchecked.status == .unchecked)
        #expect(unchecked.texts.count == 2)
    }

    @Test func promptTextIsWithheldWhenTheFileChangedSinceTheSweep() throws {
        // Text from an edited file would put the wrong prompt above a reply.
        let (root, run) = try workspace(
            devPromptsFile: "prompts/dev/dev.jsonl", pin: promptsHash,
            prompts: #"{"text": "A different first prompt."}"# + "\n")
        defer { try? fm.removeItem(at: root) }
        let changed = SweepReplies.devPrompts(
            runDirectory: run, workspaceRoot: root, recordedHashes: [])
        #expect(changed.status == .changed)
        #expect(changed.texts.isEmpty)
        #expect(changed.text(at: 0) == nil)
    }

    @Test func aMissingOrUnnamedPromptsFileIsSaidPlainly() throws {
        let missing = try workspace(
            devPromptsFile: "prompts/dev/dev.jsonl", prompts: nil)
        defer { try? fm.removeItem(at: missing.root) }
        #expect(
            SweepReplies.devPrompts(
                runDirectory: missing.run, workspaceRoot: missing.root,
                recordedHashes: []
            ).status == .missing)

        let unnamed = try workspace(devPromptsFile: nil, prompts: promptsText)
        defer { try? fm.removeItem(at: unnamed.root) }
        #expect(
            SweepReplies.devPrompts(
                runDirectory: unnamed.run, workspaceRoot: unnamed.root,
                recordedHashes: []
            ).status == .notNamed)

        // Outside the workspace is not this workspace's file to show, even
        // when a file is there: the outer folder holds one, the workspace
        // inside it is named in the snapshot by a path that climbs out.
        let outer = try workspace(devPromptsFile: nil, prompts: promptsText)
        defer { try? fm.removeItem(at: outer.root) }
        let inner = outer.root.appending(component: "inner")
        let run = inner.appending(component: "run")
        try fm.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONSerialization.data(
            withJSONObject: ["sweep": ["devPromptsFile": "../prompts/dev/dev.jsonl"]]
        ).write(to: run.appending(component: "experiment.json"))
        #expect(
            fm.fileExists(
                atPath: inner.appending(path: "../prompts/dev/dev.jsonl").path))
        #expect(
            SweepReplies.devPrompts(
                runDirectory: run, workspaceRoot: inner, recordedHashes: []
            ).status == .missing)
    }

    @Test func aRecommendationsHashChecksThePrompts() throws {
        // The run's recommendations carry the dev-prompts hash each
        // selection was made on; a drifted file fails that check even when
        // the study never pinned it.
        let (root, run) = try workspace(
            devPromptsFile: "prompts/dev/dev.jsonl", prompts: promptsText)
        defer { try? fm.removeItem(at: root) }
        func recommendations(_ hash: String) throws -> [String: SweepRunCatalog.Recommendation] {
            try SweepRunCatalog.parseRecommendations(
                Data(
                    """
                    {"fear": {"criterion": {}, "devPromptsHash": "\(hash)",
                      "metrics": {}, "sweepRun": "\(run.lastPathComponent)",
                      "winningCell": {"alpha": 0.1, "layer": 2}},
                     "hungry": "no cell passed the constraints"}
                    """.utf8))
        }
        let good = SweepRunCatalog.SweepRun(
            directory: run, rows: [], recommendations: try recommendations(promptsHash))
        #expect(SweepReplies.devPrompts(for: good, workspaceRoot: root).status == .matched)
        let drifted = SweepRunCatalog.SweepRun(
            directory: run, rows: [], recommendations: try recommendations("0000"))
        #expect(SweepReplies.devPrompts(for: drifted, workspaceRoot: root).status == .changed)
    }

    // MARK: The shipped Demo Workspace

    @Test func theDemoWorkspacesSweepCanBeReadCellByCell() throws {
        // The demo guide's step 5 sends researchers to this view to pick a
        // cell by its replies. Every carried sweep run must answer: every
        // grid cell beside the baseline, every dev prompt, with its text.
        let demos = URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(component: "DemoWorkspaces")
        let backends = (try? fm.contentsOfDirectory(
            at: demos, includingPropertiesForKeys: nil)) ?? []
        for backend in backends {
            let runs = backend.appending(component: "runs")
            let names = (try? fm.contentsOfDirectory(atPath: runs.path)) ?? []
            for name in names.sorted() where name.hasSuffix("-sweep") {
                let directory = runs.appending(component: name)
                let sweep = try SweepRunCatalog.load(directory: directory)
                guard case .recorded(let replies) = SweepReplies.load(runDirectory: directory)
                else {
                    Issue.record("\(name) carries no readable replies")
                    continue
                }
                #expect(replies.unreadableLines == 0, "\(name)")
                let prompts = SweepReplies.devPrompts(for: sweep, workspaceRoot: backend)
                #expect(prompts.status == .matched, "\(name)")
                #expect(!prompts.texts.isEmpty, "\(name)")
                for row in sweep.rows where !row.isBaseline {
                    let comparison = SweepReplies.comparison(
                        record: replies, concept: row.concept, layer: row.layer,
                        alpha: row.alpha, prompts: prompts)
                    #expect(
                        comparison.pairs.count == prompts.texts.count,
                        "\(name) \(row.concept) L\(row.layer) α\(row.alpha)")
                    #expect(
                        comparison.pairs.allSatisfy {
                            $0.prompt != nil && $0.baseline != nil && $0.steered != nil
                        },
                        "\(name) \(row.concept) L\(row.layer) α\(row.alpha)")
                }
            }
        }
    }
}
