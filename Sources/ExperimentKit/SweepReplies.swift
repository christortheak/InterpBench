import Foundation
import SteeringKit

/// A sweep's replies, read back cell by cell — the Optimizations grid's
/// reading pane.
///
/// Both engines record every dev-prompt generation a sweep makes in the run
/// directory's `dev-generations.jsonl` (`SweepRunCatalog.devGenerationsFile`;
/// server twin `sweep_evidence.DEV_GENERATIONS_FILE`, same schema). The grid
/// showed only the numbers computed from those texts, so a researcher who
/// wanted to choose a cell by reading what it wrote had to open the file by
/// hand. This reads the record back and pairs one cell's replies with the
/// unsteered baseline's replies to the same dev prompts (and, where the
/// sweep ran one, the matched-norm control's), with the prompt text from the
/// dev-prompts file the run names when that file still holds the bytes the
/// sweep used.
///
/// Pure apart from the two loaders, and the view renders exactly what
/// `comparison` returns.
public enum SweepReplies {

    /// The record's `kind`: the shared unsteered baseline (concept null,
    /// layer -1, alpha 0), a grid cell, or a matched-norm random-direction
    /// control run at a candidate cell.
    public enum Kind: String, Sendable, Equatable {
        case baseline
        case cell
        case control
    }

    /// One recorded generation.
    public struct Reply: Sendable, Equatable {
        public var kind: Kind
        public var concept: String?
        public var layer: Int
        public var alpha: Double
        public var promptIndex: Int
        public var text: String
        /// The engine capped an overlong text (`devGenerationTextLimit`).
        public var truncated: Bool

        public init(
            kind: Kind, concept: String?, layer: Int, alpha: Double,
            promptIndex: Int, text: String, truncated: Bool = false
        ) {
            self.kind = kind
            self.concept = concept
            self.layer = layer
            self.alpha = alpha
            self.promptIndex = promptIndex
            self.text = text
            self.truncated = truncated
        }
    }

    /// A parsed `dev-generations.jsonl`.
    public struct Record: Sendable, Equatable {
        /// File order, first occurrence of each (kind, concept, layer,
        /// alpha, prompt) kept — the server's resume dedupe keeps the first
        /// too, so a repeat can only come from a record written before it.
        public var replies: [Reply]
        /// Lines that did not parse: a kill mid-write leaves a torn tail,
        /// and the record is prose, not a ledger, so it is read around.
        public var unreadableLines: Int

        public init(replies: [Reply], unreadableLines: Int = 0) {
            self.replies = replies
            self.unreadableLines = unreadableLines
        }
    }

    /// What a sweep run holds for the reading pane.
    public enum Availability: Sendable, Equatable {
        /// The run has no `dev-generations.jsonl` — it predates the record
        /// (or the server's listing does not show one).
        case notRecorded
        /// The file exists but could not be read at all.
        case unreadable(String)
        case recorded(Record)
    }

    private struct Line: Decodable {
        var kind: String
        var concept: String?
        var layer: Int
        var alpha: Double
        var promptIndex: Int
        var text: String
        var truncated: Bool?
    }

    private struct Key: Hashable {
        var kind: Kind
        var concept: String?
        var layer: Int
        var alpha: Double
        var promptIndex: Int
    }

    /// Parse the record's bytes. Tolerant line by line: either engine's
    /// encoding (Swift's compact form, Python's spaced `json.dumps` with
    /// ASCII escapes), an unknown future key, and a torn tail all read.
    public static func parse(_ data: Data) -> Record {
        let decoder = JSONDecoder()
        var replies: [Reply] = []
        var seen = Set<Key>()
        var unreadable = 0
        let text = String(decoding: data, as: UTF8.self)
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard
                let decoded = try? decoder.decode(Line.self, from: Data(line.utf8)),
                let kind = Kind(rawValue: decoded.kind)
            else {
                unreadable += 1
                continue
            }
            let key = Key(
                kind: kind, concept: decoded.concept, layer: decoded.layer,
                alpha: decoded.alpha, promptIndex: decoded.promptIndex)
            guard seen.insert(key).inserted else { continue }
            replies.append(
                Reply(
                    kind: kind, concept: decoded.concept, layer: decoded.layer,
                    alpha: decoded.alpha, promptIndex: decoded.promptIndex,
                    text: decoded.text, truncated: decoded.truncated ?? false))
        }
        return Record(replies: replies, unreadableLines: unreadable)
    }

    /// Read a local sweep run directory's record.
    public static func load(runDirectory: URL) -> Availability {
        let url = runDirectory.appending(component: SweepRunCatalog.devGenerationsFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .notRecorded
        }
        do {
            return .recorded(parse(try Data(contentsOf: url)))
        } catch {
            return .unreadable(error.localizedDescription)
        }
    }

    // MARK: Dev prompts

    /// The dev prompts a run's replies answer, by prompt number.
    public struct DevPrompts: Sendable, Equatable {
        public enum Status: Sendable, Equatable {
            /// The file holds the bytes the sweep recorded using.
            case matched
            /// The file is readable, but the run recorded no hash to check
            /// it against (a sweep that selected nothing, on a study whose
            /// inputs were never pinned).
            case unchecked
            /// The file's bytes differ from the ones the sweep used.
            case changed
            /// The file the run names is not in the workspace.
            case missing
            /// The run does not name a dev-prompts file.
            case notNamed
            /// The run lives on a server; its prompts are on that server's
            /// tree, not this one.
            case notLocal
        }

        /// Empty unless `status` is `.matched` or `.unchecked`: text from a
        /// changed file would put the wrong prompt above a reply.
        public var texts: [String]
        public var status: Status

        public init(texts: [String] = [], status: Status) {
            self.texts = texts
            self.status = status
        }

        public static let notLocal = DevPrompts(status: .notLocal)

        public func text(at index: Int) -> String? {
            texts.indices.contains(index) ? texts[index] : nil
        }
    }

    private struct Snapshot: Decodable {
        struct Sweep: Decodable {
            var devPromptsFile: String?
            var devPromptsHash: String?
        }
        var sweep: Sweep?
    }

    /// The dev prompts a LOCAL sweep run used: the file its `experiment.json`
    /// snapshot names, resolved against the workspace root, checked against
    /// every hash the run recorded for it (the snapshot's pin and each
    /// recommendation's `devPromptsHash`).
    public static func devPrompts(
        for run: SweepRunCatalog.SweepRun, workspaceRoot: URL
    ) -> DevPrompts {
        let recommendationHashes = run.recommendations.values.compactMap {
            recommendation -> String? in
            if case .selected(let provenance) = recommendation {
                return provenance.devPromptsHash
            }
            return nil
        }
        return devPrompts(
            runDirectory: run.directory, workspaceRoot: workspaceRoot,
            recordedHashes: recommendationHashes)
    }

    /// `recordedHashes` are the hashes the run recorded outside its
    /// snapshot; the snapshot's own pin, when present, is added to them.
    public static func devPrompts(
        runDirectory: URL, workspaceRoot: URL, recordedHashes: [String]
    ) -> DevPrompts {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "experiment.json")),
            let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data),
            let file = snapshot.sweep?.devPromptsFile, !file.isEmpty
        else { return DevPrompts(status: .notNamed) }
        // Workspace-relative by contract; anything else is not this
        // workspace's file to show.
        let components = file.split(separator: "/")
        guard !file.hasPrefix("/"), !components.contains("..") else {
            return DevPrompts(status: .missing)
        }
        let url = workspaceRoot.appending(path: file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return DevPrompts(status: .missing)
        }
        // The sweep's own reader: a file it can no longer parse is not the
        // file the sweep read.
        guard let loaded = try? StimulusSet.loadTexts(url: url) else {
            return DevPrompts(status: .changed)
        }
        let recorded = recordedHashes + [snapshot.sweep?.devPromptsHash].compactMap { $0 }
        if recorded.isEmpty {
            return DevPrompts(texts: loaded.texts, status: .unchecked)
        }
        guard recorded.allSatisfy({ $0 == loaded.hash }) else {
            return DevPrompts(status: .changed)
        }
        return DevPrompts(texts: loaded.texts, status: .matched)
    }

    // MARK: One cell beside the baseline

    /// One selected cell's replies, prompt by prompt, beside the baseline's.
    public struct Comparison: Sendable, Equatable {
        public struct Pair: Sendable, Equatable, Identifiable {
            public var promptIndex: Int
            public var prompt: String?
            public var baseline: Reply?
            public var steered: Reply?
            /// The matched-norm random-direction control at this cell, when
            /// the sweep ran one here.
            public var control: Reply?

            public var id: Int { promptIndex }
        }

        public var concept: String
        public var layer: Int
        public var alpha: Double
        /// Ordered by prompt number: one row per prompt this cell (or its
        /// control) answered, with the baseline's reply when it has one, so
        /// a missing side shows as missing rather than dropping the row.
        public var pairs: [Pair]

        public var hasCellReplies: Bool { pairs.contains { $0.steered != nil } }
        public var hasBaseline: Bool { pairs.contains { $0.baseline != nil } }
        public var hasControl: Bool { pairs.contains { $0.control != nil } }
    }

    /// Pair the cell's replies with the baseline's by prompt number. The
    /// baseline is generated once per sweep (concept null); a baseline row
    /// that names this concept is accepted too.
    public static func comparison(
        record: Record, concept: String, layer: Int, alpha: Double,
        prompts: DevPrompts
    ) -> Comparison {
        func atCell(_ reply: Reply) -> Bool {
            reply.concept == concept && reply.layer == layer
                && abs(reply.alpha - alpha) < 1e-9
        }
        var baseline: [Int: Reply] = [:]
        var steered: [Int: Reply] = [:]
        var control: [Int: Reply] = [:]
        for reply in record.replies {
            switch reply.kind {
            case .baseline where reply.concept == nil || reply.concept == concept:
                baseline[reply.promptIndex] = baseline[reply.promptIndex] ?? reply
            case .cell where atCell(reply):
                steered[reply.promptIndex] = steered[reply.promptIndex] ?? reply
            case .control where atCell(reply):
                control[reply.promptIndex] = control[reply.promptIndex] ?? reply
            default:
                continue
            }
        }
        // Rows only for prompts this cell or its control answered: the
        // baseline is every prompt, and a cell cut short by a cancel must
        // not look like it answered them all.
        let indices = Set(steered.keys).union(control.keys).sorted()
        return Comparison(
            concept: concept, layer: layer, alpha: alpha,
            pairs: indices.map { index in
                Comparison.Pair(
                    promptIndex: index, prompt: prompts.text(at: index),
                    baseline: baseline[index], steered: steered[index],
                    control: control[index])
            })
    }
}
