import Foundation

/// Reconstructs readable panel transcripts from a run's `generations.jsonl`
/// records (plan C2).
///
/// Deliberately NOT a reader for `transcript.md`. Per-turn records are the
/// canonical measurement artifact, and `transcript.md` is a human convenience
/// the measurement path must never read — building the view from the records
/// keeps one source of truth, and it is why the file's own chrome (section
/// headers, a `Routed to:` line of agent UUIDs) can no longer leak into any
/// scored quantity.
///
/// Pure over already-parsed records: no filesystem, no networking, so the
/// grouping and ordering rules are unit-testable without a run on disk.
public enum PanelTranscript {

    /// One turn as the reader sees it.
    public struct Turn: Sendable, Equatable, Identifiable {
        public var id: String { "\(condition)#\(replicate)#\(index)#\(promptID)" }
        public let index: Int
        public let promptID: String
        public let condition: String
        public let replicate: Int
        public let speaker: String
        public let title: String
        public let prompt: String
        public let output: String
        /// Who could see this turn afterwards. Empty means nobody — a private
        /// turn. Nil means the run predates the field being recorded, which is
        /// different from "nobody" and is rendered differently.
        public let routedTo: [String]?
        public let wordCount: Int?
    }

    /// One complete play-through of the panel under one condition.
    public struct Transcript: Sendable, Equatable, Identifiable {
        public var id: String { "\(condition)#\(replicate)" }
        public let condition: String
        public let replicate: Int
        public let turns: [Turn]

        public var totalWords: Int {
            turns.reduce(0) { $0 + ($1.wordCount ?? 0) }
        }
    }

    /// Group a run's records into transcripts.
    ///
    /// One transcript per `(condition, replicateIndex)` — the unit of analysis
    /// the statistics also use — with turns in script order. Records that are
    /// not turns (instrument readouts, ordinary study generations) are ignored,
    /// so calling this on a non-panel run yields an empty array rather than
    /// nonsense.
    ///
    /// Ordering is deterministic: conditions in a fixed order with `baseline`
    /// first so the arms line up left-to-right the way a reader compares them,
    /// then by replicate, then by `promptIndex` (falling back to the record
    /// order for runs that predate it).
    public static func transcripts(from records: [RunResults.Record]) -> [Transcript] {
        var grouped: [String: [(offset: Int, record: RunResults.Record)]] = [:]
        for (offset, record) in records.enumerated() where record.isTurn {
            let key = "\(record.condition)#\(record.replicateIndex ?? 0)"
            grouped[key, default: []].append((offset, record))
        }
        guard !grouped.isEmpty else { return [] }

        var conditionOrder: [String] = []
        var seen = Set<String>()
        for record in records where record.isTurn {
            if seen.insert(record.condition).inserted {
                conditionOrder.append(record.condition)
            }
        }
        // Baseline reads on the left; the rest keep first-appearance order.
        // Appearance rank is captured BEFORE sorting — reading the array's own
        // indices inside its comparator is both an exclusivity violation and a
        // moving target.
        let rank = Dictionary(
            uniqueKeysWithValues: conditionOrder.enumerated().map { ($0.element, $0.offset) })
        conditionOrder.sort { a, b in
            if a == "baseline" { return b != "baseline" }
            if b == "baseline" { return false }
            return (rank[a] ?? 0) < (rank[b] ?? 0)
        }

        var out: [Transcript] = []
        for condition in conditionOrder {
            let replicates = grouped.keys
                .filter { $0.hasPrefix("\(condition)#") }
                .compactMap { Int($0.split(separator: "#").last ?? "") }
                .sorted()
            for replicate in replicates {
                guard let entries = grouped["\(condition)#\(replicate)"] else { continue }
                let ordered = entries.sorted {
                    ($0.record.promptIndex ?? $0.offset, $0.offset)
                        < ($1.record.promptIndex ?? $1.offset, $1.offset)
                }
                let turns = ordered.enumerated().map { position, entry -> Turn in
                    Turn(
                        index: (entry.record.promptIndex ?? position) + 1,
                        promptID: entry.record.promptID,
                        condition: condition,
                        replicate: replicate,
                        speaker: entry.record.speakerName ?? "—",
                        title: entry.record.turnTitle ?? entry.record.promptID,
                        prompt: entry.record.prompt ?? "",
                        output: entry.record.output ?? "",
                        routedTo: entry.record.routedAgentIDs,
                        wordCount: entry.record.wordCount)
                }
                out.append(
                    Transcript(condition: condition, replicate: replicate, turns: turns))
            }
        }
        return out
    }

    /// Distinct conditions present, in reading order (baseline first).
    public static func conditions(in transcripts: [Transcript]) -> [String] {
        var seen = Set<String>()
        return transcripts.compactMap { seen.insert($0.condition).inserted ? $0.condition : nil }
    }

    /// Replicate indices present, ascending — what a pager steps through.
    public static func replicates(in transcripts: [Transcript]) -> [Int] {
        Array(Set(transcripts.map(\.replicate))).sorted()
    }

    /// The transcripts for one replicate, one per condition, in reading order
    /// — the side-by-side row for comparing arms at the same play-through.
    public static func arms(
        in transcripts: [Transcript], replicate: Int
    ) -> [Transcript] {
        transcripts.filter { $0.replicate == replicate }
    }
}
