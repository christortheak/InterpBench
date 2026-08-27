import Foundation

/// The Agent Library's LIST layer: one small, `Sendable`, row-ready summary
/// per saved agent, built off the main actor, plus a separately-run evidence
/// pass for the parts that cost real IO per agent.
///
/// Why this exists (2026-08-27, Agents-tab latency): switching to the Agents
/// section ran the whole library — directory walk, per-agent decode, a
/// `runs/` walk for robustness reports, and a full-file SHA-256 of EVERY
/// saved agent — synchronously inside the panel's `onAppear`, i.e. on the
/// main thread while the switch was trying to draw. The cost scaled with the
/// number of promoted agents exactly as reported, and none of it needed to
/// happen before the tab was on screen.
///
/// The split this type draws:
///
/// - **Index** (`load`) — what a ROW displays: name, date, kind, components,
///   promotion line, and the refs the chip rules judge. One decode per
///   artifact, no hashing, no `runs/` report reads. Feeds the list.
/// - **Evidence** (`evidence(for:runsDirectory:)`) — the expensive overlay:
///   one `runs/` report scan plus a content hash per agent, to attach each
///   agent's latest robustness result. Runs AFTER the list is on screen, and
///   only for the region that shows it.
///
/// Both phases report a `LoadStats` counting seam so a test can prove the
/// index did not do the evidence pass's work (`AgentLibraryIndexTests`).
///
/// The full `ModelVariantArtifact` is NOT retained by the row path — the
/// editor and every study-side consumer keep reading `FineTuningPanel
/// .variants` for that, unchanged.
public enum AgentLibraryIndex {

    // MARK: Entry

    /// One list row, fully rendered as data. Every display string is
    /// precomputed here rather than in a view body, so scrolling and
    /// re-layout cost nothing but drawing.
    public struct Entry: Identifiable, Sendable, Equatable {
        /// Same identity as `ModelVariantRecord.id` — the artifact's path.
        /// Selection, robustness targeting, and study attach all key on it.
        public var id: String
        public var url: URL
        public var name: String
        /// The artifact's ISO stamp, verbatim (sort key).
        public var createdAt: String
        /// Display form of `createdAt` ("2026-08-27 09:15").
        public var dateLabel: String
        public var components: AgentLibrary.Components
        public var kind: AgentLibrary.Kind
        public var componentsSummary: String
        /// "from 'x' · sweep y · criterion judgeScore · L14 α1.5", or nil for
        /// a hand-created agent.
        public var promotionLine: String?
        /// The experiment named by the promotion birth certificate, when the
        /// agent has one. Carried structured (not parsed back out of
        /// `promotionLine`) so the Optimizations lifecycle strip's "Promoted"
        /// evidence reads straight off the index instead of running its own
        /// `ModelVariantStore.scan()`.
        public var promotedExperiment: String?
        /// Stamped only on a manual-override promotion.
        public var overrideReason: String?

        public var baseModelID: String { components.baseModelID }

        public init(
            id: String,
            url: URL,
            name: String,
            createdAt: String,
            dateLabel: String,
            components: AgentLibrary.Components,
            kind: AgentLibrary.Kind,
            componentsSummary: String,
            promotionLine: String? = nil,
            promotedExperiment: String? = nil,
            overrideReason: String? = nil
        ) {
            self.id = id
            self.url = url
            self.name = name
            self.createdAt = createdAt
            self.dateLabel = dateLabel
            self.components = components
            self.kind = kind
            self.componentsSummary = componentsSummary
            self.promotionLine = promotionLine
            self.promotedExperiment = promotedExperiment
            self.overrideReason = overrideReason
        }

        public init(_ record: ModelVariantRecord) {
            let components = AgentLibrary.components(of: record.artifact)
            self.init(
                id: record.id,
                url: record.url,
                name: record.artifact.name,
                createdAt: record.artifact.createdAt,
                dateLabel: AgentLibraryIndex.dateLabel(record.artifact.createdAt),
                components: components,
                kind: AgentLibrary.kind(of: components),
                componentsSummary: AgentLibrary.componentsSummary(components),
                promotionLine: record.artifact.promotion.map(
                    AgentLibraryIndex.promotionLine),
                promotedExperiment: record.artifact.promotion?.experiment,
                overrideReason: record.artifact.promotion?.overrideReason)
        }
    }

    // MARK: Counting seam

    /// What a phase actually touched. Cheap to carry, and the only honest way
    /// to assert in a test that the LIST path does not do the EVIDENCE path's
    /// work — the difference is invisible in the results themselves.
    public struct LoadStats: Sendable, Equatable {
        /// Agent artifacts read and decoded (the index's own cost).
        public var artifactsDecoded: Int
        /// Agent artifacts read a second time and SHA-256'd.
        public var artifactsHashed: Int
        /// Robustness reports read out of the `runs/` tree.
        public var robustnessReportsRead: Int

        public init(
            artifactsDecoded: Int = 0,
            artifactsHashed: Int = 0,
            robustnessReportsRead: Int = 0
        ) {
            self.artifactsDecoded = artifactsDecoded
            self.artifactsHashed = artifactsHashed
            self.robustnessReportsRead = robustnessReportsRead
        }
    }

    // MARK: Index phase

    public struct Snapshot: Sendable {
        public var entries: [Entry]
        /// The workspace root these entries describe, so a stale list can
        /// never masquerade as another workspace's library.
        public var root: URL
        public var stats: LoadStats

        public init(entries: [Entry], root: URL, stats: LoadStats) {
            self.entries = entries
            self.root = root
            self.stats = stats
        }
    }

    /// Scan the library and summarize it. Pure IO + CPU, `nonisolated` on
    /// purpose: the caller runs it off the main actor.
    public static func load(
        directory: URL,
        importedRoot: URL?,
        root: URL
    ) -> Snapshot {
        let records = ModelVariantStore.scan(
            directory: directory, importedRoot: importedRoot)
        return Snapshot(
            entries: records.map(Entry.init),
            root: root,
            stats: LoadStats(artifactsDecoded: records.count))
    }

    /// Summaries for records the caller already holds (the synchronous
    /// refresh path, and tests).
    public static func summarize(_ records: [ModelVariantRecord]) -> [Entry] {
        records.map(Entry.init)
    }

    // MARK: Evidence phase

    /// The deferred overlay: latest robustness result per entry id, plus what
    /// producing it cost.
    public struct Evidence: Sendable {
        public var byEntryID: [String: AgentEvidence.RobustnessEvidence]
        public var stats: LoadStats

        public init(
            byEntryID: [String: AgentEvidence.RobustnessEvidence],
            stats: LoadStats
        ) {
            self.byEntryID = byEntryID
            self.stats = stats
        }
    }

    /// One `runs/` report scan, then one content hash per entry to match
    /// hash-stamped reports (`AgentEvidence.latestRobustness`). Deliberately
    /// separate from `load`: this is the part that costs a second full read
    /// of every artifact, and no row needs it to draw.
    ///
    /// `nonisolated` and root-explicit for the same reason as `load`.
    public static func evidence(
        for entries: [Entry],
        runsDirectory: URL
    ) -> Evidence {
        let reports = AgentEvidence.scanRobustnessReports(runsDirectory: runsDirectory)
        var map: [String: AgentEvidence.RobustnessEvidence] = [:]
        var hashed = 0
        for entry in entries {
            let hash = try? ModelVariantStore.hash(entry.url)
            if hash != nil { hashed += 1 }
            if let match = AgentEvidence.latestRobustness(
                in: reports, variantName: entry.name, artifactHash: hash)
            {
                map[entry.id] = match
            }
        }
        return Evidence(
            byEntryID: map,
            stats: LoadStats(
                artifactsHashed: hashed, robustnessReportsRead: reports.count))
    }

    // MARK: Display helpers (precomputed once, never per frame)

    /// "2026-08-27T09:15:03Z" → "2026-08-27 09:15". The same trimming the
    /// panel did inline, moved where it runs once per agent per scan.
    public static func dateLabel(_ iso: String) -> String {
        let trimmed = iso.replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
        return String(trimmed.prefix(min(16, trimmed.count)))
    }

    /// The row's one-line birth certificate.
    public static func promotionLine(
        _ promotion: ModelVariantArtifact.Promotion
    ) -> String {
        var parts = ["from '\(promotion.experiment)'"]
        if let run = promotion.sweepRun { parts.append("sweep \(run)") }
        if let metric = promotion.criterion?.objective?.metric {
            parts.append("criterion \(metric)")
        }
        if let cell = promotion.winningCell {
            parts.append("L\(cell.layer) α\(cell.alpha)")
        }
        return parts.joined(separator: " · ")
    }
}
